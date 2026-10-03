"""Decoder-only transformer and Student-t mixture heads."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import N_FEAT

def alibi_slopes(H):
    s = 2.0 ** (-np.linspace(1.0, 10.0, max(H - 1, 1)))
    return np.concatenate([s, [0.0]])[:H] if H > 1 else np.array([0.0])


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = heads

    def forward(self, x, bias, use_sdpa):
        B, T, d = x.shape
        H = self.heads
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
        if use_sdpa:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            s = q @ k.transpose(-1, -2) / math.sqrt(d // H) + bias
            a = torch.softmax(s, dim=-1) @ v
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class VolPFN(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, K=4):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, K=K)
        self.inp = nn.Linear(N_FEAT, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head_mix = nn.Linear(d, 4 * K)
        self.head_lat = nn.Linear(d, 2)
        nn.init.normal_(self.head_mix.weight, std=0.02)
        nn.init.zeros_(self.head_mix.bias)
        self.head_mix.bias.data[3 * K :] = 1.4
        nn.init.normal_(self.head_lat.weight, std=0.02)
        nn.init.zeros_(self.head_lat.bias)
        self.register_buffer(
            'slopes', torch.tensor(alibi_slopes(heads), dtype=torch.float32), persistent=False
        )
        self._bias = {}
        self.use_sdpa = None

    def bias(self, T, device, dtype):
        key = (T, str(device), dtype)
        if key not in self._bias:
            i = torch.arange(T, device=device)
            dist = (i[:, None] - i[None, :]).to(dtype)  # i - j >= 0 below the diagonal
            b = -self.slopes.to(device=device, dtype=dtype)[:, None, None] * dist[None]
            b = b.masked_fill(dist[None] < 0, float('-inf'))
            if len(self._bias) > 8:
                self._bias.clear()
            self._bias[key] = b[None]  # (1, H, T, T)
        return self._bias[key]

    def forward(self, feats, base):
        """feats (B,T,N_FEAT), base (B,T) log baseline variance -> dict of mixture parameters for the next return."""
        x = self.inp(feats)
        T = x.shape[1]
        bias = self.bias(T, x.device, x.dtype)
        if self.use_sdpa is None:
            try:
                F.scaled_dot_product_attention(
                    x[:1, :2].view(1, 1, 2, -1),
                    x[:1, :2].view(1, 1, 2, -1),
                    x[:1, :2].view(1, 1, 2, -1),
                    attn_mask=bias[:, :1, :2, :2].to(x.dtype),
                )
                self.use_sdpa = True
            except Exception:
                self.use_sdpa = False
        for blk in self.blocks:
            x = blk(x, bias, self.use_sdpa)
        x = self.ln_f(x)
        K = self.cfg['K']
        o = self.head_mix(x)
        lat = self.head_lat(x)
        logits, a, b, c = o[..., :K], o[..., K : 2 * K], o[..., 2 * K : 3 * K], o[..., 3 * K :]
        s_base = torch.exp(0.5 * base)[..., None]
        return dict(
            logw=torch.log_softmax(logits, -1),
            loc=s_base * a,
            logscale=0.5 * base[..., None] + b.clamp(-6, 6),
            nu=2.1 + torch.exp(c.clamp(-4, 6)),
            lat_mean=base + lat[..., 0],
            lat_logsd=lat[..., 1].clamp(-5, 3),
        )


def count_params(m):
    return sum(p.numel() for p in m.parameters())
