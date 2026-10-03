#!/usr/bin/env python3
"""
volpfn.py -- VolPFN: a prior-fitted network for volatility, VaR and ES.

WHAT IT IS.  A decoder-only transformer trained once on samples from the simulated universe of volpfn_prior.py
(GARCH-type recursions, log-AR(1) and rough stochastic volatility, Markov-switching and lattice-jump variance paths,
Gaussian / Student-t / skew-t innovations, return jumps, serial correlation, drift and stale prices).  At every position
it outputs a Student-t mixture for the NEXT return (weights, locations, scales, tail indices) and a Gaussian posterior
for the log conditional variance.  Trained with the negative log predictive density (plus an auxiliary latent-variance
loss). Pure predictive log loss targets the prior's posterior predictive under a matching information set;
finite capacity, auxiliary supervision and preprocessing can change the approximation. One set of weights
forecasts assets without per-series fitting from its last L returns. The legacy known-parameter references
have privileged information and do not certify real-market optimality. The separate research/grid_reference.py
laboratory supplies a same-information exact reference for a finite Gaussian GARCH prior.

ARCHITECTURE.  Inputs per position t (all causal): asinh(r_t), log(r_t^2 + 1e-6)/4, 1[r_t = 0], sign(r_t), the log of
three causal EWMAs of r^2 (half-lives 5, 20, 100; zeros skipped) /4, and log(1+t)/8.  Pre-LayerNorm blocks with
multi-head causal attention carrying ALiBi linear biases (head h attends with the penalty m_h (i - j), one head has
m = 0), i.e. each head is an exponentially weighted average over the past -- a bank of EWMAs of learned features --
and the model extrapolates to windows longer than it was trained on.  The mixture head is parametrised around the
half-life-20 EWMA scale: loc_k = s_base a_k, scale_k = s_base exp(b_k), nu_k = 2.1 + exp(c_k).

USAGE
  python3 volpfn.py train --size small --steps 3000 --out out_volpfn            (CPU-sized run; --device auto picks cuda / mps / cpu)
  python3 volpfn.py train --size base --steps 30000 --tokens 8192 --out out_volpfn --lengths 512:0.4,1024:0.6
  python3 volpfn.py train --resume out_volpfn/volpfn_base.pt --steps 60000 ...   (continue a run)
  python3 volpfn.py predict --ckpt out_volpfn/volpfn_base_best.pt --data prices.csv --prices --col AAPL --out aapl_volpfn.csv
  python3 volpfn.py selftest                                                      (tiny end-to-end check, ~1 minute)
Library: model = load_model(path, device); out = predict_series(model, returns, levels=(0.975, 0.99)) -> dict of arrays
aligned with `returns`: h (mixture second moment), h_lat (posterior mean of the latent variance), logscore, var<level>,
es<level>, and the mixture parameters.
"""
import argparse, json, math, os, sys, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import stdtr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volpfn_prior as VP

SIZES = dict(tiny=dict(d=48, layers=2, heads=4, K=3), small=dict(d=128, layers=4, heads=4, K=4), base=dict(d=256, layers=6, heads=8, K=4), large=dict(d=384, layers=8, heads=8, K=5))
N_FEAT = 8
HALF_LIVES = (5.0, 20.0, 100.0)

def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# ----------------------------------------------------------------------------- features (numpy, causal)
def make_features(r):
    """r: (B, T) standardised returns -> feats (B, T, N_FEAT) float32, base_logvar (B, T) float32 = log EWMA_20(r^2)."""
    r = np.asarray(r, dtype=np.float64); B, T = r.shape
    feats = np.empty((B, T, N_FEAT), dtype=np.float32)
    feats[:, :, 0] = np.arcsinh(r); feats[:, :, 1] = np.log(r * r + 1e-6) / 4; feats[:, :, 2] = (r == 0); feats[:, :, 3] = np.sign(r)
    nz = r != 0
    for j, hl in enumerate(HALF_LIVES):
        lam = 0.5 ** (1.0 / hl); e = np.ones(B); col = np.empty((B, T))
        for t in range(T):
            e = np.where(nz[:, t], lam * e + (1 - lam) * r[:, t] ** 2, e); col[:, t] = e
        feats[:, :, 4 + j] = np.log(col) / 4
        if hl == 20.0:
            base = np.log(col).astype(np.float32)
    feats[:, :, 7] = np.log1p(np.arange(T))[None, :] / 8
    return feats, base

# ----------------------------------------------------------------------------- the model
def lgamma_safe(x):
    try:
        return torch.lgamma(x)
    except Exception:                                    # Stirling with a two-step shift (error < 1e-6 for x >= 1)
        y = x + 2.0
        return (y - 0.5) * torch.log(y) - y + 0.5 * math.log(2 * math.pi) + 1 / (12 * y) - 1 / (360 * y ** 3) - torch.log(x * (x + 1))

def t_logpdf_torch(z, nu):
    return lgamma_safe((nu + 1) / 2) - lgamma_safe(nu / 2) - 0.5 * torch.log(nu * math.pi) - 0.5 * (nu + 1) * torch.log1p(z * z / nu)

def alibi_slopes(H):
    s = 2.0 ** (-np.linspace(1.0, 10.0, max(H - 1, 1))); return np.concatenate([s, [0.0]])[:H] if H > 1 else np.array([0.0])

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); self.heads = heads

    def forward(self, x, bias, use_sdpa):
        B, T, d = x.shape; H = self.heads
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
        self.inp = nn.Linear(N_FEAT, d); self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)]); self.ln_f = nn.LayerNorm(d)
        self.head_mix = nn.Linear(d, 4 * K); self.head_lat = nn.Linear(d, 2)
        nn.init.normal_(self.head_mix.weight, std=0.02); nn.init.zeros_(self.head_mix.bias); self.head_mix.bias.data[3 * K:] = 1.4
        nn.init.normal_(self.head_lat.weight, std=0.02); nn.init.zeros_(self.head_lat.bias)
        self.register_buffer('slopes', torch.tensor(alibi_slopes(heads), dtype=torch.float32), persistent=False)
        self._bias = {}; self.use_sdpa = None

    def bias(self, T, device, dtype):
        key = (T, str(device), dtype)
        if key not in self._bias:
            i = torch.arange(T, device=device)
            dist = (i[:, None] - i[None, :]).to(dtype)                                   # i - j >= 0 below the diagonal
            b = -self.slopes.to(device=device, dtype=dtype)[:, None, None] * dist[None]
            b = b.masked_fill(dist[None] < 0, float('-inf'))
            if len(self._bias) > 8:
                self._bias.clear()
            self._bias[key] = b[None]                                                    # (1, H, T, T)
        return self._bias[key]

    def forward(self, feats, base):
        """feats (B,T,N_FEAT), base (B,T) log baseline variance -> dict of mixture parameters for the next return."""
        x = self.inp(feats); T = x.shape[1]; bias = self.bias(T, x.device, x.dtype)
        if self.use_sdpa is None:
            try:
                F.scaled_dot_product_attention(x[:1, :2].view(1, 1, 2, -1), x[:1, :2].view(1, 1, 2, -1), x[:1, :2].view(1, 1, 2, -1), attn_mask=bias[:, :1, :2, :2].to(x.dtype))
                self.use_sdpa = True
            except Exception:
                self.use_sdpa = False
        for blk in self.blocks:
            x = blk(x, bias, self.use_sdpa)
        x = self.ln_f(x); K = self.cfg['K']; o = self.head_mix(x); lat = self.head_lat(x)
        logits, a, b, c = o[..., :K], o[..., K:2 * K], o[..., 2 * K:3 * K], o[..., 3 * K:]
        s_base = torch.exp(0.5 * base)[..., None]
        return dict(logw=torch.log_softmax(logits, -1), loc=s_base * a, logscale=0.5 * base[..., None] + b.clamp(-6, 6),
                    nu=2.1 + torch.exp(c.clamp(-4, 6)), lat_mean=base + lat[..., 0], lat_logsd=lat[..., 1].clamp(-5, 3))

def mixture_logpdf(out, r):
    """log predictive density of r (B,T) under the mixture at the same positions."""
    z = (r[..., None] - out['loc']) / torch.exp(out['logscale'])
    return torch.logsumexp(out['logw'] + t_logpdf_torch(z, out['nu']) - out['logscale'], dim=-1)

def losses(out, r, logv, mask):
    """next-step losses: out at positions 0..T-2 predicts r[1:], logv[1:]; mask[1:] weights the targets."""
    nll = -mixture_logpdf({k: v[:, :-1] for k, v in out.items()}, r[:, 1:])
    m = out['lat_mean'][:, :-1]; ls = out['lat_logsd'][:, :-1]
    aux = 0.5 * ((logv[:, 1:] - m) * torch.exp(-ls)) ** 2 + ls
    w = mask[:, 1:]; den = w.sum().clamp_min(1.0)
    return (nll * w).sum() / den, (aux * w).sum() / den

# ----------------------------------------------------------------------------- data
def parse_lengths(s):
    d = {}
    for part in s.split(','):
        L, p = part.split(':'); d[int(L)] = float(p)
    tot = sum(d.values()); return {L: p / tot for L, p in d.items()}

def gen_batch(seed, B, L, weights=None, overlay_p=None):
    rng = np.random.default_rng(seed); b = VP.sample_batch(rng, B, L, weights=weights, overlay_p=overlay_p)
    feats, base = make_features(b['r'])
    return dict(r=b['r'], logv=b['logv'], mask=b['mask'], feats=feats, base=base, fam=[m['family'] for m in b['meta']])

def to_dev(batch, device):
    return {k: torch.as_tensor(batch[k]).to(device) for k in ('r', 'logv', 'mask', 'feats', 'base')}

class Prefetcher:
    """bounded queue of prior batches generated by a process pool (seeds are deterministic given --seed)."""
    def __init__(self, workers, seed0, lengths, tokens, weights, overlay_p, depth=6):
        from multiprocessing import Pool
        self.pool = Pool(workers) if workers > 0 else None; self.q = deque(); self.seed = seed0; self.depth = depth
        self.lengths = lengths; self.tokens = tokens; self.weights = weights; self.overlay_p = overlay_p
        self.rng = np.random.default_rng(seed0 + 99991)

    def _args(self):
        Ls = list(self.lengths); L = int(self.rng.choice(Ls, p=[self.lengths[l] for l in Ls])); B = max(1, self.tokens // L)
        self.seed += 1; return (self.seed, B, L, self.weights, self.overlay_p)

    def next(self):
        if self.pool is None:
            return gen_batch(*self._args())
        while len(self.q) < self.depth:
            self.q.append(self.pool.apply_async(gen_batch, self._args()))
        return self.q.popleft().get()

    def close(self):
        if self.pool is not None:
            self.pool.terminate()

# ----------------------------------------------------------------------------- validation sets with oracles
def build_validation(n_all=48, n_fam=8, T=1024, seed=777):
    rng = np.random.default_rng(seed)
    val = dict(all=VP.sample_batch(rng, n_all, T))
    for f in ('garch', 'gjr', 'egarch', 'sv', 'hmm', 'lattice'):
        b = VP.sample_batch(rng, n_fam, T, family=f)
        b['oracle_h'] = np.empty((n_fam, T)); b['oracle_ld'] = np.empty((n_fam, T))
        for i in range(n_fam):
            b['oracle_h'][i], b['oracle_ld'][i] = VP.oracle_predictive(b['meta'][i], b['r'][i].astype(float))
        val[f] = b
    for k, b in val.items():
        b['feats'], b['base'] = make_features(b['r'])
    return val

def univol_on_batch(b, burn=100):
    """mean predictive log density and QLIKE of univol on a validation batch (positions >= burn)."""
    from univol import run_series
    nll = []; ql = []
    for i in range(len(b['r'])):
        r = b['r'][i].astype(float); u = run_series(r, levels=())
        w = b['mask'][i][burn:]; nll.append(-np.sum(u['logscore'][burn:] * w) / w.sum()); ql.append(np.sum((r[burn:] ** 2 / u['h'][burn:] + np.log(u['h'][burn:])) * w) / w.sum())
    return float(np.mean(nll)), float(np.mean(ql))

@torch.no_grad()
def evaluate(model, val, device, burn=100, bs=8):
    model.eval(); res = {}
    for name, b in val.items():
        n, T = b['r'].shape; nll = np.zeros(n); ql2 = np.zeros(n); qll = np.zeros(n); qlo = np.zeros(n); gap = np.zeros(n)
        for i0 in range(0, n, bs):
            sl = slice(i0, min(n, i0 + bs)); bt = to_dev({k: b[k][sl] for k in ('r', 'logv', 'mask', 'feats', 'base')}, device)
            out = model(bt['feats'], bt['base']); ld = mixture_logpdf({k: v[:, :-1] for k, v in out.items()}, bt['r'][:, 1:]).cpu().numpy()
            w = b['mask'][sl][:, 1:].copy(); w[:, :burn - 1] = 0
            r2 = b['r'][sl][:, 1:].astype(float) ** 2
            h2 = mixture_second_moment(out)[:, :-1].cpu().numpy(); hl = torch.exp(out['lat_mean'] + 0.5 * torch.exp(2 * out['lat_logsd']))[:, :-1].cpu().numpy()
            nll[sl] = -np.sum(ld * w, 1) / w.sum(1)
            ql2[sl] = np.sum((r2 / h2 + np.log(h2)) * w, 1) / w.sum(1); qll[sl] = np.sum((r2 / hl + np.log(hl)) * w, 1) / w.sum(1)
            if 'oracle_ld' in b:
                gap[sl] = np.sum((b['oracle_ld'][sl][:, 1:] - ld) * w, 1) / w.sum(1)          # NLL(model) - NLL(oracle)
                ho = b['oracle_h'][sl][:, 1:]; qlo[sl] = np.sum((r2 / ho + np.log(ho)) * w, 1) / w.sum(1)
        res[name] = dict(nll=float(nll.mean()), qlike=float(ql2.mean()), qlike_lat=float(qll.mean()))
        if 'oracle_ld' in b:
            res[name]['gap'] = float(gap.mean()); res[name]['qlike_oracle'] = float(qlo.mean())
    model.train(); return res

def mixture_second_moment(out):
    nu = out['nu']; return torch.sum(torch.exp(out['logw']) * (torch.exp(2 * out['logscale']) * nu / (nu - 2) + out['loc'] ** 2), dim=-1)

# ----------------------------------------------------------------------------- training
def count_params(m):
    return sum(p.numel() for p in m.parameters())

def train(a):
    device = pick_device(a.device); torch.manual_seed(a.seed)
    if device.type == 'cpu':
        torch.set_num_threads(a.threads or max(1, os.cpu_count() or 1))
    os.makedirs(a.out, exist_ok=True); name = a.name or f'volpfn_{a.size}'
    cfg = SIZES[a.size]; model = VolPFN(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    step0 = 0; history = []
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['optimizer']); step0 = ck['step']; history = ck.get('history', [])
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        print(f'resumed {a.resume} at step {step0}', flush=True)
    lengths = parse_lengths(a.lengths); weights = VP.PRIOR_WEIGHTS if not a.prior else {k: float(v) for k, v in (kv.split('=') for kv in a.prior.split(','))}
    print(f'VolPFN {a.size}: {count_params(model) / 1e6:.2f}M parameters on {device}; lengths {lengths}; {a.tokens} tokens per batch; prior weights {weights}', flush=True)
    t0 = time.time(); val = build_validation(T=a.val_T); print(f'validation sets built in {time.time() - t0:.0f}s: all ({len(val["all"]["r"])} series) + 6 oracle families x {len(val["garch"]["r"])}', flush=True)
    ref_path = os.path.join(a.out, f'univol_val_T{a.val_T}.json')
    if a.univol_val and not os.path.exists(ref_path):
        t0 = time.time(); ref = {k: univol_on_batch(b) for k, b in val.items()}
        json.dump(ref, open(ref_path, 'w')); print(f'univol on the validation sets in {time.time() - t0:.0f}s', flush=True)
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else None
    if ref:
        print('reference (univol, positions >= 100): ' + ' | '.join(f'{k} nll {v[0]:.4f} qlike {v[1]:.4f}' for k, v in ref.items()), flush=True)
        print('oracle floors (known parameters):    ' + ' | '.join(f'{k} nll {np.mean(-val[k]["oracle_ld"][:, 100:]):.4f}' for k in val if 'oracle_ld' in val[k]), flush=True)
    overlay = None if not getattr(a, 'overlay', None) else {k: float(v) for k, v in (kv.split('=') for kv in a.overlay.split(','))}
    pf = Prefetcher(a.workers, a.seed * 100003 + step0, lengths, a.tokens, weights, overlay)
    sched = lambda s: a.lr * min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps)))))
    ema = None; tok = 0; t0 = time.time(); best = float('inf'); use_amp = device.type == 'cuda'
    for step in range(step0, a.steps):
        for g in opt.param_groups:
            g['lr'] = sched(step)
        batch = pf.next(); bt = to_dev(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            out = model(bt['feats'], bt['base'])
        out = {k: v.float() for k, v in out.items()}
        nll, aux = losses(out, bt['r'], bt['logv'], bt['mask']); loss = nll + a.aux * aux
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tok += bt['r'].numel(); v = float(nll.detach()); ema = v if ema is None else 0.98 * ema + 0.02 * v
        if (step + 1) % a.log_every == 0 or step == step0:
            el = time.time() - t0
            print(f'step {step + 1:6d}  nll {v:.4f} (ema {ema:.4f})  aux {float(aux.detach()):.3f}  gn {float(gn):.2f}  lr {sched(step):.2e}  L {bt["r"].shape[1]} B {bt["r"].shape[0]}  {tok / el:.0f} tok/s  {el / 60:.1f} min', flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ev = evaluate(model, val, device); ev['step'] = step + 1; ev['train_nll_ema'] = ema; history.append(ev)
            gaps = ' '.join(f'{k}:{ev[k]["gap"]:+.4f}' for k in ev if isinstance(ev[k], dict) and 'gap' in ev[k])
            print(f'  eval step {step + 1}: val-all nll {ev["all"]["nll"]:.4f} qlike {ev["all"]["qlike"]:.4f} (latent-head qlike {ev["all"]["qlike_lat"]:.4f}) | gap to the exact filters (nats/step) {gaps}', flush=True)
            ck = dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), step=step + 1, cfg=cfg, size=a.size, history=history, args=vars(a))
            torch.save(ck, os.path.join(a.out, f'{name}.pt'))
            if ev['all']['nll'] < best:
                best = ev['all']['nll']; torch.save(dict(state_dict=model.state_dict(), step=step + 1, cfg=cfg, size=a.size, history=history), os.path.join(a.out, f'{name}_best.pt'))
            with open(os.path.join(a.out, f'{name}_log.jsonl'), 'a') as f:
                f.write(json.dumps(ev) + '\n')
    pf.close(); print(f'done: {a.steps} steps, best val-all nll {best:.4f}; checkpoints {a.out}/{name}.pt and {name}_best.pt', flush=True)

# ----------------------------------------------------------------------------- inference
def load_model(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = VolPFN(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval(); return m.to(pick_device(device) if isinstance(device, str) else device)

@torch.no_grad()
def predict_series(model, r, L=1024, stride=None, levels=(0.975, 0.99), device=None, batch_windows=4, min_history=1):
    """One-step-ahead predictive distributions for a whole return series (any scale), causally, from sliding windows of
    length L with stride L/2 (every forecast after the first window uses at least L/2 of context; each window is
    standardised by an already observed prefix of at most 100 returns). Early forecasts use separate growing
    prefixes so their scale cannot depend on future observations. Set min_history to skip unneeded warm-up
    forecasts; this only changes which entries are returned, not retained forecast values.
    Returns dict of arrays aligned with r: h (mixture second
    moment, the QLIKE Bayes act), h_lat (posterior mean of the latent variance from the auxiliary head), logscore (log
    predictive density of the realised return), var<level> and es<level> (positive losses), and the mixture parameters
    logw, loc, scale, nu of shape (T, K).  Entry t is the forecast for r[t] made from r[:t]; entry 0 is NaN."""
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError('returns must be a finite one-dimensional array')
    if not isinstance(L, (int, np.integer)) or L < 2:
        raise ValueError('L must be an integer >= 2')
    stride = L // 2 if stride is None else stride
    if not isinstance(stride, (int, np.integer)) or not 1 <= stride <= L // 2:
        raise ValueError('stride must be an integer in [1, L//2]')
    if not isinstance(min_history, (int, np.integer)) or min_history < 1:
        raise ValueError('min_history must be a positive integer')
    if not isinstance(batch_windows, (int, np.integer)) or batch_windows < 1:
        raise ValueError('batch_windows must be a positive integer')
    T = len(r); device = next(model.parameters()).device if device is None else device
    K = model.cfg['K']; prefix = min(100, stride)
    logw = np.full((T, K), np.nan); loc = np.full((T, K), np.nan); lsc = np.full((T, K), np.nan); nu = np.full((T, K), np.nan)
    lat_m = np.full(T, np.nan); lat_s = np.full(T, np.nan)
    # Create a window as soon as it owns a forecast. Waiting until the previous
    # full window ends would retrospectively change outputs when stride < L/2.
    starts = [0] + list(range(stride, max(T - stride, 0), stride))
    wins = []
    for t in range(min_history, min(prefix, T)):
        sc = VP.initial_scale(r[:t], k=t)
        wins.append((0, r[:t + 1] / sc, sc, t))
    for k, s in enumerate(starts):
        seg = r[s:s + L]
        first = max(prefix if k == 0 else stride, min_history - s)
        if len(seg) <= first:
            continue
        sc = VP.initial_scale(seg[:prefix], k=prefix)
        wins.append((s, seg / sc, sc, first))
    i = 0
    while i < len(wins):
        group = [wins[i]]; i += 1
        while i < len(wins) and len(wins[i][1]) == len(group[0][1]) and len(group) < batch_windows:
            group.append(wins[i]); i += 1
        feats, base = make_features(np.stack([g[1] for g in group]))
        out = model(torch.as_tensor(feats).to(device), torch.as_tensor(base).to(device))
        out = {k: v.float().cpu().numpy() for k, v in out.items()}
        for j, (s, seg, sc, first) in enumerate(group):
            n = len(seg); pos = np.arange(first - 1, n - 1)                  # window positions whose prediction (for position+1) is used
            tgt = s + pos + 1
            logw[tgt] = out['logw'][j, pos]; loc[tgt] = out['loc'][j, pos] * sc; lsc[tgt] = out['logscale'][j, pos] + math.log(sc); nu[tgt] = out['nu'][j, pos]
            lat_m[tgt] = out['lat_mean'][j, pos] + 2 * math.log(sc); lat_s[tgt] = np.exp(out['lat_logsd'][j, pos])
    return summarise(r, logw, loc, np.exp(lsc), nu, lat_m, lat_s, levels)

def t_logpdf_np(z, nu):
    from scipy.special import gammaln
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - 0.5 * (nu + 1) * np.log1p(z * z / nu)

def mixture_var_es(w, loc, scale, nu, level, iters=60):
    """VaR and ES (positive losses) of t-mixtures, vectorised over rows: w, loc, scale, nu of shape (T, K)."""
    a = 1 - level
    lo = np.nanmin(loc - 60 * scale, axis=1); hi = np.nanmax(loc + 60 * scale, axis=1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi); cdf = np.sum(w * stdtr(nu, (mid[:, None] - loc) / scale), axis=1)
        lo = np.where(cdf < a, mid, lo); hi = np.where(cdf < a, hi, mid)
    q = 0.5 * (lo + hi); z = (q[:, None] - loc) / scale
    partial = loc * stdtr(nu, z) - scale * (nu + z * z) / (nu - 1) * np.exp(t_logpdf_np(z, nu))
    es = np.sum(w * partial, axis=1) / a
    return -q, -es

def summarise(r, logw, loc, scale, nu, lat_m, lat_s, levels):
    T = len(r); ok = ~np.isnan(logw[:, 0]); w = np.exp(logw)
    out = dict(logw=logw, loc=loc, scale=scale, nu=nu, h=np.full(T, np.nan), h_lat=np.full(T, np.nan), logscore=np.full(T, np.nan), mean=np.full(T, np.nan))
    out['h'][ok] = np.sum(w[ok] * (scale[ok] ** 2 * nu[ok] / (nu[ok] - 2) + loc[ok] ** 2), axis=1)
    out['mean'][ok] = np.sum(w[ok] * loc[ok], axis=1)
    out['h_lat'][ok] = np.exp(lat_m[ok] + 0.5 * lat_s[ok] ** 2)
    z = (r[ok, None] - loc[ok]) / scale[ok]
    from scipy.special import logsumexp
    out['logscore'][ok] = logsumexp(logw[ok] + t_logpdf_np(z, nu[ok]) - np.log(scale[ok]), axis=1)
    for lv in levels:
        q, es = mixture_var_es(w[ok], loc[ok], scale[ok], nu[ok], lv)
        out[f'var{lv:g}'] = np.full(T, np.nan); out[f'es{lv:g}'] = np.full(T, np.nan); out[f'var{lv:g}'][ok] = q; out[f'es{lv:g}'][ok] = es
    return out

# ----------------------------------------------------------------------------- self-test
def selftest():
    torch.manual_seed(0); device = torch.device('cpu'); torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = VolPFN(**SIZES['tiny']); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f'tiny model: {count_params(model)} parameters; sdpa available: ', end='', flush=True)
    b = gen_batch(1, 8, 256); bt = to_dev(b, device)
    with torch.no_grad():
        out = model(bt['feats'], bt['base']); print(model.use_sdpa)
        # (1) causality: perturbing the input at position t0 must not change outputs before t0
        f2 = bt['feats'].clone(); f2[:, 100] += 1.0; o2 = model(f2, bt['base'])
        print(f'causality: max |delta| before the perturbed position {float((o2["loc"][:, :100] - out["loc"][:, :100]).abs().max()):.2e}, after {float((o2["loc"][:, 100:] - out["loc"][:, 100:]).abs().max()):.2e}')
        # (2) sdpa vs manual attention agree
        model.use_sdpa = False; o3 = model(bt['feats'], bt['base']); model.use_sdpa = None
        print(f'manual vs sdpa attention: max |delta| {float((o3["loc"] - out["loc"]).abs().max()):.2e}')
    # (3) training on fresh prior batches reduces the loss on a fixed held-out batch
    vb = to_dev(gen_batch(999, 32, 256), device)
    def held_out():
        with torch.no_grad():
            o = model(vb['feats'], vb['base']); return float(losses(o, vb['r'], vb['logv'], vb['mask'])[0])
    l0 = held_out()
    for s in range(80):
        b = gen_batch(100 + s, 16, 256); bt = to_dev(b, device); out = model(bt['feats'], bt['base']); nll, aux = losses(out, bt['r'], bt['logv'], bt['mask'])
        loss = nll + 0.1 * aux; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    print(f'80 steps on 16 x 256: held-out nll {l0:.4f} -> {held_out():.4f} (must decrease)')
    # (4) inference: windowed prediction equals full-window prediction where contexts coincide; VaR/ES consistency
    model.eval(); rng = np.random.default_rng(3); bb = VP.sample_batch(rng, 1, 700, family='garch'); r = bb['r'][0].astype(float) * 0.013
    p1 = predict_series(model, r, L=700, levels=(0.975,)); p2 = predict_series(model, r, L=256, levels=(0.975,))
    same = slice(1, 256)
    print(f'windowed (L=256) vs single-window (L=700) predictions on the first window: max |delta log h| {np.nanmax(np.abs(np.log(p1["h"][same]) - np.log(p2["h"][same]))):.2e}; later positions differ (context) {np.nanmean(np.abs(np.log(p1["h"][300:]) - np.log(p2["h"][300:]))):.3f}')
    w = np.exp(p1['logw'][1:]); q = -p1['var0.975'][1:]
    cdf = np.sum(w * stdtr(p1['nu'][1:], (q[:, None] - p1['loc'][1:]) / p1['scale'][1:]), axis=1)
    print(f'mixture cdf at -VaR(97.5%): mean {cdf.mean():.5f} (target 0.025), max |err| {np.abs(cdf - 0.025).max():.1e}; ES > VaR everywhere: {np.all(p1["es0.975"][1:] > p1["var0.975"][1:])}')
    # (5) numerical ES check by Monte Carlo on one position
    t = 500; k = rng.choice(model.cfg['K'], 400000, p=w[t - 1] / w[t - 1].sum()); x = p1['loc'][t][k] + p1['scale'][t][k] * rng.standard_t(p1['nu'][t][k])
    qe = np.quantile(x, 0.025); print(f'position {t}: VaR {p1["var0.975"][t]:.5f} vs MC {-qe:.5f}; ES {p1["es0.975"][t]:.5f} vs MC {-x[x <= qe].mean():.5f}; realised r {r[t]:+.4f}, logscore {p1["logscore"][t]:.3f}')
    print('selftest done')

# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--size', default='small', choices=list(SIZES)); t.add_argument('--steps', type=int, default=3000)
    t.add_argument('--tokens', type=int, default=8192, help='tokens (series x length) per batch'); t.add_argument('--lengths', default='256:0.3,512:0.5,1024:0.2')
    t.add_argument('--lr', type=float, default=3e-4); t.add_argument('--warmup', type=int, default=300); t.add_argument('--aux', type=float, default=0.1)
    t.add_argument('--device', default='auto'); t.add_argument('--threads', type=int, default=0); t.add_argument('--workers', type=int, default=2)
    t.add_argument('--seed', type=int, default=1); t.add_argument('--out', default='out_volpfn'); t.add_argument('--name', default=None); t.add_argument('--resume', default=None)
    t.add_argument('--log-every', type=int, default=25); t.add_argument('--eval-every', type=int, default=500); t.add_argument('--val-T', type=int, default=1024)
    t.add_argument('--overlay', default=None, help='contamination probabilities, e.g. jumps=0,ar=0,drift=0,stale=0 (default: volpfn_prior.OVERLAY_P)')
    t.add_argument('--prior', default=None, help='family weights, e.g. garch=0.2,sv=0.3,...'); t.add_argument('--no-univol-val', dest='univol_val', action='store_false')
    p = sub.add_parser('predict'); p.add_argument('--ckpt', required=True); p.add_argument('--data', required=True); p.add_argument('--col', default=None); p.add_argument('--prices', action='store_true')
    p.add_argument('--levels', type=float, nargs='*', default=[0.975, 0.99]); p.add_argument('--L', type=int, default=1024); p.add_argument('--out', default=None); p.add_argument('--device', default='auto')
    sub.add_parser('selftest')
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a)
    elif a.cmd == 'selftest':
        selftest()
    else:
        import pandas as pd
        df = pd.read_csv(a.data); df = df.set_index(df.columns[0]); col = a.col or df.columns[0]; x = pd.to_numeric(df[col], errors='coerce').dropna()
        r = (x.pct_change().dropna() if a.prices else x).to_numpy(dtype=float); idx = x.index[1:] if a.prices else x.index
        model = load_model(a.ckpt, a.device); t0 = time.time(); out = predict_series(model, r, L=a.L, levels=tuple(a.levels)); ok = ~np.isnan(out['h'])
        print(f'{col}: {len(r)} returns in {time.time() - t0:.1f}s; QLIKE {np.mean(r[ok] ** 2 / out["h"][ok] + np.log(out["h"][ok])):.4f} (mixture second moment), {np.mean(r[ok] ** 2 / out["h_lat"][ok] + np.log(out["h_lat"][ok])):.4f} (latent head); mean log score {np.nanmean(out["logscore"]):.4f}')
        for lv in a.levels:
            hit = -r[ok] > out[f'var{lv:g}'][ok]; print(f'  VaR {lv:g}: violation rate {hit.mean():.4f} (nominal {1 - lv:.4f}); ES ratio {(-r[ok][hit]).mean() / out[f"es{lv:g}"][ok][hit].mean() if hit.sum() >= 5 else float("nan"):.3f}')
        if a.out:
            res = pd.DataFrame({'return': r, 'variance': out['h'], 'sigma': np.sqrt(out['h']), 'variance_latent': out['h_lat'], **{k: v for k, v in out.items() if k.startswith(('var', 'es'))}, 'logscore': out['logscore']}, index=idx)
            res.to_csv(a.out); print('wrote', a.out)

if __name__ == '__main__':
    main()
