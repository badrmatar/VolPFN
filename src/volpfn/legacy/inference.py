"""Retained prefix-normalised forecasting, for legacy checkpoints only."""
import math
import numpy as np
import torch
from .. import prior as VP
from ..model import VolPFN
from ..device import pick_device
from ..distributions import summarise
from .features import make_features

def load_model(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = VolPFN(**ck['cfg'])
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m.to(pick_device(device) if isinstance(device, str) else device)


@torch.no_grad()
def predict_series(
    model, r, L=1024, stride=None, levels=(0.975, 0.99), device=None, batch_windows=4, min_history=1
):
    """One-step-ahead predictive distributions for a whole return series (any scale), causally, from sliding windows of
    length L with stride L/2 (every forecast after the first window uses at least L/2 of context; each window is
    standardised by an already observed prefix of at most 100 returns). Early forecasts use separate growing
    prefixes so their scale cannot depend on future observations. Set min_history to skip unneeded warm-up
    forecasts; this only changes which entries are returned, not retained forecast values.
    Returns dict of arrays aligned with r: h (mixture second
    moment, the QLIKE Bayes act), h_lat (posterior mean of the latent variance from the auxiliary head), logscore (log
    predictive density of the realised return), var<level> and es<level> (positive losses), and the mixture parameters
    logw, loc, scale, nu of shape (T, K).  Entry t is the forecast for r[t] made from r[:t]; entry 0 is NaN.
    """
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
    T = len(r)
    device = next(model.parameters()).device if device is None else device
    K = model.cfg['K']
    prefix = min(100, stride)
    logw = np.full((T, K), np.nan)
    loc = np.full((T, K), np.nan)
    lsc = np.full((T, K), np.nan)
    nu = np.full((T, K), np.nan)
    lat_m = np.full(T, np.nan)
    lat_s = np.full(T, np.nan)
    # Create a window as soon as it owns a forecast. Waiting until the previous
    # full window ends would retrospectively change outputs when stride < L/2.
    starts = [0] + list(range(stride, max(T - stride, 0), stride))
    wins = []
    for t in range(min_history, min(prefix, T)):
        sc = VP.initial_scale(r[:t], k=t)
        wins.append((0, r[: t + 1] / sc, sc, t))
    for k, s in enumerate(starts):
        seg = r[s : s + L]
        first = max(prefix if k == 0 else stride, min_history - s)
        if len(seg) <= first:
            continue
        sc = VP.initial_scale(seg[:prefix], k=prefix)
        wins.append((s, seg / sc, sc, first))
    i = 0
    while i < len(wins):
        group = [wins[i]]
        i += 1
        while i < len(wins) and len(wins[i][1]) == len(group[0][1]) and len(group) < batch_windows:
            group.append(wins[i])
            i += 1
        feats, base = make_features(np.stack([g[1] for g in group]))
        out = model(torch.as_tensor(feats).to(device), torch.as_tensor(base).to(device))
        out = {k: v.float().cpu().numpy() for k, v in out.items()}
        for j, (s, seg, sc, first) in enumerate(group):
            n = len(seg)
            pos = np.arange(
                first - 1, n - 1
            )  # window positions whose prediction (for position+1) is used
            tgt = s + pos + 1
            logw[tgt] = out['logw'][j, pos]
            loc[tgt] = out['loc'][j, pos] * sc
            lsc[tgt] = out['logscale'][j, pos] + math.log(sc)
            nu[tgt] = out['nu'][j, pos]
            lat_m[tgt] = out['lat_mean'][j, pos] + 2 * math.log(sc)
            lat_s[tgt] = np.exp(out['lat_logsd'][j, pos])
    return summarise(r, logw, loc, np.exp(lsc), nu, lat_m, lat_s, levels)
