"""Corrected within-window causal forecasting."""
import numpy as np
import torch
from . import legacy as V
from .preprocessing import CAUSAL_SPEC, NORMALIZATION_SPEC, feature_batch

def load_model(path, device='cpu'):
    """Reject retained/other-normalization checkpoints at the new API boundary."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('causal_spec') != CAUSAL_SPEC:
        raise ValueError('checkpoint does not declare causal_spec=' + CAUSAL_SPEC)
    if ck.get('normalization_spec') != NORMALIZATION_SPEC:
        raise ValueError('checkpoint normalization_spec does not match the frozen causal specification')
    model = V.VolPFN(**ck['cfg'])
    model.load_state_dict(ck['state_dict'])
    model.causal_spec = CAUSAL_SPEC
    model.eval()
    return model.to(V.pick_device(device) if isinstance(device, str) else device)


@torch.no_grad()
def predict_series(model, r, L=1024, stride=None, levels=(0.975, 0.99),
                   device=None, batch_windows=4, min_history=100):
    """Raw-unit predictions with new within-window causal units and old owners.

    Entry t forecasts r[t] from strictly earlier observations. With overlapping
    windows, each later window overwrites targets starting at start+stride,
    just as the retained implementation. Future windows cannot alter an old
    target because a new owner is instantiated as soon as it owns that target.
    Warmup only suppresses returned rows, never changes their values.
    """
    r = np.asarray(r, dtype=np.float64)
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
    logw, loc, lsc, nu = [np.full((T, K), np.nan) for _ in range(4)]
    lat_m, lat_s = np.full(T, np.nan), np.full(T, np.nan)
    starts = [0] + list(range(stride, max(T - stride, 0), stride))
    wins = []
    for k, start in enumerate(starts):
        seg = r[start:start + L]
        first = max(1 if k == 0 else stride, min_history - start)
        if len(seg) > first:
            wins.append((start, seg, first))
    i = 0
    was_training = model.training
    model.eval()
    try:
        while i < len(wins):
            group = [wins[i]]
            i += 1
            while (i < len(wins) and len(wins[i][1]) == len(group[0][1])
                   and len(group) < batch_windows):
                group.append(wins[i])
                i += 1
            f = feature_batch(np.stack([g[1] for g in group]))
            out = model(torch.as_tensor(f['feats'], device=device),
                        torch.as_tensor(f['base'], device=device))
            out = {k: v.float().cpu().numpy() for k, v in out.items()}
            for j, (start, seg, first) in enumerate(group):
                pos = np.arange(first - 1, len(seg) - 1)
                tgt = start + pos + 1
                sc = f['output_scale'][j, pos]
                log_sc = np.log(sc)
                logw[tgt] = out['logw'][j, pos]
                loc[tgt] = out['loc'][j, pos] * sc[:, None]
                lsc[tgt] = out['logscale'][j, pos] + log_sc[:, None]
                nu[tgt] = out['nu'][j, pos]
                lat_m[tgt] = out['lat_mean'][j, pos] + 2 * log_sc
                lat_s[tgt] = np.exp(out['lat_logsd'][j, pos])
    finally:
        model.train(was_training)
    return V.summarise(r, logw, loc, np.exp(lsc), nu, lat_m, lat_s, levels)
