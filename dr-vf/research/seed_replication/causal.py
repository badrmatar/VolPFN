"""Causal, within-window normalization for the new seed-replication procedure.

This is deliberately separate from the retained checkpoint's preprocessing.
At input position j, every feature and output unit uses only r[:j+1]. The
next target is r[j+1] / s_j, where s_j is the nonzero prefix RMS. At the first
observation (j=0), s_j is abs(r[0]) or one for a zero. All-zero prefixes use
one. Zero observations do not enter the RMS denominator or update EWMAs.

The three EWMAs start at the first observed nonzero squared return, then use
the existing half-lives 5, 20 and 100. They are maintained in raw units and
divided by s_j**2 for the features and density head. Running scale, EWMAs and
position restart at each inference window, exactly as at a training task's
start. Window ownership is the retained inference convention. This is a new
training AND inference specification, not a claim of identical preprocessing
to the retained procedure.

All prefix accumulations use float64 log arithmetic. No standardized float32
legacy sample is reconstructed; raw_batch calls its simulators and overlays
directly before the legacy whole-prefix normalization would occur.
"""
import math

import numpy as np
import torch

import volpfn as V
import volpfn_prior as VP


CAUSAL_SPEC = 'running_nonzero_rms_window_v1'
NORMALIZATION_SPEC = {
    'id': CAUSAL_SPEC,
    'scale': 'RMS of observed nonzero returns within the current window',
    'all_zero_prefix_scale': 1.0,
    'target_units': 'r[j+1]/s[j], logv[j+1]-2*log(s[j])',
    'density_jacobian': 'raw NLL = standardized NLL + log(s[j])',
    'ewma_half_lives': list(V.HALF_LIVES),
    'ewma_initialization': 'first observed nonzero squared return; unit before it',
    'zero_handling': 'exclude from RMS count and skip EWMA updates',
    'window_state': 'reset scale, EWMAs and position for each window',
    'feature_count': V.N_FEAT,
}


def raw_batch(seed, B, L, weights=None, overlay_p=None, clean=False,
              family=None, nu=None, gamma=None):
    """Simulate legacy prior families/overlays without legacy standardization.

    r and logv remain float64 in simulator units. Extra clean/family/law
    options permit synthetic checks and do not change default training draws.
    Metadata scale=1 means that the metadata's original parameters already
    describe the returned raw units.
    """
    if not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError('B must be a positive integer')
    if not isinstance(L, (int, np.integer)) or L < 2:
        raise ValueError('L must be an integer >= 2')
    rng = np.random.default_rng(seed)
    weights = dict(VP.PRIOR_WEIGHTS if weights is None else weights)
    overlay_p = dict(VP.OVERLAY_P if overlay_p is None else overlay_p)
    if clean or family is not None:
        overlay_p = {k: 0.0 for k in overlay_p}
    fams = list(weights)
    p = np.asarray([weights[f] for f in fams], dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('family weights must be finite, nonnegative and nonzero')
    if any(f not in VP.SIMULATORS for f in fams):
        raise ValueError('unknown simulator family')
    p /= p.sum()
    fam_idx = (rng.choice(len(fams), size=B, p=p) if family is None
               else np.full(B, fams.index(family)))
    nus, gammas = VP.draw_law(rng, B)
    if nu is not None:
        nus = np.full(B, float(nu))
    if gamma is not None:
        gammas = np.full(B, float(gamma))
    r = np.empty((B, L), dtype=np.float64)
    lv = np.empty((B, L), dtype=np.float64)
    meta = [None] * B
    for j, f in enumerate(fams):
        idx = np.flatnonzero(fam_idx == j)
        if not len(idx):
            continue
        rr, ll, params = VP.SIMULATORS[f](rng, len(idx), L, nus[idx], gammas[idx])
        r[idx], lv[idx] = rr, ll
        for i, pr in zip(idx, params):
            meta[i] = dict(family=f, nu=float(nus[i]), gamma=float(gammas[i]), params=pr)
    r, lv, mask, desc = VP.apply_overlays(rng, r, lv, overlay_p)
    for i in range(B):
        d = desc[i]
        meta[i].update(overlay=d, scale=1.0,
                       clean=not (d['jumps'] or d['ar'] != 0 or d['drift'] != 0 or d['stale'] > 0))
    if not np.isfinite(r).all() or not np.isfinite(lv).all():
        raise FloatingPointError('raw simulator produced nonfinite observations or latent targets')
    return dict(r=r, logv=lv, mask=mask.astype(np.float32), meta=meta)


def feature_batch(r):
    """Prepare B x T raw observations causally inside each row/window.

    output_scale/output_log_scale here have shape B x T: the scale for the
    NEXT output at each input position. prepare_batch removes the unused final
    entry to align those arrays with its B x (T-1) training targets.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 1 or not np.isfinite(r).all():
        raise ValueError('r must be a finite B x T array with T >= 1')
    B, T = r.shape
    nz = r != 0
    logabs = np.full(r.shape, -np.inf)
    np.log(np.abs(r), out=logabs, where=nz)
    logsum = np.logaddexp.accumulate(2 * logabs, axis=1)
    count = np.cumsum(nz, axis=1)
    log_scale = np.zeros(r.shape)
    np.subtract(logsum, np.log(np.maximum(count, 1)), out=log_scale, where=count > 0)
    log_scale *= 0.5
    # Computing r/s as exp(log|r|-log s) avoids overflow in s^2.
    standardized = np.zeros(r.shape)
    standardized[nz] = np.sign(r[nz]) * np.exp(logabs[nz] - log_scale[nz])
    feats = np.empty((B, T, V.N_FEAT), dtype=np.float32)
    feats[..., 0] = np.arcsinh(standardized)
    feats[..., 1] = np.log(standardized * standardized + 1e-6) / 4
    feats[..., 2] = ~nz
    feats[..., 3] = np.sign(r)
    base = np.empty((B, T), dtype=np.float32)
    for j, half_life in enumerate(V.HALF_LIVES):
        lam = 0.5 ** (1.0 / half_life)
        log_lam, log_one_minus_lam = math.log(lam), math.log1p(-lam)
        log_e = np.zeros(B)
        seen = np.zeros(B, dtype=bool)
        col = np.empty((B, T), dtype=np.float64)
        for t in range(T):
            first = nz[:, t] & ~seen
            update = nz[:, t] & seen
            log_e[first] = 2 * logabs[first, t]
            log_e[update] = np.logaddexp(log_lam + log_e[update],
                                        log_one_minus_lam + 2 * logabs[update, t])
            seen |= nz[:, t]
            col[:, t] = log_e - 2 * log_scale[:, t]
        feats[..., 4 + j] = col / 4
        if half_life == 20.0:
            base[:] = col
    feats[..., 7] = np.log1p(np.arange(T))[None, :] / 8
    scale = np.exp(log_scale)
    return dict(feats=feats, base=base, output_scale=scale,
                output_log_scale=log_scale.astype(np.float32))


def prepare_batch(raw):
    """Return inputs B x T and matched one-step targets B x (T-1).

    The mask belongs to the target, not the observed input. Never normalize
    the target using its own value. Raw log density includes the scale Jacobian.
    """
    r = np.asarray(raw['r'], dtype=np.float64)
    lv = np.asarray(raw['logv'], dtype=np.float64)
    mask = np.asarray(raw['mask'], dtype=np.float32)
    if r.ndim != 2 or r.shape[1] < 2 or lv.shape != r.shape or mask.shape != r.shape:
        raise ValueError('r, logv and mask must share B x T shape with T >= 2')
    if not np.isfinite(lv).all() or not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError('latent targets and nonnegative masks must be finite')
    f = feature_batch(r)
    scale = f['output_scale'][:, :-1]
    log_scale = np.log(scale)
    target_r = r[:, 1:] / scale
    target_logv = lv[:, 1:] - 2 * log_scale
    with np.errstate(over='ignore'):
        target_r = target_r.astype(np.float32)
        target_logv = target_logv.astype(np.float32)
    if not np.isfinite(target_r).all() or not np.isfinite(target_logv).all():
        raise FloatingPointError('nonfinite normalized next-step target')
    return dict(feats=f['feats'], base=f['base'],
                target_r=target_r,
                target_logv=target_logv,
                target_mask=mask[:, 1:].copy(),
                output_scale=scale,
                output_log_scale=log_scale.astype(np.float32))


def to_dev(batch, device):
    """Tensorize only model/loss arrays; scales retain float64 for inspection."""
    keys = ('feats', 'base', 'target_r', 'target_logv', 'target_mask',
            'output_scale', 'output_log_scale')
    return {k: torch.as_tensor(batch[k], device=device) for k in keys}


def losses(out, batch):
    """Mean raw-unit predictive NLL and latent log-variance auxiliary loss."""
    prediction = {k: v[:, :-1] for k, v in out.items()}
    nll = -V.mixture_logpdf(prediction, batch['target_r']) + batch['output_log_scale']
    m, ls = prediction['lat_mean'], prediction['lat_logsd']
    aux = 0.5 * ((batch['target_logv'] - m) * torch.exp(-ls)) ** 2 + ls
    w = batch['target_mask']
    den = w.sum().clamp_min(1.0)
    return (nll * w).sum() / den, (aux * w).sum() / den


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
