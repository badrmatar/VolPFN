"""Running causal scale, features and aligned training targets."""
import math
import numpy as np
import torch
from . import legacy as V

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
