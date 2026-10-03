"""Features for retained prefix-normalised checkpoints."""
import numpy as np
from ..config import N_FEAT, HALF_LIVES

def make_features(r):
    """r: (B, T) standardised returns -> feats (B, T, N_FEAT) float32, base_logvar (B, T) float32 = log EWMA_20(r^2)."""
    r = np.asarray(r, dtype=np.float64)
    B, T = r.shape
    feats = np.empty((B, T, N_FEAT), dtype=np.float32)
    feats[:, :, 0] = np.arcsinh(r)
    feats[:, :, 1] = np.log(r * r + 1e-6) / 4
    feats[:, :, 2] = r == 0
    feats[:, :, 3] = np.sign(r)
    nz = r != 0
    for j, hl in enumerate(HALF_LIVES):
        lam = 0.5 ** (1.0 / hl)
        e = np.ones(B)
        col = np.empty((B, T))
        for t in range(T):
            e = np.where(nz[:, t], lam * e + (1 - lam) * r[:, t] ** 2, e)
            col[:, t] = e
        feats[:, :, 4 + j] = np.log(col) / 4
        if hl == 20.0:
            base = np.log(col).astype(np.float32)
    feats[:, :, 7] = np.log1p(np.arange(T))[None, :] / 8
    return feats, base
