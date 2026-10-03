"""Generate raw synthetic histories without legacy prefix normalisation."""
import numpy as np
from . import prior as VP

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
