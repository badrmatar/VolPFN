"""Alternative training distributions; frozen historical sources are imported only.

The independent Matérn arm is a declared alternative, not a representation of
all generic forecasting priors. Its clean log variance is a stationary Gaussian
process drawn from its finite-dimensional covariance, with no return feedback.
Overlaid auxiliary labels inherit the historical simulator's limitations.
"""
from functools import lru_cache
import numpy as np
from volpfn import prior as VP
from experiments.seed_replication import causal

ARMS = ('broad', 'garch', 'matern')
SMOOTHNESS = (.1, .5, 1.5)
LENGTH_SCALES = (30., 120., 500.)
LOG_VARIANCE_SD = (.3, 1.5)
SPECIFICATION = {
    'arms': list(ARMS),
    'broad_weights': dict(VP.PRIOR_WEIGHTS),
    'garch_weights': {'garch': 1.},
    'matern': {
        'smoothness': list(SMOOTHNESS), 'length_scales': list(LENGTH_SCALES),
        'cell_weights': 'uniform over the nine smoothness/length-scale pairs',
        'log_variance_sd_uniform': list(LOG_VARIANCE_SD),
        'log_variance_mean': 0.,
        'initialization': 'stationary finite-dimensional Gaussian draw',
        'sampler': 'Cholesky; no eigenvalue clipping or jitter; fail on non-PD covariance',
        'interpretation': 'independent latent scale; overlaps with broad rough/SV components',
    },
    'innovation': 'VP.draw_law and VP.law_sample, identical marginal law across arms',
    'overlay_probabilities': dict(VP.OVERLAY_P),
    'auxiliary_label_scope': 'inherited logv; not exact observed-return conditional variance for every overlay combination',
}


@lru_cache(maxsize=36)
def covariance_factor(length, smoothness, length_scale):
    """Declared Matérn covariance, factored without silently changing the law."""
    if length < 2 or smoothness <= 0 or length_scale <= 0:
        raise ValueError('positive kernel parameters and length >= 2 required')
    lag = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    covariance = VP.matern_cov(lag, smoothness, length_scale)
    factor = np.linalg.cholesky(covariance)
    if not np.isfinite(factor).all():
        raise FloatingPointError('nonfinite_matern_factor')
    factor.setflags(write=False)
    return factor


def matern_raw(seed, batch_size, length, clean=False):
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not isinstance(length, (int, np.integer)) or length < 2:
        raise ValueError('length must be an integer >= 2')
    rng = np.random.default_rng(seed)
    nu, gamma = VP.draw_law(rng, batch_size)
    h = rng.choice(SMOOTHNESS, batch_size)
    ell = rng.choice(LENGTH_SCALES, batch_size)
    sd = rng.uniform(*LOG_VARIANCE_SD, batch_size)
    normal = rng.standard_normal((batch_size, length))
    logv = np.empty_like(normal)
    for smoothness in SMOOTHNESS:
        for length_scale in LENGTH_SCALES:
            idx = np.flatnonzero((h == smoothness) & (ell == length_scale))
            if len(idx):
                factor = covariance_factor(length, smoothness, length_scale)
                logv[idx] = sd[idx, None] * (normal[idx] @ factor.T)
    innovation = VP.law_sample(rng, nu[:, None], gamma[:, None], (batch_size, length))
    r = np.exp(.5 * logv) * innovation
    overlay = {k: 0. if clean else v for k, v in VP.OVERLAY_P.items()}
    r, logv, mask, descriptions = VP.apply_overlays(rng, r, logv, overlay)
    metadata = [dict(family='independent_matern', nu=float(nu[i]), gamma=float(gamma[i]),
                     params=dict(H=float(h[i]), ell=float(ell[i]), sd=float(sd[i]), mean=0.),
                     overlay=descriptions[i], scale=1., clean=not (
                         descriptions[i]['jumps'] or descriptions[i]['ar'] != 0 or
                         descriptions[i]['drift'] != 0 or descriptions[i]['stale'] > 0))
                for i in range(batch_size)]
    if not np.isfinite(r).all() or not np.isfinite(logv).all():
        raise FloatingPointError('nonfinite_matern_simulation')
    return dict(r=r, logv=logv, mask=mask.astype(np.float32), meta=metadata)


def raw_batch(arm, seed, batch_size, length, clean=False):
    if arm not in ARMS:
        raise ValueError('unknown prior arm: ' + str(arm))
    if arm == 'matern':
        return matern_raw(seed, batch_size, length, clean)
    # family='garch' would disable overlays in the shared generator.
    weights = VP.PRIOR_WEIGHTS if arm == 'broad' else {'garch': 1.}
    return causal.raw_batch(seed, batch_size, length, weights=weights, clean=clean)


def prepared_batch(arm, seed, batch_size, length):
    return causal.prepare_batch(raw_batch(arm, seed, batch_size, length))


def covariance_audit(length):
    rows = []
    lag = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    for h in SMOOTHNESS:
        for ell in LENGTH_SCALES:
            factor = covariance_factor(length, h, ell)
            reference = VP.matern_cov(lag, h, ell)
            rows.append(dict(length=length, smoothness=h, length_scale=ell, jitter=0.,
                             max_reconstruction_error=float(np.max(np.abs(factor @ factor.T - reference))),
                             min_cholesky_diagonal=float(np.diag(factor).min())))
    return rows
