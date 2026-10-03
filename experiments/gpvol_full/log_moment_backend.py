"""Observe GP-Vol forecast moments in log space without changing the filter.

``make_log_moment_wrapper(original_mixture, sink)`` returns the original
``(h, logdensity)`` unchanged and calls ``sink(log_h, diagnostics)`` once.
Finite positive h uses log(h); overflow/underflow uses the same mixture's
weighted log-sum-exp, with no clipping, renormalization, or particle removal.
Exactly zero weights make zero contribution.

A runner can add ``f.out['log_h'] = np.full(len(f.x), np.nan)`` and bind a sink
that assigns ``f.out['log_h'][f.t]``.  RAPCF's existing save/load methods already
persist additional out entries; its original source and OUT_KEYS need no edit.
The caller must export/score log moments explicitly: the original returned h
can still be inf, nan, or zero when float64 cannot represent its evaluation.
"""

import numpy as np


def _components(w, mu, s2):
    w, mu, s2 = (np.asarray(value, dtype=float) for value in (w, mu, s2))
    if w.ndim != 1 or mu.shape != w.shape or s2.shape != w.shape or not w.size:
        raise ValueError("Weights, means and variances must be matching nonempty vectors")
    if not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
        raise FloatingPointError("Mixture weights must be finite, nonnegative, and not all zero")
    positive = w > 0
    if (not np.isfinite(mu[positive]).all() or not np.isfinite(s2[positive]).all()
            or (s2[positive] < 0).any()):
        raise FloatingPointError("A positive-weight component has invalid mean or variance")
    return w, mu, s2, positive


def log_weighted_lognormal_moment(w, mu, s2):
    """Compute log(sum(w * exp(mu + s2/2))) without forming exponentials."""
    w, mu, s2, positive = _components(w, mu, s2)
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.log(w[positive]) + mu[positive] + 0.5 * s2[positive]
        result = float(np.logaddexp.reduce(terms))
    if not np.isfinite(result):
        raise FloatingPointError("The weighted log moment itself is not finite")
    return result


def make_log_moment_wrapper(original_mixture, sink):
    """Wrap a predictive-mixture callable; sink receives log_h and JSON-safe metadata.

    No NumPy RNG calls occur.  Neither the original h nor logdensity is changed,
    even when h is not representable.  Invalid positive-weight components or
    an unrepresentable *log* moment fail explicitly.  Density failures are not
    repaired; callers must retain their existing density validation.
    """
    if not callable(original_mixture) or not callable(sink):
        raise TypeError("original_mixture and sink must be callable")
    log_max = float(np.log(np.finfo(float).max))
    log_min = float(np.log(np.nextafter(0.0, 1.0)))

    def wrapped(w, mu, s2, x_obs):
        _, _, _, positive = _components(w, mu, s2)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            h, logdensity = original_mixture(w, mu, s2, x_obs)
        if np.isfinite(h) and h > 0:
            log_h = float(np.log(h))
            path = "original_finite_moment"
        else:
            log_h = log_weighted_lognormal_moment(w, mu, s2)
            path = "weighted_logsumexp"
        kind = ("finite_positive" if np.isfinite(h) and h > 0 else
                "zero" if h == 0 else "nan" if np.isnan(h) else
                "positive_infinity" if h > 0 else "negative_or_invalid")
        sink(log_h, dict(moment_path=path, original_h_kind=kind,
                         float64_moment_representable=bool(log_min <= log_h <= log_max),
                         positive_weights=int(positive.sum()), zero_weights=int((~positive).sum())))
        return h, logdensity

    return wrapped
