"""Fissler-Ziegel scoring for the (VaR, ES) pair, its identification function, closed-form tails,
and the empirical (hindsight) recalibration that defines miscalibration.

Conventions: returns y, lower tail, level alpha in (0, 1/2); VaR_alpha = F^{-1}(alpha) (negative for
losses), ES_alpha = E[y | y <= VaR_alpha] <= VaR_alpha < 0. The FZ0 loss of Patton, Ziegel and Chen
(2019) is used throughout (the same formula as gp-fz/gp_fz_phase1.py); it is strictly consistent for
the pair and positively homogeneous of degree zero up to an additive log-scale term, so differences
between forecasters are scale-free.
"""
import numpy as np
from scipy import stats

E_FLOOR = 1e-8


def fz0(y, v, e, alpha):
    """FZ0 loss per observation; requires e < 0 (clipped at -E_FLOOR)."""
    y = np.asarray(y, float); v = np.asarray(v, float); e = np.minimum(np.asarray(e, float), -E_FLOOR)
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (alpha * e) + v / e + np.log(-e) - 1.0


def identification(y, v, e, alpha):
    """Strict identification function of the pair: both components have mean zero at the truth."""
    y = np.asarray(y, float); v = np.asarray(v, float); e = np.asarray(e, float)
    hit = (y <= v).astype(float)
    return hit - alpha, e - v + hit * (v - y) / alpha


def student_t_var_es(alpha, nu):
    """(VaR, ES) of the unit-variance Student-t(nu) at level alpha (nu > 2)."""
    q = stats.t.ppf(alpha, nu)
    es_raw = -(nu + q * q) / (nu - 1.0) * stats.t.pdf(q, nu) / alpha
    s = np.sqrt((nu - 2.0) / nu)
    return float(q * s), float(es_raw * s)


def normal_var_es(alpha):
    q = stats.norm.ppf(alpha)
    return float(q), float(-stats.norm.pdf(q) / alpha)


def empirical_var_es(z, alpha, assume_sorted=False):
    """(VaR, ES) of the empirical distribution of z: the lower alpha-quantile and the mean of the
    lowest alpha fraction with a fractional weight on the boundary order statistic, which is the
    minimiser of the empirical FZ0 sum over constant (v, e) with e < 0."""
    z = np.asarray(z, float) if assume_sorted else np.sort(np.asarray(z, float)); n = len(z)
    if n == 0:
        raise ValueError('empty sample')
    k = alpha * n; m = int(np.floor(k))
    if m == 0:
        v = z[0]; es = z[0]
    else:
        v = z[m - 1] if k == m else z[m]
        es = (z[:m].sum() + (k - m) * (z[m] if m < n else z[m - 1])) / k
    return float(v), float(min(es, v))


def hindsight_recalibration(y, v, e, alpha, groups):
    """Replace the forecasts in each group by the empirical (VaR, ES) of the outcomes in that group,
    with full hindsight. Returns the recalibrated arrays. The mean FZ0 gain over the original is the
    miscalibration MCB of the original forecaster with respect to that grouping (>= 0 up to the
    boundary handling of the empirical ES)."""
    y = np.asarray(y, float); groups = np.asarray(groups)
    vh = np.array(v, float, copy=True); eh = np.array(e, float, copy=True)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        vg, eg = empirical_var_es(y[idx], alpha)
        vh[idx] = vg; eh[idx] = eg
    return vh, eh


def hindsight_ratio_recalibration(y, v, alpha):
    """Hindsight recalibration in the forecaster's own scale: outcomes standardised by |v_t|, one
    empirical (VaR, ES) for the standardised sample, mapped back through |v_t|."""
    y = np.asarray(y, float); s = np.abs(np.asarray(v, float))
    q, es = empirical_var_es(y / s, alpha)
    return q * s, es * s
