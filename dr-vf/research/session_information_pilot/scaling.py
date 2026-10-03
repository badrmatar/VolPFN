"""Frozen-density scale corrections for the session-information pilot.

All arrays describe forecasts already available before their target outcome.
This module does not enforce dates, select observations, construct features, or
update neural weights. The caller must supply development-only data to fitting.
For log_s = c*x the corrected density is f(r / exp(log_s)) / exp(log_s).
Every component location and scale changes; weights and degrees of freedom do
not. In particular, log second moment changes by 2*c*x, not c*x.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp, stdtr, stdtrit


_LOG_WEIGHT_TOLERANCE = 1e-5  # Accommodate stored float32 log-softmax roundoff.


def _mixture_arrays(logw, loc, scale, nu):
    names = ("logw", "loc", "scale", "nu")
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (logw, loc, scale, nu))
    shape = arrays[0].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Mixture parameters must have nonempty shape (N, K).")
    for name, value in zip(names, arrays):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite and have shape {shape}.")
    lw, mu, sigma, degrees = arrays
    if np.any(sigma <= 0):
        raise ValueError("Every component scale must be positive.")
    if np.any(degrees <= 2):
        raise ValueError("Every nu must exceed 2 for finite second moments.")
    normalizer = logsumexp(lw, axis=1, keepdims=True)
    if np.any(np.abs(normalizer) > _LOG_WEIGHT_TOLERANCE):
        raise ValueError("Mixture log weights must already be normalized.")
    # Remove only accepted storage roundoff so numerical densities integrate to 1.
    return lw - normalizer, mu, sigma, degrees


def _vector(value, n, name, *, allow_scalar=False):
    value = np.asarray(value, dtype=np.float64)
    if allow_scalar and value.ndim == 0:
        value = np.full(n, float(value))
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector of length {n}.")
    return value


def _t_logpdf(z, nu):
    with np.errstate(divide="ignore", over="raise", invalid="raise"):
        # Avoid squaring very large standardized residuals.
        log_kernel = np.logaddexp(0.0, 2.0 * np.log(np.abs(z)) - np.log(nu))
        return (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - 0.5 * (nu + 1.0) * log_kernel
        )


def _logpdf_unchecked(r, logw, loc, scale, nu):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        z = (r[:, None] - loc) / scale
        answer = logsumexp(logw + _t_logpdf(z, nu) - np.log(scale), axis=1)
    if not np.isfinite(answer).all():
        raise ValueError("Mixture log density is not numerically finite.")
    return answer


def mixture_logpdf(r, logw, loc, scale, nu):
    """Return an N-vector of log densities for finite Student-t mixtures."""
    arrays = _mixture_arrays(logw, loc, scale, nu)
    r = _vector(r, arrays[0].shape[0], "r")
    try:
        return _logpdf_unchecked(r, *arrays)
    except FloatingPointError as exc:
        raise ValueError("Mixture log density exceeds numerical range.") from exc


def scale_mixture(logw, loc, scale, nu, log_scale):
    """Scale an entire return distribution, returning parameters and moments.

    ``log_scale`` may be scalar or length N. Returned ``variance`` is centered;
    ``second_moment`` is the appropriate forecast of squared returns. Inputs are
    never mutated. No clipping, refitting, or implicit recentering is performed.
    """
    logw, loc, scale, nu = _mixture_arrays(logw, loc, scale, nu)
    log_scale = _vector(log_scale, len(logw), "log_scale", allow_scalar=True)
    try:
        with np.errstate(over="raise", under="ignore", divide="raise", invalid="raise"):
            multiplier = np.exp(log_scale)[:, None]
            adjusted_loc, adjusted_scale = loc * multiplier, scale * multiplier
            weights = np.exp(logw)
            mean = np.sum(weights * adjusted_loc, axis=1)
            component_variance = adjusted_scale**2 * nu / (nu - 2.0)
            variance = np.sum(
                weights * (component_variance + (adjusted_loc - mean[:, None]) ** 2), axis=1
            )
            second_moment = np.sum(weights * (component_variance + adjusted_loc**2), axis=1)
    except FloatingPointError as exc:
        raise ValueError("Scaled mixture exceeds numerical range.") from exc
    if (
        np.any(adjusted_scale <= 0)
        or np.any(variance <= 0)
        or np.any(second_moment <= 0)
        or not all(np.isfinite(v).all() for v in (adjusted_loc, adjusted_scale, mean, variance, second_moment))
    ):
        raise ValueError("Scaled mixture must retain finite, positive scales and moments.")
    return {
        "logw": logw.copy(), "loc": adjusted_loc, "scale": adjusted_scale,
        "nu": nu.copy(), "mean": mean, "variance": variance,
        "second_moment": second_moment,
    }


def mixture_lower_tail(logw, loc, scale, nu, alpha):
    """Return signed return quantile and ES at probability alpha, not loss VaR.

    Uses a deterministic bisection bracketed by component alpha-quantiles and
    the analytical Student-t truncated first moment. Negate both outputs when
    positive-loss VaR/ES conventions are needed. No Gaussian tail approximation.
    """
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    logw, loc, scale, nu = _mixture_arrays(logw, loc, scale, nu)
    weights = np.exp(logw)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            component_quantiles = loc + scale * stdtrit(nu, alpha)
            lo, hi = component_quantiles.min(axis=1), component_quantiles.max(axis=1)
            for _ in range(80):
                mid = 0.5 * lo + 0.5 * hi
                cdf = np.sum(weights * stdtr(nu, (mid[:, None] - loc) / scale), axis=1)
                lo, hi = np.where(cdf < alpha, mid, lo), np.where(cdf >= alpha, mid, hi)
            quantile = 0.5 * lo + 0.5 * hi
            z = (quantile[:, None] - loc) / scale
            partial = (
                loc * stdtr(nu, z)
                - scale * (nu + z**2) / (nu - 1.0) * np.exp(_t_logpdf(z, nu))
            )
            expected_shortfall = np.sum(weights * partial, axis=1) / alpha
    except FloatingPointError as exc:
        raise ValueError("Tail calculation exceeds numerical range.") from exc
    if not np.isfinite(quantile).all() or not np.isfinite(expected_shortfall).all():
        raise ValueError("Tail calculation must be finite.")
    return {"quantile": quantile, "expected_shortfall": expected_shortfall}


def fit_pooled_coefficient(r, x, logw, loc, scale, nu, markets, bound=1.0):
    """Fit one bounded coefficient by equal-market development log score.

    Each market has equal total weight; dates within a market have equal weight.
    ``x`` must already use the frozen causal feature/normalization/clipping rule.
    This function does not inspect timestamps: development-only selection is the
    caller's responsibility. A 129-point grid, including zero and both bounds,
    is supplemented by bounded refinement around each grid-local optimum.
    This deterministic search is not a proof of a global optimum for arbitrary
    mixtures. Zero remains an explicit candidate, so training fit cannot worsen.
    This provides no out-of-sample or no-harm guarantee.
    """
    if not np.isscalar(bound) or not np.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be a positive finite scalar.")
    logw, loc, scale, nu = _mixture_arrays(logw, loc, scale, nu)
    n = len(logw)
    r, x = _vector(r, n, "r"), _vector(x, n, "x")
    raw_markets = np.asarray(markets)
    if raw_markets.shape != (n,):
        raise ValueError("markets must have one label per observation.")
    labels = []
    for value in raw_markets:
        if value is None or (isinstance(value, (int, float, np.number)) and not np.isfinite(value)):
            raise ValueError("Market labels must not be missing.")
        label = str(value).strip()
        if not label:
            raise ValueError("Market labels must not be empty.")
        labels.append(label)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(unique) * counts[inverse])
    evaluations = 0

    def objective(coefficient):
        nonlocal evaluations
        evaluations += 1
        log_s = float(coefficient) * x
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                transformed_r = r * np.exp(-log_s)
                log_scores = _logpdf_unchecked(transformed_r, logw, loc, scale, nu) - log_s
                answer = -float(weights @ log_scores)
        except FloatingPointError as exc:
            raise ValueError("Coefficient search exceeds numerical range; check feature bounds.") from exc
        if not np.isfinite(answer):
            raise ValueError("Coefficient objective is not finite.")
        return answer

    baseline = objective(0.0)
    candidates = [(0.0, baseline)]
    refinement_results = []
    grid_points = 0
    if np.any(x != 0):
        grid = np.linspace(-float(bound), float(bound), 129)
        values = np.array([objective(coefficient) for coefficient in grid])
        grid_points = len(grid)
        candidates.extend(zip(grid.tolist(), values.tolist()))
        for index in range(1, len(grid) - 1):
            if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
                result = minimize_scalar(
                    objective, bounds=(grid[index - 1], grid[index + 1]),
                    method="bounded", options={"xatol": 1e-8, "maxiter": 200},
                )
                refinement_results.append({
                    "success": bool(result.success), "coefficient": float(result.x),
                    "objective": float(result.fun), "evaluations": int(result.nfev),
                })
                if result.success and np.isfinite(result.fun):
                    candidates.append((float(result.x), float(result.fun)))
    # Prefer no adjustment (and then the smaller absolute coefficient) on ties.
    coefficient, fitted = min(candidates, key=lambda pair: (pair[1], abs(pair[0])))
    return {
        "coefficient": float(coefficient), "bound": float(bound),
        "at_boundary": bool(abs(abs(coefficient) - bound) <= max(1e-8, bound * 1e-6)),
        "baseline_log_score": -baseline, "fitted_log_score": -fitted,
        "development_log_score_gain": baseline - fitted,
        "n_observations": n, "n_markets": len(unique),
        "market_counts": {str(market): int(count) for market, count in zip(unique, counts)},
        "objective_weighting": "equal market; equal observations within market",
        "search": {
            "method": "129-point grid plus local bounded refinement; zero retained",
            "grid_points": grid_points, "evaluations": evaluations,
            "refinements": refinement_results, "zero_feature": bool(np.all(x == 0)),
            "global_optimum_guaranteed": False,
        },
    }
