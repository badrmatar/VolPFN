"""Frozen compact-rule and causal GJR Student distributions for the session pilot.

All returns/locations/Student scales use percentage-return units. ``h`` is the
predictive second moment, not a Student scale or a centered variance. Mixture
parameters have shape (observations, components); these adapters use one
component. Forecasts before their permitted initialization/start dates are NaN.

No fitting of the cross-market correction occurs here. The compact rule reuses
the archived fold-average convention in attribution.ladder; its coefficients and
normalization are fixed. GJR development and evaluation refit schedules are
separate, so adding development forecasts cannot move the existing 2015 anchor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.evaluation_strengthening.classical import forecast_classical


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS = ROOT / "experiments/outputs/attribution_driver_swap_v1/folds.json"
INITIALIZATION_RETURNS = 100


def _validate_market(market):
    r = np.asarray(market["r"], dtype=float)
    z = np.asarray(market["z"], dtype=float)
    date = np.asarray(market["date"]).astype("U10")
    normalization = float(market["scale"])
    if r.ndim != 1 or len(r) < INITIALIZATION_RETURNS:
        raise ValueError("At least 100 one-dimensional initialization returns are required")
    if z.shape != r.shape or date.shape != r.shape:
        raise ValueError("date, r and z must be aligned one-dimensional arrays")
    if not np.isfinite(r).all() or not np.isfinite(z).all():
        raise ValueError("Returns must be finite")
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Market normalization scale must be finite and positive")
    if not np.allclose(z, r / normalization, rtol=1e-12, atol=1e-12):
        raise ValueError("Normalized returns must equal r / scale")
    if np.any(date[1:] <= date[:-1]):
        raise ValueError("Market dates must be strictly increasing")
    return r, date


def _one_student(mean, variance, nu):
    """Convert a centered-variance Student law to explicit mixture parameters."""
    mean, variance, nu = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(variance, dtype=float),
        np.asarray(nu, dtype=float))
    if mean.ndim != 1:
        raise ValueError("Student parameters must be one-dimensional forecast paths")
    available = np.isfinite(mean) & np.isfinite(variance) & np.isfinite(nu)
    partly_available = np.isfinite(mean) | np.isfinite(variance) | np.isfinite(nu)
    if np.any(partly_available & ~available):
        raise ValueError("Student parameters must be jointly available or jointly NaN")
    if np.any(available & ((variance <= 0) | (nu <= 2))):
        raise ValueError("A finite positive variance and degrees of freedom above two are required")
    student_scale = np.full_like(variance, np.nan)
    student_scale[available] = np.sqrt(variance[available] * (nu[available] - 2.0) / nu[available])
    logw = np.full_like(variance, np.nan)
    logw[available] = 0.0
    second_moment = variance + mean ** 2
    return {"logw": logw[:, None], "loc": mean[:, None].copy(),
            "scale": student_scale[:, None], "nu": nu[:, None].copy(),
            "mean": mean.copy(), "variance": variance.copy(),
            "second_moment": second_moment, "h": second_moment.copy()}


def _prefix_market(market, n):
    total = len(market["r"])
    return {key: np.asarray(value)[:n].copy()
            if isinstance(value, (list, tuple, np.ndarray))
            and np.asarray(value).ndim > 0 and len(value) == total else value
            for key, value in market.items()}


def gjr_distribution(market, development_start="2011-01-01", evaluation_start="2015-01-01",
                     refit_every=250, history_window=1000):
    """Return aligned GJR-GARCH-t mixture parameters and fit provenance.

    Development is forecast from its own first origin on a series truncated
    strictly before ``evaluation_start``. Evaluation is forecast independently
    from ``evaluation_start`` using all available historical returns, preserving
    the existing strengthening study's refit schedule. Each GJR fit uses only
    returns preceding its origin. Failures propagate; no failed market/date is
    silently removed. This performs classical likelihood fitting, not neural
    training. The caller must provide a normalization derived from past data.
    """
    r, date = _validate_market(market)
    if str(development_start) >= str(evaluation_start):
        raise ValueError("development_start must precede evaluation_start")
    n = len(r)
    paths = {key: np.full(n, np.nan) for key in ("mean", "variance", "nu", "logscore")}
    fit_origin = np.full(n, "", dtype="U10")
    fits = []
    segment_metadata = {}
    boundary = int(np.searchsorted(date, str(evaluation_start)))
    segments = (("development", development_start, _prefix_market(market, boundary), boundary),
                ("evaluation", evaluation_start, market, n))
    for segment, start, supplied_market, stop in segments:
        supplied_dates = date[:stop]
        if not len(supplied_dates) or not np.any(supplied_dates >= str(start)):
            continue
        forecast = forecast_classical(supplied_market, "gjr_garch_t", start_date=start,
                                      refit_every=refit_every, history_window=history_window)
        mask = (date[:stop] >= str(start))
        if segment == "development":
            mask &= date[:stop] < str(evaluation_start)
        index = np.flatnonzero(mask)
        for key in paths:
            paths[key][index] = forecast[key][index]
        fit_origin[index] = forecast["fit_origin"][index]
        fits.extend(dict(fit, segment=segment) for fit in forecast["fits"])
        segment_metadata[segment] = forecast["metadata"]
    result = _one_student(paths["mean"], paths["variance"], paths["nu"])
    result.update(logscore=paths["logscore"], fit_origin=fit_origin, fits=fits)
    result["metadata"] = {
        "model": "gjr_garch_t", "return_units": "percent", "moment_units": "percent_squared",
        "development_start": str(development_start), "evaluation_start": str(evaluation_start),
        "refit_every": int(refit_every), "history_window": int(history_window),
        "separate_refit_anchors": True, "segments": segment_metadata,
        "distribution": "constant-mean Student-t with fitted centered variance and degrees of freedom",
        "correction_position": "post-forecast density scaling; not a modified GJR recursion",
    }
    return result


def compact_distribution(market, folds_path=DEFAULT_FOLDS):
    """Run the frozen nine-coefficient likelihood rule, with a Student-t8 density.

    Reuses ``ladder.load_real_coefficients`` and ``drivers.prepare/run_batch``
    verbatim. In the archived study, each rule was fitted on pre-2011 histories
    of the other geographic groups and assessed on 2011--2014 held-out groups.
    This adapter averages all fold coefficients and normalizations: it is NOT
    the original leave-group-out estimate, since other folds include a target
    market's pre-2011 history. It performs no fitting on the provided forecast
    path. Initial state uses the first 100 returns, which are masked out.

    The rule was selected using development evidence. Later correction fitting
    on 2011--2014 is development fitting, not an independent or prequential
    performance claim. Return-driven state evolution is causal conditional on
    the fixed archived coefficients and historical normalization.
    """
    r, _ = _validate_market(market)
    from experiments.attribution import drivers
    from experiments.attribution.ladder import load_real_coefficients
    from experiments.prior_adaptation.model import log_law

    folds_path = Path(folds_path).resolve()
    raw_folds = folds_path.read_bytes()
    fold_details = json.loads(raw_folds)
    coefficients, normalization = load_real_coefficients(folds_path)
    beta = np.asarray(coefficients["rule9"], dtype=float)
    if beta.shape != (9,) or not np.isfinite(beta).all():
        raise ValueError("The compact rule must have nine finite likelihood coefficients")
    if not all(np.isfinite(value) for value in normalization.values()):
        raise ValueError("Frozen rule normalization must be finite")
    if any(value <= 0 for key, value in normalization.items() if key.endswith("_sd")):
        raise ValueError("Frozen rule normalization standard deviations must be positive")
    keep = np.arange(len(r)) >= INITIALIZATION_RETURNS
    prepared = drivers.prepare(market, normalization, keep)
    h = drivers.run_batch(drivers.stack([prepared]), beta, "state_dep_gain", normalization)[0]
    h = np.asarray(h, dtype=float).copy()
    if not np.isfinite(h).all() or np.any(h <= 0):
        raise RuntimeError("The compact rule produced invalid variances")
    h[~keep] = np.nan
    mean = np.where(keep, 0.0, np.nan)
    nu = np.where(keep, 8.0, np.nan)
    result = _one_student(mean, h, nu)
    result["logscore"] = np.full(len(r), np.nan)
    result["logscore"][keep] = log_law(r[keep], h[keep], 8.0)
    result.update(fits=[], fit_origin=np.full(len(r), "", dtype="U10"))
    result["metadata"] = {
        "model": "compact_rule9_frozen_fold_average", "return_units": "percent",
        "moment_units": "percent_squared", "folds_path": str(folds_path),
        "folds_sha256": hashlib.sha256(raw_folds).hexdigest(),
        "fold_count": len(fold_details), "coefficient_variant": "likelihood",
        "coefficients": beta.tolist(), "normalization": normalization,
        "training_history": "archived driver-swap coefficients fitted on pre-2011 histories",
        "averaging": "arithmetic coefficient and normalization averages, as attribution.ladder",
        "leave_group_out": False, "initialization_returns": INITIALIZATION_RETURNS,
        "initialization": "log(max(mean square of first 100 percentage returns, 1e-8))",
        "development_role": "2011-2014 was used in rule selection; correction fitting is development fitting",
        "distribution": "zero-mean Student-t8, variance equals rule forecast",
        "correction_position": "post-forecast density scaling; base recursion unchanged",
    }
    return result
