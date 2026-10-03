"""Frozen-forecast transfer scoring and synchronized calendar-block inference.

This module never fits or selects a forecaster. Positive gains favour the first
named forecaster. Forecast inputs are conditional second moments; common-t8
scoring uses these as the variance of a zero-mean Student t distribution. It does
not remove the VolPFN mixture head that produced the second moment.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import gammaln


MODELS = ("R_eq", "R_class", "R_market", "VolPFN", "R_archive")
CONTRASTS = (("R_class", "R_eq"), ("VolPFN", "R_eq"),
             ("VolPFN", "R_class"), ("VolPFN", "R_market"))
SCOPE = ("Conditional on retained forecasts, fitting procedures and the fixed "
         "population; excludes uncertainty from training and prior selection.")


def common_t8_log_score(r, second_moment):
    """Log density of t8(0, scale=sqrt(3*h/4)); its variance equals h."""
    r, h = np.broadcast_arrays(np.asarray(r, float), np.asarray(second_moment, float))
    if not np.isfinite(r).all() or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("Finite returns and strictly positive finite second moments required")
    nu = 8.0
    scale2 = h * (nu - 2.0) / nu
    return (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * scale2)
            - (nu + 1.0) / 2.0 * np.log1p(r * r / (nu * scale2)))


def qlike(r, second_moment):
    """Squared-return QLIKE loss, omitting the target-only constant."""
    r, h = np.broadcast_arrays(np.asarray(r, float), np.asarray(second_moment, float))
    if not np.isfinite(r).all() or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("Finite returns and strictly positive finite second moments required")
    return r * r / h + np.log(h)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected a vector of probabilities")
    order = np.argsort(p, kind="stable")
    result = np.empty(len(p))
    result[order] = np.minimum(1.0, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return result


@lru_cache(maxsize=2)
def _calendar_counts(n, reps, block, seed):
    if any(int(v) != v or v < 1 for v in (n, reps, block)):
        raise ValueError("Positive integer dates, resamples and block length required")
    rng = np.random.default_rng(seed)
    counts = np.empty((reps, n), dtype=float)
    offsets = np.arange(block)
    for j in range(reps):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + offsets) % n).ravel()[:n]
        counts[j] = np.bincount(indices, minlength=n)
    counts.setflags(write=False)
    return counts


def _gain_frame(dates, markets, gains):
    f = pd.DataFrame({"date": dates, "market": markets, "gain": gains})
    if f.empty or f[["date", "market"]].isna().any().any():
        raise ValueError("Nonempty matched observations required")
    f["date"] = pd.to_datetime(f.date, errors="raise")
    if not np.isfinite(f.gain.to_numpy(float)).all():
        raise ValueError("Finite gains required; no outcome-based row dropping")
    if f.duplicated(["date", "market"]).any():
        raise ValueError("Duplicate market/date observations")
    return f


def _resample_groups(groups, reps, block, seed):
    """Resample the union of dates once for all populations and contrasts.

    Missing market dates remain absent. Within each draw dates are averaged
    within market and markets equally, rather than giving long series more weight.
    """
    dates = pd.DatetimeIndex(sorted(set().union(*(set(f.date) for f in groups.values()))))
    counts = _calendar_counts(len(dates), reps, block, seed)
    output = {}
    for name, f in groups.items():
        pivot = f.pivot(index="date", columns="market", values="gain").reindex(dates)
        values = pivot.to_numpy(float)
        valid = np.isfinite(values).astype(float)
        denom = counts @ valid
        if np.any(denom == 0):
            raise ValueError("A bootstrap draw omits a market; insufficient calendar coverage")
        means = np.nanmean(values, axis=0)
        draws = np.mean((counts @ np.nan_to_num(values, nan=0.0)) / denom, axis=1)
        output[name] = (float(means.mean()), draws,
                        dict(zip(pivot.columns, map(float, means))), len(dates))
    return output


def _basic_interval(estimate, draws, coverage):
    tail = (1.0 - coverage) / 2.0
    errors = draws - estimate
    q = np.quantile(errors, [tail, 1.0 - tail])
    return [float(estimate - q[1]), float(estimate - q[0])]


def _summary(estimate, draws, market_means, n_dates, rows, reps, block, seed):
    errors = draws - estimate
    p = float((1 + np.count_nonzero(np.abs(errors) >= abs(estimate))) / (reps + 1))
    return {
        "status": "available", "mean_gain": estimate,
        "interval95": _basic_interval(estimate, draws, 0.95),
        "interval90": _basic_interval(estimate, draws, 0.90),
        "interval_method": "basic interval from centered circular date-block bootstrap",
        "p_null": p, "markets_improved": sum(v > 0 for v in market_means.values()),
        "markets": len(market_means), "rows": int(rows), "calendar_dates": n_dates,
        "per_market_mean_gain": market_means, "block_observed_dates": block,
        "resamples": reps, "seed": seed,
        "weighting": "equal markets; time mean within each market",
        "p_method": "two-sided centered circular date-block bootstrap; plus-one correction",
        "scope": SCOPE,
    }


def paired_summary(dates, markets, gains, *, reps=2000, block=60, seed=927601):
    f = _gain_frame(dates, markets, gains)
    estimate, draws, means, n_dates = _resample_groups(
        {"gain": f}, reps, block, seed)["gain"]
    return _summary(estimate, draws, means, n_dates, len(f), reps, block, seed)


def _equivalence(estimate, draws, margin, alpha=0.05):
    if not np.isfinite(margin) or margin <= 0 or not 0 < alpha < 0.5:
        raise ValueError("A positive finite margin and alpha between zero and 0.5 required")
    errors = draws - estimate
    # Boundary nulls: mean <= -margin versus mean > -margin, and
    # mean >= +margin versus mean < +margin. Both must be rejected.
    lower_p = float((1 + np.count_nonzero(errors >= estimate + margin)) / (len(draws) + 1))
    upper_p = float((1 + np.count_nonzero(errors <= estimate - margin)) / (len(draws) + 1))
    interval = _basic_interval(estimate, draws, 1.0 - 2.0 * alpha)
    inside = interval[0] > -margin and interval[1] < margin
    passed = inside and max(lower_p, upper_p) <= alpha
    return {
        "status": "available", "mean_gain": estimate, "margin_nats": margin,
        "alpha": alpha, "interval90": interval,
        "interval95": _basic_interval(estimate, draws, 0.95),
        "p_lower_boundary": lower_p, "p_upper_boundary": upper_p,
        "p_tost": max(lower_p, upper_p), "interval_inside_margin": inside,
        "equivalence_supported": passed,
        "conclusion": "equivalence supported within the declared margin" if passed else "inconclusive for equivalence",
        "method": ("TOST using centered circular date-block bootstrap at boundary nulls; "
                   "plus-one correction; also requires basic 90% interval strictly inside the margin"),
        "monte_carlo_note": ("The interval and boundary tests use the same centered draws. "
                             "Both criteria are required to handle finite-resample quantile/tie differences conservatively."),
    }


def equivalence_test(dates, markets, gains, *, margin=0.003, reps=2000,
                     block=60, seed=927601):
    f = _gain_frame(dates, markets, gains)
    estimate, draws, _, _ = _resample_groups({"gain": f}, reps, block, seed)["gain"]
    return _equivalence(estimate, draws, margin)


def _unavailable(reason):
    return {"status": "unavailable", "reason": reason, "mean_gain": None,
            "interval95": None, "interval90": None, "p_null": 1.0}


def _validate_frame(frame):
    required = {"market", "date", "r"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing required columns: " + str(sorted(required - set(frame.columns))))
    f = frame.copy()
    if f[["market", "date"]].isna().any().any():
        raise ValueError("Market and date must not be missing")
    f["date"] = pd.to_datetime(f.date, errors="raise")
    if f.duplicated(["market", "date"]).any():
        raise ValueError("Duplicate market/date observations")
    return f.sort_values(["market", "date"]).reset_index(drop=True)


def _model_completeness(f, expected_names):
    failures, statuses = [], {}
    present = set(f.market)
    for model in MODELS:
        reasons = []
        for name in expected_names:
            part = f.loc[f.market == name]
            code = None
            if name not in present:
                code = "missing_series"
            elif not np.isfinite(part.r.to_numpy(float)).all():
                code = "invalid_target"
            elif model not in f:
                code = "missing_forecast_column"
            elif np.any(~np.isfinite(part[model].to_numpy(float)) | (part[model].to_numpy(float) <= 0)):
                code = "invalid_or_missing_forecast"
            if code:
                reasons.append(name + ":" + code)
                failures.append({"market": name, "model": model, "failure_code": code,
                                 "rows": len(part)})
        statuses[model] = {"available": not reasons, "reasons": reasons}
    return statuses, failures


def _complete_series_description(f, candidate, control, metric):
    """Optional complete-series summary; never used for a test or gate."""
    gains = {}
    if candidate not in f or control not in f:
        return {"mean_gain": None, "markets": 0, "per_market_mean_gain": {},
                "scope": "complete series only; descriptive; no inference"}
    for name, part in f.groupby("market", sort=True):
        a, b, r = (part[c].to_numpy(float) for c in (candidate, control, "r"))
        if not (np.isfinite(r).all() and np.isfinite(a).all() and np.isfinite(b).all()
                and (a > 0).all() and (b > 0).all()):
            continue
        d = (common_t8_log_score(r, a) - common_t8_log_score(r, b)) if metric == "log_score" else (qlike(r, b) - qlike(r, a))
        gains[name] = float(np.mean(d))
    return {"mean_gain": float(np.mean(list(gains.values()))) if gains else None,
            "markets": len(gains), "per_market_mean_gain": gains,
            "scope": "complete series only; descriptive; no inference"}


def recovery_share(frame, ref_frame, *, reps=2000, block=60, seed=927601):
    """Secondary excess transfer recovery, with matched equity residual uncertainty.

    The caller must supply full, design-matched equity reference forecasts. Both
    populations are resampled on the same union of calendar dates. A weak class
    gap prevents reporting a bounded ratio, regardless of its point estimate.
    """
    f, ref = _validate_frame(frame), _validate_frame(ref_frame)
    for data, needed in ((f, ("VolPFN", "R_eq", "R_class")), (ref, ("VolPFN", "R_eq"))):
        if data.empty or any(c not in data for c in needed):
            return {"status": "unavailable", "reason": "missing matched reference or transfer forecasts"}
        if not np.isfinite(data.r.to_numpy(float)).all() or any(
                not np.isfinite(data[c].to_numpy(float)).all() or (data[c] <= 0).any() for c in needed):
            return {"status": "unavailable", "reason": "invalid matched reference or transfer forecasts"}
    network_gain = common_t8_log_score(f.r, f.VolPFN) - common_t8_log_score(f.r, f.R_eq)
    class_gain = common_t8_log_score(f.r, f.R_class) - common_t8_log_score(f.r, f.R_eq)
    residual = common_t8_log_score(ref.r, ref.VolPFN) - common_t8_log_score(ref.r, ref.R_eq)
    groups = {"network": _gain_frame(f.date, f.market, network_gain),
              "class": _gain_frame(f.date, f.market, class_gain),
              "equity": _gain_frame(ref.date, ref.market, residual)}
    try:
        out = _resample_groups(groups, reps, block, seed)
    except ValueError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    numerator = out["network"][0] - out["equity"][0]
    num_draws = out["network"][1] - out["equity"][1]
    denominator, den_draws = out["class"][:2]
    denominator_ci = _basic_interval(denominator, den_draws, 0.95)
    result = {
        "status": "available", "role": "secondary descriptive ratio, not a mechanism test",
        "numerator": numerator, "numerator_interval95": _basic_interval(numerator, num_draws, 0.95),
        "class_gap": denominator, "class_gap_interval95": denominator_ci,
        "equity_residual": out["equity"][0],
        "equity_residual_interval95": _basic_interval(out["equity"][0], out["equity"][1], 0.95),
        "share": None, "share_interval95": None, "scope": SCOPE,
        "reference_requirement": "Equity residual must use the same fitting periods, scoring law and evaluation convention.",
    }
    if denominator_ci[0] <= 0 <= denominator_ci[1]:
        result.update(status="unavailable", reason="class-gap interval contains zero; ratio is weakly identified/unbounded")
    elif denominator_ci[1] < 0:
        result.update(status="unavailable", reason="class refitting has a supported negative gain; no positive gain to recover")
    elif (den_draws <= 0).any():
        result.update(status="unavailable", reason="bootstrap class gaps cross zero; bounded ratio interval not reported")
    else:
        result["share"] = float(numerator / denominator)
        result["share_interval95"] = _basic_interval(result["share"], num_draws / den_draws, 0.95)
        result["ratio_interval_note"] = "Basic joint-bootstrap ratio interval; reported only with a separated positive denominator and no sign-crossing draws."
    return result


def analyze_population(frame, population, expected_names, ref_frame=None, *,
                       reps=2000, block=60, seed=927601, margin=0.003):
    """Analyze one fixed population, never pooling stock and FX testing families.

    ``frame`` contains evaluation rows only. A missing expected series or any
    invalid forecast reserves the affected full-panel contrast at p=1; no row or
    series is silently dropped. Recorded fit failure codes can be attached by the
    runner in addition to this forecast-completeness table.
    """
    population = population.lower()
    if population not in ("stocks", "fx"):
        raise ValueError("population must be 'stocks' or 'fx'")
    expected_names = list(expected_names)
    if not expected_names or len(set(expected_names)) != len(expected_names):
        raise ValueError("A nonempty unique prespecified population list is required")
    f = _validate_frame(frame)
    if set(f.market) - set(expected_names):
        raise ValueError("Observed markets outside the frozen population")
    statuses, failures = _model_completeness(f, expected_names)
    scores, losses, by_market = {}, {}, {}
    for model in MODELS:
        if statuses[model]["available"]:
            scores[model] = common_t8_log_score(f.r, f[model])
            losses[model] = qlike(f.r, f[model])
            by_market[model] = {}
            for name, ids in f.groupby("market").groups.items():
                by_market[model][name] = {
                    "evaluation_log_score": float(np.mean(scores[model][list(ids)])),
                    "evaluation_qlike": float(np.mean(losses[model][list(ids)])),
                    "evaluation_rows": len(ids),
                }
    log_rows, qlike_rows = [], []
    for candidate, control in CONTRASTS:
        available = candidate in scores and control in scores
        base = {"candidate": candidate, "control": control,
                "contrast": candidate + " minus " + control}
        if available:
            try:
                row = paired_summary(f.date, f.market, scores[candidate] - scores[control],
                                     reps=reps, block=block, seed=seed)
            except ValueError as exc:
                row = _unavailable(str(exc))
        else:
            reason = "; ".join(statuses[candidate]["reasons"] + statuses[control]["reasons"])
            row = _unavailable(reason)
            row["complete_series_descriptive"] = _complete_series_description(f, candidate, control, "log_score")
        row.update(base)
        log_rows.append(row)
        # QLIKE receives no p-values or multiplicity claim.
        qrow = _complete_series_description(f, candidate, control, "qlike")
        qrow.update(base)
        qrow["status"] = "available" if available else "full_panel_unavailable"
        qrow["scope"] = "descriptive QLIKE; full panel" if available else "complete series only; descriptive; no inference"
        qlike_rows.append(qrow)
    adjusted = holm([row["p_null"] for row in log_rows])
    for row, p in zip(log_rows, adjusted):
        row["holm_p"] = float(p)
        row["supported_positive"] = bool(row["status"] == "available" and row["mean_gain"] > 0 and p <= 0.05)
        row["supported_negative"] = bool(row["status"] == "available" and row["mean_gain"] < 0 and p <= 0.05)
    eq = {"status": "unavailable", "reason": "incomplete full-panel VolPFN or R_class forecasts",
          "margin_nats": margin, "p_tost": 1.0, "equivalence_supported": False}
    if "VolPFN" in scores and "R_class" in scores:
        try:
            eq = equivalence_test(f.date, f.market, scores["VolPFN"] - scores["R_class"],
                                  margin=margin, reps=reps, block=block, seed=seed)
        except ValueError as exc:
            eq["reason"] = str(exc)
    eq["role"] = "sole primary stock equivalence endpoint" if population == "stocks" else "descriptive FX equivalence diagnostic"
    eq["confirmatory"] = population == "stocks"
    if population == "fx":
        eq["diagnostic_p_tost"] = eq.get("p_tost", 1.0)
        eq["p_tost"] = 1.0
    share = {"status": "unavailable", "reason": "matched equity reference not supplied"}
    if ref_frame is not None and all(name in scores for name in ("VolPFN", "R_eq", "R_class")):
        share = recovery_share(f, ref_frame, reps=reps, block=block, seed=seed)
    return {
        "population": population, "expected_markets": expected_names,
        "observed_markets": sorted(set(f.market)), "missing_markets": sorted(set(expected_names) - set(f.market)),
        "evaluation_rows": len(f), "model_completeness": statuses, "forecast_failures": failures,
        "scoring_convention": "zero-mean Student t8 with variance equal to each predicted second moment",
        "equivalence_primary" if population == "stocks" else "equivalence_descriptive": eq,
        "log_score_contrasts": log_rows,
        "multiplicity": {"family": population, "slots": 4, "method": "Holm within population",
                         "unavailable_slots": "p=1; never delete a planned comparison",
                         "role": "supporting superiority family, separate from sole primary stock equivalence endpoint"},
        "qlike_descriptive": qlike_rows, "per_market_evaluation_metrics": by_market,
        "recovery_share_secondary": share, "scope": SCOPE,
    }
