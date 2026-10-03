"""Conditional inference for the frozen matched-prior comparison.

Inputs are already paired daily common-t8 log-score gains (broad minus the
named alternative). This module never reads forecasts, selects rows, trains,
or averages predictive densities. It holds every declared trained model fixed.
"""
import hashlib
import math

import numpy as np
import pandas as pd


PRIMARY_SEEDS = (8101, 8102, 8103, 8104, 8105)
SECONDARY_SEEDS = (8101, 8102, 8103)
PRIMARY = "broad_minus_matern"
SECONDARY = "broad_minus_garch"
DEFAULT_REPS = 5000
DEFAULT_BLOCK = 60
DEFAULT_BOOTSTRAP_SEED = 9182601
SCOPE = (
    "Calendar uncertainty conditional on these fixed trained models and the "
    "frozen population/scoring mask. No training seeds are resampled; the seed "
    "range is descriptive. Does not cover new training seeds, study selection, "
    "prior selection, or population selection. Not a density ensemble."
)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected a nonempty vector of finite probabilities")
    order = np.argsort(p, kind="stable")
    result = np.empty(len(p))
    result[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _daily_labels(frame, label_columns):
    if frame[label_columns].isna().any().any():
        raise ValueError("Observation labels cannot be missing")
    result = frame.copy()
    result["market"] = result.market.astype(str)
    result["date"] = pd.to_datetime(result.date, errors="raise")
    if result.date.dt.tz is not None or not result.date.eq(result.date.dt.normalize()).all():
        raise ValueError("Use timezone-naive daily date labels")
    return result


def _frozen_mask(expected_rows):
    if expected_rows is None or not {"market", "date"}.issubset(expected_rows.columns):
        raise ValueError("An external frozen market/date scoring mask is required")
    mask = _daily_labels(expected_rows[["market", "date"]], ["market", "date"])
    if mask.empty or mask.duplicated(["market", "date"]).any():
        raise ValueError("Frozen scoring mask must be nonempty with unique market/date rows")
    return mask.sort_values(["market", "date"]).reset_index(drop=True)


def _validate(frame, expected_rows, expected_seeds):
    required = {"seed", "market", "date", "gain"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing columns: " + str(sorted(required - set(frame.columns))))
    seeds = tuple(str(x) for x in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected seeds must be nonempty and distinct")
    mask = _frozen_mask(expected_rows)
    f = _daily_labels(frame[list(sorted(required))], ["seed", "market", "date"])
    f["seed"] = f.seed.astype(str)
    if set(f.seed) != set(seeds):
        raise ValueError("Missing or unregistered seeds; no surviving-seed average permitted")
    if f.duplicated(["seed", "market", "date"]).any():
        raise ValueError("Duplicate seed/market/date observations")
    f["gain"] = pd.to_numeric(f.gain, errors="raise").astype(float)
    if not np.isfinite(f.gain.to_numpy()).all():
        raise ValueError("Nonfinite gain; no outcome-based row dropping permitted")
    mask_keys = set(mask.itertuples(index=False, name=None))
    for seed in seeds:
        keys = set(f.loc[f.seed == seed, ["market", "date"]].itertuples(index=False, name=None))
        if keys != mask_keys:
            raise ValueError("Scoring mask mismatch for seed %s: missing=%d extra=%d" %
                             (seed, len(mask_keys - keys), len(keys - mask_keys)))
    return f.sort_values(["seed", "market", "date"]).reset_index(drop=True), mask, seeds


def _calendar_counts(n_dates, reps, block, seed):
    """One calendar resample, reused for every market, seed and contrast."""
    n_dates = _positive_integer(n_dates, "Calendar dates")
    reps = _positive_integer(reps, "Resamples")
    block = _positive_integer(block, "Block length")
    rng = np.random.default_rng(seed)
    counts = np.empty((reps, n_dates), dtype=float)
    offsets = np.arange(block)
    for j in range(reps):
        starts = rng.integers(0, n_dates, int(np.ceil(n_dates / block)))
        selected = ((starts[:, None] + offsets) % n_dates).ravel()[:n_dates]
        counts[j] = np.bincount(selected, minlength=n_dates)
    return counts


def _basic_interval(estimate, draws, coverage):
    tail = (1. - coverage) / 2.
    low, high = np.quantile(draws - estimate, [tail, 1. - tail])
    return [float(estimate - high), float(estimate - low)]


def _equivalence(estimate, draws, margin, alpha):
    """Approximate bootstrap TOST using boundary nulls and the same errors as CI.

    H0_lower: mean <= -margin; H0_upper: mean >= +margin. Plus-one
    p-values and quantile interpolation need not agree exactly with finite B,
    so both boundary rejection and strict interval inclusion are required.
    """
    errors = draws - estimate
    # Conservative floating-point tie handling prevents a constant exactly on
    # a null boundary being accepted because aggregation moved it by one ulp.
    tolerance = 64 * np.finfo(float).eps * max(1., abs(estimate), margin)
    p_lower = float((1 + np.count_nonzero(errors >= estimate + margin - tolerance)) / (len(draws) + 1))
    p_upper = float((1 + np.count_nonzero(errors <= estimate - margin + tolerance)) / (len(draws) + 1))
    interval = _basic_interval(estimate, draws, 1. - 2. * alpha)
    inside = interval[0] > -margin + tolerance and interval[1] < margin - tolerance
    supported = inside and max(p_lower, p_upper) <= alpha
    return {
        "margin_nats": float(margin), "alpha": float(alpha),
        "interval90": interval, "p_lower_boundary": p_lower,
        "p_upper_boundary": p_upper, "p_tost": max(p_lower, p_upper),
        "interval_inside_margin": bool(inside),
        "equivalence_supported": bool(supported),
        "method": "Centered circular-block boundary tests and basic 90% interval; plus-one p-values",
        "finite_resample_rule": "Require both boundary p-values <= alpha and the interval strictly within the band",
        "floating_tie_tolerance": float(tolerance),
        "interpretation": "Equivalence within the declared band" if supported else "Inconclusive for equivalence; not evidence of parity",
    }


def _summarize(f, mask, seeds, calendar, counts, margin, alpha, materiality):
    # Exact shared row masks allow seed averaging before calendar resampling.
    # This is a mean of paired scores, not a score of a mixture of predictions.
    average = f.groupby(["date", "market"], sort=True).gain.mean().unstack("market").reindex(calendar)
    values = average.to_numpy(float)
    present = np.isfinite(values).astype(float)
    denominator = counts @ present
    if np.any(denominator == 0):
        raise ValueError("A calendar draw omits a market; do not drop that market or redraw silently")
    market_means = np.nanmean(values, axis=0)
    estimate = float(market_means.mean())
    draws = np.mean((counts @ np.nan_to_num(values, nan=0.)) / denominator, axis=1)
    errors = draws - estimate
    tolerance = 64 * np.finfo(float).eps * max(1., abs(estimate))
    p = float((1 + np.count_nonzero(errors >= estimate - tolerance)) / (len(draws) + 1))
    per_seed_market = f.groupby(["seed", "market"], sort=True).gain.mean()
    per_seed = {s: float(per_seed_market.loc[s].mean()) for s in seeds}
    improved = int(np.count_nonzero(market_means > 0))
    n_markets = len(market_means)
    required_markets = math.ceil(20 * n_markets / 28)
    digest = hashlib.sha256(mask.to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest()
    interval95 = _basic_interval(estimate, draws, .95)
    return {
        "status": "available", "mean_gain": estimate,
        "interval95": interval95,
        "positive_interval95": bool(interval95[0] > 0),
        "interval90": _basic_interval(estimate, draws, .90),
        "p_superiority": p, "p_null": "mean_gain <= 0",
        "p_method": "One-sided centered circular calendar-block bootstrap with plus-one correction",
        "interval_method": "Basic bootstrap interval; conditional on all fixed trained seeds",
        "per_seed_mean_gain": per_seed,
        "seed_range": [float(min(per_seed.values())), float(max(per_seed.values()))],
        "all_seed_means_positive": bool(all(v > 0 for v in per_seed.values())),
        "per_market_mean_gain": dict(zip(average.columns, map(float, market_means))),
        "markets": n_markets, "markets_improved": improved,
        "required_positive_markets": required_markets,
        "market_coverage_bar_met": bool(improved >= required_markets),
        "materiality_bar_nats": float(materiality),
        "point_materiality_bar_met": bool(estimate >= materiality),
        "rows_per_seed": len(mask), "calendar_dates": len(calendar),
        "scoring_mask_sha256": digest, "expected_seeds": list(seeds),
        "scope": SCOPE,
    }, draws


def analyze_contrast(frame, *, expected_rows, expected_seeds=PRIMARY_SEEDS,
                     reps=DEFAULT_REPS, block=DEFAULT_BLOCK, seed=DEFAULT_BOOTSTRAP_SEED,
                     margin=.003, alpha=.05, materiality=.005):
    """One complete contrast; malformed/incomplete inputs fail, never lose rows.

    ``expected_rows`` must be frozen independently of available model outputs.
    All expected seeds must have exactly that mask. ``gain`` is broad log score
    minus comparator log score. This standalone result has no Holm adjustment;
    use ``analyze_study`` for the registered two-slot superiority family.
    """
    if not np.isfinite(margin) or margin <= 0 or not np.isfinite(materiality) or materiality <= 0:
        raise ValueError("Finite positive equivalence and materiality margins required")
    if alpha != .05:
        raise ValueError("The frozen alpha is .05")
    f, mask, seeds = _validate(frame, expected_rows, expected_seeds)
    calendar = pd.DatetimeIndex(sorted(mask.date.unique()))
    counts = _calendar_counts(len(calendar), reps, block, seed)
    out, draws = _summarize(f, mask, seeds, calendar, counts, margin, alpha, materiality)
    out.update(resamples=int(reps), block_observed_union_dates=int(block), bootstrap_seed=int(seed),
               equivalence=_equivalence(out["mean_gain"], draws, margin, alpha))
    return out


def analyze_study(primary, *, expected_rows, secondary=None, reps=DEFAULT_REPS,
                  block=DEFAULT_BLOCK, seed=DEFAULT_BOOTSTRAP_SEED):
    """Analyze one predeclared block under fixed seeds and two test families.

    Primary is broad-minus-Matern (five pairs); optional secondary is
    broad-minus-GARCH (three pairs). A wholly unrun secondary reserves p=1 in
    Holm. An incomplete supplied contrast fails explicitly rather than quietly
    becoming a survivor analysis. The caller records failures and treats the
    failed comparison as unavailable; this function must not be rerun on fewer
    seeds/markets. No joint FWER claim spans superiority and equivalence.
    """
    f, mask, seeds = _validate(primary, expected_rows, PRIMARY_SEEDS)
    if not 20 <= mask.market.nunique() <= 28:
        raise ValueError("The primary study requires a fixed eligible population of 20 to 28 markets")
    parts = {PRIMARY: (f, mask, seeds)}
    if secondary is not None:
        parts[SECONDARY] = _validate(secondary, expected_rows, SECONDARY_SEEDS)
    calendar = pd.DatetimeIndex(sorted(mask.date.unique()))
    counts = _calendar_counts(len(calendar), reps, block, seed)
    results, equivalence = {}, None
    for name, (f, mask, seeds) in parts.items():
        result, draws = _summarize(f, mask, seeds, calendar, counts, .003, .05, .005)
        results[name] = result
        if name == PRIMARY:
            equivalence = _equivalence(result["mean_gain"], draws, .003, .05)
    if SECONDARY not in results:
        results[SECONDARY] = {
            "status": "unrun_reserved_slot", "mean_gain": None,
            "p_superiority": 1., "expected_seeds": list(map(str, SECONDARY_SEEDS)),
        }
    adjusted = holm([results[c]["p_superiority"] for c in (PRIMARY, SECONDARY)])
    for name, p_adjusted in zip((PRIMARY, SECONDARY), adjusted):
        result = results[name]
        supported = result["status"] == "available" and p_adjusted <= .05
        result["p_superiority_holm"] = float(p_adjusted)
        result["superiority_supported"] = bool(supported)
        result["decision_bar_met"] = bool(supported and result.get("positive_interval95", False)
                                           and result.get("point_materiality_bar_met", False)
                                           and result.get("market_coverage_bar_met", False))
        if supported and not result["point_materiality_bar_met"]:
            result["superiority_interpretation"] = "Statistically supported but below the declared materiality bar"
        elif supported:
            result["superiority_interpretation"] = "Statistically supported; materiality and market coverage are recorded separately"
        else:
            result["superiority_interpretation"] = "Superiority not established"
    return {
        "contrasts": results,
        "superiority_family": {"slots": [PRIMARY, SECONDARY], "method": "Holm", "alpha": .05,
                               "direction": "one-sided broad advantage"},
        "equivalence": {"contrast": PRIMARY, "family": "single primary TOST, separate from superiority",
                        **equivalence},
        "multiplicity_scope": "No joint family-wise error guarantee across superiority and equivalence families",
        "estimand": "Equal-seed mean of paired score gains; time mean within market, equal markets",
        "resamples": int(reps), "block_observed_union_dates": int(block), "bootstrap_seed": int(seed),
        "scope": SCOPE,
    }
