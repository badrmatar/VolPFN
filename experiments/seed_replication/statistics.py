"""Paired seed-mean inference; does not fit, select, or ensemble forecasts.

The headline estimand is the arithmetic mean of equal-market score gains over
the predeclared training seeds. Crossed resampling draws whole training seeds
and one shared calendar-block path, preserving pairing across markets and
contrasts. Five seeds give only a coarse view of training variability.
"""
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (8101, 8102, 8103, 8104, 8105)
SCOPE = (
    "Crossed resampling reflects empirical training-seed and calendar variation, "
    "conditional on the fixed populations, fitting rules, and training procedure. "
    "Only five planned seeds are available; inference about unseen training "
    "seeds is approximate. It does not include prior or study-selection uncertainty."
)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite probabilities")
    order = np.argsort(p, kind="stable")
    out = np.empty(len(p))
    out[order] = np.minimum(1., np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return out


def _keys(frame):
    f = frame[["market", "date"]].copy()
    if f.isna().any().any():
        raise ValueError("Market/date labels cannot be missing")
    f["market"] = f.market.astype(str)
    f["date"] = pd.to_datetime(f.date, errors="raise")
    if f.date.dt.tz is not None or not f.date.eq(f.date.dt.normalize()).all():
        raise ValueError("Use timezone-naive daily date labels")
    if f.duplicated().any():
        raise ValueError("Duplicate market/date labels")
    return f.sort_values(["market", "date"]).reset_index(drop=True)


def _resampling_counts(n_dates, n_seeds, reps, block, seed):
    """Common circular date draws and common whole-seed draws for every contrast."""
    if any(int(v) != v or v < 1 for v in (n_dates, n_seeds, reps, block)):
        raise ValueError("Dates, seeds, resamples and block length must be positive integers")
    date_rng, seed_rng = [np.random.default_rng(x) for x in
                          np.random.SeedSequence(seed).spawn(2)]
    dates = np.empty((reps, n_dates))
    seeds = np.empty((reps, n_seeds))
    offsets = np.arange(block)
    for j in range(reps):
        starts = date_rng.integers(0, n_dates, int(np.ceil(n_dates / block)))
        indices = ((starts[:, None] + offsets) % n_dates).ravel()[:n_dates]
        dates[j] = np.bincount(indices, minlength=n_dates)
        seeds[j] = np.bincount(seed_rng.integers(0, n_seeds, n_seeds), minlength=n_seeds)
    return dates, seeds


def _basic_interval(estimate, draws, coverage):
    tail = (1. - coverage) / 2.
    lo, hi = np.quantile(draws - estimate, [tail, 1. - tail])
    return [float(estimate - hi), float(estimate - lo)]


def _summary(estimate, draws, method):
    return {
        "mean_gain": float(estimate),
        "interval95": _basic_interval(estimate, draws, .95),
        "interval90": _basic_interval(estimate, draws, .90),
        "p_null": float((1 + np.count_nonzero(np.abs(draws - estimate) >= abs(estimate))) /
                        (len(draws) + 1)),
        "interval_method": "basic bootstrap interval",
        "p_method": "two-sided centered bootstrap with plus-one correction",
        "resampling_method": method,
    }


def _equivalence(estimate, draws, margin, alpha):
    errors = draws - estimate
    p_lower = float((1 + np.count_nonzero(errors >= estimate + margin)) / (len(draws) + 1))
    p_upper = float((1 + np.count_nonzero(errors <= estimate - margin)) / (len(draws) + 1))
    interval = _basic_interval(estimate, draws, 1 - 2 * alpha)
    inside = interval[0] > -margin and interval[1] < margin
    return {
        "status": "available", "mean_gain": float(estimate), "margin_nats": margin,
        "alpha": alpha, "interval90": interval,
        "p_lower_boundary": p_lower, "p_upper_boundary": p_upper,
        "p_tost": max(p_lower, p_upper), "interval_inside_margin": bool(inside),
        "equivalence_supported": bool(inside and max(p_lower, p_upper) <= alpha),
        "interpretation": "An inconclusive TOST does not establish parity or inferiority.",
    }


def _unavailable(reasons, seed_status):
    empty = {"mean_gain": None, "interval95": None, "interval90": None, "p_null": 1.}
    return {
        "status": "unavailable", "reasons": reasons,
        "seed_status": seed_status, "per_seed_mean_gain": {s: None for s in seed_status},
        "conditional": dict(empty), "crossed": dict(empty),
        "all_seeds_positive": False, "claim_survives": False,
        "holm_by_family": {},
    }


def analyze_seed_contrasts(frame, *, families, expected_seeds=DEFAULT_SEEDS,
                           expected_rows=None, model_failures=(),
                           equivalence_contrast=None, margin=.003, alpha=.05,
                           reps=2000, block=60, seed=927601):
    """Analyze long-form paired gains under frozen families and scoring masks.

    ``frame`` has columns seed, market, date, contrast, gain. Positive gain
    favours the candidate; callers form score differences on exactly paired
    model/target rows before calling. ``families`` maps family IDs to registered
    contrast IDs, including unrun reserved slots. ``expected_rows`` maps contrast
    IDs to the frozen market/date DataFrames (recommended; otherwise the union
    of submitted rows is used and its unverifiable completeness is disclosed).

    A failure record has seed (optional for a failure affecting every seed),
    contrast or contrasts, and reason/failure_code. Missing seeds, missing rows,
    invalid gains, or declared failures make the affected contrast unavailable;
    its p-value is one in every registered family. Other complete contrasts can
    still be tested in that unchanged family. No surviving-seed mean is reported.

    The crossed interval is headline; the conditional date-only interval holds
    the realized seed battery fixed. All contrasts use identical date and seed
    resampling paths. Primary stock equivalence is outside the supporting Holm
    family, exactly as in the transfer protocol.
    """
    if not families or any(not isinstance(v, (list, tuple)) or not v or len(v) != len(set(v))
                           for v in families.values()):
        raise ValueError("Provide nonempty family lists without duplicate slots")
    contrast_ids = list(dict.fromkeys(c for cs in families.values() for c in cs))
    if any(not isinstance(c, str) for c in contrast_ids):
        raise ValueError("Contrast IDs must be strings")
    seeds = tuple(str(x) for x in expected_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct expected training seeds are required")
    if not np.isfinite(margin) or margin <= 0 or not 0 < alpha < .5:
        raise ValueError("Positive finite margin and alpha in (0,.5) required")
    if any(int(v) != v or v < 1 for v in (reps, block)):
        raise ValueError("Positive integer resamples and block length required")
    required = {"seed", "market", "date", "contrast", "gain"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing columns: " + str(sorted(required - set(frame.columns))))
    f = frame.copy()
    if f[["seed", "market", "date", "contrast"]].isna().any().any():
        raise ValueError("Observation labels cannot be missing")
    f["seed"], f["market"] = f.seed.astype(str), f.market.astype(str)
    f["date"] = pd.to_datetime(f.date, errors="raise")
    if f.date.dt.tz is not None or not f.date.eq(f.date.dt.normalize()).all():
        raise ValueError("Use timezone-naive daily date labels")
    if f.duplicated(["seed", "contrast", "market", "date"]).any():
        raise ValueError("Duplicate seed/contrast/market/date observations")
    if set(f.seed) - set(seeds) or set(f.contrast) - set(contrast_ids):
        raise ValueError("Unregistered training seed or contrast")
    if equivalence_contrast is not None and equivalence_contrast not in contrast_ids:
        raise ValueError("Equivalence contrast must be registered")
    if expected_rows is not None and set(expected_rows) - set(contrast_ids):
        raise ValueError("Scoring masks contain an unregistered contrast")
    masks, mask_sources = {}, {}
    for c in contrast_ids:
        if expected_rows is not None and c in expected_rows:
            masks[c] = _keys(expected_rows[c])
            mask_sources[c] = "supplied frozen scoring mask"
        else:
            masks[c] = _keys(f.loc[f.contrast == c, ["market", "date"]].drop_duplicates())
            mask_sources[c] = "submitted union only; globally missing rows cannot be detected"
    failure_map = {}
    for failure in model_failures:
        affected = failure.get("contrasts", [failure.get("contrast")])
        if not affected or any(c not in contrast_ids for c in affected):
            raise ValueError("Every failure must identify registered affected contrasts")
        failure_seed = failure.get("seed")
        affected_seeds = seeds if failure_seed is None else (str(failure_seed),)
        if set(affected_seeds) - set(seeds):
            raise ValueError("Failure identifies an unregistered seed")
        reason = str(failure.get("reason", failure.get("failure_code", "declared_model_failure")))
        for c in affected:
            for s in affected_seeds:
                failure_map.setdefault((c, s), []).append(reason)
    available, output = {}, {}
    for c in contrast_ids:
        reasons, statuses, seed_frames = [], {}, {}
        mask_set = set(masks[c].itertuples(index=False, name=None))
        for s in seeds:
            part = f.loc[(f.contrast == c) & (f.seed == s)].copy()
            issues = list(failure_map.get((c, s), ()))
            if part.empty:
                issues.append("missing_seed_or_unrun_slot")
            actual = set(part[["market", "date"]].itertuples(index=False, name=None))
            if actual != mask_set:
                issues.append("scoring_mask_mismatch:missing=%d,extra=%d" %
                              (len(mask_set - actual), len(actual - mask_set)))
            if not np.isfinite(part.gain.to_numpy(float)).all():
                issues.append("invalid_or_missing_gain")
            statuses[s] = {"status": "unavailable" if issues else "available", "reasons": issues,
                           "rows": len(part)}
            reasons.extend(s + ":" + issue for issue in issues)
            seed_frames[s] = part
        if reasons:
            output[c] = _unavailable(reasons, statuses)
            # Report individually valid seeds without forming a survivor mean.
            for s in seeds:
                if statuses[s]["status"] == "available":
                    output[c]["per_seed_mean_gain"][s] = float(
                        seed_frames[s].groupby("market").gain.mean().mean())
        else:
            available[c] = (seed_frames, statuses)
    calendar = pd.DatetimeIndex(sorted(set().union(*(set(m.date) for m in masks.values()))))
    if available:
        date_counts, seed_counts = _resampling_counts(len(calendar), len(seeds), reps, block, seed)
    eq = None
    for c, (parts, statuses) in available.items():
        seed_means, market_means, seed_draws = {}, {}, []
        try:
            for s in seeds:
                pivot = parts[s].pivot(index="date", columns="market", values="gain").reindex(calendar)
                values = pivot.to_numpy(float)
                denom = date_counts @ np.isfinite(values).astype(float)
                if np.any(denom == 0):
                    raise ValueError("bootstrap draw omits a market; insufficient calendar coverage")
                means = np.nanmean(values, axis=0)
                market_means[s] = dict(zip(pivot.columns, map(float, means)))
                seed_means[s] = float(means.mean())
                seed_draws.append(np.mean((date_counts @ np.nan_to_num(values, nan=0.)) / denom, axis=1))
        except ValueError as exc:
            output[c] = _unavailable([str(exc)], statuses)
            continue
        paths = np.column_stack(seed_draws)
        estimate = float(np.mean(list(seed_means.values())))
        conditional, crossed = paths.mean(axis=1), np.mean(paths * seed_counts, axis=1)
        output[c] = {
            "status": "available", "seed_status": statuses,
            "per_seed_mean_gain": seed_means, "per_seed_per_market_mean_gain": market_means,
            "conditional": _summary(estimate, conditional, "synchronized circular calendar blocks; all realized seeds fixed"),
            "crossed": _summary(estimate, crossed, "crossed whole-seed and synchronized circular calendar blocks"),
            "all_seeds_positive": bool(all(v > 0 for v in seed_means.values())),
            "claim_survives": False, "holm_by_family": {},
            "seed_standard_deviation": float(np.std(list(seed_means.values()), ddof=1)),
            "markets": len(next(iter(market_means.values()))),
            "rows_per_seed": len(masks[c]), "calendar_dates": len(calendar),
        }
        if c == equivalence_contrast:
            eq = {"contrast": c, "role": "primary stock equivalence, separate from supporting Holm tests",
                  "crossed": _equivalence(estimate, crossed, margin, alpha),
                  "conditional": _equivalence(estimate, conditional, margin, alpha)}
    family_output = {}
    for family, contrasts in families.items():
        adjusted = {method: holm([output[c][method]["p_null"] for c in contrasts])
                    for method in ("crossed", "conditional")}
        for i, c in enumerate(contrasts):
            r = output[c]
            entry = {method + "_p_holm": float(adjusted[method][i]) for method in adjusted}
            for method in ("crossed", "conditional"):
                entry[method + "_claim_survives"] = bool(
                    r["status"] == "available" and r["all_seeds_positive"] and
                    r[method]["interval95"][0] > 0 and adjusted[method][i] <= alpha)
            r["holm_by_family"][family] = entry
        family_output[family] = {
            "registered_slots": list(contrasts), "size": len(contrasts),
            "all_slots_available": all(output[c]["status"] == "available" for c in contrasts),
            "unavailable_slots": [c for c in contrasts if output[c]["status"] != "available"],
        }
    for r in output.values():
        r["claim_survives"] = bool(all(x["crossed_claim_survives"] for x in r["holm_by_family"].values()))
    if equivalence_contrast is not None and eq is None:
        eq = {"contrast": equivalence_contrast, "crossed": {"status": "unavailable", "p_tost": 1.,
              "equivalence_supported": False}, "conditional": {"status": "unavailable", "p_tost": 1.,
              "equivalence_supported": False}}
    return {
        "estimand": "arithmetic mean of per-seed equal-market score gains; not a density ensemble",
        "headline_uncertainty": "crossed whole-seed and synchronized calendar-block bootstrap",
        "expected_seeds": list(seeds), "contrasts": output, "families": family_output,
        "equivalence": eq, "scoring_mask_sources": mask_sources,
        "all_scoring_masks_supplied": all(c in (expected_rows or {}) for c in contrast_ids),
        "resamples": reps, "block_observed_dates": block, "resampling_seed": seed,
        "scope": SCOPE,
        "survival_rule": "All planned seeds have positive gain, crossed 95% interval is above zero, and crossed Holm p <= alpha.",
        "missing_policy": "Keep affected registered slots with p=1; never average surviving seeds or replace failures.",
    }
