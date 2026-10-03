"""Analysis of a declared historical extension; never fits or selects forecasts.

The external bundle is authoritative for origins and targets. A missing model,
seed or forecast stops analysis; it never changes the comparison population.
Positive gains mean comparator loss minus VolPFN loss. Information groups remain
visible even though all primary contrasts share one multiplicity family.
"""
from collections import Counter, defaultdict
from datetime import date

import numpy as np

from experiments.combined_leaderboard.statistics import (
    VOLPFN_SEEDS, INFORMATION_GROUPS, _equal_market, _panel, _resample, qlike,
)
from experiments.population_transfer.statistics import _basic_interval, _calendar_counts, holm


SCOPE = (
    "Historical extension on previously exposed data, conditional on the declared "
    "forecasts, retained training seeds, calibration, adapters and markets. Calendar "
    "uncertainty only; excludes training-population, calibration-estimation, model "
    "selection and pretraining-overlap uncertainty. Ranks are descriptive, not proof "
    "of a universal winner. Sensitivities cannot replace the frozen main result."
)


class CoverageError(ValueError):
    """Incomplete/extra/duplicate forecasts, with a machine-readable audit."""
    def __init__(self, audit):
        self.audit = audit
        super().__init__("Forecast coverage does not match the declared bundle/roster")


def _records(records):
    return records.to_dict("records") if hasattr(records, "to_dict") else list(records)


def _key(row):
    if row.get("phase") not in ("calibration", "evaluation"):
        raise ValueError("Unknown record phase")
    if not isinstance(row.get("market"), str) or not row["market"]:
        raise ValueError("Nonempty market required")
    return row["phase"], row["market"], date.fromisoformat(row["date"]).isoformat()


def _seed(seed):
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("Seed must be an integer or null")
    return None if seed is None else int(seed)


def _label(seed):
    return "none" if seed is None else str(seed)


def _number(value):
    return float("nan") if value is None else float(value)


def _same(a, b):
    return a == b or (np.isnan(a) and np.isnan(b))


def _specifications(specs):
    if "volpfn" not in specs:
        raise ValueError("Declared roster must include volpfn")
    seeds = {}
    for model, spec in specs.items():
        if spec.get("family") not in INFORMATION_GROUPS:
            raise ValueError("Unknown information group for " + model)
        if (model == "volpfn") != (spec["family"] == "V"):
            raise ValueError("Only volpfn belongs to V")
        if not isinstance(spec.get("output_adapter"), str) or not spec["output_adapter"].strip():
            raise ValueError("Explicit output adapter required for " + model)
        if not isinstance(spec.get("primary", True), bool):
            raise ValueError("primary must be boolean")
        if "seeds" not in spec or not isinstance(spec["seeds"], (list, tuple)):
            raise ValueError("Explicit seed list required for " + model)
        declared = [_seed(s) for s in spec["seeds"]]
        if not declared or len(set(declared)) != len(declared):
            raise ValueError("Nonempty unique declared seeds required")
        seeds[model] = declared
    if set(seeds["volpfn"]) != set(VOLPFN_SEEDS):
        raise ValueError("All five fixed VolPFN seeds are required")
    if not specs["volpfn"].get("primary", True):
        raise ValueError("VolPFN must be primary")
    if not any(m != "volpfn" and s.get("primary", True) for m, s in specs.items()):
        raise ValueError("At least one declared primary comparator required")
    return seeds


def _bundle_rows(bundle):
    if not isinstance(bundle, dict) or not bundle.get("rows"):
        raise ValueError("An external frozen origin/target bundle is required")
    result = {}
    for original in bundle["rows"]:
        key = _key(original)
        if key in result:
            raise ValueError("Duplicate bundle origin")
        row = dict(original, r2=_number(original["r2"]), rk=_number(original.get("rk")))
        if not np.isfinite(row["r2"]) or row["r2"] < 0:
            raise ValueError("Invalid bundle squared-return target")
        eligible = bool(np.isfinite(row["rk"]) and row["rk"] > 0)
        if "primary_eligible" in row and row["primary_eligible"] != eligible:
            raise ValueError("Bundle primary mask must depend only on target RK")
        result[key] = row
    calibration = [k for k in result if k[0] == "calibration"]
    counts = Counter(k[1] for k in calibration)
    if counts != {"FTSE": 24, "N225": 24, "SPX": 24}:
        raise ValueError("Calibration requires the canonical 72 origins: FTSE/N225/SPX x24")
    evaluation = [k for k in result if k[0] == "evaluation"]
    if not evaluation:
        raise ValueError("Nonempty evaluation required")
    if {(k[1], k[2]) for k in calibration} & {(k[1], k[2]) for k in evaluation}:
        raise ValueError("Calibration/evaluation origins overlap")
    declared_markets = bundle.get("markets")
    if declared_markets is not None and set(declared_markets) != {k[1] for k in evaluation}:
        raise ValueError("Bundle market roster differs from evaluation population")
    return result


def coverage_audit(records, model_specs, bundle):
    """Read-only coverage report, usable before complete forecasts are available."""
    records = _records(records)
    seeds, expected = _specifications(model_specs), _bundle_rows(bundle)
    observed = defaultdict(Counter)
    failed = Counter()
    for row in records:
        group = row["model"], _seed(row["seed"])
        observed[group][_key(row)] += 1
        if row.get("forecast_failed") is True:
            failed[group] += 1
    declared = {(m, s) for m, ss in seeds.items() for s in ss}
    extras = [{"model": m, "seed": s, "rows": sum(observed[m, s].values())}
              for m, s in observed if (m, s) not in declared]
    models = {}
    for m, ss in seeds.items():
        detail = {}
        for s in ss:
            got = observed[m, s]
            missing = set(expected) - set(got)
            extra = set(got) - set(expected)
            duplicates = sum(n - 1 for n in got.values())
            detail[_label(s)] = {
                "expected": len(expected), "observed": sum(got.values()),
                "missing": len(missing), "extra": len(extra), "duplicates": duplicates,
                "reported_failed_forecasts": failed[m, s],
                "complete": not missing and not extra and not duplicates and not failed[m, s],
                "missing_by_phase": dict(Counter(k[0] for k in missing)),
                "missing_by_market": dict(Counter(k[1] for k in missing)),
            }
        models[m] = {"seeds": detail, "declared_seed_count": len(ss),
                     "failed_or_incomplete_seed_count": sum(not d["complete"] for d in detail.values())}
    return {"complete": not extras and all(v["failed_or_incomplete_seed_count"] == 0 for v in models.values()),
            "models": models, "unexpected_model_seeds": extras, "records": len(records)}


def _validate(records, specs, bundle):
    seeds, reference = _specifications(specs), _bundle_rows(bundle)
    coverage = coverage_audit(records, specs, bundle)
    if not coverage["complete"]:
        raise CoverageError(coverage)
    grouped = defaultdict(dict)
    for original in records:
        row = dict(original)
        key, model, seed = _key(row), row["model"], _seed(row["seed"])
        for flag in ("substituted", "floor_applied", "forecast_failed"):
            if flag == "substituted" and flag not in row:
                raise ValueError("substituted flag required")
            if not isinstance(row.get(flag, False), (bool, np.bool_)):
                raise ValueError("Forecast flags must be boolean")
            row[flag] = bool(row.get(flag, False))
        if row["forecast_failed"]:
            raise ValueError("Failed forecast cannot be scored or dropped")
        for field in ("mean", "variance", "second_moment", "r2", "rk"):
            row[field] = _number(row.get(field))
        if any(not _same(row[p], reference[key][p]) for p in ("r2", "rk")):
            raise ValueError("Target mismatch against external bundle")
        if not all(np.isfinite(row[k]) for k in ("mean", "variance", "second_moment")):
            raise ValueError("Nonfinite forecast; fixed population cannot shrink")
        if row["variance"] <= 0 or row["second_moment"] <= 0:
            raise ValueError("Positive moments required from frozen adapter")
        if not np.isclose(row["second_moment"], row["variance"] + row["mean"] ** 2, rtol=1e-7, atol=1e-10):
            raise ValueError("M2 must equal variance plus squared mean")
        if specs[model].get("zero_mean", specs[model]["family"] == "F"):
            if row["mean"] != 0 or row["variance"] != row["second_moment"]:
                raise ValueError("Declared zero-mean adapter was violated")
        grouped[model, seed][key] = row
    return grouped, seeds, reference, coverage


def _family_adjust(rows, slots, alpha=.05):
    if len(rows) != slots or slots < 1:
        raise ValueError("Every declared joint-family slot must be present")
    for row, p in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"] = float(p)
        row["directional_pass"] = bool(row["gain"] > 0 and p < alpha
                                       and row["family_interval95"][0] > 0
                                       and row["all_reference_seeds_positive"])
        # The predeclared p-value tests improvement only. A two-sided family
        # interval can nevertheless disclose adverse evidence without pretending
        # that a large positive-direction p-value tests reverse superiority.
        if row["directional_pass"]:
            row["verdict"] = "supported_positive_gain"
        elif row["gain"] < 0 and row["family_interval95"][1] < 0:
            row["verdict"] = "negative_gain_family_interval_below_zero"
        elif row["gain"] < 0:
            row["verdict"] = "negative_estimate"
        else:
            row["verdict"] = "inconclusive"
    for model in {r["model"] for r in rows}:
        pair = [r for r in rows if r["model"] == model]
        if {r["proxy"] for r in pair} != {"r2", "rk"} or len(pair) != 2:
            raise ValueError("Both proxies required for every primary comparator")
        supported = all(r["directional_pass"] for r in pair)
        for row in pair:
            row["comparison_supported_both_proxies"] = supported


def analyse(records, model_specs, *, bundle, resamples=None, blocks=None, seed=None, archive=None):
    """Calibrated dual-proxy analysis; raw scores remain descriptive.

    Bundle settings may supply resamples/blocks/bootstrap_seed. Explicit arguments
    support synthetic tests, but the caller must freeze actual execution settings.
    """
    settings = bundle.get("settings", {})
    resamples = settings.get("resamples", 20000) if resamples is None else resamples
    blocks = settings.get("blocks", (60, 20, 120)) if blocks is None else blocks
    seed = settings.get("bootstrap_seed", settings.get("seed", 2026092601)) if seed is None else seed
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples < 1:
        raise ValueError("Positive integer bootstrap draws required")
    blocks = tuple(blocks)
    if not blocks or len(set(blocks)) != len(blocks) or any(isinstance(b, bool) or int(b) != b or b < 1 for b in blocks):
        raise ValueError("Distinct positive integer block lengths required")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Nonnegative integer bootstrap seed required")
    records = _records(records)
    grouped, seeds, reference, coverage = _validate(records, model_specs, bundle)
    models = sorted(model_specs)
    primary_models = [m for m in models if m != "volpfn" and model_specs[m].get("primary", True)]
    slots = 2 * len(primary_models)
    cal_keys = sorted(k for k in reference if k[0] == "calibration")
    eval_keys = sorted(k for k in reference if k[0] == "evaluation")
    markets = np.array([k[1] for k in eval_keys]); dates = np.array([k[2] for k in eval_keys])
    cal_markets = np.array([k[1] for k in cal_keys]); population = sorted(set(markets))
    targets = {p: np.array([reference[k][p] for k in eval_keys]) for p in ("r2", "rk")}
    primary = np.isfinite(targets["rk"]) & (targets["rk"] > 0)
    if any(not np.any(primary & (markets == m)) for m in population):
        raise ValueError("Every market needs primary target coverage")
    masks = {"primary": primary, "raw": primary, "S_all": np.ones(len(eval_keys), bool),
             "S_all_raw": np.ones(len(eval_keys), bool)}
    calibration, losses, seed_scores, substitutions, floors = {}, {}, {}, {}, {}
    for model in models:
        for output in (calibration, losses, seed_scores, substitutions, floors):
            output[model] = {}
        for training_seed in seeds[model]:
            label = _label(training_seed); rows = grouped[model, training_seed]
            ratios = np.array([rows[k]["r2"] / rows[k]["second_moment"] for k in cal_keys])
            scalar, per_market = _equal_market(ratios, cal_markets)
            if not np.isfinite(scalar) or scalar <= 0:
                raise ValueError("Invalid fixed calibration scalar")
            calibration[model][label] = {"scalar": scalar, "per_market_ratio_mean": per_market, "rows": 72}
            forecasts = {"r2": np.array([rows[k]["second_moment"] for k in eval_keys]),
                         "rk": np.array([rows[k]["variance"] for k in eval_keys])}
            losses[model][label], seed_scores[model][label] = {}, {}
            for setting, mask in masks.items():
                losses[model][label][setting], seed_scores[model][label][setting] = {}, {}
                scale = 1. if setting.endswith("raw") else scalar
                for proxy in ("r2", "rk") if setting in ("primary", "raw") else ("r2",):
                    values = qlike(targets[proxy][mask], forecasts[proxy][mask] * scale)
                    average, market_loss = _equal_market(values, markets[mask])
                    losses[model][label][setting][proxy] = values
                    seed_scores[model][label][setting][proxy] = {"loss": average, "per_market_loss": market_loss}
            for flag, output in (("substituted", substitutions), ("floor_applied", floors)):
                flags = np.array([rows[k][flag] for k in eval_keys])
                output[model][label] = {"calibration": sum(rows[k][flag] for k in cal_keys),
                    "evaluation_primary": int(flags[primary].sum()), "evaluation_S_all": int(flags.sum()),
                    "per_market_primary": {m: int(np.sum(flags & primary & (markets == m))) for m in population}}
    averaged, leaderboards = {}, {}
    for setting, mask in masks.items():
        averaged[setting], leaderboards[setting] = {}, {}
        for proxy in ("r2", "rk") if setting in ("primary", "raw") else ("r2",):
            averaged[setting][proxy], table = {}, []
            for model in models:
                values = np.mean([v[setting][proxy] for v in losses[model].values()], axis=0)
                averaged[setting][proxy][model] = values
                point, per_market = _equal_market(values, markets[mask])
                spec = model_specs[model]
                scores = [v[setting][proxy]["loss"] for v in seed_scores[model].values()]
                rank_group = "realized_information_R_only" if spec["family"] == "R" else "main_V_E_F"
                table.append({"model": model, "family": spec["family"], "primary": spec.get("primary", True),
                    "information_group": spec.get("information_group", INFORMATION_GROUPS[spec["family"]]),
                    "output_adapter": spec["output_adapter"], "loss": point, "per_market_loss": per_market,
                    "seed_range": [min(scores), max(scores)], "seeds": len(scores),
                    "ranking_population": rank_group if spec.get("primary", True) else "diagnostic_only"})
            for row in table:
                row["rank"] = (1 + sum(other["loss"] < row["loss"] for other in table
                    if other["primary"] and other["ranking_population"] == row["ranking_population"])) if row["primary"] else None
            leaderboards[setting][proxy] = table

    def descriptive(setting, proxy, model):
        mask = masks[setting]
        control = averaged[setting][proxy][model]
        gains = control - averaged[setting][proxy]["volpfn"]
        gain, per_market = _equal_market(gains, markets[mask])
        individual = {s: _equal_market(control - v[setting][proxy], markets[mask])[0]
                      for s, v in losses["volpfn"].items()}
        return {"model": model, "proxy": proxy, "family": model_specs[model]["family"],
            "gain": gain, "per_market_gain": per_market, "markets": len(population),
            "markets_improved": sum(v > 0 for v in per_market.values()), "per_seed_gain": individual,
            "seed_range": [min(individual.values()), max(individual.values())],
            "positive_seeds": sum(v > 0 for v in individual.values()),
            "all_reference_seeds_positive": all(v > 0 for v in individual.values())}, gains

    secondary = {setting: [descriptive(setting, p, m)[0] for p in leaderboards[setting]
                          for m in models if m != "volpfn"] for setting in ("raw", "S_all", "S_all_raw")}
    secondary["diagnostic_primary_scale"] = [descriptive("primary", p, m)[0] for p in ("r2", "rk")
        for m in models if m != "volpfn" and not model_specs[m].get("primary", True)]
    calibration_effects = []
    for model in models:
        for proxy in ("r2", "rk"):
            raw_loss = _equal_market(averaged["raw"][proxy][model], markets[primary])[0]
            calibrated_loss = _equal_market(averaged["primary"][proxy][model], markets[primary])[0]
            effect = {"model": model, "proxy": proxy, "raw_loss": raw_loss,
                      "calibrated_loss": calibrated_loss,
                      "calibrated_minus_raw_loss": calibrated_loss - raw_loss}
            if model != "volpfn":
                raw_gain = descriptive("raw", proxy, model)[0]["gain"]
                calibrated_gain = descriptive("primary", proxy, model)[0]["gain"]
                effect.update(raw_gain=raw_gain, calibrated_gain=calibrated_gain,
                              gain_direction_changed=bool(np.sign(raw_gain) != np.sign(calibrated_gain)))
            calibration_effects.append(effect)
    calendar = sorted(set(dates[primary])); templates = []
    if archive is not None:
        archive.update(primary_calendar_dates=np.array(calendar), primary_markets=np.array(population),
                       primary_row_dates=dates[primary].copy(), primary_row_markets=markets[primary].copy())
    for model in primary_models:
        for proxy in ("r2", "rk"):
            row, gains = descriptive("primary", proxy, model)
            row.update(testing_family="all_primary_comparators_both_proxies", family_slots=slots)
            templates.append((row, _panel(dates[primary], markets[primary], gains, calendar, population)))
            if archive is not None:
                archive[f"gains__{model}__{proxy}"] = gains.copy()
    contrasts = []
    for block in blocks:
        counts = _calendar_counts(len(calendar), int(resamples), int(block), int(seed))
        if archive is not None:
            archive[f"counts__block{block}"] = counts.astype(np.int32)
        current = []
        for template, matrix in templates:
            row = dict(template); estimate, draws, _ = _resample(matrix, counts)
            np.testing.assert_allclose(estimate, row["gain"], rtol=1e-12, atol=1e-12)
            errors = draws - estimate
            row.update(block=int(block), block_role="main" if block == blocks[0] else "sensitivity",
                p_raw=float((1 + np.count_nonzero(errors >= estimate)) / (resamples + 1)),
                interval95=_basic_interval(estimate, draws, .95),
                family_interval95=_basic_interval(estimate, draws, 1 - .05 / slots),
                marginal_family_interval_coverage=1 - .05 / slots)
            current.append(row)
            if archive is not None:
                archive[f"draws__{row['model']}__{row['proxy']}__block{block}"] = draws.copy()
        _family_adjust(current, slots)
        contrasts.extend(current)
    main = [r for r in contrasts if r["block_role"] == "main"]

    def sample(mask):
        return {"rows": int(mask.sum()), "markets": len(set(markets[mask])),
                "calendar_dates": len(set(dates[mask])),
                "per_market_rows": {m: int(np.sum(mask & (markets == m))) for m in population}}

    return {"kind": "frontier_extension_20260926", "status": "complete", "scope": SCOPE,
        "settings": {"resamples": int(resamples), "blocks": list(map(int, blocks)), "main_block": int(blocks[0]),
            "seed": int(seed), "alpha": .05, "primary_family_slots": slots,
            "weighting": "equal markets, time mean within market, mean of separately calibrated seed losses",
            "p_value": "one-sided positive-gain centered circular-block bootstrap with plus-one correction; joint Holm",
            "interval": "two-sided basic intervals; Bonferroni family coverage 1-alpha/K",
            "support_rule": "positive every retained VolPFN seed; positive family lower bound and Holm p<.05 on BOTH proxies at main block",
            "negative_evidence_rule": "negative gains and two-sided family intervals below zero are labeled; positive-direction Holm is not a reverse-direction test",
            "sensitivity": "separate repeated joint family at each block; cannot replace main setting"},
        "models": model_specs, "coverage": coverage, "calibration": calibration,
        "calibration_rule": "canonical 72 origins; equal-market mean(R2/raw M2), separately per model/seed; same scalar for both proxies",
        "forecast_rule": {"r2": "c * second_moment", "rk": "c * variance"},
        "samples": {"primary": sample(primary), "S_all": sample(masks["S_all"]),
                    "primary_excluded_rk_rows": int((~primary).sum())},
        "leaderboards": leaderboards, "contrasts": contrasts,
        "comparison_support": {m: all(r["comparison_supported_both_proxies"] for r in main if r["model"] == m) for m in primary_models},
        "secondary_contrasts": secondary, "seed_scores": seed_scores,
        "calibration_effects": calibration_effects,
        "raw_rule": "after the declared adapter, floor and fallback; before scalar calibration",
        "substitution_counts": substitutions, "floor_counts": floors}
