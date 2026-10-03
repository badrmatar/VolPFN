"""Paired, frozen-forecast QLIKE comparison on a common target-only mask.

Forecast moments and both targets are in percent-squared units (the forecast
mean is in percent). This module neither forecasts nor replaces invalid outputs.
The bundle validator must additionally verify the externally frozen origin mask
and provenance: agreement between supplied models cannot establish that mask.
"""
from collections import Counter, defaultdict
from datetime import date
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.population_transfer.statistics import _basic_interval, _calendar_counts, holm


VOLPFN_SEEDS = (8101, 8102, 8103, 8104, 8105)
INFORMATION_GROUPS = {
    "V": "VolPFN: five fixed training seeds",
    "E": "E: return-only econometric/comparator information",
    "F": "F: foundation-model information",
    "R": "R: realized-measure information (separate inference family)",
}
SCOPE = (
    "Exposed historical comparison, conditional on the fixed forecasts, five trained "
    "VolPFN models, development scalars, adapters and population. Calendar uncertainty "
    "only; excludes training-seed population, calibration and research-selection "
    "uncertainty. No ensemble, untouched-data, universal or state-of-the-art claim."
)


def qlike(target, forecast):
    """Zero-safe loss y/h + log(h), omitting only target-dependent constants."""
    y, h = np.broadcast_arrays(np.asarray(target, float), np.asarray(forecast, float))
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("QLIKE targets must be finite and nonnegative")
    if not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError("QLIKE forecasts must be finite and strictly positive")
    result = y / h + np.log(h)
    if not np.isfinite(result).all():
        raise ValueError("QLIKE overflow; forecasts cannot be dropped or silently clipped")
    return result


def _seed_label(seed):
    return "none" if seed is None else str(seed)


def _number(value, *, missing=False):
    if value is None and missing:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected numeric target/moment") from exc


def _same(a, b):
    return a == b or (np.isnan(a) and np.isnan(b))


def _validate(records, specs):
    if not isinstance(specs, dict) or "volpfn" not in specs:
        raise ValueError("Model specifications must include volpfn")
    for name, spec in specs.items():
        if spec.get("family") not in INFORMATION_GROUPS:
            raise ValueError("Unknown information family: " + name)
        if not isinstance(spec.get("output_adapter"), str) or not spec["output_adapter"].strip():
            raise ValueError("An explicit output_adapter is required: " + name)
        if (name == "volpfn") != (spec["family"] == "V"):
            raise ValueError("Only volpfn may use family V")
    r_models = [m for m, s in specs.items() if s["family"] == "R"]
    if len(r_models) not in (0, 2):
        raise ValueError("R inference requires both declared controls (four proxy slots)")
    grouped = defaultdict(dict)
    required = {"model", "seed", "market", "date", "phase", "r2", "rk",
                "mean", "variance", "second_moment", "substituted"}
    for original in records:
        if not required.issubset(original):
            raise ValueError("Missing record fields: " + str(sorted(required - set(original))))
        row = dict(original)
        model, seed = row["model"], row["seed"]
        if model not in specs:
            raise ValueError("Undeclared model: " + str(model))
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
            raise ValueError("Seeds must be integers or null")
        if row["phase"] not in ("calibration", "evaluation"):
            raise ValueError("Unknown phase")
        if not isinstance(row["market"], str) or not row["market"]:
            raise ValueError("Nonempty market names required")
        try:
            row["date"] = date.fromisoformat(row["date"]).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("Dates must be ISO calendar dates") from exc
        if not isinstance(row["substituted"], (bool, np.bool_)):
            raise ValueError("substituted must be an explicit boolean")
        if not isinstance(row.get("floor_applied", False), (bool, np.bool_)):
            raise ValueError("floor_applied must be boolean when supplied")
        row["floor_applied"] = bool(row.get("floor_applied", False))
        for field in ("r2", "mean", "variance", "second_moment"):
            row[field] = _number(row[field])
        row["rk"] = _number(row["rk"], missing=True)
        if not np.isfinite(row["r2"]) or row["r2"] < 0:
            raise ValueError("Every retained R2 target must be finite and nonnegative")
        if not all(np.isfinite(row[k]) for k in ("mean", "variance", "second_moment")):
            raise ValueError("Nonfinite forecasts; no availability-based intersection allowed")
        if row["variance"] <= 0 or row["second_moment"] <= 0:
            raise ValueError("Nonpositive forecasts must be handled by the frozen adapter")
        if not np.isclose(row["second_moment"], row["variance"] + row["mean"] ** 2,
                          rtol=1e-7, atol=1e-10):
            raise ValueError("Second moment must equal variance plus squared mean")
        if specs[model]["family"] == "F" and (row["mean"] != 0 or
                row["variance"] != row["second_moment"]):
            raise ValueError("F adapter requires exactly zero mean and variance = second moment")
        key = (row["phase"], row["market"], row["date"])
        group = grouped[model, seed]
        if key in group:
            raise ValueError("Duplicate model/seed/phase/market/date record")
        group[key] = row
    observed = {m for m, _ in grouped}
    if observed != set(specs):
        raise ValueError("All declared models required; missing: " + str(sorted(set(specs) - observed)))
    seeds = {}
    for model in specs:
        found = {s for m, s in grouped if m == model}
        expected = set(VOLPFN_SEEDS) if model == "volpfn" else set(specs[model].get("seeds", found))
        if found != expected or not found:
            raise ValueError("All declared seeds required for " + model)
        if model != "volpfn" and len(found) != 1 and "seeds" not in specs[model]:
            raise ValueError("Multiple comparator seeds require an explicit seeds specification")
        seeds[model] = sorted(found, key=lambda s: -1 if s is None else s)
    reference = grouped["volpfn", VOLPFN_SEEDS[0]]
    keys = set(reference)
    cal_keys = sorted(k for k in keys if k[0] == "calibration")
    eval_keys = sorted(k for k in keys if k[0] == "evaluation")
    counts = Counter(k[1] for k in cal_keys)
    if len(cal_keys) != 72 or len(counts) != 3 or set(counts.values()) != {24}:
        raise ValueError("Calibration requires exactly 72 origins: three markets with 24 dates each")
    if not eval_keys:
        raise ValueError("Nonempty evaluation population required")
    if {(k[1], k[2]) for k in cal_keys} & {(k[1], k[2]) for k in eval_keys}:
        raise ValueError("Calibration and evaluation origins overlap")
    for (model, seed), rows in grouped.items():
        if set(rows) != keys:
            raise ValueError("Identical calibration and evaluation keys required for every model/seed: " + model)
        for key in keys:
            if not all(_same(rows[key][f], reference[key][f]) for f in ("r2", "rk")):
                raise ValueError("Target mismatch across models or seeds at " + str(key))
    return grouped, seeds, cal_keys, eval_keys, reference


def _equal_market(values, markets):
    per_market = {m: float(np.mean(values[markets == m])) for m in sorted(set(markets))}
    return float(np.mean(list(per_market.values()))), per_market


def _panel(dates, markets, gains, calendar, population):
    matrix = np.full((len(calendar), len(population)), np.nan)
    date_index = {d: i for i, d in enumerate(calendar)}
    market_index = {m: i for i, m in enumerate(population)}
    for d, m, gain in zip(dates, markets, gains):
        matrix[date_index[d], market_index[m]] = gain
    return matrix


def _resample(matrix, counts):
    """Use one union-date count vector; preserve missing market dates per draw."""
    valid = np.isfinite(matrix).astype(float)
    denominator = counts @ valid
    if np.any(denominator == 0):
        raise ValueError("A bootstrap draw omits a market; cannot silently drop draws/markets")
    market_means = np.nanmean(matrix, axis=0)
    draws = np.mean((counts @ np.nan_to_num(matrix, nan=0.0)) / denominator, axis=1)
    return float(market_means.mean()), draws, market_means


def _family_adjust(rows, slots):
    """Correct all proxy/competitor tests together within their declared family."""
    if not rows:
        return
    if len(rows) != slots:
        raise ValueError("Missing hypothesis slots in declared testing family")
    for row, p in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"] = float(p)
        row["directional_pass"] = bool(row["gain"] > 0 and p < .05 and row["family_interval95"][0] > 0)
        row["negative_supported"] = bool(row["gain"] < 0 and p < .05 and row["family_interval95"][1] < 0)
        row["verdict"] = ("supported_positive_gain" if row["directional_pass"] else
                          "supported_negative_gain" if row["negative_supported"] else "inconclusive")


def analyse(records, model_specs, resamples=20000, blocks=(10, 5, 20), seed=2026091902, *, archive=None):
    """Return JSON-compatible scores/inference; optionally fill an NPZ-ready archive.

    The primary population is finite, strictly positive RK origins, chosen from
    targets alone. R2 and RK use that identical population. S_all retains every
    supplied evaluation origin and is a descriptive R2 sensitivity. Raw forecasts
    are descriptive too; multiplicity-adjusted inference is primary/calibrated.
    """
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples < 1:
        raise ValueError("Positive integer resamples required")
    blocks = tuple(blocks)
    if not blocks or len(set(blocks)) != len(blocks) or any(
            isinstance(b, bool) or int(b) != b or b < 1 for b in blocks):
        raise ValueError("Distinct positive integer block lengths required")
    grouped, seeds, cal_keys, eval_keys, reference = _validate(records, model_specs)
    models = sorted(model_specs)
    all_markets = np.array([k[1] for k in eval_keys])
    all_dates = np.array([k[2] for k in eval_keys])
    target = {p: np.array([reference[k][p] for k in eval_keys]) for p in ("r2", "rk")}
    primary = np.isfinite(target["rk"]) & (target["rk"] > 0)
    population = sorted(set(all_markets))
    if any(not np.any(primary & (all_markets == m)) for m in population):
        raise ValueError("Each declared evaluation market must have primary target observations")
    masks = {"primary": primary, "raw": primary,
             "S_all": np.ones(len(eval_keys), dtype=bool), "S_all_raw": np.ones(len(eval_keys), dtype=bool)}
    calibration, losses, seed_scores, substitutions, floors = {}, {}, {}, {}, {}
    cal_markets = np.array([k[1] for k in cal_keys])
    for model in models:
        calibration[model], losses[model], seed_scores[model] = {}, {}, {}
        substitutions[model] = {}
        floors[model] = {}
        for training_seed in seeds[model]:
            label = _seed_label(training_seed)
            rows = grouped[model, training_seed]
            ratios = np.array([rows[k]["r2"] / rows[k]["second_moment"] for k in cal_keys])
            scalar, per_market = _equal_market(ratios, cal_markets)
            if not np.isfinite(scalar) or scalar <= 0:
                raise ValueError("Calibration scalar must be positive and finite: " + model)
            calibration[model][label] = {"scalar": scalar, "per_market_ratio_mean": per_market, "rows": 72}
            forecast = {"r2": np.array([rows[k]["second_moment"] for k in eval_keys]),
                        "rk": np.array([rows[k]["variance"] for k in eval_keys])}
            losses[model][label], seed_scores[model][label] = {}, {}
            for setting, mask in masks.items():
                scale = 1.0 if setting in ("raw", "S_all_raw") else scalar
                proxies = ("r2", "rk") if setting in ("primary", "raw") else ("r2",)
                losses[model][label][setting], seed_scores[model][label][setting] = {}, {}
                for proxy in proxies:
                    values = qlike(target[proxy][mask], scale * forecast[proxy][mask])
                    losses[model][label][setting][proxy] = values
                    average, market_loss = _equal_market(values, all_markets[mask])
                    seed_scores[model][label][setting][proxy] = {"loss": average, "per_market_loss": market_loss}
            for flag, destination in (("substituted", substitutions), ("floor_applied", floors)):
                destination[model][label] = {
                    "calibration": sum(bool(rows[k][flag]) for k in cal_keys),
                    "evaluation_S_all": sum(bool(rows[k][flag]) for k in eval_keys),
                    "evaluation_primary": sum(bool(rows[k][flag]) for k, keep in zip(eval_keys, primary) if keep),
                    "per_market_primary": {m: sum(bool(rows[k][flag]) for k, keep in zip(eval_keys, primary)
                                                   if keep and k[1] == m) for m in population},
                }
    averaged, leaderboards = {}, {}
    for setting, mask in masks.items():
        averaged[setting], leaderboards[setting] = {}, {}
        for proxy in (("r2", "rk") if setting in ("primary", "raw") else ("r2",)):
            averaged[setting][proxy], table = {}, []
            for model in models:
                # Average LOSSES over seeds; averaging moments creates an ensemble.
                values = np.mean([losses[model][s][setting][proxy] for s in losses[model]], axis=0)
                averaged[setting][proxy][model] = values
                point, per_market = _equal_market(values, all_markets[mask])
                individual = [seed_scores[model][s][setting][proxy]["loss"] for s in losses[model]]
                spec = model_specs[model]
                table.append({"model": model, "family": spec["family"],
                              "information_group": spec.get("information_group", INFORMATION_GROUPS[spec["family"]]),
                              "output_adapter": spec["output_adapter"], "loss": point,
                              "per_market_loss": per_market, "seed_range": [min(individual), max(individual)],
                              "ranking_population": "realized_information_R_only" if spec["family"] == "R" else "main_V_E_F",
                              "seeds": len(individual)})
            for row in table:
                row["rank"] = 1 + sum(other["loss"] < row["loss"] for other in table
                                      if other["ranking_population"] == row["ranking_population"])
            leaderboards[setting][proxy] = sorted(table, key=lambda r: (r["rank"], r["model"]))

    def descriptive(setting, proxy, model):
        market = all_markets[masks[setting]]
        gains = averaged[setting][proxy][model] - averaged[setting][proxy]["volpfn"]
        gain, per_market = _equal_market(gains, market)
        individual = {s: _equal_market(averaged[setting][proxy][model] - losses["volpfn"][s][setting][proxy], market)[0]
                      for s in losses["volpfn"]}
        return {"model": model, "proxy": proxy, "family": model_specs[model]["family"],
                "gain": gain, "per_market_gain": per_market,
                "markets_improved": sum(g > 0 for g in per_market.values()), "markets": len(per_market),
                "per_seed_gain": individual, "seed_range": [min(individual.values()), max(individual.values())],
                "positive_seeds": sum(g > 0 for g in individual.values())}, gains

    secondary = {s: [descriptive(s, p, m)[0] for p in leaderboards[s] for m in models if m != "volpfn"]
                 for s in ("raw", "S_all", "S_all_raw")}
    complete_case = []
    primary_keys = [k for k, keep in zip(eval_keys, primary) if keep]
    for model in models:
        if model == "volpfn":
            continue
        keep = np.array([not any(grouped[model, s][k]["substituted"] for s in seeds[model])
                         for k in primary_keys], dtype=bool)
        market = all_markets[primary][keep]
        for proxy in ("r2", "rk"):
            item = {"model": model, "proxy": proxy, "family": model_specs[model]["family"],
                    "rows": int(keep.sum()), "markets": len(set(market)),
                    "excluded_comparator_substituted_rows": int((~keep).sum()),
                    "scope": "Descriptive smaller model-specific sample, excluding any comparator-seed substituted origin; no hypothesis test; not the primary comparison"}
            if keep.any():
                control = averaged["primary"][proxy][model][keep]
                candidate = averaged["primary"][proxy]["volpfn"][keep]
                gain, per_market = _equal_market(control - candidate, market)
                individual = {s: _equal_market(control - losses["volpfn"][s]["primary"][proxy][keep], market)[0]
                              for s in losses["volpfn"]}
                item.update(status="available", gain=gain, per_market_gain=per_market,
                            control_loss=_equal_market(control, market)[0],
                            volpfn_loss=_equal_market(candidate, market)[0],
                            per_seed_gain=individual, seed_range=[min(individual.values()), max(individual.values())],
                            per_market_rows={m: int(np.sum(market == m)) for m in sorted(set(market))})
            else:
                item.update(status="unavailable", gain=None, reason="Every primary comparator origin was substituted")
            complete_case.append(item)
    secondary["complete_case"] = complete_case
    dates, markets = all_dates[primary], all_markets[primary]
    calendar = sorted(set(dates))
    main_slots = 2 * sum(s["family"] in ("E", "F") for s in model_specs.values())
    refit_slots = 4 if any(s["family"] == "R" for s in model_specs.values()) else 0
    templates = []
    if archive is not None:
        archive.update(primary_calendar_dates=np.asarray(calendar), primary_markets=np.asarray(population),
                       primary_row_dates=dates.copy(), primary_row_markets=markets.copy())
    for model in models:
        if model == "volpfn":
            continue
        for proxy in ("r2", "rk"):
            row, gains = descriptive("primary", proxy, model)
            row["testing_family"] = "realized_information" if row["family"] == "R" else "main_E_F"
            row["family_slots"] = refit_slots if row["family"] == "R" else main_slots
            matrix = _panel(dates, markets, gains, calendar, population)
            templates.append((row, matrix))
            if archive is not None:
                archive[f"gains__{model}__{proxy}"] = gains.copy()
    contrasts = []
    for block in blocks:
        counts = _calendar_counts(len(calendar), int(resamples), int(block), seed)
        if archive is not None:
            archive[f"counts__block{block}"] = counts.astype(np.int32)
        current = []
        for template, matrix in templates:
            row = dict(template)
            estimate, draws, _ = _resample(matrix, counts)
            np.testing.assert_allclose(estimate, row["gain"], rtol=1e-12, atol=1e-12)
            errors = draws - estimate
            row.update(block=int(block), block_role="main" if block == blocks[0] else "sensitivity",
                       p_raw=float((1 + np.count_nonzero(np.abs(errors) >= abs(estimate))) / (resamples + 1)),
                       interval95=_basic_interval(estimate, draws, .95),
                       family_interval95=_basic_interval(estimate, draws, 1 - .05 / row["family_slots"]),
                       marginal_family_interval_coverage=1 - .05 / row["family_slots"])
            current.append(row)
            if archive is not None:
                archive[f"draws__{row['model']}__{row['proxy']}__block{block}"] = draws.copy()
        for family, slots in (("main_E_F", main_slots), ("realized_information", refit_slots)):
            _family_adjust([r for r in current if r["testing_family"] == family], slots)
        contrasts.extend(current)

    def sample(mask):
        return {"rows": int(mask.sum()), "markets": len(set(all_markets[mask])),
                "calendar_dates": len(set(all_dates[mask])),
                "per_market_rows": {m: int(np.sum(mask & (all_markets == m))) for m in population}}

    return {"kind": "combined_dual_proxy_leaderboard_v1", "status": "complete", "scope": SCOPE,
            "settings": {"resamples": int(resamples), "blocks": list(map(int, blocks)), "main_block": int(blocks[0]),
                         "seed": int(seed), "main_family_slots": main_slots, "realized_family_slots": refit_slots,
                         "alpha": .05, "weighting": "equal markets, time mean within market, mean loss over fixed seeds",
                         "bootstrap": "synchronized union-date circular blocks; absent market dates remain absent",
                         "interval": "basic centered bootstrap; Bonferroni coverage 1-alpha/K",
                         "p_value": "two-sided centered bootstrap, plus-one correction; Holm across both proxies",
                         "sensitivity": "block-specific families; sensitivity settings cannot replace the declared main result",
                         "units": "percent-squared for targets, variance and second moment; percent for mean"},
            "calibration": calibration, "calibration_rule": "72 frozen origins, 3 markets x 24; c = equal-market mean(R2 / raw second moment), independently per model/seed; same c for both proxies",
            "forecast_rule": {"r2": "c * second_moment", "rk": "c * variance", "F": "explicit zero mean, variance = second_moment"},
            "models": model_specs, "samples": {"primary": sample(primary), "S_all": sample(masks["S_all"]),
                "primary_excluded_rk_rows": int((~primary).sum()),
                "mask_rule": "finite strictly positive RK only; common to all models and both primary proxies; no forecast-availability intersection"},
            "leaderboards": leaderboards, "contrasts": contrasts, "secondary_contrasts": secondary,
            "seed_scores": seed_scores, "substitution_counts": substitutions, "floor_counts": floors,
            "floor_count_rule": "Counts actual floor_applied flags independently of substituted flags; absent optional flags mean false; counts are model/seed forecast records",
            "omitted_optional_analyses": ["model confidence set", "ratios"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--blocks", type=int, nargs="+", default=[10, 5, 20])
    parser.add_argument("--seed", type=int, default=2026091902)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    specs = json.loads(args.spec.read_text())
    archive = {}
    result = analyse(records, specs, args.resamples, args.blocks, args.seed, archive=archive)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "bootstrap_archive.npz", **archive)
    from experiments.combined_leaderboard.report import write_report
    outputs = write_report(result, args.out)
    print(json.dumps({"status": "complete", "out": str(args.out.resolve()), "reports": outputs}, indent=2))


if __name__ == "__main__":
    main()
