"""Versioned GP-Vol addendum scoring with exact log-moment representation support.

Nothing in frontier_extension_20260926_v1 is modified. Modes:
  addendum   VolPFN (5 seeds) vs GP-Vol: 2-slot family (primary for the addendum)
  family34   GP-Vol added as a 17th comparator beside the 16 frozen ones (sensitivity)
  selftest   relabel a frozen comparator as 'gpvol' and check that gains, p-values and
             intervals reproduce the frozen results exactly
usage: python3 analyse_addendum.py addendum [--seeds 1 2] [--resamples 20000]
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
FRONTIER = ROOT / "research/outputs/frontier_extension_20260926_v1"
OUT = ROOT / "research/outputs/frontier_gpvol_20260926_v1"
KEEP = ("market", "date", "phase", "index", "r2", "rk", "primary_eligible", "model", "seed", "mean",
        "variance", "second_moment", "log_variance", "log_second_moment", "substituted", "floor_applied", "forecast_failed")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_stats():
    spec = importlib.util.spec_from_file_location(
        "gpvol_statistics_logmoments_v1", HERE / "statistics_logmoments_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cells(root, model, rename=None, seed_map=None):
    rows = []
    base = Path(root) / "cells" / model
    for seed_dir in sorted(base.iterdir()):
        for cell in sorted(seed_dir.iterdir()):
            f = cell / "forecast.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["rows"]:
                r = {k: r.get(k) for k in KEEP}
                if rename:
                    r["model"] = rename
                if seed_map is not None:
                    r["seed"] = seed_map(r["seed"])
                rows.append(r)
    return rows


def gp_spec(seeds, label="gpvol"):
    return dict(family="E", seeds=list(seeds), primary=True, zero_mean=False,
                information_group="Daily returns only",
                output_adapter=("Full GP-Vol RAPCF (200 particles, full latent chains); constant causal mean; "
                                "variance = sd^2 * particle-mixture E[exp(v_t)]" +
                                ("; released-code test-input indexing (diagnostic)" if label == "gpvol_codeidx" else "")))


def run(mode, seeds, resamples, blocks, model="gpvol", selftest_source="garch_t"):
    stats = load_stats()
    bundle = json.loads((FRONTIER / "bundle.json").read_text())
    frozen_specs = bundle["model_specs"]
    records = load_cells(FRONTIER, "volpfn")
    specs = {"volpfn": frozen_specs["volpfn"]}
    if mode == "selftest":
        records += load_cells(FRONTIER, selftest_source, rename="gpvol", seed_map=lambda s: 1)
        specs["gpvol"] = gp_spec([1])
    else:
        records += load_cells(OUT, model)
        specs[model] = gp_spec(seeds, model)
        if mode == "family34":
            for m, s in frozen_specs.items():
                if m != "volpfn":
                    records += load_cells(FRONTIER, m)
                    specs[m] = s
    archive = {}
    b = dict(bundle, settings=dict(resamples=resamples, blocks=list(blocks), bootstrap_seed=2026092601))
    result = stats.analyse(records, specs, bundle=b, archive=archive)
    return result, archive


def summarise(result, model):
    main = {}
    for c in result["contrasts"]:
        if c["model"] == model:
            main.setdefault(c["block"], {})[c["proxy"]] = {
                k: c[k] for k in ("gain", "interval95", "family_interval95", "p_raw", "p_holm", "verdict",
                                  "markets_improved", "per_seed_gain", "positive_seeds", "family_slots")}
    board = {p: {r["model"]: dict(loss=r["loss"], rank=r["rank"], seed_range=r["seed_range"])
                 for r in result["leaderboards"]["primary"][p]} for p in ("r2", "rk")}
    raw = {p: {r["model"]: r["loss"] for r in result["leaderboards"]["raw"][p]} for p in ("r2", "rk")}
    return dict(contrasts=main, primary_leaderboard=board, raw_losses=raw,
                calibration=result["calibration"].get(model),
                comparison_supported_both_proxies=result["comparison_support"].get(model))


def selftest(resamples):
    result, _ = run("selftest", [1], resamples, (60, 20, 120))
    frozen = json.loads((FRONTIER / "analysis/results.json").read_text())
    worst = 0.0
    checked = 0
    for c in result["contrasts"]:
        ref = next(f for f in frozen["contrasts"] if f["model"] == "garch_t" and f["proxy"] == c["proxy"]
                   and f["block"] == c["block"])
        for k in ("gain", "p_raw"):
            worst = max(worst, abs(c[k] - ref[k])); checked += 1
        for k in ("interval95",):
            worst = max(worst, max(abs(a - b) for a, b in zip(c[k], ref[k]))); checked += 2
    lb = next(r for r in result["leaderboards"]["primary"]["r2"] if r["model"] == "gpvol")
    ref_lb = next(r for r in frozen["leaderboards"]["primary"]["r2"] if r["model"] == "garch_t")
    worst = max(worst, abs(lb["loss"] - ref_lb["loss"]))
    ok = worst < 1e-10
    return dict(passed=ok, max_abs_difference=worst, values_checked=checked + 1, resamples=resamples,
                note="garch_t relabelled as gpvol reproduces frozen gains, raw p-values, pointwise intervals and loss")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("addendum", "family34", "selftest"))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--model", default="gpvol")
    p.add_argument("--resamples", type=int, default=20000)
    a = p.parse_args()
    if a.mode == "selftest":
        res = selftest(a.resamples)
        print(json.dumps(res, indent=2))
        (OUT / "analysis" / "logmoments_v1").mkdir(parents=True, exist_ok=True)
        (OUT / "analysis" / "logmoments_v1" / "selftest.json").write_text(json.dumps(res, indent=2))
        return
    result, archive = run(a.mode, a.seeds, a.resamples, (60, 20, 120), model=a.model)
    summary = summarise(result, a.model)
    summary.update(mode=a.mode, model=a.model, seeds=a.seeds, created_utc=datetime.now(timezone.utc).isoformat(),
                   family_slots=result["settings"]["primary_family_slots"],
                   sources=dict(analyse_addendum_sha256=sha(Path(__file__)),
                                frozen_bundle_sha256=sha(FRONTIER / "bundle.json"),
                                repaired_statistics_sha256=sha(HERE / "statistics_logmoments_v1.py"),
                                numerical_amendment="frontier_gpvol_gpu_20260926_v1/NUMERICAL_STORAGE_AMENDMENT_v1.md"))
    dest = OUT / "analysis" / "logmoments_v1"
    dest.mkdir(parents=True, exist_ok=True)
    tag = f"{a.mode}_{a.model}_seeds{'-'.join(map(str, a.seeds))}"
    for suffix in ("summary.json", "results.json", "bootstrap_archive.npz"):
        if (dest / f"{tag}_{suffix}").exists():
            raise FileExistsError("Preserve previous analysis: " + str(dest / f"{tag}_{suffix}"))
    (dest / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (dest / f"{tag}_results.json").write_text(json.dumps(result, default=float))
    np.savez_compressed(dest / f"{tag}_bootstrap_archive.npz", **archive)
    print(json.dumps(summary["contrasts"].get(60), indent=2, default=float))


if __name__ == "__main__":
    main()
