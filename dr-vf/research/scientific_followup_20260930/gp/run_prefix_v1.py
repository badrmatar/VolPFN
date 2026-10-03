"""CPU-only, bounded 200-particle GSPTSE seed-2 replay from its original burn-in.

This records an implementation diagnostic, never a completed addendum cell.
The cap of 12 filter updates prevents accidental long scientific execution.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.gpvol_full import gpvol_rapcf as original
from research.gpvol_full import run_frontier_gpvol as runner
from research.scientific_followup_20260930.gp import likelihood_v1 as repair
from research.scientific_followup_20260930.gp.test_likelihood_v1 import oracle


def numeric(value):
    value = float(value)
    return value if np.isfinite(value) else ("-inf" if value < 0 else "inf" if value > 0 else "nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6)
    a = p.parse_args()
    if not 1 <= a.steps <= 12:
        raise ValueError("This diagnostic is limited to 1–12 updates")
    a.out.mkdir(parents=True, exist_ok=False)
    settings = runner.settings_from(SimpleNamespace(seed=2, n_particles=200, burn_in=100,
        lam=.95, standardise_window=1000, code_indexing=False, chunk=32))
    bundle, rows = runner.load_plan()
    segment = next(s for s in runner.prepare_segments("GSPTSE", rows["GSPTSE"],
        bundle["market_paths"]["GSPTSE"], settings) if s["phase"] == "evaluation")
    fixed = repair.load_filter_module()
    kw = dict(n_particles=200, burn_in=100, lam=.95, seed=segment["seg_seed"],
              code_indexing=False, chunk=32)
    before = original.RAPCF(segment["x"], **kw)
    after = fixed.RAPCF(segment["x"], **kw)
    timings = []
    for _ in range(a.steps):
        start = time.perf_counter()
        after.step()
        timings.append(time.perf_counter() - start)
        before.step()
    n = slice(100, after.t)
    moment_difference = float(np.max(np.abs(after.out["h"][n] - before.out["h"][n])))
    relative_moment_difference = float(np.max(np.abs(after.out["h"][n] - before.out["h"][n])
        / np.maximum(np.abs(before.out["h"][n]), np.finfo(float).tiny)))
    density_difference = float(np.max(np.abs(after.out["logdens"][n] - before.out["logdens"][n])))
    # Actual 200-particle causal fixture: target and later x are perturbed.
    changed_x = segment["x"].copy()
    changed_x[100 + a.steps - 1:] += .5
    changed = fixed.RAPCF(changed_x, **kw)
    for _ in range(a.steps):
        changed.step()
    causal = bool(np.array_equal(after.out["h"][n], changed.out["h"][n]))
    after.save(a.out / "corrected_prefix_state.npz")
    cases = []
    for x, v in [(1e-100, -1000), (1e-200, -1000), (1.0, -710), (0.1, -1000),
                 (float(np.nextafter(0., 1.)), -1500)]:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            old, new = float(original.log_normal_obs(x, np.array(v))), float(repair.log_normal_obs(x, v))
        cases.append(dict(x=x, v=v, original=numeric(old), repaired=numeric(new),
                          decimal_oracle=str(oracle(x, v))))
    receipt = dict(scope="bounded CPU diagnostic only; not a completed forecast cell or primary ranking",
        provenance=repair.provenance(), settings=settings, market="GSPTSE", phase="evaluation",
        segment_seed=segment["seg_seed"], full_chain_length=len(segment["x"]),
        full_required_filter_updates=len(segment["x"]) - 100, completed_filter_updates=a.steps,
        start_index=segment["start"], standardization=dict(mean=segment["mu_hat"], sd=segment["sd_hat"]),
        repaired_step_seconds=timings, repaired_prefix_seconds=sum(timings),
        original_vs_repaired=dict(max_abs_moment_difference=moment_difference,
            max_relative_moment_difference=relative_moment_difference,
            max_abs_density_difference=density_difference,
            identical_rng_state=(after.rng.bit_generator.state == before.rng.bit_generator.state)),
        finite_density_prefix=bool(np.isfinite(after.out["logdens"][n]).all()),
        collapsed_prefix_steps=int(np.count_nonzero(after.out["weight_collapse"][n])),
        future_perturbation_forecast_equal=causal,
        mathematical_counterexamples=cases,
        source_bundle_sha256=runner.sha(runner.FRONTIER / "bundle.json"),
        source_market_sha256=runner.sha(runner.ROOT / bundle["market_paths"]["GSPTSE"]),
        original_file_preserved=(runner.sha(repair.ORIGINAL_PATH) == repair.provenance()["original_sha256"]))
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("completed_filter_updates", "repaired_prefix_seconds",
        "original_vs_repaired", "finite_density_prefix", "collapsed_prefix_steps",
        "future_perturbation_forecast_equal")}, indent=2))
    if not causal or not receipt["finite_density_prefix"]:
        raise AssertionError("Prefix diagnostic failed")


if __name__ == "__main__":
    main()
