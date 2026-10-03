"""Outcome-blind local readiness inventory; never fetch data or run a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments/outputs/programme_closure_20260919_v1"
ORIGIN_FIELDS = ("market", "date", "index", "phase", "target", "context_count", "context_start", "context_end")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_origins(rows, origins):
    require(len(rows) == len(origins), "Forecast origin count mismatch")
    require(all(all(row[key] == original[key] for key in ORIGIN_FIELDS)
                for row, original in zip(rows, origins)), "Forecast origin fields/order mismatch")


def inventory(root=ROOT):
    root = Path(root).resolve()
    old = root / "experiments/outputs/matched_prior_five_pairs_v1"
    exposure = read(old / "exposure_inventory.json")
    prior_manifest = read(old / "manifest.json")
    require(sha(old / "manifest.json") == (old / "manifest.sha256").read_text().strip(), "Old design manifest drift")
    require(sha(old / "exposure_inventory.json") == prior_manifest["fixed_artifacts"]["exposure_inventory.json"], "Exposure inventory drift")
    local_data = []
    for record in exposure["data"]:
        path = root / record["path"]
        require(sha(path) == record["sha256"], "Historical data array changed")
        with np.load(path, allow_pickle=False) as data:
            dates = data["date"].astype("U10")
        local_data.append(dict(market=record["market"], path=str(path), sha256=sha(path),
                               first_date=str(min(dates)), last_date=str(max(dates)),
                               later_primary_date_rows=int(np.sum((dates >= "2022-03-01") & (dates <= "2026-08-31")))))
    frontier = root / "experiments/outputs/frontier_overnight_increment_201504_201606_v1"
    spec = read(frontier / "study.json")
    require(sha(frontier / "study.json") == (frontier / "study.sha256").read_text().strip(), "Foundation study changed")
    bundle = read(frontier / "bundle/manifest.json")
    require(sha(frontier / "bundle/manifest.json") == spec["bundle_manifest_sha256"], "Foundation manifest changed")
    require(sha(frontier / "bundle/inputs.npz") == bundle["files"]["inputs.npz"], "Foundation causal contexts changed")
    frozen_outputs = read(frontier / "analysis_v1/hashes.json")
    for name, digest in frozen_outputs.items():
        require(sha(frontier / "analysis_v1" / name) == digest, "Foundation analysis artifact changed")
    forecasts = {}
    for model in ("volpfn", "tabpfn_ts", "chronos2", "moirai2"):
        cell = frontier / "models" / model
        status = read(cell / "status.json")
        digest = sha(cell / "predictions.json")
        require(status["status"] == "complete" and status["completed"] == 8777, "Incomplete foundation cell")
        require(digest == status["forecast_sha256"], "Foundation forecast hash changed")
        rows = read(cell / "predictions.json")
        validate_origins(rows, bundle["origins"])
        forecasts[model] = dict(path=str(cell / "predictions.json"), sha256=digest, completed=8777,
                                evaluation_rows=sum(r["phase"] == "evaluation" for r in rows),
                                frozen_scale=spec["scales"][model], exact_origins_verified=True,
                                reuse_role="historical reference checkpoint only" if model == "volpfn" else "immutable competitor cache reusable on this exact mask")
    contact_symbols = {"SPX": "^GSPC", "FTSE": "^FTSE", "GDAXI": "^GDAXI", "N225": "^N225"}
    mappings = [dict(market=m, verified_for_later_primary=False,
                     previously_contacted_yahoo_symbol=contact_symbols.get(m),
                     available_evidence="earlier whole-history audit contact only; not a completed identity/price-convention lock" if m in contact_symbols else "no later vendor mapping evidence in the frozen inventory")
                for m in prior_manifest["candidate_population"]]
    lock_candidates = []
    for base in (root / "experiments/data", root / "experiments/outputs"):
        for pattern in ("*data_lock*.json", "*primary_lock*.json", "*vendor_mapping*.json"):
            lock_candidates.extend(str(p) for p in base.rglob(pattern) if "source" not in p.parts)
    files = [old / "manifest.json", old / "exposure_inventory.json", frontier / "study.json",
             frontier / "bundle/manifest.json", frontier / "analysis_v1/hashes.json"]
    files += [root / p for p in ("experiments/matched_prior_confirmatory/PROTOCOL.md",
        "experiments/matched_prior_confirmatory/data_gate.py", "experiments/matched_prior_confirmatory/statistics.py",
        "experiments/matched_prior_confirmatory/study.py", "experiments/seed_replication/causal.py",
        "experiments/seed_replication/evaluation.py", "experiments/population_transfer/data.py",
        "experiments/frontier_staged/local.py", "experiments/frontier_benchmark/run.py")]
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), kind="local_evaluation_readiness_not_market_analysis",
                model_inference_performed=False, acquisition_performed=False, new_scores_computed=False,
                scope="Retained 28 historical arrays, frozen matched-prior inventory, local data/output lock filenames and completed foundation benchmark; not a global absence or freshness certificate.",
                later_primary=dict(status="not_ready_no_qualified_data_lock", start="2022-03-01", end="2026-08-31",
                    candidate_population=prior_manifest["candidate_population"], mapping_decisions=mappings,
                    independently_verified_later_mappings=0, qualified_later_markets=0,
                    qualified_count_note="Zero established by currently assembled evidence; not a claim that all vendors fail or fewer than20 could eventually qualify.",
                    local_historical_arrays=local_data, current_local_primary_rows=sum(r["later_primary_date_rows"] for r in local_data),
                    matching_local_lock_filenames=sorted(set(lock_candidates)), data_gate=prior_manifest["analysis"]["data"],
                    known_prior_contact=exposure["later_block"]["known_prior_contact"],
                    freshness="Unestablished. New downloads and later dates do not make observations untouched.",
                    acquisition_route="Create a distinct index snapshot adapter after sealing exact index identities; reuse population_transfer.data snapshot/provenance patterns, not its stock/FX population, dates, target or QC.",
                    no_automatic_fallback=True),
                historical_secondary=exposure["historical_available_block"],
                foundation_reuse=dict(status="verified_reusable_on_identical_historical_origins", forecasts=forecasts,
                    full_evaluation_rows=8774, bridge_rows=3, exact_context_shape=[8777, 1024],
                    context_archive=str(frontier / "bundle/inputs.npz"), context_sha256=bundle["files"]["inputs.npz"],
                    original_analysis_hashes=frozen_outputs,
                    comparison_policy="Retain original results; separately lock repaired CUDA VolPFN checkpoint/seed handling and score conversion before generating its forecasts. Reuse competitor predictions unchanged; no competitor rerun is needed.",
                    context_warning="Foundation confirmation uses each origin's exact preceding1024 observations and frontier native_prediction. Matched-prior primary uses predict_series L1024/stride512 with window ownership; these are distinct inference schedules.",
                    scale_warning="Competitor scales are frozen. Explicitly freeze whether the repaired VolPFN retains the original multiplier or uses a declared development-only recalibration. Never fit calibration on this exposed evaluation block.",
                    evidence_scope="Exposed historical procedure-sensitivity confirmation; not a fresh primary, not an update of the original seed8101 claim."),
                pending_tasks=[
                    "Freeze a distinct repaired-CUDA matched-training/analysis amendment and complete every broad/Matérn seed; do not reuse incomplete or scientifically incompatible old finals.",
                    "Verify all28 exact index/vendor identities and price/session conventions, retaining unresolved identities; seal mapping decisions before downloading each series.",
                    "Acquire immutable raw OHLC+metadata with at least100 valid preblock context observations; record every exclusion and source/date convention.",
                    "Apply the existing fixed outcome-blind data gate; require20–28 eligible markets and at least800 primary targets each, then hash fullcontext/returns, raw/meta/QC, population and exact shared mask before inference.",
                    "Seal ten final-checkpoint provenance and matched initial-state evidence; run the dedicated matched-prior evaluator only after both checkpoint and data gates pass.",
                    "Form five matched daily common-t8 gains and call the existing two-slot statistics engine; retain seed-level native-density/QLIKE descriptions and raw bootstrap evidence, with optional GARCH unrun slot p=1.",
                    "If needed, separately confirm repaired VolPFN on the existing frontier8774-row block using archived competitor forecasts and newly frozen repaired-model analysis rules; no historical promotion if later primary unavailable."],
                code_sources={str(p): sha(p) for p in files}, inventory_source_sha256=sha(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = inventory(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "EVALUATION_READINESS.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(destination), later_primary=result["later_primary"]["status"],
                          cached_forecasts=result["foundation_reuse"]["status"])))
