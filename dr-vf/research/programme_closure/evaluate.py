"""Gated ten-final matched-prior evaluation; no acquisition or training.

Freeze this adapter before scoring. The data lock format is documented in
EVALUATION_DATA_LOCK.md. A missing/failed seed or unqualified data lock stops
before model inference; an evaluation failure is retained without row removal.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

# Configure cuBLAS before any direct or transitive torch import/initialization.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from research.matched_prior_confirmatory import data_gate as D, statistics as ST


ROOT = Path(__file__).resolve().parents[2]
LEGACY_DESIGN = ROOT / "research/outputs/programme_closure_20260919_v1/gpu_design_v1"
SEEDS = list(ST.PRIMARY_SEEDS)
ARMS = ["broad", "matern"]
KINDS = {"plan": "programme_closure_matched_evaluation_v1", "data": "programme_closure_index_data_lock_v1"}
CFG = {"d": 256, "layers": 6, "heads": 8, "K": 4}
# Only independently rederived float64 returns use this arithmetic tolerance.
# Locked file hashes, dates and the values supplied to inference remain exact.
RETURN_RECOMPUTE_RTOL = 1e-13
RETURN_RECOMPUTE_ATOL = 1e-12  # percentage points, since r = 100 * log(close/open)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sealed(path):
    path = Path(path)
    require(sha(path) == path.with_suffix(".sha256").read_text().strip(), "Unsealed or changed manifest: " + str(path))
    return read(path)


def local_file(base, relative, files):
    require(relative in files, "File not bound by manifest: " + str(relative))
    path = (Path(base) / relative).resolve()
    require(path.is_relative_to(Path(base).resolve()), "Artifact escapes locked directory")
    require(path.is_file() and sha(path) == files[relative], "Locked file changed: " + str(path))
    return path


def verify_design(frozen):
    frozen = Path(frozen).resolve()
    manifest = sealed(frozen / "manifest.json")
    require(manifest["seeds"] == SEEDS and manifest["arms"] == ARMS, "Unregistered training allocation")
    training = manifest["training"]
    require(training["steps"] == 20000 and training["tokens"] == 8192 and training["cfg"] == CFG,
            "Training budget or architecture differs from the declared primary")
    for relative, digest in manifest["source_hashes"].items():
        require(sha(ROOT / relative) == digest and sha(frozen / "source" / relative) == digest,
                "Training source drift: " + relative)
    return manifest


def freeze_plan(frozen, out, device="cpu"):
    design = verify_design(frozen)
    require(device in ("cpu", "cuda"), "Unsupported inference backend")
    old = ROOT / "research/outputs/matched_prior_five_pairs_v1/manifest.json"
    protocol = sealed(old)
    sources = [Path(__file__), Path(D.__file__), Path(ST.__file__),
               ROOT / "research/seed_replication/causal.py", ROOT / "volpfn.py",
               ROOT / "research/population_transfer/statistics.py"]
    plan = dict(kind=KINDS["plan"], frozen_utc=datetime.now(timezone.utc).isoformat(),
                training_design=str(Path(frozen).resolve()), training_manifest_sha256=sha(Path(frozen) / "manifest.json"),
                seeds=SEEDS, arms=ARMS, analysis=protocol["analysis"],
                candidate_population=protocol["candidate_population"],
                inference=dict(device=device, dtype="float32", tf32=False, deterministic=True,
                               context=1024, stride=512, minimum_history=100, batch_windows=4,
                               normalization_spec=design["training"]["normalization_spec"]),
                sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                scope="New corrected CUDA training procedure; fixed later primary, known prior data contact, no untouched-data claim. Historical foundation results remain separate.",
                historical_fallback=False, calibration="No market-fit scale correction in common-t8 primary; raw predictive second moment",
                missing_seed_policy="Primary unavailable; never score an average of surviving seeds")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "manifest.json", plan)
    (out / "manifest.sha256").write_text(sha(out / "manifest.json") + "\n")
    return plan


def verify_plan(plan_path):
    plan = sealed(plan_path)
    require(plan["kind"] == KINDS["plan"] and plan["seeds"] == SEEDS and plan["arms"] == ARMS,
            "Unregistered evaluation plan")
    for relative, digest in plan["sources"].items():
        require(sha(ROOT / relative) == digest, "Evaluation source drift: " + relative)
    require(sha(Path(plan["training_design"]) / "manifest.json") == plan["training_manifest_sha256"], "Training design changed")
    design = verify_design(plan["training_design"])
    old = sealed(ROOT / "research/outputs/matched_prior_five_pairs_v1/manifest.json")
    require(plan["analysis"] == old["analysis"] and plan["candidate_population"] == old["candidate_population"],
            "Fixed primary analysis or candidate population changed")
    settings = plan["inference"]
    require(settings["device"] in ("cpu", "cuda") and settings["dtype"] == "float32"
            and settings["tf32"] is False and settings["deterministic"] is True
            and settings["context"] == 1024 and settings["stride"] == 512
            and settings["minimum_history"] == 100 and settings["batch_windows"] == 4
            and settings["normalization_spec"] == design["training"]["normalization_spec"],
            "Frozen inference convention changed")
    return plan


def verify_data_lock(path, candidate_population):
    """Recheck raw OHLC gates and full locked histories, without a model.

    Only the arithmetic recomputation admits float64 platform rounding. Inference
    always consumes the hash-verified locked returns, never the recomputed array.
    """
    path = Path(path).resolve()
    manifest = sealed(path)
    require(manifest["kind"] == KINDS["data"], "Wrong data-lock kind")
    require(manifest["candidate_population"] == list(candidate_population), "Candidate indices changed")
    require(manifest["data_gate"] == D.specification(), "Data/QC rules changed")
    require(manifest["locked_before_model_scoring"] is True and manifest["exposure_disclosure"], "Missing chronology/exposure disclosure")
    files, base = manifest["files"], path.parent
    for relative in files:
        local_file(base, relative, files)
    mapping = read(local_file(base, manifest["mapping_file"], files))
    require(mapping["candidate_population"] == list(candidate_population), "Mapping population changed")
    entries = mapping["markets"]
    require(len(entries) == 28 and {e["market"] for e in entries} == set(candidate_population), "Every candidate needs a mapping decision")
    decisions = {e["market"]: e for e in entries}
    reports = read(local_file(base, manifest["qc_file"], files))
    require(set(reports) == set(candidate_population), "Every candidate needs retained QC/unavailability evidence")
    require(set(manifest["market_files"]).issubset(set(candidate_population)), "Unknown market in raw data")
    series, eligible = {}, []
    for market in candidate_population:
        decision = decisions[market]
        if market not in manifest["market_files"]:
            require(reports[market].get("eligible") is False and reports[market].get("failure_codes"), "Missing series lacks an explicit exclusion")
            continue
        entry = manifest["market_files"][market]
        verified = decision.get("identity_verified") is True
        if verified:
            require(all(decision.get(k) for k in ("vendor_symbol", "exact_index_name", "identity_evidence", "price_basis", "session_date_convention")),
                    "Incomplete index identity/convention evidence")
        metadata = read(local_file(base, entry["metadata"], files))
        local_file(base, entry["vendor_raw"], files)
        require(metadata["vendor_symbol"] == decision["vendor_symbol"], "Raw source symbol does not match mapping")
        require(datetime.fromisoformat(mapping["locked_utc"]) <= datetime.fromisoformat(metadata["retrieved_utc"]),
                "Mapping was not sealed before this acquisition")
        frame = pd.read_csv(local_file(base, entry["ohlc"], files))
        report, prepared = D.check_market(frame, market, identity_verified=verified)
        require(report == reports[market], "Recorded QC differs from recomputed raw OHLC gate: " + market)
        if not report["eligible"]:
            continue
        eligible.append(market)
        with np.load(local_file(base, entry["returns"], files), allow_pickle=False) as raw:
            dates = raw["date"].astype("U10")
            returns = raw["r"].astype(float)
        require(np.array_equal(dates, prepared.date.to_numpy(dtype="U10")), "Full valid causal context dates changed")
        rederived = prepared.r.to_numpy(float)
        require(returns.shape == rederived.shape and np.isfinite(returns).all(), "Invalid return history")
        require(np.allclose(returns, rederived, rtol=RETURN_RECOMPUTE_RTOL, atol=RETURN_RECOMPUTE_ATOL,
                            equal_nan=False), "Returns differ from locked open-to-close construction")
        series[market] = dict(date=dates, r=returns)
    require(20 <= len(eligible) <= 28, "Later primary unavailable: fewer than20 eligible exact indices; no historical fallback")
    require(eligible == manifest["eligible_population"], "Eligible population changed")
    mask = pd.read_csv(local_file(base, manifest["scoring_mask"], files), dtype={"market": str, "date": str})
    require(set(mask.columns) == {"market", "date"} and not mask.duplicated(["market", "date"]).any(), "Invalid or duplicate scoring mask")
    expected = {(m, str(date)) for m in eligible for date in series[m]["date"] if D.START <= date <= D.END}
    mask = mask[["market", "date"]]
    require(set(mask.itertuples(index=False, name=None)) == expected, "Mask drops/adds eligible primary rows")
    require(len(mask) == len(expected), "Duplicate or unavailable mask origin")
    return dict(manifest=manifest, manifest_sha256=sha(path), series=series,
                mask=mask.sort_values(["market", "date"]).reset_index(drop=True), reports=reports)


def _load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy() if hasattr(value, "detach") else np.asarray(value)
        require(np.isfinite(array).all(), "Nonfinite checkpoint state")
        digest.update(key.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def verify_checkpoints(frozen, training_root, load_checkpoint=_load_checkpoint):
    """All completion/status/hash gates pass before any weight file is loaded."""
    manifest = verify_design(frozen)
    training_root = Path(training_root).resolve()
    require(read(training_root / "status.json")["status"] == "complete", "Ten-run training battery is incomplete")
    manifest_hash = sha(Path(frozen) / "manifest.json")
    reused = manifest.get("reused_checkpoints", [])
    require(isinstance(reused, list), "Invalid checkpoint reuse whitelist")
    whitelist = {}
    if reused:
        require(len(reused) == 2 and {(e["arm"], e["seed"]) for e in reused}
                == {("matern", 8101), ("matern", 8102)},
                "Reuse whitelist permits only the two completed Matérn finals")
        original = sealed(LEGACY_DESIGN / "manifest.json")
        original_hash = sha(LEGACY_DESIGN / "manifest.json")
        require(original_hash != manifest_hash and original["training"] == manifest["training"]
                and original["seeds"] == SEEDS and original["arms"] == ARMS,
                "Reused checkpoint training settings changed")
        for relative, digest in original["source_hashes"].items():
            require(sha(LEGACY_DESIGN / "source" / relative) == digest,
                    "Archived training source changed: " + relative)
        hash_fields = ("source_manifest_sha256", "status_sha256", "checkpoint_sha256",
                       "initial_state_sha256", "initial_file_sha256")
        for item in reused:
            require(all(isinstance(item.get(k), str) and len(item[k]) == 64
                        and all(c in "0123456789abcdef" for c in item[k]) for k in hash_fields),
                    "Incomplete checkpoint reuse hashes")
            require(item["source_manifest_sha256"] == original_hash,
                    "Reuse source manifest mismatch")
            whitelist[(item["arm"], item["seed"])] = item
    entries = []
    for seed in SEEDS:
        for arm in ARMS:
            directory = training_root / "runs" / (arm + "_" + str(seed))
            status = read(directory / "status.json")
            checkpoint = directory / "final.pt"
            reuse = whitelist.get((arm, seed))
            source_manifest = reuse["source_manifest_sha256"] if reuse else manifest_hash
            require(status["status"] == "complete" and status["completed_updates"] == 20000
                    and status["input_positions"] == 163840000, "Incomplete fixed-budget final: " + str(directory))
            require(status["seed"] == seed and status["arm"] == arm and status["manifest_sha256"] == source_manifest,
                    "Training provenance mismatch")
            require(status["causal_probe"]["status"] == "passed", "Checkpoint causal probe failed")
            require(sha(checkpoint) == status["checkpoint_sha256"], "Checkpoint checksum mismatch")
            status_hash, checkpoint_hash = sha(directory / "status.json"), sha(checkpoint)
            initial_file_hash = sha(directory / "initial.pt")
            if reuse:
                require(status_hash == reuse["status_sha256"]
                        and checkpoint_hash == reuse["checkpoint_sha256"]
                        and initial_file_hash == reuse["initial_file_sha256"]
                        and status["initial_state_sha256"] == reuse["initial_state_sha256"],
                        "Reused checkpoint immutable whitelist mismatch")
            entries.append(dict(seed=seed, arm=arm, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                status_sha256=status_hash, initial_state_sha256=status["initial_state_sha256"],
                                initial_file_sha256=initial_file_hash, source_manifest_sha256=source_manifest,
                                provenance="immutable_reused_matern_final" if reuse else "current_design_training"))
    for entry in entries:
        checkpoint = load_checkpoint(entry["checkpoint"])
        require(checkpoint["step"] == 20000 and checkpoint["seed"] == entry["seed"]
                and checkpoint["cfg"] == CFG and checkpoint["manifest_sha256"] == entry["source_manifest_sha256"],
                "Final checkpoint metadata mismatch")
        require(checkpoint["training_spec"] == dict(manifest["training"], arm=entry["arm"]), "Matched training specification changed")
        require(checkpoint["causal_spec"] == "running_nonzero_rms_window_v1"
                and checkpoint["normalization_spec"] == manifest["training"]["normalization_spec"], "Normalization changed")
        _state_hash(checkpoint["state_dict"])
        initial = load_checkpoint(Path(entry["checkpoint"]).with_name("initial.pt"))
        require(_state_hash(initial["state_dict"]) == initial["sha256"] == entry["initial_state_sha256"], "Initial state evidence mismatch")
    for seed in SEEDS:
        require(len({e["initial_state_sha256"] for e in entries if e["seed"] == seed}) == 1,
                "Initial weights differ within seed pair")
    return dict(training_manifest_sha256=manifest_hash, entries=entries,
                training_runtime=read(training_root / "runtime.json"),
                training_runtime_sha256=sha(training_root / "runtime.json"),
                training_status_sha256=sha(training_root / "status.json"))


def paired_scores(frames, mask):
    """Retain every seed and raw descriptive metric; never average densities."""
    from research.population_transfer.statistics import common_t8_log_score
    expected = set(mask[["market", "date"]].itertuples(index=False, name=None))
    require(set(frames) == {(arm, seed) for arm in ARMS for seed in SEEDS}, "Missing checkpoint predictions")
    descriptive, gains = {}, []
    normalized = {}
    for key, frame in frames.items():
        require(set(frame[["market", "date"]].itertuples(index=False, name=None)) == expected
                and len(frame) == len(expected) and not frame.duplicated(["market", "date"]).any(), "Forecast mask mismatch")
        frame = frame.sort_values(["market", "date"]).reset_index(drop=True).copy()
        require(np.isfinite(frame[["target", "h", "mean", "native_logscore"]]).all().all() and (frame.h > 0).all(), "Nonfinite/nonpositive forecast")
        frame["common_t8_logscore"] = common_t8_log_score(frame.target.to_numpy(), frame.h.to_numpy())
        frame["qlike"] = frame.target ** 2 / frame.h + np.log(frame.h)
        frame["native_nll"] = -frame.native_logscore
        metrics = ["common_t8_logscore", "qlike", "native_nll"]
        per = frame.groupby("market")[metrics].mean()
        descriptive[key[0] + "_" + str(key[1])] = dict(mean=per.mean().to_dict(), per_market=per.to_dict(orient="index"), rows=len(frame))
        normalized[key] = frame
    reference_targets = normalized[("broad", SEEDS[0])].target.to_numpy()
    require(all(np.array_equal(frame.target.to_numpy(), reference_targets) for frame in normalized.values()),
            "Paired targets differ across arms or seeds")
    for seed in SEEDS:
        broad, matern = normalized[("broad", seed)], normalized[("matern", seed)]
        require(np.array_equal(broad.target.to_numpy(), matern.target.to_numpy()), "Paired targets differ")
        result = broad[["market", "date"]].copy()
        result["seed"] = seed
        result["gain"] = broad.common_t8_logscore - matern.common_t8_logscore
        gains.append(result)
    return pd.concat(gains, ignore_index=True), descriptive, normalized


def run(plan_path, training_root, data_lock, out):
    plan = verify_plan(plan_path)
    data = verify_data_lock(data_lock, plan["candidate_population"])
    checkpoints = verify_checkpoints(plan["training_design"], training_root)
    # Every input is ready before this directory exists or any model is loaded.
    out = Path(out).resolve()
    protected = [Path(training_root).resolve(), Path(data_lock).resolve().parent, Path(plan_path).resolve().parent]
    require(not any(out == path or out in path.parents or path in out.parents for path in protected), "Evaluation output overlaps a protected input directory")
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from research.seed_replication import causal
    settings = plan["inference"]
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if settings["device"] == "cuda":
            require(torch.cuda.is_available(), "Declared CUDA inference unavailable; no fallback")
        save(out / "CHECKPOINT_REGISTRY.json", checkpoints)
        save(out / "INPUTS.json", dict(evaluation_plan_sha256=sha(plan_path), data_lock_sha256=data["manifest_sha256"],
             training_manifest_sha256=checkpoints["training_manifest_sha256"], data_lock=str(Path(data_lock).resolve()),
             plan=str(Path(plan_path).resolve())))
        frames = {}
        for entry in checkpoints["entries"]:
            label = entry["arm"] + "_" + str(entry["seed"])
            destination = out / "forecasts" / label
            destination.mkdir(parents=True)
            model = causal.load_model(entry["checkpoint"], settings["device"])
            cells = []
            for market, history in data["series"].items():
                prediction = causal.predict_series(model, history["r"], L=1024, stride=512,
                                                   min_history=100, batch_windows=4, levels=())
                indices = np.flatnonzero((history["date"] >= D.START) & (history["date"] <= D.END))
                require(np.all(indices >= 100), "Primary target is inside causal warmup")
                arrays = {"date": history["date"], "r": history["r"],
                          **{key: np.asarray(prediction[key]) for key in ("mean", "h", "logscore")}}
                require(all(arrays[key].shape == history["r"].shape for key in ("mean", "h", "logscore")), "Prediction length differs from history")
                np.savez_compressed(destination / (market + ".npz"), **arrays)
                cells.append(pd.DataFrame(dict(market=market, date=history["date"][indices],
                    target=history["r"][indices], mean=arrays["mean"][indices], h=arrays["h"][indices],
                    native_logscore=arrays["logscore"][indices])))
            frames[(entry["arm"], entry["seed"])] = pd.concat(cells, ignore_index=True)
            save(out / "status.json", dict(status="forecasting", completed_checkpoints=len(frames), total_checkpoints=10))
            del model
            if settings["device"] == "cuda":
                torch.cuda.empty_cache()
        gains, descriptive, scored = paired_scores(frames, data["mask"])
        result = ST.analyze_study(gains, expected_rows=data["mask"], secondary=None)
        f, mask, seeds = ST._validate(gains, data["mask"], ST.PRIMARY_SEEDS)
        calendar = pd.DatetimeIndex(sorted(mask.date.unique()))
        counts = ST._calendar_counts(len(calendar), ST.DEFAULT_REPS, ST.DEFAULT_BLOCK, ST.DEFAULT_BOOTSTRAP_SEED)
        summary, draws = ST._summarize(f, mask, seeds, calendar, counts, .003, .05, .005)
        require(np.isclose(summary["mean_gain"], result["contrasts"][ST.PRIMARY]["mean_gain"], rtol=1e-12, atol=1e-12), "Stored bootstrap evidence disagrees")
        np.savez_compressed(out / "bootstrap_evidence.npz", counts=counts.astype(np.int16), draws=draws,
                            calendar=calendar.strftime("%Y-%m-%d").to_numpy(dtype="U10"))
        gains.to_csv(out / "paired_gains.csv", index=False)
        for (arm, seed), frame in scored.items():
            frame.to_csv(out / (arm + "_" + str(seed) + "_scored.csv.gz"), index=False)
        result.update(status="complete", descriptive_by_arm_and_seed=descriptive,
                      evaluation_plan_sha256=sha(plan_path), data_lock_sha256=data["manifest_sha256"],
                      exposure=data["manifest"]["exposure_disclosure"], historical_fallback_used=False,
                      checkpoint_registry=checkpoints, limitations=plan["scope"])
        verify_plan(plan_path)
        verify_data_lock(data_lock, plan["candidate_population"])
        for entry in checkpoints["entries"]:
            require(sha(entry["checkpoint"]) == entry["checkpoint_sha256"], "Checkpoint changed during evaluation")
        save(out / "results.json", result)
        primary = result["contrasts"][ST.PRIMARY]
        lines = ["# Corrected CUDA matched-prior primary", "",
                 "Broad-prior versus specified Matérn-prior package, five fixed seed pairs; all ten finals retained.", "",
                 f"Mean common-t8 log-score gain: {primary['mean_gain']:.8g} nats. Basic 95% interval: {primary['interval95']}. Holm superiority p: {primary['p_superiority_holm']:.8g}.", "",
                 f"Material gain gate: {primary['point_materiality_bar_met']}; positive interval: {primary['positive_interval95']}; breadth gate: {primary['market_coverage_bar_met']}; combined gate: {primary['decision_bar_met']}.",
                 f"Equivalence within ±0.003 nats: {result['equivalence']['equivalence_supported']} (separate endpoint; no joint familywise guarantee).", "",
                 result["scope"], "", str(result["exposure"]), "",
                 "This does not isolate feedback or volatility structure individually. Historical foundation benchmarks are separate; this report does not replace them.", "",
                 "| Arm/seed | Native NLL | QLIKE | Common-t8 log score |", "|---|---:|---:|---:|"]
        for label, record in descriptive.items():
            values = record["mean"]
            lines.append(f"| {label} | {values['native_nll']:.8g} | {values['qlike']:.8g} | {values['common_t8_logscore']:.8g} |")
        (out / "READOUT.md").write_text("\n".join(lines) + "\n")
        save(out / "status.json", dict(status="complete", rows_per_seed=len(data["mask"]), checkpoints=10))
        save(out / "hashes.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file() and path.name != "hashes.json"})
        return result
    except BaseException as exc:
        save(out / "status.json", dict(status="failed", error=repr(exc), no_survivor_analysis=True))
        (out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "check", "run"])
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--data-lock", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze_plan(args.frozen, args.out, args.device)
    elif args.action == "check":
        plan = verify_plan(args.plan)
        data = verify_data_lock(args.data_lock, plan["candidate_population"])
        registry = verify_checkpoints(plan["training_design"], args.training)
        result = dict(status="ready", rows_per_seed=len(data["mask"]), checkpoints=len(registry["entries"]))
    else:
        result = run(args.plan, args.training, args.data_lock, args.out)
    print(json.dumps({key: result[key] for key in ("kind", "status", "rows_per_seed", "checkpoints") if key in result}))
