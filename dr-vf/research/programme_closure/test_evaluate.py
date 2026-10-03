"""Synthetic artifacts only: no market data, checkpoints, models or downloads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from research.programme_closure import evaluate as E
from research.programme_closure.evaluation_inventory import validate_origins


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal(path, value):
    write(path, value)
    path.with_suffix(".sha256").write_text(E.sha(path) + "\n")


def data_fixture(base):
    candidates = ["market_%02d" % i for i in range(28)]
    dates = pd.bdate_range("2021-01-01", "2026-08-31")
    day = np.arange(len(dates), dtype=float)
    opening = 100 + day * .05
    closing = opening * (1 + .005 * np.sin(day + .4))
    frame = pd.DataFrame(dict(date=dates.strftime("%Y-%m-%d"), open=opening,
                              high=np.maximum(opening, closing) + 1,
                              low=np.minimum(opening, closing) - 1, close=closing))
    files, mappings, reports, market_files, mask = {}, [], {}, {}, []
    base.mkdir()
    for i, market in enumerate(candidates):
        mappings.append(dict(market=market, identity_verified=i < 20, vendor_symbol="TEST_" + market,
                             exact_index_name="Synthetic unit fixture " + market, identity_evidence=["synthetic test fixture"],
                             price_basis="unadjusted", session_date_convention="local session label"))
        if i >= 20:
            reports[market] = dict(eligible=False, failure_codes=["unresolved_test_identity"])
            continue
        ohlc = market + ".csv"
        frame.to_csv(base / ohlc, index=False)
        # Re-read the saved representation, just as the production gate does.
        report, selected = E.D.check_market(pd.read_csv(base / ohlc), market, identity_verified=True)
        assert report["eligible"], report
        reports[market] = report
        metadata = market + ".metadata.json"
        write(base / metadata, dict(vendor_symbol="TEST_" + market, retrieved_utc="2026-09-19T01:00:00+00:00"))
        returns = market + ".npz"
        np.savez_compressed(base / returns, date=selected.date.to_numpy(dtype="U10"), r=selected.r.to_numpy(float))
        market_files[market] = dict(vendor_raw=ohlc, ohlc=ohlc, metadata=metadata, returns=returns)
        mask.extend((market, date) for date in selected.loc[selected.is_primary, "date"])
    write(base / "mapping.json", dict(candidate_population=candidates, locked_utc="2026-09-19T00:00:00+00:00", markets=mappings))
    write(base / "qc.json", reports)
    pd.DataFrame(mask, columns=["market", "date"]).to_csv(base / "mask.csv", index=False)
    files = {p.name: E.sha(p) for p in base.iterdir() if p.is_file()}
    lock = dict(kind=E.KINDS["data"], candidate_population=candidates, data_gate=E.D.specification(),
                locked_before_model_scoring=True, exposure_disclosure="Synthetic fixture; never acquired data.",
                files=files, mapping_file="mapping.json", qc_file="qc.json", market_files=market_files,
                eligible_population=candidates[:20], scoring_mask="mask.csv")
    seal(base / "manifest.json", lock)
    return candidates, lock


def checkpoint_fixture(base):
    frozen, battery = base / "frozen", base / "battery"
    training = dict(steps=20000, tokens=8192, cfg=E.CFG, normalization_spec={"id": "test_normalization"})
    manifest = dict(training=training, arms=E.ARMS, seeds=E.SEEDS, source_hashes={})
    seal(frozen / "manifest.json", manifest)
    digest = E.sha(frozen / "manifest.json")
    write(battery / "status.json", dict(status="complete"))
    write(battery / "runtime.json", dict(device="synthetic"))
    for seed in E.SEEDS:
        for arm in E.ARMS:
            directory = battery / "runs" / (arm + "_" + str(seed))
            state = {"weight": [float(seed), 1.]}
            state_hash = E._state_hash(state)
            checkpoint = dict(step=20000, seed=seed, cfg=E.CFG, manifest_sha256=digest,
                              training_spec=dict(training, arm=arm), causal_spec="running_nonzero_rms_window_v1",
                              normalization_spec=training["normalization_spec"], state_dict=state)
            write(directory / "final.pt", checkpoint)
            write(directory / "initial.pt", dict(state_dict=state, sha256=state_hash))
            write(directory / "status.json", dict(status="complete", completed_updates=20000,
                input_positions=163840000, seed=seed, arm=arm, manifest_sha256=digest,
                checkpoint_sha256=E.sha(directory / "final.pt"), initial_state_sha256=state_hash,
                causal_probe=dict(status="passed")))
    return frozen, battery, manifest


class DataGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "data"
        self.candidates, self.lock = data_fixture(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def update(self, name, value):
        write(self.base / name, value)
        self.lock["files"][name] = E.sha(self.base / name)
        seal(self.base / "manifest.json", self.lock)

    def test_complete_twenty_market_gate_retains_context(self):
        result = E.verify_data_lock(self.base / "manifest.json", self.candidates)
        self.assertEqual(len(result["series"]), 20)
        self.assertGreater(len(result["series"][self.candidates[0]]["r"]), len(result["mask"]) / 20 + 100)

    def change_locked_returns(self, transform, reseal=True):
        name = self.lock["market_files"][self.candidates[0]]["returns"]
        with np.load(self.base / name, allow_pickle=False) as data:
            dates, returns = data["date"], data["r"]
        changed = transform(returns.copy())
        self.assertFalse(np.array_equal(returns, changed))
        np.savez_compressed(self.base / name, date=dates, r=changed)
        if reseal:
            self.lock["files"][name] = E.sha(self.base / name)
            seal(self.base / "manifest.json", self.lock)
        return changed

    def test_return_recomputation_allows_ulp_but_preserves_locked_values(self):
        locked = self.change_locked_returns(lambda r: np.nextafter(r, np.inf))
        result = E.verify_data_lock(self.base / "manifest.json", self.candidates)
        np.testing.assert_array_equal(result["series"][self.candidates[0]]["r"], locked)

    def test_return_recomputation_rejects_substantive_drift(self):
        self.change_locked_returns(lambda r: r + 1e-6)
        with self.assertRaisesRegex(ValueError, "Returns differ"):
            E.verify_data_lock(self.base / "manifest.json", self.candidates)

    def test_unsealed_one_ulp_return_change_still_fails_hash(self):
        self.change_locked_returns(lambda r: np.nextafter(r, np.inf), reseal=False)
        with self.assertRaisesRegex(ValueError, "Locked file changed"):
            E.verify_data_lock(self.base / "manifest.json", self.candidates)

    def test_outcome_mask_removal_rejected(self):
        mask = pd.read_csv(self.base / "mask.csv")
        mask.iloc[:-1].to_csv(self.base / "mask.csv", index=False)
        self.lock["files"]["mask.csv"] = E.sha(self.base / "mask.csv")
        seal(self.base / "manifest.json", self.lock)
        with self.assertRaisesRegex(ValueError, "Mask drops"):
            E.verify_data_lock(self.base / "manifest.json", self.candidates)

    def test_preacquisition_mapping_chronology_required(self):
        mapping = E.read(self.base / "mapping.json")
        mapping["locked_utc"] = "2026-09-20T00:00:00+00:00"
        self.update("mapping.json", mapping)
        with self.assertRaisesRegex(ValueError, "Mapping was not sealed"):
            E.verify_data_lock(self.base / "manifest.json", self.candidates)

    def test_historical_target_or_changed_qc_not_allowed(self):
        self.lock["data_gate"]["start"] = "2015-01-01"
        seal(self.base / "manifest.json", self.lock)
        with self.assertRaisesRegex(ValueError, "Data/QC rules changed"):
            E.verify_data_lock(self.base / "manifest.json", self.candidates)

    def test_source_drift_rejected(self):
        with (self.base / "market_00.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Locked file changed"):
            E.verify_data_lock(self.base / "manifest.json", self.candidates)

    def test_nineteen_markets_cannot_promote_historical_fallback(self):
        mapping = E.read(self.base / "mapping.json")
        mapping["markets"][0]["identity_verified"] = False
        self.update("mapping.json", mapping)
        reports = E.read(self.base / "qc.json")
        reports["market_00"], _ = E.D.check_market(pd.read_csv(self.base / "market_00.csv"), "market_00", identity_verified=False)
        self.update("qc.json", reports)
        with self.assertRaisesRegex(ValueError, "fewer than20"):
            E.verify_data_lock(self.base / "manifest.json", self.candidates)


class CheckpointGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.frozen, self.battery, self.manifest = checkpoint_fixture(Path(self.temp.name))
        self.patcher = patch.object(E, "verify_design", return_value=self.manifest)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def test_complete_matched_ten_finals(self):
        result = E.verify_checkpoints(self.frozen, self.battery, E.read)
        self.assertEqual(len(result["entries"]), 10)

    def enable_reuse(self):
        original = Path(self.temp.name) / "original_design"
        seal(original / "manifest.json", self.manifest)
        self.assertEqual(E.sha(original / "manifest.json"), E.sha(self.frozen / "manifest.json"))
        legacy_patch = patch.object(E, "LEGACY_DESIGN", original)
        legacy_patch.start()
        self.addCleanup(legacy_patch.stop)
        reused = []
        for seed in (8101, 8102):
            directory = self.battery / "runs" / ("matern_" + str(seed))
            status = E.read(directory / "status.json")
            reused.append(dict(seed=seed, arm="matern", source_manifest_sha256=status["manifest_sha256"],
                status_sha256=E.sha(directory / "status.json"), checkpoint_sha256=E.sha(directory / "final.pt"),
                initial_state_sha256=status["initial_state_sha256"], initial_file_sha256=E.sha(directory / "initial.pt")))
        self.manifest["reused_checkpoints"] = reused
        seal(self.frozen / "manifest.json", self.manifest)
        current_hash = E.sha(self.frozen / "manifest.json")
        for seed in E.SEEDS:
            for arm in E.ARMS:
                if arm == "matern" and seed in (8101, 8102):
                    continue
                directory = self.battery / "runs" / (arm + "_" + str(seed))
                checkpoint = E.read(directory / "final.pt")
                checkpoint["manifest_sha256"] = current_hash
                write(directory / "final.pt", checkpoint)
                status = E.read(directory / "status.json")
                status.update(manifest_sha256=current_hash, checkpoint_sha256=E.sha(directory / "final.pt"))
                write(directory / "status.json", status)
        return reused

    def test_two_reused_matern_finals_keep_original_provenance(self):
        reused = self.enable_reuse()
        result = E.verify_checkpoints(self.frozen, self.battery, E.read)
        self.assertEqual(len(result["entries"]), 10)
        imported = [e for e in result["entries"] if e["provenance"] == "immutable_reused_matern_final"]
        self.assertEqual([(e["arm"], e["seed"]) for e in imported], [("matern", 8101), ("matern", 8102)])
        self.assertTrue(all(e["source_manifest_sha256"] == reused[0]["source_manifest_sha256"] for e in imported))
        self.assertNotEqual(imported[0]["source_manifest_sha256"], result["training_manifest_sha256"])

    def test_unlisted_old_checkpoint_rejected(self):
        self.enable_reuse()
        self.manifest.pop("reused_checkpoints")
        with self.assertRaisesRegex(ValueError, "Training provenance mismatch"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_broad_reuse_rejected(self):
        self.enable_reuse()[0]["arm"] = "broad"
        with self.assertRaisesRegex(ValueError, "only the two completed Matérn"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_other_matern_seed_reuse_rejected(self):
        self.enable_reuse()[0]["seed"] = 8103
        with self.assertRaisesRegex(ValueError, "only the two completed Matérn"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_wrong_reuse_manifest_rejected(self):
        self.enable_reuse()[0]["source_manifest_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Reuse source manifest mismatch"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_reuse_changed_training_settings_rejected(self):
        self.enable_reuse()
        self.manifest["training"]["tokens"] += 1
        with self.assertRaisesRegex(ValueError, "Reused checkpoint training settings changed"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_reuse_status_tampering_rejected_before_loading(self):
        self.enable_reuse()
        path = self.battery / "runs/matern_8101/status.json"
        status = E.read(path)
        status["extra"] = "changed"
        write(path, status)
        loader = Mock(side_effect=AssertionError("No weights may load"))
        with self.assertRaisesRegex(ValueError, "immutable whitelist mismatch"):
            E.verify_checkpoints(self.frozen, self.battery, loader)
        loader.assert_not_called()

    def test_reuse_initial_file_tampering_rejected(self):
        self.enable_reuse()
        path = self.battery / "runs/matern_8101/initial.pt"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "immutable whitelist mismatch"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_reuse_checkpoint_tampering_rejected(self):
        self.enable_reuse()
        path = self.battery / "runs/matern_8101/final.pt"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Checkpoint checksum mismatch"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_incomplete_reused_final_rejected(self):
        self.enable_reuse()
        path = self.battery / "runs/matern_8101/status.json"
        status = E.read(path)
        status["completed_updates"] = 19500
        write(path, status)
        with self.assertRaisesRegex(ValueError, "Incomplete fixed-budget final"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_reused_checkpoint_metadata_manifest_rechecked(self):
        reused = self.enable_reuse()
        path = self.battery / "runs/matern_8101/final.pt"
        checkpoint = E.read(path)
        checkpoint["manifest_sha256"] = "a" * 64
        write(path, checkpoint)
        status = E.read(path.with_name("status.json"))
        status["checkpoint_sha256"] = E.sha(path)
        write(path.with_name("status.json"), status)
        reused[0].update(checkpoint_sha256=E.sha(path), status_sha256=E.sha(path.with_name("status.json")))
        with self.assertRaisesRegex(ValueError, "Final checkpoint metadata mismatch"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_missing_seed_stops_before_weight_loading(self):
        path = self.battery / "runs/matern_8105/status.json"
        status = E.read(path)
        status["status"] = "failed"
        write(path, status)
        loader = Mock(side_effect=AssertionError("No weights may load"))
        with self.assertRaisesRegex(ValueError, "Incomplete fixed-budget"):
            E.verify_checkpoints(self.frozen, self.battery, loader)
        loader.assert_not_called()

    def test_wrong_step_cannot_be_promoted_to_final(self):
        path = self.battery / "runs/broad_8101/final.pt"
        checkpoint = E.read(path)
        checkpoint["step"] = 19500
        write(path, checkpoint)
        status = E.read(path.with_name("status.json"))
        status["checkpoint_sha256"] = E.sha(path)
        write(path.with_name("status.json"), status)
        with self.assertRaisesRegex(ValueError, "Final checkpoint metadata"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)

    def test_initial_state_pair_mismatch_rejected(self):
        path = self.battery / "runs/matern_8102/initial.pt"
        initial = E.read(path)
        initial["state_dict"]["weight"][0] += 1
        initial["sha256"] = E._state_hash(initial["state_dict"])
        write(path, initial)
        status = E.read(path.with_name("status.json"))
        status["initial_state_sha256"] = initial["sha256"]
        write(path.with_name("status.json"), status)
        with self.assertRaisesRegex(ValueError, "Initial weights differ"):
            E.verify_checkpoints(self.frozen, self.battery, E.read)


class PairedScoreTests(unittest.TestCase):
    def setUp(self):
        self.mask = pd.DataFrame([(f"market_{i:02}", d) for i in range(20)
                                  for d in ("2023-01-03", "2023-01-04")], columns=["market", "date"])
        self.frames = {}
        for arm in E.ARMS:
            for seed in E.SEEDS:
                frame = self.mask.copy()
                frame["target"] = 1.
                frame["mean"] = 0.
                frame["h"] = 1. if arm == "broad" else 2.
                frame["native_logscore"] = -1. if arm == "broad" else -2.
                self.frames[(arm, seed)] = frame

    def test_all_seeds_descriptive_and_gain_sign(self):
        gains, descriptions, _ = E.paired_scores(self.frames, self.mask)
        self.assertEqual(len(gains), 5 * len(self.mask))
        self.assertEqual(len(descriptions), 10)
        self.assertTrue((gains.gain > 0).all())
        self.assertAlmostEqual(descriptions["broad_8101"]["mean"]["qlike"], 1.)
        self.assertAlmostEqual(descriptions["matern_8101"]["mean"]["qlike"], .5 + np.log(2))
        result = E.ST.analyze_study(gains, expected_rows=self.mask, reps=39)
        self.assertEqual(result["contrasts"][E.ST.SECONDARY]["p_superiority"], 1.)

    def test_no_surviving_seed_average(self):
        del self.frames[("matern", 8105)]
        with self.assertRaisesRegex(ValueError, "Missing checkpoint predictions"):
            E.paired_scores(self.frames, self.mask)

    def test_target_mismatch_rejected(self):
        self.frames[("matern", 8101)].loc[0, "target"] = 2.
        with self.assertRaisesRegex(ValueError, "Paired targets differ"):
            E.paired_scores(self.frames, self.mask)

    def test_paired_but_different_target_in_another_seed_rejected(self):
        for arm in E.ARMS:
            self.frames[(arm, 8102)].loc[0, "target"] = 2.
        with self.assertRaisesRegex(ValueError, "Paired targets differ"):
            E.paired_scores(self.frames, self.mask)

    def test_nonfinite_forecast_rejected(self):
        self.frames[("broad", 8101)].loc[0, "h"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            E.paired_scores(self.frames, self.mask)

    def test_gate_failure_does_not_create_output_or_infer(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "results"
            with patch.object(E, "verify_plan", return_value={"candidate_population": []}), \
                    patch.object(E, "verify_data_lock", side_effect=ValueError("missing data lock")), \
                    patch.object(E, "verify_checkpoints") as check:
                with self.assertRaisesRegex(ValueError, "missing data lock"):
                    E.run("plan", "training", "data", out)
                check.assert_not_called()
                self.assertFalse(out.exists())

    def test_mocked_ten_final_pipeline_writes_auditable_report(self):
        from research.seed_replication import causal
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            plan_path, data_path = base / "plan/manifest.json", base / "data/manifest.json"
            write(plan_path, {"synthetic": True})
            write(data_path, {"synthetic": True})
            dates = np.r_[pd.bdate_range("2021-01-01", periods=100).strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                          np.asarray(["2023-01-03", "2023-01-04"])]
            plan = dict(candidate_population=list(self.mask.market.unique()), training_design=str(base / "design"),
                        inference=dict(device="cpu"), scope="Synthetic integration fixture")
            data = dict(manifest={"exposure_disclosure": "Synthetic fixture only"}, manifest_sha256=E.sha(data_path),
                        mask=self.mask, series={m: dict(date=dates, r=np.ones(len(dates))) for m in self.mask.market.unique()})
            entries = []
            for arm in E.ARMS:
                for seed in E.SEEDS:
                    checkpoint = base / "training/runs" / (arm + "_" + str(seed)) / "final.pt"
                    write(checkpoint, {"synthetic": True})
                    entries.append(dict(arm=arm, seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=E.sha(checkpoint)))
            registry = dict(training_manifest_sha256="synthetic", entries=entries)
            def predict(model, values, **kwargs):
                self.assertEqual(kwargs, dict(L=1024, stride=512, min_history=100, batch_windows=4, levels=()))
                broad = model.startswith("broad_")
                return dict(mean=np.zeros(len(values)), h=np.full(len(values), 1. if broad else 2.),
                            logscore=np.full(len(values), -1. if broad else -2.))
            with patch.object(E, "verify_plan", return_value=plan), patch.object(E, "verify_data_lock", return_value=data), \
                    patch.object(E, "verify_checkpoints", return_value=registry), \
                    patch.object(causal, "load_model", side_effect=lambda path, device: Path(path).parent.name) as loader, \
                    patch.object(causal, "predict_series", side_effect=predict):
                result = E.run(plan_path, base / "training", data_path, base / "evaluation")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(loader.call_count, 10)
            self.assertEqual(len(result["descriptive_by_arm_and_seed"]), 10)
            hashes = E.read(base / "evaluation/hashes.json")
            self.assertTrue(all(E.sha(base / "evaluation" / name) == digest for name, digest in hashes.items()))
            with np.load(base / "evaluation/bootstrap_evidence.npz") as evidence:
                self.assertEqual(evidence["draws"].shape, (5000,))
                self.assertEqual(evidence["counts"].shape, (5000, 2))


if __name__ == "__main__":
    unittest.main()
