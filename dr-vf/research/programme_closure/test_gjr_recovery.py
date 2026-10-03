"""Mocked closure orchestration: no GPUs, paid work or market inference."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from research.programme_closure import evaluate as E, gjr_recovery as R, run_pipeline as P
from research.programme_closure.test_evaluate import seal, write


NOW = datetime(2026, 9, 19, 13, tzinfo=timezone.utc)
ORIGINAL_DEADLINE = datetime(2026, 9, 19, 15, 56, 38, tzinfo=timezone.utc)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / "research/outputs/programme_closure_20260919_v1"
        self.old = self.base / "gpu_design_v1"
        self.frozen = self.base / "gpu_design_v2"
        self.output = self.base / "gpu_battery_v2"
        self.s = dict(steps=20000, tokens=8192, cfg=E.CFG, device="cpu",
                      normalization_spec={"id": "test_normalization"})
        old = dict(training=self.s, seeds=E.SEEDS, arms=E.ARMS, source_hashes={})
        seal(self.old / "manifest.json", old)
        self.old_hash = E.sha(self.old / "manifest.json")
        reused = []
        for seed in (8101, 8102):
            path = self.base / "reuse_inputs_v1/runs" / ("matern_" + str(seed))
            status = self.checkpoint("matern", seed, self.s, path, self.old_hash, None)
            reused.append(dict(seed=seed, arm="matern", source_manifest_sha256=self.old_hash,
                status_sha256=E.sha(path / "status.json"), checkpoint_sha256=E.sha(path / "final.pt"),
                initial_state_sha256=status["initial_state_sha256"], initial_file_sha256=E.sha(path / "initial.pt"),
                source_directory=str(path.relative_to(self.root))))
        self.manifest = dict(old, revision="exact_gjr_semivariance_v2", reused_checkpoints=reused,
            fresh_training_count=8, order=[dict(seed=seed, arm=arm) for i, seed in enumerate(E.SEEDS)
                for arm in (E.ARMS if i % 2 == 0 else E.ARMS[::-1])])
        seal(self.frozen / "manifest.json", self.manifest)
        self.digest = E.sha(self.frozen / "manifest.json")
        screen = self.base / 'broad_input_preflight_v2'
        screen.mkdir()
        (screen/'batch_hashes.jsonl').write_text('synthetic input validation fixture\n')
        write(screen/'status.json',dict(status='complete',completed=100000,total=100000,
            manifest_sha256=self.digest,batch_hashes_sha256=E.sha(screen/'batch_hashes.jsonl')))
        self.runtime = dict(python="test", numpy="test", scipy="test", torch="test", cuda="test", gpu="test")
        self.timing = self.base / "timing.json"
        write(self.timing, dict(status="complete", manifest_sha256=self.digest, runtime=self.runtime,
            records=[dict(arm="broad", full_run_projected_seconds=100), dict(arm="matern", full_run_projected_seconds=200)]))
        self.authorization = self.base / "authorization.json"
        self.receipt = dict(approved_full_battery=True, manifest_sha256=self.digest,
            timing_path=str(self.timing), timing_sha256=E.sha(self.timing), runtime=self.runtime,
            cost_ceiling_usd=3.75, user_authorization_reference="Synthetic test authorization",
            stop_deadline_utc=ORIGINAL_DEADLINE.isoformat())
        write(self.authorization, self.receipt)
        real_verify = E.verify_checkpoints
        for target, name, value in [(R, "ROOT", self.root), (R, "BASE", self.base),
                (R, "datetime", FixedDateTime), (E, "LEGACY_DESIGN", self.old)]:
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.mocks = {}
        for name, target, attr, kwargs in [
            ("design", R.G, "verify", dict(return_value=self.manifest)),
            ("eval_design", E, "verify_design", dict(return_value=self.manifest)),
            ("runtime", R.G, "runtime", dict(return_value=self.runtime)),
            ("configure", R.G, "configure", {}),
            ("panels", R.M, "shared_panels", dict(return_value={})),
            ("load", R.causal, "load_model", dict(return_value=object())),
            ("validate", R.M, "evaluate_panels", dict(return_value={"synthetic": True})),
            ("train", R.G, "train_one", dict(side_effect=self.checkpoint)),
            ("registry", E, "verify_checkpoints", dict(side_effect=lambda f, o: real_verify(f, o, E.read)))]:
            p = patch.object(target, attr, **kwargs)
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def checkpoint(arm, seed, spec, directory, manifest_hash, panels):
        directory.mkdir(parents=True, exist_ok=False)
        state = {"weight": [float(seed), 1.]}
        initial_hash = E._state_hash(state)
        write(directory / "initial.pt", dict(state_dict=state, sha256=initial_hash))
        write(directory / "final.pt", dict(step=20000, seed=seed, cfg=E.CFG, manifest_sha256=manifest_hash,
            training_spec=dict(spec, arm=arm), causal_spec="running_nonzero_rms_window_v1",
            normalization_spec=spec["normalization_spec"], state_dict=state))
        status = dict(status="complete", seed=seed, arm=arm, completed_updates=20000,
            input_positions=163840000, manifest_sha256=manifest_hash, initial_state_sha256=initial_hash,
            checkpoint_sha256=E.sha(directory / "final.pt"), causal_probe=dict(status="passed"))
        write(directory / "status.json", status)
        return status

    def run_recovery(self):
        R.run(self.frozen, self.output, self.authorization)

    def test_eight_fresh_two_unchanged_and_actual_ten_checkpoint_gate(self):
        self.run_recovery()
        fresh = [(c.args[0], c.args[1]) for c in self.mocks["train"].call_args_list]
        expected = [(e["arm"], e["seed"]) for e in self.manifest["order"]
                    if (e["arm"], e["seed"]) not in {("matern", 8101), ("matern", 8102)}]
        self.assertEqual(fresh, expected)
        self.assertEqual(len(fresh), 8)
        self.assertEqual(len(set(fresh)), 8)
        self.assertEqual(self.mocks["validate"].call_count, 2)
        self.mocks["registry"].assert_called_once_with(self.frozen, self.output)
        self.assertEqual(E.read(self.output / "status.json")["status"], "complete")
        for entry in self.manifest["reused_checkpoints"]:
            old = self.root / entry["source_directory"]
            new = self.output / "runs" / ("matern_" + str(entry["seed"]))
            for name in ("status.json", "final.pt", "initial.pt"):
                self.assertEqual((old / name).read_bytes(), (new / name).read_bytes())
            self.assertEqual(E.read(new / "final.pt")["manifest_sha256"], self.old_hash)
            self.assertEqual(E.read(new / "REUSE_PROVENANCE.json")["new_comparison_design_sha256"], self.digest)

    def test_training_failure_retained_without_retry_or_scoring(self):
        def fail(arm, seed, *args):
            if (arm, seed) == ("broad", 8103):
                raise FloatingPointError("synthetic failure")
            return self.checkpoint(arm, seed, *args)
        self.mocks["train"].side_effect = fail
        with self.assertRaisesRegex(FloatingPointError, "synthetic failure"):
            self.run_recovery()
        calls = [(c.args[0], c.args[1]) for c in self.mocks["train"].call_args_list]
        self.assertEqual(calls, [("broad", 8101), ("broad", 8102), ("broad", 8103)])
        self.assertEqual(E.read(self.output / "status.json")["status"], "incomplete_or_failed")
        self.assertTrue((self.output / "failure.txt").is_file())
        self.mocks["registry"].assert_not_called()

    def test_existing_version_output_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "sentinel.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_recovery()
        self.assertEqual(sentinel.read_text(), "preserve")
        self.mocks["train"].assert_not_called()

    def test_bad_authorization_gates_stop_before_training(self):
        variants = [dict(cost_ceiling_usd=4), dict(cost_ceiling_usd=0),
                    dict(approved_full_battery=False), dict(manifest_sha256="a" * 64),
                    dict(stop_deadline_utc=(NOW - timedelta(seconds=1)).isoformat()),
                    dict(stop_deadline_utc=(ORIGINAL_DEADLINE + timedelta(hours=1)).isoformat())]
        for changed in variants:
            with self.subTest(changed=changed):
                write(self.authorization, dict(self.receipt, **changed))
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_recovery()
                self.assertFalse(self.output.exists())
        self.mocks["train"].assert_not_called()

    def test_changed_timing_or_runtime_stop_before_training(self):
        self.timing.write_text(self.timing.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Changed timing"):
            self.run_recovery()
        self.assertFalse(self.output.exists())
        self.receipt["timing_sha256"] = E.sha(self.timing)
        write(self.authorization, self.receipt)
        self.mocks["runtime"].return_value = dict(self.runtime, torch="different")
        with self.assertRaisesRegex(ValueError, "Runtime changed"):
            self.run_recovery()
        self.mocks["train"].assert_not_called()

    def test_all_ten_validation_failure_revokes_complete_status(self):
        self.mocks["registry"].side_effect = ValueError("synthetic invalid registry")
        with self.assertRaisesRegex(ValueError, "invalid registry"):
            self.run_recovery()
        self.assertEqual(E.read(self.output / "status.json")["status"], "incomplete_or_failed")

    def test_authorization_keeps_original_allowance_deadline_and_is_exclusive(self):
        path = self.base / "new_authorization.json"
        receipt = R.authorize(self.frozen, self.timing, path, "synthetic-pod")
        self.assertEqual(receipt["total_programme_compute_allowance_usd"], 5)
        self.assertEqual(receipt["cost_ceiling_usd"], 3.75)
        self.assertEqual(receipt["stop_deadline_utc"], "2026-09-19T15:56:38Z")
        self.assertEqual(receipt["eight_fresh_training_projected_seconds"], 1100)
        with self.assertRaises(FileExistsError):
            R.authorize(self.frozen, self.timing, path, "synthetic-pod")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "research/outputs/programme_closure_20260919_v1"
        self.authorization = self.output / "authorization.json"
        write(self.authorization, dict(approved_full_battery=True, stop_deadline_utc=ORIGINAL_DEADLINE.isoformat()))
        self.evaluation = self.output / "evaluation_plan_v3/manifest.json"
        self.confirmation = self.output / "benchmark_confirmation_plan_v3/manifest.json"
        p = patch.object(P, "now", return_value=NOW)
        self.clock = p.start()
        self.addCleanup(p.stop)
        p = patch.object(P.subprocess, "Popen")
        self.launch = p.start()
        self.addCleanup(p.stop)
        self.launch.return_value = Mock(pid=12345, wait=Mock(return_value=0))

    def run_pipeline(self):
        P.run(self.root, self.authorization, self.evaluation, self.confirmation, revision="gjr-v2")

    def test_v2_paths_preserve_original_and_finish_all_stages(self):
        old = self.output / "PIPELINE_STATUS.json"
        old.write_text("old failure untouched")
        self.run_pipeline()
        commands = [c.args[0] for c in self.launch.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn("research.programme_closure.gjr_recovery", commands[0])
        self.assertIn(str(self.output / "gpu_design_v2"), commands[0])
        for command in commands:
            self.assertIn(str(self.output / "gpu_battery_v2"), command)
        self.assertIn(str(self.output / "matched_primary_v2"), commands[1])
        self.assertIn(str(self.output / "benchmark_confirmation_v2"), commands[2])
        self.assertEqual(old.read_text(), "old failure untouched")
        self.assertTrue((self.output / "PIPELINE_STARTED_v2.json").is_file())
        self.assertEqual(E.read(self.output / "PIPELINE_STATUS_v2.json")["status"], "complete_export_and_cloud_stop_required")

    def test_duplicate_pipeline_start_refused(self):
        (self.output / "PIPELINE_STARTED_v2.json").write_text("existing")
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        self.launch.assert_not_called()

    def test_failed_training_never_launches_market_stages(self):
        self.launch.return_value.wait.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "Programme stage failed"):
            self.run_pipeline()
        self.assertEqual(self.launch.call_count, 1)
        self.assertEqual(E.read(self.output / "PIPELINE_STATUS_v2.json")["status"], "failed")

    def test_expired_initial_receipt_launches_nothing(self):
        self.clock.return_value = ORIGINAL_DEADLINE + timedelta(seconds=1)
        with self.assertRaises(ValueError):
            self.run_pipeline()
        self.launch.assert_not_called()

    def test_timeout_terminates_process_group_and_no_next_stage(self):
        self.launch.return_value.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 1), 0]
        with patch.object(P.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "Worker deadline reached"):
                self.run_pipeline()
        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(self.launch.call_count, 1)
        self.assertEqual(E.read(self.output / "PIPELINE_STATUS_v2.json")["status"], "incomplete_deadline")

    def test_expiry_between_stages_does_not_launch_market_stage(self):
        def finish_after_deadline(*args, **kwargs):
            self.clock.return_value = ORIGINAL_DEADLINE + timedelta(seconds=1)
            return 0
        self.launch.return_value.wait.side_effect = finish_after_deadline
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_pipeline()
        self.assertEqual(self.launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
