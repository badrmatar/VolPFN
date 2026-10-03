"""Leakage, chronology, unit conversion and bounded synthetic runner tests."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import daily_runner as runner
from . import ginn


class DailyRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.series = runner.synthetic_smoke_series()
        self.scaler = runner.TrainingScaler.fit(self.series.training_returns())

    def teacher(self):
        standardized = self.scaler.transform(self.series.training_returns())
        return ginn.FrozenTeacher(
            ar_intercept=.02, ar_coefficients=(.4,), fallback_mean=0., garch_mean=0.,
            omega=.1, alpha=.1, beta=.8, initial_variance=1., train_count=self.scaler.count,
            train_sha256=runner.hash_values(standardized), train_residuals=(),
            train_teacher_variances=(), next_teacher_variance=1., fit_loglikelihood=0.)

    def test_runner_grid_and_seed_roster_are_finite_and_distinct(self):
        for name, count in (("garch_lstm", 2), ("ginn", 6), ("arch_lstm", 6)):
            grid = runner.registered_grid(name)
            self.assertEqual(len(grid), count)
            self.assertLessEqual(len(grid), 12)
            self.assertTrue(all(c["context"] == runner.CONTEXTS[name] for c in grid))
            self.assertTrue(all(c["optimizer_kwargs"]["eps"] == 1e-8 for c in grid))
        self.assertNotIn(runner.TUNING_SEED, runner.FINAL_SEEDS)
        self.assertEqual(runner.FINAL_SEEDS, (8101, 8102, 8103, 8104, 8105))
        self.assertEqual(runner.registered_grid("ginn")[0]["optimizer_kwargs"]["weight_decay"], .01)

    def test_runner_scaler_uses_only_training_values(self):
        changed = self.series.returns.copy()
        changed[self.series.dates > runner.TRAIN_END] = 10000.
        future_changed = replace(self.series, returns=changed)
        changed_scaler = runner.TrainingScaler.fit(future_changed.training_returns())
        self.assertEqual(self.scaler, changed_scaler)
        centered = self.scaler.transform(self.series.training_returns())
        self.assertAlmostEqual(centered.mean(), 0., places=12)
        self.assertAlmostEqual(np.mean(centered ** 2), 1., places=12)

    def test_runner_raw_mean_variance_and_second_moment_restore_units(self):
        scaler = runner.TrainingScaler(center=.2, rms=3., count=2, training_sha256="test")
        mean, variance, second = scaler.raw_params(torch.tensor([.5]), torch.tensor([2.]))
        torch.testing.assert_close(mean, torch.tensor([1.7]))
        torch.testing.assert_close(variance, torch.tensor([18.]))
        torch.testing.assert_close(second, torch.tensor([20.89]))

    def test_runner_decimal_source_multiplied_by_100_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(json.dumps({"scope": "daily open-to-close"}))
            (root / "audit.json").write_text("{}")
            roster = root / "inventory.json"
            roster.write_text(json.dumps({"retained_index_population": list(runner.EXPECTED_ROSTER)}))
            np.savez(root / "SPX.npz", date=self.series.dates,
                     returns=self.series.returns / 100., rv5=np.full(len(self.series.dates), np.nan))
            source = runner.load_series("SPX", root, roster)
            np.testing.assert_allclose(source.returns, self.series.returns, rtol=1e-15, atol=1e-15)
            self.assertEqual(source.lock()["input_conversion"], "decimal * 100 exactly once")

    def test_runner_context_excludes_target_and_dates_align(self):
        for name in ("garch_lstm", "arch_lstm"):
            windows = runner.build_windows(self.series, self.scaler, name, "validation")
            i = int(windows.indices[0])
            standardized = self.scaler.transform(self.series.returns)
            np.testing.assert_allclose(windows.inputs[0].numpy(), standardized[i-runner.CONTEXTS[name]:i],
                                       rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(windows.targets[0].item(), standardized[i], places=6)
            self.assertLess(self.series.dates[i-1], self.series.dates[i])
            self.assertTrue((self.series.dates[windows.indices] > runner.TRAIN_END).all())
            self.assertTrue((self.series.dates[windows.indices] <= runner.VALIDATION_END).all())

    def test_runner_ginn_residual_filter_is_continuous_and_teacher_before_target(self):
        teacher = self.teacher()
        windows = runner.build_windows(self.series, self.scaler, "ginn", "validation", teacher)
        i = int(windows.indices[0])
        standardized = self.scaler.transform(self.series.returns)
        first = i - runner.CONTEXTS["ginn"]
        expected = (standardized[first] - teacher.ar_intercept
                    - teacher.ar_coefficients[0] * standardized[first-1]) ** 2
        self.assertAlmostEqual(windows.inputs[0, 0].item(), expected, places=6)
        prefix = torch.tensor(standardized[:i][None, :])
        self.assertAlmostEqual(windows.teacher_variance[0].item(),
                               ginn.teacher_variances(prefix, teacher)[0, -1].item(), places=6)
        expected_mean = teacher.ar_intercept + teacher.ar_coefficients[0] * standardized[i-1]
        self.assertAlmostEqual(windows.mean[0].item(), expected_mean, places=6)

    def test_runner_target_and_future_changes_cannot_alter_first_forecast_input(self):
        teacher = self.teacher()
        before = runner.build_windows(self.series, self.scaler, "ginn", "validation", teacher)
        i = int(before.indices[0])
        changed = self.series.returns.copy()
        changed[i:] = 900.
        after = runner.build_windows(replace(self.series, returns=changed), self.scaler,
                                     "ginn", "validation", teacher)
        torch.testing.assert_close(before.inputs[0], after.inputs[0], rtol=0, atol=0)
        torch.testing.assert_close(before.mean[0], after.mean[0], rtol=0, atol=0)
        torch.testing.assert_close(before.teacher_variance[0], after.teacher_variance[0], rtol=0, atol=0)
        self.assertNotEqual(before.targets[0].item(), after.targets[0].item())

    def test_runner_incomplete_or_wrong_teacher_is_explicit_failure(self):
        with self.assertRaises(ValueError):
            runner.build_windows(self.series, self.scaler, "ginn", "train", None)
        wrong_teacher = replace(self.teacher(), train_sha256="future_fit")
        with self.assertRaises(ValueError):
            runner.build_windows(self.series, self.scaler, "ginn", "train", wrong_teacher)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "teacher_failure"
            with mock.patch.object(ginn, "fit_teacher", side_effect=RuntimeError("convergence failure")):
                with self.assertRaises(RuntimeError):
                    runner.tune(self.series, "ginn", output, manifest=None, smoke=True, epochs=1)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["market_retained_as_failure"], "SYNTHETIC")
            self.assertTrue((output / "freeze.json").exists())
            self.assertFalse((output / "result.json").exists())

    def test_runner_rejects_overwrite_missing_manifest_and_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                runner.new_output(output)
            with self.assertRaises(ValueError):
                runner.manifest_lock(None)
            lock = runner.stage_lock(self.series, None, smoke=True)
            with mock.patch.object(runner, "source_hashes", return_value={"changed": "true"}):
                with self.assertRaises(ValueError):
                    runner.verify_lock(lock, self.series, None, smoke=True)

    def test_runner_evaluation_values_do_not_change_tuning_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = runner.tune(self.series, "arch_lstm", root / "first", manifest=None, smoke=True, epochs=1)
            returns = self.series.returns.copy()
            returns[self.series.dates > runner.VALIDATION_END] = 100000.
            changed = replace(self.series, returns=returns)
            second = runner.tune(changed, "arch_lstm", root / "second", manifest=None, smoke=True, epochs=1)
            self.assertEqual(first["selected"]["config"], second["selected"]["config"])
            self.assertEqual(first["selected"]["validation_raw_r2_qlike"], second["selected"]["validation_raw_r2_qlike"])
            a = torch.load(root / "first/config_00/checkpoint.pt", weights_only=True)["state_dict"]
            b = torch.load(root / "second/config_00/checkpoint.pt", weights_only=True)["state_dict"]
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertFalse(list((root / "first").glob("**/forecasts*.csv")))

    def test_runner_bounded_end_to_end_all_models_and_forecast_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke"
            result = runner.smoke(output)
            self.assertEqual(result["status"], "complete")
            for name in runner.MODELS:
                path = output / f"{name}_evaluate/forecasts_8101.csv"
                with path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 20)
                for row in rows:
                    i = int(row["index"])
                    self.assertEqual(row["date"], self.series.dates[i])
                    self.assertLess(row["context_end"], row["date"])
                    self.assertEqual(float(row["return"]), float(row["raw_return"]))
                    self.assertEqual(row["target_sha256"], runner.hash_dated_values(
                        self.series.dates[i:i+1], self.series.returns[i:i+1]))
                    h, r = float(row["second_moment"]), float(row["return"])
                    self.assertAlmostEqual(float(row["qlike"]), r*r/h + np.log(h), places=12)
                    self.assertGreater(float(row["variance"]), 0)
                    self.assertGreater(float(row["nu"]), 2)
                    self.assertTrue(np.isfinite(float(row["common_t8_nll"])))
                status = json.loads((output / f"{name}_train/seed_8101/training_status.json").read_text())
                self.assertLessEqual(status["epochs_run"], 2)
                self.assertFalse(status["evaluation_predictions_or_scores_read"])
                self.assertIn("optimizer_kwargs", status)
            # A partial seed roster may never become a surviving-seed result.
            trained_path = output / "arch_lstm_train/result.json"
            trained = runner.read_result(trained_path)
            trained["seeds"] = []
            runner.seal_result(trained_path, trained)
            with self.assertRaises(ValueError):
                runner.evaluate(self.series, trained_path.parent, output / "invalid_evaluate",
                                manifest=None, smoke=True)

    def test_runner_result_and_artifact_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            runner.seal_result(path, {"status": "complete"})
            path.write_text('{"status":"modified"}')
            with self.assertRaises(ValueError):
                runner.read_result(path)

    def test_runner_frozen_masks_bind_source_date_index_and_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_files = {}
            for split in ("train", "validation", "evaluation"):
                path = root / f"{split}.csv"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["market", "date", "row_index", "return"])
                    writer.writeheader()
                    for i in self.series.indices(split, 0):
                        writer.writerow({"market": self.series.market, "date": self.series.dates[i],
                                         "row_index": i, "return": self.series.returns[i]})
                mask_files[split] = {"path": path.name}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"population": [self.series.market],
                "data": {"directory": "fixture_data"},
                "data_hashes": {"fixture_data/SYNTHETIC.npz": self.series.raw_file_sha256},
                "mask_files": mask_files}))
            lock = {"path": str(manifest_path)}
            runner.verify_mask_labels(self.series, lock)
            altered = self.series.returns.copy()
            altered[-1] += 1.
            with self.assertRaises(ValueError):
                runner.verify_mask_labels(replace(self.series, returns=altered), lock)


if __name__ == "__main__":
    unittest.main()
