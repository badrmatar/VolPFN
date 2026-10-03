"""Small mocked integration checks; no GPU, holdout inference, or final runs."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from research.experiment_programme_20260930 import ablations as A
from research.experiment_programme_20260930 import evaluation as E
from research.experiment_programme_20260930 import managed as M
from research.experiment_programme_20260930 import programme as P
from research.experiment_programme_20260930 import statistics as S


class IntegrationTests(unittest.TestCase):
    def _designs(self, base):
        base = Path(base)
        primary, component = base / "design/primary/manifest.json", base / "design/component/manifest.json"
        primary.parent.mkdir(parents=True)
        component.parent.mkdir(parents=True)
        primary.write_text("{}\n")
        component.write_text("{}\n")
        manifest = dict(volpfn_arms=["direct"], daily_models=[], population=["AAA"], seeds=[8101],
            external_unavailable_slots={}, component_design=dict(path="design/component/manifest.json",
                                                                 sha256=P.sha(component)))
        return primary, component, manifest

    def test_portable_bundle_contains_bound_component_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            primary, component, manifest = self._designs(base)
            source = base / "source.py"
            source.write_text("# frozen source\n")
            manifest.update(source_hashes={"source.py": P.sha(source)}, data_hashes={})
            bound_data = component.parent / "data/AAA.npz"
            bound_data.parent.mkdir()
            bound_data.write_bytes(b"bound complete source NPZ snapshot")
            seal = component.with_suffix(".sha256")
            seal.write_text(P.sha(component) + "\n")
            with mock.patch.object(P, "ROOT", base), mock.patch.object(P, "verify", return_value=manifest), \
                 mock.patch.object(A, "verify", return_value={"specification": {"smoke": False}}):
                receipt = M.bundle(primary, base / "bundle.tar.gz", ablation_manifest=component)
            with tarfile.open(base / "bundle.tar.gz") as archive:
                names = set(archive.getnames())
            self.assertIn("design/component/manifest.json", names)
            self.assertIn("design/component/manifest.sha256", names)
            self.assertIn("design/component/data/AAA.npz", names)
            self.assertEqual(receipt["component_design"], manifest["component_design"])

    def test_collect_rejects_changed_component_forecast_or_wrong_design(self):
        for corrupt in ("forecast", "design"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary).resolve()
                primary, _, manifest = self._designs(base)
                manifest["volpfn_arms"] = ["full"]
                directory = base / "results/component_evaluation/full"
                directory.mkdir(parents=True)
                table = directory / "forecasts.csv"
                table.write_text("market,date,seed,return,second_moment,qlike\nAAA,2015-01-02,8101,1,1,1\n")
                receipt = dict(status="complete", forecast_sha256=P.sha(table),
                    manifest_sha256=P.sha(primary),
                    ablation_manifest_sha256=manifest["component_design"]["sha256"])
                receipt["forecast_sha256" if corrupt == "forecast" else "manifest_sha256"] = "bad"
                (directory / "result.json").write_text(json.dumps(receipt))
                with mock.patch.object(P, "verify", return_value=manifest):
                    with self.assertRaises(ValueError):
                        M.collect(primary, base / "results", base / "collected")

    def test_daily_collection_preserves_exact_csv_target_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            primary, _, manifest = self._designs(base)
            manifest.update(volpfn_arms=[], daily_models=["arch_lstm"])
            directory = base / "results/daily/arch_lstm/AAA/evaluation"
            directory.mkdir(parents=True)
            observed = -0.45467078517172255
            frame = pd.DataFrame([dict(market="AAA", date="2015-01-02", seed=8101,
                                       **{"return": observed}, second_moment=1., qlike=observed ** 2)])
            forecast = directory / "forecasts_8101.csv"
            frame.to_csv(forecast, index=False, float_format="%.17g")
            receipt = dict(status="complete", smoke=False, seeds=[8101],
                lock={"manifest": {"sha256": P.sha(primary)}},
                forecast_sha256={forecast.name: P.sha(forecast)})
            (directory / "result.json").write_text(json.dumps(receipt))
            with mock.patch.object(P, "verify", return_value=manifest):
                paths = M.collect(primary, base / "results", base / "collected")
            copied = pd.read_csv(paths["arch_lstm"], float_precision="round_trip")
            self.assertEqual(float(copied["return"].iloc[0]), observed)

    def test_evaluation_direct_none_density_and_frozen_inference_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            primary, component, manifest = self._designs(base)
            manifest["mask_files"] = {"evaluation": {"path": "mask.csv"}}
            dates = (np.datetime64("2012-06-01") + np.arange(1027).astype("timedelta64[D]")).astype("U10")
            returns = np.linspace(-.2, .3, 1027)
            indices = np.array([1024, 1025, 1026])
            mask = pd.DataFrame(dict(market="AAA", date=dates[indices], row_index=indices,
                                     **{"return": returns[indices]}))
            mask.to_csv(primary.parent / "mask.csv", index=False, float_format="%.17g")
            run = base / "training/runs/direct_8101"
            run.mkdir(parents=True)
            (run / "final.pt").write_bytes(b"mock checkpoint; never loaded")
            (run / "status.json").write_text(json.dumps(dict(status="complete",
                checkpoint_sha256=P.sha(run / "final.pt"))))
            spec = A.specification(device="cuda")
            spec["seeds"] = [8101]
            registered = {"specification": spec}

            def issued(model, histories, batch_size, device, targets):
                self.assertEqual(device, "cuda")
                self.assertEqual(len(histories), 3)
                for history, index in zip(histories, indices):
                    np.testing.assert_array_equal(history, returns[index-1024:index])
                np.testing.assert_array_equal(targets, returns[indices])
                return dict(h=np.ones(3), native_nll=None)

            with mock.patch.object(P, "ROOT", base), mock.patch.object(P, "verify", return_value=manifest), \
                 mock.patch.object(P, "read_market", return_value=(dates, returns, np.full(1027, np.nan))), \
                 mock.patch.object(A, "verify", return_value=registered), \
                 mock.patch.object(A, "load_model", return_value=object()), \
                 mock.patch.object(A, "predict_windows", side_effect=issued), \
                 mock.patch.object(A.G, "configure") as configure:
                result = E.evaluate_arm(primary, component, base / "training", "direct", base / "evaluation")
            configure.assert_called_once_with(spec)
            self.assertEqual(result["status"], "complete")
            table = pd.read_csv(base / "evaluation/forecasts.csv", float_precision="round_trip")
            self.assertTrue(table.native_nll.isna().all())
            self.assertTrue(np.isfinite(table.common_t8_nll).all())
            np.testing.assert_array_equal(table["return"], returns[indices])

    def test_fractional_seed_labels_are_rejected(self):
        mask = pd.DataFrame([dict(market="AAA", date="2015-01-02", **{"return": 1.})])
        frame = mask.assign(seed=8101.25, second_moment=1., qlike=1.)
        with self.assertRaises(ValueError):
            S.validate_forecasts(frame, mask, [8101])

    def test_zero_exit_without_completion_receipt_records_failed_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            primary, component, manifest = self._designs(base)
            queue = dict(manifest="design/primary/manifest.json", manifest_sha256=P.sha(primary),
                ablation_manifest="design/component/manifest.json", ablation_manifest_sha256=P.sha(component),
                entries=[dict(id="missing_receipt", module="mock.module", arguments=[], completion="absent.json")])
            queue_path = base / "queue.json"
            queue_path.write_text(json.dumps(queue))
            with mock.patch.object(P, "ROOT", base), mock.patch.object(P, "verify", return_value=manifest), \
                 mock.patch.object(A, "verify", return_value={}), \
                 mock.patch.object(M.subprocess, "run", return_value=mock.Mock(returncode=0)):
                with self.assertRaises(Exception):
                    M.execute(queue_path, base / "execution")
            receipt = json.loads((base / "execution/status.json").read_text())
            self.assertEqual(receipt["status"], "failed")


if __name__ == "__main__":
    unittest.main()
