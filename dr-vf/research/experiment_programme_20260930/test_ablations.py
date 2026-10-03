"""Local CPU engineering checks only; no final training or paid resources."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from research.experiment_programme_20260930 import ablations as A
from research.seed_replication import causal, training as T
import volpfn as V


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.s = A.specification(smoke=True, device="cpu")

    def _sources(self, base):
        base = Path(base)
        data = base / "originals"
        data.mkdir()
        files = {}
        train_dates = np.datetime64("2009-01-01") + np.arange(80).astype("timedelta64[D]")
        test_dates = np.datetime64("2012-01-01") + np.arange(10).astype("timedelta64[D]")
        dates = np.concatenate([train_dates, test_dates]).astype("U10")
        for index, market in enumerate(["AAA", "BBB"]):
            returns = np.concatenate([np.linspace(-.03, .04, 80) + index * .001,
                                      np.full(10, np.nan)])
            files[market] = data / (market + ".npz")
            np.savez(files[market], date=dates, returns=returns,
                     rv5=np.full(90, np.nan), rk_parzen=np.full(90, np.nan),
                     split=np.full(90, "TEST"))
        roster = base / "inventory.json"
        roster.write_text(json.dumps({"retained_index_population": ["AAA", "BBB"]}))
        return files, roster

    def _freeze(self, base):
        files, roster = self._sources(base)
        design = Path(base) / "design"
        manifest = A.freeze(design, smoke=True, device="cpu", historical_files=files,
                            roster_path=roster)
        return design / "manifest.json", manifest, files

    def test_final_budget_and_cpu_gate(self):
        final = A.specification()
        self.assertEqual(final["cfg"], dict(d=256, layers=6, heads=8, K=4))
        self.assertEqual(final["steps"], 20000)
        self.assertEqual(final["tokens"], 8192)
        self.assertEqual(final["lengths"], {"512": .4, "1024": .6})
        self.assertEqual(final["seeds"], list(A.SEEDS))
        self.assertFalse(final["diagnostic_validation_selects"])
        with self.assertRaises(ValueError):
            A.specification(device="cpu")

    def test_shared_initialization_including_unused_heads_and_controlled_direct_head(self):
        models, identities = {}, {}
        for arm in A.ARMS:
            model, _, identity = A.initialize(arm, 8101, self.s)
            models[arm], identities[arm] = model, identity
        self.assertEqual(len({i["shared_initial_state_sha256"] for i in identities.values()}), 1)
        self.assertEqual(len({i["initial_trunk_sha256"] for i in identities.values()}), 1)
        for key, value in models["full"].state_dict().items():
            for arm in A.ARMS[1:]:
                torch.testing.assert_close(value, models[arm].state_dict()[key], rtol=0, atol=0)
        again, _, _ = A.initialize("direct", 8101, self.s)
        torch.testing.assert_close(models["direct"].head_direct.weight, again.head_direct.weight,
                                   rtol=0, atol=0)

    def test_identical_synthetic_schedule_and_batch_hashes(self):
        chains = []
        for arm in ("full", "noaux", "direct"):
            stream = A.G.OrderedBatches("broad", 8101, self.s)
            try:
                chains.append([A.batch_hash(stream.next()) for _ in range(self.s["steps"])])
            finally:
                stream.close()
        self.assertEqual(chains[0], chains[1])
        self.assertEqual(chains[0], chains[2])

    def test_historical_npz_cutoff_percent_conversion_and_unused_rv(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest, files = self._freeze(temporary)
            data = A.load_historical(path)
            self.assertEqual(manifest["input_contract"]["npz_keys_read"], ["date", "returns"])
            self.assertEqual(len(data["AAA"]["returns"]), 80)
            self.assertAlmostEqual(data["AAA"]["returns"][0], -3.0)
            batch = A.historical_batch(data, 501, 40, 16)
            self.assertNotIn("target_logv", batch)
            self.assertTrue(all(d["last_target_date"] <= A.CUTOFF for d in batch["historical_draws"]))
            # Whole originals are sealed, including unused future/source keys.
            with files["AAA"].open("ab") as source:
                source.write(b"changed")
            with self.assertRaises(RuntimeError):
                A.verify(path)

    def test_historical_target_uses_preceding_prefix_scale(self):
        raw = np.array([0., 2., 4., 8.])
        data = {"AAA": dict(returns=raw, date=np.array(["2010-01-01", "2010-01-02",
                                                       "2010-01-03", "2010-01-04"]))}
        batch = A.historical_batch(data, 50, 1, 4)
        np.testing.assert_allclose(batch["target_r"][0], [2., 2., 8 / np.sqrt(10.)], rtol=1e-6)
        np.testing.assert_allclose(batch["output_scale"][0], [1., 2., np.sqrt(10.)])
        unsafe = {"AAA": dict(returns=raw, date=np.array(["2010-01-01", "2010-01-02",
                                                        "2010-01-03", "2011-01-04"]))}
        with self.assertRaises(ValueError):
            A.historical_batch(unsafe, 50, 1, 4)

    def test_full_auxiliary_is_actual_gaussian_nll_and_noaux_needs_no_label(self):
        model, _, _ = A.initialize("full", 8101, self.s)
        batch = A.synthetic_batch(811, 2, 8)
        bt = T.to_device(batch, "cpu")
        prediction = model(bt["feats"], bt["base"])
        reference_nll, reference_aux = causal.losses(prediction, bt)
        objective, components = A.objective("full", prediction, bt)
        torch.testing.assert_close(components["aux_gaussian_nll"], reference_aux)
        torch.testing.assert_close(objective, reference_nll + .1 * reference_aux)
        observed_only = {key: value for key, value in bt.items() if key != "target_logv"}
        noaux, parts = A.objective("noaux", prediction, observed_only)
        historical, _ = A.objective("historical", prediction, observed_only)
        torch.testing.assert_close(noaux, reference_nll)
        torch.testing.assert_close(historical, reference_nll)
        self.assertEqual(set(parts), {"nll"})

    def test_direct_head_and_qlike_have_positive_outputs_and_finite_gradients(self):
        model, optimizer, _ = A.initialize("direct", 8101, self.s)
        batch = A.synthetic_batch(813, 2, 8)
        prediction = model(torch.from_numpy(batch["feats"]), torch.from_numpy(batch["base"]))
        self.assertTrue(bool(torch.isfinite(prediction["h"]).all()))
        self.assertTrue(bool((prediction["h"] > 0).all()))
        event = A.update("direct", model, optimizer, batch, 0, self.s)
        self.assertTrue(np.isfinite(list(event.values())).all())
        self.assertIsNone(model.head_lat.weight.grad)
        self.assertIsNone(model.head_mix.weight.grad)
        self.assertIsNotNone(model.head_direct.weight.grad)

    def test_predict_windows_raw_moments_nll_and_target_causality(self):
        model, _, _ = A.initialize("noaux", 8101, self.s)
        histories = [np.linspace(.3, 2.1, 8), np.linspace(-1.4, -.2, 16)]
        targets = np.array([.7, -.5])
        together = A.predict_windows(model, histories, targets=targets)
        alternate_targets = A.predict_windows(model, histories, targets=targets * 20)
        for key in ("mean", "variance", "second_moment", "h"):
            np.testing.assert_array_equal(together[key], alternate_targets[key])
        self.assertFalse(np.allclose(together["native_nll"], alternate_targets["native_nll"]))
        np.testing.assert_allclose(together["variance"], together["second_moment"] - together["mean"] ** 2)
        for i, row in enumerate(histories):
            one = A.predict(model, row, target=targets[i])
            for key in together:
                self.assertAlmostEqual(one[key], together[key][i], 6)
        scaled = A.predict_windows(model, [row * 7 for row in histories], targets=targets * 7)
        np.testing.assert_allclose(scaled["mean"], together["mean"] * 7, rtol=2e-5)
        np.testing.assert_allclose(scaled["variance"], together["variance"] * 49, rtol=2e-5)
        np.testing.assert_allclose(scaled["native_nll"], together["native_nll"] + np.log(7), rtol=2e-5)
        direct, _, _ = A.initialize("direct", 8101, self.s)
        direct_result = A.predict_windows(direct, histories, targets=targets)
        self.assertEqual(set(direct_result), {"h", "native_nll"})
        self.assertIsNone(direct_result["native_nll"])

    def test_predict_uses_exact_last_1024_history(self):
        model, _, _ = A.initialize("noaux", 8101, self.s)
        history = np.random.default_rng(44).standard_normal(1030)
        changed = history.copy()
        changed[:6] *= 1000
        full, limited, perturbed = [A.predict(model, row) for row in (history, history[-1024:], changed)]
        self.assertEqual(full, limited)
        self.assertEqual(full, perturbed)

    def test_narrow_large_mean_mixture_has_stable_positive_central_variance(self):
        model, _, _ = A.initialize("noaux", 8101, self.s)
        components = model.cfg["K"]
        with torch.no_grad():
            model.head_mix.weight.zero_()
            model.head_mix.bias.zero_()
            model.head_mix.bias[components:2 * components].fill_(1e8)
            model.head_mix.bias[2 * components:3 * components].fill_(-6.)
            model.head_mix.bias[3 * components:].fill_(1.4)
        result = A.predict(model, np.ones(8), target=1e8)
        self.assertAlmostEqual(result["mean"], 1e8)
        nu = 2.1 + np.exp(np.float32(1.4))
        expected = np.exp(-12.) * nu / (nu - 2)
        self.assertAlmostEqual(result["variance"], expected, 10)
        self.assertLess(result["native_nll"], 0)  # Finite NLL need not be positive.

    def test_cpu_tiny_end_to_end_four_arms_and_complete_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self._freeze(temporary)
            out = Path(temporary) / "training"
            records = [A.train(path, out, arm, 8101) for arm in A.ARMS]
            self.assertTrue(all(row["status"] == "complete" for row in records))
            self.assertTrue(all(row["smoke_unpromotable"] for row in records))
            self.assertTrue(all(row["input_positions"] == 64 for row in records))
            synthetic = [r for r in records if r["arm"] != "historical"]
            self.assertEqual(len({r["batch_chain_sha256"] for r in synthetic}), 1)
            self.assertEqual(len({r["initial_trunk_sha256"] for r in records}), 1)
            historical = next(r for r in records if r["arm"] == "historical")
            self.assertEqual(historical["historical_repeat_draws"],
                             historical["historical_draws"] - historical["historical_unique_windows"])
            self.assertEqual(historical["historical_repeated_input_positions"],
                             historical["input_positions"] - historical["historical_unique_observations"])
            for arm in A.ARMS:
                checkpoint = out / "runs" / (arm + "_8101") / "final.pt"
                loaded = A.load_model(checkpoint, path, "cpu")
                self.assertTrue(np.isfinite(A.predict(loaded, np.linspace(.1, 1., 8))["h"]))
            status = out / "runs/full_8101/status.json"
            before = status.read_bytes()
            with self.assertRaises(RuntimeError):
                A.train(path, out, "full", 8101, resume=True)
            self.assertEqual(status.read_bytes(), before)

    def test_strict_resume_preserves_failure_and_matches_uninterrupted_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self._freeze(temporary)
            interrupted = Path(temporary) / "interrupted"
            original_update = A.update
            count = [0]

            def stop_second(*args, **kwargs):
                count[0] += 1
                if count[0] == 2:
                    raise RuntimeError("controlled test interruption")
                return original_update(*args, **kwargs)

            with mock.patch.object(A, "update", side_effect=stop_second):
                with self.assertRaises(RuntimeError):
                    A.train(path, interrupted, "full", 8101)
            receipt = interrupted / "runs/full_8101/failure_0001.txt"
            self.assertTrue(receipt.exists())
            failed = json.loads((receipt.parent / "status.json").read_text())
            self.assertEqual(failed["completed_updates"], 1)
            resumed = A.train(path, interrupted, "full", 8101, resume=True)
            reference_out = Path(temporary) / "reference"
            reference = A.train(path, reference_out, "full", 8101)
            self.assertEqual(resumed["batch_chain_sha256"], reference["batch_chain_sha256"])
            self.assertEqual(resumed["prior_failure_receipts"], ["failure_0001.txt"])
            self.assertTrue(receipt.exists())
            a = A.load_model(receipt.parent / "final.pt", path, "cpu")
            b = A.load_model(reference_out / "runs/full_8101/final.pt", path, "cpu")
            self.assertEqual(A.M.state_hash(a), A.M.state_hash(b))

    def test_backend_setup_failure_is_preserved_without_starting_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self._freeze(temporary)
            out = Path(temporary) / "unavailable_backend"
            with mock.patch.object(A.G, "configure", side_effect=RuntimeError("backend unavailable")):
                with self.assertRaises(RuntimeError):
                    A.train(path, out, "full", 8101)
            run = out / "runs/full_8101"
            record = json.loads((run / "status.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["completed_updates"], 0)
            self.assertTrue((run / "failure_0001.txt").exists())
            self.assertFalse((run / "final.pt").exists())


if __name__ == "__main__":
    unittest.main()
