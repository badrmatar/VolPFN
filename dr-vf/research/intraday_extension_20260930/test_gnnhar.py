"""Meaningful architecture, calendar-causality and strict-coverage checks."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from research.intraday_extension_20260930 import gnnhar as G


class GNNHARChecks(unittest.TestCase):
    def test_selected_forward_is_native_graph_forward(self):
        torch.manual_seed(19)
        model = G.GNNHAR1L(9)
        x = torch.rand(7, 4, 3)
        adjacency = torch.tensor([[0., .3, .2, 0.], [.3, 0., 0., .8], [.2, 0., 0., .5], [0., .8, .5, 0.]])
        native = model(x, adjacency)
        aggregate = torch.matmul(adjacency, x)
        for node in range(4):
            selected = model.forward_selected(x[:, node], aggregate[:, node])
            torch.testing.assert_close(selected, native[:, node], rtol=2e-6, atol=2e-7)

    def test_prior_features_no_backfill_or_target_use(self):
        values = np.arange(1., 41.)
        values[:2] = np.nan; values[25] = np.nan
        features, counts = G.prior_features(values)
        self.assertTrue(np.isnan(features[21]).any())
        self.assertEqual(features[25, 0], values[24])
        self.assertEqual(counts[25], 1)
        changed = values.copy(); changed[30:] = 1e8
        other, _ = G.prior_features(changed)
        np.testing.assert_array_equal(features[:30], other[:30])
        self.assertTrue(np.isnan(values[25]))  # Target array remains missing.

    @staticmethod
    def panel():
        # B has distinct holidays: calendar labels, not row numbers, govern access.
        dates = pd.date_range("2017-01-01", periods=90).strftime("%Y-%m-%d").to_numpy(dtype="U10")
        result = {}
        for market, keep in [("A", np.ones(len(dates), bool)), ("B", np.arange(len(dates)) % 3 != 0)]:
            chosen = dates[keep]; values = np.arange(1., len(chosen) + 1.)
            feature, count = G.prior_features(values)
            result[market] = dict(date=chosen, rk=values, returns=np.sin(values), features=feature, imputations=count)
        return result

    def test_global_holidays_and_future_suffix_invariance(self):
        panel = self.panel(); adjacency = np.array([[0., 1.], [1., 0.]])
        index = np.array([70])
        own, neighbor, origin, latest, available, _ = G.make_features(panel, ["A", "B"], adjacency, "A", index)
        self.assertTrue(available[0]); self.assertLess(latest[0], origin[0])
        b_position = np.searchsorted(panel["B"]["date"], origin[0], side="left") - 1
        np.testing.assert_array_equal(neighbor[0], panel["B"]["features"][b_position])
        for item in panel.values():
            later = item["date"] >= origin[0]
            item["rk"][later] = 999999.
            item["features"], item["imputations"] = G.prior_features(item["rk"])
        own2, neighbor2, _, _, _, _ = G.make_features(panel, ["A", "B"], adjacency, "A", index)
        # Own origin is allowed; mutation at that origin changes own branch.
        self.assertFalse(np.array_equal(own, own2))
        np.testing.assert_array_equal(neighbor, neighbor2)

    def test_qlike_matches_author_denominator(self):
        target = torch.tensor([1., 2.]); forecast = torch.tensor([0., .5])
        ratio = target / (forecast + 1e-4)
        torch.testing.assert_close(G.author_qlike(target, forecast), (ratio - torch.log(ratio)).mean())

    def test_graph_never_uses_validation_or_test_returns(self):
        dates = pd.date_range("2017-01-01", periods=500).strftime("%Y-%m-%d").to_numpy(dtype="U10")
        panel = {m: dict(date=dates, returns=np.arange(500, dtype=float)) for m in ["A", "B"]}
        for item in panel.values():
            item["returns"][dates > "2017-12-31"] = 999999.
        fit = SimpleNamespace(precision_=np.eye(2), alpha_=.1, n_iter_=2,
            max_iter=100, tol=1e-4, costs_=[(1., 1e-6)])
        with patch("sklearn.covariance.GraphicalLassoCV") as factory:
            factory.return_value.fit.return_value = fit
            _, receipt = G.training_graph(panel, ["A", "B"], "2017-12-31")
            used = factory.return_value.fit.call_args.args[0]
        self.assertEqual(len(used), 365)
        self.assertLess(used.max(), 999999.)
        self.assertEqual(receipt["last_date"], "2017-12-31")

    def test_missing_seed_or_row_is_rejected(self):
        mask = pd.DataFrame({"market": ["A", "B"], "date": ["2019-01-01", "2019-01-02"]})
        rows = pd.concat([mask.assign(seed=seed, rk_forecast=1.) for seed in G.SEEDS], ignore_index=True)
        G.verify_coverage(rows, mask)
        with self.assertRaises(ValueError): G.verify_coverage(rows.iloc[:-1], mask)
        with self.assertRaises(ValueError): G.verify_coverage(rows[rows.seed != G.SEEDS[0]], mask)
        with self.assertRaises(ValueError): G.verify_coverage(pd.concat([rows, rows.iloc[:1]]), mask)

    def test_evaluation_target_mutation_cannot_change_training_or_prediction(self):
        rng = np.random.default_rng(7)
        records = pd.DataFrame(dict(market=["A"] * 10,
            date=[f"2017-01-{i + 2:02d}" for i in range(10)], index=np.arange(1, 11),
            origin_date=[f"2017-01-{i + 1:02d}" for i in range(10)],
            latest_input_date=[f"2017-01-{i + 1:02d}" for i in range(10)],
            latest_neighbor_input_date=["2016-12-31"] * 10,
            input_imputations=[0] * 10, split=["train"] * 5 + ["validation"] * 3 + ["evaluation"] * 2))
        data = SimpleNamespace(own=rng.uniform(.1, 2., (10, 3)).astype(np.float32),
            neighbor=rng.uniform(.1, 2., (10, 3)).astype(np.float32),
            target=rng.uniform(.1, 2., 10).astype(np.float32), records=records,
            evaluation_positions=np.array([8, 9]))
        # An attainable positive target makes the TRAIN gate meaningful in this
        # tiny causality test without spending a large optimization budget.
        torch.manual_seed(8101)
        teacher = G.GNNHAR1L(9)
        with torch.no_grad():
            data.target = (teacher.forward_selected(torch.from_numpy(data.own),
                torch.from_numpy(data.neighbor)) + G.EPSILON).numpy()
        first, predictions, _ = G.train_seed(data, 8101, epochs=3)
        data.target[8:] = np.nan
        other, predictions2, _ = G.train_seed(data, 8101, epochs=3)
        for name, parameter in first.state_dict().items():
            torch.testing.assert_close(parameter, other.state_dict()[name], rtol=0, atol=0)
        np.testing.assert_array_equal(predictions.rk_forecast, predictions2.rk_forecast)

    def test_checkpoint_gate_rejects_better_validation_with_bad_training_fit(self):
        target = np.array([1., 4.])
        mean = target.mean(); ratio = target / (mean + G.EPSILON)
        baseline = float(np.mean(ratio - np.log(ratio)))
        good = G.checkpoint_quality(target, target - G.EPSILON, baseline)
        bad = G.checkpoint_quality(target, np.zeros(2), baseline)
        selector = G.QualifiedCheckpoint()
        selector.consider(1, bad, .1, lambda: {"model": "invalid"})
        self.assertIsNone(selector.best_state)
        self.assertEqual(selector.stale_epochs, 0)  # No patience before qualification.
        selector.consider(2, good, 1.2, lambda: {"model": "qualified"})
        selected = selector.consider(3, bad, .01, lambda: {"model": "invalid"})
        self.assertFalse(selected)
        self.assertEqual(selector.best_state, {"model": "qualified"})
        self.assertEqual(selector.best_epoch, 2)
        self.assertEqual(selector.first_qualified_epoch, 2)
        self.assertEqual(selector.stale_epochs, 1)
        self.assertFalse(bad["train_gate_pass"])
        nonfinite = G.checkpoint_quality(target, np.array([np.nan, 1.]), baseline)
        self.assertFalse(nonfinite["train_gate_pass"])


if __name__ == "__main__":
    unittest.main()
