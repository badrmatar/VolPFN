import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.intraday_extension_20260930.transfer_xgb import (
    INTERNAL_SCALE, causal_features, dtw_distances, fit_models, forecast_market,
    make_item, repair_predictions, select_source_data,
)


def scalar_dtw(a, b):
    """Independent scalar oracle for the standard unrestricted DTW metric."""
    table = np.full((len(a) + 1, len(b) + 1), np.inf)
    table[0, 0] = 0.
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i, j] = sum((a[i-1] - b[j-1]) ** 2) + min(
                table[i-1, j], table[i, j-1], table[i-1, j-1])
    return table[-1, -1] ** .5


def synthetic_panel():
    rng = np.random.default_rng(781)
    dates = np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-01") + np.timedelta64(180, "D")).astype("U10")
    return {name: make_item(dates, np.exp(rng.normal(0, .2, len(dates))))
            for name in ("A", "B", "C")}


class TransferXgbTests(unittest.TestCase):
    def test_exact_dtw_matches_independent_oracle(self):
        rng = np.random.default_rng(9)
        candidates, recent = rng.normal(size=(8, 6, 3)), rng.normal(size=(6, 3))
        expected = np.array([scalar_dtw(c, recent) for c in candidates])
        np.testing.assert_allclose(dtw_distances(candidates, recent), expected, rtol=1e-13)
        self.assertEqual(dtw_distances(recent[None], recent)[0], 0.)

    def test_own_features_are_prior_only_and_targets_remain_missing(self):
        rk = np.arange(1., 101.)
        rk[40] = np.nan
        x, valid, imputed = causal_features(rk)
        self.assertFalse(valid[40])
        self.assertEqual(x[41, 0], rk[39])
        self.assertEqual(imputed[41], 1)
        changed = rk.copy()
        changed[60:] = 1000.
        other, _, _ = causal_features(changed)
        np.testing.assert_array_equal(x[:61], other[:61])

    def test_source_labels_and_features_cannot_see_forecast_date(self):
        panel = synthetic_panel()
        cutoff = panel["A"]["date"][130]
        x, y, info = select_source_data(panel, "A", 130)
        self.assertLess(info["latest_source_label_date"], cutoff)
        self.assertLess(info["latest_source_input_date"], cutoff)
        changed = {}
        for name, item in panel.items():
            risk = item["rk"].copy()
            risk[item["date"] >= cutoff] *= 10000.
            changed[name] = make_item(item["date"], risk)
        other_x, other_y, other_info = select_source_data(changed, "A", 130)
        np.testing.assert_array_equal(x, other_x)
        np.testing.assert_array_equal(y, other_y)
        self.assertEqual(info, other_info)

    def test_missing_training_label_is_never_carried_into_source_labels(self):
        panel = synthetic_panel()
        risk = panel["B"]["rk"].copy()
        risk[110] = np.nan
        panel["B"] = make_item(panel["B"]["date"], risk)
        _, y, _ = select_source_data(panel, "A", 130)
        self.assertTrue(np.isfinite(y).all())
        self.assertFalse(panel["B"]["valid"][110])
        self.assertTrue(np.isfinite(panel["B"]["x"][111]).all())

    def test_author_negative_repair_and_strict_zero_policy(self):
        value, count = repair_predictions(np.array([-3., 2.]), .25)
        np.testing.assert_array_equal(value, [.25, 2.])
        self.assertEqual(count, 1)
        for bad in (0., np.nan, np.inf, -np.inf):
            with self.assertRaises(RuntimeError):
                repair_predictions(np.array([bad]), .25)
        with self.assertRaises(ValueError):
            repair_predictions(np.array([-1.]), 0.)

    def test_tree_settings_internal_units_and_five_session_refit(self):
        panel = synthetic_panel()
        item = panel["A"]
        model, minimum, diagnostics = fit_models(item["x"][22:130], item["rk"][22:130], (8101,))
        params = model[8101].get_params()
        self.assertEqual((params["max_depth"], params["n_estimators"], params["gamma"]), (5, 40, .1))
        self.assertEqual((params["learning_rate"], params["subsample"], params["seed"]), (.1, .75, 8101))
        self.assertEqual(diagnostics["8101"]["tree_count"], 40)
        self.assertEqual(diagnostics["8101"]["objective"], "reg:squarederror")
        self.assertTrue(np.isfinite(diagnostics["8101"]["training_mse_native_rk_squared"]))
        self.assertAlmostEqual(minimum / INTERNAL_SCALE, item["x"][22:130, 0].min(), places=6)
        result = forecast_market(panel, "A", item["date"][130:138], seeds=(8101,))
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual([f["forecast_block_start"] for f in result["fits"]], item["date"][[130, 135]].tolist())
        for row in result["rows"]:
            self.assertGreater(row["rk_forecast"], 0.)
            self.assertLess(row["fit_end"], row["fit_block_start"])
            self.assertLess(row["latest_input_date"], row["date"])

    def test_checkpoint_resume_preserves_exact_issued_forecasts(self):
        panel = synthetic_panel()
        dates = panel["A"]["date"][130:143]
        complete = forecast_market(panel, "A", dates, seeds=(8101, 8102))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            partial = forecast_market(panel, "A", dates, seeds=(8101, 8102),
                                      pilot_refits=1, checkpoint=path, checkpoint_refits=1)
            self.assertEqual(len(partial["rows"]), 10)
            resumed = forecast_market(panel, "A", dates, seeds=(8101, 8102),
                                      checkpoint=path, checkpoint_refits=1)
        self.assertEqual(complete["rows"], resumed["rows"])
        self.assertEqual([f["selected_counts"] for f in complete["fits"]],
                         [f["selected_counts"] for f in resumed["fits"]])


if __name__ == "__main__":
    unittest.main()
