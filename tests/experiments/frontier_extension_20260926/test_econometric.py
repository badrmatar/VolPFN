"""Synthetic-only forecast timing, state, moment and failure tests."""
import copy
import json
import unittest
from unittest import mock

import numpy as np
from scipy.special import ndtr

from experiments.frontier_extension_20260926 import econometric as e


def synthetic_rk(n=240, seed=39021):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    shock = rng.normal(0., .23, n)
    y[:22] = rng.normal(0., .2, 22)
    for t in range(22, n):
        y[t] = (.03 + .45 * y[t - 1] + .2 * y[t - 5:t].mean()
                + .15 * y[t - 22:t].mean() + shock[t] - .35 * shock[t - 1])
    return np.exp(y)


def log_state(origin=122):
    return e._LogState(np.array([.05, .4, .2, .15]), -.35, .06, 0., 0., .06,
                       .001, 100., 1., origin, 22, 100)


class EconometricTests(unittest.TestCase):
    def test_features_average_transformed_observations_and_only_use_past(self):
        rk = np.arange(1., 45.) ** 2
        rk[21:23] = np.nan
        x = e._features(rk, "logrhar")
        filled = rk.copy()
        filled[21:23] = rk[20]
        expected = np.r_[1., np.log(filled[22]), np.log(filled[18:23]).mean(),
                         np.log(filled[1:23]).mean()]
        np.testing.assert_allclose(x[23], expected, rtol=0, atol=1e-15)
        self.assertNotEqual(x[23, 3], np.log(filled[1:23].mean()))
        changed = rk.copy()
        changed[23:] = 1e100
        np.testing.assert_array_equal(x[:24], e._features(changed, "logrhar")[:24])
        sx = e._features(rk, "mshar_sqrt")
        self.assertEqual(sx[23, 2], np.sqrt(filled[18:23]).mean())

    def test_MA_filter_and_forecast_match_independent_statsmodels_Kalman(self):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        rk = synthetic_rk(120)
        rk[65] = np.nan
        x = e._features(rk, "logrhar")
        state = log_state()
        endog = np.log(rk[22:100])
        model = SARIMAX(endog, exog=x[22:100], order=(0, 0, 1), trend="n",
                        initialization="stationary", enforce_invertibility=True)
        fixed = model.filter(np.r_[state.beta, state.theta, state.sigma2])
        for value, design in zip(rk[22:100], x[22:100]):
            state.observe(value, design)
        expected = fixed.get_forecast(1, exog=x[100:101])
        mu, variance = state.log_moments(x[100])
        self.assertAlmostEqual(mu, float(expected.predicted_mean[0]), places=11)
        self.assertAlmostEqual(variance, float(expected.var_pred_mean[0]), places=11)
        forecast, detail = state.predict(x[100])
        self.assertAlmostEqual(forecast, np.exp(mu + .5 * variance), places=12)
        self.assertGreater(variance, 0.)

    def test_MS_Hamilton_filter_matches_library_and_moment_is_not_squared_mean(self):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        rk = synthetic_rk(100)
        x = e._features(rk, "mshar_sqrt")
        pars = np.array([.92, .15, .1, .3, .45, .55, .12, .08, .03, .09])
        model = MarkovRegression(np.sqrt(rk[22:90]), 2, trend="c", exog=x[22:90, 1:],
                                 switching_trend=True, switching_exog=[True, False, False],
                                 switching_variance=True)
        reference = model.filter(pars)
        transition = np.array([[.92, .15], [.08, .85]])
        state = e._MSState(np.array([[.1, .45, .12, .08], [.3, .55, .12, .08]]),
                           np.array([.03, .09]), transition, np.array([.15, .08]) / .23,
                           1., 90, 22, 68)
        for value, design in zip(rk[22:90], x[22:90]):
            state.observe(value, design)
        np.testing.assert_allclose(state.filtered,
                                   reference.filter_results.filtered_marginal_probabilities[:, -1],
                                   rtol=1e-11, atol=1e-12)
        p, means = state.components(x[90])
        expected = np.sum(p * (means ** 2 + state.sigma2))
        value, detail = state.predict(x[90])
        self.assertAlmostEqual(value, expected, places=12)
        self.assertGreater(value, float(p @ means) ** 2)
        self.assertAlmostEqual(detail["negative_sd_mass"],
                               float(p @ ndtr(-means / np.sqrt(state.sigma2))), places=12)
        before = state.filtered.copy()
        state.observe(np.nan, x[90])
        np.testing.assert_allclose(state.filtered, transition @ before, rtol=0, atol=1e-15)

    def test_actual_fit_future_and_endpoint_invariance(self):
        rk = synthetic_rk(160)
        changed = rk.copy()
        changed[134:] = np.inf
        for model in e.MODELS:
            config = e.EconometricConfig(model, training_rows=100, maxiter=200)
            with self.subTest(model=model):
                before = e.forecast_series(rk, np.array([122, 125, 134]), config)
                after = e.forecast_series(changed, np.array([122, 125, 134]), config)
                self.assertEqual(before, after)
                self.assertTrue(before[0]["metadata"]["fit_audit"]["accepted"], before[0])
                json.dumps(before, allow_nan=False)

    def test_sparse_requests_keep_clock_and_all_intermediate_fit_audits(self):
        rk = synthetic_rk(180)
        config = e.EconometricConfig("loghar_meanlog", training_rows=100, refit_every=5)
        sparse = e.forecast_series(rk, np.array([122, 124, 139]), config)
        dense = e.forecast_series(rk, np.arange(122, 140), config)
        self.assertEqual([r["variance"] for r in sparse],
                         [dense[t - 122]["variance"] for t in (122, 124, 139)])
        self.assertEqual([event["origin"] for row in sparse
                          for event in row["metadata"]["fit_events"]], [122, 127, 132, 137])

    def test_failed_refit_carries_parameters_and_updates_MA_state(self):
        rk = synthetic_rk(150)
        config = e.EconometricConfig("logrhar", training_rows=100, refit_every=2,
                                     insanity_filter=False)
        accepted = log_state()
        def fitting(values, features, t, cfg):
            return ((copy.deepcopy(accepted), {"accepted": True}) if t == 122
                    else (None, {"accepted": False, "reason": "forced"}))
        with mock.patch.object(e, "_fit", side_effect=fitting):
            rows = e.forecast_series(rk, np.arange(122, 127), config)
        expected = copy.deepcopy(accepted)
        x = e._features(rk, "logrhar")
        for row in rows:
            t = row["origin"]
            self.assertAlmostEqual(row["variance"], expected.predict(x[t])[0], places=12)
            expected.observe(rk[t], x[t])
        self.assertFalse(rows[0]["metadata"]["substituted"])
        self.assertTrue(rows[2]["metadata"]["substituted"])
        self.assertNotEqual(rows[2]["variance"], rows[3]["variance"])

    def test_exact_valid_row_window_retains_interior_missing_sessions(self):
        rk = synthetic_rk(180)
        rk[[60, 70]] = np.nan
        x = e._features(rk, "logrhar")
        cfg = e.EconometricConfig("logrhar", training_rows=100, maxiter=200)
        start, observed, design = e._window(rk, x, 140, cfg)
        self.assertEqual(start, 38)
        self.assertEqual(len(observed), 102)
        self.assertEqual(int(observed.sum()), 100)
        state, audit = e._fit_log(rk, x, 140, cfg)
        self.assertTrue(audit["accepted"], audit)
        self.assertEqual(audit["missing_emissions"], 2)
        self.assertTrue(np.isfinite(state.predict(x[140])[0]))

    def test_nested_theta_zero_and_log_unit_equivariance(self):
        state = log_state()
        state.theta = 0.
        x = np.array([1., .3, .4, .5])
        value, detail = state.predict(x)
        self.assertAlmostEqual(value, np.exp(x @ state.beta + .5 * state.sigma2), places=12)
        rk = synthetic_rk(150)
        config = e.EconometricConfig("loghar_meanlog", training_rows=100)
        raw = e.forecast_series(rk, np.array([122, 135]), config)
        scaled = e.forecast_series(10000. * rk, np.array([122, 135]), config)
        np.testing.assert_allclose([r["variance"] for r in scaled],
                                   10000. * np.array([r["variance"] for r in raw]), rtol=1e-11)

    def test_actual_MS_fit_with_missing_emissions_and_postfit_gap(self):
        rk = synthetic_rk(170)
        rk[[60, 70, 141]] = np.nan
        config = e.EconometricConfig("mshar_sqrt", training_rows=100, maxiter=200)
        rows = e.forecast_series(rk, np.array([140, 143]), config)
        audit = rows[0]["metadata"]["fit_audit"]
        self.assertTrue(audit["accepted"], audit)
        self.assertEqual(audit["missing_emissions"], 2)
        self.assertEqual(audit["training_calendar_rows"], 102)
        self.assertIn({"observed_index": 141, "action": "missing_target_prediction_only"},
                      rows[1]["metadata"]["state_events"])
        self.assertTrue(np.isfinite(rows[1]["variance"]))
        json.dumps(rows, allow_nan=False)

    def test_declared_range_safeguard_and_fallback_keep_dates(self):
        rk = synthetic_rk(150)
        config = e.EconometricConfig("logrhar", training_rows=100)
        state = log_state()
        state.beta[:] = [10., 0., 0., 0.]
        with mock.patch.object(e, "_fit", return_value=(state, {"accepted": True})):
            rows = e.forecast_series(rk, np.array([122]), config)
        self.assertEqual(rows[0]["variance"], state.training_mean)
        self.assertTrue(rows[0]["metadata"]["range_replaced"])
        with mock.patch.object(e, "_fit", return_value=(None, {"accepted": False})):
            rows = e.forecast_series(rk, np.array([122, 123, 126]), config)
        self.assertEqual([r["origin"] for r in rows], [122, 123, 126])
        self.assertTrue(all(r["metadata"]["substituted"] for r in rows))
        self.assertAlmostEqual(rows[0]["variance"], rk[:122].mean(), places=12)

    def test_input_validation(self):
        rk = synthetic_rk()
        cfg = e.EconometricConfig("logrhar", training_rows=100)
        for origins in (np.array([122., 123.]), np.array([123, 122]), np.array([122, 122]),
                        np.array([21]), np.array([300])):
            with self.assertRaises(ValueError):
                e.forecast_series(rk, origins, cfg)


if __name__ == "__main__":
    unittest.main()
