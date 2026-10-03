"""Synthetic-only inference/coverage tests; no retained market scoring."""
import copy
from datetime import date, timedelta
import json
import unittest

import numpy as np

from research.frontier_extension_20260926.statistics import (
    CoverageError, VOLPFN_SEEDS, _family_adjust, analyse, coverage_audit, qlike,
)


def fixture(days=30):
    specs = {
        "volpfn": {"family": "V", "output_adapter": "native", "seeds": list(VOLPFN_SEEDS)},
        "control": {"family": "E", "output_adapter": "native", "seeds": [None]},
        "rv": {"family": "R", "output_adapter": "zero mean", "seeds": [None]},
        "logmean_diagnostic": {"family": "R", "output_adapter": "diagnostic", "seeds": [None], "primary": False},
    }
    origins = []
    for phase, count, start in (("calibration", 24, date(2012, 1, 1)),
                                ("evaluation", days, date(2019, 1, 1))):
        for market in ("SPX", "FTSE", "N225"):
            for day in range(count):
                # Uneven calendar lengths distinguish equal-market from pooled weighting.
                if phase == "evaluation" and market == "N225" and day >= days // 2:
                    continue
                origins.append(dict(phase=phase, market=market, date=str(start + timedelta(days=day)),
                    r2=1., rk=1., primary_eligible=True))
    bundle = {"rows": origins, "markets": ["FTSE", "N225", "SPX"],
              "settings": {"resamples": 199, "blocks": [3, 2], "bootstrap_seed": 17}}
    records = []
    for model, spec in specs.items():
        for seed in spec["seeds"]:
            for origin in origins:
                h = 1.
                if origin["phase"] == "evaluation":
                    h = 1 + .2 * (seed - 8101) if model == "volpfn" else {"control": 5., "rv": 7., "logmean_diagnostic": 1.}[model]
                records.append(dict(origin, model=model, seed=seed, mean=0., variance=h,
                    second_moment=h, substituted=False, floor_applied=False))
    return records, specs, bundle


def run(records, specs, bundle, **kwargs):
    return analyse(records, specs, bundle=bundle, **kwargs)


class StatisticsTests(unittest.TestCase):
    def test_zero_safe_loss_and_invalid_values(self):
        np.testing.assert_allclose(qlike([0, 1], [2, 1]), [np.log(2), 1.])
        for y, h in ((-1, 1), (1, 0), (1, float("nan"))):
            with self.assertRaises(ValueError):
                qlike(y, h)

    def test_joint_family_dynamic_diagnostics_excluded_and_sign(self):
        rows, specs, bundle = fixture()
        result = run(rows, specs, bundle)
        self.assertEqual(result["settings"]["primary_family_slots"], 4)
        self.assertEqual(len(result["contrasts"]), 8)
        self.assertEqual({r["model"] for r in result["contrasts"]}, {"control", "rv"})
        self.assertTrue(all(r["gain"] > 0 for r in result["contrasts"]))
        self.assertTrue(all(result["comparison_support"].values()))
        diagnostic = next(r for r in result["leaderboards"]["primary"]["rk"] if not r["primary"])
        self.assertIsNone(diagnostic["rank"])
        # Reverse superiority: one-sided positive-gain test must not endorse losses.
        for r in rows:
            if r["phase"] == "evaluation" and r["model"] == "control":
                r["variance"] = r["second_moment"] = 1.
        result = run(rows, specs, bundle)
        self.assertFalse(result["comparison_support"]["control"])
        self.assertTrue(all(r["p_raw"] == 1 for r in result["contrasts"] if r["model"] == "control"))
        self.assertTrue(all(r["verdict"] == "negative_gain_family_interval_below_zero"
                            for r in result["contrasts"] if r["model"] == "control"))

    def test_negative_estimate_with_crossing_family_interval_is_not_reverse_claim(self):
        rows = [dict(model="a", proxy=p, p_raw=1., gain=-.1,
                     family_interval95=[-.3, .1], all_reference_seeds_positive=False)
                for p in ("r2", "rk")]
        _family_adjust(rows, 2)
        self.assertTrue(all(r["verdict"] == "negative_estimate" for r in rows))
        self.assertTrue(all(r["p_holm"] == 1. for r in rows))
        self.assertFalse(any(r["comparison_supported_both_proxies"] for r in rows))

    def test_raw_calibrated_gain_direction_change_is_explicit(self):
        rows, specs, bundle = fixture()
        for r in rows:
            if r["model"] == "control":
                # c=1/5 repairs the control from raw5 to calibrated1, reversing
                # its comparison with the heterogeneous VolPFN seed losses.
                r["variance"] = r["second_moment"] = 5.
        result = run(rows, specs, bundle)
        for effect in result["calibration_effects"]:
            if effect["model"] == "control":
                self.assertGreater(effect["raw_gain"], 0)
                self.assertLess(effect["calibrated_gain"], 0)
                self.assertTrue(effect["gain_direction_changed"])
                self.assertLess(effect["calibrated_minus_raw_loss"], 0)

    def test_seed_mean_loss_is_not_ensemble_and_equal_market_weighting(self):
        rows, specs, bundle = fixture()
        for r in rows:
            if r["model"] == "control" and r["phase"] == "evaluation" and r["market"] == "N225":
                r["variance"] = r["second_moment"] = 12.
        result = run(rows, specs, bundle)
        table = {r["model"]: r for r in result["leaderboards"]["primary"]["r2"]}
        actual = table["volpfn"]["loss"]
        expected = np.mean([qlike(1., 1 + .2 * i) for i in range(5)])
        self.assertAlmostEqual(actual, expected)
        self.assertGreater(abs(actual - qlike(1., 1.4)), .005)
        expected_control = (2 * qlike(1., 5.) + qlike(1., 12.)) / 3
        self.assertAlmostEqual(table["control"]["loss"], expected_control)

    def test_calibration_distinct_moments_and_json(self):
        rows, specs, bundle = fixture()
        for r in rows:
            if r["model"] == "control" and r["phase"] == "calibration":
                r["variance"] = r["second_moment"] = .5
            if r["model"] == "control" and r["phase"] == "evaluation":
                r.update(mean=2., variance=1., second_moment=5.)
        result = run(rows, specs, bundle)
        self.assertEqual(result["calibration"]["control"]["none"]["scalar"], 2.)
        for proxy, h in (("r2", 10.), ("rk", 2.)):
            observed = next(r["loss"] for r in result["leaderboards"]["primary"][proxy] if r["model"] == "control")
            self.assertAlmostEqual(observed, qlike(1., h))
        json.dumps(result, allow_nan=False)

    def test_exact_bundle_mask_even_if_every_model_drops_same_origin(self):
        rows, specs, bundle = fixture()
        key = ("SPX", "2019-01-01")
        reduced = [r for r in rows if (r["market"], r["date"]) != key]
        with self.assertRaises(CoverageError) as context:
            run(reduced, specs, bundle)
        self.assertEqual(context.exception.audit["models"]["volpfn"]["failed_or_incomplete_seed_count"], 5)
        failures = [rows[:-1], rows + [rows[0]], [r for r in rows if r["seed"] != 8105]]
        for broken in failures:
            with self.assertRaises(CoverageError):
                run(broken, specs, bundle)
        audit = coverage_audit([], specs, bundle)
        self.assertFalse(audit["complete"])

    def test_target_mask_counts_floors_and_substitutions_are_separate(self):
        rows, specs, bundle = fixture()
        for r in bundle["rows"] + rows:
            if r["phase"] == "evaluation" and r["date"] == "2019-01-01":
                r["rk"] = None; r["primary_eligible"] = False
            if r["phase"] == "evaluation" and r["date"] == "2019-01-02":
                r["r2"] = 0.
        for r in rows:
            if r["model"] == "rv" and r["phase"] == "evaluation" and r["market"] == "SPX":
                r["floor_applied"] = r["date"] <= "2019-01-03"
                r["substituted"] = r["date"] == "2019-01-03"
        result = run(rows, specs, bundle)
        self.assertEqual(result["samples"]["S_all"]["rows"], 75)
        self.assertEqual(result["samples"]["primary"]["rows"], 72)
        self.assertEqual(result["floor_counts"]["rv"]["none"]["evaluation_primary"], 2)
        self.assertEqual(result["floor_counts"]["rv"]["none"]["evaluation_S_all"], 3)
        self.assertEqual(result["substitution_counts"]["rv"]["none"]["evaluation_primary"], 1)

    def test_targets_adapters_bad_seeds_and_failed_forecasts_fail_closed(self):
        rows, specs, bundle = fixture()
        for changes in (dict(rk=2.), dict(variance=0.), dict(second_moment=999.),
                        dict(forecast_failed=True), dict(substituted="false")):
            broken = copy.deepcopy(rows); broken[-1].update(changes)
            with self.assertRaises(ValueError):
                run(broken, specs, bundle)
        broken_specs = copy.deepcopy(specs); broken_specs["volpfn"]["seeds"].pop()
        with self.assertRaises(ValueError):
            run(rows, broken_specs, bundle)

    def test_holm_global_and_both_proxy_every_seed_support_gate(self):
        rows = [dict(model=m, proxy=p, p_raw=v, gain=1., family_interval95=[.1, 2.],
                     all_reference_seeds_positive=True)
                for m,p,v in (("a","r2",.001),("a","rk",.002),("b","r2",.01),("b","rk",.2))]
        _family_adjust(rows, 4)
        np.testing.assert_allclose([r["p_holm"] for r in rows], [.004, .006, .02, .2])
        self.assertEqual([r["comparison_supported_both_proxies"] for r in rows], [True, True, False, False])
        rows[0]["all_reference_seeds_positive"] = False
        _family_adjust(rows, 4)
        self.assertFalse(rows[1]["comparison_supported_both_proxies"])
        with self.assertRaises(ValueError):
            _family_adjust(rows[:-1], 4)

    def test_bootstrap_archive_counts_shared_across_all_contrasts(self):
        rows, specs, bundle = fixture(); archive = {}
        for r in rows:
            if r["model"] == "control" and r["phase"] == "evaluation":
                h = 3. + int(r["date"][-2:]) / 10 + (2 if r["market"] == "N225" else 0)
                r["variance"] = r["second_moment"] = h
        result = run(rows, specs, bundle, archive=archive)
        counts = archive["counts__block3"]
        self.assertEqual(counts.shape, (199, 30))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(199, 30))
        row = next(r for r in result["contrasts"] if r["model"] == "control" and r["proxy"] == "rk" and r["block"] == 3)
        draws = archive["draws__control__rk__block3"]
        errors = draws - row["gain"]
        self.assertEqual(row["p_raw"], (1 + np.count_nonzero(errors >= row["gain"])) / 200)
        # Independently reproduce date-count renormalization on the unequal calendar.
        row_dates = archive["primary_row_dates"]
        row_markets = archive["primary_row_markets"]
        gains = archive["gains__control__rk"]
        day_index = {d: i for i, d in enumerate(archive["primary_calendar_dates"])}
        for draw in range(10):
            market_means = []
            for market in ("SPX", "FTSE", "N225"):
                keep = row_markets == market
                weights = np.array([counts[draw, day_index[d]] for d in row_dates[keep]])
                market_means.append(np.sum(weights * gains[keep]) / weights.sum())
            self.assertAlmostEqual(draws[draw], np.mean(market_means), places=13)

    def test_actual_seed_survival_blocks_mean_gain_claim(self):
        rows, specs, bundle = fixture()
        for r in rows:
            if r["phase"] == "evaluation":
                if r["model"] == "volpfn":
                    h = 20. if r["seed"] == 8105 else 1.
                    r["variance"] = r["second_moment"] = h
                elif r["model"] == "control":
                    r["variance"] = r["second_moment"] = 5.
        result = run(rows, specs, bundle)
        contrasts = [r for r in result["contrasts"] if r["model"] == "control"]
        self.assertTrue(all(r["gain"] > 0 for r in contrasts))
        self.assertTrue(all(r["positive_seeds"] == 4 for r in contrasts))
        self.assertFalse(result["comparison_support"]["control"])


if __name__ == "__main__":
    unittest.main()
