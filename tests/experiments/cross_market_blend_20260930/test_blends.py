import unittest
import numpy as np
import pandas as pd
from experiments.cross_market_blend_20260930.run_blends import (
    aligned, annotate_weights, calibrate_series, calibration_fit, helper_metadata)
from experiments.blend_rk.blend import qlike


class BlendChecks(unittest.TestCase):
    def test_future_labels_do_not_change_earlier_forecasts(self):
        dates = np.arange(160)
        keys = [('M', int(i)) for i in dates]
        h = 1. + .2 * np.sin(dates / 7.)
        y = h * (1.1 + .2 * np.cos(dates / 9.))
        first, _ = calibrate_series(keys, dates, h, y)
        changed = y.copy()
        changed[100:] *= 100.
        second, _ = calibrate_series(keys, dates, h, changed)
        np.testing.assert_array_equal(first[:100], second[:100])

    def test_burn_in_and_metadata(self):
        d = np.arange(100)
        h = np.linspace(.5, 2., len(d))
        y = 1.4 * h
        result, weights = calibrate_series(list(zip(['M'] * len(d), d)), d, h, y)
        np.testing.assert_array_equal(result[:60], h[:60])
        calendar = [f'{i:04d}' for i in d]
        annotate_weights(weights, calendar, d, h, None, y)
        self.assertEqual(weights[3]['latest_fit_target_date'], '0059')
        self.assertEqual(weights[3]['fit_rows'], 60)
        self.assertLessEqual(weights[3]['fit_objective'], weights[3]['starting_objective'] + 1e-8)

    def test_calibration_qualification(self):
        h = np.linspace(.3, 2., 100)
        y = 1.7 * h ** .8
        a, b = calibration_fit(np.log(h), y)
        self.assertLess(np.mean(qlike(y, np.exp(a + b * np.log(h)))), np.mean(qlike(y, h)))
        self.assertAlmostEqual(b, .8, places=4)

    def test_no_invalid_forecast_repair(self):
        with self.assertRaises(ValueError):
            calibrate_series([('M', i) for i in range(3)], np.arange(3), [1, 0, 2], [1, 1, 1])

    def test_no_missing_seed_or_duplicate(self):
        mask = pd.DataFrame({'market': ['M', 'M'], 'date': ['a', 'b']})
        f = pd.DataFrame({'market': ['M', 'M'], 'date': ['a', 'b'], 'seed': ['8101', '8101']})
        with self.assertRaises(ValueError):
            aligned(f, mask)
        f = pd.concat([f, f])
        with self.assertRaises(ValueError):
            aligned(f, mask, ('8101',))

    def test_missing_and_future_helper_dates_rejected(self):
        frame = pd.DataFrame(dict(date=['2020-01-10'], origin_date=['2020-01-09'],
                                  latest_own_input_date=['2020-01-09'],
                                  latest_foreign_input_date=['2020-01-08'],
                                  rk_forecast=[1.], foreign_count=[27],
                                  train_end=['2017-12-31'], validation_end=['2018-12-31']))
        helper_metadata(frame)
        for field in ('date', 'origin_date', 'latest_own_input_date', 'latest_foreign_input_date'):
            changed = frame.copy(); changed[field] = None
            with self.assertRaises(ValueError):
                helper_metadata(changed)
        changed = frame.copy(); changed.latest_foreign_input_date = changed.origin_date
        with self.assertRaises(ValueError):
            helper_metadata(changed)


if __name__ == '__main__':
    unittest.main()
