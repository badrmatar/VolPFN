import unittest

import numpy as np
import pandas as pd

from research.experiment_programme_20260930 import programme as P
from research.experiment_programme_20260930.statistics import bootstrap, calendar_counts, contrast, validate_forecasts


class StrictScoringTests(unittest.TestCase):
    def fixture(self):
        dates = pd.date_range('2015-01-01', periods=20).strftime('%Y-%m-%d')
        mask = pd.DataFrame([(market, date, .3) for market in ['A', 'B'] for date in dates],
                            columns=['market', 'date', 'return'])
        rows = [(seed, row.market, row.date, row['return'], 1.) for seed in [1, 2]
                for _, row in mask.iterrows()]
        frame = pd.DataFrame(rows, columns=['seed', 'market', 'date', 'return', 'second_moment'])
        frame['qlike'] = .09
        return mask, frame

    def test_missing_seed_or_row_is_not_a_smaller_headline(self):
        mask, frame = self.fixture()
        with self.assertRaises(ValueError):
            validate_forecasts(frame.iloc[1:], mask, [1, 2])
        with self.assertRaises(ValueError):
            validate_forecasts(frame[frame.seed == 1], mask, [1, 2])

    def test_target_moment_and_score_are_checked(self):
        mask, frame = self.fixture()
        changed = frame.copy()
        changed.loc[0, 'qlike'] += .001
        with self.assertRaises(ValueError):
            validate_forecasts(changed, mask, [1, 2])
        changed = frame.copy()
        changed.loc[0, 'second_moment'] = -1
        with self.assertRaises(ValueError):
            validate_forecasts(changed, mask, [1, 2])
        changed = frame.copy()
        changed.loc[0, 'return'] = .4
        changed.loc[0, 'qlike'] = .16
        with self.assertRaises(ValueError):
            validate_forecasts(changed, mask, [1, 2])

    def test_gain_sign_and_identical_score_null(self):
        mask, reference = self.fixture()
        comparator = reference.copy()
        comparator['second_moment'] = 2.
        comparator['qlike'] = .09 / 2 + np.log(2)
        gains = contrast(reference, comparator, mask, [1, 2])
        result = bootstrap(gains, mask, [1, 2], repetitions=99, block=4, chunk_size=11)
        self.assertGreater(result['mean_gain'], 0)
        self.assertLessEqual(result['p_two_sided'], .05)
        null = bootstrap(contrast(reference, reference, mask, [1, 2]), mask, [1, 2], repetitions=99, block=4)
        self.assertEqual(null['mean_gain'], 0)
        self.assertEqual(null['p_two_sided'], 1)

    def test_resampling_conserves_calendar_count_and_is_reproducible(self):
        left = calendar_counts(13, 10, 4, np.random.default_rng(31))
        right = calendar_counts(13, 10, 4, np.random.default_rng(31))
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left.sum(axis=1), np.full(10, 13.))

    def test_real_data_masks_and_units_remain_audited(self):
        tables = P.masks()
        self.assertEqual({key: len(frame) for key, frame in tables.items()},
                         {'train': 70199, 'validation': 28086, 'evaluation': 50073})
        self.assertLessEqual(tables['train'].date.max(), '2010-12-31')
        self.assertGreaterEqual(tables['evaluation'].date.min(), '2015-01-01')
        with np.load(P.ROOT / P.DATA / 'SPX.npz', allow_pickle=False) as original:
            dates, returns, kernel = P.read_market('SPX')
            np.testing.assert_array_equal(dates, original['date'])
            np.testing.assert_array_equal(returns, original['returns'] * 100.)
            np.testing.assert_array_equal(kernel, original['rk_parzen'] * 10000.)


if __name__ == '__main__':
    unittest.main()
