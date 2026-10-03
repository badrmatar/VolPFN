"""Meaningful safety tests for unit, target and forecast-functional mistakes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.harmonized_leaderboard_20260930.run import align, score, summarize, run
from experiments.combined_leaderboard.statistics import qlike


class HarmonizationTests(unittest.TestCase):
    def setUp(self):
        self.mask = pd.DataFrame({'market': ['A', 'A', 'B'],
            'date': ['2020-01-01', '2020-01-02', '2020-01-01'],
            'index': [100, 101, 100], 'r2': [0., 4., 9.], 'rk': [1., 2., 3.]})
        self.frame = self.mask.copy()
        self.frame['seed'] = '8101'
        self.frame['mean'] = 2.
        self.frame['variance'] = 1.
        self.frame['second_moment'] = 5.

    def test_nonzero_mean_keeps_two_denominators(self):
        f = score(align(self.frame, self.mask, ['8101']))
        np.testing.assert_allclose(f.r2_qlike, self.mask.r2 / 5 + np.log(5))
        np.testing.assert_allclose(f.rk_qlike, self.mask.rk)
        self.assertFalse(np.allclose(f.rk_qlike, f.rk_m2_diagnostic_qlike))

    def test_direct_does_not_invent_central_variance(self):
        f = self.frame.copy()
        f[['mean', 'variance']] = np.nan
        scored = score(f, direct=True)
        self.assertTrue(scored.rk_qlike.isna().all())
        self.assertTrue(np.isfinite(scored.rk_m2_diagnostic_qlike).all())

    def test_wrong_units_wrong_keys_and_missing_seed_fail(self):
        for field, factor in [('r2', 10000), ('rk', .0001)]:
            f = self.frame.copy()
            f[field] *= factor
            with self.assertRaises(AssertionError):
                align(f, self.mask, ['8101'])
        for f in [self.frame.iloc[:-1], pd.concat([self.frame, self.frame.iloc[:1]]),
                  self.frame.assign(date=['2020-01-01', '2020-01-03', '2020-01-01'])]:
            with self.assertRaises(ValueError):
                align(f, self.mask, ['8101'])
        with self.assertRaises(ValueError):
            align(self.frame, self.mask, ['8101', '8102'])

    def test_incoherent_native_moments_and_index_fail(self):
        with self.assertRaises(AssertionError):
            score(self.frame.assign(second_moment=1.))
        with self.assertRaises(AssertionError):
            align(self.frame.assign(index=999), self.mask, ['8101'])

    def test_equal_market_seed_loss_not_forecast_ensemble(self):
        first = score(self.frame)
        second = score(self.frame.assign(seed='8102', variance=2., second_moment=6.))
        average, _, _, _ = summarize(pd.concat([first, second], ignore_index=True))
        expected = np.mean([np.mean([f.loc[f.market == m, 'r2_qlike'].mean() for m in ['A', 'B']])
                            for f in [first, second]])
        self.assertAlmostEqual(average['r2_qlike'], expected)
        self.assertNotAlmostEqual(average['r2_qlike'], np.mean(qlike(self.mask.r2, 5.5)))

    def test_consistent_unit_conversion_only_adds_common_constant(self):
        y, h = np.array([0., 2., 10.]), np.array([1., 3., 5.])
        np.testing.assert_allclose(qlike(y / 10000., h / 10000.), qlike(y, h) - np.log(10000.))
        np.testing.assert_allclose(qlike(y, h) - qlike(y, h * 2),
                                   qlike(y / 10000., h / 10000.) - qlike(y / 10000., h * 2 / 10000.))

    def test_existing_destination_is_never_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'original.txt'
            marker.write_text('preserve me')
            with self.assertRaises(FileExistsError):
                run(folder)
            self.assertEqual(marker.read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
