"""Causality, scaling, missing-evidence and preservation safeguards."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.blend_rk.blend import blend_series
from experiments.intraday_extension_20260930.run_blends import align_native, align_risk, summary, run, ROSTER


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.mask = pd.DataFrame({'market': ['A', 'A', 'B'], 'date': ['2020-01-01', '2020-01-02', '2020-01-01'],
                                  'r2': [0., 4., 9.], 'rk': [1., 2., 3.]})
        self.native = self.mask.assign(seed='8101', mean=2., variance=1., second_moment=5.)

    def test_future_targets_cannot_change_already_issued_blend(self):
        rng = np.random.default_rng(31)
        dates = np.repeat(np.arange(140), 2)
        keys = [(str(m), str(d)) for d in range(140) for m in range(2)]
        a, b = np.exp(rng.normal(size=(2, len(dates))))
        target = np.exp(rng.normal(size=len(dates)))
        first, w_first = blend_series(keys, dates, a, b, target)
        other = target.copy()
        other[dates >= 80] *= 2
        second, w_second = blend_series(keys, dates, a, b, other)
        np.testing.assert_array_equal(first[dates < 100], second[dates < 100])
        self.assertEqual(w_first[:5], w_second[:5])
        self.assertFalse(np.array_equal(first[dates >= 100], second[dates >= 100]))

    def test_native_moments_targets_and_all_seeds_are_required(self):
        valid = align_native(self.native, self.mask, ['8101'])['8101']
        np.testing.assert_array_equal(valid.second_moment, [5., 5., 5.])
        for corrupt in [self.native.iloc[:-1], self.native.assign(seed='none')]:
            with self.assertRaises(ValueError):
                align_native(corrupt, self.mask, ['8101'])
        for corrupt in [self.native.assign(rk=self.native.rk * 10000), self.native.assign(second_moment=1.)]:
            with self.assertRaises(AssertionError):
                align_native(corrupt, self.mask, ['8101'])

    def test_native_rk_output_has_no_fabricated_density(self):
        native = self.mask[['market', 'date']].assign(seed='8101', rk_forecast=2., origin_date='2019-12-31', latest_input_date='2019-12-31')
        result = align_risk(native, self.mask, ['8101'])['8101']
        self.assertNotIn('mean', result)
        with self.assertRaises(ValueError):
            align_risk(native.assign(latest_input_date=native.date), self.mask, ['8101'])
        with self.assertRaises(ValueError):
            align_risk(native.assign(rk_forecast=0.), self.mask, ['8101'])
        with self.assertRaises(ValueError):
            align_risk(native.drop(columns='latest_input_date'), self.mask, ['8101'])
        with self.assertRaisesRegex(ValueError, 'Model-specific'):
            align_risk(native, self.mask, ['8101'], required_cutoffs=('fit_end',))
        with self.assertRaises(ValueError):
            align_risk(native.assign(latest_source_label_date=native.date), self.mask, ['8101'])
        with self.assertRaises(ValueError):
            align_risk(native.assign(fit_end='2019-12-31', fit_block_start='2019-12-31'),
                       self.mask, ['8101'])

    def test_partial_frontier_seeds_cannot_be_ranked(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'All registered seeds required'):
                run(Path(folder) / 'failed_attempt', {'gnnhar': {'path': 'unused.csv',
                                                                 'seeds': ['8101']}})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Independent fitting/coverage verification'):
                run(Path(folder) / 'failed_attempt', {'qlike_har': {'path': 'unused.csv',
                                                                    'seeds': ['none']}})

    def test_equal_market_and_seed_losses(self):
        first = self.mask.assign(seed='8101', r2_qlike=[0., 0., 9.], rk_qlike=[0., 0., 9.])
        second = self.mask.assign(seed='8102', r2_qlike=[0., 0., 3.], rk_qlike=[0., 0., 3.])
        self.assertEqual(summary(pd.concat([first, second]))['rk_qlike'], 3.)

    def test_fixed_roster_and_historical_destination_refusal(self):
        self.assertEqual(len(ROSTER), 20)
        self.assertTrue({'gnnhar', 'qlike_har', 'transfer_xgboost', 'm2vn', 'deep_rgarch'}.issubset(ROSTER))
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'history.txt'
            marker.write_text('original')
            with self.assertRaises(FileExistsError):
                run(folder)
            self.assertEqual(marker.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
