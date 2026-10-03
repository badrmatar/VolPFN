"""Safeguards for combining issued forecasts without changing their estimands."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_extension_20260930 import grand_table as G


class GrandTableTests(unittest.TestCase):
    def setUp(self):
        self.mask = pd.DataFrame({'market': ['A', 'A', 'B'],
                                  'date': ['2020-01-01', '2020-01-02', '2020-01-01'],
                                  'r2': [0., 4., 9.], 'rk': [1., 2., 3.]})
        self.model = 'gjr_garch_t'
        self.native = self.mask.assign(model=self.model, seed='none',
                                        mean=2., variance=1., second_moment=5.)
        self.native['r2_qlike'] = G.loss(self.native.r2, self.native.second_moment)
        self.native['rk_qlike'] = G.loss(self.native.rk, self.native.variance)

    def test_native_nonzero_mean_keeps_distinct_functionals(self):
        f = G.native_rows(self.native, self.mask, self.model)
        np.testing.assert_array_equal(f.r2_forecast, [5., 5., 5.])
        np.testing.assert_array_equal(f.rk_forecast, [1., 1., 1.])
        self.assertEqual(f.r2_qlike.iloc[0], np.log(5.))  # Zero target retained.
        self.assertEqual(f.rk_qlike.iloc[0], 1.)
        with self.assertRaises(AssertionError):
            G.native_rows(self.native.assign(second_moment=1.), self.mask, self.model)

    def test_missing_keys_extra_seeds_units_and_row_score_corruption_fail(self):
        for broken in (self.native.iloc[:-1], self.native.assign(seed='8101'),
                       pd.concat([self.native, self.native.iloc[:1]])):
            with self.assertRaises(ValueError):
                G.native_rows(broken, self.mask, self.model)
        for broken in (self.native.assign(rk=self.native.rk * 10000),
                       self.native.assign(rk_qlike=self.native.rk_qlike + 0.1)):
            with self.assertRaises(AssertionError):
                G.native_rows(broken, self.mask, self.model)
        with self.assertRaises(ValueError):
            G.native_rows(self.native.assign(context_end=self.native.date), self.mask, self.model)

    def test_overlap_check_detects_changed_issued_forecast(self):
        f = G.native_rows(self.native, self.mask, self.model)
        G.compare_overlap(f, f.sample(frac=1., random_state=1), self.model)
        with self.assertRaises(AssertionError):
            G.compare_overlap(f, f.assign(rk_forecast=f.rk_forecast * 1.01), self.model)

    def test_loss_aggregation_weights_market_and_seed_not_date_counts(self):
        first = self.mask.assign(seed='8101', r2_qlike=[0., 0., 9.], rk_qlike=[0., 0., 9.])
        second = self.mask.assign(seed='8102', r2_qlike=[0., 0., 3.], rk_qlike=[0., 0., 3.])
        overall, _, _ = G.summarize(pd.concat([first, second]))
        self.assertEqual(overall['r2_qlike'], 3.)
        self.assertEqual(overall['seeds'], 2)
        saved = G.summarize(first)[1]
        G.verify_seed_market(first, saved, 'dummy')
        with self.assertRaises(AssertionError):
            G.verify_seed_market(first, saved.assign(rk_qlike=saved.rk_qlike + 0.01), 'dummy')

    def test_fixed_roster_excludes_diagnostics_and_unavailable_has_no_rank(self):
        self.assertEqual(len(G.DAILY), 15)
        self.assertEqual(len(G.ROSTER), 32)
        self.assertEqual(set(G.DAILY) & set(G.INTRADAY), set(G.OVERLAPS))
        self.assertFalse(set(G.DIAGNOSTICS) & set(G.ROSTER))
        records = [{'model': 'a', 'status': 'complete', 'r2_qlike': 1., 'rk_qlike': 2.},
                   {'model': 'missing', 'status': 'unavailable', 'r2_qlike': None, 'rk_qlike': None}]
        table = G.rank_table(records)
        self.assertTrue(table.loc[table.model == 'missing', ['r2_rank', 'rk_rank']].isna().all().all())
        records[1]['rk_qlike'] = 1.
        with self.assertRaises(ValueError):
            G.rank_table(records)

    def test_five_seed_point_must_retain_same_risk_forecast_both_proxies(self):
        f = pd.concat([self.mask.assign(model='gnnhar', seed=s, r2_forecast=2., rk_forecast=2.,
                                       r2_qlike=G.loss(self.mask.r2, 2.),
                                       rk_qlike=G.loss(self.mask.rk, 2.)) for s in G.SEEDS], ignore_index=True)
        good = G.risk_rows(f, self.mask, 'gnnhar')
        self.assertNotIn('mean', good)
        with self.assertRaises(ValueError):
            G.risk_rows(f.loc[f.seed != G.SEEDS[0]], self.mask, 'gnnhar')
        broken = f.assign(r2_forecast=3., r2_qlike=G.loss(f.r2, 3.))
        with self.assertRaises(AssertionError):
            G.risk_rows(broken, self.mask, 'gnnhar')

    def test_source_hash_drift_and_destination_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'original.txt'
            p.write_text('original')
            pins = G.Pins()
            pins.pin(p)
            p.write_text('changed')
            with self.assertRaises(ValueError):
                pins.recheck()
            with self.assertRaises(FileExistsError):
                G.run('unused-source', folder)
            self.assertEqual(p.read_text(), 'changed')

    def test_native_point_serialization_roundoff_is_tolerated_but_change_is_not(self):
        f = pd.concat([self.mask.assign(model='gnnhar', seed=s, r2_forecast=2., rk_forecast=2.,
                                       r2_qlike=G.loss(self.mask.r2, 2.),
                                       rk_qlike=G.loss(self.mask.rk, 2.)) for s in G.SEEDS], ignore_index=True)
        native = f[['market', 'date', 'seed']].assign(
            rk_forecast=np.nextafter(2., 3.), native_raw_forecast=1.9999,
            origin_date='2019-12-31', latest_input_date='2019-12-31',
            latest_neighbor_input_date='2019-12-30')
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'gnnhar_v2' / 'predictions.csv'
            p.parent.mkdir()
            native.to_csv(p, index=False)
            G.verify_native_point(f, self.mask, 'gnnhar', {str(p): G.sha(p)}, G.Pins())
            with self.assertRaises(AssertionError):
                G.verify_native_point(f.assign(rk_forecast=2.001), self.mask, 'gnnhar',
                                      {str(p): G.sha(p)}, G.Pins())


if __name__ == '__main__':
    unittest.main()
