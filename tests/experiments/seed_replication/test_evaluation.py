"""Synthetic tests only: no market forecast, fit, or evaluation pilot."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t

from . import evaluation as E


def index_fixture(n=80):
    dates = np.tile(pd.bdate_range('2015-01-01', periods=n).strftime('%Y-%m-%d'), 2)
    r = np.sin(np.arange(2*n))
    frame = pd.DataFrame(dict(market=np.repeat(['A', 'B'], n), date=dates,
                              return_=r, rk=.1 + r*r)).rename(columns={'return_': 'return'})
    frame['network_mean'] = .2
    frame['network_second_moment'] = 1.
    frame['network_variance'] = .96
    frame['network_logscore'] = t.logpdf(r, 8, scale=np.sqrt(.75))
    for control in E.CONTROLS:
        frame[control + '_valid'] = True
        frame[control + '_mean'] = .6
        frame[control + '_second_moment'] = 1.3
        frame[control + '_variance'] = .94
        frame[control + '_logscore'] = t.logpdf(r - .6, 8, scale=np.sqrt(.75*.94))
    return frame


class EvaluationTests(unittest.TestCase):
    def test_registry_preserves_every_family_and_reserved_slot(self):
        registry = E.comparison_registry()
        self.assertEqual(len(registry), 44)
        counts = {f: sum(r['family'] == f for r in registry) for f in set(r['family'] for r in registry)}
        self.assertEqual(counts, dict(index_density=4, index_qlike=12, calendar_a=12,
                                     calendar_b=8, transfer_stocks=4, transfer_fx=4))
        self.assertEqual(sum(r.get('reserved', False) for r in registry), 4)
        self.assertEqual(len(set(r['family']+'::'+r['comparison'] for r in registry)), 44)

    def test_index_common_law_uses_second_moment_about_zero(self):
        frame = index_fixture()
        results, gains = E.score_index_frame(frame, reps=39, block=10)
        rows = next(g for g in gains if g.comparison.iloc[0] == 'C4')
        expected = (t.logpdf(frame['return'], 8, scale=np.sqrt(.75)) -
                    t.logpdf(frame['return'], 8, scale=np.sqrt(.75*1.3)))
        np.testing.assert_allclose(rows.gain, expected, rtol=1e-12, atol=1e-12)
        incorrect = (t.logpdf(frame['return'], 8, scale=np.sqrt(.75)) -
                     t.logpdf(frame['return'], 8, scale=np.sqrt(.75*.94)))
        self.assertGreater(np.max(np.abs(rows.gain-incorrect)), .01)

    def test_index_failed_control_never_drops_its_slots(self):
        frame = index_fixture()
        frame.loc[frame.market == 'B', 'egarch_t_valid'] = False
        results, gains = E.score_index_frame(frame, reps=39, block=10)
        missing = [r for r in results if r['control'] == 'egarch_t']
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(r['p_null'] == r['holm_p'] == 1 for r in missing))
        self.assertTrue(all(r['status'] == 'unavailable_reserved_slot' for r in missing))
        self.assertEqual(sum(r['family'] == 'index_qlike' for r in results), 12)

    def test_zero_return_kept_missing_proxy_has_own_mask(self):
        frame = index_fixture()
        frame.loc[0, ['return', 'rk']] = [0., np.nan]
        _, gains = E.score_index_frame(frame, reps=39, block=10)
        r2 = next(g for g in gains if g.comparison.iloc[0] == 'r2__garch_t')
        rk = next(g for g in gains if g.comparison.iloc[0] == 'rk__garch_t')
        self.assertEqual(len(r2), len(frame))
        self.assertEqual(len(rk), len(frame)-1)

    def test_masks_reject_drops_duplicates_and_added_dates(self):
        mask = index_fixture()[['market', 'date']]
        E._assert_mask(mask.iloc[::-1], mask, 'test')
        for bad in (mask.iloc[1:], pd.concat([mask, mask.iloc[:1]])):
            with self.assertRaises(ValueError):
                E._assert_mask(bad, mask, 'test')

    def test_transfer_failed_fit_kept_and_equivalence_separate(self):
        f = index_fixture()
        panel = pd.DataFrame(dict(market=f.market, date=f.date, r=f['return'],
            VolPFN=np.ones(len(f)), R_eq=np.full(len(f), 1.3), R_class=np.full(len(f), 1.1),
            R_market=np.full(len(f), 1.2), R_archive=np.full(len(f), 1.3)))
        frames = {k: panel.copy() for k in ('indices', 'stocks', 'fx')}
        frames['stocks'].loc[frames['stocks'].market == 'B', 'R_market'] = np.nan
        result, gains, legacy = E.score_transfer_frames(frames,
            {k: ['A', 'B'] for k in frames}, reps=39, block=10)
        missing = [r for r in result if r['family'] == 'transfer_stocks' and r['control'] == 'R_market']
        self.assertEqual(missing[0]['p_null'], 1.)
        self.assertEqual(len([r for r in result if r['family'] == 'transfer_stocks']), 4)
        self.assertTrue(legacy['stocks']['equivalence_primary']['confirmatory'])
        self.assertFalse(legacy['fx']['equivalence_descriptive']['confirmatory'])
        self.assertEqual(legacy['fx']['equivalence_descriptive']['p_tost'], 1.)

    def test_frozen_input_change_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            f = Path(path) / 'input.txt'; f.write_text('one')
            inventory = dict(registry=E.comparison_registry(), hashes={str(f): E.sha(f)})
            E.verify_inventory(inventory)
            f.write_text('two')
            with self.assertRaises(RuntimeError):
                E.verify_inventory(inventory)

    def test_prediction_shape_and_variance_failures_rejected(self):
        def bad(*args, **kwargs):
            n = len(args[1])
            return dict(h=np.ones(n), mean=np.full(n, 2.), logscore=np.zeros(n),
                        **{k: np.ones((n, 1)) for k in E.PARAMS})
        with self.assertRaises(ValueError):
            E._prediction(bad, None, np.ones(150))

    def test_calendar_fit_never_uses_evaluation_outcomes(self):
        dates = pd.bdate_range('2014-01-01', periods=350).strftime('%Y-%m-%d')
        n = len(dates)
        panel = pd.DataFrame(dict(market='FAKE', date=dates,
                                  r=np.sin(np.arange(n)) * .5, rk=np.ones(n)))
        dist = dict(logw=np.zeros((n, 1)), loc=np.zeros((n, 1)),
                    scale=np.full((n, 1), np.sqrt(.75)), nu=np.full((n, 1), 8.))
        feature = panel[['market', 'date']].copy()
        feature['latest_x'] = .3 * np.cos(np.arange(n))
        feature['lagged_x'] = .2 * np.cos(np.arange(n))
        inputs = {k: dist for k in ('network', 'compact', 'gjr')}
        features = {k: feature for k in E.SCENARIOS}
        _, _, first, _ = E.score_calendar_arrays(panel, inputs, features, reps=19, block=10)
        changed = panel.copy()
        changed.loc[changed.date >= '2015-01-01', 'r'] *= 3.
        _, _, second, _ = E.score_calendar_arrays(changed, inputs, features, reps=19, block=10)
        for scenario in E.SCENARIOS:
            self.assertEqual(first['details'][scenario]['coefficients'],
                             second['details'][scenario]['coefficients'])

    def test_calendar_rejects_missing_scenario(self):
        with self.assertRaises(ValueError):
            E.score_calendar_arrays(pd.DataFrame(), {}, {'nominal': pd.DataFrame()}, reps=19)

    def test_synthetic_only_full_pipeline_never_loads_market_inputs(self):
        with tempfile.TemporaryDirectory() as path:
            with patch.object(E, 'inventory_inputs', side_effect=AssertionError('No market input')), \
                 patch('experiments.session_calendar_screen.run.load_inputs', side_effect=AssertionError('No market input')):
                result = E.synthetic_integration(path, reps=19)
            self.assertEqual(result['comparisons'], 44)
            self.assertEqual(result['kind'], 'synthetic_only_no_market_input_access')
            registry = E._json(Path(path) / 'comparison_registry.json')
            self.assertEqual(sum(r['status'] != 'available' for r in registry), 6)
            with self.assertRaises(FileExistsError):
                E.synthetic_integration(path, reps=19)


if __name__ == '__main__':
    unittest.main()
