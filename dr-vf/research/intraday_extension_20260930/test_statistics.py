"""Independent tests of calendar inference, locked inputs and both gain directions.

Run-level fixtures use five synthetic origins and controlled bootstrap errors.
The exact full-mask size validator is bypassed only for these fixtures; canonical
mask equality, target/loss checks, seed guards and inference remain active.
No historical forecasts are fitted, changed or analysed by this test suite.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.intraday_extension_20260930 import statistics as S
from research.intraday_extension_20260930.run_blends import SEEDS, expected_seeds
from research.combined_leaderboard.statistics import _panel, _resample
from research.population_transfer.statistics import _basic_interval, _calendar_counts, holm


POSITIVE = 'ttm_rk'
NEGATIVE = 'volpfn_20260930_full+loghar'


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source, self.canonical = self.folder / 'source', self.folder / 'canonical'
        self.source.mkdir()
        self.canonical.mkdir()
        (self.source / 'row_forecasts').mkdir()
        self.mask = pd.DataFrame({
            'market': ['A', 'A', 'A', 'B', 'B'],
            'date': ['2019-01-02', '2019-01-03', '2019-01-04',
                     '2019-01-02', '2019-01-04'],
            'r2': [0., 1., 4., .25, 2.25],
            'rk': [.5, 1., 2., 1.5, 2.5],
        })
        for folder in (self.source, self.canonical):
            self.mask.to_csv(folder / 'common_mask.csv', index=False)
        design = {'roster': S.ROSTER, 'reference': S.REFERENCE, 'family_slots': 38}
        (self.source / 'REGISTERED_DESIGN.json').write_text(json.dumps(design))
        self.frames = {}
        self.write_model(S.REFERENCE, [(seed, float(i + 1), float(i + 1))
                                      for i, seed in enumerate(SEEDS)])
        # Both endpoints are worse than every fixed reference seed.
        self.write_model(POSITIVE, [('none', 1e6, 1e6)])
        # Synthetic valid forecasts yielding smaller losses than every seed.
        # This is a fixture for inference, not a runnable forecasting recipe.
        self.write_model(NEGATIVE, [(seed, np.maximum(self.mask.r2, 1e-6), self.mask.rk)
                                    for seed in SEEDS])

    def write_model(self, model, forecasts):
        pieces = []
        for seed, r2_h, rk_h in forecasts:
            f = self.mask.copy()
            f['model'], f['seed'] = model, seed
            for proxy, h in (('r2', r2_h), ('rk', rk_h)):
                f[proxy + '_forecast'] = h
                f[proxy + '_qlike'] = f[proxy] / f[proxy + '_forecast'] + np.log(f[proxy + '_forecast'])
            pieces.append(f)
        self.frames[model] = pd.concat(pieces, ignore_index=True)
        self.save_frame(model)

    def save_frame(self, model):
        self.frames[model].to_csv(self.source / 'row_forecasts' / (model + '.csv.gz'),
                                  index=False, compression='gzip')

    def run_fixture(self, *, error_scale='small'):
        count_calls, resample_calls, interval_coverages = [], [], []

        def counts(n, reps, block, seed):
            count_calls.append((n, reps, block, seed))
            # A token identifies the shared count object within each block.
            return np.array([[float(block)]])

        def resample(matrix, shared_counts):
            per_market = np.nanmean(matrix, axis=0)
            estimate = float(per_market.mean())
            if error_scale == 'tail':
                errors = np.tile(np.array([-1.1, -.5, 0., .5, 1.1]), 4000) * abs(estimate)
            else:
                errors = np.tile(np.array([-.001, -.0005, 0., .0005, .001]), 4000)
            resample_calls.append((int(shared_counts[0, 0]), id(shared_counts)))
            return estimate, estimate + errors, per_market

        def interval(estimate, draws, coverage):
            interval_coverages.append(coverage)
            return _basic_interval(estimate, draws, coverage)

        destination = self.folder / ('results_' + error_scale)
        with ExitStack() as stack:
            stack.enter_context(patch.object(S, 'SOURCE', self.canonical))
            stack.enter_context(patch.object(S, 'validate_mask', lambda mask: None))
            stack.enter_context(patch.object(S, '_calendar_counts', counts))
            stack.enter_context(patch.object(S, '_resample', resample))
            stack.enter_context(patch.object(S, '_basic_interval', interval))
            stack.enter_context(redirect_stdout(io.StringIO()))
            S.run(self.source, destination)
        return (json.loads((destination / 'statistics.json').read_text()), destination,
                count_calls, resample_calls, interval_coverages)

    @staticmethod
    def equal_market(frame, column):
        return float(frame.groupby('market')[column].mean().mean())

    def test_fixed_family_shared_draws_seed_loss_averaging_and_both_directions(self):
        result, destination, calls, resamples, coverages = self.run_fixture()
        self.assertEqual(calls, [(3, 20000, block, 2026093004) for block in (60, 20, 120)])
        self.assertEqual(len(result['contrasts']), 114)
        self.assertEqual(result['family_slots'], 38)
        self.assertEqual(result['primary_block'], 60)
        self.assertEqual(result['sensitivity_blocks'], [20, 120])
        self.assertIn('conditional frozen-forecast', result['scope'])
        for excluded in ('online refit', 'training', 'pretraining', 'selection'):
            self.assertIn(excluded, result['scope'])
        for block in (60, 20, 120):
            identities = {identity for b, identity in resamples if b == block}
            self.assertEqual(len(identities), 1, 'Every model/proxy must reuse the same date draw object')
            current = [r for r in result['contrasts'] if r['block'] == block]
            self.assertEqual(len(current), 38)
            for row in current:
                if row['status'] != 'complete':
                    self.assertEqual(row['p_raw'], 1.)
                    self.assertEqual(row['p_holm'], 1.)
                    self.assertFalse(row['positive_support'])
                    self.assertFalse(row['negative_support'])
                    self.assertIsNone(row['gain'])
                    continue
                # All four nonzero synthetic effects have p=1/20001. Ties
                # retain the full 38-slot first Holm multiplier.
                self.assertAlmostEqual(row['p_raw'], 1 / 20001)
                self.assertAlmostEqual(row['p_holm'], 38 / 20001)
                proxy = row['proxy']
                comp = self.frames[row['comparator']]
                ref_scores = [self.equal_market(self.frames[S.REFERENCE].loc[
                    self.frames[S.REFERENCE].seed == seed], proxy + '_qlike') for seed in SEEDS]
                comp_score = float(comp.groupby(['seed', 'market'])[proxy + '_qlike'].mean()
                                   .groupby('seed').mean().mean())
                self.assertAlmostEqual(row['gain'], comp_score - np.mean(ref_scores))
                np.testing.assert_allclose(row['reference_seed_gains'], comp_score - np.array(ref_scores))
                positive = row['comparator'] == POSITIVE
                self.assertEqual(row['positive_support'], positive)
                self.assertEqual(row['negative_support'], not positive)
                self.assertEqual(row['reference_seeds_positive'], 5 if positive else 0)
                self.assertEqual(row['reference_seeds_negative'], 0 if positive else 5)
        self.assertEqual(coverages.count(.95), 12)
        self.assertEqual(coverages.count(1 - .05 / 38), 12)
        with np.load(destination / 'bootstrap_archive_no_pickle.npz', allow_pickle=False) as archive:
            self.assertEqual(len(archive.files), 12)
            for name in archive.files:
                self.assertEqual(archive[name].shape, (20000,))
        provenance = json.loads((destination / 'provenance.json').read_text())
        self.assertTrue({'python', 'numpy', 'pandas'}.issubset(provenance))
        paths = provenance['source_hashes']
        for helper in ('research/combined_leaderboard/statistics.py',
                       'research/population_transfer/statistics.py',
                       'research/blend_rk/blend.py',
                       'research/intraday_extension_20260930/run_blends.py'):
            self.assertTrue(any(p.endswith(helper) for p in paths))

    def test_two_sided_centered_p_value_counts_both_tails_and_uses_plus_one(self):
        result, _, _, _, _ = self.run_fixture(error_scale='tail')
        for row in result['contrasts']:
            if row['status'] == 'complete':
                self.assertAlmostEqual(row['p_raw'], 8001 / 20001)
                self.assertEqual(row['p_holm'], 1.)
                self.assertFalse(row['positive_support'])
                self.assertFalse(row['negative_support'])

    def test_joint_direction_requires_agreement_on_both_proxies(self):
        f = self.frames[POSITIVE]
        f['rk_forecast'] = f.rk
        f['rk_qlike'] = 1 + np.log(f.rk)
        self.save_frame(POSITIVE)
        result, _, _, _, _ = self.run_fixture()
        for row in result['contrasts']:
            if row['comparator'] == POSITIVE:
                self.assertGreater(row['gain'] if row['proxy'] == 'r2' else -row['gain'], 0)
                self.assertLess(row['p_holm'], .05)
                self.assertFalse(row['positive_support'])
                self.assertFalse(row['negative_support'])

    def test_positive_average_and_family_interval_do_not_replace_five_seed_support(self):
        self.write_model(POSITIVE, [('none', 4., 4.)])
        result, _, _, _, _ = self.run_fixture()
        for row in result['contrasts']:
            if row['comparator'] == POSITIVE:
                self.assertGreater(row['gain'], 0)
                self.assertGreater(row['family_interval95'][0], 0)
                self.assertLess(row['p_holm'], .05)
                self.assertEqual(row['reference_seeds_positive'], 3)
                self.assertFalse(row['positive_support'])

    def test_negative_average_and_family_interval_do_not_replace_five_seed_support(self):
        self.write_model(POSITIVE, [('none', 2., 2.)])
        result, _, _, _, _ = self.run_fixture()
        for row in result['contrasts']:
            if row['comparator'] == POSITIVE:
                self.assertLess(row['gain'], 0)
                self.assertLess(row['family_interval95'][1], 0)
                self.assertLess(row['p_holm'], .05)
                self.assertEqual(row['reference_seeds_negative'], 4)
                self.assertFalse(row['negative_support'])

    def test_reference_requires_exact_seed_identities_not_only_five_seeds(self):
        f = self.frames[S.REFERENCE]
        f['seed'] = f.seed.map({s: str(int(s) + 1000) for s in SEEDS})
        self.save_frame(S.REFERENCE)
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.run_fixture()

    def test_every_stochastic_comparator_requires_all_registered_seeds(self):
        self.frames[NEGATIVE] = self.frames[NEGATIVE].loc[self.frames[NEGATIVE].seed != SEEDS[-1]]
        self.save_frame(NEGATIVE)
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.run_fixture()

    def test_stored_losses_are_recomputed_not_trusted(self):
        self.frames[POSITIVE].loc[0, 'rk_qlike'] += .01
        self.save_frame(POSITIVE)
        with self.assertRaises(AssertionError):
            self.run_fixture()

    def test_model_target_change_cannot_pass_scored_loss_consistency(self):
        f = self.frames[POSITIVE]
        f.loc[0, 'rk'] *= 1.1
        f['rk_qlike'] = f.rk / f.rk_forecast + np.log(f.rk_forecast)
        self.save_frame(POSITIVE)
        with self.assertRaises(AssertionError):
            self.run_fixture()

    def test_coordinated_small_mask_target_change_is_rejected_against_original(self):
        altered = self.mask.copy()
        altered.loc[0, 'rk'] *= 1 + 1e-6
        altered.to_csv(self.source / 'common_mask.csv', index=False)
        for model, f in self.frames.items():
            change = (f.market == 'A') & (f.date == '2019-01-02')
            f.loc[change, 'rk'] = altered.loc[0, 'rk']
            f['rk_qlike'] = f.rk / f.rk_forecast + np.log(f.rk_forecast)
            self.save_frame(model)
        with self.assertRaises(AssertionError):
            self.run_fixture()

    def test_changed_calendar_key_is_rejected_without_availability_intersection(self):
        altered = self.mask.copy()
        altered.loc[0, 'date'] = '2019-01-01'
        altered.to_csv(self.source / 'common_mask.csv', index=False)
        with self.assertRaises(AssertionError):
            self.run_fixture()

    def test_expected_seed_roster_keeps_frontier_stochastic_and_deterministic_distinct(self):
        for model in ('gnnhar', 'transfer_xgboost', 'm2vn', 'deep_rgarch', S.REFERENCE):
            self.assertEqual(expected_seeds(model), SEEDS)
        for model in ('qlike_har', 'ttm_rk', 'gjr_garch_t+ttm_rk'):
            self.assertEqual(expected_seeds(model), ('none',))


class CalendarAndMultiplicityTests(unittest.TestCase):
    def test_asynchronous_market_means_and_draw_denominators(self):
        dates = ['d0', 'd1', 'd2', 'd0', 'd2']
        markets = ['A', 'A', 'A', 'B', 'B']
        gains = [1., 2., 3., 10., 30.]
        matrix = _panel(dates, markets, gains, ['d0', 'd1', 'd2'], ['A', 'B'])
        self.assertTrue(np.isnan(matrix[1, 1]), 'A market holiday must stay absent')
        counts = np.array([[1., 1., 1.], [2., 1., 0.], [0., 1., 2.]])
        estimate, draws, per_market = _resample(matrix, counts)
        self.assertEqual(estimate, 11.)
        np.testing.assert_allclose(per_market, [2., 20.])
        np.testing.assert_allclose(draws, [11., 17 / 3, 49 / 3])
        self.assertNotEqual(estimate, np.mean(gains), 'Long markets must not receive extra weight')
        with self.assertRaisesRegex(ValueError, 'omits a market'):
            _resample(matrix, np.array([[0., 1., 0.]]))

    def test_circular_calendar_draws_are_fixed_length_repeatable_and_read_only(self):
        first = _calendar_counts(11, 12, 4, 345)
        again = _calendar_counts(11, 12, 4, 345)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(first.sum(axis=1), np.full(12, 11.))
        self.assertFalse(first.flags.writeable)
        self.assertFalse(np.array_equal(first, _calendar_counts(11, 12, 4, 346)))

    def test_unavailable_slots_still_multiply_holm_family(self):
        p = [.001, .01] + [1.] * 36
        adjusted = holm(p)
        np.testing.assert_allclose(adjusted[:2], [.038, .37])
        np.testing.assert_array_equal(adjusted[2:], np.ones(36))
        self.assertGreater(adjusted[0], holm(p[:2])[0])

    def test_basic_interval_reflects_centered_errors(self):
        # Draws are asymmetric around the estimate: the basic interval must
        # reflect bootstrap errors rather than copy percentile endpoints.
        np.testing.assert_allclose(_basic_interval(2., np.array([1., 2., 3., 4., 5.]), .8),
                                   [-.6, 2.6])


if __name__ == '__main__':
    unittest.main()
