"""Synthetic checks of the new matched-seed, 12-slot inference contract.

The integration fixture has five keys on three dates, not scientific data. Only
the exact 22,026-row population-size validator is patched; original-mask equality,
hash receipts, the seven-model/seed roster, recomputed scoring and inference stay
active. Bootstrap errors are controlled to test tails and support logic without
fitting or scoring any historical scientific forecast.
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

from research.cross_market_blend_20260930 import statistics as S
from research.population_transfer.statistics import _basic_interval, _calendar_counts, holm
from research.combined_leaderboard.statistics import _panel, _resample


POSITIVE, NEGATIVE = 'crossmarket_xgb', 'gnnhar'


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.run_number = 0
        self.source, self.canonical = self.folder / 'source', self.folder / 'canonical'
        self.source.mkdir()
        self.canonical.mkdir()
        (self.source / 'row_forecasts').mkdir()
        (self.source / 'frozen_sources').mkdir()
        self.mask = pd.DataFrame({
            'market': ['A', 'A', 'A', 'B', 'B'],
            'date': ['2019-01-02', '2019-01-03', '2019-01-04', '2019-01-02', '2019-01-04'],
            'r2': [0., 1., 4., .25, 2.25], 'rk': [.5, 1., 2., 1.5, 2.5]})
        for folder in (self.source, self.canonical):
            self.mask.to_csv(folder / 'common_mask.csv', index=False)
        protocol = self.source / 'frozen_sources/PROTOCOL.md'
        protocol.write_text('Synthetic protocol fixture; no scientific analysis.\n')
        self.design = {'roster': S.ROSTER, 'reference': S.REFERENCE, 'family_slots': 12,
                       'seeds': S.SEEDS, 'row_count_per_seed': 22026, 'refit_dates': 20,
                       'burn_dates': 60, 'inferential_seed': 2026093005,
                       'blocks': [60, 20, 120], 'resamples': 20000,
                       'protocol_sha256': S.sha(protocol)}
        (self.source / 'REGISTERED_DESIGN.json').write_text(json.dumps(self.design))
        (self.source / 'verification.json').write_text(json.dumps({
            'status': 'passed', 'exact_common_mask': True, 'all_five_seeds': True,
            'procedures': 7, 'scored_rows': 7 * 5 * 22026}))
        (self.source / 'provenance.json').write_text(json.dumps({
            'originals_preserved': True,
            'source_hashes': {str(self.canonical / 'common_mask.csv'):
                              S.sha(self.canonical / 'common_mask.csv')}}))
        self.frames = {}
        # h is larger than each target, so adding one raises QLIKE at every
        # corresponding seed. Averaging comparator seeds before subtraction
        # would instead make the last two seed differences negative.
        reference = [40. + 10. * i for i in range(5)]
        self.write_model(S.REFERENCE, [(seed, h, h) for seed, h in zip(S.SEEDS, reference)])
        self.write_model(POSITIVE, [(seed, h + 1., h + 1.) for seed, h in zip(S.SEEDS, reference)])
        self.write_model(NEGATIVE, [(seed, self.mask.r2.clip(lower=1e-6), self.mask.rk)
                                    for seed in S.SEEDS])
        for name in S.ROSTER:
            if name not in self.frames:
                self.write_model(name, [(seed, 100., 100.) for seed in S.SEEDS])
        self.refresh_receipt()

    def write_model(self, model, forecasts):
        pieces = []
        for seed, r2_h, rk_h in forecasts:
            frame = self.mask.copy()
            frame['model'], frame['seed'] = model, seed
            for proxy, h in (('r2', r2_h), ('rk', rk_h)):
                frame[proxy + '_forecast'] = h
                frame[proxy + '_qlike'] = frame[proxy] / frame[proxy + '_forecast'] + np.log(frame[proxy + '_forecast'])
            pieces.append(frame)
        self.frames[model] = pd.concat(pieces, ignore_index=True)
        self.save_model(model)

    def save_model(self, model):
        self.frames[model].to_csv(self.source / 'row_forecasts' / (model + '.csv.gz'),
                                  index=False, compression='gzip')

    def refresh_receipt(self):
        receipt = {str(p.relative_to(self.source)): S.sha(p)
                   for p in sorted(self.source.rglob('*'))
                   if p.is_file() and p.name != 'ARTIFACT_RECEIPT.json'}
        (self.source / 'ARTIFACT_RECEIPT.json').write_text(json.dumps(receipt))

    def run_fixture(self, *, tail=False):
        calls, resamples, coverages = [], [], []

        def counts(n, reps, block, seed):
            calls.append((n, reps, block, seed))
            return np.array([[float(block)]])

        def resample(matrix, shared_counts):
            per_market = np.nanmean(matrix, axis=0)
            estimate = float(per_market.mean())
            errors = (np.tile(np.array([-1.1, -.5, 0., .5, 1.1]), 4000) * abs(estimate)
                      if tail else np.tile(np.array([-1e-7, -.5e-7, 0., .5e-7, 1e-7]), 4000))
            resamples.append((int(shared_counts[0, 0]), id(shared_counts)))
            return estimate, estimate + errors, per_market

        def interval(estimate, draws, coverage):
            coverages.append(coverage)
            return _basic_interval(estimate, draws, coverage)

        self.run_number += 1
        destination = self.folder / ('results_' + str(self.run_number))
        with ExitStack() as stack:
            stack.enter_context(patch.object(S, 'SOURCE', self.canonical))
            stack.enter_context(patch.object(S, 'validate_mask', lambda mask: None))
            stack.enter_context(patch.object(S, '_calendar_counts', counts))
            stack.enter_context(patch.object(S, '_resample', resample))
            stack.enter_context(patch.object(S, '_basic_interval', interval))
            stack.enter_context(redirect_stdout(io.StringIO()))
            S.run(self.source, destination)
        return (json.loads((destination / 'statistics.json').read_text()), destination,
                calls, resamples, coverages)

    @staticmethod
    def equal_market(frame, column):
        return float(frame.groupby('market')[column].mean().mean())

    def test_matched_seed_differences_shared_calendar_and_both_directions(self):
        result, destination, calls, resamples, coverages = self.run_fixture()
        self.assertEqual(calls, [(3, 20000, block, 2026093005) for block in (60, 20, 120)])
        self.assertEqual(result['family_slots'], 12)
        self.assertEqual(len(result['contrasts']), 36)
        self.assertEqual(result['primary_block'], 60)
        self.assertEqual(result['sensitivity_blocks'], [20, 120])
        for excluded in ('online refit', 'training', 'pretraining', 'research-selection', '38-slot'):
            self.assertIn(excluded, result['scope'])
        for block in (60, 20, 120):
            self.assertEqual(len({identity for b, identity in resamples if b == block}), 1)
        for row in result['contrasts']:
            self.assertAlmostEqual(row['p_raw'], 1 / 20001)
            self.assertAlmostEqual(row['p_holm'], 12 / 20001)
            proxy, name = row['proxy'], row['comparator']
            deltas = []
            for seed in S.SEEDS:
                comp = self.equal_market(self.frames[name].loc[self.frames[name].seed.eq(seed)], proxy + '_qlike')
                ref = self.equal_market(self.frames[S.REFERENCE].loc[self.frames[S.REFERENCE].seed.eq(seed)], proxy + '_qlike')
                deltas.append(comp - ref)
            np.testing.assert_allclose(row['matched_seed_gains'], deltas, rtol=1e-12, atol=1e-12)
            self.assertEqual(row['matched_seed_ids'], list(S.SEEDS))
            self.assertAlmostEqual(row['gain'], np.mean(deltas))
            positive = name != NEGATIVE
            self.assertEqual(row['positive_support'], positive)
            self.assertEqual(row['negative_support'], not positive)
            self.assertEqual(row['matched_seeds_positive'], 5 if positive else 0)
            self.assertEqual(row['matched_seeds_negative'], 0 if positive else 5)
        positive = next(row for row in result['contrasts'] if row['comparator'] == POSITIVE)
        comp_mean = np.mean([self.equal_market(self.frames[POSITIVE].loc[self.frames[POSITIVE].seed.eq(s)], 'r2_qlike')
                             for s in S.SEEDS])
        obsolete = [comp_mean - self.equal_market(self.frames[S.REFERENCE].loc[self.frames[S.REFERENCE].seed.eq(s)], 'r2_qlike')
                    for s in S.SEEDS]
        self.assertFalse(np.all(np.asarray(obsolete) > 0))
        self.assertTrue(np.all(np.asarray(positive['matched_seed_gains']) > 0))
        self.assertEqual(coverages.count(.95), 36)
        self.assertEqual(coverages.count(1 - .05 / 12), 36)
        with np.load(destination / 'bootstrap_archive_no_pickle.npz', allow_pickle=False) as archive:
            self.assertEqual(len(archive.files), 36)
            for name in archive.files:
                self.assertEqual(archive[name].shape, (20000,))
        provenance = json.loads((destination / 'provenance.json').read_text())
        self.assertTrue({'python', 'numpy', 'pandas', 'scipy', 'executable'}.issubset(provenance))
        self.assertTrue(any(p.endswith('cross_market_blend_20260930/run_blends.py')
                            for p in provenance['source_hashes']))
        self.assertTrue((destination / 'frozen_sources/research/cross_market_blend_20260930/statistics.py').exists())

    def test_two_sided_centered_plus_one_counts_both_tails(self):
        result, _, _, _, _ = self.run_fixture(tail=True)
        for row in result['contrasts']:
            self.assertAlmostEqual(row['p_raw'], 8001 / 20001)
            self.assertEqual(row['p_holm'], 1.)
            self.assertFalse(row['positive_support'])
            self.assertFalse(row['negative_support'])

    def test_joint_support_requires_both_proxies_same_direction(self):
        f = self.frames[POSITIVE]
        f['rk_forecast'] = f.rk
        f['rk_qlike'] = 1 + np.log(f.rk)
        self.save_model(POSITIVE)
        self.refresh_receipt()
        result, _, _, _, _ = self.run_fixture()
        for row in result['contrasts']:
            if row['comparator'] == POSITIVE:
                self.assertGreater(row['gain'] if row['proxy'] == 'r2' else -row['gain'], 0)
                self.assertFalse(row['positive_support'])
                self.assertFalse(row['negative_support'])

    def test_positive_average_cannot_replace_all_matched_seed_directions(self):
        self.write_model(POSITIVE, [(seed, h, h) for seed, h in zip(S.SEEDS, [39., 100., 100., 100., 100.])])
        self.refresh_receipt()
        result, _, _, _, _ = self.run_fixture()
        for row in result['contrasts']:
            if row['comparator'] == POSITIVE:
                self.assertGreater(row['gain'], 0)
                self.assertGreater(row['family_interval95'][0], 0)
                self.assertLess(row['p_holm'], .05)
                self.assertEqual(row['matched_seeds_positive'], 4)
                self.assertFalse(row['positive_support'])

    def test_reference_and_comparators_require_exact_seeds(self):
        for model in (S.REFERENCE, POSITIVE):
            with self.subTest(model=model):
                self.frames[model]['seed'] = self.frames[model].seed.map({s: str(int(s) + 1000) for s in S.SEEDS})
                self.save_model(model)
                self.refresh_receipt()
                with self.assertRaisesRegex(ValueError, 'seed'):
                    self.run_fixture()
                self.frames[model]['seed'] = self.frames[model].seed.map({str(int(s) + 1000): s for s in S.SEEDS})
                self.save_model(model)

    def test_missing_seed_duplicate_key_extra_key_fail_closed(self):
        original = self.frames[POSITIVE].copy()
        altered = [original.loc[original.seed.ne(S.SEEDS[-1])],
                   pd.concat([original, original.iloc[[0]]], ignore_index=True),
                   original.assign(date=lambda f: f.date.where(f.index != 0, '2019-01-01'))]
        for frame in altered:
            self.frames[POSITIVE] = frame
            self.save_model(POSITIVE)
            self.refresh_receipt()
            with self.assertRaises(ValueError):
                self.run_fixture()

    def test_stored_loss_target_and_nonpositive_forecast_rejected(self):
        original = self.frames[POSITIVE].copy()
        for field, value, error in [('rk_qlike', .01, AssertionError), ('rk', 99., AssertionError),
                                    ('rk_forecast', -1., ValueError)]:
            self.frames[POSITIVE] = original.copy()
            self.frames[POSITIVE].loc[0, field] = value
            if field == 'rk':
                f = self.frames[POSITIVE]
                f['rk_qlike'] = f.rk / f.rk_forecast + np.log(f.rk_forecast)
            self.save_model(POSITIVE)
            self.refresh_receipt()
            with self.assertRaises(error):
                self.run_fixture()

    def test_source_row_hash_change_is_rejected_and_failure_preserved(self):
        self.frames[POSITIVE].loc[0, 'rk_qlike'] += .01
        self.save_model(POSITIVE)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_fixture()
        failure = json.loads((self.folder / 'results_1/FAILED_ATTEMPT.json').read_text())
        self.assertEqual(failure['status'], 'failed')
        self.assertTrue(failure['originals_preserved'])

    def test_missing_model_receipt_and_changed_upstream_source_rejected(self):
        (self.source / 'row_forecasts' / (NEGATIVE + '.csv.gz')).unlink()
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError, 'artifact receipt'):
            self.run_fixture()

    def test_upstream_hash_changes_rejected_even_with_untouched_model_rows(self):
        (self.canonical / 'common_mask.csv').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen upstream source changed'):
            self.run_fixture()

    def test_coordinated_source_mask_change_cannot_replace_canonical(self):
        changed = self.mask.copy()
        changed.loc[0, 'rk'] *= 1.000001
        changed.to_csv(self.source / 'common_mask.csv', index=False)
        for name, f in self.frames.items():
            selected = f.market.eq('A') & f.date.eq('2019-01-02')
            f.loc[selected, 'rk'] = changed.loc[0, 'rk']
            f['rk_qlike'] = f.rk / f.rk_forecast + np.log(f.rk_forecast)
            self.save_model(name)
        self.refresh_receipt()
        with self.assertRaises(AssertionError):
            self.run_fixture()

    def test_design_changes_old_family_and_source_protocol_hash_rejected(self):
        self.design['family_slots'] = 38
        (self.source / 'REGISTERED_DESIGN.json').write_text(json.dumps(self.design))
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError, 'design drift'):
            self.run_fixture()

    def test_output_is_exclusive(self):
        _, destination, _, _, _ = self.run_fixture()
        with self.assertRaises(FileExistsError):
            S.run(self.source, destination)


class DirectContractTests(unittest.TestCase):
    def test_matched_seed_gains_are_equal_market_not_pooled_rows(self):
        reference = np.zeros((5, 3, 2))
        comparator = np.tile(np.array([[1., 1.], [1., 1.], [10., 10.]]), (5, 1, 1))
        np.testing.assert_allclose(S.matched_seed_gains(comparator, reference, ['A', 'A', 'B'], 0), [5.5] * 5)
        with self.assertRaises(ValueError):
            S.matched_seed_gains(comparator[:-1], reference[:-1], ['A', 'A', 'B'], 0)

    def test_calendar_preserves_holidays_and_equal_markets(self):
        matrix = _panel(['d0', 'd1', 'd2', 'd0', 'd2'], ['A', 'A', 'A', 'B', 'B'],
                        [1., 2., 3., 10., 30.], ['d0', 'd1', 'd2'], ['A', 'B'])
        self.assertTrue(np.isnan(matrix[1, 1]))
        estimate, draws, means = _resample(matrix, np.array([[1., 1., 1.], [2., 1., 0.], [0., 1., 2.]]))
        self.assertEqual(estimate, 11.)
        np.testing.assert_allclose(means, [2., 20.])
        np.testing.assert_allclose(draws, [11., 17 / 3, 49 / 3])
        with self.assertRaisesRegex(ValueError, 'omits a market'):
            _resample(matrix, np.array([[0., 1., 0.]]))

    def test_circular_calendar_fixed_length_repeatability(self):
        first = _calendar_counts(11, 12, 4, 345)
        np.testing.assert_array_equal(first, _calendar_counts(11, 12, 4, 345))
        np.testing.assert_array_equal(first.sum(axis=1), [11.] * 12)
        self.assertFalse(first.flags.writeable)
        self.assertFalse(np.array_equal(first, _calendar_counts(11, 12, 4, 346)))

    def test_joint_holm_uses_exact_new_family_and_basic_interval_reflects_errors(self):
        np.testing.assert_allclose(holm([.001, .01] + [1.] * 10)[:2], [.012, .11])
        np.testing.assert_allclose(_basic_interval(2., np.array([1., 2., 3., 4., 5.]), .8), [-.6, 2.6])


if __name__ == '__main__':
    unittest.main()
