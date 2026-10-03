"""CPU checks for the declared simulator substitutions and causal interface.

These tests verify the engineering contract. They do not establish matched
training difficulty, posterior-predictive exactness, or market performance.
Run with one BLAS thread, e.g. ``python -m unittest
research.matched_prior.test_simulators -v``.
"""
import unittest
from unittest import mock

import numpy as np

import volpfn_prior as VP
from research.matched_prior import simulators as S
from research.seed_replication import causal


class CovarianceTests(unittest.TestCase):
    def test_all_nine_declared_cells_reconstruct_unit_marginal_covariance(self):
        length = 24
        lag = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
        for smoothness in S.SMOOTHNESS:
            for scale in S.LENGTH_SCALES:
                with self.subTest(smoothness=smoothness, scale=scale):
                    factor = S.covariance_factor(length, smoothness, scale)
                    covariance = factor @ factor.T
                    reference = VP.matern_cov(lag, smoothness, scale)
                    np.testing.assert_allclose(covariance, reference, atol=5e-13,
                                               rtol=5e-13)
                    np.testing.assert_allclose(np.diag(covariance), 1., atol=5e-13)
                    np.testing.assert_allclose(factor, np.tril(factor), atol=0., rtol=0.)
                    self.assertTrue(np.all(np.diag(factor) > 0.))
                    self.assertFalse(factor.flags.writeable)
                    # Closed forms independently check two of the kernel branches.
                    if smoothness == .5:
                        expected = np.exp(-lag / scale)
                        np.testing.assert_allclose(covariance, expected, atol=5e-13)
                    elif smoothness == 1.5:
                        distance = np.sqrt(3.) * lag / scale
                        expected = (1. + distance) * np.exp(-distance)
                        np.testing.assert_allclose(covariance, expected, atol=5e-13)

    def test_covariance_audit_records_no_jitter(self):
        rows = S.covariance_audit(24)
        self.assertEqual(len(rows), 9)
        self.assertEqual({(r['smoothness'], r['length_scale']) for r in rows},
                         {(h, ell) for h in S.SMOOTHNESS for ell in S.LENGTH_SCALES})
        self.assertTrue(all(r['jitter'] == 0. for r in rows))
        self.assertTrue(all(r['max_reconstruction_error'] < 5e-13 for r in rows))
        self.assertTrue(all(r['min_cholesky_diagonal'] > 0. for r in rows))


class SimulatorTests(unittest.TestCase):
    def test_reproducibility_and_output_contract_for_every_arm(self):
        for arm in S.ARMS:
            with self.subTest(arm=arm):
                first = S.raw_batch(arm, 48017, 6, 48)
                repeated = S.raw_batch(arm, 48017, 6, 48)
                different = S.raw_batch(arm, 48018, 6, 48)
                for key in ('r', 'logv', 'mask'):
                    np.testing.assert_array_equal(first[key], repeated[key])
                    self.assertEqual(first[key].shape, (6, 48))
                    self.assertTrue(np.isfinite(first[key]).all())
                self.assertEqual(first['meta'], repeated['meta'])
                self.assertFalse(np.array_equal(first['r'], different['r']))
                self.assertTrue(np.isin(first['mask'], [0., 1.]).all())

    def test_garch_arm_preserves_declared_overlay_probabilities(self):
        with mock.patch.object(VP, 'apply_overlays', wraps=VP.apply_overlays) as apply:
            batch = S.raw_batch('garch', 41029, 4, 40)
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[3], VP.OVERLAY_P)
        self.assertTrue(all(m['family'] == 'garch' for m in batch['meta']))
        # A family= dispatch would silently supply zero probabilities instead.
        self.assertTrue(all(v > 0. for v in apply.call_args.args[3].values()))

    def test_clean_mode_removes_overlays_for_every_arm(self):
        for arm in S.ARMS:
            with self.subTest(arm=arm):
                with mock.patch.object(VP, 'apply_overlays', wraps=VP.apply_overlays) as apply:
                    batch = S.raw_batch(arm, 42111, 4, 48, clean=True)
                self.assertTrue(all(v == 0. for v in apply.call_args.args[3].values()))
                np.testing.assert_array_equal(batch['mask'], np.ones((4, 48)))
                self.assertTrue(all(m['clean'] for m in batch['meta']))

    def test_clean_variance_labels_recover_gaussian_unit_innovations(self):
        def gaussian_law(_rng, count):
            return np.full(count, VP.NU_GAUSS), np.ones(count)

        # Fix the law to Gaussian so a finite-sample variance check is meaningful;
        # the production prior permits nu < 4 and hence infinite fourth moments.
        for arm in S.ARMS:
            with self.subTest(arm=arm):
                with mock.patch.object(VP, 'draw_law', side_effect=gaussian_law):
                    batch = S.raw_batch(arm, 52307, 64, 256, clean=True)
                innovation = batch['r'] * np.exp(-.5 * batch['logv'])
                self.assertLess(abs(float(innovation.mean())), .04)
                self.assertLess(abs(float(np.mean(innovation ** 2)) - 1.), .07)
                self.assertTrue(all(m['nu'] == VP.NU_GAUSS and m['gamma'] == 1.
                                    for m in batch['meta']))

    def test_invalid_requests_fail(self):
        with self.assertRaises(ValueError):
            S.raw_batch('unknown', 1, 2, 16)
        for arm in S.ARMS:
            for batch_size, length in ((0, 16), (2, 1)):
                with self.subTest(arm=arm, batch_size=batch_size, length=length):
                    with self.assertRaises(ValueError):
                        S.raw_batch(arm, 1, batch_size, length)


class CausalPreparationTests(unittest.TestCase):
    def test_future_observations_cannot_change_earlier_inputs_or_forecast_units(self):
        cut = 19
        for arm in S.ARMS:
            with self.subTest(arm=arm):
                raw = S.raw_batch(arm, 65001, 4, 48, clean=True)
                # Exercise the all-zero fallback followed by a first observation.
                raw['r'][:, :3] = 0.
                raw['r'][:, 3] = np.arange(1., 5.)
                original = causal.prepare_batch(raw)
                changed = {key: value.copy() if isinstance(value, np.ndarray) else value
                           for key, value in raw.items()}
                changed['r'][:, cut:] = 500. - 71. * raw['r'][:, cut:]
                changed['logv'][:, cut:] += 8.
                changed['mask'][:, cut:] = 0.
                mutated = causal.prepare_batch(changed)
                for key in ('feats', 'base', 'output_scale', 'output_log_scale'):
                    np.testing.assert_array_equal(original[key][:, :cut],
                                                  mutated[key][:, :cut])
                for key in ('target_r', 'target_logv', 'target_mask'):
                    np.testing.assert_array_equal(original[key][:, :cut - 1],
                                                  mutated[key][:, :cut - 1])
                np.testing.assert_array_equal(original['output_scale'][:, :3],
                                              np.ones((4, 3)))
                # The first changed target has changed, while its units have not.
                np.testing.assert_allclose(mutated['target_r'][:, cut - 1],
                                           changed['r'][:, cut] /
                                           original['output_scale'][:, cut - 1],
                                           rtol=2e-7)

    def test_targets_and_masks_use_next_observation_for_every_arm(self):
        for arm in S.ARMS:
            with self.subTest(arm=arm):
                raw = S.raw_batch(arm, 65011, 4, 40)
                raw['mask'][:, ::3] = 0.
                prepared = causal.prepare_batch(raw)
                scale = prepared['output_scale']
                self.assertEqual(prepared['target_r'].shape, (4, 39))
                np.testing.assert_allclose(prepared['target_r'], raw['r'][:, 1:] / scale,
                                           rtol=2e-7, atol=1e-7)
                np.testing.assert_allclose(prepared['target_logv'],
                                           raw['logv'][:, 1:] - 2. * np.log(scale),
                                           rtol=2e-7, atol=1e-7)
                np.testing.assert_array_equal(prepared['target_mask'], raw['mask'][:, 1:])
                np.testing.assert_allclose(prepared['output_log_scale'], np.log(scale),
                                           rtol=2e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
