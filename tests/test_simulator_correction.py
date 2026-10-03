"""Regression checks for the observed stale-price aggregation failure."""
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.special import logsumexp

from volpfn import prior as VP
from volpfn import data, training

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(__file__).parent / 'fixtures/legacy_prior.py'


def archived_prior():
    spec = importlib.util.spec_from_file_location('archived_volpfn_prior', ARCHIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failing_arguments():
    specification = training.spec()
    specification['workers'] = 0
    batches = training.Batches(8103, specification)
    for _ in range(4575):
        args = batches.args()
    return args


class SimulatorCorrectionTests(unittest.TestCase):
    def test_exact_failed_draw_raises_before_and_is_finite_after(self):
        args = failing_arguments()
        self.assertEqual(args[:3], (810328884, 8, 1024))
        with patch.object(data, 'VP', archived_prior()), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaisesRegex(FloatingPointError, 'raw simulator'):
                training.generated_batch(*args)
        prepared = training.generated_batch(*args)
        training.validate_batch(prepared)
        for value in prepared.values():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.isfinite(value).all())

    def test_all_fresh_is_identity_even_after_huge_shocks(self):
        r = np.array([1e20, 2., -3., 1e-20])
        lv = np.array([1000., -1000., 4., -4.])
        observed, observed_lv = VP._aggregate_stale_intervals(r, lv, np.arange(4))
        np.testing.assert_array_equal(observed, r)
        np.testing.assert_array_equal(observed_lv, lv)

    def test_segment_sums_match_independent_reference_and_exclude_trailing_days(self):
        r = np.array([1., 2., 3., 4., 5., 6., 999.])
        lv = np.log(np.array([2., 3., 5., 7., 11., 13., 999.]))
        fresh = np.array([1, 2, 5])
        observed, observed_lv = VP._aggregate_stale_intervals(r, lv, fresh)
        start = 0
        for end in fresh:
            self.assertAlmostEqual(observed[end], math.fsum(r[start:end + 1]))
            self.assertAlmostEqual(observed_lv[end], logsumexp(lv[start:end + 1]))
            start = end + 1
        np.testing.assert_array_equal(observed[[0, 3, 4, 6]], 0.)
        np.testing.assert_array_equal(observed_lv[[0, 3, 4, 6]], lv[[0, 3, 4, 6]])
        self.assertEqual(observed.sum(), r[:6].sum())

    def test_interval_log_variance_remains_finite_outside_exp_range(self):
        r = np.array([1., 2., -1., 4.])
        lv = np.array([1000., 999., -1000., -1001.])
        _, observed_lv = VP._aggregate_stale_intervals(r, lv, np.array([1, 3]))
        np.testing.assert_allclose(observed_lv[[1, 3]],
                                   [logsumexp(lv[:2]), logsumexp(lv[2:])], rtol=0, atol=1e-12)

    def test_ordinary_overlays_preserve_rng_metadata_masks_and_numerical_values(self):
        old = archived_prior()
        for seed in range(12):
            source = np.random.default_rng(seed)
            lv = source.normal(0., 1., (8, 1024))
            r = np.exp(lv / 2) * source.standard_normal(lv.shape)
            rng_old = np.random.default_rng(seed + 901)
            rng_new = np.random.default_rng(seed + 901)
            settings = dict(jumps=.3, ar=.3, drift=.5, stale=1.)
            before = old.apply_overlays(rng_old, r, lv, settings)
            after = VP.apply_overlays(rng_new, r, lv, settings)
            np.testing.assert_allclose(after[0], before[0], rtol=1e-10, atol=2e-12)
            np.testing.assert_allclose(after[1], before[1], rtol=1e-9, atol=1e-10)
            np.testing.assert_array_equal(after[2], before[2])
            self.assertEqual(after[3], before[3])
            self.assertEqual(rng_old.bit_generator.state, rng_new.bit_generator.state)

    def test_no_stale_overlay_is_bitwise_unchanged(self):
        old = archived_prior()
        rng = np.random.default_rng(9901)
        lv = rng.normal(size=(8, 1024))
        r = np.exp(lv / 2) * rng.normal(size=lv.shape)
        settings = dict(jumps=.3, ar=.3, drift=.5, stale=0.)
        before = old.apply_overlays(np.random.default_rng(123), r, lv, settings)
        after = VP.apply_overlays(np.random.default_rng(123), r, lv, settings)
        for a, b in zip(before[:3], after[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(before[3], after[3])


if __name__ == '__main__':
    unittest.main()
