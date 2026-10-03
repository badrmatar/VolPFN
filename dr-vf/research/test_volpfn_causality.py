"""Forecast invariance tests for the actual public VolPFN prediction API."""
import unittest

import numpy as np
import torch

import volpfn as V
import volpfn_prior as VP


class CausalPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(927401)
        cls.model = V.VolPFN(d=16, layers=1, heads=2, K=2).eval()

    def predict(self, r, **kwargs):
        return V.predict_series(self.model, r, levels=(), **kwargs)

    def assert_forecasts_equal(self, a, b, through):
        for name in ('logw', 'loc', 'scale', 'nu', 'h', 'h_lat'):
            np.testing.assert_allclose(a[name][:through], b[name][:through],
                                       rtol=2e-5, atol=2e-6, equal_nan=True,
                                       err_msg=name)

    def test_sparse_scale_never_uses_later_returns(self):
        a = np.zeros(250); a[2] = 3.; a[20] = -4.
        b = a.copy(); b[100:] = 1e6
        self.assertEqual(VP.initial_scale(a), VP.initial_scale(b))
        self.assertAlmostEqual(VP.initial_scale(a), np.sqrt(12.5))
        self.assertEqual(VP.initial_scale(np.r_[np.zeros(100), 1e9]), 1.)

    def test_suffix_changes_cannot_change_forecasts_at_or_before_cut(self):
        r = np.random.default_rng(1).normal(size=290)
        # Sparse initialization in a later overlapping window.
        r[120:220] = 0.; r[123] = 3.
        full = self.predict(r, L=240, stride=120)
        for t in (1, 5, 49, 99, 100, 150, 240, 260):
            changed = r.copy(); changed[t:] = np.linspace(-50, 70, len(r)-t)
            other = self.predict(changed, L=240, stride=120)
            self.assert_forecasts_equal(full, other, t+1)

    def test_prefix_truncation_and_warmup_skipping(self):
        r = np.random.default_rng(2).normal(size=180)
        full = self.predict(r, L=64, stride=20)
        for t in (2, 19, 20, 63, 64, 101, 150):
            short = self.predict(r[:t+1], L=64, stride=20)
            self.assert_forecasts_equal(full, short, t+1)
        skipped = self.predict(r, L=64, stride=20, min_history=75)
        self.assertTrue(np.isnan(skipped['h'][:75]).all())
        for name in ('logw', 'loc', 'scale', 'nu', 'h', 'h_lat'):
            np.testing.assert_allclose(full[name][75:], skipped[name][75:],
                                       rtol=2e-5, atol=2e-6)

    def test_empty_short_and_zero_histories(self):
        for n in (0, 1, 2, 12):
            out = self.predict(np.zeros(n), L=16)
            self.assertEqual(len(out['h']), n)
            self.assertTrue(np.isfinite(out['h'][1:]).all())

    def test_invalid_window_and_input(self):
        for kw in ({'L': 1}, {'L': 10, 'stride': 10}, {'min_history': 0}):
            with self.assertRaises(ValueError):
                self.predict(np.ones(20), **kw)
        with self.assertRaises(ValueError):
            self.predict(np.array([1., np.nan]))


if __name__ == '__main__':
    unittest.main()
