import unittest
import numpy as np
from research.intraday_extension_20260930.qlike_har import causal_features, fit


class QlikeHarTests(unittest.TestCase):
    def test_feature_changes_never_go_backwards(self):
        risk = np.arange(1., 81.)
        first, _ = causal_features(risk)
        risk[60:] *= 10
        other, _ = causal_features(risk)
        np.testing.assert_array_equal(first[:61], other[:61])
        self.assertFalse(np.array_equal(first[61:], other[61:]))

    def test_missing_inputs_carried_but_target_not_imputed(self):
        risk = np.arange(1., 81.)
        risk[30] = np.nan
        x, valid = causal_features(risk)
        self.assertFalse(valid[30])
        self.assertEqual(x[31, 0], risk[29])
        self.assertEqual(x[30, 0], risk[29])

    def test_scale_equivariance_and_positive_fit(self):
        rng = np.random.default_rng(4)
        x = np.exp(rng.normal(size=(250, 3)))
        y = 0.1 + x @ np.array([0.6, 0.2, 0.1])
        value, coef, scale, attempts = fit(x, y, np.array([1., 2., 3.]))
        other, _, _, _ = fit(x / 10000, y / 10000, np.array([1., 2., 3.]) / 10000)
        self.assertAlmostEqual(value, other * 10000, places=6)
        self.assertTrue(attempts[-1]['success'])
        self.assertAlmostEqual(value, 1.4, places=4)

    def test_bad_training_is_rejected_not_omitted(self):
        x, y = np.ones((150, 3)), np.ones(150)
        y[4] = -1.
        with self.assertRaises(ValueError):
            fit(x, y, np.ones(3))


if __name__ == '__main__':
    unittest.main()
