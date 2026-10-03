"""Compare the packaged implementation with pre-reorganisation outputs."""
from pathlib import Path
import unittest

import numpy as np
import torch

from volpfn import VolPFN, data, inference, legacy, losses, preprocessing


class ReorganisationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2703)
        self.model = VolPFN(d=16, layers=1, heads=2, K=2).eval()
        self.expected = np.load(Path(__file__).parent / 'fixtures/core_expected.npz')
        self.addCleanup(self.expected.close)

    def test_simulation_features_and_losses_match_original(self):
        raw = data.raw_batch(273, 2, 16)
        batch = preprocessing.prepare_batch(raw)
        for key in ('r', 'logv'):
            np.testing.assert_array_equal(raw[key], self.expected['raw_' + key])
        for key in ('feats', 'base'):
            np.testing.assert_array_equal(batch[key], self.expected[key])
        with torch.no_grad():
            prediction = self.model(torch.tensor(batch['feats']), torch.tensor(batch['base']))
            actual = losses.losses(prediction, preprocessing.to_dev(batch, 'cpu'))
        np.testing.assert_allclose([float(x) for x in actual], self.expected['loss'], rtol=1e-6)

    def test_corrected_and_legacy_forecasts_match_original(self):
        for prefix, predictor in [('prediction_', inference.predict_series),
                                  ('legacy_', legacy.predict_series)]:
            result = predictor(self.model, self.expected['r'], L=16, stride=5,
                               levels=(.975,), min_history=4)
            for key, value in result.items():
                np.testing.assert_allclose(value, self.expected[prefix + key],
                                           rtol=1e-6, atol=1e-8, equal_nan=True)
