"""Regression checks for safe corrected/legacy checkpoint dispatch."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import volpfn as V
from research import forecast_checkpoint as F
from research.seed_replication import causal


class CheckpointDispatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.model = V.VolPFN(**V.SIZES['tiny']).eval()
        self.returns = np.random.default_rng(12).normal(size=140) * np.linspace(.3, 2., 140)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def checkpoint(self, name, **metadata):
        path = Path(self.directory.name) / name
        torch.save(dict(cfg=self.model.cfg, state_dict=self.model.state_dict(), **metadata), path)
        return path

    def test_corrected_dispatch_matches_final_api_and_differs_from_legacy(self):
        path = self.checkpoint('corrected.pt', causal_spec=causal.CAUSAL_SPEC,
                               normalization_spec=causal.NORMALIZATION_SPEC)
        actual, identity = F.forecast(path, self.returns, L=64, min_history=12, levels=())
        expected = causal.predict_series(self.model, self.returns, L=64, min_history=12, levels=())
        for key in ('h', 'mean', 'logscore', 'loc', 'scale', 'nu'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
        self.assertEqual(identity['preprocessing'], causal.CAUSAL_SPEC)
        legacy = V.predict_series(self.model, self.returns, L=64, min_history=12, levels=())
        self.assertGreater(np.nanmax(np.abs(np.log(actual['h']) - np.log(legacy['h']))), .01)

    def test_legacy_requires_opt_in_and_matches_legacy_api(self):
        path = self.checkpoint('legacy.pt')
        with self.assertRaisesRegex(ValueError, 'allow-legacy'):
            F.load_checkpoint(path)
        actual, identity = F.forecast(path, self.returns, allow_legacy=True, L=64, min_history=12, levels=())
        expected = V.predict_series(self.model, self.returns, L=64, min_history=12, levels=())
        np.testing.assert_allclose(actual['h'], expected['h'], rtol=0, atol=0, equal_nan=True)
        self.assertTrue(identity['preprocessing'].startswith('legacy_'))

    def test_unknown_or_inconsistent_identity_rejected(self):
        for i, metadata in enumerate((dict(causal_spec='unknown'),
                    dict(causal_spec=causal.CAUSAL_SPEC),
                    dict(normalization_spec=causal.NORMALIZATION_SPEC))):
            path = self.checkpoint(f'invalid_{i}.pt', **metadata)
            with self.assertRaisesRegex(ValueError, 'preprocessing metadata'):
                F.load_checkpoint(path, allow_legacy=True)

    def test_nonfinite_parameters_rejected(self):
        self.model.head_mix.bias.data[0] = float('nan')
        path = self.checkpoint('nan.pt')
        with self.assertRaisesRegex(ValueError, 'finite'):
            F.load_checkpoint(path, allow_legacy=True)


if __name__ == '__main__':
    unittest.main()
