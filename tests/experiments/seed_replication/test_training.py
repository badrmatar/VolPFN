import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import training


class TrainingIntegrityTests(unittest.TestCase):
    def configuration(self):
        s = training.spec('pilot', 'cpu')
        s.update(size='tiny', steps=2, tokens=128, lengths={'128': 1.}, workers=0,
                 val_n=2, val_T=128, eval_every=1, log_every=1, warmup=1)
        return s

    def test_exact_budget_and_synthetic_checkpoint_choice(self):
        s = self.configuration()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'seed'
            with patch.object(training, 'evaluate', side_effect=[{'nll': 1.}, {'nll': 2.}]):
                result = training.train_seed(7101, s, out, 'synthetic-test-manifest')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['completed_updates'], 2)
            self.assertEqual(result['input_positions'], 256)
            self.assertEqual(result['selected_step'], 1)
            ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
            self.assertEqual(ck['step'], 1)
            self.assertEqual(ck['normalization_spec'], training.causal.NORMALIZATION_SPEC)
            training.causal.load_model(out/'best.pt', 'cpu')
            with self.assertRaises(FileExistsError):
                training.train_seed(7101, s, out, 'synthetic-test-manifest')

    def test_nonfinite_seed_is_retained_as_failure(self):
        s = self.configuration()
        batch = training.generated_batch(123, 2, 128, s['prior'], s['overlay'])
        batch['feats'][0, 1, 0] = np.nan
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'seed'
            with patch.object(training, 'generated_batch', return_value=batch):
                result = training.train_seed(7101, s, out, 'synthetic-test-manifest')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['completed_updates'], 0)
            self.assertIn('nonfinite_generated_feats', result['failure_code'])
            self.assertTrue((out/'failure.txt').exists())
            self.assertFalse((out/'best.pt').exists())

    def test_batch_sequence_is_reproducible(self):
        s = self.configuration()
        a, b = training.Batches(7101, s), training.Batches(7101, s)
        try:
            for _ in range(3):
                x, y = a.next(), b.next()
                for key in ('feats', 'base', 'target_r', 'target_mask', 'output_log_scale'):
                    np.testing.assert_array_equal(x[key], y[key])
        finally:
            a.close(); b.close()

    def test_interruption_records_attempt_and_is_not_swallowed(self):
        s = self.configuration()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'seed'
            with patch.object(training.Batches, 'next', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    training.train_seed(7101, s, out, 'synthetic-test-manifest')
            record = json.loads((out/'status.json').read_text())
            self.assertEqual(record['status'], 'interrupted')
            self.assertEqual(record['attempted_updates'], 1)
            self.assertEqual(record['completed_updates'], 0)


if __name__ == '__main__':
    unittest.main()
