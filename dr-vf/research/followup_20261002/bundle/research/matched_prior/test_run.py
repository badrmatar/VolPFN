"""Allocation, selection, causality and failure tests for the separate runner."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import run as R


class RunnerTests(unittest.TestCase):
    def config(self):
        s = R.specification()
        s.update(steps=3, tokens=64, lengths={'32': 1.}, warmup=1,
                 cfg=dict(d=16, layers=1, heads=4, K=4),
                 val_T=32, val_n=2, validation_burn=8, eval_every=1, log_every=1)
        return s

    def test_shared_schedule_and_exact_allocation(self):
        s = self.config()
        a, b = list(R.schedule(9101, s)), list(R.schedule(9101, s))
        self.assertEqual(a, b)
        self.assertNotEqual(a, list(R.schedule(9102, s)))
        self.assertEqual(sum(batch * length for _, batch, length in a), s['tokens'] * s['steps'])
        s['tokens'] = 65
        with self.assertRaises(ValueError):
            list(R.schedule(9101, s))

    def test_fixed_final_checkpoint_not_validation_best_and_reloads(self):
        s = self.config()
        validation = [dict(equal_arm_nll=nll, cells={a: {'nll': nll} for a in R.S.ARMS})
                      for nll in (2., 1., 1.5, 1.8)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'arm'
            with patch.object(R, 'evaluate_panels', side_effect=validation):
                result = R.train_arm('broad', 9101, s, out, 'test', {})
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['final_step'], 3)
            self.assertEqual(result['input_positions'], 192)
            checkpoint = torch.load(out/'final.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['step'], 3)
            self.assertEqual(checkpoint['training_spec']['arm'], 'broad')
            self.assertEqual(checkpoint['history'][-1]['equal_arm_nll'], 1.8)
            self.assertFalse((out/'best.pt').exists())
            self.assertEqual(result['checkpoint_probe']['status'], 'passed')
            with self.assertRaises(FileExistsError):
                R.train_arm('broad', 9101, s, out, 'test', {})

    def test_real_validation_and_same_initialization_across_arms(self):
        s = self.config()
        s['steps'] = 1
        panels = R.shared_panels(s)
        with tempfile.TemporaryDirectory() as d:
            records = [R.train_arm(a, 9101, s, Path(d)/a, 'test', panels) for a in R.S.ARMS]
        self.assertTrue(all(r['status'] == 'complete' for r in records))
        self.assertEqual(len({r['initial_state_sha256'] for r in records}), 1)
        self.assertEqual(len({r['initial_validation_nll'] for r in records}), 1)
        self.assertTrue(all(r['input_positions'] == s['tokens'] for r in records))

    def test_nonfinite_batch_records_failure_without_checkpoint_substitution(self):
        s = self.config()
        panels = R.shared_panels(s)
        broken = R.S.prepared_batch('garch', 85004, 2, 32)
        broken['target_r'][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'arm'
            with patch.object(R.S, 'prepared_batch', return_value=broken):
                result = R.train_arm('garch', 9101, s, out, 'test', panels)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['completed_updates'], 0)
            self.assertEqual(result['attempted_updates'], 1)
            self.assertIn('nonfinite_generated_target_r', result['failure_code'])
            self.assertFalse((out/'final.pt').exists())
            self.assertTrue((out/'failure.txt').exists())

    def test_interruption_is_retained_and_reraised(self):
        s = self.config()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'arm'
            with patch.object(R, 'evaluate_panels', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    R.train_arm('matern', 9101, s, out, 'test', {})
            saved = json.loads((out/'status.json').read_text())
            self.assertEqual(saved['status'], 'interrupted')
            self.assertFalse((out/'final.pt').exists())

    def test_freeze_rejects_overwrite_and_detects_snapshot_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'frozen'
            manifest = R.freeze(out)
            self.assertEqual(R.verify(out), manifest)
            with self.assertRaises(FileExistsError):
                R.freeze(out)
            snapshot = out/'source'/next(iter(manifest['source_hashes']))
            snapshot.write_text(snapshot.read_text() + '\n# corrupted test snapshot\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                R.verify(out)

    def test_manifest_corruption_detected(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'frozen'
            R.freeze(out)
            path = out/'manifest.json'
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                R.verify(out)

    def test_full_spec_does_not_reuse_replication_seeds_or_selector(self):
        s = R.specification('full', 'mps')
        self.assertEqual(s['cfg'], R.volpfn.SIZES['base'])
        self.assertEqual(s['seeds'], [9201, 9202, 9203])
        self.assertEqual(s['steps'], 20000)
        self.assertIn('fixed final', s['checkpoint_selection'])
        self.assertFalse(set(s['seeds']) & set(R.T.BASE_SEEDS))

    def test_incomplete_battery_cannot_pass_gate(self):
        s = self.config()
        self.assertFalse(R.summarize([], s)['engineering_gate'])
        records = [dict(status='complete', seed=seed, arm=arm, initial_state_sha256=str(seed),
                        input_positions=s['tokens'] * s['steps'])
                   for seed in s['seeds'] for arm in R.S.ARMS]
        self.assertTrue(R.summarize(records, s)['engineering_gate'])
        records[-1]['status'] = 'failed'
        self.assertFalse(R.summarize(records, s)['engineering_gate'])


if __name__ == '__main__':
    unittest.main()
