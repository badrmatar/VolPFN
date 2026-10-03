"""Synthetic driver contracts; never train or forecast a real market."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from . import evaluation as E
from . import run as R
from . import training as T


SEEDS = [8101, 8102, 8103, 8104, 8105]


def synthetic_aggregate_inputs(out):
    source = out / 'synthetic_source'
    E.synthetic_integration(source, reps=19)
    gains = pd.read_csv(source / 'loss_differences.csv.gz', dtype={'market': str, 'date': str})
    gains['contrast'] = gains.family + '::' + gains.comparison
    mask_paths = {}
    for spec in E.comparison_registry():
        key = spec['family'] + '::' + spec['comparison']
        rows = gains.loc[gains.contrast == key, ['market', 'date']]
        if rows.empty:
            # Reserved slots have the same synthetic dates as their family.
            sibling = gains.loc[gains.family == spec['family'], 'contrast'].iloc[0]
            rows = gains.loc[gains.contrast == sibling, ['market', 'date']]
        path = out / 'masks' / (key.replace('::', '__') + '.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False)
        mask_paths[key] = str(path)
    for i, seed in enumerate(SEEDS):
        path = out / 'evaluation' / str(seed)
        path.mkdir(parents=True)
        frame = gains.drop(columns='contrast').copy()
        frame['gain'] += (i - 2) * 1e-5
        frame.to_csv(path / 'loss_differences.csv.gz', index=False)
    return dict(base=dict(seeds=SEEDS), expected_masks=mask_paths,
                statistics=dict(reps=39, block=20, seed=927601, margin=.003))


class DriverContractTests(unittest.TestCase):
    def test_five_fake_outputs_aggregate_all44_slots_and_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = synthetic_aggregate_inputs(out)
            result = R.aggregate(out, manifest, SEEDS, [])
            self.assertEqual(len(result['contrasts']), 44)
            self.assertTrue(result['all_scoring_masks_supplied'])
            self.assertEqual(result['expected_seeds'], list(map(str, SEEDS)))
            self.assertEqual({f: r['size'] for f, r in result['families'].items()},
                             dict(index_density=4, index_qlike=12, calendar_a=12,
                                  calendar_b=8, transfer_stocks=4, transfer_fx=4))
            self.assertEqual(result['equivalence']['contrast'], 'transfer_stocks::VolPFN__R_class')
            self.assertEqual(result['equivalence']['crossed']['status'], 'available')
            unavailable = [r for r in result['contrasts'].values() if r['status'] == 'unavailable']
            self.assertEqual(len(unavailable), 6)
            self.assertTrue(all(r['crossed']['p_null'] == 1 for r in unavailable))
            # save_json is strict about nonfinite values; re-serialization checks
            # the full nested API contract, not only the report's visible rows.
            json.dumps(result, allow_nan=False)
            self.assertTrue((out / 'REPORT.md').exists())

    def test_failed_seed_never_creates_surviving_seed_headline(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = synthetic_aggregate_inputs(out)
            result = R.aggregate(out, manifest, SEEDS[:-1], [SEEDS[-1]])
            self.assertTrue(all(r['status'] == 'unavailable' for r in result['contrasts'].values()))
            self.assertTrue(all(r['crossed']['mean_gain'] is None for r in result['contrasts'].values()))
            self.assertFalse(result['equivalence']['crossed']['equivalence_supported'])

    def test_all_base_seeds_terminal_before_first_market_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'manifest.json').write_text('{}')
            manifest = dict(base=dict(seeds=SEEDS), evaluation_inventory={},
                            inference=dict(device='cpu', threads=2),
                            statistics=dict(reps=19, block=10, seed=927601, margin=.003))
            events = []
            def train(seed, spec, path, digest):
                events.append(('train', seed))
                path.mkdir(parents=True)
                (path / 'best.pt').write_bytes(b'synthetic-not-a-torch-checkpoint')
                (path / 'final.pt').write_bytes(b'synthetic-not-a-torch-checkpoint')
                record = dict(status='failed' if seed == SEEDS[2] else 'complete', seed=seed,
                              specification=spec, manifest_sha256=digest,
                              best_checkpoint_sha256=T.sha(path/'best.pt'),
                              final_checkpoint_sha256=T.sha(path/'final.pt'))
                T.save_json(path / 'status.json', record)
                return record
            def evaluate(checkpoint, path, **kwargs):
                self.assertEqual([s for kind, s in events if kind == 'train'], SEEDS)
                events.append(('evaluate', int(checkpoint.parent.name)))
                path.mkdir(parents=True)
                T.save_json(path / 'completed_artifacts.json',
                            dict(checkpoint_sha256=T.sha(checkpoint), hashes={}))
            with patch.object(R, 'verified_manifest', return_value=manifest), \
                 patch.object(R, 'pilot', return_value={'status': 'passed'}), \
                 patch.object(T, 'train_seed', side_effect=train), \
                 patch.object(E, 'evaluate_checkpoint', side_effect=evaluate), \
                 patch.object(R, 'aggregate', return_value={}):
                R.run_all(out)
            self.assertEqual(events[:5], [('train', s) for s in SEEDS])
            self.assertEqual(events[5:], [('evaluate', s) for s in SEEDS if s != SEEDS[2]])
            status = json.loads((out / 'status.json').read_text())
            self.assertEqual(status['status'], 'complete_with_failures')
            self.assertEqual(status['failed_seeds'], [SEEDS[2]])

    def test_completed_driver_refuses_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'manifest.json').write_text('{}')
            T.save_json(out / 'status.json', dict(status='complete'))
            with patch.object(R, 'verified_manifest', return_value={}), \
                 patch.object(R, 'pilot', side_effect=AssertionError('Must not restart')):
                with self.assertRaises(RuntimeError):
                    R.run_all(out)

    def test_completed_checkpoint_tampering_is_rejected_on_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'best.pt').write_bytes(b'original')
            (out/'final.pt').write_bytes(b'final')
            record = dict(status='complete', seed=8101, manifest_sha256='frozen', specification={'steps':20},
                          best_checkpoint_sha256=T.sha(out/'best.pt'), final_checkpoint_sha256=T.sha(out/'final.pt'))
            T.save_json(out/'status.json',record)
            self.assertEqual(R.existing_seed(out,manifest_sha='frozen',specification={'steps':20},seed=8101),record)
            with self.assertRaises(RuntimeError):
                R.existing_seed(out,manifest_sha='wrong')
            (out/'best.pt').write_bytes(b'replaced')
            with self.assertRaises(RuntimeError):
                R.existing_seed(out,manifest_sha='frozen')

    def test_failed_pilot_gate_is_recorded_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'manifest.json').write_text('{}')
            manifest = {'pilot':{'seeds':[7101,7102]}}
            def fake_train(seed,spec,path,digest):
                path.mkdir(parents=True)
                record={'status':'complete','seed':seed}
                T.save_json(path/'status.json',record)
                return record
            with patch.object(R,'verified_manifest',return_value=manifest), \
                 patch.object(T,'train_seed',side_effect=fake_train) as trainer, \
                 patch.object(R,'checkpoint_probe',side_effect=ValueError('synthetic gate failure')) as probe:
                first=R.pilot(out)
                second=R.pilot(out)
            self.assertEqual(first['status'],'failed')
            self.assertEqual(second['status'],'failed')
            self.assertEqual(trainer.call_count,1)
            self.assertEqual(probe.call_count,1)
            self.assertEqual(len(first['gate_failures']),1)


if __name__ == '__main__':
    unittest.main()
