"""Synthetic runner contracts. No market inference, downloads or paid compute."""
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from . import run as runner
from .prepare import model_specs
from experiments.combined_leaderboard.common import ROOT, read, save, sha


def fixture():
    rows = []
    for market in ('FTSE', 'N225', 'SPX'):
        for i in range(24):
            rows.append(dict(market=market, date=(date(2012, 1, 4) + timedelta(days=i)).isoformat(),
                phase='calibration', index=1024+i, r2=1., rk=1., primary_eligible=True))
        for i in range(2):
            rows.append(dict(market=market, date=(date(2019, 1, 1) + timedelta(days=i)).isoformat(),
                phase='evaluation', index=1100+i, r2=1., rk=1., primary_eligible=True))
    return dict(rows=rows, markets=['FTSE', 'N225', 'SPX'], model_specs=model_specs(), inputs={},
                start='2019-01-01', end='2022-02-25', exposure='synthetic historical fixture',
                settings=dict(resamples=20, blocks=[2, 1, 3], bootstrap_seed=10, primary_family_slots=32))


def history(n=1200):
    dates = np.array([str(date(2015, 1, 1) + timedelta(days=i)) for i in range(n)])
    return dict(name='X', date=dates, r=np.linspace(-1., 1., n), rk=np.linspace(.1, 2., n))


def history_bundle(data):
    rows = [dict(market='X', date=str(data['date'][i]), index=i,
                 phase='calibration' if j == 0 else 'evaluation',
                 r2=float(data['r'][i])**2, rk=float(data['rk'][i]), primary_eligible=True)
            for j, i in enumerate((1024, 1180, 1190))]
    return dict(rows=rows, markets=['X'])


class PersistenceTests(unittest.TestCase):
    def test_atomic_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.json'
            runner.atomic(path, {'value': 1})
            with self.assertRaises(FileExistsError): runner.atomic(path, {'value': 2})
            self.assertEqual(read(path), {'value': 1})

    def test_cell_resume_hash_identity_and_overwrite_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); rel = runner.relative_cell('ttm_rk', None, 'SPX')
            self.assertIsNone(runner.cell(out, rel, 'freeze'))
            value = runner.cell(out, rel, 'freeze', {'rows': [], 'audit': {}})
            self.assertEqual(runner.cell(out, rel, 'freeze'), value)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                runner.cell(out, rel, 'freeze', {'rows': []})
            with self.assertRaisesRegex(ValueError, 'another freeze'):
                runner.cell(out, rel, 'other')
            (out / 'cells' / rel / 'forecast.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                runner.cell(out, rel, 'freeze')

    def test_partial_cell_and_path_escape_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); (out / 'cells' / 'partial').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Unsealed'):
                runner.cell(out, 'partial', 'freeze')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                runner.cell(out, '../elsewhere', 'freeze')

    def test_run_requires_freeze_before_any_adapter(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'forecast_market') as forecast:
            with self.assertRaisesRegex(ValueError, 'requires an approved frozen'):
                runner.run(Path(folder), SimpleNamespace())
            forecast.assert_not_called()

    def test_freeze_binds_approval_test_sources_and_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); bundle = fixture()
            save(out / 'source_bundle.json', bundle)
            bundle.update(source_bundle_path=str(out / 'source_bundle.json'),
                          source_bundle_sha256=sha(out / 'source_bundle.json'))
            bundle['inputs'][str(out / 'source_bundle.json')] = sha(out / 'source_bundle.json')
            save(out / 'bundle.json', bundle)
            (out / 'PROTOCOL.md').write_text('Synthetic protocol')
            core = [runner.HERE / name for name in ('run.py', 'prepare.py', 'econometric.py', 'ttm.py', 'statistics.py', 'supervise_local.py')]
            hashes = {str(p.relative_to(ROOT)): sha(p) for p in core}
            save(out / 'TESTS.json', dict(passed=True, source_hashes=hashes))
            save(out / 'APPROVAL.json', dict(approved=True, bundle_sha256='wrong', protocol_sha256=sha(out / 'PROTOCOL.md')))
            with patch.object(runner, 'source_paths', return_value=core):
                with self.assertRaisesRegex(ValueError, 'Approval predates bundle'):
                    runner.freeze(out)
                changed = read(out / 'bundle.json')
                changed['rows'][0]['date'] = '2012-01-03'
                save(out / 'bundle.json', changed, replace=True)
                with self.assertRaisesRegex(ValueError, 'Calibration keys differ'):
                    runner.freeze(out)
                save(out / 'bundle.json', bundle, replace=True)
                save(out / 'APPROVAL.json', dict(approved=True, bundle_sha256=sha(out / 'bundle.json'),
                    protocol_sha256=sha(out / 'PROTOCOL.md')), replace=True)
                runner.freeze(out)
                actual, sealed = runner.verify(out)
                self.assertEqual(actual, bundle)
                self.assertIn(str(out / 'APPROVAL.json'), sealed['sources'])
                with self.assertRaisesRegex(ValueError, 'Already frozen'): runner.freeze(out)
                (out / 'PROTOCOL.md').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'Frozen content changed'): runner.verify(out)


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(batch_size=2, threads=2, device='cpu')
        self.data = history(); self.bundle = history_bundle(self.data)

    def test_full_canonical_history_has_no_2016_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'market.npz'
            np.savez(path, **{k: self.data[k] for k in ('date', 'r', 'rk')})
            bundle = dict(self.bundle, market_paths={'X': str(path)})
            observed = runner.market_data(bundle, 'X')
            self.assertEqual(len(observed['r']), 1200)
            self.assertGreater(observed['date'][-1], '2016-06-30')
            self.assertEqual(len(runner.origins_for(bundle, 'X', observed)), 3)

    def test_canonical_metadata_tampering_is_rejected(self):
        self.bundle['rows'][1]['r2'] = 200.
        with self.assertRaisesRegex(ValueError, 'target mismatch'):
            runner.origins_for(self.bundle, 'X', self.data)

    def test_classical_full_union_single_call_and_flags(self):
        def fake(data, model, origins):
            self.assertEqual(len(data['r']), 1200)
            self.assertEqual(origins.tolist(), [1024, 1180, 1190])
            return dict(mean=np.array([.1, .2, .3]), variance=np.ones(3),
                second_moment=1+np.array([.1, .2, .3])**2,
                substituted=np.array([False, True, False]), floor_applied=np.array([False, False, True]), audit={'model': model})
        with patch.object(runner, 'market_data', return_value=self.data), patch('experiments.combined_leaderboard.classical.forecast', side_effect=fake) as method:
            rows, audit = runner.forecast_market(self.bundle, 'hist250', None, 'X', self.args, {})
        self.assertEqual(method.call_count, 1)
        self.assertTrue(rows[1]['substituted']); self.assertTrue(rows[2]['floor_applied'])
        self.assertEqual(rows[0]['phase'], 'calibration')
        self.assertEqual(rows[0]['index'], 1024)
        runner.validate_rows(rows, self.bundle['rows'], 'hist250', None)
        rows[0]['rk'] = 1234.
        with self.assertRaisesRegex(ValueError, 'Changed canonical'):
            runner.validate_rows(rows, self.bundle['rows'], 'hist250', None)

    def test_foundation_uses_strict_past_squared_track_and_preserves_negative_point(self):
        class Fake:
            def __init__(self): self.contexts = []
            def predict(self, past, track):
                self.contexts.append(past.copy()); self.track = track
                return dict(point=-2., point_semantics='test median')
        fake = Fake()
        self.bundle['foundation_models'] = {'chronos2': {'provenance': {'revision': 'test'}}}
        with patch.object(runner, 'market_data', return_value=self.data):
            rows, _ = runner.forecast_market(self.bundle, 'chronos2', None, 'X', self.args, {'chronos2': fake})
        self.assertEqual(fake.track, 'squared')
        for context, base in zip(fake.contexts, self.bundle['rows']):
            np.testing.assert_array_equal(context, self.data['r'][base['index']-1024:base['index']])
        self.assertTrue(all(r['floor_applied'] and not r['substituted'] for r in rows))
        self.assertTrue(all(r['mean'] == 0 and r['variance'] == r['second_moment'] for r in rows))
        self.assertEqual(rows[0]['raw_native_point'], -2.)
        changed = {k: v.copy() if hasattr(v, 'copy') else v for k, v in self.data.items()}
        # Future perturbation can change later forecasts, never the first one.
        changed['r'][1025:] = 999.
        single_bundle = dict(self.bundle, rows=[self.bundle['rows'][0]])
        with patch.object(runner, 'market_data', return_value=changed):
            again, _ = runner.forecast_market(single_bundle, 'chronos2', None, 'X', self.args, {'chronos2': Fake()})
        self.assertEqual(rows[0], again[0])

    def test_econometric_aliases_and_diagnostic_primary_roster(self):
        def fake(rk, origins, config):
            self.assertEqual(config.model, 'mshar_sqrt')
            return [dict(origin=int(i), variance=2., metadata=dict(substituted=False)) for i in origins]
        with patch.object(runner, 'market_data', return_value=self.data), patch('experiments.frontier_extension_20260926.econometric.forecast_series', side_effect=fake):
            rows, _ = runner.forecast_market(self.bundle, 'ms_har', None, 'X', self.args, {})
        self.assertTrue(all(r['model'] == 'ms_har' and r['variance'] == 2 for r in rows))
        specs = model_specs()
        self.assertEqual(2*sum(s['primary'] for m, s in specs.items() if m != 'volpfn'), 32)
        self.assertEqual(specs['volpfn']['seeds'], [8101, 8102, 8103, 8104, 8105])
        self.assertTrue(all(s['seeds'] == [None] for m, s in specs.items() if m != 'volpfn'))
        self.assertFalse(specs['ttm_sqrt_rk']['primary']); self.assertFalse(specs['loghar_meanlog']['primary'])

    def test_record_rejects_incoherent_nonfinite_and_metadata_overwrite(self):
        row = self.bundle['rows'][0]
        for values in ((0., 0., 0.), (1., 1., 1.), (0., np.nan, np.nan)):
            with self.assertRaises(ValueError): runner.record(row, 'test', None, values)
        with self.assertRaisesRegex(ValueError, 'overwrite canonical'):
            runner.record(row, 'test', None, (0., 1., 1.), r2=2.)


class AnalysisContracts(unittest.TestCase):
    def test_report_includes_all_models_and_joint_family_without_market_scores(self):
        from .statistics import analyse
        bundle = fixture()
        rows = [runner.record(base, model, seed, (0., 1., 1.))
            for model, spec in bundle['model_specs'].items() for seed in spec['seeds'] for base in bundle['rows']]
        result = analyse(rows, bundle['model_specs'], bundle=bundle)
        with tempfile.TemporaryDirectory() as folder:
            runner.write_report(result, folder, bundle)
            table = read(Path(folder) / 'leaderboard.json')
            text = (Path(folder) / 'REPORT.md').read_text()
        self.assertEqual(len(table), 19)
        self.assertIn('32 tests', text)
        self.assertIn('2019-01-01 to 2022-02-25', text)
        self.assertIn('ttm_sqrt_rk', text)
        self.assertEqual(result['settings']['blocks'], [2, 1, 3])


class SupervisorContracts(unittest.TestCase):
    def test_cleanup_kills_and_reaps_child_ignoring_term(self):
        from .supervise_local import stop_children
        with tempfile.TemporaryDirectory() as folder:
            log = (Path(folder) / 'log').open('x')
            process = subprocess.Popen([sys.executable, '-B', '-c',
                'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'],
                stdout=subprocess.PIPE, text=True, start_new_session=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), 'ready')
                stop_children([dict(process=process, log=log)], grace_seconds=.05)
                self.assertIsNotNone(process.poll())
                self.assertTrue(log.closed)
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
                process.stdout.close()
                log.close()

    def test_declared_environments_limits_and_finite_job_commands(self):
        from .supervise_local import build_jobs, deadline_timestamp
        bundle = fixture()
        bundle['runtime_requirements'] = {m: dict(executable='/declared/'+m+'/python')
            for m in bundle['model_specs'] if m != 'tabpfn_ts'}
        jobs = build_jobs(bundle, '/tmp/synthetic', models=['ttm_rk', 'volpfn'], seeds=[8102], markets=['SPX'])
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]['command'][0], '/declared/ttm_rk/python')
        self.assertEqual(jobs[0]['command'][-1], '64')
        self.assertEqual(jobs[1]['command'][-2:], ['--seeds', '8102'])
        self.assertTrue(all(j['markets'] == ['SPX'] for j in jobs))
        with self.assertRaises(ValueError): build_jobs(bundle, '/tmp/x', models=['tabpfn_ts'])
        with self.assertRaises(ValueError): build_jobs(bundle, '/tmp/x', models=['volpfn'], seeds=[0])
        self.assertEqual(deadline_timestamp(1000., 1.), 4600.)
        self.assertEqual(deadline_timestamp(1000., 1., '1970-01-01T00:20:00+00:00'), 1200.)
        for hours in (0., 9.):
            with self.assertRaises(ValueError): deadline_timestamp(1000., hours)
        with self.assertRaises(ValueError): deadline_timestamp(1000., 1., '1970-01-01T00:20:00')
        with self.assertRaises(ValueError): deadline_timestamp(1000., 1., '1970-01-01T00:00:01+00:00')

    def test_runtime_requirements_match_package_subset_and_python_patch(self):
        actual = dict(python='3.12.12', packages=dict(torch='2.14.0', numpy='2.5.3', other='1'))
        expected = dict(python='3.12.12', packages=dict(torch='2.14.0', numpy='2.5.3'))
        self.assertTrue(runner.runtime_matches(actual, expected))
        self.assertFalse(runner.runtime_matches(actual, dict(expected, python='3.12.14')))
        self.assertFalse(runner.runtime_matches(actual, dict(packages=dict(torch='2.8.0'))))


if __name__ == '__main__':
    unittest.main()
