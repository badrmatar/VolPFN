"""Synthetic remote contracts; never loads TabPFN or forecasts market data."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from research.frontier_extension_20260926 import remote_tabpfn as worker


class FakeAdapter:
    def __init__(self, point=2.0):
        self.calls = []
        self.point = point

    def predict(self, past, track, x, xn):
        if track != 'squared' or x.shape != (1023, 8) or xn.shape != (1, 8):
            raise AssertionError('Adapter contract changed')
        self.calls.append(past.copy())
        return dict(point=self.point, quantiles=np.linspace(0, 4, 9),
                    point_semantics='TabPFN-TS explicit mean output', rearranged=False)


class RemoteContracts(unittest.TestCase):
    def fixture(self, root):
        r = np.random.default_rng(99).normal(size=1200)
        r[::17] = 0
        rk = r*r + .2
        dates = (np.datetime64('2000-01-01') + np.arange(len(r))).astype('U10')
        np.savez(root/'market.npz', date=dates, r=r, rk=rk)
        rows = [dict(market='M',date=str(dates[t]),index=t,
                     phase='calibration' if t < 1096 else 'evaluation',
                     r2=float(r[t])**2,rk=float(rk[t]),primary_eligible=True)
                for t in range(1024,1110)]
        (root/'checkpoint.bin').write_bytes(b'synthetic-checkpoint')
        source=Path(__file__).resolve().parents[1]/'outputs/runpod_tabpfn_bundle_v1/tabpfn_adapter.py'
        shutil.copyfile(source,root/'adapter.py')
        shutil.copyfile(worker.__file__,root/'remote_tabpfn.py')
        bundle=dict(rows=rows,markets=['M'],market_paths={'M':'market.npz'},
                    inputs={p:worker.sha(root/p) for p in ('checkpoint.bin','market.npz')})
        worker.atomic(root/'bundle.json',bundle)
        worker.atomic(root/'refs.json',[dict(row,h=2.) for row in rows if row['phase']=='calibration'])
        runtime=dict(packages=worker.PACKAGES,python='3.12.9',cuda='12.9',gpu='Synthetic GPU',
                     precision='float32',tf32=False,random_state=worker.SEED,n_estimators=8,thread_limit=1)
        manifest=dict(kind=worker.KIND,frozen=True,bundle='bundle.json',
                      bundle_sha256=worker.sha(root/'bundle.json'),worker='remote_tabpfn.py',
                      adapter='adapter.py',checkpoint='checkpoint.bin',packages=worker.PACKAGES,
                      approved_runtime=runtime,bridge_references='refs.json',
                      bridge_tolerances=worker.BRIDGE_TOLERANCES,
                      files={p:worker.sha(root/p) for p in ('bundle.json','remote_tabpfn.py',
                                                          'adapter.py','checkpoint.bin','market.npz','refs.json')})
        worker.atomic(root/'manifest.json',manifest)
        return bundle,manifest,r

    def test_portable_features_are_bitwise_identical_to_frozen_canonical_features(self):
        from research.seed_replication.causal import feature_batch
        values=np.random.default_rng(4).normal(size=1024)
        values[::11]=0
        for past in (values,np.zeros(1024),np.arange(1024,dtype=float)/100):
            x,xn=worker.covariates(past)
            expected=feature_batch(past[None])['feats'][0]
            np.testing.assert_array_equal(x,expected[:-1])
            np.testing.assert_array_equal(xn,expected[-1:])
        changed=values.copy();changed[501:]=999
        np.testing.assert_array_equal(worker.covariates(values)[0][:501],
                                      worker.covariates(changed)[0][:501])

    def test_pilot_never_selects_evaluation_and_shards_partition_every_origin(self):
        with tempfile.TemporaryDirectory() as d:
            bundle,_,_=self.fixture(Path(d))
            for count in (16,32):
                chosen=worker.select_rows(bundle,pilot=count)
                self.assertEqual(len(chosen),count)
                self.assertTrue(all(r['phase']=='calibration' for r in chosen))
            shards=[worker.select_rows(bundle,shard_index=i,shard_count=4) for i in range(4)]
            self.assertEqual(sorted(r['index'] for s in shards for r in s),
                             sorted(r['index'] for r in bundle['rows']))
            with self.assertRaises(ValueError):worker.select_rows(bundle,pilot=20)
            with self.assertRaises(ValueError):worker.select_rows(bundle,pilot=16,shard_count=2)

    def test_source_bundle_market_and_frozen_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle,manifest,_=self.fixture(root)
            digest=worker.sha(root/'manifest.json')
            worker.verify(root,root/'manifest.json',digest,evaluation=True)
            with self.assertRaisesRegex(ValueError,'manifest SHA'):
                worker.verify(root,root/'manifest.json','bad')
            manifest['frozen']=False;worker.atomic(root/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'frozen'):
                worker.verify(root,root/'manifest.json',worker.sha(root/'manifest.json'),evaluation=True)
            worker.verify(root,root/'manifest.json',worker.sha(root/'manifest.json'),evaluation=False)
            (root/'adapter.py').write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                worker.verify(root,root/'manifest.json',worker.sha(root/'manifest.json'))

    def test_cache_context_future_invariance_and_statistics_record_floor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle,_,_=self.fixture(root)
            cache=worker.MarketCache(root,bundle)
            row=bundle['rows'][0]
            original=cache.context(row)
            cache.cache['M']['r'][row['index']+1:]=1e6
            np.testing.assert_array_equal(original,cache.context(row))
            fake=FakeAdapter(point=-2)
            x,xn=worker.covariates(original)
            result=worker.record(row,fake.predict(original,'squared',x,xn),original,.5)
            self.assertEqual(result['mean'],0.)
            self.assertEqual(result['variance'],result['second_moment'])
            self.assertEqual(result['variance'],max(1e-12,1e-8*np.mean(original**2)))
            self.assertTrue(result['floor_applied'])
            self.assertFalse(result['substituted'])
            self.assertIsNone(result['seed'])
            self.assertEqual(result['raw_point'],-2)
            self.assertNotIn('loss',result)

    def test_chunk_resume_preserves_bytes_and_completed_attempt_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle,manifest,_=self.fixture(root)
            out=root/'out';rows=bundle['rows'][:3];adapter=FakeAdapter()
            selection=dict(pilot=16,shard_index=0,shard_count=1)
            result=worker.run(root,out,manifest,bundle,'frozen',rows,selection,adapter,{},chunk_size=2)
            self.assertEqual(result['completed'],3)
            self.assertEqual(len(adapter.calls),3)
            saved={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('forecast*')}
            with self.assertRaises(FileExistsError):
                worker.run(root,out,manifest,bundle,'frozen',rows,selection,adapter,{},chunk_size=2)
            # Simulate a process dying after chunks commit but before terminal publication.
            (out/'terminal.json').unlink()
            resumed=FakeAdapter()
            worker.run(root,out,manifest,bundle,'frozen',rows,selection,resumed,{},chunk_size=2)
            self.assertEqual(len(resumed.calls),0)
            self.assertEqual(saved,{str(p.relative_to(out)):p.read_bytes() for p in out.rglob('forecast*')})
            block=worker.read(worker.chunk_path(out,0))
            self.assertEqual(block['records'][0]['model'],'tabpfn_ts')
            self.assertEqual(block['records'][0]['phase'],'calibration')

    def test_bridge_failure_is_terminal_and_frozen_tolerance_is_not_relaxed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle,manifest,_=self.fixture(root)
            row=bundle['rows'][0]
            refs=worker.read(root/'refs.json');refs[0]['h']=9.
            worker.atomic(root/'refs.json',refs)
            manifest['files']['refs.json']=worker.sha(root/'refs.json')
            with self.assertRaisesRegex(RuntimeError,'bridge failed'):
                worker.run(root,root/'failed',manifest,bundle,'frozen',[row],
                           {'pilot':16},FakeAdapter(),{})
            self.assertEqual(worker.read(root/'failed'/'terminal.json')['status'],'failed')
            self.assertFalse(worker.read(root/'failed'/'failed_bridge.json')['passed'])
            self.assertFalse(list((root/'failed').rglob('forecast.json')))

    def test_import_requires_all_shards_and_freeze_identity_before_any_cell(self):
        from research.frontier_extension_20260926 import run as local
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle,manifest,_=self.fixture(root)
            out=root/'local';out.mkdir()
            shutil.copyfile(root/'bundle.json',out/'bundle.json')
            worker.atomic(out/'FREEZE.json',{'synthetic':True})
            manifest['extension_freeze_sha256']=worker.sha(out/'FREEZE.json')
            worker.atomic(root/'manifest.json',manifest)
            digest=worker.sha(root/'manifest.json')
            dirs=[]
            for shard in range(2):
                directory=root/('shard'+str(shard));dirs.append(directory)
                rows=worker.select_rows(bundle,shard_index=shard,shard_count=2)
                worker.run(root,directory,manifest,bundle,digest,rows,
                           dict(pilot=None,shard_index=shard,shard_count=2),FakeAdapter(),
                           manifest['approved_runtime'],chunk_size=16)
            with patch.object(local,'verify',return_value=(bundle,{})) as gate:
                with self.assertRaisesRegex((ValueError,RuntimeError),'retain every'):
                    worker.import_attempts(out,root,root/'manifest.json',digest,dirs[:1])
                self.assertFalse((out/'cells').exists())
                identity_path=dirs[0]/'attempt.json';saved=identity_path.read_bytes()
                identity=worker.read(identity_path);identity['runtime']['gpu']='Unapproved GPU'
                worker.atomic(identity_path,identity)
                with self.assertRaisesRegex(ValueError,'approved pilot runtime'):
                    worker.import_attempts(out,root,root/'manifest.json',digest,dirs)
                self.assertFalse((out/'cells').exists());identity_path.write_bytes(saved)
                chunk_path=worker.chunk_path(dirs[0],0);saved=chunk_path.read_bytes()
                chunk=worker.read(chunk_path);chunk['bridge']=[]
                worker.atomic(chunk_path,chunk)
                worker.atomic(chunk_path.with_suffix('.sha256.json'),{'sha256':worker.sha(chunk_path)})
                with self.assertRaisesRegex(ValueError,'bridge diagnostics'):
                    worker.import_attempts(out,root,root/'manifest.json',digest,dirs)
                self.assertFalse((out/'cells').exists());chunk_path.write_bytes(saved)
                worker.atomic(chunk_path.with_suffix('.sha256.json'),{'sha256':worker.sha(chunk_path)})
                result=worker.import_attempts(out,root,root/'manifest.json',digest,dirs)
                self.assertEqual(result['imported_records'],len(bundle['rows']))
                self.assertGreaterEqual(gate.call_count,2)
                worker.import_attempts(out,root,root/'manifest.json',digest,dirs)
            final=worker.read(out/'cells/tabpfn_ts/none/M/forecast.json')
            self.assertEqual(len(final['rows']),len(bundle['rows']))
            self.assertEqual(len(final['audit']['calibration_bridge']),72)

    def test_missing_empty_or_changed_reference_set_and_unapproved_runtime_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bundle,manifest,_=self.fixture(root)
            valid=worker.read(root/'refs.json')
            absent=dict(manifest);absent.pop('bridge_references')
            with self.assertRaisesRegex(ValueError,'72 calibration'):
                worker.calibration_references(root,absent,bundle)
            for refs in ([],valid[:-1],valid[:-1]+[valid[0]]):
                worker.atomic(root/'refs.json',refs)
                manifest['files']['refs.json']=worker.sha(root/'refs.json')
                with self.assertRaisesRegex(ValueError,'72'):
                    worker.calibration_references(root,manifest,bundle)
            worker.atomic(root/'refs.json',valid)
            manifest['files']['refs.json']=worker.sha(root/'refs.json')
            manifest['bridge_tolerances']={'rtol':.01,'atol':.0002}
            with self.assertRaisesRegex(ValueError,'tolerances'):
                worker.calibration_references(root,manifest,bundle)
            manifest['bridge_tolerances']=worker.BRIDGE_TOLERANCES
            unapproved=dict(manifest);unapproved.pop('approved_runtime')
            with self.assertRaisesRegex(ValueError,'approved pilot runtime'):
                worker.approved_runtime(unapproved)
            for key in ('python','cuda','gpu','precision'):
                changed=dict(manifest['approved_runtime']);changed[key]='different'
                with self.assertRaisesRegex(ValueError,'approved pilot runtime'):
                    worker.approved_runtime(manifest,changed)
            worker.atomic(root/'manifest.json',unapproved)
            digest=worker.sha(root/'manifest.json')
            worker.verify(root,root/'manifest.json',digest,evaluation=False)
            with self.assertRaisesRegex(ValueError,'approved pilot runtime'):
                worker.verify(root,root/'manifest.json',digest,evaluation=True)


if __name__=='__main__':unittest.main()
