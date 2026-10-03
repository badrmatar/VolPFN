"""Document the analytic GJR moment correction; no training or market scoring."""
import ast
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from volpfn import prior as VP
from experiments.matched_prior import run as M,simulators as S
from experiments.programme_closure import gpu_training as G,test_gjr_moment_correction as tests
from experiments.seed_replication import causal,training as T

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/outputs/programme_closure_20260919_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_hashes(path):
    text=Path(path).read_text()
    return {n.name:hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()
            for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def main():
    before=tests.archived_prior()
    a,b=function_hashes(tests.ARCHIVE),function_hashes(ROOT/'src/volpfn/prior.py')
    dependencies=['draw_law','law_sample','skew_moments','t_absmean','matern_cov','apply_overlays','_aggregate_stale_intervals']
    fingerprints={name:dict(before=a[name],after=b[name],unchanged=a[name]==b[name]) for name in dependencies}
    assert all(x['unchanged'] for x in fingerprints.values())
    old_manifest=json.loads((OUT/'gpu_design_v1/manifest.json').read_text())
    caller_paths=['experiments/matched_prior/simulators.py','experiments/seed_replication/causal.py','src/volpfn/model.py']
    callers={p:dict(before=old_manifest['source_hashes'][p],after=sha(ROOT/p)) for p in caller_paths}
    assert all(x['before']==x['after'] for x in callers.values())
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'experiments.programme_closure.test_gjr_moment_correction',
        'experiments.programme_closure.test_simulator_correction',
        'experiments.matched_prior.test_simulators','experiments.seed_replication.test_causal'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'GJR_CORRECTION_TESTS.log').write_text(stream.getvalue())
    assert result.wasSuccessful(),stream.getvalue()
    quadrature=[]
    for nu in (2.5,2.541481637298658,3.,8.,50.,VP.NU_GAUSS):
        for gamma in (.7,.85,1.,1.15,1.4):
            exact=float(VP.skew_negative_second_moment(nu,gamma))
            reference=tests.numerical_negative_moment(nu,gamma)
            quadrature.append(dict(nu=nu,gamma=gamma,analytic=exact,density_integral=reference,
                                   absolute_error=abs(exact-reference)))
    assert max(x['absolute_error'] for x in quadrature)<2e-8
    spec=G.specification('cpu')
    stress=[]
    with threadpool_limits(limits=2):
        for seed in range(8101,8106):
            begun=time.perf_counter()
            for step,args in enumerate(M.schedule(seed,spec),1):
                if step>128:break
                prepared=S.prepared_batch('broad',*args);T.validate_batch(prepared)
            stress.append(dict(arm='broad',seed=seed,batches=128,status='passed',seconds=time.perf_counter()-begun))
        # Both previously failing neighborhoods are checked under the final repair.
        checked=0
        for step,args in enumerate(M.schedule(8103,spec),1):
            if step>8836:break
            if not(4555<=step<=4595 or 8796<=step<=8836):continue
            prepared=S.prepared_batch('broad',*args);T.validate_batch(prepared);checked+=1
        assert checked==82
        # Full-size samples supplement the static unchanged-function/path audit.
        paired=0
        for seed in (8101,8102):
            for step,args in enumerate(M.schedule(seed,spec),1):
                if step>64:break
                with patch.object(S,'VP',before):old=S.prepared_batch('matern',*args)
                new=S.prepared_batch('matern',*args)
                for key in old:np.testing.assert_array_equal(old[key],new[key])
                paired+=1
    raw=causal.raw_batch(810333125,8,1024)
    gjr=[dict(row=i,**meta) for i,meta in enumerate(raw['meta']) if meta['family']=='gjr']
    record=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
        original_source_sha256=sha(tests.ARCHIVE),corrected_source_sha256=sha(ROOT/'src/volpfn/prior.py'),
        changed_existing_functions=[k for k in a if a[k]!=b.get(k)],added_functions=sorted(set(b)-set(a)),
        matern_dependency_fingerprints=fingerprints,unchanged_caller_sources=callers,
        unchanged_prior_constants={k:getattr(before,k)==getattr(VP,k) for k in ('PRIOR_WEIGHTS','OVERLAY_P','NU_GAUSS','FAMILIES')},
        quadrature=quadrature,max_quadrature_absolute_error=max(x['absolute_error'] for x in quadrature),
        tests=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors)),
        broad_stress=stress,failure_neighborhood_batches=checked,full_size_matern_bitwise_batches=paired,
        repaired_8816_batch_gjr_metadata=gjr,
        environment=dict(python=platform.python_version(),numpy=np.__version__),
        compatibility='Matérn active simulator/normalization dependency functions and caller files are unchanged;128 full-size prepared batches match bitwise. This is dependency compatibility evidence, not itself checkpoint provenance or permission to reuse.',
        broad_compatibility='All GJR moments now use the exact law; broad data/optimization are not backward-identical. Historical checkpoints remain distinct.',
        random_stream='Historical20000 innovation draws still consumed; GARCH/EGARCH RNG states and outputs checked unchanged; GJR coefficients corrected for every draw.',
        no_clipping=True,no_resampling=True,no_model_training=True,no_market_scoring=True,
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(tests.__file__),ROOT/'src/volpfn/prior.py']})
    (OUT/'GJR_CORRECTION.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=record['status'],source_sha256=record['corrected_source_sha256'],tests=result.testsRun,
                         broad_batches=640+checked,matern_bitwise_batches=paired,max_quadrature_error=record['max_quadrature_absolute_error'])),flush=True)


if __name__=='__main__':main()
