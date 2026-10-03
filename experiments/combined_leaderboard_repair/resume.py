"""Dated numerical-gate amendment; forecast and statistical code remain frozen.

The original run is read-only. This creates v2 with a uniform, independently
checked cross-runtime tolerance and reports every original-gate exceedance.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import uuid
from collections import defaultdict
import numpy as np
from experiments.combined_leaderboard.common import *
from experiments.combined_leaderboard import pipeline as original
from experiments.combined_leaderboard.neural import Predictor

HERE=Path(__file__).resolve().parent
AUDIT=ROOT/'experiments/outputs/combined_leaderboard_numerical_repair_v1'
OUT=ROOT/'experiments/outputs/combined_leaderboard_v2'
RTOL=1e-5
ORIGINAL_RTOL=1e-6
CHUNK=64

def compare(moment,archived):
    moment,archived=np.asarray(moment,float),np.asarray(archived,float)
    require(moment.ndim==1 and moment.shape==archived.shape and moment.size>0,'Invalid bridge shapes')
    require(np.isfinite(moment).all() and np.isfinite(archived).all() and
            np.all(moment>0) and np.all(archived>0),'Invalid bridge values')
    error=np.abs(moment-archived)/archived
    return dict(passed=bool(np.all(error<=RTOL)),checked=len(error),rtol=RTOL,atol=0.,
                maximum_relative_error=float(error.max()),relative_errors=error.tolist(),
                original_rtol=ORIGINAL_RTOL,original_passed=bool(np.all(error<=ORIGINAL_RTOL)),
                original_exceedance_count=int(np.sum(error>ORIGINAL_RTOL)))

def archived_rows(seed):
    return {key(r):r for phase in ('calibration','evaluation')
            for r in read(CACHE/f'{seed}_{phase}.json')}

def validated_old_cells():
    expected=read(AUDIT/'VALIDATION.json')['old_cells']
    actual=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
            for p in sorted((OUTPUT/'cells').rglob('forecast.json'))]
    require(actual==expected,'Original cell inventory differs from independent validation')
    return {r['path']:r['sha256'] for r in expected}

def verify():
    original.verify(OUTPUT)
    bundle,sealed=original.verify(OUT)
    require(sealed['base_freeze_sha256']==sha(OUTPUT/'FREEZE.json'),'Original freeze identity changed')
    require(sealed['settings']['bridge_rtol']==RTOL,'Wrong amended numerical envelope')
    require(sealed['settings']['original_bridge_rtol']==ORIGINAL_RTOL,'Missing original gate')
    return bundle,sealed

def freeze():
    bundle,base=original.verify(OUTPUT)
    validation=read(AUDIT/'VALIDATION.json')
    require(validation['passed'],'Independent numerical validation failed')
    require(validation['checked']==524,'Incomplete predeclared numerical validation')
    require(validation['manual_checked']==10,'Incomplete independent reference cross-check')
    require(validation['freeze_sha256']==sha(OUTPUT/'FREEZE.json'),'Validation used a different original freeze')
    require(validation['validation_code_sha256']==sha(HERE/'validation.py'),'Validation executable changed')
    require(validation['reference_helper_sha256']==sha(HERE/'diagnose.py'),'Reference executable changed')
    require(validation['diagnostic_plan_freeze_sha256']==sha(AUDIT/'DIAGNOSTIC_PLAN_FREEZE.json'),'Validation plan identity changed')
    tests=read(AUDIT/'TESTS.json');require(tests['passed'],'Repair tests failed')
    for name,digest in tests['source_hashes'].items():
        require(sha(ROOT/name)==digest,'Tests predate repair executable change: '+name)
    planned=read(AUDIT/'DIAGNOSTIC_PLAN_FREEZE.json')
    require(sha(ROOT/planned['source'])==planned['sha256'],'Diagnostic plan changed')
    require(planned['uniform_rtol']==RTOL and planned['original_rtol']==ORIGINAL_RTOL,'Validation tolerance changed')
    old_cells=validated_old_cells()
    require(not OUT.exists(),'Amended output already exists; never overwrite it')
    OUT.mkdir()
    for name in ('bundle.json','model_specs.json'):
        (OUT/name).write_bytes((OUTPUT/name).read_bytes())
    sources=dict(base['sources'])
    sources.update(old_cells)
    for p in [*HERE.glob('*.py'),*HERE.glob('*.md'),*AUDIT.glob('*.json')]:
        sources[str(p.relative_to(ROOT))]=sha(p)
    amendment=dict(created_utc=now(),reason='Cross-device/library float32 reproducibility exceeds original 1ppm gate',
        base_freeze_sha256=sha(OUTPUT/'FREEZE.json'),original_failure_files={str(p.relative_to(ROOT)):sha(p)
            for p in (OUTPUT/'failures').glob('*.json')},
        numerical_change=dict(original_rtol=ORIGINAL_RTOL,amended_rtol=RTOL,atol=0.),
        preserved_cell_sha256=old_cells,
        invariants=['same frozen input histories and checkpoints','same float32 forecast code and runtime',
                    'same means/variances/second moments','same population and calibration',
                    'same scores, bootstrap and Holm families'],
        evidence=dict(validation_sha256=sha(AUDIT/'VALIDATION.json'),diagnostic_plan_sha256=planned['sha256']),
        limitations='Prompted by a numerical failure; 524-cell reference check is not a global error bound. Full all-row gate remains. No new leaderboard performance inspected.',
        resume_policy='Copy validated completed rows unchanged; no per-row backend switch, no old-GPU value substitution; preserve and report original 1ppm violations')
    original.atomic(OUT/'AMENDMENT.json',amendment)
    sources.update({str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ('bundle.json','model_specs.json','AMENDMENT.json')})
    sources.update(amendment['original_failure_files'])
    settings=dict(base['settings'],bridge_rtol=RTOL,original_bridge_rtol=ORIGINAL_RTOL)
    sealed=dict(base,kind='combined_leaderboard_v2_numerical_amendment',created_utc=now(),
                base_freeze_sha256=sha(OUTPUT/'FREEZE.json'),sources=sources,settings=settings,
                analysis_gate='All 44230 native second moments within uniform 1e-5; every original 1e-6 exceedance reported; all original scientific settings retained')
    original.atomic(OUT/'FREEZE.json',sealed)
    (OUT/'FREEZE.sha256').write_text(sha(OUT/'FREEZE.json')+'\n')
    return dict(status='frozen',identity=sha(OUT/'FREEZE.json'))

def migrate():
    bundle,sealed=verify();identity=sha(OUT/'FREEZE.json');old_id=sealed['base_freeze_sha256']
    require(validated_old_cells()==read(OUT/'AMENDMENT.json')['preserved_cell_sha256'],'Original inventory changed since amendment')
    records=[];lookups={s:archived_rows(s) for s in SEEDS}
    for path in sorted((OUTPUT/'cells').rglob('forecast.json')):
        relative=path.parent.relative_to(OUTPUT/'cells').with_suffix('.json')
        value=original.cell(OUTPUT,relative,old_id)
        saved=original.cell(OUT,relative,identity)
        if saved is not None:
            require(saved['rows']==value['rows'],'Migrated forecasts changed')
        else:
            payload={k:copy.deepcopy(v) for k,v in value.items() if k not in ('freeze_sha256','created_utc')}
            if relative.parts[0]=='neural':
                seed=payload['rows'][0]['seed']
                bridge=compare([r['second_moment'] for r in payload['rows']],
                               [lookups[seed][key(r)]['h'] for r in payload['rows']])
                require(bridge['original_passed'],'Previously accepted cell no longer passes original gate')
                payload['audit']['original_bridge']=payload['audit']['bridge']
                payload['audit']['bridge']=bridge
            payload['migration']=dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),rows_unchanged=True)
            saved=original.cell(OUT,relative,identity,payload)
            require(saved['rows']==value['rows'],'Migration changed numerical rows')
        records.append(dict(source=str(path.relative_to(ROOT)),sha256=sha(path),rows=len(value['rows']),
                            model=value['rows'][0]['model']))
    receipt=dict(created_utc=now(),base_freeze_sha256=old_id,cells=records,
                 neural_rows=sum(r['rows'] for r in records if r['model']=='volpfn'),
                 all_rows_unchanged=True)
    if (OUT/'MIGRATION.json').exists():
        require(read(OUT/'MIGRATION.json')['cells']==records,'Migration inventory changed')
    else: original.atomic(OUT/'MIGRATION.json',receipt)
    return dict(status='migrated',cells=len(records),neural_rows=receipt['neural_rows'])

def runtime(expected):
    actual=dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in expected['packages']},
                device='cpu',threads=2,batch_size=1)
    require(all(actual[k]==expected[k] for k in actual),'Use the pinned frontier_v1 neural interpreter')
    return actual

def run(limit=0):
    bundle,sealed=verify();identity=sha(OUT/'FREEZE.json');meta=runtime(sealed['neural_runtime'])
    require(limit==0 or limit>=CHUNK,'Limit must be zero or at least one complete 64-row chunk')
    with (OUT/'.neural.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with np.load(PLAN.parent/'contexts.npz') as z:
            contexts=np.concatenate([z['calibration'],z['evaluation']])
        tick=time.perf_counter();generated=0
        state=dict(status='running',started_utc=now(),pid=os.getpid(),new_forecasts=0)
        original.atomic(OUT/'PIPELINE_STATUS.json',state,replace=True)
        try:
            for seed in SEEDS:
                lookup=archived_rows(seed);p=None
                for start in range(0,len(contexts),CHUNK):
                    relative=Path('neural')/str(seed)/f'{start:06d}.json'
                    previous=original.cell(OUT,relative,identity)
                    if previous is not None:
                        require(previous['audit']['runtime']==meta,'Mixed neural runtimes')
                        continue
                    if p is None:p=Predictor(seed,'cpu',2)
                    end=min(start+CHUNK,len(contexts))
                    values=np.concatenate([p.predict(contexts[i:i+1]) for i in range(start,end)])
                    selected=bundle['rows'][start:end]
                    old=np.array([lookup[key(r)]['h'] for r in selected])
                    bridge=compare(values[:,2],old)
                    evidence=dict(seed=seed,start=start,end=end,bridge=bridge,runtime=meta,
                                  device='cpu',batch_size=1,threads=2)
                    if not bridge['passed']:
                        original.atomic(OUT/'failures'/f'{seed}_{start}_{uuid.uuid4().hex}.json',dict(**evidence,values=values,archived=old))
                        raise RuntimeError('Amended uniform numerical envelope exceeded; stop, no tolerance escalation')
                    rows=[original.record(r,'volpfn',seed,v) for r,v in zip(selected,values)]
                    original.cell(OUT,relative,identity,dict(rows=rows,audit=evidence))
                    generated+=len(rows)
                    state.update(seed=seed,completed_in_seed=end,total_in_seed=len(contexts),new_forecasts=generated,
                                 elapsed_seconds=time.perf_counter()-tick,original_gate_exceedances_in_latest_chunk=bridge['original_exceedance_count'])
                    original.atomic(OUT/'PIPELINE_STATUS.json',state,replace=True)
                    print('NEURAL',seed,end,'new',generated,'seconds',round(time.perf_counter()-tick,1),flush=True)
                    if limit and generated>=limit:
                        state.update(status='repair_check_complete_ready_to_resume',ended_utc=now())
                        original.atomic(OUT/'PIPELINE_STATUS.json',state,replace=True)
                        return state
            state.update(status='forecasts_complete_analysis_pending',ended_utc=now())
            original.atomic(OUT/'PIPELINE_STATUS.json',state,replace=True)
            return state
        except BaseException as exc:
            state.update(status='failed',ended_utc=now(),error=f'{type(exc).__name__}: {exc}')
            original.atomic(OUT/'PIPELINE_STATUS.json',state,replace=True)
            raise

def analyse():
    bundle,sealed=verify();identity=sha(OUT/'FREEZE.json')
    require(not (OUT/'analysis').exists(),'Analysis already exists; no overwrite')
    for name in ('numpy','scipy','pandas'):
        require(importlib.metadata.version(name)==sealed['classical_runtime'][name],'Use original classical interpreter for analysis')
    expected={key(r):r for r in bundle['rows']};observed=defaultdict(set)
    lookup={s:archived_rows(s) for s in SEEDS};records=[];violations=[];fit_diagnostics=[];errors=[]
    for path in sorted((OUT/'cells').rglob('forecast.json')):
        relative=path.parent.relative_to(OUT/'cells').with_suffix('.json')
        value=original.cell(OUT,relative,identity)
        for r in value['rows']:
            k=key(r)
            require(k in expected and r['r2']==expected[k]['r2'] and r['rk']==expected[k]['rk'],'Frozen origin/target mismatch')
            require(k not in observed[r['model'],r['seed']],'Duplicate row')
            observed[r['model'],r['seed']].add(k)
        records.extend(value['rows'])
        if relative.parts[0]=='neural':
            rr=value['rows'];seed=rr[0]['seed']
            bridge=compare([r['second_moment'] for r in rr],[lookup[seed][key(r)]['h'] for r in rr])
            require(bridge==value['audit']['bridge'] and bridge['passed'],'Independent all-row bridge recheck failed')
            errors.extend(bridge['relative_errors'])
            for r,error in zip(rr,bridge['relative_errors']):
                if error>ORIGINAL_RTOL:
                    violations.append(dict(seed=seed,market=r['market'],date=r['date'],phase=r['phase'],relative_error=error))
        elif relative.parts[0]=='classical':
            a=value['audit']
            fit_diagnostics.append(dict(model=a['model'],market=a['market'],fit_attempts=a['fit_attempt_count'],
                fit_failures=a['fit_failure_count'],numerical_failures=a['numerical_failure_count'],substitutions=a['substitution_count'],floors=a['floor_count']))
    specs=read(OUT/'model_specs.json');required={(m,s) for m,v in specs.items() for s in v['seeds']}
    require(set(observed)==required and all(k==set(expected) for k in observed.values()),'Full model/seed/population required')
    require(len(errors)==44230,'Incomplete neural bridge population')
    from experiments.combined_leaderboard.statistics import analyse as statistics
    from experiments.combined_leaderboard.report import write_report
    settings=sealed['settings'];archive={}
    result=statistics(records,specs,settings['resamples'],settings['blocks'],settings['seed'],archive=archive)
    result['fit_diagnostics']=fit_diagnostics
    result['numerical_amendment']=dict(amendment_sha256=sha(OUT/'AMENDMENT.json'),original_rtol=ORIGINAL_RTOL,
        rtol=RTOL,checked=len(errors),maximum_relative_error=max(errors),original_exceedances=violations,
        forecast_arithmetic_unchanged=True)
    stage=OUT/'work'/('analysis_'+uuid.uuid4().hex);stage.mkdir(parents=True)
    original.atomic(stage/'results.json',result)
    np.savez_compressed(stage/'bootstrap_archive.npz',**archive)
    with (stage/'records.jsonl').open('x') as stream:
        for r in records:stream.write(json.dumps(r,allow_nan=False)+'\n')
    write_report(result,stage)
    note=(f'Numerical amendment: uniform relative tolerance {RTOL:g}; the original {ORIGINAL_RTOL:g} gate '
          f'was exceeded on {len(violations)} of {len(errors)} neural forecasts. Maximum discrepancy '
          f'{max(errors):.9g}. Forecast arithmetic and scientific settings are unchanged. '
          'The original failed freeze and independent higher-precision checks are preserved.\n')
    (stage/'NUMERICAL_AMENDMENT.md').write_text(note)
    for name in ('REPORT.md','LEADERBOARD.md'):
        with (stage/name).open('a') as stream:stream.write('\n'+note)
    original.atomic(stage/'verification.json',dict(freeze_sha256=identity,records=len(records),bridge_checked=len(errors),
        numerical_amendment=result['numerical_amendment'],outputs={p.name:sha(p) for p in sorted(stage.iterdir()) if p.is_file()}))
    os.rename(stage,OUT/'analysis')
    original.atomic(OUT/'PIPELINE_STATUS.json',dict(status='complete',ended_utc=now(),report=str(OUT/'analysis/LEADERBOARD.md')),replace=True)
    return dict(status='complete',report=str(OUT/'analysis/LEADERBOARD.md'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','migrate','run','analyse','status'))
    p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    if args.command=='run':value=run(args.limit)
    elif args.command=='status':value=read(OUT/'PIPELINE_STATUS.json')
    else:value=globals()[args.command]()
    print(json.dumps(value,indent=2,default=default))

if __name__=='__main__':main()
