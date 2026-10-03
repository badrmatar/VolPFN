"""Immutable, resumable stages for the single-sample combined leaderboard.

No infrastructure is created by this command. Classical and cached stages are
CPU-only; the neural stage supports an explicitly selected existing device.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import numpy as np
from .common import *

HERE = Path(__file__).resolve().parent
CLASSICAL = ('hist250','ewma','garch_t','gjr_garch_t','egarch_t','aparch_t',
             'compact9','univol','har','loghar')
CHUNK = 64

def specs():
    adapters = {
        'volpfn':'Native Student-t mixture mean and second moment; five seeds scored separately',
        'hist250':'250-return sample mean and sample variance; M2 = variance + mean squared',
        'ewma':'Zero-mean EWMA .94; variance = second moment',
        'garch_t':'Constant mean, fitted Student-t GARCH(1,1)',
        'gjr_garch_t':'Constant mean, fitted Student-t GJR(1,1,1)',
        'egarch_t':'Constant mean, fitted Student-t EGARCH(1,1,1)',
        'aparch_t':'Constant mean, fitted Student-t APARCH(1,1,1)',
        'compact9':'Archived held-out-geography nine-parameter likelihood rule; zero mean',
        'univol':'Default density-weighted geometric point, mean=0 and variance=M2=h; not native mixture moments',
        'chronos2':'Cached squared-return median; proxy adapter mean=0 and variance=M2=h',
        'moirai2':'Cached squared-return median; proxy adapter mean=0 and variance=M2=h',
        'tabpfn_ts':'Cached squared-return mean; proxy adapter mean=0 and variance=M2=h',
        'har':'RK HAR OLS; zero-mean working variance/M2; declared floor',
        'loghar':'Log of RK arithmetic averages; exp(prediction + residual variance/2); zero mean',
    }
    return {m:dict(family='V' if m=='volpfn' else 'F' if m in FMS else 'R' if m in ('har','loghar') else 'E',
                   output_adapter=a, seeds=list(SEEDS) if m=='volpfn' else [None],
                   information_group='Lagged realised kernel (richer information)' if m in ('har','loghar') else 'Daily returns only')
            for m,a in adapters.items()}

def atomic(path, value, *, replace=False):
    """Write fully, then publish without replacing completed artifacts."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        save(temporary,value)
        if replace:
            os.replace(temporary,path)
        else:
            os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

def preflight(out):
    from .integrity import archived_reproduction
    from .optional_audit import r_probe, foundation_cache_probe
    bundle=load_bundle(out)
    for name in ('pilot_neural_all_calibration_v1.json','pilot_classical_v1.json',
                 'pilot_neural_frontier_calibration_v1.json','pilot_neural_random_eval_v1.json',
                 'pilot_neural_oldruntime_cpu_v1.json'):
        require((out/name).is_file(),'Missing measured pilot: '+name)
    selected=read(out/'pilot_neural_frontier_calibration_v1.json')
    pilots=selected['reports']
    evaluation=read(out/'pilot_neural_oldruntime_cpu_v1.json')
    require({r['seed'] for r in pilots}==set(SEEDS), 'Pilot needs every corrected seed')
    require(all(r['bridge']['passed'] for r in pilots),'Pilot bridge failed; investigate without scoring')
    require({r['seed'] for r in evaluation}==set(SEEDS) and all(r['bridge']['passed'] for r in evaluation),
            'Selected-runtime sampled evaluation bridge failed')
    original=read(out/'pilot_neural_random_eval_v1.json')
    require(all(a['indices']==b['indices'] for a,b in zip(original,evaluation)),
            'Runtime diagnosis must retain identical sampled indices')
    packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','arch','torch','matplotlib')}
    result=dict(status='passed',created_utc=now(),counts=bundle['counts'],
                packages=packages,python=sys.version,platform=platform.platform(),
                reproduction=archived_reproduction(),R=r_probe(),
                optional_foundation_inventory=foundation_cache_probe(),
                neural_pilot=selected,neural_evaluation_pilot=evaluation,
                rejected_runtime_pilot=original,
                runtime_diagnosis='Original Torch2.8/NumPy1.26 sampled bridge failed; identical fixed contexts pass in existing Torch2.14/NumPy2.5 runtime. All-row gate remains; exact numerical cause not isolated.',
                classical_pilot=read(out/'pilot_classical_v1.json'),
                execution='No new paid compute provisioned; full forecasts not implied by preflight')
    atomic(out/'PREFLIGHT.json',result)
    return result

def freeze(out):
    bundle=load_bundle(out)
    require(read(out/'PREFLIGHT.json')['status']=='passed','Preflight incomplete')
    require(not (out/'FREEZE.json').exists(),'Already frozen; do not overwrite')
    require((out/'TESTS.json').is_file() and read(out/'TESTS.json')['passed'], 'Verified tests required')
    for name,digest in read(out/'TESTS.json')['source_hashes'].items():
        require(sha(ROOT/name)==digest,'Tests predate executable change: '+name)
    paths=set(HERE.glob('*.py')) | set(HERE.glob('*.md'))
    paths.update(ROOT.glob('*.py'))
    for folder in ('experiments/seed_replication','experiments/tier_a_confirmation','experiments/frontier_benchmark',
                   'experiments/population_transfer','experiments/prior_adaptation','experiments/attribution'):
        paths.update((ROOT/folder).glob('*.py'))
    from experiments.evaluation_expansion.compact import dependency_paths
    paths.update(dependency_paths())
    paths.add(ROOT/'experiments/return_only_benchmark/volpfn_adapter.py')
    paths.add(ROOT/'experiments/return_only_benchmark/__init__.py')
    paths.add(ROOT/'experiments/evaluation_expansion/__init__.py')
    paths.add(ROOT/'experiments/return_only_benchmark/foundation_adapters.py')
    paths.add(ROOT/'experiments/tier_a_confirmation/PROTOCOL.md')
    paths.add(ROOT/'experiments/programme_closure/confirm_benchmark.py')
    for name in ('bundle.json','PREFLIGHT.json','TESTS.json','pilot_neural_all_calibration_v1.json','pilot_classical_v1.json'):
        paths.add(out/name)
    paths.update(out.glob('pilot_neural_*.json'))
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    atomic(out/'model_specs.json',specs())
    sources[str((out/'model_specs.json').relative_to(ROOT))]=sha(out/'model_specs.json')
    for name in ('PROTOCOL.md','OPTIONAL_MODELS.md'):
        content=(HERE/name).read_text()
        with (out/name).open('x') as stream: stream.write(content)
        sources[str((out/name).relative_to(ROOT))]=sha(out/name)
    head=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    status=subprocess.run(['git','status','--short'],capture_output=True,text=True).stdout.strip()
    manifest=dict(kind='combined_leaderboard_v1',created_utc=now(),inputs=bundle['inputs'],sources=sources,
                  git_head=head,git_status=status,git_disclosure='Executable SHA256 freeze; uncommitted/untracked project, not a claimed committed freeze',
                  settings=dict(resamples=20000,blocks=[10,5,20],seed=2026091902,
                                bridge_rtol=1e-6,chunk_size=CHUNK,primary_family=22,reference_family=4),
                  neural_runtime=read(out/'PREFLIGHT.json')['neural_pilot']['runtime'],
                  classical_runtime={k:v for k,v in read(out/'PREFLIGHT.json')['packages'].items()
                                     if k in ('numpy','scipy','pandas','arch')},
                  analysis_gate='Every required forecast plus all 44230 native-moment bridges before any new scoring')
    atomic(out/'FREEZE.json',manifest)
    with (out/'FREEZE.sha256').open('x') as f: f.write(sha(out/'FREEZE.json')+'\n')
    return manifest

def verify(out):
    require(sha(out/'FREEZE.json')==(out/'FREEZE.sha256').read_text().strip(),'Freeze manifest changed')
    sealed=read(out/'FREEZE.json')
    for group in ('sources','inputs'):
        for name,digest in sealed[group].items():
            require(sha(ROOT/name)==digest,'Frozen content changed: '+name)
    return read(out/'bundle.json'),sealed

def record(base, model, seed, values, substituted=False, floor=False):
    return {k:base[k] for k in ('market','date','phase','r2','rk')} | dict(
        model=model,seed=seed,mean=float(values[0]),variance=float(values[1]),second_moment=float(values[2]),
        substituted=bool(substituted),floor_applied=bool(floor))

def cell(out, relative, sealed_id, data=None):
    # Forecast plus digest are published as one directory rename. Interrupted
    # writes remain in work/, outside the committed cell inventory.
    path=(out/'cells'/relative).with_suffix('')
    if path.exists():
        require((path/'forecast.sha256').exists(),'Unsealed cell (recover explicitly): '+str(path))
        require(sha(path/'forecast.json')==(path/'forecast.sha256').read_text().strip(),'Cell hash changed: '+str(path))
        found=read(path/'forecast.json')
        require(found['freeze_sha256']==sealed_id,'Cell belongs to another freeze')
        return found
    if data is None: return None
    value=dict(freeze_sha256=sealed_id,created_utc=now(),**data)
    staging=out/'work'/('cell_'+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    save(staging/'forecast.json',value)
    (staging/'forecast.sha256').write_text(sha(staging/'forecast.json')+'\n')
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'Another process already committed this cell')
    os.rename(staging,path)
    return value

def _cached(out,bundle,identity):
    for model in FMS:
        relative=Path('cached')/(model+'.json')
        if cell(out,relative,identity) is not None: continue
        source={}
        for directory,phase in ((DEV,'calibration'),(FRONTIER,'evaluation')):
            source.update({key(r):r for r in read(directory/'models'/model/'predictions.json') if r['phase']==phase})
        rows=[record(r,model,None,(0.,source[key(r)]['h'],source[key(r)]['h']),
                     floor=source[key(r)].get('floor_activated',False)) for r in bundle['rows']]
        cell(out,relative,identity,dict(rows=rows,audit=dict(source='Pinned archived forecasts; zero-mean proxy adapter',no_inference=True)))
        print('CACHED',model,len(rows),flush=True)

def _classical(out,bundle,identity,args):
    from .classical import forecast
    for name,version in read(out/'FREEZE.json')['classical_runtime'].items():
        require(importlib.metadata.version(name)==version,'Classical runtime differs: '+name)
    models=args.models or CLASSICAL
    markets=args.markets or bundle['markets']
    require(set(models)<=set(CLASSICAL) and set(markets)<=set(bundle['markets']),'Undeclared model/market')
    for model in models:
        for name in markets:
            relative=Path('classical')/model/(name+'.json')
            if cell(out,relative,identity) is not None: continue
            origins=sorted([r for r in bundle['rows'] if r['market']==name],key=lambda r:r['index'])
            tick=time.perf_counter()
            predictions=forecast(market_data(bundle,name),model,np.array([r['index'] for r in origins]))
            rows=[record(r,model,None,[predictions[k][i] for k in ('mean','variance','second_moment')],
                         predictions['substituted'][i],predictions['floor_applied'][i]) for i,r in enumerate(origins)]
            cell(out,relative,identity,dict(rows=rows,audit=predictions['audit'],elapsed_seconds=time.perf_counter()-tick))
            print('CLASSICAL',model,name,len(rows),round(time.perf_counter()-tick,2),flush=True)

def _neural(out,bundle,identity,args):
    from .neural import Predictor,compare
    selected=args.seeds or SEEDS
    require(set(selected)<=set(SEEDS),'Undeclared seed')
    expected=read(out/'FREEZE.json')['neural_runtime']
    runtime=dict(python=platform.python_version(),
                 packages={k:importlib.metadata.version(k) for k in expected['packages']},
                 device=args.device,threads=args.threads,batch_size=args.batch_size)
    require(all(runtime[k]==expected[k] for k in runtime),
            'Neural runtime differs from passing preflight; use its pinned interpreter/settings')
    with np.load(PLAN.parent/'contexts.npz',allow_pickle=False) as z:
        contexts=np.concatenate([z['calibration'],z['evaluation']])
    require(len(contexts)==len(bundle['rows']),'Context row mismatch')
    for seed in selected:
        predictor=None; tick=time.perf_counter()
        archived=np.array([r['h'] for phase in ('calibration','evaluation') for r in read(CACHE/f'{seed}_{phase}.json')])
        for start in range(0,len(contexts),CHUNK):
            relative=Path('neural')/str(seed)/f'{start:06d}.json'
            previous=cell(out,relative,identity)
            if previous is not None:
                require(previous['audit']['runtime']==runtime,'Cannot silently mix neural runtimes')
                continue
            if predictor is None: predictor=Predictor(seed,args.device,args.threads)
            end=min(start+CHUNK,len(contexts))
            values=np.concatenate([predictor.predict(contexts[i:min(i+args.batch_size,end)])
                                   for i in range(start,end,args.batch_size)])
            bridge=compare(values[:,2],archived[start:end])
            evidence=dict(bridge=bridge,seed=seed,start=start,end=end,device=args.device,
                          batch_size=args.batch_size,threads=args.threads,runtime=runtime)
            if not bridge['passed']:
                # Keep the failed values in a distinct attempt. Never widen the
                # declared tolerance, delete a row, or silently use the cache.
                atomic(out/'failures'/f'bridge_{seed}_{start}_{uuid.uuid4().hex}.json',dict(**evidence,values=values,archived=archived[start:end]))
                raise RuntimeError('Native second-moment bridge failed; preserved failure, analysis blocked')
            rows=[record(r,'volpfn',seed,v) for r,v in zip(bundle['rows'][start:end],values)]
            cell(out,relative,identity,dict(rows=rows,audit=evidence))
            atomic(out/f'status_neural_{seed}.json',dict(status='running',seed=seed,completed=end,total=len(contexts),elapsed_seconds=time.perf_counter()-tick),replace=True)
            print('NEURAL',seed,end,len(contexts),round(time.perf_counter()-tick,2),flush=True)
        atomic(out/f'status_neural_{seed}.json',dict(status='complete',seed=seed,completed=len(contexts),total=len(contexts)),replace=True)

def run(out,args):
    bundle,_=verify(out); identity=sha(out/'FREEZE.json')
    require(args.batch_size>=1 and args.threads>=1,'Positive batch size and threads required')
    groups=('cached','classical','neural') if args.group=='all' else (args.group,)
    for group in groups:
        with (out/f'.{group}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status_path=out/f'status_{group}.json'
            atomic(status_path,dict(status='running',started_utc=now(),pid=os.getpid()),replace=True)
            try:
                if group=='cached': _cached(out,bundle,identity)
                elif group=='classical': _classical(out,bundle,identity,args)
                else: _neural(out,bundle,identity,args)
            except BaseException as exc:
                atomic(status_path,dict(status='failed',ended_utc=now(),exception=f'{type(exc).__name__}: {exc}'),replace=True)
                raise
            atomic(status_path,dict(status='requested_scope_complete',ended_utc=now()),replace=True)

def status(out):
    counts={}
    for path in sorted((out/'cells').rglob('forecast.json')):
        relative=path.parent.relative_to(out/'cells')
        label='/'.join(relative.parts[:2]) if relative.parts[0]!='cached' else 'cached'
        counts[label]=counts.get(label,0)+1
    return dict(created_utc=now(),frozen=(out/'FREEZE.sha256').exists(),cells=counts,
                stages={p.stem:read(p) for p in sorted(out.glob('status_*.json'))},
                report_complete=(out/'analysis/verification.json').exists(),
                note='Counts are progress only; analyse verifies full population and every checksum')

def analyse(out):
    bundle,sealed=verify(out); identity=sha(out/'FREEZE.json')
    for name in ('numpy','scipy','pandas'):
        require(importlib.metadata.version(name)==sealed['classical_runtime'][name],
                'Analysis runtime differs: '+name)
    require(not (out/'analysis').exists(),'Analysis directory exists; do not overwrite')
    records=[]; bridges=[]; fit_diagnostics=[]
    for path in sorted((out/'cells').rglob('forecast.json')):
        relative=path.parent.relative_to(out/'cells').with_suffix('.json')
        value=cell(out,relative,identity)
        records.extend(value['rows'])
        if path.relative_to(out/'cells').parts[0]=='neural': bridges.append(value['audit']['bridge'])
        if path.relative_to(out/'cells').parts[0]=='classical':
            a=value['audit']
            fit_diagnostics.append(dict(model=a['model'],market=a['market'],
                fit_attempts=a['fit_attempt_count'],fit_failures=a['fit_failure_count'],
                numerical_failures=a['numerical_failure_count'],substitutions=a['substitution_count'],floors=a['floor_count']))
    expected={key(r):r for r in bundle['rows']}; model_specs=read(out/'model_specs.json')
    from collections import defaultdict
    observed=defaultdict(set)
    for r in records:
        k=key(r); require(k in expected,'Unregistered origin')
        for field in ('rk','r2'): require(r[field]==expected[k][field],'Frozen target mismatch')
        require(k not in observed[r['model'],r['seed']],'Duplicate forecast')
        observed[r['model'],r['seed']].add(k)
    required={(m,s) for m,v in model_specs.items() for s in v['seeds']}
    require(set(observed)==required,'Missing declared models/seeds; no available-case report')
    require(all(v==set(expected) for v in observed.values()),'Incomplete required origins')
    require(all(b['passed'] for b in bridges) and sum(b['checked'] for b in bridges)==len(expected)*5,
            'Full all-seed bridge gate not passed')
    from .statistics import analyse as score
    from .report import write_report
    archive={}; settings=sealed['settings']
    result=score(records,model_specs,settings['resamples'],settings['blocks'],settings['seed'],archive=archive)
    result['fit_diagnostics']=fit_diagnostics
    destination=out/'work'/('analysis_'+uuid.uuid4().hex); destination.mkdir(parents=True)
    atomic(destination/'results.json',result)
    np.savez_compressed(destination/'bootstrap_archive.npz',**archive)
    with (destination/'records.jsonl').open('x') as stream:
        for r in records: stream.write(json.dumps(r,allow_nan=False)+'\n')
    write_report(result,destination)
    atomic(destination/'verification.json',dict(freeze_sha256=identity,records=len(records),
            bridge_checked=sum(b['checked'] for b in bridges),max_relative_error=max(b['maximum_relative_error'] for b in bridges),
            outputs={p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()}))
    require(not (out/'analysis').exists(),'Another process already published the analysis')
    os.rename(destination,out/'analysis')
    return dict(status='complete',report=str(out/'analysis/REPORT.md'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','preflight','freeze','run','status','analyse'))
    p.add_argument('--out',type=Path,default=OUTPUT)
    p.add_argument('--group',choices=('all','cached','classical','neural'),default='all')
    p.add_argument('--models',nargs='+');p.add_argument('--markets',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--device',choices=('cpu','cuda','mps'),default='cpu')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=1)
    args=p.parse_args();out=args.out.resolve()
    if args.command=='prepare':
        from .prepare import prepare
        result=prepare(out)['counts']
    elif args.command=='run': result=run(out,args)
    else: result=globals()[args.command](out)
    if result is not None: print(json.dumps(result,indent=2,default=default))

if __name__=='__main__': main()
