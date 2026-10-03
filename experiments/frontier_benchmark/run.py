"""Frozen, development-only integration pilot; never opens final scoring dates."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t
from experiments.seed_replication import training as T
from . import data,metrics

ROOT=data.ROOT
PACKAGE=Path(__file__).resolve().parent
BUILD=ROOT/'experiments/outputs/frontier_build_20260918'
DEFAULT_OUT=ROOT/'experiments/outputs/frontier_development_pilot_v1'
PROJECT_PYTHON=Path('/Users/badrmatar/Downloads/Study/Dissertation Masters/Code/Diss_Project/venv/bin/python')
ENVIRONMENTS={'chronos2':ROOT/'experiments/environments/frontier_v1/bin/python',
    'moirai2':ROOT/'experiments/environments/moirai_v1/bin/python',
    'tabpfn_ts':ROOT/'experiments/environments/frontier_v1/bin/python',
    'volpfn':PROJECT_PYTHON,'lstm':PROJECT_PYTHON}
MODELS=list(ENVIRONMENTS)


def versions():
    result={'python':platform.python_version(),'executable':sys.executable}
    for package in ('torch','numpy','scipy','pandas','chronos-forecasting','transformers',
                    'uni2ts','tabpfn','tabpfn-time-series','huggingface-hub'):
        try: result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    return result


def freeze(out=DEFAULT_OUT):
    out=Path(out).resolve()
    if out.exists(): raise FileExistsError('Pilot attempts are immutable; select a new attempt directory')
    out.mkdir(parents=True)
    arrays={};boundaries={};hashes={}
    for market in data.markets():
        a=data.load_development(market)
        arrays[market+'__date']=a['date'];arrays[market+'__r']=a['r']
        boundaries[market]={'first':str(a['date'][0]),'last':str(a['date'][-1]),'rows':len(a['r'])}
        source=data.DATA/(market+'.npz');hashes[str(source)]=T.sha(source)
    np.savez_compressed(out/'development_data.npz',**arrays)
    spx_date=arrays['SPX__date']
    origins=np.flatnonzero((spx_date>='2014-01-01')&(spx_date<='2014-12-31'))[:4]
    if len(origins)!=4 or min(origins)<100: raise ValueError('Required four development origins unavailable')
    identities={}
    for name,file in [('chronos2','amazon__chronos-2_weights.json'),('moirai2','Salesforce__moirai-2.0-R-small_weights.json')]:
        entry=json.loads((BUILD/file).read_text())
        for path,digest in entry['hashes'].items():
            if T.sha(path)!=digest: raise RuntimeError('Pinned model file changed: '+path)
        identities[name]=dict(status='available',path=entry['local_path'],provenance=entry)
        hashes[str(BUILD/file)]=T.sha(BUILD/file)
    identities['tabpfn_ts']=dict(status='pending_license_acceptance',
        model_id='Prior-Labs/tabpfn_3_5',revision='06bf2ba35c80a92a3b9abb436b99cf49e7a0365e',
        checkpoint='tabpfn-v3.5-20260909.safetensors',
        reason='Official TabPFN9 downloader requires PriorLabs license acceptance; no authorized local checkpoint supplied; no cloud or older-model substitution')
    checkpoint=ROOT/'experiments/outputs/seed_replication_session_recovery_20260918/base/8101/final.pt'
    identities['volpfn']=dict(status='available',path=str(checkpoint),sha256=T.sha(checkpoint),
        selection='completed fixed-final seed8101 for engineering only, not a headline-selected model')
    hashes[str(checkpoint)]=T.sha(checkpoint)
    identities['lstm']=dict(status='to_fit',steps=32,width=544,length=128,batch_size=2,seed=91801,
        held_out='SPX',fit_end='2010-12-31',purpose='bounded engineering fit, not a competitive training budget')
    sources={}
    files=list(PACKAGE.glob('*.py'))+[PACKAGE/'PROTOCOL.md',ROOT/'src/volpfn/model.py',ROOT/'src/volpfn/prior.py',
        ROOT/'experiments/seed_replication/causal.py',ROOT/'experiments/seed_replication/training.py']
    for source in files:
        relative=str(source.relative_to(ROOT));destination=out/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        sources[relative]=T.sha(source)
    runtime={}
    for model,python in ENVIRONMENTS.items():
        runtime[model]=json.loads(subprocess.check_output([str(python),'-m',
            'experiments.frontier_benchmark.run','versions'],cwd=ROOT,text=True))
    manifest=dict(created_utc=T.now(),kind='four_origin_development_engineering_only',
        models=identities,order=MODELS,sources=sources,hashes=hashes,environment=runtime,
        development_snapshot_sha256=T.sha(out/'development_data.npz'),data_boundaries=boundaries,
        origins=[dict(market='SPX',index=int(t),date=str(spx_date[t])) for t in origins],
        tracks=['raw_return_quantiles','squared_return_point_and_quantiles'],quantiles=metrics.QUANTILES.tolist(),
        context_length=1024,min_history=100,thread_limit=1,device='cpu',timeout_seconds_per_model=1800,
        selection='no winners or final performance inference from this pilot',
        scale_calibration='not fitted on four origins; development-panel calibration is separate before final scoring',
        final_period_scoring_authorized=False)
    T.save_json(out/'manifest.json',manifest)
    (out/'manifest.sha256').write_text(T.sha(out/'manifest.json')+'\n')
    T.save_json(out/'status.json',dict(status='frozen',created_utc=T.now()))
    verify(out)
    return manifest


def verify(out):
    out=Path(out)
    if T.sha(out/'manifest.json')!=(out/'manifest.sha256').read_text().strip():
        raise RuntimeError('Pilot manifest drift')
    m=json.loads((out/'manifest.json').read_text())
    if m['final_period_scoring_authorized'] is not False: raise RuntimeError('Wrong runner scope')
    if T.sha(out/'development_data.npz')!=m['development_snapshot_sha256']:
        raise RuntimeError('Development snapshot changed')
    for path,digest in m['hashes'].items():
        if T.sha(path)!=digest: raise RuntimeError('Frozen input drift: '+path)
    for path,digest in m['sources'].items():
        if T.sha(ROOT/path)!=digest or T.sha(out/'source'/path)!=digest:
            raise RuntimeError('Frozen source drift: '+path)
    for model in ('chronos2','moirai2'):
        for path,digest in m['models'][model]['provenance']['hashes'].items():
            if T.sha(path)!=digest: raise RuntimeError('Model weight/config drift')
    return m


def locked_data(out,m):
    with np.load(Path(out)/'development_data.npz',allow_pickle=False) as z:
        result={name:{'date':z[name+'__date'].copy(),'r':z[name+'__r'].copy()} for name in m['data_boundaries']}
    for a in result.values():
        if np.any(a['date']>data.END): raise RuntimeError('Postdevelopment observation in snapshot')
    return result


def native_prediction(model,past):
    from . import lstm
    return lstm.predict(model,past)


def worker(out,name):
    import torch
    from . import adapters,lstm
    from experiments.seed_replication import causal
    out=Path(out).resolve();m=verify(out)
    power_safe()
    if versions()!=m['environment'][name]: raise RuntimeError('Worker runtime changed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9182026)
    entry=m['models'][name];cell=out/'models'/name;cell.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();record=dict(model=name,status='running',runtime=versions(),started_utc=T.now())
    T.save_json(cell/'status.json',record)
    try:
        if entry['status']=='pending_license_acceptance':
            record.update(status='blocked_license',reason=entry['reason'])
            return record
        panel=locked_data(out,m)
        if name=='lstm':
            histories={k:{'date':v['date'][v['date']<=entry['fit_end']],
                         'r':v['r'][v['date']<=entry['fit_end']]} for k,v in panel.items() if k!=entry['held_out']}
            model,fit=lstm.fit(entry['held_out'],cell/'fit',steps=entry['steps'],width=entry['width'],
                length=entry['length'],batch_size=entry['batch_size'],seed=entry['seed'],histories=histories)
            record['fit']=fit
        elif name=='volpfn':
            model=causal.load_model(entry['path'],'cpu').eval()
            record['parameters']=sum(p.numel() for p in model.parameters())
        else:
            model=adapters.make(name,entry['path'])
        rows=[];audit=[]
        for i,origin in enumerate(m['origins']):
            power_safe()
            a=panel[origin['market']];t=origin['index']
            if str(a['date'][t])!=origin['date']: raise RuntimeError('Origin date mismatch')
            past=data.context(a['r'],t,length=m['context_length'],min_history=m['min_history'])
            altered=a['r'].copy();altered[t:]=12345.
            np.testing.assert_array_equal(past,data.context(altered,t,length=m['context_length']))
            np.testing.assert_array_equal(past,data.context(a['r'][:t],t,length=m['context_length']))
            before=time.perf_counter()
            if name in ('lstm','volpfn'):
                raw=native_prediction(model,past)
                h=float(raw['h']);q=raw['quantiles'];floor_hit=False;floor=None
                semantics='native mixture second moment';rearranged=False
                logdensity=float(logsumexp(
                    raw['logw']+student_t.logpdf(
                    (a['r'][t]-raw['loc'])/raw['scale'],raw['nu'])-np.log(raw['scale'])))
                squared_q=None
            else:
                raw=model.predict(past,'raw');sq=model.predict(past,'squared')
                q=raw['quantiles'];h,floor_hit,floor=metrics.positive_point(sq['point'],past)
                semantics=sq['point_semantics'];rearranged=raw['rearranged'] or sq['rearranged']
                logdensity=None;squared_q=sq['quantiles'].tolist()
            elapsed=time.perf_counter()-before
            if i==0:
                repeat=native_prediction(model,past)['quantiles'] if name in ('lstm','volpfn') else model.predict(past,'raw')['quantiles']
                np.testing.assert_allclose(q,repeat,rtol=1e-6,atol=1e-7)
                audit.append(dict(repeated_inference='passed',max_abs_delta=float(np.max(np.abs(q-repeat)))))
            if h<=0 or not np.isfinite(h) or np.any(np.diff(q)<0): raise FloatingPointError('Invalid prediction')
            rows.append(dict(**origin,target=float(a['r'][t]),context_start=str(a['date'][max(0,t-m['context_length'])]),
                context_end=str(a['date'][t-1]),context_count=len(past),raw_quantiles=np.asarray(q).tolist(),
                squared_quantiles=squared_q,h=h,h_semantics=semantics,positive_floor=floor,
                floor_activated=floor_hit,quantiles_rearranged=rearranged,
                mean_pinball=float(metrics.pinball(np.asarray(a['r'][t]),q)),
                qlike_descriptive=float(metrics.qlike(a['r'][t]**2,h)),native_log_density=logdensity,
                inference_seconds=elapsed))
            T.save_json(cell/'predictions.json',rows)
            print('PILOT_ORIGIN',name,origin['date'],round(elapsed,3),'seconds',flush=True)
        record.update(status='complete_engineering_only',origins=len(rows),checks=audit,
            mean_inference_seconds=float(np.mean([r['inference_seconds'] for r in rows])),
            forecast_sha256=T.sha(cell/'predictions.json'),not_evidence_of_superiority=True)
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));(cell/'failure.txt').write_text(traceback.format_exc())
        print('PILOT_FAILED',name,repr(exc),flush=True)
    finally:
        record.update(elapsed_seconds=time.perf_counter()-started,finished_utc=T.now())
        T.save_json(cell/'status.json',record)
    return record


def power_safe():
    if sys.platform!='darwin': return True
    import re
    power=subprocess.check_output(['pmset','-g','batt'],text=True)
    match=re.search(r'(\d+)%;\s*discharging',power)
    if match and int(match.group(1))<=10:
        raise RuntimeError('Low discharging battery: defer model computation until power is restored')
    return True


def run(out=DEFAULT_OUT):
    out=Path(out).resolve();m=verify(out);power_safe()
    with (out/'started.json').open('x') as stream: json.dump({'utc':T.now(),'pid':os.getpid()},stream)
    records=[]
    for name in m['order']:
        verify(out);power_safe()
        T.save_json(out/'status.json',dict(status='running',model=name,pid=os.getpid(),updated_utc=T.now()))
        with (out/(name+'.log')).open('x') as log:
            try:
                result=subprocess.run([str(ENVIRONMENTS[name]),'-u','-m','experiments.frontier_benchmark.run',
                    'worker','--out',str(out),'--model',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=m['timeout_seconds_per_model'])
                path=out/'models'/name/'status.json'
                record=json.loads(path.read_text()) if path.exists() else dict(model=name,status='failed',returncode=result.returncode)
            except subprocess.TimeoutExpired:
                record=dict(model=name,status='failed_timeout',seconds=m['timeout_seconds_per_model'])
                T.save_json(out/'models'/name/'status.json',record)
        records.append(record);T.save_json(out/'records.json',records)
    write_readout(out,records)
    T.save_json(out/'status.json',dict(status='development_pilot_finished',finished_utc=T.now(),
        completed=[r['model'] for r in records if r['status']=='complete_engineering_only'],
        unavailable=[r['model'] for r in records if r['status']!='complete_engineering_only'],
        final_market_scoring_performed=False))
    return records


def write_readout(out,records):
    lines=['# Stronger-model development integration pilot','',
        'Four SPX development origins in 2014 only. These results test integration and runtime, not superiority.',
        'No post-2014 observations reach a model. Final-period acquisition and scoring are not performed.','',
        '| Model | Status | Mean forecast seconds |','|---|---|---:|']
    for r in records:
        lines.append('| '+r['model']+' | '+r['status']+' | '+str(r.get('mean_inference_seconds','unavailable'))+' |')
    lines+=['','Raw quantiles are scored using pinball loss. Squared-return point semantics are retained; '
        'Chronos/Moirai medians are not claimed to be variance expectations. Native-point QLIKE is descriptive, '
        'with floor activations recorded. Full development-only scale calibration is still required.',
        '', 'The 32-update held-out-SPX LSTM checks execution only; it is not a competitive trained baseline. '
        'VolPFN seed8101 is an engineering checkpoint, not the final headline.','',
        'Next: complete TabPFN license/local model access; establish a properly trained and tuned LSTM; '
        'run broader development calibration and timing; freeze final contrasts and common dates; then score the locked later block once.']
    (Path(out)/'READOUT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','run','worker','verify','versions'))
    p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--model',choices=MODELS)
    a=p.parse_args()
    if a.command=='versions': print(json.dumps(versions()))
    elif a.command=='freeze': freeze(a.out);print('FROZEN',a.out)
    elif a.command=='verify': verify(a.out);print('VERIFIED')
    elif a.command=='worker':
        r=worker(a.out,a.model)
        if r['status']=='failed': raise SystemExit(1)
    else: run(a.out)


if __name__=='__main__': main()
