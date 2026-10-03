import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.frontier_benchmark import data
from research.frontier_comparison import run as C
from research.runpod_transfer.worker import sha,save,verify
ROOT=data.ROOT
OUT=ROOT/'research/outputs/frontier_overnight_increment_201504_201606_v1'
BUNDLE=ROOT/'research/outputs/runpod_tabpfn_bundle_v1'
GPU=ROOT/'research/outputs/runpod_tabpfn_operation_v1/tabpfn_cuda_attempt_v1'
SCALE=ROOT/'research/outputs/frontier_development_comparison_cuda_v1/comparison.json'
MASK=ROOT/'research/outputs/evaluation_strengthening_v1/scoring_mask.csv'


def evaluation_origins(panel,mask):
    rows=mask.loc[(mask.date>='2015-04-01')&(mask.date<='2016-06-30')].copy()
    if rows.duplicated(['market','date']).any():raise ValueError('Duplicate scoring row')
    if set(rows.market)!=set(panel):raise ValueError('Incomplete population')
    origins=[]
    for r in rows.sort_values(['date','market']).itertuples():
        dates=panel[r.market]['date'];i=int(np.searchsorted(dates,r.date))
        if i>=len(dates) or str(dates[i])!=r.date or i<1024:raise ValueError('Invalid scoring origin')
        origins.append(dict(market=r.market,index=i,date=r.date,phase='evaluation'))
    return origins


def build():
    parent=C.verify(C.DEFAULT_OUT);verify(BUNDLE)
    original=json.loads((BUNDLE/'manifest.json').read_text());scale=json.loads(SCALE.read_text())
    assert scale['status']=='complete'
    panel={};source_hashes={}
    for name in data.markets():
        p=data.DATA/(name+'.npz');source_hashes[str(p)]=sha(p)
        with np.load(p) as z:
            keep=z['date']<='2016-06-30'
            panel[name]=dict(date=z['date'][keep],r=100*z['returns'][keep].astype(float))
    origins=evaluation_origins(panel,pd.read_csv(MASK))
    print('Building',len(origins),'new origins',flush=True)
    origins=[dict(o,phase='bridge') for o in original['origins'][:3]]+origins
    OUT.mkdir(exist_ok=False);bundle=OUT/'bundle';bundle.mkdir()
    pasts=[];xs=[];xns=[];full=[]
    for ordinal,o in enumerate(origins):
        if ordinal%500==0: print('INPUT',ordinal,len(origins),flush=True)
        a=panel[o['market']];i=o['index'];past=data.context(a['r'],i)
        x,xn=data.covariates(past)
        np.testing.assert_array_equal(past,a['r'][i-1024:i])
        pasts.append(past);xs.append(x);xns.append(xn)
        full.append(dict(o,target=float(a['r'][i]),context_count=len(past),context_start=str(a['date'][i-1024]),context_end=str(a['date'][i-1])))
    np.savez_compressed(bundle/'inputs.npz',past=np.array(pasts),features=np.array(xs),future_features=np.array(xns))
    for f in ['worker.py','tabpfn_adapter.py','requirements.txt','MODEL_LICENSE',original['checkpoint']]:shutil.copy2(BUNDLE/f,bundle/f)
    shutil.copy2(Path(__file__).with_name('PROTOCOL.md'),bundle/'PROTOCOL.md')
    refs=[dict(r,phase='bridge') for r in original['bridge_references']]
    manifest=dict(original,kind='fixed_overnight_extension_exposed_historical',origins=full,bridge_references=refs,
                  parent_transfer_manifest_sha256=sha(BUNDLE/'manifest.json'),files={p.name:sha(p) for p in bundle.iterdir()})
    save(bundle/'manifest.json',manifest);(bundle/'manifest.sha256').write_text(sha(bundle/'manifest.json')+'\n')
    verify(bundle)
    sources={}
    for f in list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('PROTOCOL.md')]:
        target=OUT/'source'/f.name;target.parent.mkdir(exist_ok=True);shutil.copy2(f,target);sources[str(f.resolve())]=sha(f)
    source_hashes.update({str(MASK):sha(MASK),str(SCALE):sha(SCALE)})
    for key in ['volpfn','chronos2','moirai2']:
        model=parent['models'][key]
        if key=='volpfn':source_hashes[model['path']]=sha(model['path'])
        else:source_hashes.update(model['provenance']['hashes'])
    for p in ['research/frontier_benchmark/adapters.py','research/frontier_benchmark/lstm.py','research/frontier_benchmark/metrics.py','research/frontier_benchmark/data.py','research/seed_replication/causal.py','volpfn.py','volpfn_prior.py']:
        source_hashes[str(ROOT/p)]=sha(ROOT/p)
    spec=dict(kind=manifest['kind'],models=parent['models'],environments=parent['environment'],order=list(C.ORDER),
              evaluation_start='2015-04-01',evaluation_end='2016-06-30',markets=data.markets(),
              evaluation_count=len(full)-3,scales={k:v['scale'] for k,v in scale['models'].items()},
              calibration='fixed three-market 2012-2013 pilot; provisional transfer, not full-panel calibration',
              bundle_manifest_sha256=sha(bundle/'manifest.json'),hashes=source_hashes,sources=sources,
              checkpoint_policy='fixed-final 8101 only; not seed mean',inferential_claims=False)
    spec['reused_january']={str(p.resolve()):sha(p) for p in (ROOT/'research/outputs/frontier_one_month_201501_v1').glob('models/*/predictions.json')}
    spec['reused_quarter']={str(p.resolve()):sha(p) for p in (ROOT/'research/outputs/frontier_quarter_increment_20150203_v1').glob('models/*/predictions.json')}
    assert len(full)-3==8774
    spec['uncertainty']=dict(main_block=10,sensitivity_blocks=[5,20],resamples=20000,seed=2026091902,family_slots=9,period='extension_only')
    spec['sources'].update({str((ROOT/p).resolve()):sha(ROOT/p) for p in ['research/frontier_staged/local.py','research/frontier_staged/report.py']})
    save(OUT/'study.json',spec);(OUT/'study.sha256').write_text(sha(OUT/'study.json')+'\n')
    print('Frozen',len(full)-3,'evaluation forecasts, 3 bridge origins, 28 markets')

if __name__=='__main__':build()
