"""Frozen inference only on the new ETF population; no new model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from historical import RegularizedVolPFN, A, V

HERE=Path(__file__).resolve().parent
OUT=HERE/'external_confirmation_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(seeds):
    design=json.loads((OUT/'protocol.json').read_text())
    execution=json.loads((OUT/'execution_protocol.json').read_text())
    if sha(__file__)!=execution['worker_sha256']:raise ValueError('Worker source drift')
    receipt=json.loads((OUT/'DATA_RECEIPT.json').read_text())
    if receipt['status']!='complete' or receipt['protocol_sha256']!=sha(OUT/'protocol.json') or receipt['mask_sha256']!=sha(OUT/'mask.csv'):
        raise ValueError('Incomplete or changed data/design')
    ids=json.loads((OUT/'checkpoint_identity.json').read_text())
    if sha(OUT/'checkpoint_identity.json')!=receipt['model_identity_sha256']:raise ValueError('Checkpoint roster drift')
    mask=pd.read_csv(OUT/'mask.csv',dtype={'date':str})
    data={}
    for market in design['tickers']:
        path=OUT/'data'/(market+'.npz')
        if sha(path)!=receipt['files'][str(path.relative_to(OUT))]:raise ValueError('New data changed')
        with np.load(path,allow_pickle=False) as z:data[market]={k:z[k].copy() for k in ['date','returns']}
    s=json.loads((HERE/'protocol.json').read_text())['training'];A.G.configure(s)
    for seed in seeds:
        if seed not in design['seeds']:raise ValueError('Unregistered seed')
        run=OUT/'neural_runs'/str(seed);run.mkdir(parents=True,exist_ok=False)
        start=time.time();input_hashes={};all_rows=[]
        for arm in ['full','noaux','historical_regularized']:
            if arm=='historical_regularized':
                path=HERE/'runs'/str(seed)/'best.pt';status=json.loads(path.with_name('status.json').read_text())
                if status['status']!='complete' or sha(path)!=status['checkpoint_sha256'] or status['smoke_unpromotable']:
                    raise ValueError('New historical model is not complete and verified')
                ck=torch.load(path,map_location='cpu',weights_only=False)
                if ck['protocol_sha256']!=sha(HERE/'protocol.json'):raise ValueError('Historical procedure drift')
                model=RegularizedVolPFN(dropout=ck['dropout'],**ck['cfg'])
            else:
                identity=next(x for x in ids if x['arm']==arm and x['seed']==seed)
                path=OUT/identity['file']
                if sha(path)!=identity['sha256']:raise ValueError('Original inference weight drift')
                ck=torch.load(path,map_location='cpu',weights_only=False);model=V.VolPFN(**ck['cfg'])
            model.load_state_dict(ck['state_dict']);model.cuda().eval();input_hashes[arm]=sha(path)
            rows=[]
            for market,g in mask.groupby('market',sort=True):
                raw=data[market];indices=g.row_index.to_numpy(int)
                if np.any(indices<1024) or not np.array_equal(raw['date'][indices],g.date.to_numpy(str)):raise ValueError('Target alignment')
                for begin in range(0,len(indices),32):
                    ix=indices[begin:begin+32];r=raw['returns'][ix]
                    pred=A.predict_windows(model,[raw['returns'][i-1024:i] for i in ix],batch_size=32,targets=r)
                    for j,i in enumerate(ix):
                        h=float(pred['second_moment'][j])
                        rows.append(dict(model=arm,seed=seed,market=market,date=str(raw['date'][i]),row_index=int(i),
                            context_end=str(raw['date'][i-1]),return_value=float(r[j]),second_moment=h,
                            mean=float(pred['mean'][j]),variance=float(pred['variance'][j]),
                            qlike=float(r[j]**2/h+np.log(h)),native_nll=float(pred['native_nll'][j])))
            frame=pd.DataFrame(rows).rename(columns={'return_value':'return'})
            if len(frame)!=len(mask) or frame.duplicated(['market','date']).any():raise ValueError('Complete exact mask required')
            frame.to_csv(run/(arm+'.csv'),index=False,float_format='%.17g');all_rows.extend(rows)
            print('EXTERNAL',seed,arm,len(frame),flush=True)
            del model;torch.cuda.empty_cache()
        result=dict(status='complete',seed=seed,rows_per_model=len(mask),input_checkpoint_hashes=input_hashes,
                    protocol_sha256=sha(OUT/'protocol.json'),execution_protocol_sha256=sha(OUT/'execution_protocol.json'),
                    runtime=A.G.runtime(),elapsed_seconds=time.time()-start,
                    exposure_classification='new recorded ETF population; human exposure unverified; do not label untouched',
                    forecast_hashes={p.name:sha(p) for p in run.glob('*.csv')})
        (run/'status.json').write_text(json.dumps(result,indent=2)+'\n')
        print('EXTERNAL COMPLETE',seed,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--seeds',type=int,nargs='+',required=True);a=cli.parse_args();worker(a.seeds)
