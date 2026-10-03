import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['TABPFN_DISABLE_TELEMETRY']='1';os.environ['HF_HUB_OFFLINE']='1'
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from research.runpod_transfer.worker import sha,save,verify
from research.frontier_benchmark import adapters,metrics,run as R
from research.seed_replication import causal


def main(out,name):
 import torch
 out=Path(out);spec=json.loads((out/'study.json').read_text())
 if sha(out/'study.json')!=(out/'study.sha256').read_text().strip():raise RuntimeError('Study drift')
 for p,h in {**spec['hashes'],**spec['sources']}.items():
  if sha(p)!=h:raise RuntimeError('Frozen source/input drift: '+p)
 m=verify(out/'bundle')
 if sha(out/'bundle/manifest.json')!=spec['bundle_manifest_sha256']:raise RuntimeError('Bundle drift')
 if R.versions()!=spec['environments'][name]:raise RuntimeError('Runtime drift')
 R.power_safe();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9182026)
 cell=out/'models'/name;cell.mkdir(parents=True,exist_ok=False);start=time.time();rows=[]
 save(cell/'status.json',dict(status='running',pid=os.getpid(),started_unix=start))
 try:
  path=spec['models'][name]['path'];model=causal.load_model(path,'cpu') if name=='volpfn' else adapters.make(name,path)
  with np.load(out/'bundle/inputs.npz') as z:
   for i,o in enumerate(m['origins']):
    R.power_safe();past=z['past'][i];before=time.perf_counter()
    if name=='volpfn':
     p=R.native_prediction(model,past);q=p['quantiles'];h=p['h'];sq=None;floor=None;hit=False;rearranged=False;semantics='native mixture second moment'
    else:
     p=model.predict(past,'raw');s=model.predict(past,'squared');q=p['quantiles'];sq=s['quantiles'].tolist();h,hit,floor=metrics.positive_point(s['point'],past);rearranged=p['rearranged'] or s['rearranged'];semantics=s['point_semantics']
    rows.append(dict(o,raw_quantiles=np.asarray(q).tolist(),squared_quantiles=sq,h=h,h_semantics=semantics,positive_floor=floor,floor_activated=hit,quantiles_rearranged=rearranged,inference_seconds=time.perf_counter()-before))
    save(cell/'predictions.json',rows);save(cell/'status.json',dict(status='running',completed=len(rows),total=len(m['origins']),elapsed_seconds=time.time()-start))
    if len(rows)%25==0:print(name,len(rows),len(m['origins']),flush=True)
  save(cell/'status.json',dict(status='complete',completed=len(rows),forecast_sha256=sha(cell/'predictions.json'),elapsed_seconds=time.time()-start))
 except BaseException as e:
  save(cell/'status.json',dict(status='failed',completed=len(rows),error=repr(e)));(cell/'failure.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--model',required=True,choices=['volpfn','chronos2','moirai2']);a=p.parse_args();main(a.out,a.model)
