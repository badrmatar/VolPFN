"""Portable CUDA-only TabPFN continuation; predictions are a separate attempt."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
for key in ('TABPFN_DISABLE_TELEMETRY','HF_HUB_OFFLINE','HF_HUB_DISABLE_TELEMETRY','DO_NOT_TRACK'):
    os.environ[key]='1'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify(root):
    root=Path(root)
    if sha(root/'manifest.json')!=(root/'manifest.sha256').read_text().strip():
        raise RuntimeError('Manifest drift')
    m=json.loads((root/'manifest.json').read_text())
    for name,digest in m['files'].items():
        if sha(root/name)!=digest: raise RuntimeError('Input drift: '+name)
    with np.load(root/'inputs.npz',allow_pickle=False) as z:
        if z['past'].shape!=(len(m['origins']),1024): raise RuntimeError('Context shape changed')
        if z['features'].shape!=(len(m['origins']),1023,8): raise RuntimeError('Feature shape changed')
        if z['future_features'].shape!=(len(m['origins']),1,8): raise RuntimeError('Future feature shape changed')
        if not all(np.isfinite(z[k]).all() for k in z.files): raise RuntimeError('Nonfinite input')
    return m


def agrees(row,reference,tolerances):
    return all(np.allclose(row[key],reference[key],**tolerances)
               for key in ('raw_quantiles','squared_quantiles','h'))


def main(root,out):
    import torch
    from tabpfn_adapter import TabPFN
    root=Path(root).resolve();out=Path(out).resolve();m=verify(root)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    if out.exists(): raise FileExistsError('Preserve every attempt; choose a new output directory')
    out.mkdir(parents=True)
    runtime={name:importlib.metadata.version(name) for name in m['packages']}
    for name,expected in m['packages'].items():
        if runtime[name].split('+')[0]!=expected: raise RuntimeError('Package mismatch: '+name)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9182026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(out/'runtime.json',dict(packages=runtime,python=platform.python_version(),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),precision='float32',tf32=False,
        manifest_sha256=sha(root/'manifest.json')))
    rows=[];bridge=[];started=time.time()
    save(out/'status.json',dict(status='running',started_unix=started,pid=os.getpid()))
    try:
        model=TabPFN(root/m['checkpoint'])
        with np.load(root/'inputs.npz',allow_pickle=False) as z:
            for i,origin in enumerate(m['origins']):
                before=time.perf_counter()
                past=z['past'][i];x=z['features'][i];xn=z['future_features'][i]
                raw=model.predict(past,'raw',x,xn);sq=model.predict(past,'squared',x,xn)
                torch.cuda.synchronize()
                duration=time.perf_counter()-before
                floor=max(1e-12,1e-8*float(np.mean(past**2)))
                h=max(sq['point'],floor);target=origin['target']
                error=target-raw['quantiles'];levels=np.arange(1,10)/10
                row=dict(origin,raw_quantiles=raw['quantiles'].tolist(),
                    squared_quantiles=sq['quantiles'].tolist(),h=h,h_semantics=sq['point_semantics'],
                    positive_floor=floor,floor_activated=bool(sq['point']<floor),
                    quantiles_rearranged=bool(raw['rearranged'] or sq['rearranged']),
                    mean_pinball=float(np.maximum(levels*error,(levels-1)*error).mean()),
                    qlike_descriptive=float(target**2/h+np.log(h)),native_log_density=None,
                    inference_seconds=duration)
                if i<len(m['bridge_references']):
                    ref=m['bridge_references'][i]
                    ok=agrees(row,ref,m['bridge_tolerances'])
                    bridge.append(dict(index=i,passed=ok,seconds=duration,
                        maximum_raw_quantile_delta=float(np.max(np.abs(np.array(row['raw_quantiles'])-ref['raw_quantiles']))),
                        variance_delta=float(h-ref['h'])))
                    save(out/'bridge.json',bridge)
                    if not ok: raise RuntimeError('CPU/CUDA bridge failed; stop without tuning tolerances')
                rows.append(row);save(out/'predictions.json',rows)
                save(out/'status.json',dict(status='running',completed=len(rows),total=len(m['origins']),
                    elapsed_seconds=time.time()-started,latest_seconds=duration))
                print('FORECAST',len(rows),len(m['origins']),origin['market'],origin['date'],round(duration,3),flush=True)
        save(out/'status.json',dict(status='complete',completed=len(rows),elapsed_seconds=time.time()-started,
            forecast_sha256=sha(out/'predictions.json'),bridge_passed=all(r['passed'] for r in bridge)))
    except BaseException as exc:
        save(out/'status.json',dict(status='failed',completed=len(rows),error=repr(exc)))
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
