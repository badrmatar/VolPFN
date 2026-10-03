"""Numerical identity only: no evaluation targets or forecast losses are used."""
from __future__ import annotations
import argparse
import contextlib
import platform
import time
import numpy as np
import torch
from research.combined_leaderboard.common import ROOT,OUTPUT,PLAN,CACHE,read,save,sha,now
from research.combined_leaderboard.pipeline import verify
from research.combined_leaderboard.neural import Predictor
from research.seed_replication import causal

OUT=ROOT/'research/outputs/combined_leaderboard_numerical_repair_v1'

def forward(p,context,mode):
    f=causal.feature_batch(np.asarray(context,float)[None])
    dtype=torch.float64 if mode=='float64_reference' else torch.float32
    features=torch.from_numpy(f['feats']).to(p.device,dtype=dtype)
    base=torch.from_numpy(f['base']).to(p.device,dtype=dtype)
    backend=(torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
             if mode=='math_float32' else contextlib.nullcontext())
    with torch.no_grad(),backend:
        pred=p.model(features,base)
    last={k:pred[k][0,-1].cpu().numpy() for k in ('logw','loc','logscale','nu')}
    unit=float(f['output_scale'][0,-1])
    w=np.exp(last['logw']);loc=last['loc']*unit;scale=np.exp(last['logscale'])*unit;nu=last['nu']
    second=float(np.sum(w*(loc**2+scale**2*nu/(nu-2))))
    mean=float(np.sum(w*loc))
    return [mean,second-mean**2,second]

def probe(indices,seed,modes):
    verify(OUTPUT)
    with np.load(PLAN.parent/'contexts.npz') as z:
        contexts=np.concatenate([z['calibration'],z['evaluation']])
    archived=np.array([r['h'] for phase in ('calibration','evaluation')
                       for r in read(CACHE/f'{seed}_{phase}.json')])
    result={}
    for mode in modes:
        p=Predictor(seed,'cpu',2)
        if mode=='float64_reference': p.model.double()
        if mode=='manual_float32': p.model.use_sdpa=False
        start=time.perf_counter()
        values=np.array([forward(p,contexts[i],mode) for i in indices])
        error=np.abs(values[:,2]-archived[indices])/archived[indices]
        result[mode]=dict(values=values,relative_errors=error,
            maximum_relative_error=float(error.max()),failed_indices=[int(indices[i]) for i in np.flatnonzero(error>1e-6)],
            seconds=time.perf_counter()-start)
        print(mode,'max',float(error.max()),'failed',result[mode]['failed_indices'],flush=True)
    return dict(created_utc=now(),seed=seed,indices=indices,archived=archived[indices],modes=result,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        freeze_sha256=sha(OUTPUT/'FREEZE.json'),code_sha256=sha(__file__),
        scope='Same weights, contexts and forecast functional; numerical diagnosis only; no new losses scored')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indices',type=int,nargs='+',default=[3741])
    p.add_argument('--start',type=int);p.add_argument('--count',type=int,default=64)
    p.add_argument('--seed',type=int,default=8101)
    p.add_argument('--modes',nargs='+',default=['default_float32','math_float32','manual_float32','float64_reference'])
    p.add_argument('--name',required=True)
    args=p.parse_args()
    indices=args.indices if args.start is None else list(range(args.start,args.start+args.count))
    result=probe(indices,args.seed,args.modes)
    save(OUT/(args.name+'.json'),result)

if __name__=='__main__':main()
