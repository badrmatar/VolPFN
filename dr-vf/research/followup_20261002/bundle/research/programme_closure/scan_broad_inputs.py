"""Finite pre-training numerical screen of every scheduled broad-prior batch.

No model, weights or market data. A failure aborts; no sample is replaced.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback
import numpy as np
from research.programme_closure import gpu_training as G, evaluate as E
from research.matched_prior import run as M


def check(job):
    seed,step,data_seed,batch,length=job
    try:
        data=G.generated(('broad',data_seed,batch,length))
        h=hashlib.sha256()
        for key in sorted(data):
            if isinstance(data[key],np.ndarray):
                h.update(key.encode());h.update(data[key].tobytes())
        return dict(seed=seed,step=step,data_seed=data_seed,batch=batch,length=length,sha256=h.hexdigest())
    except BaseException as exc:
        return dict(seed=seed,step=step,data_seed=data_seed,batch=batch,length=length,error=repr(exc),traceback=traceback.format_exc())


def run(frozen,out,workers=12):
    manifest=G.verify(frozen);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    jobs=[(seed,i+1,*entry) for seed in manifest['seeds'] for i,entry in enumerate(M.schedule(seed,manifest['training']))]
    assert len(jobs)==100000
    start=time.monotonic();done=0
    with multiprocessing.get_context('spawn').Pool(workers) as pool, (out/'batch_hashes.jsonl').open('x') as log:
        for result in pool.imap_unordered(check,jobs,chunksize=8):
            log.write(json.dumps(result)+'\n');done+=1
            if 'error' in result:
                E.save(out/'status.json',dict(status='failed',completed=done,total=len(jobs),failure=result))
                raise RuntimeError('Numerical preflight failed; no training authorized for this revision')
            if done%250==0:
                elapsed=time.monotonic()-start
                E.save(out/'status.json',dict(status='running',completed=done,total=len(jobs),elapsed_seconds=elapsed,
                                            projected_total_seconds=elapsed*len(jobs)/done))
                print('PREFLIGHT',done,len(jobs),round(elapsed,2),flush=True)
    E.save(out/'status.json',dict(status='complete',completed=done,total=len(jobs),elapsed_seconds=time.monotonic()-start,
        manifest_sha256=E.sha(Path(frozen)/'manifest.json'),batch_hashes_sha256=E.sha(out/'batch_hashes.jsonl'),
        scope='Every registered broad training input passed finite-array checks; no model fit or market score.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);a=p.parse_args();run(a.frozen,a.out,a.workers)
