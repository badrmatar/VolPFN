"""Operational continuation: unchanged native predictions, atomic forecast blocks.

Separate seeds may run concurrently; each still uses the frozen single-origin
prediction function. No new model, scoring date, selection or statistical rule.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from experiments.programme_closure import confirm_benchmark as C, evaluate as E

BLOCK = 25


def now():
    return datetime.now(timezone.utc)


def load_blocks(directory, origins, identity):
    rows = []
    for path in sorted(Path(directory).glob('block_*.json')):
        block = E.read(path)
        E.require(block['identity'] == identity and block['start'] == len(rows), 'Cache identity or sequence mismatch')
        E.require(path.name == 'block_%06d.json' % len(rows), 'Unexpected cache block name')
        saved = block['rows']
        E.require(0 < len(saved) <= BLOCK and len(rows)+len(saved) <= len(origins), 'Invalid cache block length')
        for j, row in enumerate(saved):
            E.require(all(row[k] == v for k,v in origins[len(rows)+j].items()), 'Cached origin mismatch')
            E.require(np.isfinite(row['h']) and row['h'] > 0 and np.isfinite(row['raw_quantiles']).all(), 'Invalid cached prediction')
        rows.extend(saved)
    return rows


def configure():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def worker(plan_path, out, seed, limit):
    import torch
    from experiments.seed_replication import causal
    plan = C.verify_plan(plan_path)
    registry = E.read(out/'CHECKPOINT_REGISTRY.json')
    entry = next(e for e in registry['entries'] if e['seed']==seed and e['arm']=='broad')
    E.require(E.sha(entry['checkpoint']) == entry['checkpoint_sha256'], 'Changed checkpoint')
    configure()
    model = causal.load_model(entry['checkpoint'],plan['inference']['device'])
    identity = dict(plan=E.sha(plan_path), checkpoint=entry['checkpoint_sha256'],
                    runner=E.sha(__file__), seed=seed)
    start = time.monotonic(); generated = 0
    with np.load(Path(plan_path).parent/'contexts.npz',allow_pickle=False) as contexts:
        for phase,origins in [('calibration',plan['calibration_origins']),('evaluation',plan['origins'])]:
            directory=out/'cache'/str(seed)/phase; directory.mkdir(parents=True,exist_ok=True)
            rows=load_blocks(directory,origins,identity)
            while len(rows)<len(origins) and (limit==0 or generated<limit):
                n=min(BLOCK,len(origins)-len(rows),limit-generated if limit else BLOCK)
                begin=len(rows); new=[]
                for i in range(begin,begin+n):
                    prediction=C.native_prediction(model,contexts[phase][i],plan['inference']['device'])
                    new.append(dict(origins[i],**prediction,h_semantics='native mixture second moment',floor_activated=False))
                target=directory/('block_%06d.json'%begin)
                E.require(not target.exists(),'Refusing to overwrite cache')
                E.save(target,dict(identity=identity,start=begin,rows=new))
                rows.extend(new);generated+=n
                E.save(out/('status_%d.json'%seed),dict(status='running',seed=seed,phase=phase,
                       completed=len(rows),total=len(origins),new_forecasts=generated,elapsed_seconds=time.monotonic()-start))
            if len(rows)==len(origins):
                target=out/('%d_%s.json'%(seed,phase))
                if target.exists(): E.require(E.read(target)==rows,'Completed forecast file changed')
                else: E.save(target,rows)
            if limit and generated>=limit:break
    E.save(out/('status_%d.json'%seed),dict(status='pilot_complete' if limit else 'complete',seed=seed,
           new_forecasts=generated,elapsed_seconds=time.monotonic()-start))


def finalize(plan_path,out,registry):
    plan=C.verify_plan(plan_path)
    seed_rows={s:E.read(out/('%d_evaluation.json'%s)) for s in C.SEEDS}
    calibration={s:E.read(out/('%d_calibration.json'%s)) for s in C.SEEDS}
    for s in C.SEEDS:
        C.validate(seed_rows[s],plan['origins'])
        C.validate(calibration[s],plan['calibration_origins'])
    frontier=Path(plan['frontier']); old=E.read(frontier/'bundle/manifest.json')
    competitors={m:C.validate(E.read(frontier/'models'/m/'predictions.json'),old['origins']) for m in C.COMPETITORS}
    result,draws,counts=C.score_complete(seed_rows,calibration,competitors,plan['competitor_scales'],plan['statistics'])
    for entry in registry['entries']:
        E.require(E.sha(entry['checkpoint'])==entry['checkpoint_sha256'],'Checkpoint changed during inference')
    np.savez_compressed(out/'draws.npz',**draws);np.savez_compressed(out/'counts.npz',**counts)
    E.save(out/'FORECAST_HASHES.json',{p.name:E.sha(p) for p in out.glob('81*_*.json')})
    E.save(out/'results.json',dict(result,status='complete',plan_sha256=E.sha(plan_path)))
    lines=['# Corrected procedure: historical benchmark confirmation','',result['scope']+'.','',
           'Same frozen forecasts and score definitions; resumable per-seed execution. Positive gain favours VolPFN.','',
           '| Competitor | Seed-mean QLIKE gain | Family95% interval | Holm3 p | Positive seeds | Verdict |',
           '|---|---:|---|---:|---:|---|']
    for r in result['contrasts']:
        if r['block']==plan['statistics']['main_block']:
            lo,hi=r['family_interval95']
            lines.append('| %s | %.6g | [%.6g, %.6g] | %.6g | %d/5 | %s |'%
                         (r['model'],r['gain'],lo,hi,r['p_holm3'],r['positive_seeds'],r['verdict']))
    lines+=['','All five seeds and 8,774 evaluation rows per seed retained. Original checkpoint result remains separate.']
    (out/'READOUT.md').write_text('\n'.join(lines)+'\n')


def run(plan_path,training,out,deadline,pilot):
    plan=C.verify_plan(plan_path);registry=E.verify_checkpoints(plan['training_design'],training)
    E.require(registry['training_manifest_sha256']==plan['training_manifest_sha256'],'Wrong training procedure')
    out.mkdir(parents=True,exist_ok=True)
    binding=dict(plan_sha256=E.sha(plan_path),registry=registry,runner_sha256=E.sha(__file__))
    if (out/'BINDING.json').exists():E.require(E.read(out/'BINDING.json')==binding,'Resume binding changed')
    else:E.save(out/'BINDING.json',binding)
    E.save(out/'CHECKPOINT_REGISTRY.json',registry)
    lock=out/'active.lock';fd=os.open(str(lock),os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.close(fd)
    children=[];start=time.monotonic()
    def halt(signum=None,frame=None):
        for proc,log in children:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
        for proc,log in children:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if signum is not None:raise RuntimeError('Worker deadline/signal; cloud stop still required')
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    try:
        E.save(out/'status.json',dict(status='pilot' if pilot else 'running',started_utc=now().isoformat(),deadline_utc=deadline.isoformat()))
        for seed in C.SEEDS:
            log=(out/('worker_%d.log'%seed)).open('a')
            command=[sys.executable,'-u','-m','experiments.programme_closure.resume_confirmation','worker',
                     '--plan',str(plan_path),'--out',str(out),'--seed',str(seed),'--limit',str(50 if pilot else 0)]
            children.append((subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True),log))
        while any(p.poll() is None for p,l in children):
            E.require(now()<deadline,'Authorized worker deadline reached')
            E.require(not any(p.poll() not in (None,0) for p,l in children),'Seed worker failed; no survivor-only scoring')
            time.sleep(2)
        E.require(all(p.returncode==0 for p,l in children),'Seed worker failed')
        if not pilot:
            E.require(now()<deadline,'Deadline before statistical analysis')
            finalize(plan_path,out,registry)
        E.save(out/('PILOT.json' if pilot else 'status.json'),dict(status='pilot_complete' if pilot else 'complete_export_and_cloud_stop_required',
            elapsed_seconds=time.monotonic()-start,ended_utc=now().isoformat(),worker_statuses=[E.read(out/('status_%d.json'%s)) for s in C.SEEDS]))
    except BaseException as exc:
        halt();E.save(out/'status.json',dict(status='incomplete_or_failed',error=repr(exc)))
        (out/('failure_'+now().strftime('%H%M%S')+'.txt')).write_text(traceback.format_exc());raise
    finally:
        for p,l in children:l.close()
        lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','run','pilot']);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--training',type=Path);p.add_argument('--deadline');p.add_argument('--seed',type=int);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.action=='worker':worker(a.plan,a.out,a.seed,a.limit)
    else:run(a.plan,a.training,a.out,datetime.fromisoformat(a.deadline.replace('Z','+00:00')),a.action=='pilot')
