"""Finite, fixed-budget paired training and a separate synthetic timing gate.

No market inputs, scores, model selection or automatic paid escalation here.
The CLI benchmarks by default; a full run requires a separate launch receipt.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
import numpy as np
import scipy
import torch
import volpfn
from research.matched_prior import run as M, simulators as S
from research.seed_replication import training as T, causal

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [8101, 8102, 8103, 8104, 8105]
ARMS = ('broad', 'matern')


def specification(device='cuda'):
    s = M.specification('full', device)
    s.update(seeds=SEEDS, workers=2, prefetch_depth=4,
             checkpoint_every=2500, backend='CUDA float32; TF32 disabled; deterministic algorithms',
             checkpoint_selection='fixed final update 20000 only; validation is diagnostic',
             seed_failure='retain failure; no automatic retry, seed replacement or survivor-only headline')
    return s


def configure(s):
    torch.set_num_threads(s['threads'])
    if s['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent backend substitution')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                platform=platform.platform())


def generated(args):
    arm, data_seed, batch, length = args
    result = S.prepared_batch(arm, data_seed, batch, length)
    T.validate_batch(result)
    return result


class OrderedBatches:
    """Process timing cannot change which synthetic batch owns a step."""
    def __init__(self, arm, seed, s, start=0, stop=None):
        self.jobs = iter([(arm, *entry) for entry in list(M.schedule(seed, s))[start:stop]])
        self.queue = deque()
        self.depth = max(1, s['prefetch_depth'])
        self.pool = None
        if s['workers']:
            import multiprocessing
            self.pool = multiprocessing.get_context('spawn').Pool(s['workers'])
        self.exhausted = False

    def next(self):
        if not self.pool:
            return generated(next(self.jobs))
        while len(self.queue) < self.depth and not self.exhausted:
            try:
                self.queue.append(self.pool.apply_async(generated, (next(self.jobs),)))
            except StopIteration:
                self.exhausted = True
        if not self.queue:
            raise StopIteration
        return self.queue.popleft().get()

    def close(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()


def initialize(seed, s):
    torch.manual_seed(seed)
    # Initialize on CPU identically for both arms, then transfer to the GPU.
    model = volpfn.VolPFN(**s['cfg'])
    initial = M.state_hash(model)
    model.to(s['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['lr'],
        betas=tuple(s['optimizer']['betas']), weight_decay=s['optimizer']['weight_decay'],
        eps=s['optimizer']['eps'])
    return model, optimizer, initial


def update(model, optimizer, batch, step, s):
    bt = T.to_device(batch, s['device'])
    for group in optimizer.param_groups:
        group['lr'] = T.learning_rate(step, s)
    model.train()
    prediction = model(bt['feats'], bt['base'])
    for key, value in prediction.items():
        T.finite_tensor(value, 'output_' + key)
    nll, aux = causal.losses(prediction, bt)
    loss = nll + s['aux'] * aux
    T.finite_tensor(loss, 'loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), s['gradient_norm_clip'], error_if_nonfinite=True)
    T.finite_tensor(gradient, 'gradient')
    optimizer.step()
    T.finite_parameters(model)
    return float(nll.detach().cpu()), float(aux.detach().cpu()), float(gradient.detach().cpu())


def source_paths():
    paths = M.source_paths() + [Path(__file__), ROOT/'research/programme_closure/GPU_PROTOCOL.md',
        ROOT/'research/matched_prior_confirmatory/PROTOCOL.md',
        ROOT/'research/matched_prior_confirmatory/statistics.py',
        ROOT/'research/matched_prior_confirmatory/data_gate.py']
    return sorted(set(paths))


def freeze(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = T.sha(path)
    original = ROOT/'research/outputs/matched_prior_five_pairs_v1/manifest.json'
    m = dict(created_utc=T.now(), training=specification(), seeds=SEEDS, arms=list(ARMS),
        source_hashes=hashes, previous_design_sha256=T.sha(original),
        reuse='No old checkpoint enters the corrected CUDA primary. Preserve old results as separate evidence.',
        exposure='Amendment follows viewed seed8101 historical foundation-model results; no claim of pre-exposure amendment.',
        order=[dict(seed=seed, arm=arm) for i, seed in enumerate(SEEDS)
               for arm in (ARMS if i % 2 == 0 else ARMS[::-1])],
        runtime_policy='One runtime receipt from timing pilot must match every full run; relocation allowed, stack/GPU type fixed',
        full_launch='Requires separately recorded approved cost ceiling after timing report; never automatic',
        market_access='None in this module; primary data lock and evaluation are separate')
    T.save_json(out/'manifest.json', m)
    (out/'manifest.sha256').write_text(T.sha(out/'manifest.json')+'\n')
    return m


def verify(out):
    out = Path(out)
    if T.sha(out/'manifest.json') != (out/'manifest.sha256').read_text().strip():
        raise RuntimeError('Manifest hash mismatch')
    m = json.loads((out/'manifest.json').read_text())
    for relative, expected in m['source_hashes'].items():
        if T.sha(ROOT/relative) != expected or T.sha(out/'source'/relative) != expected:
            raise RuntimeError('Source drift: '+relative)
    return m


def benchmark(frozen, out, steps=350, warmup=50):
    if steps != 350 or warmup != 50:
        raise ValueError('Frozen timing gate is 350 updates per arm, first50 excluded from throughput')
    m = verify(frozen)
    s = m['training']
    configure(s)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    T.save_json(out/'runtime.json', runtime())
    records = []
    for arm in ARMS:
        stream = None
        start = time.perf_counter()
        try:
            model, optimizer, initial = initialize(7103, s)
            stream = OrderedBatches(arm, 7103, s, stop=steps)
            events = []
            for step in range(steps):
                t = time.perf_counter()
                batch = stream.next()
                generation_wait = time.perf_counter()-t
                nll, aux, grad = update(model, optimizer, batch, step, s)
                # Scalars synchronize CUDA; wall includes generation wait and host transfer.
                elapsed = time.perf_counter()-t
                event = dict(step=step+1, length=batch['feats'].shape[1], seconds=elapsed,
                             generation_wait=generation_wait, nll=nll, aux=aux, gradient=grad)
                events.append(event)
                if (step+1) % 25 == 0:
                    T.save_json(out/'status.json', dict(status='running', arm=arm, completed=step+1, total=steps))
                    print('TIMING', arm, step+1, round(elapsed,4), flush=True)
            panel_start = time.perf_counter()
            panels = M.shared_panels(s)
            panel_generation = time.perf_counter()-panel_start
            val_start = time.perf_counter()
            validation = M.evaluate_panels(model, panels, s)
            validation_seconds = time.perf_counter()-val_start
            save_start = time.perf_counter()
            T.checkpoint(out/(arm+'_pilot.pt'), model, optimizer, steps, [], dict(s,arm=arm), 7103, T.sha(Path(frozen)/'manifest.json'))
            save_seconds = time.perf_counter()-save_start
            post = events[warmup:]
            by_length = {str(length):dict(n=sum(x['length']==length for x in post),
                seconds=float(np.mean([x['seconds'] for x in post if x['length']==length])))
                for length in (512,1024)}
            weighted = sum(s['lengths'][key]*value['seconds'] for key,value in by_length.items())
            projected = s['steps']*weighted + 41*validation_seconds + 8*save_seconds + panel_generation
            records.append(dict(arm=arm,status='complete',initial_sha256=initial,by_length=by_length,
                validation_seconds=validation_seconds, panel_generation_seconds=panel_generation,
                checkpoint_seconds=save_seconds, update_seconds_weighted=weighted,
                full_run_projected_seconds=projected, elapsed_seconds=time.perf_counter()-start,
                validation=validation, events=events))
            del model, optimizer
            torch.cuda.empty_cache()
        except BaseException as exc:
            T.save_json(out/'status.json',dict(status='failed',arm=arm,error=repr(exc),at=T.now()))
            (out/'failure.txt').write_text(traceback.format_exc())
            raise
        finally:
            if stream:
                stream.close()
    if len({r['initial_sha256'] for r in records}) != 1:
        raise AssertionError('Initial weights differ by prior arm')
    result = dict(status='complete', manifest_sha256=T.sha(Path(frozen)/'manifest.json'),
                  runtime=runtime(), records=records,
                  ten_runs_projected_hours=5*sum(r['full_run_projected_seconds'] for r in records)/3600,
                  note='Synthetic timing only. Pilot checkpoints are never primary. No paid escalation.')
    T.save_json(out/'timing.json',result)
    T.save_json(out/'status.json',dict(status='complete',at=T.now()))
    return result


def train_one(arm, seed, s, out, manifest_hash, panels):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    configure(s)
    model, optimizer, initial = initialize(seed,s)
    torch.save(dict(state_dict=model.state_dict(),sha256=initial),out/'initial.pt')
    record = dict(status='running',arm=arm,seed=seed,initial_state_sha256=initial,
                  started_utc=T.now(),completed_updates=0,input_positions=0,manifest_sha256=manifest_hash)
    T.save_json(out/'status.json',record)
    stream = None
    start = time.perf_counter()
    try:
        history = [dict(step=0,**M.evaluate_panels(model,panels,s))]
        stream = OrderedBatches(arm,seed,s)
        for step in range(s['steps']):
            record['attempted_updates'] = step+1
            batch = stream.next()
            nll,aux,grad = update(model,optimizer,batch,step,s)
            record.update(completed_updates=step+1,input_positions=(step+1)*s['tokens'])
            if (step+1)%25==0:
                event = dict(step=step+1,nll=nll,aux=aux,gradient=grad,seconds=time.perf_counter()-start)
                with (out/'training.jsonl').open('a') as stream_log:
                    stream_log.write(json.dumps(event,allow_nan=False)+'\n')
                T.save_json(out/'status.json',record)
                print('TRAIN',arm,seed,step+1,round(event['seconds'],2),flush=True)
            if (step+1)%s['eval_every']==0:
                history.append(dict(step=step+1,**M.evaluate_panels(model,panels,s)))
                T.save_json(out/'validation.json',history)
            if (step+1)%s['checkpoint_every']==0:
                T.checkpoint(out/('step_%05d.pt'%(step+1)),model,optimizer,step+1,history,
                             dict(s,arm=arm),seed,manifest_hash)
        shutil.copy2(out/'step_20000.pt',out/'final.pt')
        record.update(status='complete',final_step=20000,checkpoint_sha256=T.sha(out/'final.pt'),
                      causal_probe=M.checkpoint_probe(out/'final.pt'))
    except BaseException as exc:
        record.update(status='failed',error=repr(exc))
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if stream: stream.close()
        record.update(ended_utc=T.now(),elapsed_seconds=time.perf_counter()-start)
        T.save_json(out/'status.json',record)
    return record


def full(frozen, out, authorization):
    m=verify(frozen)
    a=json.loads(Path(authorization).read_text())
    if (a.get('manifest_sha256') != T.sha(Path(frozen)/'manifest.json') or
        a.get('approved_full_battery') is not True or
        not np.isfinite(a.get('cost_ceiling_usd',0)) or a.get('cost_ceiling_usd',0)<=0 or
        not a.get('user_authorization_reference') or not a.get('timing_sha256') or
        not a.get('stop_deadline_utc') or not a.get('stop_enforcement')):
        raise RuntimeError('Missing concrete full-battery cost authorization after timing report')
    timing_path=Path(a['timing_path'])
    if T.sha(timing_path)!=a['timing_sha256']:
        raise RuntimeError('Timing receipt hash mismatch')
    timing=json.loads(timing_path.read_text())
    if timing.get('status')!='complete' or timing.get('manifest_sha256')!=a['manifest_sha256']:
        raise RuntimeError('No completed compatible timing gate')
    if timing['runtime']!=a['runtime']:
        raise RuntimeError('Authorization runtime differs from measured runtime')
    from datetime import datetime, timezone
    if datetime.fromisoformat(a['stop_deadline_utc'].replace('Z','+00:00'))<=datetime.now(timezone.utc):
        raise RuntimeError('Paid stop deadline has expired')
    expected=a['runtime']
    actual=runtime()
    for key in ('python','numpy','scipy','torch','cuda','gpu'):
        if expected[key]!=actual[key]: raise RuntimeError('Runtime mismatch: '+key)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    s=m['training'];configure(s)
    panels=M.shared_panels(s)
    np.savez_compressed(out/'shared_validation.npz',**{arm+'__'+k:v for arm,p in panels.items() for k,v in p.items() if isinstance(v,np.ndarray)})
    T.save_json(out/'runtime.json',actual)
    records=[]
    try:
        for entry in m['order']:
            verify(frozen)
            T.save_json(out/'status.json',dict(status='training',current=entry,completed=len(records)))
            r=train_one(entry['arm'],entry['seed'],s,out/'runs'/('%s_%d'%(entry['arm'],entry['seed'])),T.sha(Path(frozen)/'manifest.json'),panels)
            records.append(r)
        for seed in SEEDS:
            pair=[r for r in records if r['seed']==seed]
            if len(pair)!=2 or len({r['initial_state_sha256'] for r in pair})!=1:
                raise RuntimeError('Unmatched initial state for seed '+str(seed))
        T.save_json(out/'status.json',dict(status='complete',records=records))
    except BaseException as exc:
        T.save_json(out/'status.json',dict(status='incomplete_or_failed',records=records,error=repr(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','benchmark','full'])
    p.add_argument('--frozen',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--authorization',type=Path)
    a=p.parse_args()
    if a.action=='freeze': print(json.dumps(freeze(a.out),indent=2))
    elif a.action=='benchmark': benchmark(a.frozen,a.out)
    else: full(a.frozen,a.out,a.authorization)
