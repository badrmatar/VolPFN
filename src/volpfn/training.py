"""Finite-checked, deterministic-budget training for the new causal procedure.

Only synthetic validation may select checkpoints. No real-market code is
imported here. Interrupted or failed seeds are retained and never restarted.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
import volpfn
from . import _causal as causal

BASE_SEEDS = (8101, 8102, 8103, 8104, 8105)
PILOT_SEEDS = (7101, 7102)
PRIOR = dict(garch=.20, gjr=.15, egarch=.10, sv=.20, rough=.10, hmm=.12, lattice=.13)
OVERLAY = dict(jumps=.30, ar=.30, drift=.50, stale=.25)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def spec(kind='base', device='mps'):
    if kind not in ('base', 'pilot'):
        raise ValueError('Unknown run kind')
    base = kind == 'base'
    return dict(kind=kind, size='base' if base else 'small',
                steps=20000 if base else 200, tokens=8192 if base else 1024,
                lengths={'512': .4, '1024': .6} if base else {'128': .5, '256': .5},
                lr=.0003, warmup=300 if base else 20, aux=.1,
                eval_every=500 if base else 100, val_T=1024 if base else 256,
                val_n=48 if base else 8, validation_seed=777,
                validation_burn=100, evaluation_batch_size=1,
                seeds=list(BASE_SEEDS if base else PILOT_SEEDS),
                device=device, threads=2, workers=2, prefetch_depth=4,
                log_every=25, prior=PRIOR.copy(), overlay=OVERLAY.copy(),
                optimizer=dict(name='AdamW', betas=[.9, .98], weight_decay=.01, eps=1e-8),
                gradient_norm_clip=1., precision='float32_no_autocast',
                checkpoint_selection='minimum broad-prior raw-return validation NLL at fixed evaluation steps; first minimum wins',
                normalization_spec=causal.NORMALIZATION_SPEC,
                seed_failure='record and retain, no retry/substitution; all five needed for headline')


def generated_batch(seed, B, L, weights, overlays):
    return causal.prepare_batch(causal.raw_batch(seed, B, L, weights=weights, overlay_p=overlays))


class Batches:
    def __init__(self, seed, specification):
        self.s = specification
        self.seed = int(seed) * 100003
        self.rng = np.random.default_rng(self.seed + 99991)
        self.lengths = [int(k) for k in self.s['lengths']]
        self.probabilities = list(self.s['lengths'].values())
        self.pool = None
        self.queue = deque()
        if self.s['workers']:
            import multiprocessing
            self.pool = multiprocessing.get_context('spawn').Pool(self.s['workers'])

    def args(self):
        L = int(self.rng.choice(self.lengths, p=self.probabilities))
        self.seed += 1
        return self.seed, max(1, self.s['tokens'] // L), L, self.s['prior'], self.s['overlay']

    def next(self):
        if self.pool is None:
            return generated_batch(*self.args())
        while len(self.queue) < self.s['prefetch_depth']:
            self.queue.append(self.pool.apply_async(generated_batch, self.args()))
        return self.queue.popleft().get()

    def close(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()


def finite_tensor(value, stage):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('nonfinite_' + stage)


def finite_parameters(model):
    if not bool(torch.stack([torch.isfinite(v).all() for v in model.parameters()]).all()):
        raise FloatingPointError('nonfinite_parameters_after_update')


def learning_rate(step, specification):
    s = specification
    return s['lr'] * min(1., (step + 1) / s['warmup']) * (
        .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., step / max(1, s['steps'])))))


def to_device(batch, device):
    # Stable log scales are sufficient; MPS cannot store float64 output_scale.
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in batch.items() if isinstance(v, np.ndarray) and
            k in ('feats', 'base', 'target_r', 'target_logv', 'target_mask', 'output_log_scale')}


def validate_batch(batch):
    for k in ('feats', 'base', 'target_r', 'target_logv', 'target_mask', 'output_log_scale'):
        if not np.isfinite(batch[k]).all():
            raise FloatingPointError('nonfinite_generated_' + k)
    if batch['target_mask'].sum() <= 0:
        raise FloatingPointError('no_scored_training_targets')


@torch.no_grad()
def evaluate(model, validation, device, burn=100, bs=8):
    """Equal-series raw-return validation NLL, with targets before burn omitted."""
    model.eval()
    scores = []
    try:
        for i in range(len(validation['feats'])):
            # Per-series weighting intentionally matches the retained selector.
            batch = {k: v[i:i+1].copy() for k, v in validation.items() if isinstance(v, np.ndarray)}
            batch['target_mask'][:, :max(burn - 1, 0)] = 0
            validate_batch(batch)
            bt = to_device(batch, device)
            prediction = model(bt['feats'], bt['base'])
            for key, value in prediction.items():
                finite_tensor(value, 'validation_output_' + key)
            nll, aux = causal.losses(prediction, bt)
            finite_tensor(nll, 'validation_nll')
            finite_tensor(aux, 'validation_aux')
            scores.append(float(nll.cpu()))
        return dict(nll=float(np.mean(scores)), per_series_nll=scores, series=len(scores), burn=burn)
    finally:
        model.train()


def checkpoint(path, model, optimizer, step, history, specification, seed, manifest_sha):
    value = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                 cfg=model.cfg, size=specification['size'], step=step, seed=seed,
                 history=history, normalization_spec=causal.NORMALIZATION_SPEC, causal_spec=causal.CAUSAL_SPEC,
                 training_spec=specification, manifest_sha256=manifest_sha,
                 optimizer=optimizer.state_dict())
    tmp = Path(path).with_suffix('.pt.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def train_seed(seed, specification, output, manifest_sha):
    """Train a seed exactly once; return a terminal record even after failure."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    if seed not in specification['seeds']:
        raise ValueError('Seed outside frozen battery')
    started = time.perf_counter()
    record = dict(status='running', seed=seed, started_utc=now(),
                  manifest_sha256=manifest_sha, specification=specification,
                  completed_updates=0, attempted_updates=0, input_positions=0,
                  input_positions_seen=0, scored_targets=0,
                  environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                   machine=platform.machine(), device=specification['device']))
    save_json(out/'status.json', record)
    data = None
    best = float('inf')
    history = []
    validation_seconds = 0.
    generation_seconds = 0.
    try:
        device = torch.device(specification['device'])
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Frozen MPS backend unavailable; CPU is not silently substituted')
        torch.set_num_threads(specification['threads'])
        torch.manual_seed(seed)
        model = volpfn.VolPFN(**volpfn.SIZES[specification['size']]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=specification['lr'],
                                     betas=tuple(specification['optimizer']['betas']),
                                     weight_decay=specification['optimizer']['weight_decay'],
                                     eps=specification['optimizer']['eps'])
        validation = generated_batch(specification['validation_seed'], specification['val_n'], specification['val_T'],
                                     specification['prior'], specification['overlay'])
        validate_batch(validation)
        data = Batches(seed, specification)
        print('TRAIN', specification['kind'], seed, 'parameters', volpfn.count_params(model), 'device', device, flush=True)
        for step in range(specification['steps']):
            record['attempted_updates'] = step + 1
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, specification)
            generated = time.perf_counter()
            batch = data.next()
            record['input_positions_seen'] += int(np.prod(batch['feats'].shape[:2]))
            validate_batch(batch)
            generation_seconds += time.perf_counter() - generated
            bt = to_device(batch, device)
            prediction = model(bt['feats'], bt['base'])
            if not bool(torch.stack([torch.isfinite(v).all() for v in prediction.values()]).all()):
                raise FloatingPointError('nonfinite_training_output')
            nll, aux = causal.losses(prediction, bt)
            loss = nll + specification['aux'] * aux
            finite_tensor(loss, 'loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), specification['gradient_norm_clip'], error_if_nonfinite=True)
            finite_tensor(gradient_norm, 'gradient_norm')
            optimizer.step()
            finite_parameters(model)
            record['completed_updates'] = step + 1
            record['input_positions'] += int(np.prod(batch['feats'].shape[:2]))
            record['scored_targets'] += int(batch['target_mask'].sum())
            if step == 0 or (step + 1) % specification['log_every'] == 0:
                elapsed = time.perf_counter() - started
                event = dict(step=step+1, nll=float(nll.detach().cpu()), aux=float(aux.detach().cpu()),
                             gradient_norm=float(gradient_norm.cpu()), seconds=elapsed,
                             input_positions=record['input_positions'], scored_targets=record['scored_targets'])
                with (out/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, allow_nan=False)+'\n')
                print('STEP', seed, json.dumps(event), flush=True)
                save_json(out/'status.json', record)
            if (step+1) % specification['eval_every'] == 0 or step+1 == specification['steps']:
                t = time.perf_counter()
                result = evaluate(model, validation, device, specification['validation_burn'], specification['evaluation_batch_size'])
                validation_seconds += time.perf_counter()-t
                result.update(step=step+1)
                history.append(result)
                checkpoint(out/'final.pt', model, optimizer, step+1, history, specification, seed, manifest_sha)
                if result['nll'] < best:
                    best = result['nll']
                    checkpoint(out/'best.pt', model, optimizer, step+1, history, specification, seed, manifest_sha)
                    record['selected_step'] = step+1
                save_json(out/'validation.json', history)
                print('VALIDATION', seed, step+1, result['nll'], flush=True)
        record.update(status='complete', best_validation_nll=best, best_checkpoint_sha256=sha(out/'best.pt'),
                      final_checkpoint_sha256=sha(out/'final.pt'))
    except (KeyboardInterrupt, SystemExit) as error:
        record.update(status='interrupted', failure_type=type(error).__name__,
                      failure_code='interrupted_no_restart')
        (out/'failure.txt').write_text(traceback.format_exc())
        save_json(out/'status.json', record)
        raise
    except Exception as error:
        record.update(status='failed', failure_type=type(error).__name__, failure_code=str(error))
        (out/'failure.txt').write_text(traceback.format_exc())
        print('FAILED', seed, repr(error), flush=True)
    finally:
        if data is not None:
            data.close()
        record.update(completed_utc=now(), elapsed_seconds=time.perf_counter()-started,
                      validation_seconds=validation_seconds, generation_wait_seconds=generation_seconds)
        save_json(out/'status.json', record)
    return record
