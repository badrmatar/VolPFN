"""Freeze and run independent matched-prior training; never reads market data."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
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
from research.seed_replication import causal, training as T
from . import simulators as S, diagnostics

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def specification(kind='pilot', device='cpu'):
    if kind not in ('pilot', 'full'):
        raise ValueError('unknown kind')
    if kind == 'pilot' and device != 'cpu':
        raise ValueError('engineering pilot is CPU-only')
    full = kind == 'full'
    s = T.spec('base' if full else 'pilot', device)
    s.update(kind='matched_prior_' + kind, size='base' if full else 'matched_pilot',
             cfg=dict(volpfn.SIZES['base']) if full else dict(d=64, layers=2, heads=4, K=4),
             seeds=[9201, 9202, 9203] if full else [9101, 9102],
             steps=20000 if full else 200, tokens=8192 if full else 512,
             lengths={'512': .4, '1024': .6} if full else {'64': .5, '128': .5},
             val_T=1024 if full else 128, val_n=48 if full else 12,
             validation_burn=100 if full else 32, eval_every=500 if full else 100,
             validation_seeds=dict(zip(S.ARMS, (99101, 99102, 99103))),
             threads=1 if not full else 2, workers=0, prefetch_depth=0,
             checkpoint_selection='fixed final update; validation diagnostic only; no fallback',
             seed_failure='retain every failure/interruption; no automatic retry or replacement',
             simulator_spec=S.SPECIFICATION)
    # A single 'prior' field could mislabel the different arms.
    s.pop('prior', None)
    s.pop('validation_seed', None)
    return s


def schedule(seed, s):
    """Independent of arm and simulation RNG consumption."""
    rng = np.random.default_rng(int(seed) * 100003 + 99991)
    lengths = np.array([int(k) for k in s['lengths']])
    probabilities = list(s['lengths'].values())
    for step in range(s['steps']):
        length = int(rng.choice(lengths, p=probabilities))
        if s['tokens'] % length:
            raise ValueError('input-position budget must be divisible by every length')
        yield int(seed) * 100003 + step + 1, s['tokens'] // length, length


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_paths():
    return sorted(list(PACKAGE.glob('*.py')) + [PACKAGE/'PROTOCOL.md',
                  ROOT/'volpfn.py', ROOT/'volpfn_prior.py',
                  ROOT/'research/seed_replication/causal.py',
                  ROOT/'research/seed_replication/training.py',
                  ROOT/'research/seed_replication/__init__.py'])


def verify(out):
    out = Path(out)
    path = out/'manifest.json'
    digest = T.sha(path)
    if digest != (out/'manifest.sha256').read_text().strip():
        raise RuntimeError('manifest changed')
    m = json.loads(path.read_text())
    for relative, expected in m['source_hashes'].items():
        if T.sha(ROOT/relative) != expected or T.sha(out/'source'/relative) != expected:
            raise RuntimeError('frozen source changed: ' + relative)
    return m


def freeze(out, kind='pilot', device='cpu'):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing experiment directory: ' + str(out))
    s = specification(kind, device)
    out.mkdir(parents=True)
    hashes = {}
    for path in source_paths():
        relative = str(path.relative_to(ROOT))
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative] = T.sha(path)
    manifest = dict(created_utc=T.now(), kind=kind, specification=s, source_hashes=hashes,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     torch=torch.__version__, platform=platform.platform(),
                                     thread_environment={k: os.environ.get(k) for k in (
                                         'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                         'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}),
                    market_access='none; synthetic engineering/training only',
                    scientific_claim='none from pilot', runtime_comparison='descriptive; shared machine')
    T.save_json(out/'manifest.json', manifest)
    (out/'manifest.sha256').write_text(T.sha(out/'manifest.json') + '\n')
    verify(out)
    return manifest


def shared_panels(s):
    return {arm: S.prepared_batch(arm, s['validation_seeds'][arm], s['val_n'], s['val_T'])
            for arm in S.ARMS}


def evaluate_panels(model, panels, s):
    cells = {arm: T.evaluate(model, panels[arm], s['device'], burn=s['validation_burn'])
             for arm in S.ARMS}
    return dict(equal_arm_nll=float(np.mean([x['nll'] for x in cells.values()])), cells=cells)


@torch.no_grad()
def checkpoint_probe(path):
    """Change future returns and ensure earlier predictive parameters agree."""
    model = causal.load_model(path, 'cpu')
    rng = np.random.default_rng(99771)
    raw = rng.standard_normal((2, 96))
    raw[0, :5] = 0.
    altered = raw.copy()
    altered[:, 49:] *= 37.
    a, b = causal.feature_batch(raw), causal.feature_batch(altered)
    pa = model(torch.from_numpy(a['feats']), torch.from_numpy(a['base']))
    pb = model(torch.from_numpy(b['feats']), torch.from_numpy(b['base']))
    delta = 0.
    for key in pa:
        x, y = pa[key][:, :49].numpy(), pb[key][:, :49].numpy()
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
        delta = max(delta, float(np.max(np.abs(x - y))))
    return dict(status='passed', maximum_predictive_parameter_difference=delta)


def train_arm(arm, seed, s, out, manifest_sha, panels):
    if arm not in S.ARMS or seed not in s['seeds']:
        raise ValueError('arm or seed outside frozen battery')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = dict(status='running', arm=arm, seed=seed, started_utc=T.now(),
                  manifest_sha256=manifest_sha, attempted_updates=0, completed_updates=0,
                  input_positions_seen=0, input_positions=0, scored_targets=0,
                  generation_seconds=0., validation_seconds=0.)
    history = []
    T.save_json(out/'status.json', record)
    try:
        device = torch.device(s['device'])
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Requested MPS backend unavailable; no silent fallback')
        torch.set_num_threads(s['threads'])
        torch.manual_seed(seed)
        model = volpfn.VolPFN(**s['cfg']).to(device)
        record.update(initial_state_sha256=state_hash(model), parameters=volpfn.count_params(model))
        optimizer = torch.optim.AdamW(model.parameters(), lr=s['lr'],
                                     betas=tuple(s['optimizer']['betas']),
                                     weight_decay=s['optimizer']['weight_decay'], eps=s['optimizer']['eps'])
        t = time.perf_counter()
        history.append(dict(step=0, **evaluate_panels(model, panels, s)))
        record['validation_seconds'] += time.perf_counter() - t
        for step, (data_seed, batch_size, length) in enumerate(schedule(seed, s)):
            record['attempted_updates'] = step + 1
            t = time.perf_counter()
            batch = S.prepared_batch(arm, data_seed, batch_size, length)
            record['generation_seconds'] += time.perf_counter() - t
            record['input_positions_seen'] += batch_size * length
            T.validate_batch(batch)
            bt = T.to_device(batch, device)
            for group in optimizer.param_groups:
                group['lr'] = T.learning_rate(step, s)
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
            record['completed_updates'] = step + 1
            record['input_positions'] += batch_size * length
            record['scored_targets'] += int(batch['target_mask'].sum())
            if (step + 1) % s['log_every'] == 0 or step == 0:
                event = dict(step=step+1, nll=float(nll.detach().cpu()), aux=float(aux.detach().cpu()),
                             seconds=time.perf_counter()-started, input_positions=record['input_positions'])
                with (out/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, allow_nan=False) + '\n')
                T.save_json(out/'status.json', record)
            if (step + 1) % s['eval_every'] == 0 or step + 1 == s['steps']:
                t = time.perf_counter()
                row = dict(step=step+1, **evaluate_panels(model, panels, s))
                history.append(row)
                record['validation_seconds'] += time.perf_counter() - t
                T.save_json(out/'validation.json', history)
                print('VALIDATION', arm, seed, step+1, row['equal_arm_nll'], flush=True)
        if record['input_positions'] != s['steps'] * s['tokens']:
            raise AssertionError('input budget mismatch')
        arm_specification = dict(s, arm=arm)
        T.checkpoint(out/'final.pt', model, optimizer, s['steps'], history, arm_specification, seed, manifest_sha)
        record.update(checkpoint_sha256=T.sha(out/'final.pt'),
                      checkpoint_probe=checkpoint_probe(out/'final.pt'),
                      final_step=s['steps'], status='complete', completed_utc=T.now(),
                      initial_validation_nll=history[0]['equal_arm_nll'],
                      final_validation_nll=history[-1]['equal_arm_nll'],
                      final_validation_by_arm={a: c['nll'] for a, c in history[-1]['cells'].items()})
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      failure_code=type(exc).__name__ + ': ' + str(exc), failed_utc=T.now())
        (out/'failure.txt').write_text(traceback.format_exc())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        record['elapsed_seconds'] = time.perf_counter() - started
        T.save_json(out/'status.json', record)
    return record


def summarize(records, s):
    complete = len(records) == len(s['seeds']) * len(S.ARMS) and all(r['status'] == 'complete' for r in records)
    matched = True
    for seed in s['seeds']:
        group = [r for r in records if r['seed'] == seed]
        matched &= (len(group) == len(S.ARMS) and
                    len({r.get('initial_state_sha256') for r in group}) == 1 and
                    all(r['input_positions'] == s['steps'] * s['tokens'] for r in group))
    return dict(status='complete' if complete and matched else 'incomplete_or_failed',
                engineering_gate=bool(complete and matched), matched_initialization_and_budget=bool(matched),
                scope='synthetic engineering check only; no market or prior-superiority conclusion', records=records)


def write_readout(out, result):
    rows = ['# Matched-prior engineering readout', '',
            'Engineering gate: **' + ('PASS' if result['engineering_gate'] else 'NOT PASSED') + '**.', '',
            result['scope'] + '.', '',
            '| Arm | Seed | Status | Updates | Initial shared NLL | Final shared NLL | Seconds |',
            '|---|---:|---|---:|---:|---:|---:|']
    for r in result['records']:
        initial, final = r.get('initial_validation_nll'), r.get('final_validation_nll')
        rows.append('| {} | {} | {} | {} | {} | {} | {:.1f} |'.format(
            r['arm'], r['seed'], r['status'], r['completed_updates'],
            '—' if initial is None else f'{initial:.6f}',
            '—' if final is None else f'{final:.6f}', r['elapsed_seconds']))
    rows += ['', 'Lower NLL is better. The shared mean weights each of the three generating priors equally; '
             'per-prior scores are retained in summary.json and each validation.json. These small models '
             'and panels do not rank the priors for markets.', '',
             'All named runs and failures are retained. No validation-best substitution, market scoring, '
             'or automatic full-scale escalation occurs. See manifest.json for source hashes and configuration, '
             'synthetic_diagnostics.json for prior-distribution differences, and covariance_audit.json for numerical checks.', '',
             'The active replication uses a different checkpoint selector and is reported separately. '
             'Before full-scale market inference, finalize the scoring protocol, seed-uncertainty treatment '
             'and data-exposure inventory. A positive engineering gate is not evidence of increased novelty.', '']
    (out/'READOUT.md').write_text('\n'.join(rows))


def run(out):
    out = Path(out).resolve()
    m = verify(out)
    s = m['specification']
    # Exclusive marker makes concurrent launches and accidental retries fail closed.
    with (out/'started.json').open('x') as stream:
        json.dump(dict(started_utc=T.now(), pid=os.getpid()), stream)
    torch.set_num_threads(s['threads'])
    records = []
    try:
        covariance = [row for length in sorted(set(map(int, s['lengths'])) | {s['val_T']})
                      for row in S.covariance_audit(length)]
        if max(row['max_reconstruction_error'] for row in covariance) > 1e-10:
            raise FloatingPointError('covariance reconstruction tolerance exceeded')
        T.save_json(out/'covariance_audit.json', covariance)
        T.save_json(out/'synthetic_diagnostics.json', diagnostics.run())
        panels = shared_panels(s)
        # Retain actual shared arrays as well as deterministic generator seeds.
        arrays = {arm + '__' + key: value for arm, panel in panels.items() for key, value in panel.items()}
        np.savez_compressed(out/'shared_validation.npz', **arrays)
        for i, seed in enumerate(s['seeds']):
            order = S.ARMS[i % len(S.ARMS):] + S.ARMS[:i % len(S.ARMS)]
            for arm in order:
                verify(out)
                T.save_json(out/'status.json', dict(status='training', current_arm=arm, current_seed=seed,
                                                   completed_runs=len(records), updated_utc=T.now()))
                record = train_arm(arm, seed, s, out/'runs'/f'{arm}_{seed}', T.sha(out/'manifest.json'), panels)
                records.append(record)
                print('TERMINAL', arm, seed, record['status'], flush=True)
        verify(out)
        result = summarize(records, s)
        result['shared_validation_sha256'] = T.sha(out/'shared_validation.npz')
        T.save_json(out/'summary.json', result)
        write_readout(out, result)
        T.save_json(out/'status.json', dict(status=result['status'], engineering_gate=result['engineering_gate'],
                                          finished_utc=T.now(), completed_runs=len(records)))
        return result
    except BaseException as exc:
        T.save_json(out/'status.json', dict(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                                          failure_code=type(exc).__name__ + ': ' + str(exc), completed_runs=len(records)))
        (out/'failure.txt').write_text(traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--out', required=True)
    f.add_argument('--kind', choices=('pilot', 'full'), default='pilot')
    f.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'freeze':
        freeze(a.out, a.kind, a.device)
        print('FROZEN', a.out)
    elif a.command == 'verify':
        verify(a.out)
        print('VERIFIED', a.out)
    else:
        result = run(a.out)
        if not result['engineering_gate']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
