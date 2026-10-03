"""Finite CPU queue for the frozen extension; never provisions infrastructure.

At most three child processes, each with two numerical-library threads. Each
child owns one model/seed and resumes immutable market cells. A failed job is
recorded once and is not retried or substituted automatically. Deadline stopping
may leave an unfinished market cell, which is explicitly absent on the next run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .prepare import OUT
from .run import atomic, cell, relative_cell, validate_rows, verify
from research.combined_leaderboard.common import now, require, sha


def build_jobs(bundle, out, *, models=None, markets=None, seeds=None):
    """Use prepared interpreter paths; no PATH-based runtime substitution."""
    models = models or ['ttm_rk', 'ttm_sqrt_rk', 'ms_har', 'log_rhar'] + [
        m for m in bundle['model_specs'] if m not in ('tabpfn_ts', 'ttm_rk', 'ttm_sqrt_rk', 'ms_har', 'log_rhar')]
    markets = markets or bundle['markets']
    require(len(set(models)) == len(models) and set(models) <= set(bundle['runtime_requirements']), 'Undeclared local models')
    require(len(set(markets)) == len(markets) and set(markets) <= set(bundle['markets']), 'Undeclared markets')
    require(seeds is None or ('volpfn' in models and len(set(seeds)) == len(seeds)
            and set(seeds) <= set(bundle['model_specs']['volpfn']['seeds'])), 'Undeclared seeds')
    jobs = []
    for model in models:
        for seed in (seeds or bundle['model_specs'][model]['seeds']) if model == 'volpfn' else [None]:
            runtime = bundle['runtime_requirements'][model]
            command = [runtime['executable'], '-B', '-m', 'research.frontier_extension_20260926.run',
                       'run', '--out', str(Path(out).absolute()), '--models', model,
                       '--markets', *markets, '--device', 'cpu', '--threads', '2',
                       '--batch-size', '64' if model.startswith('ttm_') else '1']
            if seed is not None:
                command += ['--seeds', str(seed)]
            jobs.append(dict(model=model, seed=seed, markets=list(markets), command=command))
    return jobs


def deadline_timestamp(start, hours, requested=None):
    require(0 < hours <= 8, 'Duration must be positive and no greater than eight hours')
    deadline = start + hours * 3600
    if requested:
        parsed = datetime.fromisoformat(requested.replace('Z', '+00:00'))
        require(parsed.tzinfo is not None, 'Explicit timezone required for deadline')
        deadline = min(deadline, parsed.timestamp())
    require(deadline > start, 'Deadline has already elapsed')
    return deadline


def complete(out, identity, job, bundle):
    for market in job['markets']:
        value = cell(out, relative_cell(job['model'], job['seed'], market), identity)
        if value is None:
            return False
        origins = [r for r in bundle['rows'] if r['market'] == market]
        validate_rows(value['rows'], origins, job['model'], job['seed'])
    return True


def stop_children(active, grace_seconds=10):
    """Bounded cleanup also covers supervisor exceptions and process-exit races."""
    def signal_group(item, sig):
        if item['process'].poll() is None:
            try:
                os.killpg(item['process'].pid, sig)
            except ProcessLookupError:
                pass
    for item in active:
        signal_group(item, signal.SIGTERM)
    cutoff = time.monotonic() + grace_seconds
    while any(i['process'].poll() is None for i in active) and time.monotonic() < cutoff:
        time.sleep(.2)
    for item in active:
        signal_group(item, signal.SIGKILL)
        item['process'].wait()
        item['log'].close()


def supervise(args):
    out = Path(args.out)
    bundle, _ = verify(out)
    require(1 <= args.workers <= 3, 'At most three local CPU workers are authorized')
    identity = sha(out / 'FREEZE.json')
    require(bundle.get('budget', {}).get('absolute_stop_utc'), 'Frozen absolute operation deadline required')
    started = time.time()
    deadline = deadline_timestamp(started, args.hours, bundle['budget']['absolute_stop_utc'])
    if args.deadline_utc:
        deadline = min(deadline, deadline_timestamp(started, args.hours, args.deadline_utc))
    jobs = build_jobs(bundle, out, models=args.models, markets=args.markets, seeds=args.seeds)
    run_id = uuid.uuid4().hex
    destination = out / 'supervisor' / run_id
    destination.mkdir(parents=True)
    atomic(destination / 'plan.json', dict(created_utc=now(), freeze_sha256=identity,
           started_timestamp=started, deadline_timestamp=deadline, workers=args.workers, jobs=jobs))
    environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       VECLIB_MAXIMUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    queued = [j for j in jobs if not complete(out, identity, j, bundle)]
    skipped = len(jobs) - len(queued)
    active, finished, failed, stopped = [], [], [], []
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    def event(value):
        with (destination / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=now(), **value)) + '\n')
        print(json.dumps(value), flush=True)
    try:
        with (out / '.local_supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while queued or active:
                if time.time() >= deadline or stop_requested:
                    stop_children(active)
                    stopped.extend(item['job'] for item in active)
                    event(dict(event='stopped', reason='requested' if stop_requested else 'deadline',
                               active_jobs=len(stopped), queued_jobs=len(queued)))
                    active = []
                    break
                while queued and len(active) < args.workers:
                    job = queued.pop(0); attempt = uuid.uuid4().hex
                    log = (destination / (attempt + '.log')).open('x')
                    process = subprocess.Popen(job['command'], cwd=str(Path(__file__).resolve().parents[2]),
                        env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active.append(dict(job=job, process=process, log=log, attempt=attempt))
                    event(dict(event='started', model=job['model'], seed=job['seed'], pid=process.pid,
                               log=str(destination / (attempt + '.log'))))
                for item in list(active):
                    code = item['process'].poll()
                    if code is None:
                        continue
                    item['log'].close(); active.remove(item)
                    success = code == 0 and complete(out, identity, item['job'], bundle)
                    (finished if success else failed).append(item['job'])
                    atomic(destination / (item['attempt'] + '_terminal.json'), dict(
                        ended_utc=now(), exit_code=code, complete=success, job=item['job']))
                    event(dict(event='finished' if success else 'failed', model=item['job']['model'],
                               seed=item['job']['seed'], exit_code=code))
                if queued or active:
                    time.sleep(min(2., max(0., deadline - time.time())))
    finally:
        # Exceptions also stop our children; no orphaned unlimited queue survives.
        stop_children(active)
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
    result = dict(status='complete' if not failed and not queued and not stopped else 'incomplete',
                  ended_utc=now(), skipped_complete=skipped, completed=finished, failed=failed,
                  stopped=stopped, queued=queued, freeze_sha256=identity)
    atomic(destination / 'terminal.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--hours', type=float, default=8.)
    parser.add_argument('--deadline-utc', help='ISO timestamp with explicit UTC offset')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--markets', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    print(json.dumps(supervise(parser.parse_args())), flush=True)


if __name__ == '__main__':
    main()
