"""Synthetic-only timing pilot. Hardware selection never consults market scores."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/volpfn-matplotlib')

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from experiments.programme_closure import gpu_training as G
from experiments.seed_replication import training as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--warmup', type=int, default=25)
    args = parser.parse_args()
    if args.steps <= args.warmup or args.warmup < 0:
        raise ValueError('Timing pilot needs observations after warmup')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    s = G.specification(args.device)
    s['steps'] = args.steps
    s['workers'] = 2
    G.configure(s)
    runtime = dict(python=platform.python_version(), numpy=np.__version__,
                   scipy=scipy.__version__, torch=str(torch.__version__),
                   cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None)
    T.save_json(out/'runtime.json', runtime)
    stream = None
    started = time.perf_counter()
    try:
        model, optimizer, initial = G.initialize(717301, s)
        stream = G.OrderedBatches('broad', 717301, s, stop=args.steps)
        events = []
        for step in range(args.steps):
            start = time.perf_counter()
            batch = stream.next()
            generation_wait = time.perf_counter() - start
            nll, aux, gradient = G.update(model, optimizer, batch, step, s)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            events.append(dict(step=step+1, length=batch['feats'].shape[1],
                               seconds=time.perf_counter()-start,
                               generation_wait=generation_wait, nll=nll, aux=aux,
                               gradient=gradient))
            if (step+1) % 25 == 0:
                T.save_json(out/'status.json', dict(status='running', updates=step+1))
                print('TIMING', step+1, events[-1]['seconds'], flush=True)
        post = events[args.warmup:]
        by_length = {}
        for length in (512, 1024):
            measured = [e['seconds'] for e in post if e['length'] == length]
            if not measured:
                raise ValueError('Timing pilot did not observe both length configurations after warmup')
            by_length[str(length)] = dict(n=len(measured), seconds=float(np.mean(measured)))
        weighted = sum(s['lengths'][key]*entry['seconds'] for key,entry in by_length.items())
        result = dict(status='complete', runtime=runtime, initial_sha256=initial,
                      steps=args.steps, warmup=args.warmup, by_length=by_length,
                      seconds_per_update=weighted,
                      projected_20000_updates_seconds=20000*weighted,
                      elapsed_seconds=time.perf_counter()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
                      selection='Timing and finite-gradient checks only; no market data or model selection')
        T.save_json(out/'events.json', events)
        T.save_json(out/'result.json', result)
        T.save_json(out/'status.json', {'status':'complete','updates':args.steps})
        print(json.dumps(result), flush=True)
    except Exception as error:
        T.save_json(out/'status.json', dict(status='failed', error=repr(error)))
        raise
    finally:
        if stream is not None:
            stream.close()


if __name__ == '__main__':
    main()
