"""New development-selected historical procedure; old experiment is immutable."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
import volpfn.legacy as V
from experiments.experiment_programme_20260930 import ablations as A
from experiments.seed_replication import training as T


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    p = json.loads((HERE / 'protocol.json').read_text())
    for name, digest in p['input_hashes'].items():
        if sha(HERE / name) != digest:
            raise RuntimeError('Input/source drift: ' + name)
    return p


class RegularizedVolPFN(V.VolPFN):
    def __init__(self, dropout=.1, **cfg):
        super().__init__(**cfg)
        # Modules have no parameters, preserving the seeded initial weights.
        for block in self.blocks:
            block.proj = nn.Sequential(block.proj, nn.Dropout(dropout))
            block.mlp = nn.Sequential(*list(block.mlp), nn.Dropout(dropout))


def read_data(p):
    result = {}
    for market in p['markets']:
        with np.load(HERE / 'bundle/data' / (market + '.npz'), allow_pickle=False) as z:
            dates = z['date'].astype('U10')
            returns = z['returns'].astype(np.float64) * 100.
            kernel = z['rk_parzen'].astype(np.float64) * 10000.
        if np.any(dates[1:] <= dates[:-1]) or not np.isfinite(returns).all():
            raise ValueError('Invalid chronological source')
        result[market] = dict(date=dates, returns=returns, kernel=kernel)
    return result


@torch.no_grad()
def evaluate(model, mask, data, batch_size=32, save=None, seed=None):
    records = []
    for market, block in mask.groupby('market', sort=True):
        raw = data[market]
        indices = block.row_index.to_numpy(int)
        if np.any(indices < 1024) or not np.array_equal(raw['date'][indices], block.date.to_numpy(str)):
            raise ValueError('Origin/date/context mismatch')
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            observed = raw['returns'][idx]
            pred = A.predict_windows(model, [raw['returns'][i-1024:i] for i in idx],
                                     batch_size=batch_size, targets=observed)
            h = pred['second_moment']
            for j, i in enumerate(idx):
                kernel = raw['kernel'][i]
                records.append(dict(model='historical_regularized', seed=seed, market=market,
                    date=str(raw['date'][i]), row_index=int(i), context_count=1024,
                    context_end=str(raw['date'][i-1]), return_value=float(observed[j]),
                    second_moment=float(h[j]), mean=float(pred['mean'][j]),
                    variance=float(pred['variance'][j]), qlike=float(observed[j]**2/h[j]+np.log(h[j])),
                    native_nll=float(pred['native_nll'][j]),
                    kernel=float(kernel) if np.isfinite(kernel) and kernel > 0 else None,
                    kernel_qlike=float(kernel/h[j]+np.log(h[j])) if np.isfinite(kernel) and kernel > 0 else None))
    frame = pd.DataFrame(records).rename(columns={'return_value': 'return'})
    if len(frame) != len(mask) or frame.duplicated(['market', 'date']).any():
        raise ValueError('Incomplete or duplicate forecasts')
    summary = frame.groupby('market')[['qlike', 'native_nll']].mean().mean().to_dict()
    if save:
        frame.to_csv(save, index=False, float_format='%.17g')
    return summary


def train(seed, smoke=False):
    p = verify()
    if seed not in p['seeds']:
        raise ValueError('Unregistered seed')
    run = HERE / ('smoke' if smoke else 'runs') / str(seed)
    run.mkdir(parents=True, exist_ok=False)
    s = p['training'].copy()
    if smoke:
        s.update(steps=2, eval_every=2)
    A.G.configure(s)
    data = read_data(p)
    train_data = {m: {k: v[raw['date'] <= p['train_end']] for k, v in raw.items() if k != 'kernel'}
                  for m, raw in data.items()}
    dev = pd.read_csv(HERE / 'development_mask.csv', dtype={'date': str})
    if not dev.date.between(p['development_start'], p['development_end']).all():
        raise ValueError('Development selection escaped time gate')
    evaluation = pd.read_csv(HERE / 'evaluation_mask.csv', dtype={'date': str})
    if smoke:
        dev = dev.groupby('market').head(1).head(2)
        evaluation = evaluation.head(32)
    torch.manual_seed(seed)
    model = RegularizedVolPFN(dropout=p['dropout'], **s['cfg']).cuda()
    initial = A.M.state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['lr'], betas=tuple(s['optimizer']['betas']),
                                 weight_decay=s['optimizer']['weight_decay'], eps=s['optimizer']['eps'])
    start = time.time()
    status = dict(status='running', seed=seed, smoke_unpromotable=smoke, protocol_sha256=sha(HERE/'protocol.json'),
                  initial_state_sha256=initial, runtime=A.G.runtime(), best_step=None, completed_updates=0)
    T.save_json(run/'status.json', status)
    best, best_step, stale, chain = np.inf, None, 0, hashlib.sha256(b'historical-followup-v1').hexdigest()
    seen, observed = set(), {m: np.zeros(len(v['returns']), dtype=bool) for m, v in train_data.items()}
    try:
        initial_dev = evaluate(model, dev, data, seed=seed)
        T.save_json(run/'initial_development_diagnostic.json', initial_dev)
        for step, (data_seed, batch_size, length) in enumerate(A.M.schedule(seed, s)):
            batch = A.historical_batch(train_data, data_seed, batch_size, length)
            digest = A.batch_hash(batch)
            chain = hashlib.sha256(bytes.fromhex(chain)+bytes.fromhex(digest)).hexdigest()
            stats = A.update('historical', model, optimizer, batch, step, s)
            for draw in batch['historical_draws']:
                seen.add((draw['market'], draw['start'], length))
                observed[draw['market']][draw['start']:draw['stop']] = True
            event = dict(step=step+1, batch_sha256=digest, batch_chain_sha256=chain,
                         elapsed_seconds=time.time()-start, **stats)
            with (run/'training.jsonl').open('a') as f:
                f.write(json.dumps(event, allow_nan=False)+'\n')
            status['completed_updates'] = step+1
            if (step+1) % s['eval_every'] == 0 or step+1 == s['steps']:
                result = evaluate(model, dev, data, seed=seed)
                event = dict(step=step+1, **result)
                with (run/'development.jsonl').open('a') as f:
                    f.write(json.dumps(event, allow_nan=False)+'\n')
                if result['qlike'] < best:
                    best, best_step, stale = result['qlike'], step+1, 0
                    torch.save(dict(state_dict=model.state_dict(), cfg=s['cfg'], dropout=p['dropout'],
                                    seed=seed, step=best_step, development=result,
                                    protocol_sha256=sha(HERE/'protocol.json'), smoke_unpromotable=smoke), run/'best.pt')
                else:
                    stale += 1
                status.update(best_step=best_step, development_qlike=best, stale_checks=stale,
                              elapsed_seconds=time.time()-start)
                T.save_json(run/'status.json', status)
                print('DEVELOPMENT', seed, event, 'best', best_step, 'stale', stale, flush=True)
                if stale >= p['patience_checks']:
                    break
            elif (step+1) % 100 == 0:
                T.save_json(run/'status.json', status)
                print('TRAIN', seed, step+1, stats, flush=True)
        ck = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        if ck['protocol_sha256'] != sha(HERE/'protocol.json'):
            raise RuntimeError('Selected checkpoint changed protocol')
        model.load_state_dict(ck['state_dict'])
        verify()
        status['status'] = 'evaluating'
        T.save_json(run/'status.json', status)
        result = evaluate(model, evaluation, data, seed=seed, save=run/'forecasts.csv')
        status.update(status='complete', evaluation=result, forecast_rows=len(evaluation),
                      checkpoint_sha256=sha(run/'best.pt'), forecast_sha256=sha(run/'forecasts.csv'),
                      training_unique_windows=len(seen), training_unique_observations=sum(int(v.sum()) for v in observed.values()),
                      input_positions=status['completed_updates']*s['tokens'], batch_chain_sha256=chain,
                      elapsed_seconds=time.time()-start)
        T.save_json(run/'status.json', status)
        print('COMPLETE', json.dumps(status), flush=True)
    except BaseException as e:
        status.update(status='failed', error=repr(e))
        T.save_json(run/'status.json', status)
        (run/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--seed', type=int, required=True)
    cli.add_argument('--smoke', action='store_true')
    args = cli.parse_args()
    train(args.seed, args.smoke)
