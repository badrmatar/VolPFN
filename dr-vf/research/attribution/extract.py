"""Leg C step 1: extract the checkpoint's own log-variance updates on the development index panel.

Runs out_volpfn/volpfn_base_best.pt causally over each of the 28 Oxford-Man development indices with
the repaired public API, records the mixture second moment h_t aligned with the returns, and records
which sliding window produced each forecast so that window-boundary transitions can be excluded: at a
boundary log h changes because the context restarted, not because of the shock, and including those
transitions would corrupt every regression that follows.

Writes research/outputs/attribution_extract_v1/<market>.npz and a manifest.
usage: python3 research/attribution/extract.py [--checkpoint out_volpfn/volpfn_base_best.pt]
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, time
from pathlib import Path
import numpy as np

from research.prior_adaptation.experiment import DATA, GROUPS, load_market, save, sha

L_WINDOW, STRIDE, MIN_HISTORY = 1024, 512, 100


def window_owner(T, L=L_WINDOW, stride=STRIDE, min_history=MIN_HISTORY):
    """Replay predict_series' window construction and return owner[t]: the index of the sliding window
    whose forecast is retained at position t, or -1 where no forecast exists. Mirrors volpfn.predict_series
    exactly, including that later windows overwrite earlier ones."""
    prefix = min(100, stride)
    owner = np.full(T, -1, dtype=np.int64)
    wins = []
    for t in range(min_history, min(prefix, T)):
        wins.append((0, t + 1, t))                       # growing prefix window: start 0, length t+1, first t
    starts = [0] + list(range(stride, max(T - stride, 0), stride))
    for k, s in enumerate(starts):
        n = len(range(s, min(s + L, T)))
        first = max(prefix if k == 0 else stride, min_history - s)
        if n <= first:
            continue
        wins.append((s, n, first))
    for w, (s, n, first) in enumerate(wins):
        owner[s + first: s + n] = w                      # targets s+first .. s+n-1, later windows overwrite
    return owner


def run(args):
    import volpfn as V          # imported here so that window_owner needs no torch (tests, ablation readout)
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint)
    sources = [Path(__file__), Path('volpfn.py'), Path('volpfn_prior.py'),
               Path('research/attribution/PROTOCOL.md'), checkpoint]
    hashes = {str(p): sha(p) for p in sources}
    save(root / 'protocol.json', {
        'scope': 'Attribution study, development data only. Extracts the saved checkpoint own variance '
                 'updates; does not train, tune, select a checkpoint, or open reserved outcomes.',
        'inputs': hashes, 'window': {'L': L_WINDOW, 'stride': STRIDE, 'min_history': MIN_HISTORY}})
    model = V.load_model(str(checkpoint), 'cpu')
    names = sorted(n for g in GROUPS.values() for n in g)
    started = time.perf_counter(); manifest = {}
    for name in names:
        market = load_market(DATA / (name + '.npz'))
        r, date = market['r'], market['date']
        out = V.predict_series(model, r, L=L_WINDOW, stride=STRIDE, min_history=MIN_HISTORY, levels=())
        owner = window_owner(len(r))
        h = out['h']
        finite = np.isfinite(h)
        if np.any(finite != (owner >= 0)):
            raise RuntimeError(f'{name}: window ownership map disagrees with which forecasts exist')
        np.savez_compressed(root / f'{name}.npz', date=date, r=r, h=h, h_lat=out['h_lat'],
                            logscore=out['logscore'], owner=owner)
        manifest[name] = {'observations': int(len(r)), 'forecasts': int(finite.sum()),
                          'first_date': str(date[0]), 'last_date': str(date[-1]),
                          'windows': int(owner.max()) + 1}
        print(f'{name}: {finite.sum()} forecasts over {owner.max() + 1} windows', flush=True)
    save(root / 'manifest.json', {'markets': manifest, 'elapsed_seconds': time.perf_counter() - started})
    if any(sha(Path(p)) != v for p, v in hashes.items()):
        raise RuntimeError('a frozen source changed during the run')
    save(root / 'status.json', {'status': 'complete'})
    print('COMPLETE', len(names), 'markets', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default='out_volpfn/volpfn_base_best.pt')
    ap.add_argument('--out', default='research/outputs/attribution_extract_v1')
    run(ap.parse_args())
