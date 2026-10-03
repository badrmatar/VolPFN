"""Conditional frozen-forecast uncertainty with every registered slot retained."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.intraday_extension_20260930.run_blends import (
    REFERENCE, ROSTER, SOURCE, expected_seeds, validate_mask, sha, write_json)
from experiments.blend_rk.blend import qlike
from experiments.combined_leaderboard.statistics import _panel, _resample
from experiments.population_transfer.statistics import _calendar_counts, _basic_interval, holm


def run(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    design = json.loads((source / 'REGISTERED_DESIGN.json').read_text())
    if tuple(design['roster']) != ROSTER or design['reference'] != REFERENCE or design['family_slots'] != 38:
        raise ValueError('Registered design drift')
    pins = {str(source / 'REGISTERED_DESIGN.json'): sha(source / 'REGISTERED_DESIGN.json'),
            str(Path(__file__).resolve()): sha(__file__)}
    root = Path(__file__).resolve().parents[2]
    for path in (Path(__file__).with_name('run_blends.py'),
                 root / 'experiments/combined_leaderboard/statistics.py',
                 root / 'experiments/population_transfer/statistics.py',
                 root / 'experiments/blend_rk/blend.py', SOURCE / 'common_mask.csv'):
        pins[str(path)] = sha(path)
    mask = pd.read_csv(source / 'common_mask.csv', dtype={'market': str, 'date': str})
    validate_mask(mask)
    original = pd.read_csv(SOURCE / 'common_mask.csv', dtype={'market': str, 'date': str})
    pd.testing.assert_frame_equal(mask, original, check_exact=False, rtol=1e-12, atol=1e-12)
    pins[str(source / 'common_mask.csv')] = sha(source / 'common_mask.csv')
    dates, markets = mask.date.to_numpy(str), mask.market.to_numpy(str)
    calendar, population = sorted(set(dates)), sorted(set(markets))
    expected = pd.MultiIndex.from_frame(mask[['market', 'date']])
    losses, seed_losses = {}, {}
    for name in ROSTER:
        path = source / 'row_forecasts' / (name + '.csv.gz')
        if not path.exists():
            continue
        pins[str(path)] = sha(path)
        frame = pd.read_csv(path, dtype={'market': str, 'date': str, 'seed': str})
        if set(frame.seed) != set(expected_seeds(name)) or set(frame.model) != {name}:
            raise ValueError('Registered model/seed identities required: ' + name)
        per_seed = {}
        for seed, f in frame.groupby('seed', sort=True):
            f = f.set_index(['market', 'date'])
            if f.index.has_duplicates or len(f) != len(expected) or set(f.index) != set(expected):
                raise ValueError('Incomplete statistical population')
            f = f.reindex(expected)
            for proxy in ('r2', 'rk'):
                np.testing.assert_allclose(f[proxy], mask[proxy], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(f[proxy + '_qlike'],
                                           qlike(f[proxy], f[proxy + '_forecast']),
                                           rtol=1e-12, atol=1e-12)
            per_seed[seed] = f[['r2_qlike', 'rk_qlike']].to_numpy(float)
        values = np.stack(list(per_seed.values()))
        if not np.isfinite(values).all():
            raise ValueError('Invalid scored rows')
        seed_losses[name], losses[name] = values, values.mean(axis=0)
    if REFERENCE not in losses or seed_losses[REFERENCE].shape[0] != 5:
        raise ValueError('Complete five-seed reference required')
    results, archive = [], {}
    for block in (60, 20, 120):
        counts = _calendar_counts(len(calendar), 20000, block, 2026093004)
        block_rows = []
        for name in ROSTER:
            if name == REFERENCE:
                continue
            for j, proxy in enumerate(('r2', 'rk')):
                row = {'block': block, 'comparator': name, 'reference': REFERENCE,
                       'proxy': proxy, 'family_slots': 38}
                if name not in losses:
                    row.update(status='unverified/unavailable', p_raw=1., gain=None,
                               interval95=None, family_interval95=None,
                               reference_seeds_positive=None, positive_support=False,
                               negative_support=False)
                else:
                    gains = losses[name][:, j] - losses[REFERENCE][:, j]
                    estimate, draws, per_market = _resample(_panel(dates, markets, gains, calendar, population), counts)
                    p = float((1 + np.count_nonzero(np.abs(draws - estimate) >= abs(estimate))) / 20001)
                    seed_gains = []
                    for values in seed_losses[REFERENCE]:
                        delta = losses[name][:, j] - values[:, j]
                        seed_gains.append(float(np.mean([delta[markets == m].mean() for m in population])))
                    row.update(status='complete', p_raw=p, gain=float(estimate),
                               interval95=_basic_interval(estimate, draws, .95),
                               family_interval95=_basic_interval(estimate, draws, 1 - .05 / 38),
                               reference_seed_gains=seed_gains,
                               reference_seeds_positive=int(np.sum(np.array(seed_gains) > 0)),
                               reference_seeds_negative=int(np.sum(np.array(seed_gains) < 0)),
                               markets_improved=int(np.sum(per_market > 0)),
                               positive_support=False, negative_support=False)
                    archive[f'{block}|{proxy}|{name}'] = draws
                block_rows.append(row)
        adjusted = holm([r['p_raw'] for r in block_rows])
        for row, value in zip(block_rows, adjusted):
            row['p_holm'] = float(value)
        for name in ROSTER:
            if name == REFERENCE:
                continue
            both = [r for r in block_rows if r['comparator'] == name]
            joint = all(r['status'] == 'complete' and r['gain'] > 0 and r['p_holm'] < .05 and
                        r['family_interval95'][0] > 0 and r['reference_seeds_positive'] == 5 for r in both)
            negative = all(r['status'] == 'complete' and r['gain'] < 0 and r['p_holm'] < .05 and
                           r['family_interval95'][1] < 0 and r['reference_seeds_negative'] == 5 for r in both)
            for row in both:
                row['positive_support'] = joint
                row['negative_support'] = negative
        results.extend(block_rows)
    write_json(out / 'statistics.json', {
        'reference': REFERENCE, 'roster': ROSTER, 'family_slots': 38,
        'resamples': 20000, 'seed': 2026093004, 'blocks': [60, 20, 120],
        'direction': 'comparator loss minus reference loss; positive favours reference',
        'p_method': 'two-sided centered calendar bootstrap, plus-one and joint Holm',
        'primary_block': 60, 'sensitivity_blocks': [20, 120],
        'scope': 'retrospective conditional frozen-forecast inference; excludes online refit, training, pretraining and selection uncertainty',
        'contrasts': results})
    pd.DataFrame(results).to_csv(out / 'contrasts.csv', index=False)
    np.savez_compressed(out / 'bootstrap_archive_no_pickle.npz', **archive)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed statistical input: ' + path)
    write_json(out / 'provenance.json', {'source_hashes': pins, 'originals_preserved': True,
               'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__})
    write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
               for p in sorted(out.rglob('*')) if p.is_file()})
    print('Completed', len(results), 'contrasts; main support:',
          sorted({r['comparator'] for r in results if r['block'] == 60 and r['positive_support']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.source, args.out)
