"""Registered 12-slot, conditional inference for the cross-market helper follow-up.

Forecasts are never fitted, pooled into an ensemble, repaired or dropped here.
Every comparison retains the same five seed IDs, and seed-direction checks use
comparator_i minus reference_i, rather than a comparator seed-average. The
calendar bootstrap is conditional on these particular frozen fitted procedures.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from research.cross_market_blend_20260930.run_blends import (
    FAMILY, REFERENCE, ROOT, ROSTER, SEEDS, SOURCE, sha, validate_mask, write_json)
from research.blend_rk.blend import qlike
from research.combined_leaderboard.statistics import _panel, _resample
from research.population_transfer.statistics import _basic_interval, _calendar_counts, holm


RESAMPLES, INFERENTIAL_SEED = 20000, 2026093005
BLOCKS, PROXIES = (60, 20, 120), ('r2', 'rk')
SCOPE = ('Retrospective conditional frozen-forecast inference on an already exposed '
         'panel; excludes online refit, helper/model training, pretraining and '
         'research-selection uncertainty. The new 12-slot family is separate from '
         'the preserved earlier 38-slot family. Seed-ID matching does not establish '
         'paired training randomness for unrelated architectures.')


def matched_seed_gains(comparator, reference, markets, proxy_index):
    """Equal-market loss gains for each corresponding registered seed ID."""
    a, b, markets = np.asarray(comparator, float), np.asarray(reference, float), np.asarray(markets)
    if (a.shape != b.shape or a.ndim != 3 or a.shape != (len(SEEDS), len(markets), len(PROXIES))
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or proxy_index not in range(len(PROXIES)) or not len(markets)):
        raise ValueError('Complete aligned five-seed loss arrays required')
    population = sorted(set(markets))
    return [float(np.mean([delta[markets == market].mean() for market in population]))
            for delta in a[:, :, proxy_index] - b[:, :, proxy_index]]


def validate_design(design):
    expected = {'reference': REFERENCE, 'family_slots': FAMILY,
                'row_count_per_seed': 22026, 'refit_dates': 20, 'burn_dates': 60,
                'inferential_seed': INFERENTIAL_SEED, 'resamples': RESAMPLES}
    if (any(design.get(k) != v for k, v in expected.items())
            or tuple(design.get('roster', ())) != ROSTER
            or tuple(design.get('seeds', ())) != SEEDS
            or tuple(design.get('blocks', ())) != BLOCKS
            or FAMILY != 12 or len(ROSTER) != 7 or len(set(ROSTER)) != 7):
        raise ValueError('Registered 12-slot design drift')


def verify_receipt(source, pin):
    """Require an immutable complete comparison bundle, including all seven rows."""
    receipt = json.loads(pin(source / 'ARTIFACT_RECEIPT.json').read_text())
    required = {'REGISTERED_DESIGN.json', 'common_mask.csv', 'verification.json',
                'provenance.json', 'frozen_sources/PROTOCOL.md'}
    required.update('row_forecasts/' + name + '.csv.gz' for name in ROSTER)
    if not isinstance(receipt, dict) or not required.issubset(receipt):
        raise ValueError('Complete registered comparison artifact receipt required')
    for relative, digest in receipt.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not isinstance(digest, str) or sha(pin(path)) != digest:
            raise ValueError('Comparison artifact hash mismatch: ' + relative)
    verification = json.loads((source / 'verification.json').read_text())
    if (verification.get('status') != 'passed' or verification.get('exact_common_mask') is not True
            or verification.get('all_five_seeds') is not True or verification.get('procedures') != 7
            or verification.get('scored_rows') != 7 * len(SEEDS) * 22026):
        raise ValueError('Complete gated comparison required')
    provenance = json.loads((source / 'provenance.json').read_text())
    if provenance.get('originals_preserved') is not True or not provenance.get('source_hashes'):
        raise ValueError('Preserved source provenance required')
    for path, digest in provenance['source_hashes'].items():
        if sha(pin(path)) != digest:
            raise ValueError('Frozen upstream source changed: ' + path)


def load_losses(source, mask, pin):
    """Canonical rows, exact seeds, positive predictions, and recomputed scores."""
    expected = pd.MultiIndex.from_frame(mask[['market', 'date']])
    required = {'market', 'date', 'model', 'seed', 'r2', 'rk',
                'r2_forecast', 'rk_forecast', 'r2_qlike', 'rk_qlike'}
    losses, seed_losses = {}, {}
    for name in ROSTER:
        path = pin(source / 'row_forecasts' / (name + '.csv.gz'))
        frame = pd.read_csv(path, dtype={'market': str, 'date': str, 'seed': str, 'model': str})
        if (not required.issubset(frame) or frame[list(required)].isna().any().any()
                or set(frame.seed) != set(SEEDS) or set(frame.model) != {name}):
            raise ValueError('Registered model/seed identities and complete fields required: ' + name)
        per_seed = []
        for seed in SEEDS:
            f = frame.loc[frame.seed.eq(seed)].set_index(['market', 'date'])
            if f.index.has_duplicates or len(f) != len(expected) or set(f.index) != set(expected):
                raise ValueError('Incomplete or extra statistical population: ' + name + '|' + seed)
            f = f.reindex(expected)
            recomputed = []
            for proxy in PROXIES:
                np.testing.assert_allclose(f[proxy], mask[proxy], rtol=1e-12, atol=1e-12,
                                           err_msg='Original target/scaling mismatch')
                score = qlike(f[proxy], f[proxy + '_forecast'])
                np.testing.assert_allclose(f[proxy + '_qlike'], score, rtol=1e-12, atol=1e-12,
                                           err_msg='Saved QLIKE mismatch')
                recomputed.append(score)
            per_seed.append(np.column_stack(recomputed))
        values = np.stack(per_seed)
        if not np.isfinite(values).all():
            raise ValueError('Invalid recomputed scored rows')
        seed_losses[name], losses[name] = values, values.mean(axis=0)
    return losses, seed_losses


def run(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins = {}

    def pin(path):
        path = Path(path).resolve()
        digest = sha(path)
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError('Statistical input changed while loading: ' + str(path))
        pins[str(path)] = digest
        return path

    try:
        verify_receipt(source, pin)
        design = json.loads(pin(source / 'REGISTERED_DESIGN.json').read_text())
        validate_design(design)
        if sha(source / 'frozen_sources/PROTOCOL.md') != design.get('protocol_sha256'):
            raise ValueError('Registered frozen protocol hash mismatch')
        mask = pd.read_csv(pin(source / 'common_mask.csv'), dtype={'market': str, 'date': str})
        validate_mask(mask)
        original = pd.read_csv(pin(SOURCE / 'common_mask.csv'), dtype={'market': str, 'date': str})
        pd.testing.assert_frame_equal(mask, original, check_exact=False, rtol=1e-12, atol=1e-12)
        if (mask[['market', 'date']].isna().any().any()
                or not mask.date.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all()):
            raise ValueError('Nonmissing canonical ISO dates/markets required')
        pd.to_datetime(mask.date, format='%Y-%m-%d', errors='raise')
        dates, markets = mask.date.to_numpy(str), mask.market.to_numpy(str)
        calendar, population = sorted(set(dates)), sorted(set(markets))
        losses, seed_losses = load_losses(source, mask, pin)

        dependencies = (Path(__file__), Path(__file__).with_name('test_statistics.py'),
                        Path(__file__).with_name('PROTOCOL.md'), Path(__file__).with_name('run_blends.py'),
                        ROOT / 'research/intraday_extension_20260930/run_blends.py',
                        ROOT / 'research/combined_leaderboard/statistics.py',
                        ROOT / 'research/population_transfer/statistics.py',
                        ROOT / 'research/blend_rk/blend.py')
        frozen = out / 'frozen_sources'
        for path in dependencies:
            path = pin(path)
            destination = frozen / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if sha(destination) != pins[str(path)]:
                raise ValueError('Frozen statistical source hash mismatch')
        for path in (sys.executable, np.__file__, pd.__file__, scipy.__file__):
            pin(path)
        registration = {
            'reference': REFERENCE, 'roster': ROSTER, 'seeds': SEEDS, 'family_slots': FAMILY,
            'resamples': RESAMPLES, 'seed': INFERENTIAL_SEED, 'blocks': BLOCKS,
            'primary_block': 60, 'sensitivity_blocks': [20, 120],
            'rows_per_seed': len(mask), 'markets': len(population), 'calendar_dates': len(calendar),
            'period': [calendar[0], calendar[-1]],
            'direction': 'comparator loss minus reference loss; positive favours reference',
            'weighting': 'dates within each market; equal markets; equal seed losses, never forecast ensembles',
            'p_method': 'two-sided centered circular calendar bootstrap; plus-one; joint Holm over 12 slots',
            'interval_method': 'basic centered bootstrap; ordinary 95% and Bonferroni family 95%',
            'seed_direction_method': 'corresponding seed comparator_i minus reference_i, in registered seed-ID order',
            'scope': SCOPE, 'source_comparison': str(source),
        }
        write_json(out / 'STATISTICS_DESIGN.json', registration)
        results, archive = [], {}
        for block in BLOCKS:
            counts = _calendar_counts(len(calendar), RESAMPLES, block, INFERENTIAL_SEED)
            block_rows = []
            for name in ROSTER:
                if name == REFERENCE:
                    continue
                for j, proxy in enumerate(PROXIES):
                    gains = losses[name][:, j] - losses[REFERENCE][:, j]
                    estimate, draws, per_market = _resample(
                        _panel(dates, markets, gains, calendar, population), counts)
                    if (draws.shape != (RESAMPLES,) or not np.isfinite(draws).all()
                            or not np.isfinite(estimate) or not np.isfinite(per_market).all()):
                        raise ValueError('Complete finite registered bootstrap draws required')
                    p = float((1 + np.count_nonzero(np.abs(draws - estimate) >= abs(estimate)))
                              / (RESAMPLES + 1))
                    seed_gains = matched_seed_gains(seed_losses[name], seed_losses[REFERENCE], markets, j)
                    np.testing.assert_allclose(np.mean(seed_gains), estimate, rtol=1e-12, atol=1e-12)
                    row = {'block': block, 'comparator': name, 'reference': REFERENCE,
                           'proxy': proxy, 'family_slots': FAMILY, 'status': 'complete',
                           'p_raw': p, 'gain': float(estimate),
                           'interval95': _basic_interval(estimate, draws, .95),
                           'family_interval95': _basic_interval(estimate, draws, 1 - .05 / FAMILY),
                           'matched_seed_ids': SEEDS, 'matched_seed_gains': seed_gains,
                           'matched_seeds_positive': int(np.sum(np.array(seed_gains) > 0)),
                           'matched_seeds_negative': int(np.sum(np.array(seed_gains) < 0)),
                           'markets_improved': int(np.sum(per_market > 0)),
                           'per_market_gains': dict(zip(population, map(float, per_market))),
                           'positive_support': False, 'negative_support': False}
                    block_rows.append(row)
                    archive[f'{block}|{proxy}|{name}'] = draws
            if len(block_rows) != FAMILY:
                raise ValueError('Missing registered inference slots')
            for row, p_adjusted in zip(block_rows, holm([r['p_raw'] for r in block_rows])):
                row['p_holm'] = float(p_adjusted)
            for name in ROSTER:
                if name == REFERENCE:
                    continue
                both = [r for r in block_rows if r['comparator'] == name]
                joint = (len(both) == len(PROXIES) and all(
                    r['gain'] > 0 and r['p_holm'] < .05 and r['family_interval95'][0] > 0
                    and r['matched_seeds_positive'] == len(SEEDS) for r in both))
                negative = (len(both) == len(PROXIES) and all(
                    r['gain'] < 0 and r['p_holm'] < .05 and r['family_interval95'][1] < 0
                    and r['matched_seeds_negative'] == len(SEEDS) for r in both))
                for row in both:
                    row['positive_support'], row['negative_support'] = joint, negative
            results.extend(block_rows)
        write_json(out / 'statistics.json', {**registration, 'contrasts': results})
        pd.DataFrame(results).to_csv(out / 'contrasts.csv', index=False)
        np.savez_compressed(out / 'bootstrap_archive_no_pickle.npz', **archive)
        for path, digest in pins.items():
            if sha(path) != digest:
                raise ValueError('Changed statistical input: ' + path)
        write_json(out / 'provenance.json', {
            'source_hashes': pins, 'originals_preserved': True,
            'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__})
        write_json(out / 'verification.json', {
            'status': 'passed', 'exact_common_mask': True, 'all_five_matched_seeds': True,
            'family_slots': FAMILY, 'contrasts': len(results), 'archive_arrays': len(archive),
            'hashes_unchanged': True, 'old_38_slot_family_unchanged': True})
        write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
                   for p in sorted(out.rglob('*')) if p.is_file()})
        print('Completed', len(results), 'contrasts; primary support:',
              sorted({r['comparator'] for r in results if r['block'] == 60 and r['positive_support']}))
    except Exception as exc:
        write_json(out / 'FAILED_ATTEMPT.json', {
            'status': 'failed', 'exception': type(exc).__name__, 'reason': str(exc),
            'source_comparison': str(source), 'observed_source_hashes': pins,
            'originals_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.source, args.out)
