"""Strict paired historical confirmation; no row/seed/model survivor selection.

Loss gains are comparator QLIKE minus reference QLIKE. Positive favors the
reference. Empirical exposure/selection is not corrected by this bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiment_programme_20260930 import programme as P
from research.matched_prior_confirmatory.statistics import holm


def validate_forecasts(frame, mask, seeds):
    required = {'market', 'date', 'seed', 'return', 'second_moment', 'qlike'}
    if not required.issubset(frame):
        raise ValueError('Missing forecast columns: ' + str(sorted(required - set(frame))))
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing labels/primary forecasts')
    frame = frame.copy()
    seed_values = pd.to_numeric(frame.seed, errors='raise').to_numpy(float)
    if not np.isfinite(seed_values).all() or not np.equal(seed_values, np.floor(seed_values)).all():
        raise ValueError('Seed identifiers must be finite integers')
    frame['seed'] = seed_values.astype(int)
    frame['date'] = pd.to_datetime(frame.date, format='%Y-%m-%d', errors='raise').dt.strftime('%Y-%m-%d')
    if set(frame.seed) != set(seeds) or frame.duplicated(['seed', 'market', 'date']).any():
        raise ValueError('Unexpected seeds or duplicate rows')
    expected = mask[['market', 'date']].copy()
    expected['date'] = expected.date.astype(str)
    expected_index = pd.MultiIndex.from_frame(expected)
    for seed in seeds:
        available = pd.MultiIndex.from_frame(frame.loc[frame.seed == seed, ['market', 'date']])
        if len(available) != len(expected_index) or set(available) != set(expected_index):
            raise ValueError('Coverage differs from frozen mask for seed ' + str(seed))
    numeric = frame[['return', 'second_moment', 'qlike']].to_numpy(float)
    if not np.isfinite(numeric).all() or (frame.second_moment <= 0).any():
        raise ValueError('Nonfinite forecast or nonpositive second moment')
    computed = frame['return'].to_numpy(float) ** 2 / frame.second_moment.to_numpy(float) + np.log(frame.second_moment.to_numpy(float))
    if not np.allclose(computed, frame.qlike.to_numpy(float), rtol=2e-10, atol=2e-10):
        raise ValueError('QLIKE is inconsistent with issued moment and observed return')
    if 'return' in mask:
        joined = frame.merge(mask[['market', 'date', 'return']], on=['market', 'date'], suffixes=('', '_locked'), validate='many_to_one')
        if not np.allclose(joined['return'], joined['return_locked'], rtol=1e-12, atol=1e-12):
            raise ValueError('Return target differs from the frozen data mask')
    return frame.sort_values(['seed', 'market', 'date']).reset_index(drop=True)


def contrast(reference, comparator, mask, seeds):
    left = validate_forecasts(reference, mask, seeds)
    right = validate_forecasts(comparator, mask, seeds)
    labels = ['seed', 'market', 'date']
    joined = left[labels + ['return', 'qlike']].merge(right[labels + ['return', 'qlike']],
        on=labels, suffixes=('_reference', '_comparator'), validate='one_to_one')
    if not np.allclose(joined.return_reference, joined.return_comparator, rtol=0, atol=0):
        raise ValueError('Paired target values are not identical')
    joined['gain'] = joined.qlike_comparator - joined.qlike_reference
    return joined[labels + ['gain']]


def calendar_counts(n_dates, repetitions, block, rng):
    if min(n_dates, repetitions, block) < 1:
        raise ValueError('Positive calendar/resample/block sizes required')
    starts = rng.integers(0, n_dates, size=(repetitions, int(np.ceil(n_dates / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % n_dates).reshape(repetitions, -1)[:, :n_dates]
    counts = np.zeros((repetitions, n_dates), dtype=np.float64)
    for row, index in enumerate(indices):
        counts[row] = np.bincount(index, minlength=n_dates)
    return counts


def _summary(estimate, draws):
    errors = draws - estimate
    quantiles = np.quantile(errors, [.025, .975])
    tolerance = 64 * np.finfo(float).eps * max(1., abs(estimate))
    probability = (1 + np.count_nonzero(np.abs(errors) >= abs(estimate) - tolerance)) / (len(draws) + 1)
    return {'mean_gain': float(estimate), 'interval95': [float(estimate - quantiles[1]), float(estimate - quantiles[0])],
            'p_two_sided': float(probability), 'positive_favors': 'reference',
            'interval_method': 'Basic centered circular-calendar bootstrap',
            'probability_method': 'Two-sided centered plus-one bootstrap probability'}


def bootstrap(gains, mask, seeds, repetitions=20000, block=60, rng_seed=2026093001, chunk_size=250):
    calendar = sorted(mask.date.astype(str).unique())
    markets = sorted(mask.market.unique())
    columns = pd.MultiIndex.from_product([seeds, markets], names=['seed', 'market'])
    wide = gains.pivot(index='date', columns=['seed', 'market'], values='gain').reindex(index=calendar, columns=columns)
    values = wide.to_numpy(float)
    present = np.isfinite(values).astype(float)
    estimate_by_seed_market = np.nanmean(values, axis=0).reshape(len(seeds), len(markets))
    estimate = float(estimate_by_seed_market.mean())
    rng = np.random.default_rng(rng_seed)
    seed_rng = np.random.default_rng(rng_seed + 1)
    fixed_draws, crossed_draws = [], []
    for start in range(0, repetitions, chunk_size):
        n = min(chunk_size, repetitions - start)
        counts = calendar_counts(len(calendar), n, block, rng)
        denominator = counts @ present
        if (denominator <= 0).any():
            raise ValueError('A calendar draw omitted a market; no silent redraw/drop')
        means = ((counts @ np.nan_to_num(values, nan=0.)) / denominator).reshape(n, len(seeds), len(markets))
        seed_means = means.mean(axis=2)
        fixed_draws.extend(seed_means.mean(axis=1))
        selection = seed_rng.integers(0, len(seeds), size=(n, len(seeds)))
        crossed_draws.extend(np.take_along_axis(seed_means, selection, axis=1).mean(axis=1))
    result = _summary(estimate, np.asarray(fixed_draws))
    result.update({'status': 'available', 'resamples': repetitions, 'calendar_block': block,
        'seed': rng_seed, 'seeds': list(seeds), 'rows_per_seed': len(mask),
        'per_seed_mean_gain': dict(zip(map(str, seeds), map(float, estimate_by_seed_market.mean(axis=1)))),
        'per_market_mean_gain': dict(zip(markets, map(float, estimate_by_seed_market.mean(axis=0)))),
        'scope': 'Calendar uncertainty conditional on these fixed models, markets, masks and exposed study. '
            'Does not account for choosing the study after seeing historical results.',
        'crossed_seed_calendar_exploratory': dict(_summary(estimate, np.asarray(crossed_draws)),
            scope='Approximate resampling of only five seed pairs; exploratory, not the primary p-value')})
    return result


def run(manifest_path, forecast_paths, out, repetitions=None):
    manifest = P.verify(manifest_path)
    base = Path(manifest_path)
    if base.is_dir():
        base = base / 'manifest.json'
    mask = pd.read_csv(base.parent / manifest['mask_files']['evaluation']['path'], dtype={'date': str, 'market': str}, float_precision='round_trip')
    tables = {key: pd.read_csv(path, dtype={'date': str, 'market': str}, float_precision='round_trip') for key, path in forecast_paths.items()}
    mapping = {'full_vs_noaux': ('full', 'noaux'), 'noaux_vs_historical': ('noaux', 'historical'),
        'noaux_vs_direct': ('noaux', 'direct'), 'full_vs_arch_lstm': ('full', 'arch_lstm'),
        'full_vs_garch_lstm': ('full', 'garch_lstm'), 'full_vs_ginn': ('full', 'ginn'),
        'full_vs_rech': ('full', 'rech')}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    count = manifest['uncertainty']['resamples'] if repetitions is None else repetitions
    if count != manifest['uncertainty']['resamples']:
        raise ValueError('Final inference must use the frozen resample count')
    for family, slots in manifest['statistical_families'].items():
        family_results = {}
        for slot in slots:
            reference, comparator = mapping[slot]
            if reference not in tables or comparator not in tables:
                family_results[slot] = {'status': 'unavailable', 'p_two_sided': 1.,
                    'reason': 'At least one required complete forecast table is unavailable; no superiority claim'}
                continue
            try:
                gains = contrast(tables[reference], tables[comparator], mask, manifest['seeds'])
                gains.to_csv(out / (slot + '_paired_gains.csv'), index=False, float_format='%.17g')
                result = bootstrap(gains, mask, manifest['seeds'], count,
                    manifest['uncertainty']['calendar_block'], manifest['uncertainty']['seed'])
                result['block_sensitivities'] = {str(block): bootstrap(gains, mask, manifest['seeds'], count, block,
                    manifest['uncertainty']['seed']) for block in manifest['uncertainty']['sensitivity_blocks']}
                family_results[slot] = result
            except (ValueError, KeyError) as error:
                # Invalid coverage does not become a reduced-population estimate.
                family_results[slot] = {'status': 'failed_coverage_or_metric_gate', 'p_two_sided': 1., 'reason': str(error)}
        corrected = holm([family_results[slot]['p_two_sided'] for slot in slots])
        for slot, probability in zip(slots, corrected):
            family_results[slot]['p_holm'] = float(probability)
            family_results[slot]['family'] = family
        results[family] = family_results
    result = {'manifest_sha256': P.sha(base), 'status': 'analysis_available_with_explicit_unavailable_slots',
        'families': results, 'forecast_hashes': {key: P.sha(path) for key, path in forecast_paths.items()},
        'interpretation': 'No joint familywise guarantee across component and external families or secondary metrics. '
            'A positive lower95% interval and corrected two-sided p<=.05 support a reference advantage '
            'conditional on this exposed historical design; they do not establish universal superiority.'}
    P.write_json(out / 'statistics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--forecasts', required=True, help='JSON mapping arm/model to complete CSV path')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run(args.manifest, json.loads(Path(args.forecasts).read_text()), args.out)
    print(json.dumps({'status': result['status'], 'out': args.out}))


if __name__ == '__main__':
    main()
