"""Rescore preserved forecasts on a target-only mask; never train or calibrate."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.combined_leaderboard.statistics import qlike, _panel, _resample
from experiments.population_transfer.statistics import _calendar_counts, _basic_interval, holm

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'experiments/outputs/frontier_extension_20260926_v1'
NEW = ROOT / 'experiments/outputs/experiment_programme_20260930'
SEEDS = [8101, 8102, 8103, 8104, 8105]
REFERENCE = 'volpfn_20260930_full'
DAILY_OLD = ['volpfn', 'hist250', 'ewma', 'garch_t', 'gjr_garch_t', 'egarch_t',
             'aparch_t', 'univol', 'chronos2', 'moirai2', 'tabpfn_ts']
REALIZED = ['har', 'loghar', 'ttm_rk', 'ms_har', 'log_rhar']
OLD_DIAGNOSTICS = ['compact9', 'loghar_meanlog', 'ttm_sqrt_rk']
NEW_MAIN = ['full', 'arch_lstm', 'garch_lstm', 'ginn']
NEW_DIAGNOSTICS = ['noaux', 'direct', 'historical']
NEW_IDS = {'full': REFERENCE, 'noaux': 'volpfn_20260930_noaux',
           'historical': 'volpfn_20260930_historical', 'direct': 'volpfn_20260930_direct',
           'arch_lstm': 'arch_lstm', 'garch_lstm': 'garch_lstm', 'ginn': 'ginn'}
LABELS = {'volpfn_20260926': 'VolPFN (Sep 26)', REFERENCE: 'VolPFN (Sep 30 full)',
          'volpfn_20260930_noaux': 'VolPFN without auxiliary head',
          'volpfn_20260930_historical': 'VolPFN historical-training arm',
          'volpfn_20260930_direct': 'VolPFN direct-second-moment arm',
          'arch_lstm': 'ARCH-LSTM', 'garch_lstm': 'GARCH-LSTM', 'ginn': 'GINN',
          'hist250': 'Historical variance (250)', 'ewma': 'EWMA', 'garch_t': 'GARCH-t',
          'gjr_garch_t': 'GJR-GARCH-t', 'egarch_t': 'EGARCH-t', 'aparch_t': 'APARCH-t',
          'univol': 'UniVol', 'chronos2': 'Chronos-2', 'moirai2': 'Moirai-2',
          'tabpfn_ts': 'TabPFN-TS', 'har': 'HAR', 'loghar': 'Log-HAR',
          'ttm_rk': 'TTM-RK', 'ms_har': 'MS-HAR', 'log_rhar': 'Log-Rough-HAR',
          'compact9': 'Compact nine-parameter rule', 'loghar_meanlog': 'Mean-log HAR diagnostic',
          'ttm_sqrt_rk': 'TTM square-root RK diagnostic'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def pin(path, hashes, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Source hash changed: ' + str(path))
    if str(path) in hashes and hashes[str(path)] != digest:
        raise ValueError('Source changed during loading: ' + str(path))
    hashes[str(path)] = digest
    return digest


def align(frame, mask, seeds):
    """Require exact key coverage, target identity and all declared seeds."""
    f = frame.copy()
    if f[['market', 'date', 'seed']].isna().any().any():
        raise ValueError('Missing identifiers')
    if set(f.seed) != set(seeds) or f.duplicated(['seed', 'market', 'date']).any():
        raise ValueError('Wrong seed set or duplicate forecast keys')
    key = pd.MultiIndex.from_frame(mask[['market', 'date']])
    ordered = []
    for seed in seeds:
        rows = f.loc[f.seed == seed].set_index(['market', 'date'])
        if len(rows) != len(key) or set(rows.index) != set(key):
            raise ValueError('Missing or extra scoring keys: ' + str(seed))
        rows = rows.reindex(key).reset_index()
        np.testing.assert_allclose(rows.r2, mask.r2, rtol=1e-12, atol=1e-12,
                                   err_msg='Squared-return units/targets differ')
        for index_field in ['index', 'row_index']:
            if index_field in rows:
                np.testing.assert_array_equal(rows[index_field], mask['index'],
                                              err_msg='Saved target index differs')
        if 'rk' in rows:
            np.testing.assert_allclose(rows.rk, mask.rk, rtol=1e-12, atol=1e-12,
                                       err_msg='Kernel units/targets differ')
        rows['rk'] = mask.rk.to_numpy(float)
        rows['seed'] = seed
        ordered.append(rows)
    return pd.concat(ordered, ignore_index=True)


def score(frame, *, direct=False):
    """Two explicit functionals; never convert M2 into variance by assumption."""
    f = frame.copy()
    f['r2_qlike'] = qlike(f.r2, f.second_moment)
    if direct:
        if f[['mean', 'variance']].notna().any().any():
            raise ValueError('Direct arm must not invent native moments')
        f['rk_qlike'] = np.nan
        f['rk_m2_diagnostic_qlike'] = qlike(f.rk, f.second_moment)
    else:
        moments = f[['mean', 'variance', 'second_moment']].to_numpy(float)
        if not np.isfinite(moments).all() or (f.variance <= 0).any():
            raise ValueError('Invalid native moments; no clipping or dropping')
        np.testing.assert_allclose(f.second_moment, f.variance + f['mean'] ** 2,
                                   rtol=1e-7, atol=1e-10, err_msg='M2 != variance + mean²')
        f['rk_qlike'] = qlike(f.rk, f.variance)
        f['rk_m2_diagnostic_qlike'] = qlike(f.rk, f.second_moment)
    if 'original_r2_qlike' in f:
        np.testing.assert_allclose(f.r2_qlike, f.original_r2_qlike, rtol=2e-10, atol=2e-10)
    if 'original_kernel_m2_qlike' in f:
        np.testing.assert_allclose(f.rk_m2_diagnostic_qlike, f.original_kernel_m2_qlike,
                                   rtol=2e-10, atol=2e-10)
    return f


def summarize(frame):
    metrics = ['r2_qlike', 'rk_qlike', 'rk_m2_diagnostic_qlike']
    seed_market = frame.groupby(['seed', 'market'], sort=True)[metrics].mean().reset_index()
    seed = seed_market.groupby('seed')[metrics].mean().reset_index()
    market = seed_market.groupby('market')[metrics].mean().reset_index()
    overall = {m: None if seed[m].isna().all() else float(seed[m].mean()) for m in metrics}
    return overall, seed_market, seed, market


def target_mask(hashes, bundle, manifest):
    rows = [r for r in bundle['rows'] if r['phase'] == 'evaluation' and r['primary_eligible']]
    mask = pd.DataFrame(rows).sort_values(['market', 'date']).reset_index(drop=True)
    if len(mask) != 22026 or mask.market.nunique() != 28 or mask.date.nunique() != 827:
        raise ValueError('Original population changed')
    if mask.duplicated(['market', 'date']).any() or not np.isfinite(mask.rk).all() or (mask.rk <= 0).any():
        raise ValueError('Invalid original target-only population')
    for market in sorted(mask.market.unique()):
        target = mask.loc[mask.market == market]
        old_path = ROOT / bundle['market_paths'][market]
        raw_path = ROOT / manifest['data']['directory'] / (market + '.npz')
        pin(old_path, hashes, bundle['inputs'][str(old_path.relative_to(ROOT))])
        pin(raw_path, hashes, manifest['data_hashes'][str(raw_path.relative_to(ROOT))])
        with np.load(old_path, allow_pickle=False) as old, np.load(raw_path, allow_pickle=False) as raw:
            dates_old = old['date'].astype('U10')
            dates_raw = raw['date'].astype('U10')
            i = np.searchsorted(dates_old, target.date)
            j = np.searchsorted(dates_raw, target.date)
            np.testing.assert_array_equal(dates_old[i], target.date)
            np.testing.assert_array_equal(dates_raw[j], target.date)
            r = raw['returns'][j].astype(float) * 100.
            rk = raw['rk_parzen'][j].astype(float) * 10000.
            np.testing.assert_allclose(old['r'][i], r, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(old['rk'][i], rk, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(target.r2, r*r, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(target.rk, rk, rtol=1e-12, atol=1e-12)
    return mask


def load_old(model, mask, freeze, hashes):
    spec = freeze['model_specs'][model]
    seeds = ['none' if s is None else str(s) for s in spec['seeds']]
    rows = []
    keys = set(zip(mask.market, mask.date))
    for seed in seeds:
        for market in sorted(mask.market.unique()):
            path = OLD / 'cells' / model / seed / market / 'forecast.json'
            pin(path, hashes)
            cell = read_json(path)
            if cell['freeze_sha256'] != hashes[str((OLD / 'FREEZE.json').resolve())]:
                raise ValueError('Old forecast has wrong frozen source identity')
            for row in cell['rows']:
                if row['model'] != model or ('none' if row['seed'] is None else str(row['seed'])) != seed:
                    raise ValueError('Old model/seed differs from cell identity')
                if row['phase'] != 'evaluation' or (row['market'], row['date']) not in keys:
                    continue
                if row.get('forecast_failed', False):
                    raise ValueError('Failed issued forecast retained')
                row = dict(row)
                row['seed'] = seed
                rows.append(row)
    frame = align(pd.DataFrame(rows), mask, seeds)
    if spec.get('zero_mean') and ((frame['mean'] != 0).any() or (frame.variance != frame.second_moment).any()):
        raise ValueError('Declared zero-mean adapter differs')
    return frame, spec


def load_new(model, mask, provenance, manifest, hashes):
    entry = provenance['models'][model]
    path = Path(entry['forecast_path'])
    pin(path, hashes, entry['forecast_sha256'])
    f = pd.read_csv(path, dtype={'market': str, 'date': str}, float_precision='round_trip')
    if set(f.model) != {model} or set(f.seed) != set(SEEDS):
        raise ValueError('New forecast model/seed labels changed')
    if f.duplicated(['seed', 'market', 'date']).any():
        raise ValueError('Duplicate original rows')
    # Verify the full original mask before cutting; a cutoff must not hide failed coverage.
    mask_path = NEW / 'programme_design_v1' / manifest['mask_files']['evaluation']['path']
    pin(mask_path, hashes, manifest['mask_files']['evaluation']['sha256'])
    full_mask = pd.read_csv(mask_path, dtype={'market': str, 'date': str}, float_precision='round_trip')
    original_keys = set(zip(full_mask.market, full_mask.date))
    if len(full_mask) != 50073:
        raise ValueError('Original overnight mask changed')
    for seed in SEEDS:
        selected = f.loc[f.seed == seed]
        if len(selected) != len(full_mask) or set(zip(selected.market, selected.date)) != original_keys:
            raise ValueError('Original overnight full coverage differs')
    f = f.merge(full_mask[['market', 'date', 'return']].rename(columns={'return': 'locked_return'}),
                on=['market', 'date'], how='left', validate='many_to_one')
    np.testing.assert_allclose(f['return'], f.locked_return, rtol=1e-12, atol=1e-12)
    if not (f.context_end < f.date).all() or not (f.context_start <= f.context_end).all():
        raise ValueError('Saved context sees target/future date')
    if 'raw_r2' in f:
        np.testing.assert_allclose(f.raw_r2, f['return'] ** 2, rtol=1e-12, atol=1e-12)
    if 'raw_return' in f:
        # "raw" here means unstandardized percentage return, not decimal return.
        np.testing.assert_allclose(f.raw_return, f['return'], rtol=1e-12, atol=1e-12)
    f['r2'] = f['return'] ** 2
    f['original_r2_qlike'] = f.qlike
    if 'kernel' in f:
        f['rk'] = f.kernel
        f['original_kernel_m2_qlike'] = f.kernel_qlike
    f['seed'] = f.seed.astype(str)
    key = pd.MultiIndex.from_frame(mask[['market', 'date']])
    selected = pd.MultiIndex.from_frame(f[['market', 'date']]).isin(key)
    return align(f.loc[selected].copy(), mask, [str(s) for s in SEEDS])


def inference(loss_arrays, mask, main_models, resamples, blocks):
    reference = loss_arrays[REFERENCE]
    slots = 2 * (len(main_models) - 1)
    calendar = sorted(mask.date.unique())
    markets = sorted(mask.market.unique())
    dates, row_markets = mask.date.to_numpy(), mask.market.to_numpy()
    templates = []
    archive = {'row_dates': dates, 'row_markets': row_markets, 'calendar': np.array(calendar)}
    for model in main_models:
        if model == REFERENCE:
            continue
        for metric in ['r2_qlike', 'rk_qlike']:
            control = loss_arrays[model][metric].mean(axis=0)
            gains = control - reference[metric].mean(axis=0)
            matrix = _panel(dates, row_markets, gains, calendar, markets)
            gain = float(np.nanmean(matrix, axis=0).mean())
            seed_gains = [float(np.mean([np.mean((control - ref)[row_markets == m]) for m in markets]))
                          for ref in reference[metric]]
            templates.append((model, metric, matrix, gain, seed_gains))
            archive[f'gains__{model}__{metric}'] = gains
    results = []
    for block in blocks:
        print('bootstrap block', block, flush=True)
        counts = _calendar_counts(len(calendar), resamples, block, 2026093003)
        archive[f'counts__block{block}'] = counts.astype(np.int16)
        current = []
        for model, metric, matrix, gain, seed_gains in templates:
            estimate, draws, _ = _resample(matrix, counts)
            np.testing.assert_allclose(estimate, gain, rtol=1e-12, atol=1e-12)
            errors = draws - estimate
            tolerance = 64 * np.finfo(float).eps * max(1., abs(gain))
            p = (1 + np.count_nonzero(np.abs(errors) >= abs(gain) - tolerance)) / (resamples + 1)
            row = {'model': model, 'metric': metric, 'block': block, 'gain': gain,
                   'p_two_sided': float(p), 'interval95': _basic_interval(gain, draws, .95),
                   'family_interval95': _basic_interval(gain, draws, 1 - .05 / slots),
                   'reference_seed_gains': dict(zip(map(str, SEEDS), seed_gains)),
                   'all_reference_seeds_positive': bool(min(seed_gains) > 0)}
            current.append(row)
            archive[f'draws__{model}__{metric}__block{block}'] = draws
        for row, p in zip(current, holm([r['p_two_sided'] for r in current])):
            row['p_holm'] = float(p)
            row['positive_supported'] = bool(row['gain'] > 0 and p < .05 and
                row['family_interval95'][0] > 0 and row['all_reference_seeds_positive'])
            row['negative_supported'] = bool(row['gain'] < 0 and p < .05 and row['family_interval95'][1] < 0)
        for row in current:
            row['both_metrics_positive_supported'] = all(r['positive_supported'] for r in current if r['model'] == row['model'])
        results.extend(current)
    return {'reference': REFERENCE, 'family_slots': slots, 'resamples': resamples,
            'blocks': list(blocks), 'main_block': blocks[0], 'seed': 2026093003,
            'conditional_retrospective': True, 'contrasts': results}, archive


def markdown_table(table):
    lines = ['| Model | r² QLIKE | RK QLIKE (variance) | Seeds |',
             '|---|---:|---:|---:|']
    for row in table.itertuples():
        rk = 'Unavailable' if pd.isna(row.rk_qlike) else f'{row.rk_qlike:.6f}'
        lines.append(f'| {row.label} | {row.r2_qlike:.6f} | {rk} | {row.seeds} |')
    return '\n'.join(lines)


def run(out, resamples=20000, blocks=(60, 20, 120)):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Refuse to overwrite preserved output: ' + str(out))
    hashes = {}
    for path in [OLD / 'bundle.json', OLD / 'FREEZE.json', OLD / 'analysis/results.json',
                 OLD / 'analysis/leaderboard.csv', NEW / 'FINAL_REPORT.md',
                 NEW / 'programme_design_v1/manifest.json', NEW / 'final_provenance_v1/provenance.json',
                 Path(__file__), Path(__file__).with_name('PROTOCOL.md'),
                 ROOT / 'experiments/combined_leaderboard/statistics.py',
                 ROOT / 'experiments/population_transfer/statistics.py']:
        pin(path, hashes)
    bundle = read_json(OLD / 'bundle.json')
    freeze = read_json(OLD / 'FREEZE.json')
    manifest = read_json(NEW / 'programme_design_v1/manifest.json')
    provenance = read_json(NEW / 'final_provenance_v1/provenance.json')
    old_analysis = read_json(OLD / 'analysis/results.json')
    mask = target_mask(hashes, bundle, manifest)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'run_status.json', {'status': 'started', 'preservation': 'New directory only',
        'resamples': resamples, 'blocks': list(blocks), 'source_sha256_initial': hashes})
    (out / 'PROTOCOL.md').write_text(Path(__file__).with_name('PROTOCOL.md').read_text())
    mask.to_csv(out / 'common_mask.csv', index=False, float_format='%.17g')
    results, coverage, checks, arrays, model_specs = [], [], [], {}, {}
    detail_tables, seed_tables, market_tables = [], [], []
    (out / 'aligned_forecasts').mkdir()
    roster = [('old', m) for m in DAILY_OLD + REALIZED + OLD_DIAGNOSTICS] + [('new', m) for m in NEW_MAIN + NEW_DIAGNOSTICS]
    for origin, model in roster:
        print('validating and scoring', origin, model, flush=True)
        if origin == 'old':
            frame, spec = load_old(model, mask, freeze, hashes)
            model_id = 'volpfn_20260926' if model == 'volpfn' else model
            group = 'daily' if model in DAILY_OLD else 'intraday' if model in REALIZED else 'diagnostic'
        else:
            frame = load_new(model, mask, provenance, manifest, hashes)
            model_id = NEW_IDS[model]
            group = 'daily' if model in NEW_MAIN else 'diagnostic'
            spec = {'configuration': provenance['models'][model]['configuration'],
                    'information_group': 'Daily returns only',
                    'output_adapter': 'Native raw moments' if model != 'direct' else 'M2 only; central variance unavailable'}
        direct = origin == 'new' and model == 'direct'
        frame = score(frame, direct=direct)
        frame['model'] = model_id
        overall, detail, seeds, market = summarize(frame)
        row = dict(model=model_id, label=LABELS[model_id], group=group, seeds=len(seeds), **overall)
        for metric in ['r2_qlike', 'rk_qlike']:
            row[metric + '_seed_min'] = None if seeds[metric].isna().all() else float(seeds[metric].min())
            row[metric + '_seed_max'] = None if seeds[metric].isna().all() else float(seeds[metric].max())
        results.append(row)
        for table, store in [(detail, detail_tables), (seeds, seed_tables), (market, market_tables)]:
            table['model'] = model_id
            store.append(table)
        model_specs[model_id] = {'origin': origin, 'original_model': model, 'group': group, **spec}
        arrays[model_id] = {metric: np.stack([frame.loc[frame.seed == s, metric].to_numpy(float)
                                           for s in seeds.seed]) for metric in ['r2_qlike', 'rk_qlike']}
        for (seed, m), part in frame.groupby(['seed', 'market']):
            c = {'model': model_id, 'seed': seed, 'market': m, 'rows': len(part),
                 'rk_variance_rows': int(part.rk_qlike.notna().sum())}
            for flag in ['substituted', 'floor_applied', 'forecast_failed', 'native_variance_cap_event']:
                c[flag] = None if flag not in part else int(part[flag].sum())
            coverage.append(c)
        if origin == 'old':
            for metric, proxy in [('r2_qlike', 'r2'), ('rk_qlike', 'rk')]:
                archived = next(r for r in old_analysis['leaderboards']['raw'][proxy] if r['model'] == model)['loss']
                np.testing.assert_allclose(overall[metric], archived, rtol=2e-12, atol=2e-12)
                checks.append({'model': model_id, 'check': 'reproduce_original_raw_' + proxy,
                               'absolute_difference': abs(overall[metric] - archived)})
        columns = ['model', 'seed', 'market', 'date', 'r2', 'rk', 'mean', 'variance', 'second_moment',
                   'r2_qlike', 'rk_qlike', 'rk_m2_diagnostic_qlike']
        columns += [c for c in ['original_r2_qlike', 'original_kernel_m2_qlike', 'substituted',
                               'floor_applied', 'forecast_failed', 'native_variance_cap_event',
                               'context_start', 'context_end', 'context_count', 'context_sha256',
                               'target_sha256'] if c in frame]
        frame[columns].to_csv(out / 'aligned_forecasts' / (model_id + '.csv.gz'), index=False,
                            float_format='%.17g', compression={'method': 'gzip', 'mtime': 0})
    table = pd.DataFrame(results)
    for metric in ['r2_qlike', 'rk_qlike']:
        table[metric + '_rank'] = np.nan
        for group in ['daily', 'intraday']:
            subset = table.group == group
            table.loc[subset, metric + '_rank'] = table.loc[subset, metric].rank(method='min')
    table.to_csv(out / 'all_scores.csv', index=False, float_format='%.17g')
    main = table.loc[table.group == 'daily'].sort_values('r2_qlike')
    intraday = table.loc[table.group == 'intraday'].sort_values('r2_qlike')
    diagnostics = table.loc[table.group == 'diagnostic'].sort_values('r2_qlike')
    for name, t in [('daily_leaderboard', main), ('intraday_information_separate', intraday), ('diagnostics_unranked', diagnostics)]:
        t.to_csv(out / (name + '.csv'), index=False, float_format='%.17g')
    for name, items in [('per_seed_market', detail_tables), ('per_seed', seed_tables), ('per_market', market_tables)]:
        pd.concat(items, ignore_index=True).to_csv(out / (name + '.csv'), index=False, float_format='%.17g')
    pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
    # Retain original calibrated results under their own labels, never in the new ranking.
    (out / 'original_calibrated_leaderboard.csv').write_bytes((OLD / 'analysis/leaderboard.csv').read_bytes())
    stats, archive = inference(arrays, mask, list(main.model), resamples, blocks)
    write_json(out / 'statistics.json', stats)
    np.savez_compressed(out / 'bootstrap_archive.npz', **archive)
    preservation = {p: sha(p) == digest for p, digest in hashes.items()}
    if not all(preservation.values()):
        raise ValueError('Preserved inputs changed during analysis')
    write_json(out / 'verification.json', {'status': 'passed', 'all_source_hashes_unchanged': True,
        'source_files_checked': len(hashes), 'rows_per_seed': len(mask), 'markets': 28,
        'all_rows_unit_checked_against_both_raw_sources': True, 'exact_seed_key_target_moment_checks': True,
        'old_raw_score_reproductions': checks, 'original_calibrated_copy_identical':
        sha(out / 'original_calibrated_leaderboard.csv') == sha(OLD / 'analysis/leaderboard.csv')})
    write_json(out / 'provenance.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': hashes, 'models': model_specs, 'reference': REFERENCE,
        'units': {'mean': 'percentage points', 'moments_and_targets': 'percentage points squared'},
        'posthoc_calibration': 'None for every model; old calibrated leaderboard preserved separately',
        'forecast_functionals': {'r2': 'second_moment', 'rk': 'central variance'},
        'rows_per_seed': len(mask), 'calendar_dates': 827, 'markets': 28,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                    'executable': sys.executable},
        'unavailable': ['GP-Vol complete compatible coverage', 'RECH', 'DCC/GN-GARCH', 'New nonlinear intraday experiments'],
        'original_kernel_m2_scores_preserved': True, 'retraining_or_refitting': False})
    main_stats = [r for r in stats['contrasts'] if r['block'] == blocks[0]]
    supported = sorted({r['model'] for r in main_stats if r['both_metrics_positive_supported']})
    report = f'''# Harmonized saved-forecast leaderboard

Separate retrospective reanalysis of **22,026 identical market/date rows**, 28 markets,
2019–2022. No training, refitting, posthoc calibration, clipping or deletion.
Every original source checked before and after processing ({len(hashes)} pinned files).

## Daily-return information

Lower QLIKE is better. Tables average losses over time within market, then markets
and seeds equally. Five seeds are separate scored runs, not an ensemble.

{markdown_table(main)}

## Consistent scaling and scoring

Returns/forecast means use percent units; both proxies and forecast moments use
percent-squared units. All 22,026 targets were checked against both source archives:
decimal return ×100 once, decimal kernel ×10,000 once. All old raw scores reproduce.
Squared returns are scored against **M2**, RK against **central variance** throughout.
Do not treat these two forecast quantities as interchangeable when the mean is nonzero.
All forecasts were already in original units, so none was standardized again.

The old primary leaderboard used separately estimated 2012–2013 calibration scalars.
Those scores remain in `original_calibrated_leaderboard.csv` and the original archive,
outside this raw ranking. New models were selected on 2011–2014 validation; those old
dates cannot supply independent calibration. Reusing old calibrated scores alongside
new raw scores would be inconsistent. This table does **not** replace the old experiment.

The overnight saved RK score used M2. Its values/files remain untouched. Here kernel
QLIKE is recomputed from central variance to match the older scoring functional.
The full2015–2022 overnight study and its p-values are not spliced into this shorter study.

## Additional intraday information — separate descriptive ranks

{markdown_table(intraday)}

These procedures see past RK values, unlike the daily-only group. No blend was
constructed and this table does not establish equal-information superiority.

## Preserved supplementary configurations — no main ranks

{markdown_table(diagnostics)}

The direct arm has only M2, so RK/central-variance QLIKE is unavailable by design.
Its original M2-based RK diagnostic is retained in the detailed outputs. Compact9
and old alternative RK adapters remain preserved without main-table ranks.

## New conditional uncertainty analysis

The daily family has {stats['family_slots']} slots (14 comparators × both proxies),
20,000 synchronized calendar-block draws; 60 dates main, 20/120 sensitivities.
Two-sided centered p-values use plus-one correction and joint Holm. Save ordinary
95% and Bonferroni family intervals; no historical p-values are copied. Positive
support also requires every reference seed positive and both proxies passing.
Supported on both at the main block: {', '.join(supported) or 'none'}.
Detailed directions, intervals and sensitivities: `statistics.json`.
Intraday and design-variant tables are descriptive outside this family.

These are conditional, exposed historical pipeline comparisons. Contexts, budgets,
fitting opportunities, pretraining exposure and model adaptations differ. They do not
isolate architecture or establish universal/first-ever superiority. Kernel equivalence
to the open-to-close target remains unverified. Moirai/HAR floor events and extreme
EGARCH forecasts remain part of their issued procedures; do not claim the huge losses
prove their underlying model families cannot work. GP-Vol/RECH/DCC and new intraday
frontier models have no compatible completed comparison here, so are not defeated.

## Reproduction and preservation

Run the module `experiments.harmonized_leaderboard_20260930.run` with the recorded Python
runtime and a **fresh** `--out` directory. Existing destinations are refused.
See `PROTOCOL.md`, `provenance.json`, `verification.json`, coverage/per-seed/per-market
CSVs and compressed aligned row forecasts. `artifact_receipt.json` hashes all outputs.
Original datasets, checkpoints, frozen sources, failures and older results were not
edited. No GPU or paid resource was used for this postprocessing.
'''
    (out / 'REPORT.md').write_text(report)
    write_json(out / 'artifact_receipt.json', {'status': 'complete', 'files': {
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    table = run(args.out)
    print(table.loc[table.group == 'daily', ['model', 'r2_qlike', 'rk_qlike']].sort_values('r2_qlike').to_string(index=False))


if __name__ == '__main__':
    main()
