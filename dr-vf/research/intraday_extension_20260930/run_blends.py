"""New raw-component blends and full comparator register; never overwrite history."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from research.blend_rk.blend import blend_series, qlike

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research/outputs/harmonized_leaderboard_20260930_v1'
SEEDS = ('8101', '8102', '8103', '8104', '8105')
REFERENCE = 'volpfn_20260930_full+ttm_rk'
PAIRS = (
    ('volpfn_20260930_full', 'ttm_rk'),
    ('volpfn_20260926', 'ttm_rk'),
    ('volpfn_20260930_full', 'loghar'),
    ('volpfn_20260930_full', 'ms_har'),
    ('gjr_garch_t', 'ttm_rk'),
    ('gjr_garch_t', 'loghar'),
    ('ms_har', 'loghar'),
)
BASES = ('volpfn_20260930_full', 'volpfn_20260926', 'gjr_garch_t',
         'har', 'loghar', 'ttm_rk', 'ms_har', 'log_rhar')
FRONTIER = ('gnnhar', 'qlike_har', 'transfer_xgboost', 'm2vn', 'deep_rgarch')
ROSTER = tuple(a + '+' + b for a, b in PAIRS) + BASES + FRONTIER
assert len(ROSTER) == 20 and len(set(ROSTER)) == 20


def expected_seeds(model):
    """Retain every registered stochastic seed; deterministic controls stay one run."""
    if model not in ROSTER:
        raise ValueError('Undeclared procedure: ' + model)
    return SEEDS if (model.startswith('volpfn') or
                     model in FRONTIER and model != 'qlike_har') else ('none',)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def validate_mask(mask):
    if len(mask) != 22026 or mask.market.nunique() != 28 or mask.date.nunique() != 827:
        raise ValueError('Original exact common population required')
    if mask.duplicated(['market', 'date']).any() or not mask.equals(mask.sort_values(['market', 'date']).reset_index(drop=True)):
        raise ValueError('Unique, canonical market/date order required')
    if not np.isfinite(mask[['r2', 'rk']]).all().all() or (mask.r2 < 0).any() or (mask.rk <= 0).any():
        raise ValueError('Invalid original targets')


def align_native(frame, mask, seeds):
    """Reject partial runs, target/unit changes or invalid saved native moments."""
    f = frame.copy()
    f['seed'] = f.seed.astype(str)
    if set(f.seed) != set(seeds) or f.duplicated(['seed', 'market', 'date']).any():
        raise ValueError('Wrong seed coverage or duplicate forecast keys')
    keys = pd.MultiIndex.from_frame(mask[['market', 'date']])
    groups = {}
    for seed in seeds:
        rows = f.loc[f.seed == seed].set_index(['market', 'date'])
        if len(rows) != len(keys) or set(rows.index) != set(keys):
            raise ValueError('Incomplete or extra native forecast keys')
        rows = rows.reindex(keys).reset_index()
        for target in ('r2', 'rk'):
            np.testing.assert_allclose(rows[target], mask[target], rtol=1e-12, atol=1e-12,
                                       err_msg='Target/scaling mismatch: ' + target)
        moments = rows[['mean', 'variance', 'second_moment']].to_numpy(float)
        if not np.isfinite(moments).all() or (moments[:, 1:] <= 0).any():
            raise ValueError('Invalid saved native moments')
        np.testing.assert_allclose(rows.second_moment, rows.variance + rows['mean'] ** 2,
                                   rtol=1e-7, atol=1e-10)
        groups[seed] = rows
    return groups


def align_risk(frame, mask, seeds, *, required_cutoffs=()):
    """Validate native RK point forecasts without fabricating a return density."""
    if not {'origin_date', 'latest_input_date'}.issubset(frame):
        raise ValueError('Native risk forecasts require explicit causal origin/input cutoffs')
    if not set(required_cutoffs).issubset(frame):
        raise ValueError('Model-specific causal cutoff evidence required')
    f = frame.copy()
    f['seed'] = f.seed.astype(str)
    if set(f.seed) != set(seeds) or f.duplicated(['seed', 'market', 'date']).any():
        raise ValueError('Wrong seed coverage or duplicate risk forecast keys')
    keys = pd.MultiIndex.from_frame(mask[['market', 'date']])
    groups = {}
    for seed in seeds:
        rows = f.loc[f.seed == seed].set_index(['market', 'date'])
        if len(rows) != len(keys) or set(rows.index) != set(keys):
            raise ValueError('Incomplete or extra risk forecast keys')
        rows = rows.reindex(keys).reset_index()
        if not np.isfinite(rows.rk_forecast).all() or (rows.rk_forecast <= 0).any():
            raise ValueError('Nonpositive/nonfinite native risk forecast')
        # Origins and all reported input cutoffs must be earlier than target day.
        for field in ('origin_date', 'latest_input_date', 'latest_neighbor_input_date',
                      'latest_source_label_date', 'fit_end'):
            if field in rows and (rows[field].isna().any() or (rows[field] >= rows.date).any()):
                raise ValueError('Missing or noncausal cutoff: ' + field)
        if 'fit_block_start' in rows:
            if (rows.fit_block_start.isna().any() or
                    (rows.fit_block_start > rows.date).any() or
                    'fit_end' not in rows or (rows.fit_end >= rows.fit_block_start).any()):
                raise ValueError('Fit must precede its forecast block')
        groups[seed] = rows
    return groups


def summary(frame):
    per = frame.groupby(['seed', 'market'])[['r2_qlike', 'rk_qlike']].mean()
    per_seed = per.groupby('seed').mean()
    values = per_seed.mean()
    return {'r2_qlike': float(values.r2_qlike), 'rk_qlike': float(values.rk_qlike),
            'seeds': len(per_seed), 'markets': frame.market.nunique(),
            'rows_per_seed': len(frame) // len(per_seed)}


def scored_rows(mask, model, seed, h_r2, h_rk):
    rows = mask[['market', 'date', 'r2', 'rk']].copy()
    rows['model'], rows['seed'] = model, seed
    rows['r2_forecast'], rows['rk_forecast'] = h_r2, h_rk
    rows['r2_qlike'], rows['rk_qlike'] = qlike(rows.r2, h_r2), qlike(rows.rk, h_rk)
    return rows


def run(out, extras=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    extras = extras or {}
    if set(extras) - set(FRONTIER):
        raise ValueError('Undeclared additional procedure')
    for name, spec in extras.items():
        if tuple(map(str, spec['seeds'])) != expected_seeds(name):
            raise ValueError('All registered seeds required: ' + name)
        if 'quality_verification' not in spec:
            raise ValueError('Independent fitting/coverage verification required: ' + name)
    pins = {}
    def pin(path):
        path = Path(path).resolve()
        value = sha(path)
        if str(path) in pins and pins[str(path)] != value:
            raise ValueError('Source changed during read')
        pins[str(path)] = value
        return path

    mask = pd.read_csv(pin(SOURCE / 'common_mask.csv'), dtype={'market': str, 'date': str})
    validate_mask(mask)
    calendar = sorted(mask.date.unique())
    didx = np.searchsorted(calendar, mask.date)
    keys = list(zip(mask.market, mask.date))
    sources = {}
    for name in BASES:
        path = pin(SOURCE / 'aligned_forecasts' / (name + '.csv.gz'))
        f = pd.read_csv(path, dtype={'market': str, 'date': str, 'seed': str})
        seeds = SEEDS if name.startswith('volpfn') else ('none',)
        sources[name] = align_native(f, mask, seeds)
    for path in (Path(__file__), Path(__file__).with_name('PROTOCOL.md'),
                 ROOT / 'research/blend_rk/blend.py', SOURCE / 'provenance.json',
                 SOURCE / 'verification.json'):
        pin(path)
    write_json(out / 'REGISTERED_DESIGN.json', {
        'reference': REFERENCE, 'roster': ROSTER, 'family_slots': 38,
        'proxy_functionals': {'r2': 'saved M2; native RK risk model uses issued risk forecast',
                             'rk': 'saved central variance; native RK risk model uses issued risk forecast'},
        'blend': {'refit': 20, 'burn': 60, 'weight': 'unconstrained log-linear',
                  'fit': 'pooled past market/day QLIKE, separate by proxy and seed',
                  'scalars': 'none; raw components; fitted intercept still calibrates'},
        'seed_ids': SEEDS, 'seed_ids_by_model': {n: expected_seeds(n) for n in ROSTER},
        'period': [min(calendar), max(calendar)], 'rows': len(mask),
        'bootstrap_if_run': {'resamples': 20000, 'seed': 2026093004, 'blocks': [60, 20, 120]},
        'status': 'retrospective; roster fixed before extension scores; no fresh holdout'})
    mask.to_csv(out / 'common_mask.csv', index=False)
    (out / 'row_forecasts').mkdir()
    frames, weights, records = {}, {}, []
    for model, seeds in sources.items():
        frames[model] = pd.concat([scored_rows(mask, model, s, f.second_moment, f.variance)
                                  for s, f in seeds.items()], ignore_index=True)
    for a, b in PAIRS:
        model, built = a + '+' + b, []
        seeds = SEEDS if a.startswith('volpfn') else ('none',)
        for seed in seeds:
            fa, fb = sources[a][seed], sources[b]['none']
            forecasts = {}
            for proxy, field in (('r2', 'second_moment'), ('rk', 'variance')):
                forecasts[proxy], wts = blend_series(keys, didx, fa[field], fb[field], mask[proxy])
                for row in wts:
                    row['block_start_date'] = calendar[row['from_date']]
                    row['latest_fit_target_date'] = (calendar[row['from_date'] - 1]
                                                     if row['from_date'] >= 60 else None)
                weights[model + '|' + seed + '|' + proxy] = wts
            built.append(scored_rows(mask, model, seed, forecasts['r2'], forecasts['rk']))
        frames[model] = pd.concat(built, ignore_index=True)
    for name, spec in extras.items():
        quality_path = pin(spec['quality_verification'])
        if json.loads(quality_path.read_text()).get('status') != 'passed':
            raise ValueError('Independent quality gate did not pass: ' + name)
        path = pin(spec['path'])
        f = pd.read_csv(path, dtype={'market': str, 'date': str, 'seed': str})
        required = {'gnnhar': ('latest_neighbor_input_date',),
                    'qlike_har': ('fit_end',),
                    'transfer_xgboost': ('fit_end', 'fit_block_start', 'latest_source_label_date')}
        groups = align_risk(f, mask, tuple(map(str, spec['seeds'])),
                            required_cutoffs=required.get(name, ()))
        frames[name] = pd.concat([scored_rows(mask, name, s, f.rk_forecast, f.rk_forecast)
                                 for s, f in groups.items()], ignore_index=True)
        for extra_pin in spec.get('source_pins', []):
            pin(extra_pin)
    for name in ROSTER:
        if name in frames:
            f = frames[name]
            f.to_csv(out / 'row_forecasts' / (name + '.csv.gz'), index=False, compression='gzip')
            f.groupby(['seed', 'market'])[['r2_qlike', 'rk_qlike']].mean().reset_index().to_csv(
                out / (name + '_per_seed_market.csv'), index=False)
            records.append({'model': name, 'status': 'complete', **summary(f)})
        else:
            records.append({'model': name, 'status': 'unverified/not yet reproduced',
                            'r2_qlike': None, 'rk_qlike': None, 'seeds': 0, 'markets': 0,
                            'rows_per_seed': 0})
    table = pd.DataFrame(records)
    for proxy in ('r2', 'rk'):
        table[proxy + '_rank'] = table[proxy + '_qlike'].rank(method='min')
    table.to_csv(out / 'leaderboard.csv', index=False)
    write_json(out / 'weights.json', weights)
    write_json(out / 'verification.json', {
        'exact_common_keys': True, 'all_declared_seeds_required': True,
        'units': 'percent-squared, no new scaling or old posthoc scalar',
        'available_procedures': len(frames), 'registered_procedures': len(ROSTER),
        'row_count': sum(len(f) for f in frames.values()),
        'historical_outputs_overwritten': False,
        'statistics': 'not run at blend-building stage; no significance claim'})
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Source modified: ' + path)
    write_json(out / 'provenance.json', {'source_hashes': pins, 'python': platform.python_version(),
               'numpy': np.__version__, 'pandas': pd.__version__,
               'reference': REFERENCE, 'originals_preserved': True})
    lines = ['# Intraday-information extension — saved blends', '',
             'Same 22,026 keys, 28 markets, 827 dates; raw issued components.',
             'Reference is frozen Sep 30 full VolPFN + saved TTM-RK. No VolPFN retraining.', '',
             '| Procedure | r² QLIKE | RK QLIKE | Seeds |', '|---|---:|---:|---:|']
    for _, row in table.loc[table.status == 'complete'].sort_values('rk_qlike').iterrows():
        lines.append(f"| {row.model} | {row.r2_qlike:.6f} | {row.rk_qlike:.6f} | {row.seeds} |")
    lines += ['', 'Unverified frontier slots: ' + ', '.join(n for n in FRONTIER if n not in frames), '',
              'Ranks are descriptive; no significance analysis has run at this stage.',
              'Blends fit separate proxy/seed weights causally from earlier completed dates.',
              'These are risk-level forecasts, not jointly coherent return densities.',
              'The intercept gives blends online calibration; raw standalone comparators lack it.',
              'Matched GJR+TTM and GJR+Log-HAR controls receive the same fitting opportunity.',
              'Old calibrated results remain in their original archives and are not spliced here.',
              'This exposed panel is not independent confirmation or a universal frontier test.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
               for p in sorted(out.rglob('*')) if p.is_file()})
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--extras-json', help='Registered native RK forecasts and declared seeds')
    args = parser.parse_args()
    extras = json.loads(Path(args.extras_json).read_text()) if args.extras_json else None
    print(run(args.out, extras).to_string(index=False))
