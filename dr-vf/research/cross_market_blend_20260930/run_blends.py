"""Fixed cross-market helper controls; retain all original forecasts/results."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.blend_rk.blend import blend_series, qlike
from research.intraday_extension_20260930.run_blends import (
    ROOT, SEEDS, scored_rows, sha, summary, validate_mask, write_json)

SOURCE = ROOT / 'research/outputs/intraday_extension_20260930_comparison_final_v3'
REFERENCE = 'volpfn_full+crossmarket_xgb'
ROSTER = (REFERENCE, 'crossmarket_xgb_calibrated', 'gjr+crossmarket_xgb',
          'crossmarket_xgb', 'volpfn_20260930_full+ttm_rk', 'gnnhar', 'transfer_xgboost')
FAMILY = 2 * (len(ROSTER) - 1)


def aligned(frame, mask, seeds=SEEDS):
    f = frame.copy()
    f['seed'] = f.seed.astype(str)
    if set(f.seed) != set(seeds) or f.duplicated(['market', 'date', 'seed']).any():
        raise ValueError('Exact declared seeds and unique keys required')
    index = pd.MultiIndex.from_frame(mask[['market', 'date']])
    result = {}
    for seed in seeds:
        block = f.loc[f.seed.eq(seed)].set_index(['market', 'date'])
        if len(block) != len(index) or set(block.index) != set(index):
            raise ValueError('Full original mask required per seed')
        result[seed] = block.reindex(index).reset_index()
    return result


def helper_metadata(frame):
    fields = ('date', 'origin_date', 'latest_own_input_date', 'latest_foreign_input_date')
    for field in fields:
        values = frame[field]
        if values.isna().any() or not values.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            raise ValueError('Missing/invalid helper input date: ' + field)
        pd.to_datetime(values, format='%Y-%m-%d', errors='raise')
    if (not np.isfinite(frame.rk_forecast).all() or (frame.rk_forecast <= 0).any()
            or (frame.origin_date >= frame.date).any()
            or (frame.latest_own_input_date > frame.origin_date).any()
            or (frame.latest_foreign_input_date >= frame.origin_date).any()
            or (frame.foreign_count != 27).any()):
        raise ValueError('Positive, fully covered causal cross-market inputs required')
    if not frame.train_end.eq('2017-12-31').all() or not frame.validation_end.eq('2018-12-31').all():
        raise ValueError('Registered frozen helper cutoffs required')


def calibration_fit(log_h, y):
    log_h, y = np.asarray(log_h, float), np.asarray(y, float)
    if (log_h.ndim != 1 or log_h.shape != y.shape or not len(y)
            or not np.isfinite(log_h).all() or not np.isfinite(y).all() or (y < 0).any()):
        raise ValueError('Finite aligned calibration histories required')
    def objective(p):
        z = p[0] + p[1] * log_h
        with np.errstate(over='ignore', invalid='ignore'):
            value = np.mean(y * np.exp(-z) + z)
        return float(value) if np.isfinite(value) else np.inf
    start = np.array([0., 1.])
    fit = minimize(objective, start, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 2000})
    if (not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun)
            or fit.fun > objective(start) + 1e-8):
        raise RuntimeError('Helper-only calibration failed: ' + fit.message)
    return fit.x


def calibrate_series(keys, didx, h, y):
    didx, h, y = np.asarray(didx), np.asarray(h, float), np.asarray(y, float)
    if (didx.shape != h.shape or h.shape != y.shape or y.ndim != 1 or not len(y)
            or len(keys) != len(y) or len(set(keys)) != len(keys)
            or not np.issubdtype(didx.dtype, np.integer) or (didx < 0).any()
            or not np.isfinite(h).all() or (h <= 0).any()
            or not np.isfinite(y).all() or (y < 0).any()):
        raise ValueError('Valid positive complete calibration forecasts required')
    out, weights, log_h = np.empty(len(y)), [], np.log(h)
    for start in range(0, int(didx.max()) + 1, 20):
        params = np.array([0., 1.])
        if start >= 60:
            past = didx < start
            params = calibration_fit(log_h[past], y[past])
        selected = (didx >= start) & (didx < start + 20)
        out[selected] = np.exp(params[0] + params[1] * log_h[selected])
        weights.append({'from_date': start, 'a': float(params[0]), 'b': float(params[1])})
    if not np.isfinite(out).all() or (out <= 0).any():
        raise RuntimeError('Invalid issued calibrated forecast')
    return out, weights


def annotate_weights(weights, calendar, didx, ha, hb, y):
    """Audit the starting-objective gate without changing the historical fit."""
    for row in weights:
        start = row['from_date']
        row['block_start_date'] = calendar[start]
        row['latest_fit_target_date'] = calendar[start - 1] if start >= 60 else None
        row['fit_rows'] = int(np.count_nonzero(didx < start)) if start >= 60 else 0
        if start < 60:
            continue
        past = didx < start
        if 'w' in row:
            log_a, log_b = np.log(ha[past]), np.log(hb[past])
            fitted = row['a'] + row['w'] * log_a + (1 - row['w']) * log_b
            initial = .5 * (log_a + log_b)
        else:
            initial = np.log(ha[past])
            fitted = row['a'] + row['b'] * initial
        fitted_loss = float(np.mean(y[past] * np.exp(-fitted) + fitted))
        initial_loss = float(np.mean(y[past] * np.exp(-initial) + initial))
        if not np.isfinite(fitted_loss) or fitted_loss > initial_loss + 1e-8:
            raise RuntimeError('Registered calibration starting-objective gate failed')
        row['fit_objective'], row['starting_objective'] = fitted_loss, initial_loss


def run(helper, out):
    helper, out = Path(helper).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins = {}
    def pin(path):
        p = Path(path).resolve()
        digest = sha(p)
        if str(p) in pins and pins[str(p)] != digest:
            raise ValueError('Already-pinned input modified: ' + str(p))
        pins[str(p)] = digest
        return p
    for path in (Path(__file__), Path(__file__).with_name('PROTOCOL.md'),
                 Path(__file__).with_name('test_blends.py'),
                 ROOT / 'research/blend_rk/blend.py',
                 ROOT / 'research/intraday_extension_20260930/run_blends.py'):
        pin(path)
    artifacts = json.loads(pin(SOURCE / 'ARTIFACT_RECEIPT.json').read_text())
    for relative, digest in artifacts.items():
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts or sha(pin(SOURCE / name)) != digest:
            raise ValueError('Frozen original artifact changed: ' + relative)
    mask = pd.read_csv(pin(SOURCE / 'common_mask.csv'), dtype={'market': str, 'date': str})
    validate_mask(mask)
    source_provenance = json.loads(pin(SOURCE / 'provenance.json').read_text())
    for path, expected in source_provenance['source_hashes'].items():
        if sha(pin(path)) != expected:
            raise ValueError('Frozen original source changed: ' + path)
    quality = json.loads(pin(helper / 'QUALITY_VERIFICATION.json').read_text())
    if (quality['status'] != 'passed' or quality['checkpoints'] != 140
            or quality['evaluation_rows'] != 5 * len(mask)
            or not quality['all_28_markets'] or not quality['all_five_seeds']
            or not quality['once_only_percent_squared_scaling']
            or not quality['causal_metadata_rebuilt']):
        raise ValueError('Independent all-market helper verification required')
    required_helper_pins = {str((helper / name).resolve()) for name in ('predictions.csv', 'DESIGN.json', 'RECEIPT.json')}
    if not required_helper_pins.issubset(quality['source_hashes']):
        raise ValueError('Helper verification must bind design, receipt and forecasts')
    for path, expected in quality['source_hashes'].items():
        if sha(pin(path)) != expected:
            raise ValueError('Verified helper source changed: ' + path)
    for name in ('DESIGN.json', 'RECEIPT.json'):
        pin(helper / name)
    hf = pd.read_csv(pin(helper / 'predictions.csv'),
                     dtype={'market': str, 'date': str, 'seed': str})
    helpers = aligned(hf, mask)
    for f in helpers.values():
        helper_metadata(f)
    loaded = {}
    for name in ('volpfn_20260930_full', 'gjr_garch_t',
                 'volpfn_20260930_full+ttm_rk', 'gnnhar', 'transfer_xgboost'):
        f = pd.read_csv(pin(SOURCE / 'row_forecasts' / (name + '.csv.gz')),
                        dtype={'market': str, 'date': str, 'seed': str})
        loaded[name] = aligned(f, mask, ('none',) if name == 'gjr_garch_t' else SEEDS)
        for group in loaded[name].values():
            for proxy in ('r2', 'rk'):
                np.testing.assert_allclose(group[proxy], mask[proxy], rtol=1e-12, atol=1e-12)
                values = group[proxy + '_forecast']
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError('Invalid frozen native forecast')
                np.testing.assert_allclose(group[proxy + '_qlike'], qlike(group[proxy], values),
                                           rtol=1e-12, atol=1e-12)
        if name in ('volpfn_20260930_full', 'gjr_garch_t'):
            for f in loaded[name].values():
                if (f.r2_forecast + 1e-10 < f.rk_forecast).any():
                    raise ValueError('Saved M2 must include central variance')
    calendar = sorted(mask.date.unique())
    didx, keys = np.searchsorted(calendar, mask.date), list(zip(mask.market, mask.date))
    frames, weights = {}, {}
    for name in ('volpfn_20260930_full+ttm_rk', 'gnnhar', 'transfer_xgboost'):
        frames[name] = pd.concat(list(loaded[name].values()), ignore_index=True)
    for seed in SEEDS:
        h = helpers[seed].rk_forecast.to_numpy(float)
        raw = scored_rows(mask, 'crossmarket_xgb', seed, h, h)
        frames.setdefault('crossmarket_xgb', []).append(raw)
        for model, component in ((REFERENCE, loaded['volpfn_20260930_full'][seed]),
                                 ('gjr+crossmarket_xgb', loaded['gjr_garch_t']['none']),
                                 ('crossmarket_xgb_calibrated', None)):
            forecasts = {}
            for proxy in ('r2', 'rk'):
                y = mask[proxy].to_numpy(float)
                if component is None:
                    forecast, fitted = calibrate_series(keys, didx, h, y)
                    annotate_weights(fitted, calendar, didx, h, None, y)
                else:
                    own = component[proxy + '_forecast'].to_numpy(float)
                    forecast, fitted = blend_series(keys, didx, own, h, y)
                    annotate_weights(fitted, calendar, didx, own, h, y)
                forecasts[proxy] = forecast
                weights[model + '|' + seed + '|' + proxy] = fitted
            frames.setdefault(model, []).append(scored_rows(
                mask, model, seed, forecasts['r2'], forecasts['rk']))
    frames = {name: pd.concat(value, ignore_index=True) if isinstance(value, list) else value
              for name, value in frames.items()}
    rows_folder = out / 'row_forecasts'
    rows_folder.mkdir()
    records = []
    for name in ROSTER:
        f = frames[name]
        if set(f.seed.astype(str)) != set(SEEDS):
            raise ValueError('All five registered seeds required')
        f.to_csv(rows_folder / (name + '.csv.gz'), index=False, compression='gzip')
        f.groupby(['seed', 'market'])[['r2_qlike', 'rk_qlike']].mean().reset_index().to_csv(
            out / (name + '_per_seed_market.csv'), index=False)
        records.append({'model': name, 'status': 'complete', **summary(f)})
    table = pd.DataFrame(records)
    for proxy in ('r2', 'rk'):
        table[proxy + '_rank'] = table[proxy + '_qlike'].rank(method='min')
    table.to_csv(out / 'leaderboard.csv', index=False)
    mask.to_csv(out / 'common_mask.csv', index=False)
    write_json(out / 'weights.json', weights)
    for p in (Path(__file__), Path(__file__).with_name('PROTOCOL.md'),
              ROOT / 'research/blend_rk/blend.py'):
        pin(p)
    write_json(out / 'REGISTERED_DESIGN.json', {'reference': REFERENCE, 'roster': ROSTER,
        'seeds': SEEDS, 'family_slots': FAMILY, 'row_count_per_seed': len(mask),
        'refit_dates': 20, 'burn_dates': 60,
        'inferential_seed': 2026093005, 'blocks': [60, 20, 120], 'resamples': 20000,
        'scope': 'new retrospective conditional follow-up; old 38-slot family unchanged',
        'helper': str(helper), 'protocol_sha256': sha(Path(__file__).with_name('PROTOCOL.md'))})
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Input modified while comparing: ' + path)
    write_json(out / 'provenance.json', {'source_hashes': pins, 'originals_preserved': True})
    write_json(out / 'verification.json', {'status': 'passed', 'exact_common_mask': True,
        'all_five_seeds': True, 'procedures': len(frames),
        'scored_rows': sum(len(f) for f in frames.values()), 'no_volpfn_retraining': True})
    frozen = out / 'frozen_sources'
    frozen.mkdir()
    for name in ('run_blends.py', 'test_blends.py', 'PROTOCOL.md'):
        shutil.copy2(Path(__file__).with_name(name), frozen / name)
    write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
               for p in sorted(out.rglob('*')) if p.is_file()})
    print(table.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.helper, args.out)
