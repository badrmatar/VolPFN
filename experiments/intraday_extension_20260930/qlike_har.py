"""Strict-domain QLIKE HAR on RK, with daily causal 1,000-session fits.

Adaptation of Puke/Schweikert's generic author objective. The application-specific
square-root target is deliberately replaced by variance-level RK to match this
comparison; no exact reproduction of the paper's data/target is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
MASK = ROOT / 'experiments/outputs/harmonized_leaderboard_20260930_v1/common_mask.csv'
DATA = ROOT / 'experiments/data/oxfordman_barde/calendar_full_v1'
AUTHOR_PIN = 'ddca68c48510c55445e6041c91435fbabed65240'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def causal_features(risk):
    """Own-session 1/5/22 nested means, using prior-only carried missing inputs.

    Missing target observations are never replaced: filling is confined to
    feature construction, and a missing current RK cannot affect its own input.
    """
    risk = np.asarray(risk, float)
    valid = np.isfinite(risk) & (risk > 0)
    filled = np.full(len(risk), np.nan)
    last = np.nan
    for i, value in enumerate(risk):
        if valid[i]:
            last = value
        filled[i] = last
    x = np.full((len(risk), 3), np.nan)
    for i in range(22, len(risk)):
        previous = filled[i-22:i]
        if np.isfinite(previous).all():
            x[i] = [previous[-1], previous[-5:].mean(), previous.mean()]
    return x, valid


def fit(x, y, forecast_x):
    """Same generic QLIKE/Nelder-Mead objective, rejecting every invalid domain.

    R's na.rm=TRUE could remove invalid fitted observations from the objective.
    Here any h<=0 rejects the whole objective. Scaling uses only this fit window.
    A documented deterministic retry is allowed only on this training objective.
    """
    x, y, forecast_x = map(lambda z: np.asarray(z, float), (x, y, forecast_x))
    if x.shape != (len(y), 3) or forecast_x.shape != (3,) or len(y) < 100:
        raise ValueError('At least100 valid causal fit observations required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Invalid training values')
    scale = float(y.mean())
    design = np.column_stack([np.ones(len(y)), x / scale])
    target = y / scale
    initial = np.linalg.lstsq(design, target, rcond=None)[0]
    baseline = np.array([1., 0., 0., 0.])
    if not np.isfinite(initial).all() or (design @ initial <= 0).any():
        initial = baseline
    def objective(parameters):
        h = design @ parameters
        if not np.isfinite(h).all() or (h <= 0).any():
            return float('inf')
        loss = np.mean(np.log(h) + target / h)
        return float(loss) if np.isfinite(loss) else float('inf')
    attempts = []
    for start, cap in ((initial, 3000), (baseline, 10000)):
        result = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': cap, 'xatol': 1e-7, 'fatol': 1e-10})
        attempts.append({'success': bool(result.success), 'iterations': int(result.nit),
                         'objective': float(result.fun) if np.isfinite(result.fun) else None,
                         'message': str(result.message), 'maxiter': cap})
        if result.success and np.isfinite(result.x).all() and np.isfinite(result.fun):
            h_fit = design @ result.x
            prediction = float(np.r_[1., forecast_x / scale] @ result.x * scale)
            if np.isfinite(prediction) and prediction > 0 and (h_fit > 0).all():
                if objective(result.x) > objective(baseline) + 1e-8:
                    attempts[-1]['training_quality_rejection'] = 'worse than positive constant baseline'
                    continue
                return prediction, result.x, scale, attempts
            attempts[-1]['invalid_forecast'] = True
    raise RuntimeError('No accepted positive QLIKE-HAR fit: ' + json.dumps(attempts))


def forecast_market(market, wanted_dates, *, window=1000, limit=None):
    source = DATA / (market + '.npz')
    digest = sha(source)
    with np.load(source, allow_pickle=False) as data:
        dates = data['date'].astype(str)
        risk = data['rk_parzen'].astype(float) * 10000.
    if len(set(dates)) != len(dates) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError('Noncanonical source dates')
    x, valid_target = causal_features(risk)
    indices = np.searchsorted(dates, wanted_dates)
    if np.any(indices >= len(dates)) or not np.array_equal(dates[indices], wanted_dates):
        raise ValueError('Missing requested original key')
    if limit is not None:
        indices = indices[:limit]
    rows, fits = [], []
    start = time.perf_counter()
    for i in indices:
        low = max(22, i - window)
        fit_indices = np.arange(low, i)
        keep = valid_target[fit_indices] & np.isfinite(x[fit_indices]).all(axis=1)
        train = fit_indices[keep]
        prediction, coefficients, scale, attempts = fit(x[train], risk[train], x[i])
        rows.append({'market': market, 'date': dates[i], 'seed': 'none',
                     'rk_forecast': prediction, 'origin_date': dates[i-1],
                     'latest_input_date': dates[i-1], 'fit_end': dates[train[-1]],
                     'fit_start': dates[train[0]], 'fit_count': len(train)})
        fits.append({'date': dates[i], 'fit_end': dates[train[-1]], 'fit_count': len(train),
                     'scale': scale, 'coefficients': coefficients.tolist(), 'attempts': attempts})
    if sha(source) != digest:
        raise ValueError('Source modified during fit')
    return {'market': market, 'rows': rows, 'fits': fits, 'source_path': str(source),
            'source_sha256': digest, 'elapsed_seconds': time.perf_counter() - start,
            'window': window}


def run(out, workers=4, pilot=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    mask_hash = sha(MASK)
    mask = pd.read_csv(MASK, dtype={'market': str, 'date': str})
    markets = sorted(mask.market.unique())
    if pilot is not None:
        markets = ['SPX']
    design = {'model': 'QLIKE-HAR RK variance-level adaptation', 'author_pin': AUTHOR_PIN,
              'target': 'RK percent-squared; generic author loss with identity transform',
              'window_sessions': 1000, 'refit': 'each target session; strictly earlier labels',
              'features': 'own prior1/5/22-session means; prior-only input carry, masked missinglabels',
              'optimizer': 'strict-domain Nelder-Mead, OLSstart; cap3000; constantstart retrycap10000',
              'forecast_policy': 'reject failed fits; no outcome-dependent clipping/carry/drop',
              'seed': 'none, deterministic', 'mask_sha256': mask_hash,
              'pilot': pilot is not None, 'pilot_rows': pilot, 'workers': workers}
    (out / 'DESIGN.json').write_text(json.dumps(design, indent=2))
    (out / 'markets').mkdir()
    all_rows, receipts = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(forecast_market, market,
                           mask.loc[mask.market == market, 'date'].to_numpy(str), limit=pilot): market
                for market in markets}
        for future in as_completed(jobs):
            result = future.result()
            pd.DataFrame(result.pop('rows')).to_csv(out / 'markets' / (result['market'] + '.csv'), index=False)
            fits = result.pop('fits')
            (out / 'markets' / (result['market'] + '_fits.json')).write_text(json.dumps(fits, allow_nan=False))
            receipts.append(result)
            all_rows.append(pd.read_csv(out / 'markets' / (result['market'] + '.csv')))
            print(result['market'], result['elapsed_seconds'], flush=True)
    predictions = pd.concat(all_rows, ignore_index=True).sort_values(['market', 'date']).reset_index(drop=True)
    if pilot is None:
        if not predictions[['market', 'date']].equals(mask[['market', 'date']]) or len(predictions) != 22026:
            raise ValueError('Exact common coverage failed')
    predictions.to_csv(out / 'predictions.csv', index=False)
    if sha(MASK) != mask_hash:
        raise ValueError('Mask modified')
    receipt = {'status': 'pilot_complete' if pilot else 'complete', 'rows': len(predictions),
               'markets': len(markets), 'elapsed_seconds': time.perf_counter() - started,
               'source_receipts': receipts, 'code_sha256': sha(__file__),
               'predictions_sha256': sha(out / 'predictions.csv'), 'originals_preserved': True,
               'scoring': 'not performed here'}
    (out / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pilot', type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.workers, args.pilot), indent=2))
