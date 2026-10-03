"""Paired losses and synchronized calendar-block inference; no model selection."""
from functools import lru_cache
import numpy as np
import pandas as pd


def qlike(target, forecast):
    y, h = np.broadcast_arrays(np.asarray(target, float), np.asarray(forecast, float))
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('QLIKE targets must be finite and nonnegative')
    if not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError('QLIKE forecasts must be finite and positive')
    return y / h + np.log(h)


def squared_error(target, forecast):
    y, h = np.broadcast_arrays(np.asarray(target, float), np.asarray(forecast, float))
    if not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError('Squared-error inputs must be finite')
    return (y - h) ** 2


def holm(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected a vector of probabilities')
    order = np.argsort(p, kind='stable')
    result = np.empty(len(p))
    result[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


@lru_cache(maxsize=6)
def _calendar_counts(n, reps, block, seed):
    if min(n, reps, block) < 1:
        raise ValueError('Positive calendar length, resample count and block length required')
    rng = np.random.default_rng(seed)
    counts = np.empty((reps, n), dtype=np.float64)
    for j in range(reps):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        counts[j] = np.bincount(indices, minlength=n)
    counts.setflags(write=False)
    return counts


def paired_summary(dates, markets, gains, reps=2000, block=60, seed=927601):
    """Positive gains favour the candidate. Rows must already use matched masks.

    Bootstrap dates jointly across markets; average sampled dates within market,
    then markets equally. p_null is a two-sided centered bootstrap test, not the
    fraction of uncentered draws on the opposite side of zero.
    """
    f = pd.DataFrame({'date': dates, 'market': markets, 'gain': gains})
    if f.empty or f[['date', 'market']].isna().any().any() or not np.isfinite(f.gain).all():
        raise ValueError('Nonempty finite matched observations required')
    if f.duplicated(['date', 'market']).any():
        raise ValueError('Duplicate market/date observations')
    pivot = f.pivot(index='date', columns='market', values='gain').sort_index()
    arr = pivot.to_numpy(float)
    valid = np.isfinite(arr).astype(float)
    totals = np.nan_to_num(arr, nan=0.)
    means = np.nanmean(arr, axis=0)
    estimate = float(means.mean())
    counts = _calendar_counts(len(arr), reps, block, seed)
    denom = counts @ valid
    if np.any(denom == 0):
        raise ValueError('A bootstrap draw omits a market; insufficient calendar coverage')
    draws = np.mean((counts @ totals) / denom, axis=1)
    null_draws = draws - estimate
    p_null = (1 + np.count_nonzero(np.abs(null_draws) >= abs(estimate))) / (reps + 1)
    return {
        'mean_gain': estimate, 'interval95': np.quantile(draws, [.025, .975]).tolist(),
        'p_null': float(p_null), 'markets_improved': int(np.sum(means > 0)),
        'markets': len(means), 'rows': len(f), 'calendar_dates': len(arr),
        'per_market_mean_gain': dict(zip(pivot.columns, map(float, means))),
        'block_observed_dates': block, 'resamples': reps, 'seed': seed,
        'weighting': 'equal markets; time mean within each market',
        'p_method': 'two-sided centered circular date-block bootstrap; plus-one correction',
        'scope': 'conditional on retained forecasts and chosen markets; excludes training and selection uncertainty',
    }
