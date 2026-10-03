"""Outcome-blind OHLC eligibility rules for the prospective index block.

No model or model predictions are imported. A supplied OHLC history must have
its source/index identity verified separately; passing numerical checks cannot
establish source identity or actual exchange-data delivery times.
"""
import math
import numpy as np
import pandas as pd

START = '2022-03-01'
END = '2026-08-31'
MIN_MARKETS = 20
MIN_TARGET_DAYS = 800
MIN_CONTEXT_DAYS = 100
MAX_EQUAL_PREVIOUS_CLOSE = .20
MAX_INVALID_FRACTION = .01


def specification():
    return dict(start=START, end=END, target='100*log(close/open)',
                price_policy='unadjusted index OHLC; source/index identity must be verified; no proxy substitutions',
                date_policy='unique increasing local trading-session dates; no forward fill',
                minimum_markets=MIN_MARKETS, minimum_target_days=MIN_TARGET_DAYS,
                minimum_preblock_context_days=MIN_CONTEXT_DAYS,
                maximum_invalid_primary_row_fraction=MAX_INVALID_FRACTION,
                open_equals_previous_close_maximum_fraction=MAX_EQUAL_PREVIOUS_CLOSE,
                equality='numpy.isclose rtol=1e-10 atol=1e-12 on positive finite prices',
                same_day_open_equals_close='report descriptive fraction; no additional outcome-dependent exclusion',
                fallback='exclude indices failing predeclared QC; if N<20 primary unavailable; no close-to-close switch and no secondary promotion',
                breadth_threshold='ceil(20*N/28) strictly positive per-market mean paired gains',
                exclusions='data-quality rules only, fixed before any model scoring; log every rejected row and market')


def breadth_threshold(markets):
    if not isinstance(markets, (int, np.integer)) or not MIN_MARKETS <= markets <= 28:
        raise ValueError('Primary requires a fixed population of 20 through 28 indices')
    return int(math.ceil(20 * markets / 28))


def check_market(frame, market, identity_verified=False):
    required = ('date', 'open', 'high', 'low', 'close')
    if any(k not in frame for k in required):
        raise ValueError('OHLC input needs date, open, high, low, close')
    f = frame.loc[:, required].copy()
    dates = pd.to_datetime(f['date'], errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Unique increasing session dates required; no silent deduplication/sort')
    if getattr(dates.dt, 'tz', None) is not None:
        raise ValueError('Use documented local session dates, not timezone-bearing instants')
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Input must contain session dates at midnight, not intraday timestamps')
    f['date'] = dates.dt.strftime('%Y-%m-%d')
    for k in required[1:]:
        f[k] = pd.to_numeric(f[k], errors='raise')
    prices = f.loc[:, required[1:]].to_numpy(float)
    finite_positive = np.isfinite(prices).all(1) & (prices > 0).all(1)
    tolerance = 1e-10 * np.maximum(np.nan_to_num(np.abs(prices), nan=0.).max(1), 1.) + 1e-12
    o, h, l, c = (f[k].to_numpy(float) for k in required[1:])
    ordered = (l <= np.minimum(o, c) + tolerance) & (h + tolerance >= np.maximum(o, c)) & (h + tolerance >= l)
    valid = finite_positive & ordered
    in_block = (f.date >= START).to_numpy() & (f.date <= END).to_numpy()
    earlier = (f.date < START).to_numpy()
    previous = np.r_[np.nan, c[:-1]]
    paired = in_block & np.isfinite(o) & (o > 0) & np.isfinite(previous) & (previous > 0)
    equal_previous = np.isclose(o, previous, rtol=1e-10, atol=1e-12) & paired
    equal_same = np.isclose(o, c, rtol=1e-10, atol=1e-12) & in_block & valid
    available = int(in_block.sum())
    target_count, context_count = int((valid & in_block).sum()), int((valid & earlier).sum())
    invalid_fraction = float((in_block & ~valid).sum() / available) if available else None
    equality_fraction = float(equal_previous.sum() / paired.sum()) if paired.sum() else None
    failures = []
    if not identity_verified:
        failures.append('index_and_source_identity_not_verified')
    if target_count < MIN_TARGET_DAYS:
        failures.append('fewer_than_800_valid_primary_days')
    if context_count < MIN_CONTEXT_DAYS:
        failures.append('fewer_than_100_valid_preblock_days')
    if invalid_fraction is None or invalid_fraction > MAX_INVALID_FRACTION:
        failures.append('invalid_primary_OHLC_fraction_exceeds_0.01_or_empty')
    if equality_fraction is None or equality_fraction > MAX_EQUAL_PREVIOUS_CLOSE:
        failures.append('open_equals_previous_close_fraction_exceeds_0.20_or_unavailable')
    years = dates.dt.year.to_numpy()
    yearly = {}
    for year in sorted(set(years[in_block])):
        selected = paired & (years == year)
        yearly[str(year)] = dict(pairs=int(selected.sum()),
                                open_equals_previous_close_fraction=float(equal_previous[selected].mean()) if selected.any() else None)
    report = dict(market=market, eligible=not failures, failure_codes=failures,
                  primary_rows=available, valid_primary_rows=target_count, valid_preblock_rows=context_count,
                  invalid_primary_fraction=invalid_fraction, equality_denominator=int(paired.sum()),
                  open_equals_previous_close_fraction=equality_fraction,
                  open_equals_same_day_close_fraction=float(equal_same.sum() / target_count) if target_count else None,
                  yearly=yearly,
                  excluded_primary_dates=f.loc[in_block & ~valid, 'date'].tolist(),
                  interpretation='QC flag is not proof that the vendor fabricated or misreported opening prices')
    # Context and targets retain only valid observations; missing days stay absent.
    selected = f.loc[valid & (f.date <= END)].copy()
    selected['r'] = 100. * np.log(selected['close'] / selected['open'])
    if not np.isfinite(selected['r'].to_numpy()).all():
        raise FloatingPointError('nonfinite_open_to_close_return')
    selected['market'] = str(market)
    selected['is_primary'] = (selected.date >= START) & (selected.date <= END)
    return report, selected[['market', 'date', 'r', 'is_primary']]
