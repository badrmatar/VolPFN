"""Synthetic-only distribution diagnostics, not market-based prior selection."""
import numpy as np
from . import simulators as S
from research.seed_replication import causal


def summary(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise FloatingPointError('nonfinite_diagnostic')
    return dict(mean=float(x.mean()), minimum=float(x.min()), maximum=float(x.max()),
                quantiles=dict(zip(('q05', 'q50', 'q95'), map(float, np.quantile(x, [.05, .5, .95])))))


def mean_acf(rows, lag):
    a, b = rows[:, :-lag], rows[:, lag:]
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    denominator = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = denominator > 0
    correlations = (a[ok] * b[ok]).sum(1) / denominator[ok]
    return dict(mean=float(correlations.mean()) if len(correlations) else None,
                usable_series=int(ok.sum()), constant_series=int((~ok).sum()))


def run(seed=99201, series=96, length=256):
    result = {'seed': seed, 'series_per_cell': series, 'length': length,
              'scope': 'synthetic descriptive checks; no market access or parameter tuning', 'cells': {}}
    for arm in S.ARMS:
        for clean in (True, False):
            raw = S.raw_batch(arm, seed, series, length, clean=clean)
            prepared = causal.prepare_batch(raw)
            r, lv, meta = raw['r'], raw['logv'], raw['meta']
            observed = prepared['target_mask'] > 0
            result['cells'][arm + ('_clean' if clean else '_overlaid')] = {
                'mean_return_square': summary((r * r).mean(1)),
                'log_variance_mean': summary(lv.mean(1)),
                'log_variance_sd': summary(lv.std(1)),
                'log_variance_increment_rms': summary(np.sqrt(np.mean(np.diff(lv, axis=1) ** 2, axis=1))),
                'log_variance_acf': {str(k): mean_acf(lv, k) for k in (1, 5, 20, 100) if k < length},
                'squared_return_acf': {str(k): mean_acf(r * r, k) for k in (1, 5, 20) if k < length},
                'absolute_normalized_target': summary(np.abs(prepared['target_r'][observed])),
                'zero_fraction': float((r == 0).mean()),
                'scored_fraction': float(raw['mask'].mean()),
                'gaussian_law_fraction': float(np.mean([m['nu'] >= 1e6 for m in meta])),
                'asymmetric_law_fraction': float(np.mean([m['gamma'] != 1 for m in meta])),
                'overlay_fractions': {k: float(np.mean([bool(m['overlay'][k]) for m in meta]))
                                     for k in ('jumps', 'ar', 'drift', 'stale')},
            }
    return result
