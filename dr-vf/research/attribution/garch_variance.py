"""Recover the rolling refit GARCH-t predictive variance so it can be scored on equal terms.

The saved `garch_t` column in the prior study is a native score: it uses GARCH-t's own fitted degrees
of freedom and its fitted mean. The attribution filters are scored zero-mean under a common Student-t8,
as the mechanism audit does for VolPFN and UniVol. Comparing the two therefore mixes variance quality
with density shape and mean handling, which is exactly what the common law is meant to remove.

This reruns the identical causal refit recursion from research.prior_adaptation.experiment, keeps the
predictive variance, and emits a `garch_common_t8` score built from GARCH-t's second moment about
zero (h + mu^2), matching how vol_common_t8 and uni_common_t8 were constructed. It verifies itself by
reproducing the saved native scores exactly; if it cannot, it refuses to write.

usage: python3 research/attribution/garch_variance.py
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

from research.prior_adaptation.experiment import DATA, GROUPS, load_market, save, sha
from research.prior_adaptation.model import log_law

NU_SCORE = 8.0


def garch_path(market):
    """The recursion of experiment.conventional_scores, retaining h and the fitted parameters.
    Any change here breaks the native-score check below, which is the point of that check."""
    from arch import arch_model
    z = market['z']; date = market['date']; scale = market['scale']
    native = np.full(len(z), np.nan); second = np.full(len(z), np.nan); fits = []
    origins = np.flatnonzero(date >= '2011-01-01')
    if not len(origins):
        raise ValueError('Missing evaluation dates')
    for start in range(int(origins[0]), len(z), 250):
        history = z[max(0, start - 1000):start]
        fit = arch_model(history, mean='Constant', vol='GARCH', p=1, q=1,
                         dist='t', rescale=False).fit(disp='off', show_warning=False)
        if int(fit.convergence_flag) != 0:
            raise RuntimeError('GARCH benchmark failed to converge; no selective date removal')
        p = fit.params; h = float(fit.conditional_volatility[-1] ** 2)
        fits.append({'origin': str(date[start]), 'nu': float(p['nu']), 'mu': float(p['mu'])})
        for t in range(start, min(start + 250, len(z))):
            h = float(p['omega'] + p['alpha[1]'] * (z[t - 1] - p['mu']) ** 2 + p['beta[1]'] * h)
            native[t] = log_law(np.array([z[t] - p['mu']]), np.array([h]),
                                np.array([p['nu']]))[0] - np.log(scale)
            second[t] = h + p['mu'] ** 2                        # second moment about zero, in z units
    common = np.full(len(z), np.nan); ok = np.isfinite(second)   # GARCH-t forecasts only from 2011
    common[ok] = log_law(z[ok], second[ok], np.full(int(ok.sum()), NU_SCORE)) - np.log(scale)
    return native, second, common, fits


def main(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    reference = pd.read_csv(args.reference, dtype={'date': str})
    hashes = {str(p): sha(Path(p)) for p in [__file__, 'research/prior_adaptation/experiment.py',
                                             args.reference]}
    save(root / 'protocol.json', {
        'scope': 'Recovers the existing GARCH-t benchmark predictive variance so it can be scored under '
                 'the same common Student-t8 law as the other methods. No new model, no tuning.',
        'inputs': hashes, 'nu_score': NU_SCORE,
        'construction': 'Identical causal refit recursion to experiment.conventional_scores; the common '
                        'score uses the second moment about zero, h + mu^2, matching the mechanism audit.'})
    started = time.perf_counter(); parts = []; checks = {}
    for name in sorted(n for g in GROUPS.values() for n in g):
        market = load_market(DATA / (name + '.npz'))
        native, second, common, fits = garch_path(market)
        rows = reference.loc[reference.market == name]
        idx = pd.Index(market['date']).get_indexer(rows.date)
        if np.any(idx < 0):
            raise RuntimeError(f'{name}: reference dates not found in the market calendar')
        err = float(np.max(np.abs(native[idx] - rows.garch_t.to_numpy())))
        # A genuine change to the recursion moves scores by order 1e-2; refitting the same model with a
        # different optimiser build moves them by order 1e-5. The tolerance separates the two.
        if not err < 1e-4:
            raise RuntimeError(f'{name}: native GARCH-t scores not reproduced (max abs error {err:.3e}); '
                               f'the recursion has drifted from the one that produced the saved column')
        checks[name] = {'native_max_abs_error': err,
                        'fitted_nu': [f['nu'] for f in fits], 'fitted_mu': [f['mu'] for f in fits]}
        parts.append(pd.DataFrame({'date': rows.date.to_numpy(), 'market': name,
                                   'garch_second_moment': second[idx] * market['scale'] ** 2,
                                   'garch_common_t8': common[idx]}))
        print(f'{name}: native scores reproduced (max abs error {err:.2e}); '
              f'fitted nu {np.min(checks[name]["fitted_nu"]):.1f}-{np.max(checks[name]["fitted_nu"]):.1f}',
              flush=True)
    frame = pd.concat(parts, ignore_index=True)
    frame.to_csv(root / 'garch_common.csv', index=False)
    merged = reference.merge(frame, on=['date', 'market'], how='inner', validate='one_to_one')
    if len(merged) != len(reference):
        raise RuntimeError('common-law GARCH-t scores did not align with the reference panel')
    gap = (merged.groupby('market').garch_common_t8.mean() - merged.groupby('market').garch_t.mean()).mean()
    nus = np.concatenate([c['fitted_nu'] for c in checks.values()])
    save(root / 'verification.json', {'markets': checks, 'rows': int(len(frame)),
                                      'mean_common_minus_native': float(gap),
                                      'fitted_nu_median': float(np.median(nus)),
                                      'fitted_nu_range': [float(nus.min()), float(nus.max())]})
    save(root / 'status.json', {'status': 'complete', 'elapsed_seconds': time.perf_counter() - started})
    print(f'\nCOMPLETE. Imposing t8 changes the GARCH-t score by {gap:+.6f} nats/return on average; '
          f'its own fitted nu has median {np.median(nus):.1f} (range {nus.min():.1f}-{nus.max():.1f}).',
          flush=True)
    print('A negative number means GARCH-t was being flattered by its own density shape in the '
          'earlier comparison, a positive one means it was being penalised.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', default='research/outputs/prior_adaptation_mechanism_v1/scores.csv')
    ap.add_argument('--out', default='research/outputs/attribution_garch_common_v1')
    main(ap.parse_args())
