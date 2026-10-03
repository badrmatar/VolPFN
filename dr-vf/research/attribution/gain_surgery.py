"""Leg C steps 2-3: is the checkpoint's variance updating compressible into an explicit rule?

Stage 1 (distillation) regresses the network's own log-variance update g_t = log h_{t+1} - log h_t on
causal state variables, in nested specifications of increasing freedom, holding markets out by
geographic group. Stage 2 runs each fitted rule forward as a standalone recursive filter that never
touches the network, and scores it under the same common Student-t8 law the mechanism audit used, so
the answer is expressed in the units of the +0.01025356 advantage being explained.

Reported quantity: gain recovery, the share of that advantage the standalone filter retains on
held-out markets. Thresholds are fixed in research/attribution/PROTOCOL.md.

usage: python3 research/attribution/gain_surgery.py
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.prior_adaptation.experiment import DATA, GROUPS, load_market, paired_calendar_interval, save, sha
from research.prior_adaptation.model import log_law

HALF_LIVES = (5.0, 20.0, 100.0)          # the network's own EWMA channels
NU_SCORE = 8.0                           # common law for scoring, as in the mechanism audit
BURN = 500                               # EWMA transient; every evaluation date is far beyond it
ABS_CENTRE = float(np.sqrt(2.0 / np.pi))  # E|z| under a standard normal; the intercept absorbs the rest

# Every regressor the study may use, in a fixed order. Each specification selects a subset; the filter
# always evaluates the full vector with zeros in unused slots, so one tight recursion serves all three
# and the specifications are exactly nested by construction.
PARAM_NAMES = ['const', 'log_h', 'score', 'absz', 'signed', 'd1', 'd2', 'score_x_d1', 'score_x_d2']
SPECS = {
    'constant_gain':   ['const', 'log_h', 'score'],                        # GARCH / GAS form
    'asymmetric_gain': ['const', 'log_h', 'score', 'absz', 'signed'],      # Beta-t-EGARCH form
    'state_dep_gain':  PARAM_NAMES,                                        # the only non-standard one
}


def expand(beta, spec):
    full = np.zeros(len(PARAM_NAMES))
    for b, name in zip(beta, SPECS[spec]):
        full[PARAM_NAMES.index(name)] = b
    return full


def ewma_logs(r):
    """Causal log EWMA of r^2 at the network's three half-lives, matching volpfn.make_features: the
    value at t already includes r_t, and zero returns do not advance the average."""
    T = len(r); nz = r != 0.
    out = np.empty((T, len(HALF_LIVES)))
    for j, hl in enumerate(HALF_LIVES):
        lam = 0.5 ** (1.0 / hl); e = 1.0
        for t in range(T):
            if nz[t]:
                e = lam * e + (1 - lam) * r[t] * r[t]
            out[t, j] = e
    return np.log(out)


def raw_state(r):
    """Channel disagreements, unnormalised. Both are log ratios, so the return scale cancels."""
    lw = ewma_logs(r)
    return lw[:, 0] - lw[:, 1], lw[:, 1] - lw[:, 2]


def norm_from(frame):
    return {'d1_mean': float(frame.d1_raw.mean()), 'd1_sd': float(frame.d1_raw.std() + 1e-12),
            'd2_mean': float(frame.d2_raw.mean()), 'd2_sd': float(frame.d2_raw.std() + 1e-12),
            'logh_mean': float(frame.log_h.mean())}


def design_matrix(frame, spec, norm):
    log_h = frame.log_h.to_numpy() - norm['logh_mean']; r = frame.r.to_numpy()
    d1 = (frame.d1_raw.to_numpy() - norm['d1_mean']) / norm['d1_sd']
    d2 = (frame.d2_raw.to_numpy() - norm['d2_mean']) / norm['d2_sd']
    z2 = r * r / np.exp(frame.log_h.to_numpy()); az = np.sqrt(np.maximum(z2, 0.)); score = z2 - 1.
    cols = {'const': np.ones_like(z2), 'log_h': log_h, 'score': score,
            'absz': az - ABS_CENTRE, 'signed': np.sign(r) * np.minimum(az, 5.),
            'd1': d1, 'd2': d2, 'score_x_d1': score * d1, 'score_x_d2': score * d2}
    return np.column_stack([cols[c] for c in SPECS[spec]])


def run_filter(r2, absr, sgn, d1, d2, full, logh_mean, log_h0):
    """The fitted rule as a self-contained recursive variance filter reading only returns. A tight
    scalar loop: the likelihood optimiser calls this thousands of times."""
    T = len(r2); log_h = np.empty(T); lh = log_h0
    b0, b1, b2, b3, b4, b5, b6, b7, b8 = full
    for t in range(T):
        log_h[t] = lh
        if t + 1 == T:
            break
        inv = np.exp(-lh); z2 = r2[t] * inv; az = absr[t] * np.sqrt(inv); score = z2 - 1.
        g = (b0 + b1 * (lh - logh_mean) + b2 * score + b3 * (az - ABS_CENTRE)
             + b4 * sgn[t] * (az if az < 5. else 5.) + b5 * d1[t] + b6 * d2[t]
             + b7 * score * d1[t] + b8 * score * d2[t])
        g = -3. if g < -3. else (3. if g > 3. else g)
        lh = lh + g
        lh = -20. if lh < -20. else (20. if lh > 20. else lh)
    return np.exp(log_h)


def run_filter_batch(R2, ABSR, SGN, D1, D2, valid, full, logh_mean, lh0):
    """The same recursion advanced over many series at once. Series are padded to a common length and
    masked; padding cannot affect a real series because each row's own state is carried independently.
    Used inside the optimiser, where the per-step Python overhead of one series at a time dominates."""
    n, T = R2.shape; LH = np.empty((n, T)); lh = np.asarray(lh0, float).copy()
    b0, b1, b2, b3, b4, b5, b6, b7, b8 = full
    for t in range(T):
        LH[:, t] = lh
        if t + 1 == T:
            break
        inv = np.exp(-lh); z2 = R2[:, t] * inv; az = np.sqrt(np.abs(ABSR[:, t]) ** 2 * inv)
        score = z2 - 1.
        g = (b0 + b1 * (lh - logh_mean) + b2 * score + b3 * (az - ABS_CENTRE)
             + b4 * SGN[:, t] * np.minimum(az, 5.) + b5 * D1[:, t] + b6 * D2[:, t]
             + b7 * score * D1[:, t] + b8 * score * D2[:, t])
        lh = np.clip(lh + np.clip(g, -3., 3.), -20., 20.)
        lh = np.where(valid[:, t], lh, LH[:, t])          # frozen past each series' own end
    return np.exp(LH)


def stack(prepared):
    """Pad a list of prepared series into rectangular arrays for the batched recursion."""
    T = max(len(p[2]) for p in prepared); n = len(prepared)
    R2 = np.zeros((n, T)); ABSR = np.zeros((n, T)); SGN = np.zeros((n, T))
    D1 = np.zeros((n, T)); D2 = np.zeros((n, T)); valid = np.zeros((n, T), bool)
    keep = np.zeros((n, T), bool); Y = np.zeros((n, T)); lh0 = np.zeros(n)
    NETLH = np.zeros((n, T)); netok = np.zeros((n, T), bool)
    for i, (r, k, r2, absr, sgn, d1, d2, l0, netlh) in enumerate(prepared):
        m = len(r2)
        R2[i, :m] = r2; ABSR[i, :m] = absr; SGN[i, :m] = sgn
        D1[i, :m] = d1; D2[i, :m] = d2; valid[i, :m] = True
        keep[i, :m] = k; Y[i, :m] = r; lh0[i] = l0
        good = np.isfinite(netlh)
        NETLH[i, :m] = np.where(good, netlh, 0.); netok[i, :m] = k & good
    return dict(R2=R2, ABSR=ABSR, SGN=SGN, D1=D1, D2=D2, valid=valid, keep=keep, Y=Y, lh0=lh0,
                NETLH=NETLH, netok=netok)


def prepare(market, norm, keep, net_h=None):
    """Precompute everything the recursion needs that does not depend on the running variance.
    net_h is the network's own predictive variance, carried so a free-running fit can target it."""
    r = market['r']; d1_raw, d2_raw = raw_state(r)
    d1 = (d1_raw - norm['d1_mean']) / norm['d1_sd']; d2 = (d2_raw - norm['d2_mean']) / norm['d2_sd']
    lh0 = float(np.log(max(np.mean(r[:100] ** 2), 1e-8)))
    if net_h is None:
        netlh = np.full(len(r), np.nan)
    else:
        if len(net_h) != len(r):
            raise ValueError('network variance path does not align with the market returns')
        netlh = np.where(np.isfinite(net_h) & (net_h > 0), np.log(np.maximum(net_h, 1e-300)), np.nan)
    return (r, keep, r * r, np.abs(r), np.sign(r), d1, d2, lh0, netlh)


def negloglik(beta, packed, spec, logh_mean):
    """Mean negative log score of the standalone filter over the packed fitting series."""
    full = expand(beta, spec)
    H = run_filter_batch(packed['R2'], packed['ABSR'], packed['SGN'], packed['D1'], packed['D2'],
                         packed['valid'], full, logh_mean, packed['lh0'])
    k = packed['keep']
    if not np.all(np.isfinite(H[k])) or np.any(H[k] <= 0):
        return 1e6
    return -float(np.sum(log_law(packed['Y'][k], H[k], NU_SCORE))) / max(int(k.sum()), 1)


def freerun_loss(beta, packed, spec, logh_mean):
    """Mean squared mismatch between the filter's free-running log-variance path and the network's.

    The ordinary distilled fit matches one-step updates while sitting on the network's own trajectory.
    That is teacher forcing, and it does not bound the error of the filter once it runs on its own
    trajectory for thousands of steps: a specification can match updates well and still drift. This
    objective asks the question leg C is actually about, whether the network's variance path is
    reproducible by the rule, and it uses no market outcome, so it stays a distillation."""
    full = expand(beta, spec)
    H = run_filter_batch(packed['R2'], packed['ABSR'], packed['SGN'], packed['D1'], packed['D2'],
                         packed['valid'], full, logh_mean, packed['lh0'])
    k = packed['netok']
    if not np.all(np.isfinite(H[k])) or np.any(H[k] <= 0):
        return 1e6
    return float(np.mean((np.log(H[k]) - packed['NETLH'][k]) ** 2))


def build_frame(extract_root, mask_by_market):
    """One row per usable transition: same sliding window at both endpoints, forecast present at both,
    past the EWMA transient. Window-boundary transitions are excluded because there the change in
    log h reflects a context restart rather than a response to the shock.

    The scored-date mask is per market. The reference panel does not score every market on every date
    (906 to 960 dates each against a union of 1014), so a shared mask would silently claim forecasts
    on dates a market was never scored on."""
    parts = []
    for group, names in GROUPS.items():
        for name in sorted(names):
            a = np.load(Path(extract_root) / f'{name}.npz', allow_pickle=False)
            date, r, h, owner = a['date'].astype('U10'), a['r'], a['h'], a['owner']
            d1_raw, d2_raw = raw_state(r)
            idx = np.arange(len(r) - 1)
            usable = ((owner[:-1] >= 0) & (owner[1:] >= 0) & (owner[:-1] == owner[1:])
                      & (idx >= BURN) & np.isfinite(h[:-1]) & np.isfinite(h[1:]) & (h[:-1] > 0))
            k = np.flatnonzero(usable)
            parts.append(pd.DataFrame({
                'market': name, 'group': group, 'date': date[k], 'next_date': date[k + 1],
                'in_mask': np.isin(date[k + 1], mask_by_market[name]), 'r': r[k], 'log_h': np.log(h[k]),
                'd1_raw': d1_raw[k], 'd2_raw': d2_raw[k], 'g': np.log(h[k + 1]) - np.log(h[k])}))
    return pd.concat(parts, ignore_index=True)


def main(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    reference = pd.read_csv(args.reference, dtype={'date': str})
    mask_by_market = {m: np.sort(g.date.unique()) for m, g in reference.groupby('market')}
    hashes = {str(p): sha(Path(p)) for p in [__file__, 'research/attribution/PROTOCOL.md',
                                             'research/attribution/extract.py', args.reference]}
    save(root / 'protocol.json', {
        'scope': 'Attribution of the saved checkpoint advantage on development data. No training, no '
                 'checkpoint selection, no reserved outcomes opened.',
        'inputs': hashes, 'specifications': SPECS, 'nu_score': NU_SCORE, 'burn': BURN,
        'fit_window': 'pre-2011 history of the fitting groups; evaluation on held-out groups 2011-2014',
        'thresholds': {'compressible': .50, 'partially_compressible': .30}})
    started = time.perf_counter()
    frame = build_frame(args.extract, mask_by_market)
    print(f'{len(frame)} usable transitions over {frame.market.nunique()} markets; '
          f'{int(frame.in_mask.sum())} on scored dates', flush=True)
    markets = {n: load_market(DATA / (n + '.npz')) for n in sorted(frame.market.unique())}
    net_h = {}
    for n in markets:
        a = np.load(Path(args.extract) / f'{n}.npz', allow_pickle=False)
        if not np.array_equal(a['date'].astype('U10'), markets[n]['date']):
            raise RuntimeError(f'{n}: extracted calendar does not match the market calendar')
        net_h[n] = a['h']
    denominator = float((reference.groupby('market').vol_common_t8.mean()
                         - reference.groupby('market').uni_common_t8.mean()).mean())

    rows = []; fold_detail = {}
    for held_out in GROUPS:
        train = frame[(frame.group != held_out) & (frame.next_date < '2011-01-01')]
        held = frame[(frame.group == held_out) & frame.in_mask]
        norm = norm_from(train)
        fit_packed = stack([prepare(markets[n], norm,
                                    (markets[n]['date'] < '2011-01-01')
                                    & (np.arange(len(markets[n]['r'])) >= BURN), net_h[n])
                            for g, ns in GROUPS.items() if g != held_out for n in sorted(ns)])
        eval_series = [(n, prepare(markets[n], norm, np.isin(markets[n]['date'], mask_by_market[n]),
                                   net_h[n]))
                       for n in sorted(GROUPS[held_out])]
        detail = {}
        for spec in SPECS:
            X = design_matrix(train, spec, norm); y = train.g.to_numpy()
            beta_d, *_ = np.linalg.lstsq(X, y, rcond=None)
            in_r2 = 1 - np.var(y - X @ beta_d) / np.var(y)
            Xh = design_matrix(held, spec, norm); yh = held.g.to_numpy()
            out_r2 = 1 - np.var(yh - Xh @ beta_d) / np.var(yh)
            if args.skip_likelihood:
                res = argparse.Namespace(x=np.asarray(beta_d, float), success=False, fun=None, nfev=0)
                fre = argparse.Namespace(x=np.asarray(beta_d, float), success=False, fun=None, nfev=0)
            else:
                # Bound the search by function evaluations, not outer iterations: one Powell iteration
                # is ~9 line searches and can cost thousands of evaluations, which makes wall time
                # unpredictable. The start is the ordinary least-squares fit, so a modest budget is
                # enough; whether it was is recorded below rather than assumed.
                opts = {'maxfev': args.maxfev, 'xtol': 1e-3, 'ftol': 1e-6}
                res = minimize(negloglik, np.asarray(beta_d, float),
                               args=(fit_packed, spec, norm['logh_mean']), method='Powell', options=opts)
                fre = minimize(freerun_loss, np.asarray(beta_d, float),
                               args=(fit_packed, spec, norm['logh_mean']), method='Powell', options=opts)
            for label, beta in (('distilled', beta_d), ('freerun', fre.x), ('likelihood', res.x)):
                full = expand(beta, spec)
                for name, (r, keep, r2, absr, sgn, d1, d2, lh0, _netlh) in eval_series:
                    h = run_filter(r2, absr, sgn, d1, d2, full, norm['logh_mean'], lh0)
                    rows.append(pd.DataFrame({'date': markets[name]['date'][keep], 'market': name,
                                              'group': held_out, 'spec': spec, 'variant': label,
                                              'filter_common_t8': log_law(r[keep], h[keep], NU_SCORE)}))
            detail[spec] = {'free_parameters': len(SPECS[spec]),
                            'coefficients_distilled': dict(zip(SPECS[spec], np.asarray(beta_d).tolist())),
                            'coefficients_likelihood': dict(zip(SPECS[spec], np.asarray(res.x).tolist())),
                            'coefficients_freerun': dict(zip(SPECS[spec], np.asarray(fre.x).tolist())),
                            'freerun_converged': bool(fre.success),
                            'freerun_mean_squared_log_gap': None if fre.fun is None else float(fre.fun),
                            'freerun_mean_squared_log_gap_at_distilled':
                                float(freerun_loss(np.asarray(beta_d, float), fit_packed, spec,
                                                   norm['logh_mean'])),
                            'distillation_r2_in_fold': float(in_r2), 'distillation_r2_held_out': float(out_r2),
                            'likelihood_converged': bool(res.success),
                            'likelihood_evaluations': int(getattr(res, 'nfev', 0)),
                            'freerun_evaluations': int(getattr(fre, 'nfev', 0)),
                            'likelihood_mean_nll': None if res.fun is None else float(res.fun),
                            'normalisation': norm}
            print(f'  {held_out:13s} {spec:16s} distillation R2  in-fold {in_r2:6.3f}  held-out {out_r2:6.3f}',
                  flush=True)
        fold_detail[held_out] = detail

    scored = pd.concat(rows, ignore_index=True)
    merged = scored.merge(reference[['date', 'market', 'vol_common_t8', 'uni_common_t8', 'garch_t']],
                          on=['date', 'market'], how='inner', validate='many_to_one')
    if len(merged) != len(scored):
        raise RuntimeError('scored dates did not align one-to-one with the reference mask')
    # The saved garch_t column is a native score: GARCH-t's own fitted degrees of freedom and fitted
    # mean. Comparing it against a zero-mean t8 filter mixes variance quality with density shape, so
    # attach the common-law version when garch_variance.py has produced it.
    gpath = Path(args.garch_common)
    if gpath.exists():
        gframe = pd.read_csv(gpath, dtype={'date': str})[['date', 'market', 'garch_common_t8']]
        before = len(merged)
        merged = merged.merge(gframe, on=['date', 'market'], how='inner', validate='many_to_one')
        if len(merged) != before:
            raise RuntimeError('common-law GARCH-t scores did not cover the scored panel')
    else:
        print(f'note: {gpath} absent, so the GARCH-t comparison stays on the native score and is not '
              f'like-for-like. Run research/attribution/garch_variance.py to fix that.', flush=True)
    summary = {}
    print('')
    for (spec, variant), block in merged.groupby(['spec', 'variant']):
        vs_univol = paired_calendar_interval(block, 'filter_common_t8', 'uni_common_t8')
        vs_garch = paired_calendar_interval(block, 'filter_common_t8', 'garch_t')
        vs_net = paired_calendar_interval(block, 'filter_common_t8', 'vol_common_t8')
        recovery = vs_univol['mean_difference'] / denominator
        entry = {'gain_recovery': recovery, 'versus_univol_common_t8': vs_univol,
                 'versus_garch_t_native': vs_garch, 'versus_network_common_t8': vs_net,
                 'free_parameters': len(SPECS[spec])}
        if 'garch_common_t8' in block.columns:
            entry['versus_garch_common_t8'] = paired_calendar_interval(block, 'filter_common_t8',
                                                                       'garch_common_t8')
        summary[f'{spec}|{variant}'] = entry
        g = entry.get('versus_garch_common_t8', vs_garch)
        tag = 'GARCH-t(t8)' if 'versus_garch_common_t8' in entry else 'GARCH-t(native)'
        print(f'  {spec:16s} {variant:10s} recovery {recovery:+7.1%}   vs UniVol '
              f'{vs_univol["mean_difference"]:+.6f} ({vs_univol["markets_improved"]}/28)   '
              f'vs {tag} {g["mean_difference"]:+.6f} ({g["markets_improved"]}/28)', flush=True)
    merged.to_csv(root / 'scores.csv', index=False)
    save(root / 'folds.json', fold_detail)
    save(root / 'summary.json', {'denominator_checkpoint_advantage': denominator, 'results': summary,
                                 'elapsed_seconds': time.perf_counter() - started})
    best_key, best = max(summary.items(), key=lambda kv: kv[1]['gain_recovery'])
    verdict = ('compressible' if best['gain_recovery'] >= .50 else
               'partially compressible' if best['gain_recovery'] >= .30 else 'not compressible')
    save(root / 'status.json', {'status': 'complete', 'best': best_key, 'verdict': verdict,
                                'best_recovery': best['gain_recovery']})
    print(f'\nVERDICT: {verdict}. Best is {best_key}, recovering {best["gain_recovery"]:.1%} of the '
          f'{denominator:.8f} nat/return advantage.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extract', default='research/outputs/attribution_extract_v1')
    ap.add_argument('--reference', default='research/outputs/prior_adaptation_mechanism_v1/scores.csv')
    ap.add_argument('--out', default='research/outputs/attribution_gain_v1')
    ap.add_argument('--garch-common', default='research/outputs/attribution_garch_common_v1/garch_common.csv',
                    help='like-for-like GARCH-t scores from garch_variance.py; skipped if absent')
    ap.add_argument('--maxfev', type=int, default=800,
                    help='objective evaluations per optimisation; bounds wall time predictably')
    ap.add_argument('--skip-likelihood', action='store_true',
                    help='report the distilled filter only; skips the maximum-likelihood refit')
    main(ap.parse_args())
