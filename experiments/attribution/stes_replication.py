"""Study 3 of PROTOCOL_AMENDMENT_20260907.md: Ho et al. (2022) STES gates on the 28-index panel.

Taylor's smooth transition exponential smoothing, h_{t+1} = a_t r_t^2 + (1 - a_t) h_t with
a_t = 1 / (1 + exp(b + sum_i g_i V_{i,t})), is fitted per market on the market's own pre-2011 history
and run forward over the 2011-2014 mask, for every transition set of Ho et al.'s Table 1 that daily
returns can reproduce, plus three sets expressing the present paper's driver inside their family.
Each set is fitted under two criteria (their eq. 11, the sum of squared errors of h against r^2;
and the common Student-t8 log score) and evaluated under their metrics (MAE, MSE/RMSE of h against
the r^2 proxy) and ours (common-t8 log score, QLIKE). Rules S1-S3 of the amendment are evaluated.

usage: PYTHONPATH=. python3 experiments/attribution/stes_replication.py [--workers 4]
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from experiments.prior_adaptation.experiment import GROUPS, paired_calendar_interval, save, sha
from experiments.prior_adaptation.model import log_law
from experiments.attribution.gain_surgery import NU_SCORE
import experiments.attribution.drivers as D
import experiments.prior_adaptation.experiment as _E

FIT_END = '2011-01-01'; BURN = 100
# Transition sets: name -> variables. e, ae, se are the current shock (sign, size, square); rvw and
# rvm are the 5- and 22-day means of r^2 ending at t (Ho's weekly and monthly realised variance with
# r^2 as the daily measure); ls_5_22 and lrvm are their log slope and log level; d2 is the
# checkpoint's own 20/100-day log-EWMA slope.
TRANSITIONS = {
    'ewma': [], 'E': ['e'], 'AE': ['ae'], 'SE': ['se'], 'EAE': ['e', 'ae'], 'ESE': ['e', 'se'],
    'RVw': ['rvw'], 'RVm': ['rvm'], 'RVdw': ['se', 'rvw'], 'RVdm': ['se', 'rvm'], 'RVwm': ['rvw', 'rvm'],
    'RVdwm': ['se', 'rvw', 'rvm'],
    'slope_5_22': ['ls_5_22'], 'slope_20_100': ['d2'], 'level_22': ['lrvm'],
    # added 7 Sep 2026 after the first run (attribution_stes_v1), to separate the log transform, the
    # horizon and the slope constraint: additive log levels nest the corresponding slope exactly.
    'loglevels_5_22': ['lrvw', 'lrvm'], 'loglevels_20_100': ['l20', 'l100'], 'level_100': ['l100'],
}
CRITERIA = ('mse', 'loglik')
COMPARISONS = [('RVdwm', 'SE', 'S1/S2'), ('slope_20_100', 'SE', 'S2'), ('slope_5_22', 'SE', 'S2'), ('RVwm', 'SE', 'S2'),
               ('RVdwm', 'ewma', 'context'), ('SE', 'ewma', 'context'), ('slope_20_100', 'level_22', 'slope_vs_level'),
               ('slope_5_22', 'level_22', 'slope_vs_level'), ('slope_20_100', 'RVdwm', 'context'),
               ('loglevels_20_100', 'slope_20_100', 'nesting'), ('loglevels_5_22', 'slope_5_22', 'nesting'),
               ('slope_20_100', 'level_100', 'slope_vs_level'), ('slope_20_100', 'slope_5_22', 'horizon'),
               ('loglevels_20_100', 'RVdwm', 'log_vs_raw'), ('level_100', 'level_22', 'horizon')]
METRICS = ['logscore', 'neg_mae', 'neg_mse', 'neg_qlike']       # all oriented so that higher is better


def rolling_mean(x, w):
    c = np.cumsum(np.insert(x, 0, 0.)); out = np.empty(len(x))
    for t in range(len(x)):
        lo = max(0, t + 1 - w); out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def variables(r):
    r2 = r * r; rvw = rolling_mean(r2, 5); rvm = rolling_mean(r2, 22)
    lw = D.ewma_logs(r, (20., 100.))
    return {'e': r, 'ae': np.abs(r), 'se': r2, 'rvw': rvw, 'rvm': rvm,
            'ls_5_22': np.log(rvw + 1e-12) - np.log(rvm + 1e-12), 'lrvw': np.log(rvw + 1e-12), 'lrvm': np.log(rvm + 1e-12),
            'd2': lw[:, 0] - lw[:, 1], 'l20': lw[:, 0], 'l100': lw[:, 1]}


def stes_path(r2, alpha, h0):
    """h_{t+1} = alpha_t r_t^2 + (1 - alpha_t) h_t, started at h0; returns h_t for every t."""
    T = len(r2); h = np.empty(T); x = float(h0)
    a = alpha.tolist(); q = r2.tolist()
    for t in range(T):
        h[t] = x
        x = a[t] * q[t] + (1. - a[t]) * x
    return h


def alpha_of(theta, V):
    if V.shape[1] == 0:
        return np.full(V.shape[0], expit(-theta[0]))
    return expit(-(theta[0] + V @ np.asarray(theta[1:])))


def objective(theta, r, r2, V, h0, fit, criterion):
    h = stes_path(r2, alpha_of(theta, V), h0)
    hf = h[fit]
    if not np.all(np.isfinite(hf)) or np.any(hf <= 0):
        return 1e12
    if criterion == 'mse':
        return float(np.mean((r2[fit] - hf) ** 2))
    return -float(np.mean(log_law(r[fit], hf, NU_SCORE)))


def fit_market(task):
    """All transition sets and both criteria for one market; returns scored rows and fit records."""
    name, r, date, mask_dates, maxfev = task
    r2 = r * r; T = len(r); V_all = variables(r)
    fit = (date < FIT_END) & (np.arange(T) >= BURN); ev = np.isin(date, mask_dates)
    h0 = float(max(np.mean(r2[:BURN]), 1e-8))
    rows = []; fits = {}
    for set_name, vars_ in TRANSITIONS.items():
        V = np.column_stack([V_all[v] for v in vars_]) if vars_ else np.zeros((T, 0))
        if V.shape[1]:
            mu = V[fit].mean(axis=0); sd = V[fit].std(axis=0); sd = np.where(sd < 1e-12, 1., sd); V = (V - mu) / sd
        for criterion in CRITERIA:
            best = None
            for a0 in (0.06, 0.20):
                x0 = np.concatenate([[math.log(1. / a0 - 1.)], np.zeros(V.shape[1])])
                res = minimize(objective, x0, args=(r, r2, V, h0, fit, criterion), method='Powell',
                               options={'maxfev': maxfev, 'xtol': 1e-3, 'ftol': 1e-9})
                if best is None or res.fun < best.fun:
                    best = res
            h = stes_path(r2, alpha_of(best.x, V), h0)
            k = ev & np.isfinite(h) & (h > 0)
            rows.append(pd.DataFrame({'date': date[k], 'market': name, 'set': set_name, 'criterion': criterion,
                                      'logscore': log_law(r[k], h[k], NU_SCORE), 'neg_mae': -np.abs(r2[k] - h[k]),
                                      'neg_mse': -(r2[k] - h[k]) ** 2,
                                      'neg_qlike': -(r2[k] / h[k] - np.log(np.maximum(r2[k], 1e-12) / h[k]) - 1.)}))
            fits[f'{set_name}|{criterion}'] = {'theta': best.x.tolist(), 'objective': float(best.fun), 'converged': bool(best.success),
                                              'evaluations': int(best.nfev), 'mean_alpha_fit': float(alpha_of(best.x, V)[fit].mean()),
                                              'variables': vars_}
    return name, pd.concat(rows, ignore_index=True), fits


def decide(scored, log):
    """Market means and the pre-registered contrasts, per criterion and metric."""
    means = scored.groupby(['set', 'criterion', 'market'])[METRICS].mean().groupby(['set', 'criterion']).mean()
    comparisons = {}
    for criterion in CRITERIA:
        for metric in METRICS:
            block = scored[scored.criterion == criterion]
            w = block.pivot_table(index=['date', 'market'], columns='set', values=metric).reset_index(); w.columns.name = None
            for cand, comp, family in COMPARISONS:
                if cand in w.columns and comp in w.columns:
                    comparisons[f'{criterion}|{metric}|{cand}-{comp}'] = dict(D.paired_block(w, cand, comp), family=family,
                                                                             candidate=cand, comparator=comp,
                                                                             criterion=criterion, metric=metric)
    c = comparisons.get
    rules = {}
    s1 = c('mse|neg_mae|RVdwm-SE')
    if s1:
        covers0 = s1['calendar_block_interval95'][0] <= 0 <= s1['calendar_block_interval95'][1]
        rules['S1_their_null_replicates'] = {'pass': bool(covers0 or s1['mean_difference'] <= 0), 'RVdwm_minus_SE_mse_mae': s1,
                                            'RVdwm_minus_SE_mse_mse': c('mse|neg_mse|RVdwm-SE')}
    s2a = c('loglik|logscore|RVdwm-SE'); s2b = c('loglik|logscore|slope_20_100-SE')
    if s2a and s2b:
        ok = lambda x: x['mean_difference'] > 0 and x['calendar_block_interval95'][0] > 0 and x['markets_improved'] >= 17
        rules['S2_load_bearing_under_log_score'] = {'pass': bool(ok(s2a) and ok(s2b)), 'RVdwm_minus_SE': s2a, 'slope_20_100_minus_SE': s2b}
    rules['S3_cross'] = {'mse_fitted_under_logscore': {k: v for k, v in comparisons.items() if k.startswith('mse|logscore|')},
                         'loglik_fitted_under_mae': {k: v for k, v in comparisons.items() if k.startswith('loglik|neg_mae|')}}
    sl = c('loglik|logscore|slope_20_100-level_22')
    if sl:
        rules['slope_vs_level_within_STES'] = {'pass': bool(sl['mean_difference'] > 0 and sl['calendar_block_interval95'][0] > 0), 'detail': sl}
    correction = ('S1_their_null_replicates' in rules and rules['S1_their_null_replicates']['pass']
                  and 'S2_load_bearing_under_log_score' in rules and rules['S2_load_bearing_under_log_score']['pass'])
    verdict = ('correction licensed: the loss function, not the mechanism, produced the earlier negative result' if correction else
               'correction not licensed' + ('' if 'S1_their_null_replicates' not in rules else
                                            (' (their null did not replicate on this panel)' if not rules['S1_their_null_replicates']['pass'] else
                                             ' (the gate is not load-bearing under the log score here)')))
    log('')
    for k, v in rules.items():
        log(f'  {k:36s} {"PASS" if v.get("pass") else ("FAIL" if "pass" in v else "-")}')
    log(f'\nVERDICT: {verdict}')
    return {'market_means': {f'{s}|{cr}': row.to_dict() for (s, cr), row in means.iterrows()}, 'comparisons': comparisons,
            'rules': rules, 'verdict': verdict}


def write_md(root, result):
    L = ['# STES replication (Ho et al. 2022) on the 28-index development panel', '',
         '## Market-mean metrics by transition set and criterion (higher is better; MAE/MSE/QLIKE negated)', '',
         '| set | criterion | log score (t8) | -MAE | -MSE | -QLIKE |', '|---|---|---:|---:|---:|---:|']
    for k, v in result['market_means'].items():
        s, cr = k.split('|')
        L.append(f'| {s} | {cr} | {v["logscore"]:+.6f} | {v["neg_mae"]:+.4f} | {v["neg_mse"]:+.3f} | {v["neg_qlike"]:+.5f} |')
    L += ['', '## Pre-registered contrasts', '', '| family | criterion | metric | candidate - comparator | difference | 95% block interval | markets | boot p |',
          '|---|---|---|---|---:|---|---:|---:|']
    for k, v in result['comparisons'].items():
        if v['family'] == 'context' and v['metric'] not in ('logscore', 'neg_mae'):
            continue
        L.append(f'| {v["family"]} | {v["criterion"]} | {v["metric"]} | {v["candidate"]} - {v["comparator"]} | {v["mean_difference"]:+.6f} | '
                 f'[{v["calendar_block_interval95"][0]:+.6f}, {v["calendar_block_interval95"][1]:+.6f}] | {v["markets_improved"]}/{v["markets"]} | '
                 f'{v["bootstrap_p_two_sided"]:.3f} |')
    L += ['', '## Rules', '']
    for k, v in result['rules'].items():
        L.append(f'* {k}: **{"PASS" if v.get("pass") else ("FAIL" if "pass" in v else "descriptive")}**')
    L += ['', f'**Verdict:** {result["verdict"]}', '']
    (Path(root) / 'SUMMARY.md').write_text('\n'.join(L) + '\n')


def main(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
    logf = open(root / 'run.log', 'a')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    reference = pd.read_csv(args.reference, dtype={'date': str})
    mask = {m: np.sort(g.date.unique()) for m, g in reference.groupby('market')}
    here = Path(__file__).resolve().parent
    save(root / 'protocol.json', {
        'scope': 'Study 3 of the 7 September amendment: STES transition sets of Ho et al. (2022) refitted per market on the '
                 'development panel under two criteria and four metrics. No reserved outcomes.',
        'inputs': {str(p): sha(Path(p)) for p in [__file__, here / 'drivers.py', here / 'PROTOCOL_AMENDMENT_20260907.md', args.reference]},
        'transitions': TRANSITIONS, 'criteria': CRITERIA, 'metrics': METRICS, 'comparisons': COMPARISONS,
        'fit': f'per market, dates < {FIT_END}, after {BURN} initial observations; h0 = mean r^2 of the first {BURN}',
        'maxfev': args.maxfev, 'nu_score': NU_SCORE})
    started = time.perf_counter()
    names = sorted(n for g in GROUPS.values() for n in g)
    if args.markets:
        names = [n for n in names if n in args.markets.split(',')]
    tasks = []
    for n in names:
        m = D.load_market(_E.DATA / (n + '.npz'))
        tasks.append((n, m['r'], m['date'], mask[n], args.maxfev))
    log(f'{len(tasks)} markets x {len(TRANSITIONS)} transition sets x {len(CRITERIA)} criteria; {args.workers} workers')
    parts = []; fits = {}
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for name, rows, f in pool.imap_unordered(fit_market, tasks):
                parts.append(rows); fits[name] = f; log(f'  {name}: done ({(time.perf_counter() - started) / 60:.1f} min)')
    else:
        for task in tasks:
            name, rows, f = fit_market(task); parts.append(rows); fits[name] = f
            log(f'  {name}: done ({(time.perf_counter() - started) / 60:.1f} min)')
    scored = pd.concat(parts, ignore_index=True)
    merged = scored.merge(reference[['date', 'market', 'uni_common_t8', 'vol_common_t8']], on=['date', 'market'], how='inner')
    scored.to_csv(root / 'scores.csv', index=False); save(root / 'fits.json', fits)
    result = decide(scored, log)
    for criterion in CRITERIA:
        for s in ('SE', 'RVdwm', 'slope_20_100', 'ewma'):
            b = merged[(merged.set == s) & (merged.criterion == criterion)]
            if len(b):
                result.setdefault('versus_univol_and_network', {})[f'{s}|{criterion}'] = {
                    'vs_univol_common_t8': paired_calendar_interval(b, 'logscore', 'uni_common_t8'),
                    'vs_network_common_t8': paired_calendar_interval(b, 'logscore', 'vol_common_t8')}
    save(root / 'summary.json', dict(result, elapsed_seconds=time.perf_counter() - started))
    save(root / 'decisions.json', {'rules': result['rules'], 'verdict': result['verdict']})
    write_md(root, result)
    save(root / 'status.json', {'status': 'complete', 'verdict': result['verdict'], 'elapsed_seconds': time.perf_counter() - started})
    log(f'done in {(time.perf_counter() - started) / 60:.1f} min; outputs in {root}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', default='experiments/outputs/prior_adaptation_mechanism_v1/scores.csv')
    ap.add_argument('--out', default='experiments/outputs/attribution_stes_v1')
    ap.add_argument('--maxfev', type=int, default=1500)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--markets', default=None, help='comma list to restrict (smoke tests)')
    main(ap.parse_args())
