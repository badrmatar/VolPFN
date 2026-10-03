"""One-shot reserved-block confirmation of the dissertation's headline claim (pre-registered here).

Claim on the development panel (2011-2014): the prior-fitted network beats refit GARCH-t by
+0.0171 nats/return (28/28 markets), UniVol by +0.0121 (27/28), and GARCH-t under a common
Student-t8 law by +0.0103. This script scores the same comparisons ONCE on the reserved block
(split == 2: 2015-01 to 2022-02, including March 2020) of the 28 development markets, with the
network's forecasts from experiments/outputs/attribution_extract_full_v1 (base checkpoint, no
training, no tuning), refit GARCH-t computed with the same code path as the development study
(arch, 1,000-day history, refits every 250 days, own Student-t law), UniVol run over the full
series (numpy filter), and the block bootstrap of the development study.

Rules (frozen before running; predictions in protocol.json):
  C1  network (own law) - refit GARCH-t (own law):  mean > 0, bootstrap p < 0.05, >= 20/28 markets
  C2  network (own law) - UniVol (own law):          same thresholds
  C3  network (own law) - EWMA(0.94)-t8:            same thresholds (plain RiskMetrics comparator)
  C4  network (t8 law on its h) - GARCH-t (t8 law on its h): same thresholds (scale contribution)
Run:  PYTHONPATH=. python3 experiments/attribution/reserved_confirmation.py
Disclosure: before freezing, the pipeline was exercised on SPX alone (network, UniVol and EWMA
reserved means printed once, no GARCH); no rule or threshold was changed afterwards.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.prior_adaptation.experiment import GROUPS, paired_calendar_interval, save, sha
from experiments.prior_adaptation.model import log_law
from experiments.baselines.univol import run_series
from volpfn.prior import initial_scale          # the development study's scale convention (experiment.load_market)

DATA = Path('experiments/data/oxfordman_barde/calendar_full_v1')
EXTRACT = Path('experiments/outputs/attribution_extract_full_v1')
RULES = {'C1': ('network', 'garch_t'), 'C2': ('network', 'univol'), 'C3': ('network', 'ewma_t8'),
         'C4': ('network_common_t8', 'garch_common_t8')}
PREDICTIONS = {'C1': 'pass, margin smaller than the development +0.0171 (GARCH-t refits adapt through 2020)',
               'C2': 'pass', 'C3': 'pass', 'C4': 'pass; open whether the margin holds through the 2020 shock'}


def load_full(name):
    with np.load(DATA / f'{name}.npz', allow_pickle=False) as a:
        date = a['date'].astype('U10'); r = a['returns'].astype(float) * 100.; split = a['split'].astype(int)
    scale = initial_scale(r[:100])
    return {'name': name, 'date': date, 'r': r, 'z': r / scale, 'scale': scale, 'split': split}


def garch_t_scores(market, start_date='2015-01-01'):
    """Same construction as prior_adaptation.experiment.conventional_scores, origins from start_date;
    returns the own-law log score and the conditional variance path (z units) for the common law."""
    from arch import arch_model
    z = market['z']; date = market['date']; scale = market['scale']
    out = np.full(len(z), np.nan); hpath = np.full(len(z), np.nan); fits = []
    origins = np.flatnonzero(date >= start_date)
    for start in range(int(origins[0]), len(z), 250):
        history = z[max(0, start - 1000):start]
        fit = arch_model(history, mean='Constant', vol='GARCH', p=1, q=1, dist='t', rescale=False).fit(disp='off', show_warning=False)
        if int(fit.convergence_flag) != 0:
            raise RuntimeError(f"{market['name']}: GARCH benchmark failed to converge at {date[start]}")
        fits.append({'origin': str(date[start]), 'history_n': len(history), 'nu': float(fit.params['nu'])})
        p = fit.params; h = float(fit.conditional_volatility[-1] ** 2)
        for t in range(start, min(start + 250, len(z))):
            h = float(p['omega'] + p['alpha[1]'] * (z[t - 1] - p['mu']) ** 2 + p['beta[1]'] * h)
            hpath[t] = h
            out[t] = log_law(np.array([z[t] - p['mu']]), np.array([h]), np.array([p['nu']]))[0] - np.log(scale)
    return out, hpath, fits


def ewma_t8_scores(market, lam=0.94):
    r = market['r']; h = np.empty(len(r)); h[0] = max(float(np.mean(r[:10] ** 2)), 1e-4)
    for t in range(1, len(r)):
        h[t] = lam * h[t - 1] + (1 - lam) * r[t - 1] ** 2
    return log_law(r, h, np.full(len(r), 8.0))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='experiments/outputs/reserved_confirmation_v1')
    ap.add_argument('--markets', nargs='*'); a = ap.parse_args(argv)
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True); (root / 'cells').mkdir(exist_ok=True)
    names = a.markets or sorted(n for g in GROUPS.values() for n in g)
    if not (root / 'protocol.json').exists():
      save(root / 'protocol.json', {'scope': 'One-shot reserved-block confirmation of the development headline; no training, tuning or '
                                           'selection; rules and predictions frozen before scoring.', 'rules': RULES, 'thresholds':
                                  {'p': 0.05, 'markets_improved': 20, 'markets': 28}, 'predictions': PREDICTIONS,
                                  'inputs': {str(p): sha(Path(p)) for p in [__file__, 'experiments/baselines/univol.py', 'experiments/prior_adaptation/experiment.py',
                                                                            'experiments/prior_adaptation/model.py']},
                                  'data': str(DATA), 'extract': str(EXTRACT), 'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    rows = []; fits_all = {}; t0 = time.time()
    for name in names:
        cache = root / 'cells' / f'{name}.csv'
        if cache.exists():                                   # resumable: one market per file
            rows.extend(pd.read_csv(cache, dtype={'date': str}).to_dict('records')); fits_all[name] = json.loads((root / 'cells' / f'{name}_fits.json').read_text()); continue
        m = load_full(name); mask = m['split'] == 2
        with np.load(EXTRACT / f'{name}.npz', allow_pickle=False) as e:
            if not np.array_equal(e['date'].astype('U10'), m['date']):
                raise RuntimeError(f'{name}: extract calendar differs from the full series')
            net_own = e['logscore'].astype(float); net_h = e['h'].astype(float)
        g_own, g_h, fits = garch_t_scores(m); fits_all[name] = fits
        ok_n = np.isfinite(net_h) & (net_h > 0); ok_g = np.isfinite(g_h) & (g_h > 0)      # placeholders outside the scored mask only
        net_common = np.where(ok_n, log_law(m['r'], np.where(ok_n, net_h, 1.0), np.full(len(m['r']), 8.0)), np.nan)
        g_common = np.where(ok_g, log_law(m['z'], np.where(ok_g, g_h, 1.0), np.full(len(m['z']), 8.0)) - np.log(m['scale']), np.nan)
        uni = run_series(m['r'], levels=())['logscore']
        ew = ewma_t8_scores(m)
        idx = np.flatnonzero(mask); new = []
        for i in idx:
            new.append(dict(date=m['date'][i], market=name, network=net_own[i], garch_t=g_own[i], univol=uni[i], ewma_t8=ew[i],
                            network_common_t8=net_common[i], garch_common_t8=g_common[i]))
        pd.DataFrame(new).to_csv(cache, index=False); (root / 'cells' / f'{name}_fits.json').write_text(json.dumps(fits)); rows.extend(new)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {len(idx)} reserved days, {len(fits)} GARCH refits, "
              f"net-garch {np.nanmean(net_own[idx] - g_own[idx]):+.4f}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(root / 'scores.csv', index=False)
    L = ['# Reserved-block confirmation (2015-01 to 2022-02, 28 markets, one pass)', '',
         f'{len(frame)} market-days scored; GARCH refits per market: {sorted({len(v) for v in fits_all.values()})}.', '',
         '| rule | comparison | mean diff (nats/return) | 95% block interval | p (two-sided) | markets improved | verdict |', '|---|---|---:|---|---:|---:|---|']
    decisions = {}
    for rule, (cand, ctrl) in RULES.items():
        sub = frame.dropna(subset=[cand, ctrl]); res = paired_calendar_interval(sub, cand, ctrl)
        pivot = sub.pivot(index='date', columns='market', values=[cand, ctrl]); diff = (pivot[cand] - pivot[ctrl]).to_numpy()
        n = len(diff); rng = np.random.default_rng(927601); sampled = []
        for _ in range(20):
            starts = rng.integers(0, n, size=(100, int(np.ceil(n / 60)))); ind = ((starts[:, :, None] + np.arange(60)) % n).reshape(100, -1)[:, :n]
            sampled.extend(np.nanmean(np.nanmean(diff[ind], axis=1), axis=1).tolist())
        sampled = np.asarray(sampled); p = max(min(1.0, 2 * min(np.mean(sampled <= 0), np.mean(sampled >= 0))), 1 / len(sampled))
        ok = res['mean_difference'] > 0 and p < 0.05 and res['markets_improved'] >= 20
        decisions[rule] = dict(candidate=cand, control=ctrl, passed=bool(ok), p=float(p), **{k: res[k] for k in ('mean_difference', 'calendar_block_interval95', 'markets_improved', 'markets')})
        lo, hi = res['calendar_block_interval95']
        L.append(f"| {rule} | {cand} − {ctrl} | {res['mean_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} | {res['markets_improved']}/{res['markets']} | {'PASS' if ok else 'FAIL'} |")
    per_market = frame.groupby('market')[['network', 'garch_t', 'univol', 'ewma_t8', 'network_common_t8', 'garch_common_t8']].mean()
    per_market.to_csv(root / 'per_market.csv')
    L += ['', 'Predictions registered before the run: ' + '; '.join(f'{k}: {v}' for k, v in PREDICTIONS.items()), '',
          f'Elapsed {time.time() - t0:.0f}s.']
    (root / 'REPORT.md').write_text('\n'.join(L) + '\n'); save(root / 'decisions.json', decisions); save(root / 'garch_fits.json', fits_all)
    save(root / 'status.json', {'status': 'complete'}); print('\n'.join(L))


if __name__ == '__main__':
    main()
