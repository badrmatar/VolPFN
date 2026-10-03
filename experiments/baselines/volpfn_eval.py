#!/usr/bin/env python3
"""volpfn_eval.py -- VolPFN against UniVol and the GARCH family on many return series, in the
protocol of Wu, Hernandez-Lobato & Ghahramani (2014): one-step-ahead forecasts, predictive log density, QLIKE, VaR/ES
backtests, per-series ranks and Wilcoxon signed-rank tests.  Reuses the loaders, baselines and scoring of univol_eval.py.

Methods:
  volpfn        the prior-fitted network (checkpoint --ckpt): sliding windows of --L returns, stride L/2; variance
                forecast = second moment of the predictive t-mixture; VaR by numerical inversion; ES from mixture truncated moments
  volpfn-lat    same predictive distribution; the point variance forecast is the posterior mean of the latent variance
                from the auxiliary head (ablation of the point-forecast rule)
  hedge         Fixed-Share (rate 1/(t+1), Bayes weights) over {volpfn, univol} predictive densities: density mixture; VaR/ES from the combined mixture, variance
                forecast = weight-averaged point forecasts
  univol, riskmetrics, garch-t, gjr-t, egarch-t, oracle-fixed   as in univol_eval.py (oracle-fixed is NOT causal)
Initial-scale and tail-calibration restrictions are documented in each baseline module.

usage: python3 volpfn_eval.py --ckpt out_volpfn/volpfn_base_best.pt --real yf_stocks --procs 6 --out out_volpfn
       python3 volpfn_eval.py --ckpt ... --real yf_fx --out out_volpfn
       python3 volpfn_eval.py --ckpt ... --synthetic --out out_volpfn            (jump / smooth paths with known truth)
       python3 volpfn_eval.py --ckpt ... --real csv:returns.csv --no-arch          (quick: skip the GARCH refits)
"""
import argparse, csv, math, os, sys, time, warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.special import stdtr, logsumexp
from scipy.stats import wilcoxon, rankdata
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.baselines import univol_eval as UE
from experiments.baselines.univol import UniVol, initial_scale, qlike, t_logpdf as u_t_logpdf
from volpfn import t_logpdf_np
from experiments.baselines import gpvol
from experiments.baselines import deepvol_tcn
def log(msg):
    print(msg, flush=True)


# univol with its mixture recorded
def run_univol_mix(r, levels=(0.99, 0.975)):
    """like univol.run_series but also records the predictive mixture (weights, variances, nus) at every step."""
    r = np.asarray(r, dtype=float)
    T = len(r)
    f = UniVol(scale=initial_scale(r))
    out = {k: np.full(T, np.nan) for k in ('h', 'h_mean', 'logscore')}
    for lv in levels:
        out[f'var{lv:g}'] = np.full(T, np.nan)
        out[f'es{lv:g}'] = np.full(T, np.nan)
    mix = [None] * T
    for t in range(T):
        if f.theta is not None:
            hm, hg, nu, hl = f.predict()
            out['h'][t], out['h_mean'][t] = hg, hm
            out['logscore'][t] = f.log_score(r[t])
            w, h, nus = f._mixture()
            mix[t] = (w.copy(), h.copy(), nus.copy())
            for lv in levels:
                q, es = f.tail(lv)
                out[f'var{lv:g}'][t], out[f'es{lv:g}'][t] = q, es
        f.update(r[t])
    return out, mix


def mixture_var_es_rows(w, loc, scale, nu, level, iters=45):
    a = 1 - level
    lo = np.min(loc - 60 * scale, axis=1)
    hi = np.max(loc + 60 * scale, axis=1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = np.sum(w * stdtr(nu, (mid[:, None] - loc) / scale), axis=1)
        lo = np.where(cdf < a, mid, lo)
        hi = np.where(cdf < a, hi, mid)
    q = 0.5 * (lo + hi)
    z = (q[:, None] - loc) / scale
    partial = loc * stdtr(nu, z) - scale * (nu + z * z) / (nu - 1) * np.exp(t_logpdf_np(z, nu))
    return -q, -np.sum(w * partial, axis=1) / a


def hedge(r, v, u, umix, levels=(0.99, 0.975)):
    """Fixed-Share over the two predictive densities. Returns h, logscore, var/es dict, and the volpfn weight path."""
    T = len(r)
    lw = np.log(0.5)
    pi = np.full(T, np.nan)
    ls = np.full(T, np.nan)
    h = np.full(T, np.nan)
    ok = ~np.isnan(v['logscore']) & ~np.isnan(u['logscore'])
    logp = np.array([math.log(0.5), math.log(0.5)])
    n = 0
    for t in range(T):
        if not ok[t]:
            continue
        p = np.exp(logp - logsumexp(logp))
        pi[t] = p[0]
        ls[t] = logsumexp(logp + np.array([v['logscore'][t], u['logscore'][t]])) - logsumexp(logp)
        h[t] = p[0] * v['h'][t] + p[1] * u['h'][t]
        n += 1
        bt = 1.0 / (n + 1.0)
        logp = logp + np.array([v['logscore'][t], u['logscore'][t]])
        logp -= logsumexp(logp)
        logp = np.logaddexp(math.log(1 - bt) + logp, math.log(bt / 2))
    out = dict(h=h, logscore=ls, pi=pi)
    idx = np.flatnonzero(ok & ~np.isnan(pi))
    K = v['logw'].shape[1]
    nu_max = max(len(m[0]) for m in umix if m is not None)
    W = np.zeros((len(idx), K + nu_max))
    LOC = np.zeros_like(W)
    SC = np.ones_like(W)
    NU = np.full_like(W, 5.0)
    for j, t in enumerate(idx):
        W[j, :K] = pi[t] * np.exp(v['logw'][t])
        LOC[j, :K] = v['loc'][t]
        SC[j, :K] = v['scale'][t]
        NU[j, :K] = v['nu'][t]
        w, hh, nn = umix[t]
        m = len(w)
        W[j, K : K + m] = (1 - pi[t]) * w
        SC[j, K : K + m] = np.sqrt(hh * (nn - 2) / nn)
        NU[j, K : K + m] = nn
    for lv in levels:
        q, es = mixture_var_es_rows(W, LOC, SC, NU, lv)
        out[f'var{lv:g}'] = np.full(T, np.nan)
        out[f'es{lv:g}'] = np.full(T, np.nan)
        out[f'var{lv:g}'][idx] = q
        out[f'es{lv:g}'][idx] = es
    return out


# one series (worker)
def eval_series(args):
    name, r, a, v = args
    r = np.asarray(r, dtype=float) * 100.0
    T = len(r)
    burn = a.burn
    res = {}
    t0 = time.time()
    sc = lambda h, ls, o, tag: UE.score_method(
        r, h, ls, o['var0.99'], o['es0.99'], o['var0.975'], o['es0.975'], burn, tag=f'{tag}__{name}'
    )
    # volpfn (its outputs were computed on the same percent returns in the main process)
    res['volpfn'] = sc(v['h'], v['logscore'], v, 'volpfn')
    res['volpfn-lat'] = sc(v['h_lat'], v['logscore'], v, 'volpfn-lat')
    res['volpfn']['nu_eff'] = float(
        np.nanmedian(np.sum(np.exp(v['logw'][burn:]) * v['nu'][burn:], axis=1))
    )
    # univol with its mixture, then the hedge
    u, umix = run_univol_mix(r)
    res['univol'] = sc(u['h'], u['logscore'], u, 'univol')
    hd = hedge(r, v, u, umix)
    res['hedge'] = sc(hd['h'], hd['logscore'], hd, 'hedge')
    res['hedge']['pi_volpfn'] = float(np.nanmean(hd['pi'][burn:]))
    # GP-Vol (Wu, Hernandez-Lobato & Ghahramani 2014), RAPCF with a finite-memory GP -- see gpvol.py
    if getattr(a, 'no_gpvol', False):
        res['gpvol'] = {
            k: float('nan')
            for k in (
                'qlike',
                'logscore',
                'hit0.99',
                'kupiec0.99',
                'christ0.99',
                'esratio0.99',
                'hit0.975',
                'kupiec0.975',
                'christ0.975',
                'esratio0.975',
            )
        }
        res['gpvol']['n'] = 0
    else:
        gp = gpvol.run_gpvol(
            r,
            burn=burn,
            levels=(0.99, 0.975),
            N=getattr(a, 'gpvol_particles', 60),
            window=getattr(a, 'gpvol_window', 60),
        )
        res['gpvol'] = sc(gp['h'], gp['logscore'], gp, 'gpvol')
    # DeepVol-RM (Moreno-Pino & Zohren 2024), dilated causal convs on a 22-day realised-measures window -- see deepvol_tcn.py
    if getattr(a, 'no_deepvol', False):
        res['deepvol'] = {
            k: float('nan')
            for k in (
                'qlike',
                'logscore',
                'hit0.99',
                'kupiec0.99',
                'christ0.99',
                'esratio0.99',
                'hit0.975',
                'kupiec0.975',
                'christ0.975',
                'esratio0.975',
            )
        }
        res['deepvol']['n'] = 0
    else:
        dv = deepvol_tcn.run_deepvol(
            r,
            burn=burn,
            levels=(0.99, 0.975),
            epochs=getattr(a, 'deepvol_epochs', 60),
            refit_every=getattr(a, 'deepvol_refit', 250),
        )
        res['deepvol'] = sc(dv['h'], dv['logscore'], dv, 'deepvol')
    # RiskMetrics
    h = UE.riskmetrics(r)
    v1, e1 = UE.gaussian_var_es(h, 0.99)
    v25, e25 = UE.gaussian_var_es(h, 0.975)
    res['riskmetrics'] = UE.score_method(
        r, h, UE.gaussian_logscore(r, h), v1, e1, v25, e25, burn, tag=f'riskmetrics__{name}'
    )
    # GARCH family
    for kind in ('garch', 'gjr', 'egarch'):
        if a.no_arch:
            res[kind + '-t'] = {k: float('nan') for k in res['univol']}
            res[kind + '-t']['n'] = 0
            continue
        h, nu, mu = UE.arch_family(r, kind, a.refit, burn)
        e = r - np.nan_to_num(mu)
        ls = UE.t_logscore(e, h, nu)
        v1, e1 = UE.t_var_es(h, nu, 0.99)
        v25, e25 = UE.t_var_es(h, nu, 0.975)
        m = np.nan_to_num(mu)
        res[kind + '-t'] = UE.score_method(
            r, h, ls, v1 - m, e1 - m, v25 - m, e25 - m, burn, tag=f'{kind}-t__{name}'
        )
    # hindsight oracle
    H, g, nu = UE.fixed_gain_paths(
        r, np.geomspace(1e-3, 0.5, 12), np.array([3.0, 4.5, 7.0, 12.0, 40.0]), initial_scale(r)
    )
    ql = [np.mean(qlike(H[burn:, j], r[burn:] ** 2)) for j in range(H.shape[1])]
    j = int(np.argmin(ql))
    hj = H[:, j]
    ls = UE.t_logscore(r, hj, nu[j])
    v1, e1 = UE.t_var_es(hj, nu[j], 0.99)
    v25, e25 = UE.t_var_es(hj, nu[j], 0.975)
    res['oracle-fixed'] = UE.score_method(
        r, hj, ls, v1, e1, v25, e25, burn, tag=f'oracle-fixed__{name}'
    )
    res['oracle-fixed']['gain'] = float(g[j])
    res['oracle-fixed']['nu'] = float(nu[j])
    log(
        f'  {name}: T={T}, {time.time() - t0:.0f}s; log density volpfn {res["volpfn"]["logscore"]:.4f} univol {res["univol"]["logscore"]:.4f} gpvol {res["gpvol"]["logscore"]:.4f} deepvol {res["deepvol"]["logscore"]:.4f} hedge {res["hedge"]["logscore"]:.4f} garch-t {res["garch-t"]["logscore"]:.4f} | QLIKE volpfn {res["volpfn"]["qlike"]:.4f} univol {res["univol"]["qlike"]:.4f} gpvol {res["gpvol"]["qlike"]:.4f} deepvol {res["deepvol"]["qlike"]:.4f} garch-t {res["garch-t"]["qlike"]:.4f}'
    )
    return name, res


# report
METHODS = [
    'volpfn',
    'volpfn-lat',
    'hedge',
    'univol',
    'gpvol',
    'deepvol',
    'riskmetrics',
    'garch-t',
    'gjr-t',
    'egarch-t',
    'oracle-fixed',
]


def report(results, a):
    L = []
    P = L.append
    names = [n for n, _ in results]
    P(
        f'VolPFN -- out-of-sample evaluation on {len(names)} series (evaluation from observation {a.burn}; GARCH refits every {a.refit}; VolPFN window {a.L}, checkpoint {os.path.basename(a.ckpt)})'
    )
    for metric, label, better in (
        ('logscore', 'mean predictive log density (higher is better)', 'high'),
        ('qlike', 'mean QLIKE (r^2 proxy; lower is better)', 'low'),
    ):
        P(f'\n{label}')
        P(f'{"series":>16}' + ''.join(f'{m:>13}' for m in METHODS))
        M = np.array([[res[m][metric] for m in METHODS] for _, res in results])
        for n, row in zip(names, M):
            P(f'{n:>16}' + ''.join(f'{v:>13.4f}' for v in row))
        ranks = np.array([rankdata(row if better == 'low' else -row) for row in M])
        P(f'{"mean":>16}' + ''.join(f'{v:>13.4f}' for v in np.nanmean(M, 0)))
        P(f'{"mean rank":>16}' + ''.join(f'{v:>13.2f}' for v in ranks.mean(0)))
        i0 = METHODS.index('volpfn')
        wins = [
            (
                int(np.sum(M[:, i0] < M[:, j]))
                if better == 'low'
                else int(np.sum(M[:, i0] > M[:, j]))
            )
            for j in range(len(METHODS))
        ]
        P(f'{"volpfn wins":>16}' + ''.join(f'{w:>13d}' for w in wins))
        pv = []
        for j in range(len(METHODS)):
            d = M[:, i0] - M[:, j]
            d = d[~np.isnan(d)]
            pv.append(
                float('nan')
                if j == i0 or len(d) < 6 or not np.any(d != 0)
                else float(wilcoxon(d).pvalue)
            )
        P(f'{"Wilcoxon p":>16}' + ''.join(f'{v:>13.3g}' if v == v else f'{"":>13}' for v in pv))
    P(
        '\nVaR / ES backtests (mean over series): violation rate at 1% and 2.5%, share of series passing Kupiec and Christoffersen at 5%, ES ratio at 2.5% (1 = calibrated)'
    )
    P(
        f'{"method":>16}{"hit1%":>9}{"kupiec":>8}{"christ":>8}{"hit2.5%":>9}{"kupiec":>8}{"christ":>8}{"ES ratio":>10}'
    )
    for m in METHODS:
        g = lambda k: np.array([res[m][k] for _, res in results], dtype=float)
        P(
            f'{m:>16}{np.nanmean(g("hit0.99")):>9.4f}{np.nanmean(g("kupiec0.99") > 0.05):>8.2f}{np.nanmean(g("christ0.99") > 0.05):>8.2f}{np.nanmean(g("hit0.975")):>9.4f}{np.nanmean(g("kupiec0.975") > 0.05):>8.2f}{np.nanmean(g("christ0.975") > 0.05):>8.2f}{np.nanmean(g("esratio0.975")):>10.3f}'
        )
    P(
        '\ndiagnostics per series: volpfn median effective nu | hedge mean weight on volpfn | oracle-fixed (gain, nu)'
    )
    for n, res in results:
        P(
            f'{n:>16}: nu_eff {res["volpfn"]["nu_eff"]:5.1f} | hedge weight on volpfn {res["hedge"]["pi_volpfn"]:.2f} | oracle gain {res["oracle-fixed"]["gain"]:.4f} nu {res["oracle-fixed"]["nu"]:.1f}'
        )
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--real', default=None)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--T', type=int, default=4096)
    ap.add_argument('--burn', type=int, default=500)
    ap.add_argument('--refit', type=int, default=250)
    ap.add_argument('--L', type=int, default=1024)
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default='out_volpfn')
    ap.add_argument('--max-series', type=int, default=None)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--no-arch', action='store_true')
    ap.add_argument('--no-gpvol', action='store_true')
    ap.add_argument('--gpvol-particles', type=int, default=60)
    ap.add_argument('--gpvol-window', type=int, default=60)
    ap.add_argument('--no-deepvol', action='store_true')
    ap.add_argument('--deepvol-epochs', type=int, default=60)
    ap.add_argument('--deepvol-refit', type=int, default=250)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = UE.synthetic_series(a) if a.synthetic else UE.load_real(a.real)
    items = list(data.items())[: a.max_series]
    log(
        f'{len(items)} series; lengths {min(len(v) for _, v in items)}-{max(len(v) for _, v in items)}'
    )
    from volpfn import load_model, predict_series

    model = load_model(a.ckpt, a.device)
    dev = next(model.parameters()).device
    t0 = time.time()
    vouts = {}
    for n, r in items:
        vouts[n] = predict_series(
            model, np.asarray(r, dtype=float) * 100.0, L=a.L, levels=(0.99, 0.975), device=dev
        )
    log(f'volpfn forecasts for {len(items)} series in {time.time() - t0:.0f}s on {dev}')
    with Pool(a.procs) as pool:
        results = pool.map(eval_series, [(n, r, a, vouts[n]) for n, r in items])
    rep = report(results, a)
    log('\n' + rep)
    tag = ('synthetic' if a.synthetic else a.real.replace('csv:', '').replace('/', '_')) + '_volpfn'
    with open(os.path.join(a.out, f'{tag}_report.txt'), 'w') as f:
        f.write(rep)
    with open(os.path.join(a.out, f'{tag}_per_series.csv'), 'w', newline='') as f:
        keys = sorted(set(k for _, res in results for m in res for k in res[m]))
        w = csv.writer(f)
        w.writerow(['series', 'method'] + keys)
        for n, res in results:
            for m in METHODS:
                w.writerow([n, m] + [res[m].get(k, '') for k in keys])
    log(f'wrote {a.out}/{tag}_report.txt and {tag}_per_series.csv')


if __name__ == '__main__':
    main()
