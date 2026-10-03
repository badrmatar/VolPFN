#!/usr/bin/env python3
"""univol_eval.py -- out-of-sample evaluation of the universal volatility filter (univol.py) against GARCH-family
baselines on many return series, in the protocol of Wu, Hernandez-Lobato & Ghahramani (2014): one-step-ahead
forecasts, predictive log-likelihood, per-series ranks and Wilcoxon signed-rank tests.

Methods (see initial-scale conventions in the individual implementations):
  univol        the universal filter (bank of score-driven EWMA experts x Student-t laws + FLH restarts, Fixed-Share mixture)
  univol-bank   ablation: the bank only, no restart experts
  univol-sym    ablation: no leverage experts (symmetric response to the sign of the return)
  riskmetrics   EWMA on squared returns, lambda = 0.94, Gaussian predictive
  garch-t, gjr-t, egarch-t   GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1) with Student-t innovations, maximum likelihood
                (arch package), refitted every --refit observations on all past data, forecasts run forward with the
                fitted parameters between refits
  oracle-fixed  NOT causal: the single (gain, law) expert of the bank with the best realised QLIKE on the evaluation
                window -- the best fixed-gain filter in hindsight, the theory's comparator
Metrics on the evaluation window (t >= --burn): mean QLIKE with r^2 as the variance proxy, mean predictive log density,
VaR violation rates at 1% and 2.5% with Kupiec (unconditional coverage) and Christoffersen (independence) p-values,
and the 2.5% ES ratio (mean realised loss on violation days / mean ES forecast on those days).
usage: python3 univol_eval.py --real yf_stocks --procs 6 --out out_univol      (20 US stocks 1990-2022 via yfinance)
       python3 univol_eval.py --real sp500_20 --procs 2 --out out_univol       (same stocks from the skfolio package, Python >= 3.10)
       python3 univol_eval.py --real yf_fx --procs 6 --out out_univol           (20 FX rates via yfinance, needs internet)
       python3 univol_eval.py --real csv:returns.csv --out out_univol            (date column + one column per series)
       python3 univol_eval.py --synthetic --out out_univol                      (jump / smooth paths with known truth)
"""
import argparse, csv, math, os, sys, time, warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import chi2, wilcoxon, rankdata
from scipy.special import gammaln, stdtr
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from univol import UniVol, run_series, t_score, lgam, qlike, initial_scale


def log(msg):
    print(msg, flush=True)


# baselines
def riskmetrics(r, lam=0.94):
    T = len(r)
    h = np.empty(T)
    h[0] = initial_scale(r)
    for t in range(1, T):
        h[t] = lam * h[t - 1] + (1 - lam) * r[t - 1] ** 2
    return h


def gaussian_logscore(r, h):
    return -0.5 * np.log(2 * np.pi * h) - 0.5 * r * r / h


def t_logscore(r, h, nu):
    s = np.sqrt(h * (nu - 2.0) / nu)
    z = r / s
    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - 0.5 * (nu + 1) * np.log1p(z * z / nu)
        - np.log(s)
    )


def t_var_es(h, nu, level):
    """VaR and ES (positive losses) of a unit-variance Student-t(nu) scaled to variance h."""
    from scipy.stats import t as st

    s = np.sqrt(h * (nu - 2.0) / nu)
    a = 1 - level
    q = st.ppf(a, nu)
    es = (nu + q * q) / (nu - 1) * st.pdf(q, nu) / a
    return -q * s, es * s


def gaussian_var_es(h, level):
    from scipy.stats import norm

    a = 1 - level
    q = norm.ppf(a)
    return -q * np.sqrt(h), norm.pdf(q) / a * np.sqrt(h)


def arch_family(r, kind, refit, burn):
    """expanding-window maximum likelihood every `refit` obs; between refits the fitted recursion runs forward.
    Returns (h forecasts for t, nu used at t, mu used at t) with NaN before burn."""
    from arch import arch_model

    T = len(r)
    h = np.full(T, np.nan)
    nu = np.full(T, np.nan)
    mu = np.full(T, np.nan)
    spec = dict(
        garch=dict(vol='GARCH', p=1, o=0, q=1),
        gjr=dict(vol='GARCH', p=1, o=1, q=1),
        egarch=dict(vol='EGARCH', p=1, o=1, q=1),
    )[kind]
    am = arch_model(r, mean='Constant', dist='t', **spec)
    for t0 in range(burn, T, refit):
        t1 = min(t0 + refit, T)
        try:
            res = am.fit(last_obs=t0, disp='off', show_warning=False)
            fixed = am.fix(res.params.values)
            fc = fixed.forecast(horizon=1, start=t0 - 1, reindex=False)
            v = (
                fc.variance.values.ravel()
            )  # row i is the forecast for observation t0 + i (made at t0 - 1 + i)
            h[t0:t1] = v[: t1 - t0]
            nu[t0:t1] = float(res.params['nu'])
            mu[t0:t1] = float(res.params['mu'])
        except Exception as e:
            h[t0:t1] = np.nan
    return h, nu, mu


def fixed_gain_paths(r, gains, nus, scale):
    """forecast path of every (gain, nu) score-driven EWMA expert, run alone (for the hindsight oracle)."""
    G, N = np.meshgrid(gains, nus, indexing='ij')
    g = G.ravel()
    nu = N.ravel()
    newton = 2 * (nu + 3) / nu
    T = len(r)
    th = np.full(len(g), math.log(scale))
    H = np.empty((T, len(g)))
    for t in range(T):
        H[t] = np.exp(th)
        u = r[t] ** 2 / H[t]
        th = np.maximum(th + g * newton * t_score(u, nu), math.log(1e-8 * scale))
    return H, g, nu


# backtests
def kupiec_p(hits, a):
    """Kupiec unconditional-coverage likelihood-ratio test of P(violation) = a."""
    n = len(hits)
    x = int(hits.sum())
    if n == 0:
        return float('nan')
    if x == 0:
        lr = -2.0 * n * math.log(1 - a)
    elif x == n:
        lr = -2.0 * n * math.log(a)
    else:
        p = x / n
        lr = -2.0 * (
            x * math.log(a)
            + (n - x) * math.log(1 - a)
            - x * math.log(p)
            - (n - x) * math.log(1 - p)
        )
    return float(1 - chi2.cdf(lr, 1))


def christoffersen_p(hits):
    h = hits.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(h[:-1], h[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(pa, pb):
        return (
            (n00 * math.log(1 - pa) if n00 and pa < 1 else 0)
            + (n01 * math.log(pa) if n01 and pa > 0 else 0)
            + (n10 * math.log(1 - pb) if n10 and pb < 1 else 0)
            + (n11 * math.log(pb) if n11 and pb > 0 else 0)
        )

    lr = -2 * (ll(p, p) - ll(p01, p11))
    return 1 - chi2.cdf(max(lr, 0.0), 1)


def score_method(r, h, ls, var1, es1, var25, es25, burn, tag=None):
    ok = np.arange(len(r)) >= burn
    ok &= ~np.isnan(h)
    r2 = r * r
    loss = -r
    if tag:
        from dump_utils import dump

        dump(tag, h[ok], r2[ok])
    out = dict(
        qlike=float(np.mean(qlike(h[ok], r2[ok]))), logscore=float(np.mean(ls[ok])), n=int(ok.sum())
    )
    for lv, var, es in ((0.99, var1, es1), (0.975, var25, es25)):
        hit = loss[ok] > var[ok]
        a = 1 - lv
        out[f'hit{lv:g}'] = float(hit.mean())
        out[f'kupiec{lv:g}'] = kupiec_p(hit, a)
        out[f'christ{lv:g}'] = christoffersen_p(hit)
        out[f'esratio{lv:g}'] = (
            float(loss[ok][hit].mean() / es[ok][hit].mean()) if hit.sum() >= 5 else float('nan')
        )
    return out


# one series
def eval_series(args):
    name, r, a = args
    r = (
        np.asarray(r, dtype=float) * 100.0
    )  # percent returns (arch's preferred scale; QLIKE is scale-free)
    T = len(r)
    burn = a.burn
    res = {}
    t0 = time.time()
    # univol and its ablation
    for label, kw in (
        ('univol', {}),
        ('univol-bank', dict(use_flh=False)),
        ('univol-sym', dict(leverages=(0.0,))),
    ):
        u = run_series(r, levels=(0.99, 0.975), **kw)
        res[label] = score_method(
            r,
            u['h'],
            u['logscore'],
            u['var0.99'],
            u['es0.99'],
            u['var0.975'],
            u['es0.975'],
            burn,
            tag=f'{label}__{name}',
        )
        res[label]['nu_eff'] = float(np.nanmedian(u['nu_eff'][burn:]))
        res[label]['halflife'] = float(np.nanmedian(u['halflife'][burn:]))
    # RiskMetrics
    h = riskmetrics(r)
    v1, e1 = gaussian_var_es(h, 0.99)
    v25, e25 = gaussian_var_es(h, 0.975)
    res['riskmetrics'] = score_method(
        r, h, gaussian_logscore(r, h), v1, e1, v25, e25, burn, tag=f'riskmetrics__{name}'
    )
    # GARCH family
    for kind in ('garch', 'gjr', 'egarch'):
        h, nu, mu = arch_family(r, kind, a.refit, burn)
        e = r - np.nan_to_num(mu)
        ls = t_logscore(e, h, nu)
        v1, e1 = t_var_es(h, nu, 0.99)
        v25, e25 = t_var_es(h, nu, 0.975)
        res[kind + '-t'] = score_method(
            r,
            h,
            ls,
            v1 - np.nan_to_num(mu),
            e1 - np.nan_to_num(mu),
            v25 - np.nan_to_num(mu),
            e25 - np.nan_to_num(mu),
            burn,
            tag=f'{kind}-t__{name}',
        )
    # hindsight oracle: best single (gain, nu) expert on the evaluation window
    H, g, nu = fixed_gain_paths(
        r, np.geomspace(1e-3, 0.5, 12), np.array([3.0, 4.5, 7.0, 12.0, 40.0]), initial_scale(r)
    )
    ql = [np.mean(qlike(H[burn:, j], r[burn:] ** 2)) for j in range(H.shape[1])]
    j = int(np.argmin(ql))
    hj = H[:, j]
    ls = t_logscore(r, hj, nu[j])
    v1, e1 = t_var_es(hj, nu[j], 0.99)
    v25, e25 = t_var_es(hj, nu[j], 0.975)
    res['oracle-fixed'] = score_method(
        r, hj, ls, v1, e1, v25, e25, burn, tag=f'oracle-fixed__{name}'
    )
    res['oracle-fixed']['gain'] = float(g[j])
    res['oracle-fixed']['nu'] = float(nu[j])
    log(
        f'  {name}: T={T}, done in {time.time()-t0:.0f}s; QLIKE univol {res["univol"]["qlike"]:.4f} garch-t {res["garch-t"]["qlike"]:.4f} oracle {res["oracle-fixed"]["qlike"]:.4f}'
    )
    return name, res


# data
def load_real(name):
    import pandas as pd

    if name == 'sp500_20':  # the 20-stock sample bundled with skfolio (needs Python >= 3.10)
        try:
            from skfolio.datasets import load_sp500_dataset
        except ImportError as exc:
            raise SystemExit(
                'skfolio is not installed / not supported on this Python; use --real yf_stocks (same 20 tickers via yfinance)'
            ) from exc
        px = load_sp500_dataset()
        ret = px.pct_change().dropna()
        return {c: ret[c].to_numpy() for c in ret.columns}
    if name == 'yf_stocks':  # the same 20 tickers from Yahoo Finance, adjusted closes 1990-2022
        import yfinance as yf

        tickers = (
            'AAPL AMD BAC BBY CVX GE HD JNJ JPM KO LLY MRK MSFT PEP PFE PG RRC UNH WMT XOM'.split()
        )
        px = yf.download(
            tickers, start='1990-01-01', end='2022-12-31', auto_adjust=True, progress=False
        )['Close'].dropna(axis=1, thresh=3000)
        return {c: px[c].dropna().pct_change().dropna().to_numpy() for c in px.columns}
    if name == 'yf_fx':
        import yfinance as yf

        pairs = [
            'EURUSD=X',
            'GBPUSD=X',
            'JPY=X',
            'CHF=X',
            'CAD=X',
            'AUDUSD=X',
            'NZDUSD=X',
            'SEK=X',
            'NOK=X',
            'DKK=X',
            'MXN=X',
            'ZAR=X',
            'BRL=X',
            'INR=X',
            'KRW=X',
            'SGD=X',
            'HKD=X',
            'PLN=X',
            'HUF=X',
            'CZK=X',
        ]
        px = yf.download(
            pairs, start='2005-01-01', end='2025-08-31', auto_adjust=True, progress=False
        )['Close'].dropna(axis=1, thresh=3000)
        return {c: px[c].dropna().pct_change().dropna().to_numpy() for c in px.columns}
    if name.startswith('csv:'):
        df = pd.read_csv(name[4:], index_col=0, parse_dates=True).apply(
            pd.to_numeric, errors='coerce'
        )
        return {c: df[c].dropna().to_numpy() for c in df.columns}
    raise ValueError(name)


def synthetic_series(a):
    import falsify_I13 as F

    rng = np.random.default_rng(1)
    out = {}
    for fam, V in (('sparse', 16.0), ('dense', 8.0)):
        for innov in ('gaussian', 't4'):
            for k in range(3):
                theta = F.make_path(a.T, V, fam, rng)[0]
                hs = np.exp(theta)
                eps = (
                    rng.standard_normal(a.T)
                    if innov == 'gaussian'
                    else rng.standard_t(4, a.T) / math.sqrt(2.0)
                )
                out[f'{fam}-{innov}-{k}'] = 0.01 * np.sqrt(hs) * eps
    return out


# report
METHODS = [
    'univol',
    'univol-bank',
    'univol-sym',
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
        f'UNIVERSAL VOLATILITY FILTER -- out-of-sample evaluation on {len(names)} series (evaluation from observation {a.burn}; GARCH refits every {a.refit})'
    )
    for metric, label, better in (
        ('qlike', 'mean QLIKE (r^2 proxy; lower is better)', 'low'),
        ('logscore', 'mean predictive log density (higher is better)', 'high'),
    ):
        P(f'\n{label}')
        P(f'{"series":>16}' + ''.join(f'{m:>13}' for m in METHODS))
        M = np.array([[res[m][metric] for m in METHODS] for _, res in results])
        for n, row in zip(names, M):
            P(f'{n:>16}' + ''.join(f'{v:>13.4f}' for v in row))
        ranks = np.array([rankdata(row if better == 'low' else -row) for row in M])
        P(f'{"mean":>16}' + ''.join(f'{v:>13.4f}' for v in M.mean(0)))
        P(f'{"mean rank":>16}' + ''.join(f'{v:>13.2f}' for v in ranks.mean(0)))
        i0 = METHODS.index('univol')
        wins = [
            (
                int(np.sum(M[:, i0] < M[:, j]))
                if better == 'low'
                else int(np.sum(M[:, i0] > M[:, j]))
            )
            for j in range(len(METHODS))
        ]
        P(f'{"univol wins":>16}' + ''.join(f'{w:>13d}' for w in wins))
        pv = []
        for j in range(len(METHODS)):
            if j == i0 or len(names) < 6:
                pv.append(float('nan'))
                continue
            d = M[:, i0] - M[:, j]
            pv.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        P(f'{"Wilcoxon p":>16}' + ''.join(f'{v:>13.3g}' if v == v else f'{"":>13}' for v in pv))
    P(
        '\nVaR / ES backtests (mean over series): violation rate at 1% and 2.5%, share of series passing Kupiec and Christoffersen at 5%, ES ratio at 2.5% (1 = calibrated)'
    )
    P(
        f'{"method":>16}{"hit1%":>9}{"kupiec":>8}{"christ":>8}{"hit2.5%":>9}{"kupiec":>8}{"christ":>8}{"ES ratio":>10}'
    )
    for m in METHODS:
        h1 = np.mean([res[m]['hit0.99'] for _, res in results])
        k1 = np.mean([res[m]['kupiec0.99'] > 0.05 for _, res in results])
        c1 = np.mean([res[m]['christ0.99'] > 0.05 for _, res in results])
        h2 = np.mean([res[m]['hit0.975'] for _, res in results])
        k2 = np.mean([res[m]['kupiec0.975'] > 0.05 for _, res in results])
        c2 = np.mean([res[m]['christ0.975'] > 0.05 for _, res in results])
        er = np.nanmean([res[m]['esratio0.975'] for _, res in results])
        P(f'{m:>16}{h1:>9.4f}{k1:>8.2f}{c1:>8.2f}{h2:>9.4f}{k2:>8.2f}{c2:>8.2f}{er:>10.3f}')
    P(
        '\nunivol diagnostics per series: median effective nu / median effective half-life (steps); oracle-fixed: chosen (gain, nu)'
    )
    for n, res in results:
        P(
            f'{n:>16}: nu_eff {res["univol"]["nu_eff"]:5.1f}  half-life {res["univol"]["halflife"]:6.1f} | oracle gain {res["oracle-fixed"]["gain"]:.4f} nu {res["oracle-fixed"]["nu"]:.1f}'
        )
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default=None)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--T', type=int, default=4096)
    ap.add_argument('--burn', type=int, default=500)
    ap.add_argument('--refit', type=int, default=250)
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default='out_univol')
    ap.add_argument('--max-series', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = synthetic_series(a) if a.synthetic else load_real(a.real)
    items = list(data.items())[: a.max_series]
    log(
        f'{len(items)} series; lengths {min(len(v) for _, v in items)}-{max(len(v) for _, v in items)}'
    )
    with Pool(a.procs) as pool:
        results = pool.map(eval_series, [(n, r, a) for n, r in items])
    rep = report(results, a)
    log('\n' + rep)
    tag = 'synthetic' if a.synthetic else a.real.replace('csv:', '').replace('/', '_')
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
