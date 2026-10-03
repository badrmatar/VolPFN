"""Market leg: the online wrappers around real black-box (VaR, ES) forecasters on the 28-index
Oxford-Man panel used by dr-vf (calendar-corrected copy, daily log returns in percent).

Every forecaster is causal: the pair (v_t, e_t) for day t depends on returns strictly before t and
on parameters refitted on an expanding window every REFIT_EVERY days (first fit at FIRST_FIT; the
first-fit parameters are also used, in-sample, before that date, which is never scored). The
wrappers run online from t = 0 with a WARM-day pass-through, so when the development window
(split == 1, 2011-2014) is scored they hold a decade of standardised history. The reserved block
(2015+) is not read in the default mode; `--reserved` switches to the full-series data directory
and scores split == 2, to be run once after MARKET_PROTOCOL.md's development rules are read.

Black boxes: ewma_gauss (RiskMetrics, lambda .94, Gaussian tail); ewma_t5 (same variance, unit-
variance t(5) tail); hist250 (rolling 250-day historical simulation); garch_t (GARCH(1,1) with
Student-t innovations, expanding refits); garch_gauss (the garch_t variance path with a Gaussian
tail: right dynamics, wrong tail law); gas_fz0 (Patton-Ziegel-Chen 2019 one-factor GAS fitted by
FZ0 minimisation, expanding refits, one model per level); volpfn_t8 (the dr-vf network's extracted
variance path with the unit-variance t(8) law).
"""
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln
from scipy import stats
from .fz import fz0, identification, empirical_var_es, hindsight_ratio_recalibration, \
    student_t_var_es, normal_var_es
from .calibeater import RatioCalibeater, BinCalibeater

HERE = Path(__file__).resolve().parent.parent
DRVF = Path(os.environ.get('CALIBEAT_DRVF', HERE.parent))
DATA_DEV = DRVF / 'experiments/data/oxfordman_barde/calendar_development_v2'
DATA_FULL = DRVF / 'experiments/data/oxfordman_barde/calendar_full_v1'
EXTRACT_DEV = DRVF / 'experiments/outputs/attribution_extract_v1'
EXTRACT_FULL = DRVF / 'experiments/outputs/attribution_extract_full_v1'
ALPHAS = (0.05, 0.025, 0.01)
FORECASTERS = ['ewma_gauss', 'ewma_t5', 'hist250', 'garch_gauss', 'garch_t', 'gas_fz0', 'volpfn_t8', 'fhs_garch']
MISSPECIFIED = ['ewma_gauss', 'garch_gauss', 'hist250']          # M1 set
SPECIALISTS = ['garch_t', 'gas_fz0', 'volpfn_t8']                # M2 set
RATIOS = ['ratio_ftl', 'ratio_win1000', 'ratio_win2000', 'ratio_sa']    # M4 choice set
SWITCHES = ['switch_' + r[6:] for r in RATIOS]                          # original-vs-wrapped follow-the-leader, all history
GUARDS = ['guard_' + r[6:] for r in RATIOS]                             # the same over the last GUARD_WINDOW days with a margin
CALIBEATERS = RATIOS + SWITCHES + GUARDS + ['adapt', 'bins5']       # adapt: FTL over {original, ftl, win1000, win2000}, W = GUARD_WINDOW, margin GUARD_MARGIN
SWITCH_MIN = 250                                                        # comparisons before a switch may leave the original
GUARD_WINDOW = 1000
REGRET_BUDGET = {0.05: 0.0035, 0.025: 0.006, 0.01: 0.013}   # M2 budget: sandbox oracle excess in the panel's geometry + 2 SE
GUARD_MARGIN = {a: 0.5 * b for a, b in REGRET_BUDGET.items()}         # the wrapped pair must beat the original by this per day
WARM = 250; FIRST_FIT = 750; REFIT_EVERY = 250; V_CAP = -1e-3
BOOT = dict(seed=927601, reps=2000, block=60)


# ----------------------------------------------------------------------------- data
def market_names(data):
    return sorted(p.stem for p in Path(data).glob('*.npz'))


def load_market(name, data):
    with np.load(Path(data) / f'{name}.npz', allow_pickle=False) as a:
        date = a['date'].astype('U10'); r = a['returns'].astype(float) * 100.; split = a['split'].astype(int)
    if not np.isfinite(r).all():
        raise ValueError(f'{name}: non-finite returns')
    return dict(name=name, date=date, r=r, split=split)


def load_network_variance(name, extract, fallback):
    """The network's one-step variance path for percent returns; NaN prefix (min history) filled
    with the fallback path. Returns None when no extract exists for the market."""
    p = Path(extract) / f'{name}.npz'
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as a:
        h = a['h'].astype(float)
    if len(h) != len(fallback):
        return None
    bad = ~np.isfinite(h) | (h <= 0); h = h.copy(); h[bad] = fallback[bad]
    return h


# ----------------------------------------------------------------------------- variance black boxes
def ewma_variance(r, lam=0.94):
    T = len(r); x = np.empty(T); x[0] = max(float(np.mean(r[:10] ** 2)), 1e-4); x[1:] = (1 - lam) * r[:-1] ** 2
    return lfilter([1.0], [1.0, -lam], x)


def garch_variance(r, w, a, b, h0):
    x = np.empty(len(r)); x[0] = h0; x[1:] = w + a * r[:-1] ** 2
    return lfilter([1.0], [1.0, -b], x)


def t_loglik(r, h, nu):
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi * h)
            - (nu + 1) / 2 * np.log1p(r * r / ((nu - 2) * h)))


def _garch_unpack(th, h0):
    w = np.exp(th[0]) * h0; a = 0.6 / (1 + np.exp(-th[1])); b = (0.9995 - a) / (1 + np.exp(-th[2])); nu = 2.05 + np.exp(th[3])
    return w, a, b, nu


def fit_garch_t(r, x0=None):
    """MLE of GARCH(1,1) with unit-variance Student-t(nu) innovations; returns (theta_raw, params)."""
    h0 = float(np.mean(r ** 2))

    def nll(th):
        w, a, b, nu = _garch_unpack(th, h0); h = garch_variance(r, w, a, b, h0)
        if not np.all(h > 0) or nu > 300:
            return 1e6
        return -float(np.mean(t_loglik(r[1:], h[1:], nu)))
    starts = ([np.asarray(x0, float)] if x0 is not None else []) + \
        [np.array([np.log(0.05), np.log(0.08 / 0.52), np.log(0.90 / 0.0195), np.log(4.0)]),
         np.array([np.log(0.02), np.log(0.05 / 0.55), np.log(0.94 / 0.0095), np.log(2.0)])]
    best = None
    for s in starts:
        res = minimize(nll, s, method='BFGS', options=dict(maxiter=300, gtol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, _garch_unpack(best.x, h0), float(best.fun)


def refit_schedule(T, first=FIRST_FIT, every=REFIT_EVERY):
    return list(range(first, T, every))


def garch_t_paths(r):
    """Expanding-window refits: variance and nu paths, each day's values from the latest fit whose
    fitting sample ends before that day (first-fit parameters before FIRST_FIT, unscored)."""
    T = len(r); h = np.empty(T); nu = np.empty(T); fits = refit_schedule(T); th = None; log = []
    for k, t0 in enumerate(fits):
        th, (w, a, b, n), f = fit_garch_t(r[:t0], th); hk = garch_variance(r, w, a, b, float(np.mean(r[:t0] ** 2)))
        lo = 0 if k == 0 else t0; hi = fits[k + 1] if k + 1 < len(fits) else T
        h[lo:hi] = hk[lo:hi]; nu[lo:hi] = n; log.append(dict(t=t0, omega=w, alpha=a, beta=b, nu=n, nll=f))
    return np.maximum(h, 1e-6), nu, log


def fhs_paths(r, garch_log, alpha):
    """Textbook filtered historical simulation (Barone-Adesi, Giannopoulos and Vosper 1999) on the
    GARCH-t fits: at each refit, the in-sample standardised residuals of the fitted model over the
    estimation window give an empirical (VaR, ES) that is rescaled by the model's conditional
    scale until the next refit. Same refit schedule as garch_t; the residual pair is frozen between
    refits (the wrapper, by contrast, updates its pair daily from out-of-sample residuals)."""
    T = len(r); v = np.empty(T); e = np.empty(T); fits = [f['t'] for f in garch_log]
    for k, f in enumerate(garch_log):
        t0 = f['t']; hk = garch_variance(r, f['omega'], f['alpha'], f['beta'], float(np.mean(r[:t0] ** 2)))
        res = r[1:t0] / np.sqrt(np.maximum(hk[1:t0], 1e-12)); q, es = empirical_var_es(res, alpha)
        lo = 0 if k == 0 else t0; hi = fits[k + 1] if k + 1 < len(fits) else T; sk = np.sqrt(np.maximum(hk[lo:hi], 1e-12))
        v[lo:hi] = q * sk; e[lo:hi] = es * sk
    return v, e


def hist_sim(r, alpha, window=250, min_obs=20):
    T = len(r); v = np.empty(T); e = np.empty(T)
    for t in range(T):
        s = r[max(0, t - window):t] if t >= min_obs else r[:min_obs]
        v[t], e[t] = empirical_var_es(s, alpha)
    return v, e


# ----------------------------------------------------------------------------- PZC one-factor GAS
def gas_path(r, a, b, beta, gamma, alpha, kappa0=0.0):
    """Patton-Ziegel-Chen (2019) one-factor GAS: v_t = a exp(k_t), e_t = b exp(k_t),
    k_{t+1} = beta k_t + gamma (1/e_t)(1{y_t <= v_t} y_t / alpha - e_t); a < 0, b < a, omega = 0
    (absorbed by a, b). Forecasts for day t use k_t, a function of y_{<t}."""
    T = len(r); v = np.empty(T); e = np.empty(T); k = float(kappa0); ia = 1.0 / alpha
    for t in range(T):
        ek = np.exp(k); vt = a * ek; et = b * ek; v[t] = vt; e[t] = et; y = r[t]
        force = (y * ia / et - 1.0) if y <= vt else -1.0
        k = beta * k + gamma * force
        if k > 4.0: k = 4.0
        elif k < -4.0: k = -4.0
    return v, e


def _gas_unpack(th):
    a = -np.exp(th[0]); b = a - np.exp(th[1]); beta = 1.0 / (1.0 + np.exp(-th[2])); return a, b, beta, float(th[3])


def fit_gas(r, alpha, x0=None, burn=50, maxfev=400):
    q0, es0 = empirical_var_es(r, alpha)

    def loss(th):
        a, b, beta, gamma = _gas_unpack(th)
        if abs(gamma) > 1.0 / alpha:
            return 1e6
        v, e = gas_path(r, a, b, beta, gamma, alpha)
        return float(np.mean(fz0(r[burn:], v[burn:], e[burn:], alpha)))
    base = [np.log(-q0), np.log(max(q0 - es0, 1e-3))]
    starts = ([np.asarray(x0, float)] if x0 is not None else []) + \
        [np.array(base + [3.9, 0.001 / alpha]), np.array(base + [2.2, 0.0003 / alpha])]
    best = None
    for s in starts:
        res = minimize(loss, s, method='Nelder-Mead', options=dict(maxfev=maxfev, xatol=1e-4, fatol=1e-8))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, _gas_unpack(best.x), float(best.fun)


def gas_paths(r, alpha):
    T = len(r); v = np.empty(T); e = np.empty(T); fits = refit_schedule(T); th = None; log = []
    for k, t0 in enumerate(fits):
        th, (a, b, beta, gamma), f = fit_gas(r[:t0], alpha, th); vk, ek = gas_path(r, a, b, beta, gamma, alpha)
        lo = 0 if k == 0 else t0; hi = fits[k + 1] if k + 1 < len(fits) else T
        v[lo:hi] = vk[lo:hi]; e[lo:hi] = ek[lo:hi]; log.append(dict(t=t0, a=a, b=b, beta=beta, gamma=gamma, fz0=f))
    return v, e, log


# ----------------------------------------------------------------------------- black boxes -> (v, e)
def scale_pair(sigma, q, es):
    return q * sigma, es * sigma


def forecast(name, r, alpha, cache):
    """(v, e) for every day; cache holds the market-level shared paths (see prepare_cache)."""
    if name == 'ewma_gauss':
        return scale_pair(np.sqrt(cache['h_ewma']), *normal_var_es(alpha))
    if name == 'ewma_t5':
        return scale_pair(np.sqrt(cache['h_ewma']), *student_t_var_es(alpha, 5.0))
    if name == 'hist250':
        return hist_sim(r, alpha)
    if name == 'garch_gauss':
        return scale_pair(np.sqrt(cache['h_garch']), *normal_var_es(alpha))
    if name == 'garch_t':
        s = np.sqrt(cache['h_garch']); nu = cache['nu_garch']; v = np.empty(len(r)); e = np.empty(len(r))
        for n in np.unique(nu):
            q, es = student_t_var_es(alpha, float(n)); idx = nu == n; v[idx] = q * s[idx]; e[idx] = es * s[idx]
        return v, e
    if name == 'gas_fz0':
        v, e, log = gas_paths(r, alpha); cache.setdefault('gas_log', {})[alpha] = log; return v, e
    if name == 'fhs_garch':
        return fhs_paths(r, cache['garch_log'], alpha)
    if name == 'volpfn_t8':
        if cache.get('h_net') is None:
            return None
        return scale_pair(np.sqrt(cache['h_net']), *student_t_var_es(alpha, 8.0))
    raise KeyError(name)


def prepare_cache(r, name, extract):
    cache = dict(h_ewma=ewma_variance(r))
    cache['h_garch'], cache['nu_garch'], cache['garch_log'] = garch_t_paths(r)
    cache['h_net'] = load_network_variance(name, extract, cache['h_ewma'])
    return cache


def calibeat(name, y, v, e, alpha):
    if name == 'ratio_ftl':
        return RatioCalibeater(alpha, warm=WARM, mode='ftl').run(y, v, e)
    if name.startswith('ratio_win'):
        return RatioCalibeater(alpha, warm=WARM, mode='window', window=int(name[9:])).run(y, v, e)
    if name == 'ratio_sa':
        return RatioCalibeater(alpha, warm=WARM, mode='sa', eta=0.002).run(y, v, e)
    if name.startswith('bins'):
        return BinCalibeater(alpha, K=int(name[4:]), warm=WARM).run(y, v, e)
    raise KeyError(name)


def switch(s_orig, s_wrap, v_orig, e_orig, v_wrap, e_wrap, start=WARM, min_days=SWITCH_MIN, window=None, margin=0.0):
    """Follow-the-leader over two experts, the original pair and the wrapped pair: day t takes the
    wrapped pair iff its cumulative FZ0 over the comparison days before t (all days from `start`, or
    the last `window` of them) is below the original's by more than `margin` per day; ties and the
    first min_days days go to the original. Causal: the choice for day t never uses day t."""
    T = len(s_orig); v = v_orig.copy(); e = e_orig.copy(); live = (np.arange(T) >= start).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(live * (s_wrap - s_orig))]); cn = np.concatenate([[0.0], np.cumsum(live)])
    for t in range(start + min_days, T):
        lo = 0 if window is None else max(0, t - window)
        gap = cs[t] - cs[lo]; n = cn[t] - cn[lo]                 # sums over days lo..t-1
        if n > 0 and gap < -margin * n:
            v[t] = v_wrap[t]; e[t] = e_wrap[t]
    return v, e


def scored(y, v, e, alpha, sc):
    s = fz0(y, v, e, alpha)
    return dict(score=float(np.mean(s[sc])), hit=float(np.mean(y[sc] <= v[sc])),
                es_resid=float(np.mean(identification(y[sc], v[sc], e[sc], alpha)[1]))), s


def market_cell(args):
    """All forecasters x levels x wrappers for one market; daily scores on the scoring mask."""
    name, data, extract, alphas, split, out = args[:6]; forecasters = args[6] if len(args) > 6 else FORECASTERS
    t0 = time.time(); m = load_market(name, data); r = m['r']; sc = np.flatnonzero(m['split'] == split)
    if len(sc) == 0:
        return dict(market=name, skipped='no scoring rows')
    cache = prepare_cache(r, name, extract); rows = []; daily = {}
    for alpha in alphas:
        for fc in forecasters:
            pair = forecast(fc, r, alpha, cache)
            if pair is None:
                continue
            v = np.minimum(pair[0], V_CAP); e = np.minimum(pair[1], v - 1e-6)
            base, s = scored(r, v, e, alpha, sc)
            row = dict(market=name, forecaster=fc, alpha=alpha, original=base['score'],
                       original_hit=base['hit'], original_es_resid=base['es_resid'], scored_days=int(len(sc)),
                       hits_scored=int(np.sum(r[sc] <= v[sc])))
            daily[f'{fc}|{alpha}|original'] = s[sc]
            vr, er = hindsight_ratio_recalibration(r, v, alpha)           # constants from the whole series
            row['hindsight_full'] = float(np.mean(fz0(r[sc], vr[sc], er[sc], alpha)))
            vw, ew = hindsight_ratio_recalibration(r[sc], v[sc], alpha)   # constants from the window itself
            row['hindsight_window'] = float(np.mean(fz0(r[sc], vw, ew, alpha)))
            wrapped = {}
            for cb in CALIBEATERS:
                if cb == 'adapt':
                    from .adaptive import ftl_window_select
                    names = ['ratio_ftl', 'ratio_win1000', 'ratio_win2000']
                    Lm = np.stack([fz0(r, v, e, alpha)] + [fz0(r, *wrapped[k], alpha) for k in names], axis=1)
                    ch = ftl_window_select(Lm, start=WARM + SWITCH_MIN, W=GUARD_WINDOW, margin=GUARD_MARGIN.get(alpha, 0.0))
                    vh = v.copy(); eh = e.copy()
                    for k, nm in enumerate(names, start=1):
                        idx = ch == k; vh[idx] = wrapped[nm][0][idx]; eh[idx] = wrapped[nm][1][idx]
                elif cb.startswith('switch_') or cb.startswith('guard_'):
                    kind, tail = cb.split('_', 1); vw, ew = wrapped['ratio_' + tail]
                    vh, eh = switch(fz0(r, v, e, alpha), fz0(r, vw, ew, alpha), v, e, vw, ew,
                                    window=GUARD_WINDOW if kind == 'guard' else None,
                                    margin=GUARD_MARGIN.get(alpha, 0.0) if kind == 'guard' else 0.0)
                else:
                    vh, eh = calibeat(cb, r, v, e, alpha); wrapped[cb] = (vh, eh)
                res, s = scored(r, vh, eh, alpha, sc)
                row[cb] = res['score']; row[cb + '_hit'] = res['hit']; row[cb + '_es_resid'] = res['es_resid']
                daily[f'{fc}|{alpha}|{cb}'] = s[sc]
            rows.append(row)
    out = Path(out); (out / 'cells').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'cells' / f'{name}.npz', date=m['date'][sc], **daily)
    fits = dict(garch=cache['garch_log'], gas={str(k): v for k, v in cache.get('gas_log', {}).items()},
                network_available=cache['h_net'] is not None, seconds=time.time() - t0)
    (out / 'cells' / f'{name}.json').write_text(json.dumps(dict(rows=rows, fits=fits), indent=1))
    return dict(market=name, rows=len(rows), seconds=time.time() - t0)


# ----------------------------------------------------------------------------- panel statistics
def paired_block(diff, seed=BOOT['seed'], reps=BOOT['reps'], block=BOOT['block']):
    """Calendar-block bootstrap of a (dates x markets) matrix of paired daily differences (NaN where
    a market has no observation): the same construction as dr-vf's paired_calendar_interval."""
    n = diff.shape[0]; rng = np.random.default_rng(seed); sampled = []
    for _ in range(reps // 100):
        starts = rng.integers(0, n, size=(100, int(np.ceil(n / block))))
        idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(100, -1)[:, :n]
        sampled.extend(np.nanmean(np.nanmean(diff[idx], axis=1), axis=1).tolist())
    sampled = np.asarray(sampled); delta = np.nanmean(diff, axis=0)
    p = min(1.0, 2 * min(float(np.mean(sampled <= 0)), float(np.mean(sampled >= 0)))); p = max(p, 1.0 / len(sampled))
    return dict(mean_difference=float(np.nanmean(delta)), interval95=np.quantile(sampled, [.025, .975]).tolist(),
                p_two_sided=float(p), markets_improved=int(np.sum(delta < 0)), markets=int(len(delta)))


def load_cells(out, extra=()):
    """Rows and daily scores of a run, merged with any extra run directories (e.g. a comparator black
    box computed later for the same markets and window; the dates must agree)."""
    rows = []; daily = {}; dates = {}
    for root in [out] + list(extra):
        for p in sorted(Path(root).glob('cells/*.json')):
            d = json.loads(p.read_text()); rows.extend(d['rows'])
            with np.load(p.with_suffix('.npz'), allow_pickle=False) as a:
                dd = a['date'].astype('U10')
                if p.stem in dates and not np.array_equal(dates[p.stem], dd):
                    raise RuntimeError(f'{p.stem}: dates differ between {out} and {root}')
                dates[p.stem] = dd; daily.setdefault(p.stem, {}).update({k: a[k] for k in a.files if k != 'date'})
    return rows, daily, dates


def diff_matrix(daily, dates, key_a, key_b):
    markets = [m for m in daily if key_a in daily[m] and key_b in daily[m]]
    if not markets:
        return None, []
    all_dates = np.array(sorted(set(np.concatenate([dates[m] for m in markets]))))
    out = np.full((len(all_dates), len(markets)), np.nan)
    for j, m in enumerate(markets):
        out[np.searchsorted(all_dates, dates[m]), j] = daily[m][key_a] - daily[m][key_b]
    return out, markets


def fmt(x, w=8, d=4):
    return ('%*.*f' % (w, d, x)) if x == x and np.isfinite(x) else '%*s' % (w, 'nan')


def summarise(out, alphas=ALPHAS, variant='guard_ftl', extra=()):
    out = Path(out); rows, daily, dates = load_cells(out, extra)
    markets = sorted(daily); L = []; P = L.append; decisions = []; panel = {}
    P('CALIBEATING THE (VaR, ES) PAIR ON THE OXFORD-MAN PANEL (MARKET_PROTOCOL.md)')
    P(f'{len(markets)} markets; scoring window rows per market: ' +
      ', '.join(f'{m}={len(dates[m])}' for m in markets[:6]) + (' ...' if len(markets) > 6 else ''))
    P('Mean FZ0 over the scoring window, panel average over markets (lower is better); hind_full = the'
      ' constant-ratio recalibration fitted on the whole series, hind_win = fitted on the window itself.')
    for alpha in alphas:
        P(''); P(f'=== alpha = {alpha} ===')
        P('%-12s %8s %8s %8s | %8s %8s %8s %8s %8s %8s | %6s %6s %5s' % ('forecaster', 'original', 'hindfull', 'hindwin',
          'ftl', 'win1000', 'win2000', 'sa', 'gd_ftl', 'bins5', 'hit_o', 'hit_gd', 'n'))
        for fc in FORECASTERS:
            sub = [r for r in rows if r['forecaster'] == fc and r['alpha'] == alpha]
            if not sub:
                continue
            mean = lambda k: float(np.mean([r[k] for r in sub]))
            cell = dict(original=mean('original'), hindsight_full=mean('hindsight_full'), hindsight_window=mean('hindsight_window'),
                        hit_original=mean('original_hit'), markets=len(sub))
            for cb in CALIBEATERS:
                cell[cb] = mean(cb); cell[cb + '_hit'] = mean(cb + '_hit')
                d, ms = diff_matrix(daily, dates, f'{fc}|{alpha}|{cb}', f'{fc}|{alpha}|original')
                cell[cb + '_boot'] = paired_block(d) if d is not None else None
            cell['mcb_full'] = cell['original'] - cell['hindsight_full']; cell['mcb_window'] = cell['original'] - cell['hindsight_window']
            cell['gain'] = cell['original'] - cell[variant]
            cell['capture_full'] = cell['gain'] / cell['mcb_full'] if abs(cell['mcb_full']) > 1e-9 else float('nan')
            cell['capture_window'] = cell['gain'] / cell['mcb_window'] if abs(cell['mcb_window']) > 1e-9 else float('nan')
            panel[f'{fc}|{alpha}'] = cell
            P('%-12s %s %s %s | %s %s %s %s %s %s | %s %s %5d' % (fc, fmt(cell['original']), fmt(cell['hindsight_full']),
              fmt(cell['hindsight_window']), fmt(cell['ratio_ftl']), fmt(cell['ratio_win1000']), fmt(cell['ratio_win2000']),
              fmt(cell['ratio_sa']), fmt(cell['guard_ftl']), fmt(cell['bins5']), fmt(cell['hit_original'], 6, 3), fmt(cell['guard_ftl_hit'], 6, 3), len(sub)))
        P(f'-- {variant} minus original: mean difference [block-bootstrap 95% interval] p, markets improved; capture = gain / MCB')
        for fc in FORECASTERS:
            c = panel.get(f'{fc}|{alpha}')
            if c is None or c[variant + '_boot'] is None:
                continue
            b = c[variant + '_boot']
            P('%-12s %+.4f [%+.4f, %+.4f] p=%.3f  %2d/%2d   capture(full)=%s capture(window)=%s' % (fc, b['mean_difference'],
              b['interval95'][0], b['interval95'][1], b['p_two_sided'], b['markets_improved'], b['markets'],
              fmt(c['capture_full'], 6, 2), fmt(c['capture_window'], 6, 2)))
    # ---- decisions: binding on `variant` (the switch), the pure wrapper reported alongside as informational
    P(''); P(f'DECISIONS (binding rules evaluated on {variant}; the pure wrapper and the unguarded switch shown as informational)')
    pure = 'ratio_' + variant.split('_', 1)[1] if variant.startswith(('switch_', 'guard_')) else 'ratio_ftl'
    plain = 'switch_' + variant.split('_', 1)[1] if variant.startswith('guard_') else None
    for wv, binding in ((variant, True), (pure, False)) + (((plain, False),) if plain else ()):
        tag = '' if binding else (' (informational: pure wrapper)' if wv == pure else ' (informational: unguarded switch)')
        for alpha in alphas:
            budget = REGRET_BUDGET.get(alpha, float('nan'))
            for fc in MISSPECIFIED:
                c = panel.get(f'{fc}|{alpha}')
                if c is None or c[wv + '_boot'] is None:
                    continue
                b = c[wv + '_boot']; ok = (b['mean_difference'] < 0) and (b['p_two_sided'] < 0.05) and (b['markets_improved'] >= 20)
                info = (not binding) or fc == 'hist250'
                decisions.append(dict(rule='M1_gain', wrapper=wv, alpha=alpha, forecaster=fc, passed=bool(ok), informational=info,
                                      gain=-b['mean_difference'], p=b['p_two_sided'], markets_improved=b['markets_improved'], markets=b['markets']))
                P(f"M1_gain      {wv:<14} {fc:<12} alpha={alpha:<6} {'PASS' if ok else 'FAIL'}  gain={-b['mean_difference']:+.4f} "
                  f"p={b['p_two_sided']:.3f} {b['markets_improved']}/{b['markets']}{tag}{' (informational: outside the class)' if fc == 'hist250' and binding else ''}")
            for fc in SPECIALISTS:
                c = panel.get(f'{fc}|{alpha}')
                if c is None or c[wv + '_boot'] is None:
                    continue
                b = c[wv + '_boot']; excess = c[wv] - c['original']; ok = (excess <= 0) or (excess <= budget)
                decisions.append(dict(rule='M2_no_harm', wrapper=wv, alpha=alpha, forecaster=fc, passed=bool(ok), informational=not binding,
                                      excess=excess, budget=budget, p=b['p_two_sided'], markets_improved=b['markets_improved'], markets=b['markets']))
                P(f"M2_no_harm   {wv:<14} {fc:<12} alpha={alpha:<6} {'PASS' if ok else 'FAIL'}  excess={excess:+.4f} budget={budget:+.4f} "
                  f"p={b['p_two_sided']:.3f} improved {b['markets_improved']}/{b['markets']}{tag}")
            for fc in ['ewma_gauss', 'garch_gauss']:
                c = panel.get(f'{fc}|{alpha}')
                if c is None:
                    continue
                cap = (c['original'] - c[wv]) / c['mcb_full'] if abs(c['mcb_full']) > 1e-9 else float('nan')
                capw = (c['original'] - c[wv]) / c['mcb_window'] if abs(c['mcb_window']) > 1e-9 else float('nan')
                ok = cap == cap and cap >= 0.5; info = (not binding) or alpha < 0.02
                decisions.append(dict(rule='M3_capture', wrapper=wv, alpha=alpha, forecaster=fc, passed=bool(ok), informational=info,
                                      capture_full=cap, capture_window=capw))
                P(f"M3_capture   {wv:<14} {fc:<12} alpha={alpha:<6} {'PASS' if ok else 'FAIL'}  capture_full={fmt(cap, 5, 2)} capture_window={fmt(capw, 5, 2)}"
                  f"{tag}{' (informational at this level)' if alpha < 0.02 and binding else ''}")
    totals = {cb: float(np.nansum([panel[k][cb] - panel[k]['original'] for k in panel])) for cb in RATIOS}
    choice = min(totals, key=totals.get)
    decisions.append(dict(rule='M4_variant_choice', chosen=choice, chosen_guard='guard_' + choice[6:], total_gain_by_variant={k: -v for k, v in totals.items()}))
    P('M4_variant   ratio variant with the largest total gain over all forecasters and levels: %s -> reserved-block rules on guard_%s   (%s)' % (
      choice, choice[6:], ', '.join(f'{k}={-v:+.4f}' for k, v in totals.items())))
    for alpha in alphas:
        for fc in SPECIALISTS:
            c = panel.get(f'{fc}|{alpha}')
            if c:
                P(f"M5_bins      {fc:<12} alpha={alpha:<6} bins5-original={c['bins5'] - c['original']:+.4f}  ratio_ftl-original={c['ratio_ftl'] - c['original']:+.4f}  {variant}-original={c[variant] - c['original']:+.4f}")
    for alpha in alphas:
        n = int(np.mean([r['scored_days'] for r in rows])); lo, hi = stats.binom.ppf([.025, .975], n, alpha) / n
        for fc in FORECASTERS:
            sub = [r for r in rows if r['forecaster'] == fc and r['alpha'] == alpha]
            if sub:
                P(f"M6_hits      {fc:<12} alpha={alpha:<6} within binomial band: original {sum(lo <= r['original_hit'] <= hi for r in sub)}/{len(sub)}, "
                  f"ratio_ftl {sum(lo <= r['ratio_ftl_hit'] <= hi for r in sub)}/{len(sub)}, {variant} {sum(lo <= r[variant + '_hit'] <= hi for r in sub)}/{len(sub)}")
    n_pass = sum(1 for d in decisions if 'passed' in d and d['passed'] and not d.get('informational')); n_all = sum(1 for d in decisions if 'passed' in d and not d.get('informational'))
    P(f'SUMMARY: {n_pass}/{n_all} binding decisions pass')
    (out / 'REPORT.txt').write_text('\n'.join(L) + '\n'); (out / 'decisions.json').write_text(json.dumps(decisions, indent=1))
    (out / 'panel.json').write_text(json.dumps(panel, indent=1, default=float))
    print('\n'.join(L))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['run', 'summarise'])
    ap.add_argument('--out', default='outputs/market_dev_v1'); ap.add_argument('--markets', nargs='*')
    ap.add_argument('--alphas', nargs='*', type=float, default=list(ALPHAS)); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--reserved', action='store_true', help='full-series data, score split == 2 (one pass only)')
    ap.add_argument('--variant', default='guard_ftl', help='wrapper the binding rules are evaluated on (reserved block: the M4 choice)')
    ap.add_argument('--forecasters', nargs='*', default=None, help='subset of black boxes to run (default all)')
    ap.add_argument('--extra', nargs='*', default=[], help='summarise: extra run directories to merge (comparators)')
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = DATA_FULL if a.reserved else DATA_DEV; extract = EXTRACT_FULL if a.reserved else EXTRACT_DEV; split = 2 if a.reserved else 1
    if a.command == 'summarise':
        return summarise(out, a.alphas, a.variant, a.extra)
    names = a.markets or market_names(DATA_DEV)          # the 28 development markets, in both modes
    missing = [n for n in names if not (Path(data) / f'{n}.npz').exists()]
    if missing:
        raise SystemExit(f'markets without data under {data}: {missing}')
    todo = [n for n in names if not (out / 'cells' / f'{n}.json').exists()]
    (out / 'protocol.json').write_text(json.dumps(dict(data=str(data), extract=str(extract), split=split, alphas=a.alphas,
        markets=names, forecasters=FORECASTERS, calibeaters=CALIBEATERS, warm=WARM, first_fit=FIRST_FIT, refit_every=REFIT_EVERY,
        boot=BOOT, code_sha256={f: sha(HERE / 'calibeat' / f) for f in ['market.py', 'calibeater.py', 'fz.py']},
        protocol_sha256=sha(DRVF / 'docs/protocols/tail/MARKET_PROTOCOL.md') if (DRVF / 'docs/protocols/tail/MARKET_PROTOCOL.md').exists() else None,
        started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
    print(f'{len(todo)} markets to run ({len(names) - len(todo)} done), workers {a.workers}', flush=True)
    jobs = [(n, str(data), str(extract), a.alphas, split, str(out), a.forecasters or FORECASTERS) for n in todo]
    if a.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for res in pool.imap_unordered(market_cell, jobs):
                print(f"[{time.strftime('%H:%M:%S')}] {res}", flush=True)
    else:
        for j in jobs:
            print(f"[{time.strftime('%H:%M:%S')}] {market_cell(j)}", flush=True)
    print('run finished', flush=True)


if __name__ == '__main__':
    main()
