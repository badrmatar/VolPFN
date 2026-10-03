"""Shared machinery for the driver-swap study and the prior-family mechanism readout.

State variables, a generalised standalone recursion, nested specifications, the fold loop and the
paired statistics. gain_surgery.py is left untouched: this module reproduces its asymmetric_gain and
state_dep_gain specifications exactly (same recursion, same normalisation, same fitting window)
and adds the level, additive-level, nested and horizon-sweep specifications of
research/attribution/PROTOCOL_AMENDMENT_20260907.md.

The recursion is log h_{t+1} = log h_t + g_t with

    g_t = b0 + b1 (log h_t - m) + gain_t (z_t^2 - 1) + b3 (|z_t| - E|z|) + b4 sign(r_t) min(|z_t|, 5)
          + sum_j a_j X_j(t)
    gain_t = b2 + sum_j e_j X_j(t) + e_lev lev_t + e_logh (log h_t - m)

where X_j are causal exogenous state variables computed from returns (slopes and levels of the
5/20/.../2000-day log-EWMA channels), lev_t = log h_t - log(scale^2) is the filter's own lagged
log-variance relative to the market's first-100-observation scale, and m is the pooled fold mean of
log h. Every exogenous variable is standardised by the training fold's mean and standard deviation.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon

from research.prior_adaptation.experiment import GROUPS, load_market
from research.prior_adaptation.model import log_law
from research.attribution.gain_surgery import ABS_CENTRE, BURN, NU_SCORE
import research.prior_adaptation.experiment as _E

HALF_LIVES_ALL = (5., 20., 50., 100., 250., 500., 2000.)
LEVELS = tuple(f'l{int(h)}' for h in HALF_LIVES_ALL)
CORE = ['const', 'log_h', 'score', 'absz', 'signed']

# Exogenous state variables as functions of the market-referenced log-EWMA levels.
STATE = {
    'd1': lambda L: L['l5'] - L['l20'],
    'd2': lambda L: L['l20'] - L['l100'],
    'd20_50': lambda L: L['l20'] - L['l50'],
    'd20_250': lambda L: L['l20'] - L['l250'],
    'd20_500': lambda L: L['l20'] - L['l500'],
    'd20_2000': lambda L: L['l20'] - L['l2000'],
    'l5': lambda L: L['l5'], 'l20': lambda L: L['l20'], 'l100': lambda L: L['l100'],
}

# spec -> (drift state names, gate state names, gate on lev, gate on pooled log h)
SPECS = {
    'asymmetric_gain':      ([], [], False, False),
    'state_dep_gain':       (['d1', 'd2'], ['d1', 'd2'], False, False),
    'drift_only':           (['d1', 'd2'], [], False, False),
    'gate_only':            ([], ['d1', 'd2'], False, False),
    'gate_d2':              ([], ['d2'], False, False),
    'gate_d1':              ([], ['d1'], False, False),
    'gate_level_h':         ([], [], True, False),      # Bauwens-Storti (2009) / Ding (2023) driver
    'gate_level_h_pooled':  ([], [], False, True),
    'gate_level_l100':      ([], ['l100'], False, False),
    'gate_level_l20':       ([], ['l20'], False, False),
    'gate_levels_5_20_100': ([], ['l5', 'l20', 'l100'], False, False),
    'state_dep_plus_l100':  (['d1', 'd2'], ['d1', 'd2', 'l100'], False, False),
    'state_dep_plus_lev':   (['d1', 'd2'], ['d1', 'd2'], True, False),
    'gate_d20_50':          ([], ['d20_50'], False, False),
    'gate_d20_250':         ([], ['d20_250'], False, False),
    'gate_d20_500':         ([], ['d20_500'], False, False),
    'gate_d20_2000':        ([], ['d20_2000'], False, False),
    # used by the ablation readout only: every gate at once, for a single descriptive regression
    'readout':              (['d1', 'd2'], ['d1', 'd2', 'l100'], True, False),
}
SWEEP = ['gate_d20_50', 'gate_d2', 'gate_d20_250', 'gate_d20_500', 'gate_d20_2000', 'gate_level_l20']
SWEEP_H = [50, 100, 250, 500, 2000, float('inf')]


def param_names(spec):
    drift, gate, lev, logh = SPECS[spec]
    return (CORE + list(drift) + [f'score_x_{g}' for g in gate]
            + (['score_x_lev'] if lev else []) + (['score_x_log_h'] if logh else []))


def n_params(spec):
    return len(param_names(spec))


def ewma_logs(r, half_lives=HALF_LIVES_ALL):
    """Causal log EWMA of r^2 at each half-life; the value at t includes r_t and zero returns do not
    advance the average (as in volpfn.make_features and gain_surgery.ewma_logs)."""
    T = len(r); nz = r != 0.
    out = np.empty((T, len(half_lives)))
    for j, hl in enumerate(half_lives):
        lam = 0.5 ** (1.0 / hl); e = 1.0
        for t in range(T):
            if nz[t]:
                e = lam * e + (1 - lam) * r[t] * r[t]
            out[t, j] = e
    return np.log(out)


def levels_for(market):
    """Market-referenced log-EWMA levels: log EWMA_k(r^2) - log(scale^2), one column per half-life."""
    lw = ewma_logs(market['r']) - 2.0 * np.log(market['scale'])
    return {name: lw[:, j] for j, name in enumerate(LEVELS)}


def state_raw(market):
    L = levels_for(market)
    return {name: f(L) for name, f in STATE.items()}


def build_frame(extract_root, mask_by_market, markets):
    """One row per usable transition of the network's own update, as gain_surgery.build_frame, with
    every exogenous state variable and the market-referenced level of log h attached."""
    from pathlib import Path
    parts = []
    for group, names in GROUPS.items():
        for name in sorted(names):
            a = np.load(Path(extract_root) / f'{name}.npz', allow_pickle=False)
            date, r, h, owner = a['date'].astype('U10'), a['r'], a['h'], a['owner']
            m = markets[name]
            if not np.array_equal(date, m['date']) or not np.allclose(r, m['r']):
                raise RuntimeError(f'{name}: extracted returns or calendar do not match the market file')
            S = state_raw(m)
            idx = np.arange(len(r) - 1)
            usable = ((owner[:-1] >= 0) & (owner[1:] >= 0) & (owner[:-1] == owner[1:])
                      & (idx >= BURN) & np.isfinite(h[:-1]) & np.isfinite(h[1:]) & (h[:-1] > 0))
            k = np.flatnonzero(usable)
            cols = {'market': name, 'group': group, 'date': date[k], 'next_date': date[k + 1],
                    'in_mask': np.isin(date[k + 1], mask_by_market[name]), 'r': r[k],
                    'log_h': np.log(h[k]), 'lev_raw': np.log(h[k]) - 2.0 * np.log(m['scale']),
                    'g': np.log(h[k + 1]) - np.log(h[k])}
            for s in STATE:
                cols[s + '_raw'] = S[s][k]
            parts.append(pd.DataFrame(cols))
    return pd.concat(parts, ignore_index=True)


def norm_from(train):
    """Fold normalisation for every state variable at once, so nested specifications share it."""
    norm = {'logh_mean': float(train.log_h.mean())}
    for s in list(STATE) + ['lev']:
        v = train[s + '_raw']
        norm[s + '_mean'] = float(v.mean()); norm[s + '_sd'] = float(v.std() + 1e-12)
    return norm


def _std(frame, s, norm):
    return (frame[s + '_raw'].to_numpy() - norm[s + '_mean']) / norm[s + '_sd']


def design_matrix(frame, spec, norm):
    drift, gate, lev, logh = SPECS[spec]
    log_h = frame.log_h.to_numpy() - norm['logh_mean']; r = frame.r.to_numpy()
    z2 = r * r / np.exp(frame.log_h.to_numpy()); az = np.sqrt(np.maximum(z2, 0.)); score = z2 - 1.
    cols = [np.ones_like(z2), log_h, score, az - ABS_CENTRE, np.sign(r) * np.minimum(az, 5.)]
    cols += [_std(frame, s, norm) for s in drift]
    cols += [score * _std(frame, s, norm) for s in gate]
    if lev:
        cols.append(score * _std(frame, 'lev', norm))
    if logh:
        cols.append(score * log_h)
    return np.column_stack(cols)


def split_beta(beta, spec):
    drift, gate, lev, logh = SPECS[spec]
    beta = np.asarray(beta, float); i = 5
    core = beta[:5]
    a = beta[i:i + len(drift)]; i += len(drift)
    e = beta[i:i + len(gate)]; i += len(gate)
    e_lev = float(beta[i]) if lev else 0.; i += int(lev)
    e_logh = float(beta[i]) if logh else 0.
    return core, a, e, e_lev, e_logh


def prepare(market, norm, keep, net_h=None):
    """Per-series precomputation: everything the recursion needs that does not depend on the running
    variance. States are standardised here; the drift/gate combinations are formed per spec."""
    r = market['r']; S = state_raw(market)
    X = {s: (S[s] - norm[s + '_mean']) / norm[s + '_sd'] for s in STATE}
    lh0 = float(np.log(max(np.mean(r[:100] ** 2), 1e-8)))
    if net_h is None:
        netlh = np.full(len(r), np.nan)
    else:
        if len(net_h) != len(r):
            raise ValueError('network variance path does not align with the market returns')
        netlh = np.where(np.isfinite(net_h) & (net_h > 0), np.log(np.maximum(net_h, 1e-300)), np.nan)
    return dict(r=r, keep=keep, r2=r * r, absr=np.abs(r), sgn=np.sign(r), X=X, lh0=lh0,
                lev_ref=2.0 * np.log(market['scale']), netlh=netlh)


def stack(prepared):
    T = max(len(p['r']) for p in prepared); n = len(prepared)
    P = {k: np.zeros((n, T)) for k in ('R2', 'ABSR', 'SGN', 'Y', 'NETLH')}
    P.update(valid=np.zeros((n, T), bool), keep=np.zeros((n, T), bool), netok=np.zeros((n, T), bool),
             lh0=np.zeros(n), lev_ref=np.zeros(n), X={s: np.zeros((n, T)) for s in STATE})
    for i, p in enumerate(prepared):
        m = len(p['r'])
        P['R2'][i, :m] = p['r2']; P['ABSR'][i, :m] = p['absr']; P['SGN'][i, :m] = p['sgn']
        P['Y'][i, :m] = p['r']; P['valid'][i, :m] = True; P['keep'][i, :m] = p['keep']
        P['lh0'][i] = p['lh0']; P['lev_ref'][i] = p['lev_ref']
        good = np.isfinite(p['netlh'])
        P['NETLH'][i, :m] = np.where(good, p['netlh'], 0.); P['netok'][i, :m] = p['keep'] & good
        for s in STATE:
            P['X'][s][i, :m] = p['X'][s]
    return P


def run_batch(P, beta, spec, norm):
    """The fitted rule as a self-contained recursive filter reading only returns, advanced over many
    series at once (padded and masked; each row carries its own state)."""
    drift, gate, lev, logh = SPECS[spec]
    (b0, b1, b2, b3, b4), a, e, e_lev, e_logh = split_beta(beta, spec)
    n, T = P['R2'].shape
    DRIFT = np.zeros((n, T)); GATE = np.zeros((n, T))
    for coef, s in zip(a, drift):
        DRIFT += coef * P['X'][s]
    for coef, s in zip(e, gate):
        GATE += coef * P['X'][s]
    m = norm['logh_mean']; lev_mean, lev_sd = norm['lev_mean'], norm['lev_sd']
    R2, ABSR, SGN, valid, lev_ref = P['R2'], P['ABSR'], P['SGN'], P['valid'], P['lev_ref']
    LH = np.empty((n, T)); lh = np.asarray(P['lh0'], float).copy()
    for t in range(T):
        LH[:, t] = lh
        if t + 1 == T:
            break
        inv = np.exp(-lh); z2 = R2[:, t] * inv; az = ABSR[:, t] * np.sqrt(inv); score = z2 - 1.
        gain = b2 + GATE[:, t]
        if lev:
            gain = gain + e_lev * ((lh - lev_ref) - lev_mean) / lev_sd
        if logh:
            gain = gain + e_logh * (lh - m)
        g = (b0 + b1 * (lh - m) + gain * score + b3 * (az - ABS_CENTRE)
             + b4 * SGN[:, t] * np.minimum(az, 5.) + DRIFT[:, t])
        lh = np.clip(lh + np.clip(g, -3., 3.), -20., 20.)
        lh = np.where(valid[:, t], lh, LH[:, t])
    return np.exp(LH)


def negloglik(beta, P, spec, norm):
    H = run_batch(P, beta, spec, norm); k = P['keep']
    if not np.all(np.isfinite(H[k])) or np.any(H[k] <= 0):
        return 1e6
    return -float(np.sum(log_law(P['Y'][k], H[k], NU_SCORE))) / max(int(k.sum()), 1)


def freerun_loss(beta, P, spec, norm):
    H = run_batch(P, beta, spec, norm); k = P['netok']
    if not np.all(np.isfinite(H[k])) or np.any(H[k] <= 0):
        return 1e6
    return float(np.mean((np.log(H[k]) - P['NETLH'][k]) ** 2))


def fit_spec(train, fit_packed, spec, norm, maxfev, do_freerun=True, do_likelihood=True):
    """Distilled OLS, then the two Powell refits started from it. Returns the three coefficient
    vectors and the fit diagnostics."""
    X = design_matrix(train, spec, norm); y = train.g.to_numpy()
    beta_d, *_ = np.linalg.lstsq(X, y, rcond=None)
    in_r2 = 1 - np.var(y - X @ beta_d) / np.var(y)
    out = {'distilled': np.asarray(beta_d, float), 'distillation_r2_in_fold': float(in_r2)}
    opts = {'maxfev': maxfev, 'xtol': 1e-3, 'ftol': 1e-6}
    if do_likelihood:
        res = minimize(negloglik, np.asarray(beta_d, float), args=(fit_packed, spec, norm),
                       method='Powell', options=opts)
        out.update(likelihood=np.asarray(res.x, float), likelihood_converged=bool(res.success),
                   likelihood_evaluations=int(getattr(res, 'nfev', 0)), likelihood_mean_nll=float(res.fun))
    else:
        out.update(likelihood=np.asarray(beta_d, float), likelihood_converged=False,
                   likelihood_evaluations=0, likelihood_mean_nll=None)
    if do_freerun:
        fre = minimize(freerun_loss, np.asarray(beta_d, float), args=(fit_packed, spec, norm),
                       method='Powell', options=opts)
        out.update(freerun=np.asarray(fre.x, float), freerun_converged=bool(fre.success),
                   freerun_evaluations=int(getattr(fre, 'nfev', 0)),
                   freerun_mean_squared_log_gap=float(fre.fun))
    else:
        out.update(freerun=np.asarray(beta_d, float), freerun_converged=False, freerun_evaluations=0,
                   freerun_mean_squared_log_gap=None)
    out['freerun_mean_squared_log_gap_at_distilled'] = float(freerun_loss(np.asarray(beta_d, float),
                                                                            fit_packed, spec, norm))
    return out


def load_panel(frame, extract_root):
    """Markets and the network's own variance paths for every market in the frame."""
    from pathlib import Path
    markets = {n: load_market(_E.DATA / (n + '.npz')) for n in sorted(frame.market.unique())}
    net_h = {}
    for n in markets:
        a = np.load(Path(extract_root) / f'{n}.npz', allow_pickle=False)
        if not np.array_equal(a['date'].astype('U10'), markets[n]['date']):
            raise RuntimeError(f'{n}: extracted calendar does not match the market calendar')
        net_h[n] = a['h']
    return markets, net_h


# ----------------------------------------------------------------------------- paired statistics
def paired_block(frame, candidate, control, seed=927601, reps=2000, block=60):
    """research.prior_adaptation.experiment.paired_calendar_interval, reproduced exactly, plus the
    bootstrap samples, a two-sided bootstrap p-value and a per-market Wilcoxon signed-rank test."""
    pivot = frame.pivot(index='date', columns='market', values=[candidate, control])
    diff = (pivot[candidate] - pivot[control]).to_numpy()
    n = len(diff); rng = np.random.default_rng(seed); sampled = []
    for _ in range(reps // 100):
        starts = rng.integers(0, n, size=(100, int(np.ceil(n / block))))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(100, -1)[:, :n]
        sampled.extend(np.nanmean(np.nanmean(diff[indices], axis=1), axis=1).tolist())
    sampled = np.asarray(sampled)
    market_means = frame.groupby('market')[[candidate, control]].mean()
    delta = (market_means[candidate] - market_means[control]).to_numpy()
    p_boot = 2 * min(float(np.mean(sampled <= 0)), float(np.mean(sampled >= 0)))
    p_boot = max(p_boot, 1.0 / len(sampled))
    try:
        p_wil = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
    except ValueError:
        p_wil = 1.0
    if not np.isfinite(p_wil):
        p_wil = 1.0
    return {'mean_difference': float(delta.mean()),
            'calendar_block_interval95': np.quantile(sampled, [.025, .975]).tolist(),
            'bootstrap_p_two_sided': float(p_boot), 'wilcoxon_p_two_sided': p_wil,
            'markets_improved': int((delta > 0).sum()), 'markets': int(len(delta)),
            'seed': seed, 'resamples': int(len(sampled)), 'block_observed_dates': block,
            'scope': 'Exploratory interval conditional on fitted models and chosen markets; resamples '
                     'dates jointly across markets, not training/fold uncertainty.'}


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1., running)
    return adj.tolist()


def wide(scored, variant, value='filter_common_t8'):
    """One column per specification for a given fit variant, indexed by (date, market)."""
    block = scored[scored.variant == variant]
    w = block.pivot_table(index=['date', 'market'], columns='spec', values=value).reset_index()
    w.columns.name = None
    return w
