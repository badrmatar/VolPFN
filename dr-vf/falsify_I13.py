#!/usr/bin/env python3
"""
falsify_I13.py -- numerical falsification sandbox for two claims about observation-driven
volatility filtering under QLIKE loss (memo "Three Theorem Targets", Idea I).

CLAIMS UNDER TEST
  I.1 (rate)  Against the TRUE log-variance path theta*_t (total variation V_T), a gain-adaptive
              causal filter has dynamic regret ~ T^{1/3} V_T^{2/3}, while any fixed-gain filter
              (EWMA / fixed-parameter GARCH(1,1)) is ~ sqrt(T V_T) in the worst case over paths.
  I.3 (gap)   sup_paths[best fixed gain in hindsight] / sup_paths[adaptive] ~ (T/V_T)^{1/6}.

WHAT IS SIMULATED (honest construction)
  * TRUE piecewise-constant log-variance paths with EXACT known total variation V (asserted, not
    estimated), in two families:
      'sparse' : K = ceil(V/DMAX) jumps of size V/K <= DMAX   (few big jumps -- the family that is
                 worst for constant-gain filters)
      'dense'  : K = round((T V^2)^{1/3}) jumps of size V/K   (many small jumps -- the family that
                 is minimax-hard for EVERY filter; jumps have size ~ (V/T)^{1/3})
  * returns r_t = exp(theta*_t / 2) eta_t, eta Gaussian or standardized Student-t.
  * loss l(h, r) = r^2/h + log h (QLIKE).  Dynamic regret = sum_t [l(hhat_t, r_t) - l(h*_t, r_t)],
    scored DIRECTLY against the true path.  No fitted oracle anywhere in the scoring.
  * Every filter is causal (hhat_t uses r_1..r_{t-1}) and is warm-started at the true h*_1.

FILTERS
  fixed-gain, best parameters IN HINDSIGHT per cell (strongest fair baseline):
      ewma_best   : h_{t+1} = (1-g) h_t + g r_t^2, g on a 40-point log grid
      garch_best  : h_{t+1} = w + a r_t^2 + b h_t, (a, b, w) on a 14 x 10 x 5 grid (w = (1-a-b) * level)
  adaptive, NO hindsight (stand-ins for the Theorem I.2 construction -- they are NOT that construction):
      fs_ewma     : Fixed-Share over 12 EWMA experts (the user's suggested stand-in)
      fs_restart  : Fixed-Share over 'dyadic-restart' experts: expert k restarts every 2^k steps
                    (k = 1..log2 T) at the ensemble forecast and then runs a DECAYING gain
                    gamma_n = max(1/(n+2), 2^-k) -- a running mean that turns into an EWMA with memory 2^k.
                    NOTE (corrected after the first sweep): a Fixed-Share bank of constant-gain EWMAs is ALSO
                    gain-adaptive -- weight migrates from fast experts right after a jump to slow experts
                    later, so the mixture's effective gain decays inside a segment.  Restart experts are a
                    more direct version of the same mechanism, not the only one.
      fs_combined : Fixed-Share over both banks ("gains x restarts" in the cheapest form).
      fs_cusum    : Fixed-Share over the EWMA bank wrapped in a detection-triggered restart (two-sided CUSUM on
                    r^2/hhat with threshold log T; on alarm all experts re-anchor and run a decaying gain back
                    to their own constant gain).  Between alarms it IS fs_ewma; a clean "gain-adaptive" filter.
      saol_restart: sleeping-experts (SAOL-style) aggregation over the dyadic-restart experts: an expert's weight
                    depends only on its advantage over the ensemble since its own interval started, so fresh
                    experts are not held back by stale history.  Closest stand-in for the Theorem I.2 wrapper.
      exp-weights rate ETA = 1/2 (= the exact Bayesian mixture for QLIKE), switching rate 1/T,
      aggregation in log-variance (Jensen-safe since l is convex in log h).
  reference (CHEATS -- uses the true jump times, never a candidate):
      oracle_restart : running mean of r^2 restarted exactly at the true jump times.  A floor only when jumps
                    are large (sparse family); on the dense family restarting at every tiny jump is a BAD
                    strategy, so there it is not a floor.

WHAT THE THEORY PREDICTS PER FAMILY (so the sweep can be read without motivated reasoning)
  sparse family : fixed-gain best ~ sqrt(T * V * DMAX)            -> exponents (T, V) ~ (0.5, 0.5)
                  gain-adaptive filters ~ K * (const + log T)      -> exponents ~ (~0.1, 1.0) for an ideal wrapper;
                  no-hindsight stand-ins pay detection/switching overheads that grow like log T or (log T)^2, so
                  (0.2-0.3, 0.7-0.8) at T <= 16k is what "the gap opens" looks like in practice.
                  FS over EWMA only: ALSO gain-adaptive via within-segment switching (fast experts after a jump,
                  slow ones later) -- the first draft of this file predicted (0.5,0.5) for it; the sweep falsified
                  that prediction (bT ~ 0.3), and the reasoning behind it (per-segment best expert) was wrong.
  dense family  : fixed-gain best  ~ T^{1/3} V^{2/3}               -> (0.33, 0.67)  (equal segments =>
                  one gain is near-optimal for all of them; constant gain is NOT suboptimal here)
                  adaptive ~ T^{1/3} V^{2/3} * log                -> (0.33, 0.67)
  => the (T/V)^{1/6} gap of Claim I.3 is a ratio of two SUPREMA over paths, attained on different
     families.  On any single family the ratio is O(1) (dense) or ~ sqrt(T)/log T (sparse).  The
     sandbox therefore reports per-family exponents AND the sup-over-families ratio.  With T spanning
     a factor of 8, (T/V)^{1/6} moves by 8^{1/6} = 1.41: the gap test has little power at these sizes
     and is reported as a consistency check, not a confirmation.

USAGE
  python3 falsify_I13.py --quick            # ~1 min smoke run
  python3 falsify_I13.py --full             # T in {2k,4k,8k,16k} x V in {4,8,16,32} x 2 families x 5 seeds
  python3 falsify_I13.py --full --heavy     # + Student-t(nu=4) rerun, adaptive updates raw vs clipped
  outputs: results.csv, report.txt, fig_rates.png, fig_gap.png, fig_heavy.png (in --out dir)
"""
import argparse, math, os, sys, time, csv, itertools
import numpy as np
from scipy.signal import lfilter
from scipy.special import logsumexp
from multiprocessing import Pool

# ----------------------------------------------------------------------------- configuration
THETA_LO, THETA_HI = -3.0, 3.0   # log-variance domain (width 6 > DMAX); variance factor e^6 ~ 400 end to end
DMAX = 2.0                       # largest single jump in log-variance (variance factor e^2 ~ 7.4)
H_FLOOR = 1e-8
ETA = 0.5                        # exp-weights rate: l/2 = -log N(r; 0, h) + const, so ETA=1/2 is Bayes
N0_PRIOR = 2.0                   # pseudo-count for restarted running-mean experts
CLIP_C = 9.0                     # clipped update: r^2 -> min(r^2, CLIP_C * h_expert)  (|r| <= 3 sigma_expert)
GAMMAS_HINDSIGHT = np.geomspace(5e-4, 0.5, 40)
ALPHAS = np.geomspace(0.002, 0.5, 14)
BETAS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
LEVEL_MULT = [0.25, 0.5, 1.0, 2.0, 4.0]
GAMMAS_BANK = np.geomspace(1e-3, 0.5, 12)

# ----------------------------------------------------------------------------- paths & returns
def make_path(T, V, family, rng):
    """Piecewise-constant log-variance path with EXACT total variation V. Returns (theta, K, delta, times)."""
    if family == 'sparse':
        K = max(1, int(math.ceil(V / DMAX - 1e-12)))
    elif family == 'dense':
        K = max(1, int(round((T * V * V) ** (1.0 / 3.0))))
    else:
        raise ValueError(family)
    delta = V / K
    seg = T / (K + 1)
    base = np.arange(1, K + 1) * seg
    times = np.round(base + rng.uniform(-0.35, 0.35, size=K) * seg).astype(int)
    times = np.unique(np.clip(times, 1, T - 1))
    while len(times) < K:                              # (never triggers for the grids used; guard only)
        times = np.unique(np.concatenate([times, rng.integers(1, T, size=K - len(times))]))
    theta = np.empty(T)
    cur = rng.uniform(THETA_LO + delta, THETA_HI - delta)
    prev = 0
    for tj in times:
        theta[prev:tj] = cur
        s = 1.0 if rng.random() < 0.5 else -1.0
        if cur + s * delta > THETA_HI or cur + s * delta < THETA_LO:
            s = -s
        cur += s * delta
        prev = tj
    theta[prev:] = cur
    V_exact = float(np.abs(np.diff(theta)).sum())
    assert abs(V_exact - V) < 1e-9 * max(1.0, V), (V_exact, V)
    return theta, K, delta, times

def simulate_returns(theta, rng, innov='gaussian', nu=4.0):
    T = len(theta)
    if innov == 'gaussian':
        eta = rng.standard_normal(T)
    elif innov == 't':
        eta = rng.standard_t(nu, size=T) / math.sqrt(nu / (nu - 2.0))   # unit variance
    else:
        raise ValueError(innov)
    return np.exp(theta / 2.0) * eta

# ----------------------------------------------------------------------------- scoring
def qlike(h, r2):
    return r2 / h + np.log(h)

def regret(hhat, r2, hstar):
    hhat = np.maximum(hhat, H_FLOOR)
    return float(np.sum(qlike(hhat, r2) - qlike(hstar, r2)))

# ----------------------------------------------------------------------------- fixed-gain filters (linear recursions -> lfilter)
def ewma_forecasts(r2, g, h1):
    y = lfilter([g], [1.0, -(1.0 - g)], r2, zi=[(1.0 - g) * h1])[0]      # y_t = (1-g) y_{t-1} + g r2_t
    return np.concatenate([[h1], y[:-1]])                                # forecast for t uses r2_{<t}

def garch_forecasts(r2, w, a, b, h1):
    y = lfilter([1.0], [1.0, -b], w + a * r2, zi=[b * h1])[0]            # y_t = w + a r2_t + b y_{t-1}
    return np.concatenate([[h1], y[:-1]])

def best_ewma(r2, hstar):
    best = (np.inf, None)
    for g in GAMMAS_HINDSIGHT:
        R = regret(ewma_forecasts(r2, g, hstar[0]), r2, hstar)
        if R < best[0]:
            best = (R, float(g))
    return best

def best_garch(r2, hstar):
    m = float(r2.mean())
    best = (np.inf, None)
    for a in ALPHAS:
        for b in BETAS:
            if a + b >= 1.0:
                continue
            for lm in LEVEL_MULT:
                w = (1.0 - a - b) * m * lm
                R = regret(garch_forecasts(r2, w, a, b, hstar[0]), r2, hstar)
                if R < best[0]:
                    best = (R, (float(a), float(b), float(w)))
    return best

# ----------------------------------------------------------------------------- adaptive filters (no hindsight)
def restart_periods(T):
    return [2 ** k for k in range(1, int(math.floor(math.log2(T))) + 1)]

def fixed_share_bank(r2, hstar, bank='combined', clip=False, seed_switch=None):
    """Fixed-Share over a bank of experts; forecasts aggregated in log-variance. Fully causal."""
    T = len(r2)
    specs = []
    if bank in ('ewma', 'combined'):
        specs += [('ewma', float(g)) for g in GAMMAS_BANK]
    if bank in ('restart', 'combined'):
        specs += [('restart', float(p)) for p in restart_periods(T)]
    n = len(specs)
    is_ewma = np.array([s[0] == 'ewma' for s in specs])
    gam = np.array([s[1] if s[0] == 'ewma' else 0.0 for s in specs])
    per = np.array([int(s[1]) if s[0] == 'restart' else 0 for s in specs])
    is_rst = ~is_ewma
    hf = np.full(n, hstar[0])
    nobs = np.zeros(n)                                   # observations since last restart (restart experts)
    logw = np.full(n, -math.log(n))
    beta_s = 1.0 / T if seed_switch is None else seed_switch
    log_beta_mix = math.log(beta_s / n)
    log_1mb = math.log(1.0 - beta_s)
    hhat = np.empty(T)
    tt = np.arange(1, T + 1)
    for t in range(T):
        w = np.exp(logw - logw.max()); w /= w.sum()
        th = float(np.dot(w, np.log(hf)))
        h_ens = math.exp(th)
        hhat[t] = h_ens
        x = r2[t]
        xe = np.minimum(x, CLIP_C * hf) if clip else np.full(n, x)
        loss = xe / hf + np.log(hf)                       # per-expert loss (clipped innovation if clip)
        logw = logw - ETA * loss
        logw = logw - logsumexp(logw)
        logw = np.logaddexp(log_1mb + logw, log_beta_mix)  # fixed share
        # expert state updates
        hf[is_ewma] = (1.0 - gam[is_ewma]) * hf[is_ewma] + gam[is_ewma] * xe[is_ewma]
        if is_rst.any():
            rst_now = is_rst & (per > 0) & ((tt[t] % np.maximum(per, 1)) == 0)
            hf[rst_now] = h_ens; nobs[rst_now] = 0.0          # restart: re-anchor at the ensemble forecast
            nobs[is_rst] += 1.0
            gain = np.maximum(1.0 / (nobs[is_rst] + N0_PRIOR), 1.0 / np.maximum(per[is_rst], 1))   # running mean -> EWMA(1/P)
            hf[is_rst] = hf[is_rst] + gain * (xe[is_rst] - hf[is_rst])
        hf = np.maximum(hf, H_FLOOR)
    return hhat

def saol_restart(r2, hstar, clip=False):
    """Sleeping-experts / SAOL-style aggregation over dyadic-restart experts (closest stand-in for the
    Theorem I.2 'restart wrapper').  Expert k restarts at every multiple of 2^k at the ensemble forecast and
    runs the decaying gain gamma_n = max(1/(n+2), 2^-k) (running mean -> EWMA with memory 2^k).  Its weight depends ONLY on its cumulative advantage over the ensemble SINCE ITS OWN
    INTERVAL STARTED (uniform prior over levels, rate ETA), so a freshly restarted expert is not held back
    by stale history and long-window experts take over naturally as they accumulate data.  Fully causal."""
    T = len(r2)
    per = np.array(restart_periods(T)); n = len(per)
    hf = np.full(n, hstar[0]); nobs = np.zeros(n); adv = np.zeros(n)
    hhat = np.empty(T); tt = np.arange(1, T + 1)
    for t in range(T):
        hf = np.maximum(hf, H_FLOOR)
        logw = ETA * adv; w = np.exp(logw - logw.max()); w /= w.sum()
        th = float(np.dot(w, np.log(hf))); h_ens = math.exp(th); hhat[t] = h_ens
        x = r2[t]
        xe = np.minimum(x, CLIP_C * hf) if clip else np.full(n, x)
        x_ens = min(x, CLIP_C * h_ens) if clip else x
        adv += (x_ens / h_ens + th) - (xe / hf + np.log(hf))     # ensemble loss - expert loss
        rst = (tt[t] % per) == 0
        hf[rst] = h_ens; nobs[rst] = 0.0; adv[rst] = 0.0     # restart: re-anchor at the ensemble, wipe history
        nobs += 1.0
        gain = np.maximum(1.0 / (nobs + N0_PRIOR), 1.0 / per)   # running mean that turns into EWMA(1/P)
        hf = hf + gain * (xe - hf)
    return hhat

def fs_cusum_bank(r2, hstar, clip=False, rhos=(2.0, 4.0, 8.0)):
    """Gain-adaptive stand-in #3: Fixed-Share over the EWMA bank, wrapped in a DETECTION-TRIGGERED restart.
    A small bank of two-sided CUSUMs on z_t = r_t^2 / hhat_t (scale-change log-likelihood ratios for x rho and
    x 1/rho, rho in {2,4,8}; threshold b = log T + log(#detectors), i.e. false-alarm rate ~ 1/T) watches the
    ensemble's standardized innovations.  On an alarm, every EWMA expert is re-anchored at the mean of r^2 since
    the alarming CUSUM started climbing and then runs the decaying gain gamma_n = max(1/(n+2), gamma_j) back
    down to its own constant gain.  Between alarms it is exactly Fixed-Share over constant-gain EWMAs.
    Nothing is tuned in hindsight.  Fully causal."""
    T = len(r2)
    gam = np.array(GAMMAS_BANK, dtype=float); n = len(gam)
    hf = np.full(n, hstar[0]); nobs = np.full(n, 1e12)          # pure EWMA until the first alarm
    logw = np.full(n, -math.log(n))
    beta_s = 1.0 / T; log_beta_mix = math.log(beta_s / n); log_1mb = math.log(1.0 - beta_s)
    rr = np.array(list(rhos) + [1.0 / r for r in rhos])          # up-scales then down-scales
    cc = 0.5 * (1.0 - 1.0 / rr); kk = 0.5 * np.log(rr)           # LLR_i(z) = cc_i z - kk_i
    nd = len(rr); b = math.log(T) + math.log(nd)
    S = np.zeros(nd); t0 = np.zeros(nd, dtype=int)
    hhat = np.empty(T)
    for t in range(T):
        w = np.exp(logw - logw.max()); w /= w.sum()
        th = float(np.dot(w, np.log(hf))); h_ens = math.exp(th); hhat[t] = h_ens
        x = r2[t]
        xe = np.minimum(x, CLIP_C * hf) if clip else np.full(n, x)
        z = min(x, CLIP_C * h_ens) / h_ens if clip else x / h_ens
        loss = xe / hf + np.log(hf)
        logw = logw - ETA * loss; logw = logw - logsumexp(logw); logw = np.logaddexp(log_1mb + logw, log_beta_mix)
        nobs += 1.0
        gain = np.maximum(1.0 / (nobs + N0_PRIOR), gam)
        hf = np.maximum(hf + gain * (xe - hf), H_FLOOR)
        # detectors
        S = S + cc * z - kk
        zero = S <= 0.0
        S[zero] = 0.0; t0[zero] = t + 1
        if S.max() > b:
            i = int(np.argmax(S)); seg = r2[t0[i]:t + 1]
            fresh = float(np.mean(np.minimum(seg, CLIP_C * h_ens))) if clip else float(np.mean(seg))
            hf[:] = max(fresh, H_FLOOR); nobs[:] = float(len(seg))
            S[:] = 0.0; t0[:] = t + 1
    return hhat

def oracle_restart(r2, hstar, times):
    """CHEATS: running mean of r^2 restarted at the true jump times (prior pseudo-count at its own last value)."""
    T = len(r2)
    hhat = np.empty(T)
    cnt, ssum = N0_PRIOR, N0_PRIOR * hstar[0]
    restart = np.zeros(T, dtype=bool); restart[np.asarray(times)] = True
    for t in range(T):
        h = ssum / cnt
        hhat[t] = h
        if restart[t]:
            cnt, ssum = N0_PRIOR, N0_PRIOR * h
        cnt += 1.0; ssum += r2[t]
    return hhat

# ----------------------------------------------------------------------------- one cell
def run_cell(args):
    T, V, family, seed, innov, nu, clip_variants = args
    rng = np.random.default_rng(seed * 1_000_003 + int(T) * 7919 + int(V * 13) + (0 if family == 'sparse' else 1) + (0 if innov == 'gaussian' else 100))
    theta, K, delta, times = make_path(T, V, family, rng)
    r = simulate_returns(theta, rng, innov, nu)
    r2 = r * r
    hstar = np.exp(theta)
    out = dict(T=T, V=V, family=family, seed=seed, innov=innov, nu=nu if innov == 't' else float('nan'), K=K, delta=delta)
    R_e, g_e = best_ewma(r2, hstar);   out['ewma_best'] = R_e;  out['ewma_gamma'] = g_e
    R_g, p_g = best_garch(r2, hstar);  out['garch_best'] = R_g; out['garch_alpha'], out['garch_beta'], out['garch_omega'] = p_g
    out['fixed_best'] = min(R_e, R_g)
    out['oracle_restart'] = regret(oracle_restart(r2, hstar, times), r2, hstar)
    for clip in clip_variants:
        suf = '_clip' if clip else ''
        out['fs_ewma' + suf] = regret(fixed_share_bank(r2, hstar, bank='ewma', clip=clip), r2, hstar)
        out['fs_restart' + suf] = regret(fixed_share_bank(r2, hstar, bank='restart', clip=clip), r2, hstar)
        out['fs_combined' + suf] = regret(fixed_share_bank(r2, hstar, bank='combined', clip=clip), r2, hstar)
        out['saol_restart' + suf] = regret(saol_restart(r2, hstar, clip=clip), r2, hstar)
        out['fs_cusum' + suf] = regret(fs_cusum_bank(r2, hstar, clip=clip), r2, hstar)
    return out

# ----------------------------------------------------------------------------- fitting
def fit_loglog(rows, key):
    """OLS of log(mean regret) on [1, log T, log V] over the (T, V) cells in rows. Returns dict."""
    cells = {}
    for r in rows:
        cells.setdefault((r['T'], r['V']), []).append(r[key])
    pts = [(T, V, float(np.mean(v))) for (T, V), v in sorted(cells.items())]
    pts = [(T, V, m) for T, V, m in pts if m > 0]
    if len(pts) < 4:
        return None
    X = np.array([[1.0, math.log(T), math.log(V)] for T, V, m in pts])
    y = np.array([math.log(m) for T, V, m in pts])
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(1, len(y) - 3)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()) if len(y) > 1 else float('nan')
    return dict(bT=coef[1], seT=se[1], bV=coef[2], seV=se[2], r2=r2, n=len(y))

def fmt_exp(f):
    if f is None:
        return '   (n/a)'
    return f"bT={f['bT']:+.2f}±{f['seT']:.2f}  bV={f['bV']:+.2f}±{f['seV']:.2f}  R²={f['r2']:.3f}"

# ----------------------------------------------------------------------------- main sweep
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--heavy', action='store_true', help='add Student-t rerun with raw vs clipped adaptive updates')
    ap.add_argument('--nu', type=float, default=4.0)
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--procs', type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument('--out', default='out_I13')
    ap.add_argument('--report-only', action='store_true', help='rebuild report/figures from an existing results.csv in --out')
    a = ap.parse_args()
    if a.full:
        Ts, Vs, seeds = [2000, 4000, 8000, 16000], [4.0, 8.0, 16.0, 32.0], (a.seeds or 5)
        Th, Vh = [4000, 16000], [8.0, 32.0]
    else:
        Ts, Vs, seeds = [2000, 8000], [4.0, 32.0], (a.seeds or 2)
        Th, Vh = [8000], [32.0]
    os.makedirs(a.out, exist_ok=True)
    if a.report_only:
        rows = []
        with open(os.path.join(a.out, 'results.csv')) as f:
            for r in csv.DictReader(f):
                rr = {}
                for k, v in r.items():
                    if k in ('family', 'innov'):
                        rr[k] = v
                    elif k in ('T', 'K', 'seed'):
                        rr[k] = int(float(v))
                    else:
                        rr[k] = float(v) if v not in ('', 'nan') else float('nan')
                rows.append(rr)
        report = build_report(rows, Ts, Vs, a)
        with open(os.path.join(a.out, 'report.txt'), 'w') as f:
            f.write(report)
        print(report)
        make_figures(rows, Ts, Vs, a)
        return
    jobs = [(T, V, fam, s, 'gaussian', a.nu, (False,)) for T in Ts for V in Vs for fam in ('sparse', 'dense') for s in range(seeds)]
    if a.heavy:
        jobs += [(T, V, fam, s, 't', a.nu, (False, True)) for T in Th for V in Vh for fam in ('sparse', 'dense') for s in range(seeds)]
    t0 = time.time()
    print(f"running {len(jobs)} cells on {a.procs} process(es) ...", flush=True)
    with Pool(a.procs) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run_cell, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  {i+1}/{len(jobs)} cells done, {time.time()-t0:.0f}s", flush=True)
    keys = sorted(set(k for r in rows for k in r.keys()))
    with open(os.path.join(a.out, 'results.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for r in sorted(rows, key=lambda r: (r['innov'], r['family'], r['T'], r['V'], r['seed'])):
            wr.writerow(r)
    report = build_report(rows, Ts, Vs, a)
    with open(os.path.join(a.out, 'report.txt'), 'w') as f:
        f.write(report)
    print(report)
    try:
        make_figures(rows, Ts, Vs, a)
    except Exception as e:  # figures are optional
        print('figure generation failed:', repr(e))

# ----------------------------------------------------------------------------- report
ADAPT = ['fs_ewma', 'fs_restart', 'fs_combined', 'saol_restart', 'fs_cusum']
FIXED = ['ewma_best', 'garch_best', 'fixed_best']

def mean_table(rows, key, Ts, Vs):
    out = {}
    for T in Ts:
        for V in Vs:
            v = [r[key] for r in rows if r['T'] == T and r['V'] == V]
            out[(T, V)] = (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float('nan'))
    return out

def build_report(rows, Ts, Vs, a):
    L = []
    P = L.append
    g_rows = [r for r in rows if r['innov'] == 'gaussian']
    P('=' * 100)
    P('FALSIFICATION SANDBOX -- Claims I.1 (rate) and I.3 (gap), QLIKE dynamic regret vs the TRUE log-variance path')
    P(f'T in {Ts}, V in {Vs}, families sparse/dense, seeds per cell = {len(set(r["seed"] for r in g_rows))}, Gaussian innovations')
    P('Regret = sum_t [l(hhat_t,r_t) - l(h*_t,r_t)], l(h,r) = r^2/h + log h; all filters causal and warm-started at h*_1.')
    P('fixed-gain baselines = best parameters in hindsight per cell; adaptive banks = no hindsight; oracle_restart CHEATS (true jump times).')
    P('=' * 100)
    for fam in ('sparse', 'dense'):
        fr = [r for r in g_rows if r['family'] == fam]
        Ks = sorted(set((r['T'], r['V'], r['K'], round(r['delta'], 3)) for r in fr))
        P('')
        P(f'--- family: {fam}  (K jumps, jump size delta by cell: ' + ', '.join(f'T={T},V={V:g}:K={K},d={d}' for T, V, K, d in Ks) + ')')
        P('mean regret over seeds (± s.e.), rows = filter, cols = (T, V):')
        header = 'filter'.ljust(16) + ''.join(f'{f"T={T//1000}k,V={V:g}":>16}' for T in Ts for V in Vs)
        P(header)
        for key in FIXED + ADAPT + ['oracle_restart']:
            tab = mean_table(fr, key, Ts, Vs)
            P(key.ljust(16) + ''.join(f'{tab[(T, V)][0]:>11.1f}±{tab[(T, V)][1]:<4.0f}' for T in Ts for V in Vs))
        P('')
        P('log-log fits  log(mean regret) = a + bT log T + bV log V   (OLS over the T x V cells)')
        pred = {'sparse': {'ewma_best': '(0.50, 0.50) predicted', 'garch_best': '(0.50, 0.50) predicted', 'fixed_best': '(0.50, 0.50) predicted',
                           'fs_ewma': '(0.2-0.3, 0.7-0.8) expected: gain-adaptive via within-segment switching (first-draft prediction (0.5,0.5) was wrong)',
                           'fs_restart': '(~0.1, ~1.0) predicted: K*(const+log T) with K = V/DMAX',
                           'fs_combined': '(~0.1, ~1.0) predicted (EWMA + restart experts)',
                           'saol_restart': '(~0.1, ~1.0) predicted (sleeping-experts restart wrapper)',
                           'fs_cusum': '(~0.1, ~1.0) predicted (EWMA bank + detection-triggered restart)', 'oracle_restart': '(~0.1, ~1.0) floor'},
                'dense': {k: '(0.33, 0.67) predicted for EVERY filter (equal segments)' for k in FIXED + ADAPT + ['oracle_restart']}}[fam]
        for key in FIXED + ADAPT + ['oracle_restart']:
            P(f'  {key:<16}{fmt_exp(fit_loglog(fr, key))}    {pred[key]}')
    # ------------------------------------------------------------------ sup over families
    P('')
    P('--- Claim I.3: ratio of SUPREMA over path families, per (T, V):  sup_fam fixed_best / sup_fam adaptive')
    P('    predicted ~ (T/V)^{1/6}: exponents (bT, bV) = (+0.17, -0.17).  NOTE: over T in [2k,16k] the prediction moves by only 8^{1/6} = 1.41,')
    P('    and log factors in the adaptive rate are of the same size, so this is a weak test at these sizes (a consistency check).')
    sup_rows = []
    for T in Ts:
        for V in Vs:
            supF = max(np.mean([r['fixed_best'] for r in g_rows if r['T'] == T and r['V'] == V and r['family'] == fam]) for fam in ('sparse', 'dense'))
            rec = dict(T=T, V=V, seed=0, supF=supF)
            for key in ADAPT:
                supA = max(np.mean([r[key] for r in g_rows if r['T'] == T and r['V'] == V and r['family'] == fam]) for fam in ('sparse', 'dense'))
                rec['ratio_' + key] = supF / supA
                rec['supA_' + key] = supA
            sup_rows.append(rec)
    P('ratio (sup fixed / sup adaptive):')
    P('adaptive'.ljust(16) + ''.join(f'{f"T={T//1000}k,V={V:g}":>16}' for T in Ts for V in Vs))
    for key in ADAPT:
        P(key.ljust(16) + ''.join(f'{[s for s in sup_rows if s["T"] == T and s["V"] == V][0]["ratio_" + key]:>16.2f}' for T in Ts for V in Vs))
    for key in ADAPT:
        P(f'  ratio fit {key:<12}{fmt_exp(fit_loglog(sup_rows, "ratio_" + key))}    predicted (+0.17, -0.17)')
    P('which family attains each sup (fixed_best / adaptive):')
    for T in Ts:
        for V in Vs:
            famF = max(('sparse', 'dense'), key=lambda fam: np.mean([r['fixed_best'] for r in g_rows if r['T'] == T and r['V'] == V and r['family'] == fam]))
            famA = {key: max(('sparse', 'dense'), key=lambda fam: np.mean([r[key] for r in g_rows if r['T'] == T and r['V'] == V and r['family'] == fam])) for key in ADAPT}
            P(f'  T={T}, V={V:g}: fixed sup on {famF:<6} | ' + ', '.join(f'{k}: {v}' for k, v in famA.items()))
    # ------------------------------------------------------------------ verdicts
    P('')
    P('--- PLAIN-LANGUAGE VERDICTS (heuristic thresholds; read the numbers, not the labels)')
    fs = {fam: {k: fit_loglog([r for r in g_rows if r['family'] == fam], k) for k in FIXED + ADAPT} for fam in ('sparse', 'dense')}
    def near(f, tT, tV, tol=0.12):
        return f is not None and abs(f['bT'] - tT) <= tol and abs(f['bV'] - tV) <= tol
    fsp, fde = fs['sparse'], fs['dense']
    P(f"  [I.1 fixed-gain worst case ~ sqrt(TV)]  sparse-family fixed_best exponents {fmt_exp(fsp['fixed_best'])}: "
      + ('CONSISTENT with (0.5, 0.5)' if near(fsp['fixed_best'], 0.5, 0.5) else 'NOT within ±0.12 of (0.5, 0.5) -> claim in doubt at these sizes'))
    P(f"  [I.1 adaptive ~ T^(1/3) V^(2/3) on the hard family]  dense-family fs_cusum {fmt_exp(fde['fs_cusum'])}, saol_restart {fmt_exp(fde['saol_restart'])}, fs_combined {fmt_exp(fde['fs_combined'])}: "
      + ('CONSISTENT with (0.33, 0.67)' if (near(fde['fs_cusum'], 1/3, 2/3, 0.15) or near(fde['saol_restart'], 1/3, 2/3, 0.15) or near(fde['fs_combined'], 1/3, 2/3, 0.15)) else 'NOT within ±0.15 of (0.33, 0.67) -> rate claim in doubt (or log factors dominate at these T)'))
    P(f"  [I.1 fixed-gain on the hard family]  dense fixed_best {fmt_exp(fde['fixed_best'])}: theory says constant gain is NOT rate-suboptimal here (equal segments) -> expect (0.33, 0.67) too.")
    P(f"  [I.3 mechanism]  sparse family: adaptive restart bank should grow like ~V*log T while fixed-gain grows like sqrt(TV):")
    P(f"      fs_cusum {fmt_exp(fsp['fs_cusum'])}  |  saol_restart {fmt_exp(fsp['saol_restart'])}  vs  fixed_best {fmt_exp(fsp['fixed_best'])}  -> "
      + ('gap OPENS with T (adaptive bT << fixed bT)' if (fsp['fs_cusum'] and fsp['fixed_best'] and fsp['fixed_best']['bT'] - min(fsp['fs_cusum']['bT'], fsp['saol_restart']['bT']) > 0.2) else 'NO widening gap in T -> Claim I.3 mechanism not visible'))
    P(f"  [EWMA-bank note]  sparse family fs_ewma {fmt_exp(fsp['fs_ewma'])}: a Fixed-Share bank of constant gains is itself gain-adaptive (within-segment switching), so a T-exponent below the fixed-gain 0.5 is expected; the first draft of this sandbox predicted (0.5,0.5) for it and was wrong.")
    famA_all = [max(('sparse', 'dense'), key=lambda fam: np.mean([r['fs_cusum'] for r in g_rows if r['T'] == T and r['V'] == V and r['family'] == fam])) for T in Ts for V in Vs]
    P(f"  [I.3 diagnostic power]  the adaptive supremum is attained on the sparse family in {famA_all.count('sparse')}/{len(famA_all)} cells (fs_cusum). Claim I.3's (T/V)^(1/6) comes from sparse(fixed) / DENSE(adaptive);")
    P("      until the adaptive sup moves to the dense family (needs T^(1/3) V^(2/3) > V*(const+log T), i.e. much larger T), the fitted ratio exponents measure sparse/sparse and agreement with (0.17,-0.17) is consistency, not confirmation.")
    P('  What failure would look like: adaptive bT ~ 0.5 and bV ~ 0.5 on the dense family (no T^(1/3) regime); fixed_best bV ~ 1.0 on sparse (no sqrt(V));')
    P('  restart bank bT ~ 0.5 on sparse (restarts buy nothing); sup-ratio falling with T.')
    # ------------------------------------------------------------------ heavy tails
    h_rows = [r for r in rows if r['innov'] == 't']
    if h_rows:
        P('')
        P(f"--- HEAVY-TAIL RERUN: Student-t(nu={a.nu:g}) innovations (nu=4: infinite 4th moment, OUTSIDE assumption (A1); (A1') with delta<2)")
        P('    adaptive banks run twice: raw updates vs CLIPPED updates (r^2 -> min(r^2, 9 h_expert) inside the expert/weight updates only; scoring loss unclipped)')
        Th = sorted(set(r['T'] for r in h_rows)); Vh = sorted(set(r['V'] for r in h_rows))
        for fam in ('sparse', 'dense'):
            fr = [r for r in h_rows if r['family'] == fam]
            P(f'  family {fam}:')
            P('  ' + 'filter'.ljust(20) + ''.join(f'{f"T={T//1000}k,V={V:g}":>16}' for T in Th for V in Vh))
            for key in FIXED + ['fs_ewma', 'fs_ewma_clip', 'fs_restart', 'fs_restart_clip', 'fs_combined', 'fs_combined_clip', 'saol_restart', 'saol_restart_clip', 'fs_cusum', 'fs_cusum_clip', 'oracle_restart']:
                tab = mean_table(fr, key, Th, Vh)
                P('  ' + key.ljust(20) + ''.join(f'{tab[(T, V)][0]:>11.1f}±{tab[(T, V)][1]:<4.0f}' for T in Th for V in Vh))
            for key in ('fs_cusum', 'saol_restart', 'fs_restart', 'fs_combined', 'fs_ewma'):
                imp = [(r[key] - r[key + '_clip']) / r[key] for r in fr if r[key] > 0]
                P(f'    clipping changes {key} regret by {100*np.mean(imp):+.1f}% on average (positive = clipping helps), cells: {len(imp)}')
            # compare against gaussian same cells
            for key in ('fixed_best', 'fs_cusum', 'fs_cusum_clip', 'saol_restart', 'saol_restart_clip'):
                gt = mean_table([r for r in g_rows if r['family'] == fam and r['T'] in Th and r['V'] in Vh], key.replace('_clip', ''), Th, Vh)
                ht = mean_table(fr, key, Th, Vh)
                P(f'    {key:<16} t/gaussian regret ratio: ' + ', '.join(f'{ht[(T, V)][0]/gt[(T, V)][0]:.2f}' for T in Th for V in Vh))
        P('  READING THE CLIPPING RESULT: hard-clipping r^2 at 9*h inside an update is a BIASED estimator of the variance (E[min(r^2, 9h)] < h*; for t(4)')
        P('  innovations the bias is of order 15-25%), and QLIKE penalises under-forecasting exponentially, so the bias costs ~0.02-0.03 nats per step, i.e. a')
        P('  LINEAR-in-T penalty that swamps the variance reduction.  This is exactly the bias term Theorem I.4 must control: the clipping level has to grow')
        P('  with T (or the update must use a bounded score that stays mean-zero under the innovation law, as the Student-t/DCS score does), not sit at 3 sigma.')
        P('  The one place fixed clipping helps is the CUSUM detector (fs_cusum): unclipped, a single outlier r^2/h ~ 100 exceeds the log T threshold and')
        P('  triggers a false restart; clipped, the detector survives.  Two different roles for clipping -- update bias vs alarm robustness.')
    P('')
    P('CAVEATS: (1) adaptive banks are stand-ins, not the Theorem I.2 construction; (2) exponents are fitted over T spanning a factor 8 and V a factor 8,')
    P('so ±0.1 on an exponent is the realistic resolution; (3) Claim I.3 is a statement about suprema over paths -- per-family ratios are NOT the claim;')
    P("(4) 'dense' family K = round((TV^2)^(1/3)) is the theory's own hard instance; a different hard family could give different constants but not exponents.")
    return '\n'.join(L)

# ----------------------------------------------------------------------------- figures
def make_figures(rows, Ts, Vs, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
    C = {'ewma_best': '#2a78d6', 'garch_best': '#eb6834', 'fs_ewma': '#1baf7a', 'saol_restart': '#eda100', 'fs_cusum': '#e87ba4', 'oracle_restart': '#898781'}
    NAME = {'ewma_best': 'EWMA, best gain in hindsight', 'garch_best': 'GARCH(1,1), best params in hindsight', 'fs_ewma': 'Fixed-Share over EWMA bank (no hindsight)',
            'saol_restart': 'SAOL-style dyadic-restart wrapper (no hindsight)', 'fs_cusum': 'EWMA bank + CUSUM-triggered restart (no hindsight)', 'oracle_restart': 'oracle restart (cheats: true jump times)'}
    g_rows = [r for r in rows if r['innov'] == 'gaussian']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=SURF)
    for i, fam in enumerate(('sparse', 'dense')):
        fr = [r for r in g_rows if r['family'] == fam]
        for j, mode in enumerate(('T', 'V')):
            ax = axes[i, j]; ax.set_facecolor(SURF)
            if mode == 'T':
                Vfix = Vs[-1]; xs = Ts
                getter = lambda key: [np.mean([r[key] for r in fr if r['T'] == T and r['V'] == Vfix]) for T in Ts]
                ax.set_xlabel('T (log scale)'); ax.set_title(f'{fam} family, V = {Vfix:g}: regret vs T', loc='left', fontsize=11)
            else:
                Tfix = Ts[-1]; xs = Vs
                getter = lambda key: [np.mean([r[key] for r in fr if r['T'] == Tfix and r['V'] == V]) for V in Vs]
                ax.set_xlabel('V_T (log scale)'); ax.set_title(f'{fam} family, T = {Tfix}: regret vs V_T', loc='left', fontsize=11)
            for key in ('ewma_best', 'garch_best', 'fs_ewma', 'saol_restart', 'fs_cusum', 'oracle_restart'):
                ys = getter(key)
                ax.plot(xs, ys, color=C[key], lw=2, marker='o', ms=6, ls='--' if key == 'oracle_restart' else '-', label=NAME[key])
            # slope guides anchored at the first fixed_best point
            y0 = getter('fixed_best')[0]; x0 = xs[0]
            guides = [(0.5, 'slope 1/2'), (1/3, 'slope 1/3')] if mode == 'T' else [(0.5, 'slope 1/2'), (2/3, 'slope 2/3'), (1.0, 'slope 1')]
            for s, lab in guides:
                xg = np.array([xs[0], xs[-1]]); ax.plot(xg, y0 * (xg / x0) ** s, color=MUTED, lw=1, ls=':', alpha=0.9)
                ax.annotate(lab, (xg[-1], y0 * (xg[-1] / x0) ** s), color=MUTED, fontsize=8, xytext=(4, 0), textcoords='offset points', va='center')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel('dynamic regret (QLIKE units)')
            ax.grid(True, which='major', color=GRID, lw=0.8); ax.grid(False, which='minor')
            for s in ('top', 'right'): ax.spines[s].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Claim I.1 -- QLIKE dynamic regret against the true path (mean over seeds); dotted guides are reference slopes', x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96)); fig.savefig(os.path.join(a.out, 'fig_rates.png'), dpi=150, facecolor=SURF); plt.close(fig)
    # gap figure
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURF); ax.set_facecolor(SURF)
    for k, V in enumerate(Vs):
        ratios = []
        for T in Ts:
            supF = max(np.mean([r['fixed_best'] for r in g_rows if r['T'] == T and r['V'] == V and r['family'] == fam]) for fam in ('sparse', 'dense'))
            supA = max(np.mean([r['fs_cusum'] for r in g_rows if r['T'] == T and r['V'] == V and r['family'] == fam]) for fam in ('sparse', 'dense'))
            ratios.append(supF / supA)
        col = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'][k % 4]
        ax.plot(Ts, ratios, color=col, lw=2, marker='o', ms=6, label=f'V = {V:g}')
        ax.annotate(f'V={V:g}', (Ts[-1], ratios[-1]), color=INK2, fontsize=8, xytext=(5, 0), textcoords='offset points', va='center')
    xg = np.array([Ts[0], Ts[-1]]); y0 = ax.get_ylim()[0] * 1.02
    ax.plot(xg, y0 * (xg / xg[0]) ** (1 / 6), color=MUTED, lw=1, ls=':'); ax.annotate('slope 1/6 (predicted)', (xg[0], y0), color=MUTED, fontsize=8, xytext=(4, -10), textcoords='offset points', va='top')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('T (log scale)'); ax.set_ylabel('sup fixed_best / sup fs_cusum')
    ax.set_title('Claim I.3 -- sup fixed-gain / sup adaptive (CUSUM-restart bank)', loc='left', fontsize=11)
    ax.grid(True, which='major', color=GRID, lw=0.8); ax.grid(False, which='minor'); ax.legend(frameon=False, fontsize=8)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'fig_gap.png'), dpi=150, facecolor=SURF); plt.close(fig)
    # heavy-tail figure
    h_rows = [r for r in rows if r['innov'] == 't']
    if h_rows:
        Th = sorted(set(r['T'] for r in h_rows)); Vh = sorted(set(r['V'] for r in h_rows))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURF)
        for i, fam in enumerate(('sparse', 'dense')):
            ax = axes[i]; ax.set_facecolor(SURF)
            fr = [r for r in h_rows if r['family'] == fam]
            cells = [(T, V) for T in Th for V in Vh]; x = np.arange(len(cells)); wd = 0.13
            series = [('fixed_best', 'fixed, best in hindsight', '#2a78d6'), ('fs_cusum', 'CUSUM-restart bank, raw', '#eda100'), ('fs_cusum_clip', 'CUSUM-restart bank, clipped', '#e87ba4'),
                      ('saol_restart', 'SAOL restart wrapper, raw', '#1baf7a'), ('saol_restart_clip', 'SAOL restart wrapper, clipped', '#eb6834'), ('oracle_restart', 'oracle restart', '#898781')]
            for k, (key, lab, col) in enumerate(series):
                ys = [np.mean([r[key] for r in fr if r['T'] == T and r['V'] == V]) for T, V in cells]
                ax.bar(x + (k - 2.5) * wd, ys, width=wd * 0.92, color=col, label=lab, edgecolor=SURF, linewidth=1)
            ax.set_xticks(x); ax.set_xticklabels([f'T={T//1000}k\nV={V:g}' for T, V in cells], fontsize=8)
            ax.axhline(0, color='#c3c2b7', lw=1); ax.set_ylabel('dynamic regret (QLIKE units)'); ax.set_title(f'Student-t(nu={a.nu:g}) rerun, {fam} family (negative bar = realized regret below zero, noise)', loc='left', fontsize=10)
            ax.grid(True, axis='y', color=GRID, lw=0.8); ax.grid(False, which='minor')
            for s in ('top', 'right'): ax.spines[s].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'fig_heavy.png'), dpi=150, facecolor=SURF); plt.close(fig)

if __name__ == '__main__':
    main()
