#!/usr/bin/env python3
"""
phase1.py -- Phase 1 of the Idea-I programme: sharpen the claims numerically before proving them.
Requires falsify_I13.py in the same directory (baselines and stand-ins are imported from it).

SUBCOMMANDS
  flh-bayes   Steps 1-3.  DGP: exact lattice jump prior (each step jumps with prob lam = K/T by +-delta; sign
              fair).  Filters: hindsight-best fixed gain (EWMA/GARCH grids), the stand-ins from the first
              sandbox (fs_ewma, fs_cusum), the REAL construction FLH-FTL (Hazan-Seshadhri 2009 Follow-the-
              Leading-History with the efficient O(log T) working set; base learner = follow-the-leader on the
              QLIKE loss, which for this loss is exactly the running mean of r^2 = the Newton step), and the
              EXACT Bayes-optimal causal filter for the DGP (finite-state HMM forward filter on the jump
              lattice; Bayes act under QLIKE = log posterior-mean variance).  Bayes risk <= minimax risk, so
              the Bayes filter is a true floor for every causal filter on this prior.
              Regimes: sparse (delta = 2, K = V/2) with T up to 2^18; dense (K = (T V^2)^{1/3}, delta = V/K).
              Reports per-jump cost vs log T (log-linear vs power-law fit), exponents, and FLH/Bayes ratios.
  heavy       Step 4.  Student-t(nu) innovations, nu in {3, 4, 6, gaussian}.  FLH-FTL raw vs relative clipping
              at c in {9, 30, 100} vs bounded Student-t (DCS) score experts with a working nu_w = 5 and with the
              true nu; Bayes filter (knows nu) as floor; hindsight fixed gain as reference.
  stoch       Step 5.  (a) AR(1) log-variance (the SV model) with an exact grid-HMM Bayes floor: constant gain
              should be near-optimal (GARCH regime).  (b) stationary Matern(nu) log-variance -- rough for nu<1/2
              -- with the optimal LINEAR filter (GP regression with the true kernel, window W) as reference;
              tests the per-step error exponent 2/(2H+1) in the increment scale and adaptivity across H.

USAGE
  python3 phase1.py flh-bayes --scale small|full [--procs N] [--out out_phase1]
  python3 phase1.py heavy     --scale small|full
  python3 phase1.py stoch     --scale small|full
  outputs: <out>/<subcommand>_results.csv and <out>/<subcommand>_report.txt
"""
import argparse, math, os, sys, time, csv
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')          # one BLAS thread per worker: avoids the macOS oversubscription hang
import numpy as np
from scipy.special import logsumexp
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.baselines import falsify_I13 as F
INNOV_CODE = {'gaussian': 0, '3': 3, '4': 4, '6': 6}   # deterministic seeding (hash() is randomised per process)
ETA = F.ETA            # 1/2 = Bayes mixture rate for QLIKE
N0 = F.N0_PRIOR        # pseudo-count of newborn experts
H_FLOOR = F.H_FLOOR
LOGCHI2_VAR = math.pi ** 2 / 2.0          # Var(log chi^2_1)
LOGCHI2_MEAN = -1.2703628454614782        # E[log chi^2_1] = -(gamma + log 2)

# ----------------------------------------------------------------------------- innovations
def draw_innov(T, rng, innov):
    if innov == 'gaussian':
        return rng.standard_normal(T)
    nu = float(innov)
    return rng.standard_t(nu, size=T) / math.sqrt(nu / (nu - 2.0))

def loglik_r2(r2, h, innov):
    """log p(r | variance h) up to a constant, for Gaussian or standardized Student-t innovations."""
    if innov == 'gaussian':
        return -0.5 * np.log(h) - 0.5 * r2 / h
    nu = float(innov)
    return -0.5 * np.log(h) - 0.5 * (nu + 1.0) * np.log1p(r2 / ((nu - 2.0) * h))

def xregret(hhat, hstar):
    """EXPECTED dynamic regret given the forecast path: sum_t d(theta_hat_t || theta*_t), d = e^{a} - a - 1, a = theta* - theta_hat.
    Equals E[realized QLIKE regret | forecasts] exactly (E[r_t^2 | F_{t-1}] = h*_t), so it scores against the TRUE path
    with the innovation noise removed from the SCORING only (the filters still see noisy data).  Far lower variance,
    essential under heavy tails where realized regret is dominated by a handful of extreme returns."""
    a = np.log(hstar) - np.log(np.maximum(hhat, H_FLOOR))
    return float(np.sum(np.exp(a) - a - 1.0))

def fixed_share_over_forecasts(hhats, r2, eta=ETA):
    """Fixed-Share (rate eta, switching 1/T) over already-computed causal forecast paths; log-variance averaging."""
    T = len(r2); n = len(hhats)
    H = np.log(np.maximum(np.column_stack(hhats), H_FLOOR))
    logw = np.full(n, -math.log(n)); beta = 1.0 / T
    lb, l1 = math.log(beta / n), math.log(1.0 - beta)
    out = np.empty(T)
    for t in range(T):
        w = np.exp(logw - logw.max()); w /= w.sum()
        out[t] = math.exp(float(np.dot(w, H[t])))
        loss = r2[t] * np.exp(-H[t]) + H[t]
        logw = logw - eta * loss; logw -= logsumexp(logw)
        logw = np.logaddexp(l1 + logw, lb)
    return out

# ----------------------------------------------------------------------------- DGP 1: lattice jump prior (exact)
def make_jump_path(T, K_target, delta, rng, theta0=0.0):
    lam = K_target / T
    jumps = rng.random(T) < lam
    jumps[0] = False
    signs = np.where(rng.random(T) < 0.5, -1.0, 1.0)
    theta = theta0 + np.cumsum(np.where(jumps, signs * delta, 0.0))
    return theta, lam, int(jumps.sum()), float(np.abs(np.diff(theta)).sum())

def bayes_jump_filter(r2, lam, delta, theta0, innov, J):
    """Exact Bayes-optimal causal QLIKE forecaster for the lattice jump prior. States theta0 + delta*j, |j|<=J."""
    T = len(r2)
    levels = theta0 + delta * np.arange(-J, J + 1)
    h_lev = np.exp(levels)
    logp = np.full(2 * J + 1, -np.inf); logp[J] = 0.0          # theta*_1 = theta0 known (same warm start as others)
    hhat = np.empty(T)
    l_stay, l_move = math.log(1.0 - lam), math.log(lam / 2.0)
    for t in range(T):
        if t > 0:                                                # predict step: theta*_t | r_{1:t-1}
            up = np.full_like(logp, -np.inf); up[1:] = logp[:-1] + l_move
            dn = np.full_like(logp, -np.inf); dn[:-1] = logp[1:] + l_move
            logp = np.logaddexp(logp + l_stay, np.logaddexp(up, dn))
        p = np.exp(logp - logp.max()); p /= p.sum()
        hhat[t] = float(np.dot(p, h_lev))                        # Bayes act for QLIKE: posterior-mean variance
        logp = logp + loglik_r2(r2[t], h_lev, innov)             # update with r_t
        logp -= logsumexp(logp)
    return hhat

# ----------------------------------------------------------------------------- the construction: FLH over FTL experts
def t_score(u, nu):
    """Score of the standardized Student-t(nu) log-likelihood w.r.t. log-variance, u = r^2 e^{-theta}. Bounded."""
    return 0.5 * ((nu + 1.0) * u / (nu - 2.0 + u) - 1.0)

NU_GRID = np.array([2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 50.0, 200.0])
LGAM_TERM = np.array([math.lgamma((v + 1.0) / 2.0) - math.lgamma(v / 2.0) - 0.5 * math.log((v - 2.0) * math.pi) for v in NU_GRID])

def flh_ftl(r2, hstar, expert='ftl', clip_c=None, nu_w=5.0, eta=ETA, weight_loss='qlike', nu_bank=None, nu_hat=False):
    """Follow-the-Leading-History (Hazan & Seshadhri 2009) with the efficient working set (expert born at
    time i lives 2^(k+2)+1 steps, k = number of trailing zeros of i), over base learners that are follow-the-
    leader on the QLIKE loss:
        expert='ftl' : running mean of r^2 since birth (prior pseudo-count N0 at the ensemble forecast at birth);
                       with clip_c, the innovation entering the mean is min(r^2, clip_c * h_expert).
        expert='dcs' : M-estimator in log-variance driven by the bounded Student-t(nu_w) score with Newton
                       scaling 2(nu_w+3)/nu_w and decaying gain 1/(n+1+N0) (Harvey's DCS as an FTL base).
                       nu_bank=[...]: experts are born with working nu cycling through the list (a bank over
                       innovation laws; no knowledge of the true law).
    weight_loss='qlike'  : meta-weights updated with the (unbounded) QLIKE loss  r^2/h + log h;
    weight_loss='tloglik': meta-weights updated with the expert's own Student-t predictive log-loss
                           log h + (nu+1) log(1 + r^2/((nu-2) h))  (= QLIKE as nu -> inf; grows only
                           logarithmically in r^2, so a single outlier cannot jerk the weights).
    New expert gets weight 1/t, others are scaled by (1-1/t); multiplicative update exp(-eta * loss);
    prediction = weight-average in log-variance.  Fully causal, no hindsight."""
    T = len(r2)
    M = 6 * int(math.log2(T) + 1) + 12
    alive = np.zeros(M, dtype=bool)
    death = np.zeros(M, dtype=np.int64)
    n = np.zeros(M); s = np.zeros(M); h0 = np.zeros(M); th = np.zeros(M); nus = np.full(M, float(nu_w))
    logv = np.full(M, -np.inf)
    hhat = np.empty(T)
    h_prev = float(hstar[0])
    lp_nu = np.zeros(len(NU_GRID)); nu_cur = float(nu_w)      # online profile likelihood for the innovation law
    for t in range(1, T + 1):
        k = (t & -t).bit_length() - 1
        free = np.flatnonzero(~alive)
        slot = int(free[0])
        alive[slot] = True; death[slot] = t + (1 << (k + 2)) + 1
        n[slot] = 0.0; s[slot] = 0.0; h0[slot] = h_prev; th[slot] = math.log(h_prev)
        nus[slot] = float(nu_bank[(t - 1) % len(nu_bank)]) if nu_bank else (nu_cur if nu_hat else float(nu_w))
        if t == 1:
            logv[slot] = 0.0
        else:
            logv[alive] += math.log(1.0 - 1.0 / t); logv[slot] = math.log(1.0 / t)
        dead = alive & (death <= t)
        if dead.any():
            alive[dead] = False; logv[dead] = -np.inf
            logv[alive] -= logsumexp(logv[alive])
        idx = np.flatnonzero(alive)
        if expert == 'ftl':
            hf = (N0 * h0[idx] + s[idx]) / (N0 + n[idx])
        else:
            hf = np.exp(th[idx])
        hf = np.maximum(hf, H_FLOOR)
        w = np.exp(logv[idx] - logv[idx].max()); w /= w.sum()
        theta_hat = float(np.dot(w, np.log(hf)))
        h_ens = math.exp(theta_hat); hhat[t - 1] = h_ens
        x = r2[t - 1]
        xe = np.minimum(x, clip_c * hf) if clip_c else np.full(len(idx), x)
        u = x / hf
        if nu_hat:                                                 # update the law estimate with the ensemble-standardized residual
            ue = x / h_ens
            lp_nu += -0.5 * (NU_GRID + 1.0) * np.log1p(ue / (NU_GRID - 2.0)) + LGAM_TERM
            nu_cur = float(NU_GRID[int(np.argmax(lp_nu))])
            nus[idx] = nu_cur
        if weight_loss == 'tloglik':
            nv = nus[idx]
            loss = np.log(hf) + (nv + 1.0) * np.log1p(u / (nv - 2.0))
        else:
            loss = xe / hf + np.log(hf)
        logv[idx] -= eta * loss
        logv[idx] -= logsumexp(logv[idx])
        if expert == 'ftl':
            n[idx] += 1.0; s[idx] += xe
        else:
            nv = nus[idx]
            th[idx] += (2.0 * (nv + 3.0) / nv) * t_score(u, nv) / (n[idx] + 1.0 + N0)
            n[idx] += 1.0
        h_prev = h_ens
    return hhat

# ----------------------------------------------------------------------------- DGP 2: AR(1) log-variance with grid-HMM Bayes floor
def bayes_ar1_filter(r2, phi, sig_w, theta0, innov, ngrid=161, width=4.0):
    """Exact-up-to-discretisation Bayes filter for theta_t = phi theta_{t-1} + sig_w eps_t on a grid."""
    T = len(r2)
    sd = sig_w / math.sqrt(1.0 - phi * phi)
    grid = np.linspace(-width * sd, width * sd, ngrid) + theta0
    h_lev = np.exp(grid)
    # transition matrix P[i, j] = P(theta_t = grid_j | theta_{t-1} = grid_i)
    mu = phi * (grid[:, None] - theta0) + theta0
    logP = -0.5 * ((grid[None, :] - mu) / sig_w) ** 2
    logP -= logsumexp(logP, axis=1, keepdims=True)
    P = np.exp(logP)
    p = np.zeros(ngrid); p[np.argmin(np.abs(grid - theta0))] = 1.0
    hhat = np.empty(T)
    for t in range(T):
        if t > 0:
            p = p @ P
        hhat[t] = float(np.dot(p, h_lev))
        p = p * np.exp(loglik_r2(r2[t], h_lev, innov) - loglik_r2(r2[t], h_lev, innov).max())
        p /= p.sum()
    return hhat

# ----------------------------------------------------------------------------- DGP 3: stationary Matern log-variance + optimal linear filter
def matern_cov(tau, nu, ell, amp=1.0):
    tau = np.abs(np.asarray(tau, dtype=float))
    if nu == 0.5:
        return amp * np.exp(-tau / ell)
    if nu == 1.5:
        a = math.sqrt(3.0) * tau / ell
        return amp * (1.0 + a) * np.exp(-a)
    from scipy.special import kv, gamma
    a = math.sqrt(2.0 * nu) * tau / ell
    out = np.empty_like(a)
    out[a == 0] = amp
    az = a[a > 0]
    out[a > 0] = amp * (2 ** (1 - nu) / gamma(nu)) * (az ** nu) * kv(nu, az)
    return out

def sample_stationary(T, cov_fn, rng):
    """Circulant embedding (Davies-Harte); negative eigenvalues (if any) are zeroed -- reported as a warning."""
    Mlen = 1
    while Mlen < 2 * T:
        Mlen *= 2
    row = cov_fn(np.concatenate([np.arange(Mlen // 2 + 1), np.arange(Mlen // 2 - 1, 0, -1)]))
    eig = np.fft.fft(row).real
    neg = float(-eig.min()) if eig.min() < 0 else 0.0
    eig = np.maximum(eig, 0.0)
    z = rng.standard_normal(Mlen) + 1j * rng.standard_normal(Mlen)
    x = np.fft.fft(np.sqrt(eig / Mlen) * z)
    return x.real[:T], neg

def linear_gp_filter(y, cov_fn, noise_var, W, m_small=64):
    """Optimal linear one-step-ahead predictor of theta_t from the last W noisy observations y = theta + noise
    (stationary process, true kernel), plus the Gaussian QLIKE act correction theta_hat = m + s^2/2 (posterior-mean
    variance).  Full-window weights are computed once (O(W^3)) and applied as a dot product; the first W steps use
    growing windows up to m_small observations (negligible share of the regret)."""
    lags = np.arange(W)
    k0 = float(cov_fn(np.array([0.0]))[0])
    kstar = cov_fn(np.arange(1, W + 1))            # cov(theta_t, y_{t-1..t-W})
    def weights(m):
        Km = cov_fn(np.abs(lags[:m, None] - lags[None, :m])) + noise_var * np.eye(m)
        w = np.linalg.solve(Km, kstar[:m])
        return w, k0 - float(w @ kstar[:m])
    cache = {m: weights(m) for m in range(1, m_small + 1)}
    wW, s2W = weights(W)
    T = len(y)
    out = np.empty(T)
    for t in range(T):
        if t == 0:
            out[t] = 0.0 + k0 / 2.0
        elif t < W:
            m = min(t, m_small); w, s2 = cache[m]
            out[t] = float(w @ y[t - m:t][::-1]) + s2 / 2.0
        else:
            out[t] = float(wW @ y[t - W:t][::-1]) + s2W / 2.0
    return out

def yule_walker_ar(cov_fn, p):
    from scipy.linalg import toeplitz
    r = cov_fn(np.arange(p + 1).astype(float))
    a = np.linalg.solve(toeplitz(r[:p]), r[1:p + 1])
    sig2 = float(r[0] - a @ r[1:p + 1])
    return a, max(sig2, 1e-12)

def pf_ar_reference(r2, cov_fn, rng, p=64, N=2000, innov='gaussian'):
    """Near-Bayes reference that KNOWS the latent covariance (to lag p): bootstrap particle filter for the
    AR(p) Yule-Walker approximation of the stationary latent process, chi^2 likelihood, systematic resampling.
    Forecast = posterior-mean variance (Bayes act for QLIKE).  Causal.  Cheats (knows the model)."""
    a, sig2 = yule_walker_ar(cov_fn, p); sig = math.sqrt(sig2)
    T = len(r2)
    X = np.zeros((N, p))
    for _ in range(4 * p):                                        # burn-in from the prior to reach stationarity
        new = X @ a + sig * rng.standard_normal(N)
        X = np.column_stack([new, X[:, :-1]])
    hhat = np.empty(T)
    for t in range(T):
        new = X @ a + sig * rng.standard_normal(N)                # predictive particles for theta_t
        h = np.exp(new)
        hhat[t] = float(np.mean(h))
        lw = loglik_r2(r2[t], h, innov); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
        u = (rng.random() + np.arange(N)) / N                     # systematic resampling
        idx = np.searchsorted(np.cumsum(w), u); idx = np.minimum(idx, N - 1)
        X = np.column_stack([new[idx], X[idx, :-1]])
    return hhat

# ----------------------------------------------------------------------------- cells
def score(out, key, hhat, r2, hstar):
    out[key] = F.regret(hhat, r2, hstar); out[key + '_x'] = xregret(hhat, hstar)

def cell_flh_bayes(args):
    regime, T, V, seed, run_standins = args
    rng = np.random.default_rng(10_000 + 7919 * seed + 13 * int(V) + T + (0 if regime == 'sparse' else 1_000_000))
    if regime == 'sparse':
        delta = F.DMAX; K = V / delta
    else:
        K = (T * V * V) ** (1.0 / 3.0); delta = V / K
    theta, lam, K_real, V_real = make_jump_path(T, K, delta, rng)
    J = max(int(4.0 * math.sqrt(max(K, 1.0))) + 8, int(np.max(np.abs(np.round(theta / delta)))) + 2)
    r = np.exp(theta / 2.0) * draw_innov(T, rng, 'gaussian'); r2 = r * r; hstar = np.exp(theta)
    out = dict(regime=regime, T=T, V=V, seed=seed, K_target=K, K_real=K_real, V_real=V_real, delta=delta)
    t0 = time.time()
    Re, ge = F.best_ewma(r2, hstar); Rg, pg = F.best_garch(r2, hstar)
    he = F.ewma_forecasts(r2, ge, hstar[0]); hg = F.garch_forecasts(r2, pg[2], pg[0], pg[1], hstar[0])
    score(out, 'ewma_best', he, r2, hstar); score(out, 'garch_best', hg, r2, hstar)
    score(out, 'fixed_best', he if Re <= Rg else hg, r2, hstar)
    score(out, 'bayes', bayes_jump_filter(r2, lam, delta, 0.0, 'gaussian', J), r2, hstar)
    h_flh = flh_ftl(r2, hstar, 'ftl'); score(out, 'flh_ftl', h_flh, r2, hstar)
    if run_standins:
        h_fse = F.fixed_share_bank(r2, hstar, bank='ewma'); score(out, 'fs_ewma', h_fse, r2, hstar)
        score(out, 'fs_cusum', F.fs_cusum_bank(r2, hstar), r2, hstar)
        score(out, 'meta2', fixed_share_over_forecasts([h_flh, h_fse], r2), r2, hstar)
    else:
        for k in ('fs_ewma', 'fs_cusum', 'meta2'):
            out[k] = float('nan'); out[k + '_x'] = float('nan')
    out['secs'] = time.time() - t0
    return out

def cell_heavy(args):
    regime, T, V, seed, innov = args
    rng = np.random.default_rng(20_000 + 7919 * seed + 13 * int(V) + T + (0 if regime == 'sparse' else 1_000_000) + INNOV_CODE.get(innov, 7))
    if regime == 'sparse':
        delta = F.DMAX; K = V / delta
    else:
        K = (T * V * V) ** (1.0 / 3.0); delta = V / K
    theta, lam, K_real, V_real = make_jump_path(T, K, delta, rng)
    J = max(int(4.0 * math.sqrt(max(K, 1.0))) + 8, int(np.max(np.abs(np.round(theta / delta)))) + 2)
    r = np.exp(theta / 2.0) * draw_innov(T, rng, innov); r2 = r * r; hstar = np.exp(theta)
    out = dict(regime=regime, T=T, V=V, seed=seed, innov=innov, K_real=K_real, V_real=V_real, delta=delta)
    Re, ge = F.best_ewma(r2, hstar); Rg, pg = F.best_garch(r2, hstar)
    he = F.ewma_forecasts(r2, ge, hstar[0]); hg = F.garch_forecasts(r2, pg[2], pg[0], pg[1], hstar[0])
    score(out, 'fixed_best', he if Re <= Rg else hg, r2, hstar)
    score(out, 'bayes_knows_nu', bayes_jump_filter(r2, lam, delta, 0.0, innov, J), r2, hstar)
    h_raw = flh_ftl(r2, hstar, 'ftl'); score(out, 'flh_raw', h_raw, r2, hstar)
    for c in (9.0, 30.0, 100.0):
        score(out, f'flh_clip{int(c)}', flh_ftl(r2, hstar, 'ftl', clip_c=c), r2, hstar)
    score(out, 'flh_dcs_nu5', flh_ftl(r2, hstar, 'dcs', nu_w=5.0), r2, hstar)
    nu_true = 200.0 if innov == 'gaussian' else float(innov)
    score(out, 'flh_dcs_true_nu', flh_ftl(r2, hstar, 'dcs', nu_w=nu_true), r2, hstar)
    h_fse = F.fixed_share_bank(r2, hstar, bank='ewma'); score(out, 'fs_ewma', h_fse, r2, hstar)
    score(out, 'meta2', fixed_share_over_forecasts([h_raw, h_fse], r2), r2, hstar)
    score(out, 'fs_cusum_raw', F.fs_cusum_bank(r2, hstar), r2, hstar)
    score(out, 'fs_cusum_clip9', F.fs_cusum_bank(r2, hstar, clip=True), r2, hstar)
    return out

NU_BANK = [3.0, 4.0, 6.0, 10.0, 200.0]

def cell_heavy2(args):
    regime, T, V, seed, innov = args
    rng = np.random.default_rng(50_000 + 7919 * seed + 13 * int(V) + T + (0 if regime == 'sparse' else 1_000_000) + INNOV_CODE.get(innov, 7))
    if regime == 'sparse':
        delta = F.DMAX; K = V / delta
    else:
        K = (T * V * V) ** (1.0 / 3.0); delta = V / K
    theta, lam, K_real, V_real = make_jump_path(T, K, delta, rng)
    J = max(int(4.0 * math.sqrt(max(K, 1.0))) + 8, int(np.max(np.abs(np.round(theta / delta)))) + 2)
    r = np.exp(theta / 2.0) * draw_innov(T, rng, innov); r2 = r * r; hstar = np.exp(theta)
    out = dict(regime=regime, T=T, V=V, seed=seed, innov=innov, K_real=K_real, V_real=V_real, delta=delta)
    Re, ge = F.best_ewma(r2, hstar); Rg, pg = F.best_garch(r2, hstar)
    he = F.ewma_forecasts(r2, ge, hstar[0]); hg = F.garch_forecasts(r2, pg[2], pg[0], pg[1], hstar[0])
    score(out, 'fixed_best', he if Re <= Rg else hg, r2, hstar)
    score(out, 'bayes_knows_nu', bayes_jump_filter(r2, lam, delta, 0.0, innov, J), r2, hstar)
    nu_true = 200.0 if innov == 'gaussian' else float(innov)
    score(out, 'flh_raw', flh_ftl(r2, hstar, 'ftl'), r2, hstar)
    score(out, 'dcs_true_qw', flh_ftl(r2, hstar, 'dcs', nu_w=nu_true), r2, hstar)                       # bounded base, unbounded weights
    score(out, 'dcs_true_tw', flh_ftl(r2, hstar, 'dcs', nu_w=nu_true, weight_loss='tloglik'), r2, hstar)  # bounded base, bounded weights
    score(out, 'dcs_nuhat_tw', flh_ftl(r2, hstar, 'dcs', weight_loss='tloglik', nu_hat=True), r2, hstar)  # unknown law: online profile nu
    score(out, 'dcs_nuhat_qw', flh_ftl(r2, hstar, 'dcs', nu_hat=True), r2, hstar)
    h_fse = F.fixed_share_bank(r2, hstar, bank='ewma'); score(out, 'fs_ewma', h_fse, r2, hstar)
    return out

def report_heavy2(rows_raw, Ts, innovs):
    KEYS = ['fixed_best', 'bayes_knows_nu', 'flh_raw', 'dcs_true_qw', 'dcs_true_tw', 'dcs_nuhat_tw', 'dcs_nuhat_qw', 'fs_ewma']
    rows = xrows(rows_raw, KEYS)
    L = []; P = L.append
    P('=' * 110); P('PHASE 1, STEP 4b: heavy tails -- does BOUNDED aggregation fix the drift?  (expected regret; Bayes filter knows nu)')
    P('dcs_true_qw = t-score base with the true nu, QLIKE meta-weights (the step-4 object that still drifted);')
    P('dcs_true_tw = same base, meta-weights by the bounded t predictive log-loss; dcs_nuhat_* = NO knowledge of the law: nu is profiled')
    P('online on a grid from the ensemble-standardized residuals (argmax of the cumulative t log-likelihood) and used in every score.')
    P(consistency_line(rows_raw, KEYS))
    P('=' * 110)
    for regime in ('sparse', 'dense'):
        rr = [r for r in rows if r['regime'] == regime]
        P(''); P(f'--- regime {regime}: ratio to the Bayes floor (mean over seeds); a flat row across T is the target')
        P('filter'.ljust(16) + ''.join(f'{f"T={T//1000}k,{inn}":>15}' for T in Ts for inn in innovs))
        for k in KEYS:
            P(k.ljust(16) + ''.join(f'{np.mean([r[k] / r["bayes_knows_nu"] for r in rr if r["T"] == T and r["innov"] == inn]):>15.2f}' for T in Ts for inn in innovs))
        P('mean expected regret:')
        P('filter'.ljust(16) + ''.join(f'{f"T={T//1000}k,{inn}":>15}' for T in Ts for inn in innovs))
        for k in KEYS:
            P(k.ljust(16) + ''.join('{:>10.0f}±{:<4.0f}'.format(*cellmean(rr, k, lambda r, T=T, inn=inn: r['T'] == T and r['innov'] == inn)[:2]) for T in Ts for inn in innovs))
    P('')
    P('reading: if dcs_true_tw is flat in T where dcs_true_qw drifts, the drift was the aggregation (unbounded QLIKE weights), and Theorem I.4')
    P('needs bounded losses in BOTH the base learner and the meta-weights; if dcs_nuhat_tw is close to dcs_true_tw, the innovation law need not')
    P('be known -- a 1-d static parameter profiled online costs O(1) per path, which makes I.4 assumption-free within the t family.')
    return '\n'.join(L)

def cell_stoch_ar1(args):
    T, phi, sig_w, seed = args
    rng = np.random.default_rng(30_000 + 7919 * seed + int(1000 * sig_w) + T)
    sd = sig_w / math.sqrt(1.0 - phi * phi)
    theta = np.empty(T); theta[0] = 0.0
    eps = rng.standard_normal(T)
    for t in range(1, T):
        theta[t] = phi * theta[t - 1] + sig_w * eps[t]
    r = np.exp(theta / 2.0) * draw_innov(T, rng, 'gaussian'); r2 = r * r; hstar = np.exp(theta)
    out = dict(T=T, phi=phi, sig_w=sig_w, seed=seed, sd_theta=sd, V_real=float(np.abs(np.diff(theta)).sum()))
    Re, ge = F.best_ewma(r2, hstar); Rg, pg = F.best_garch(r2, hstar)
    he = F.ewma_forecasts(r2, ge, hstar[0]); hg = F.garch_forecasts(r2, pg[2], pg[0], pg[1], hstar[0])
    out['ewma_gamma'] = ge
    score(out, 'fixed_best', he if Re <= Rg else hg, r2, hstar)
    score(out, 'bayes_grid', bayes_ar1_filter(r2, phi, sig_w, 0.0, 'gaussian'), r2, hstar)
    h_flh = flh_ftl(r2, hstar, 'ftl'); score(out, 'flh_ftl', h_flh, r2, hstar)
    h_fse = F.fixed_share_bank(r2, hstar, bank='ewma'); score(out, 'fs_ewma', h_fse, r2, hstar)
    score(out, 'fs_cusum', F.fs_cusum_bank(r2, hstar), r2, hstar)
    score(out, 'meta2', fixed_share_over_forecasts([h_flh, h_fse], r2), r2, hstar)
    return out

def cell_stoch_matern(args):
    T, nu, ell, amp, seed, W = args
    rng = np.random.default_rng(40_000 + 7919 * seed + int(100 * nu) + int(ell) + int(100 * amp) + T)
    cov_fn = lambda tau: matern_cov(tau, nu, ell, amp)
    theta, neg = sample_stationary(T, cov_fn, rng)
    r = np.exp(theta / 2.0) * draw_innov(T, rng, 'gaussian'); r2 = r * r; hstar = np.exp(theta)
    inc_sd = math.sqrt(max(2.0 * (amp - float(cov_fn(np.array([1.0]))[0])), 1e-12))
    out = dict(T=T, nu=nu, ell=ell, amp=amp, seed=seed, W=W, inc_sd=inc_sd, V_real=float(np.abs(np.diff(theta)).sum()), embed_neg=neg)
    Re, ge = F.best_ewma(r2, hstar); Rg, pg = F.best_garch(r2, hstar)
    he = F.ewma_forecasts(r2, ge, hstar[0]); hg = F.garch_forecasts(r2, pg[2], pg[0], pg[1], hstar[0])
    out['ewma_gamma'] = ge
    score(out, 'fixed_best', he if Re <= Rg else hg, r2, hstar)
    y = np.log(np.maximum(r2, 1e-300)) - LOGCHI2_MEAN
    score(out, 'linear_logspace', np.exp(linear_gp_filter(y, cov_fn, LOGCHI2_VAR, W)), r2, hstar)
    score(out, 'pf_knows_model', pf_ar_reference(r2, cov_fn, rng), r2, hstar)
    h_flh = flh_ftl(r2, hstar, 'ftl'); score(out, 'flh_ftl', h_flh, r2, hstar)
    h_fse = F.fixed_share_bank(r2, hstar, bank='ewma'); score(out, 'fs_ewma', h_fse, r2, hstar)
    score(out, 'fs_cusum', F.fs_cusum_bank(r2, hstar), r2, hstar)
    score(out, 'meta2', fixed_share_over_forecasts([h_flh, h_fse], r2), r2, hstar)
    return out

# ----------------------------------------------------------------------------- fitting helpers
def fit_power(xs, ys):
    """log y = a + b log x; returns (b, se_b, R2)."""
    X = np.column_stack([np.ones(len(xs)), np.log(xs)]); yv = np.log(ys)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ coef; dof = max(1, len(yv) - 2)
    se = math.sqrt(float(res @ res) / dof * np.linalg.inv(X.T @ X)[1, 1])
    r2 = 1 - float(res @ res) / float(((yv - yv.mean()) ** 2).sum()) if len(yv) > 1 else float('nan')
    return coef[1], se, r2

def fit_loglinear(xs, ys):
    """y = c1 + c2 log x; returns (c1, c2, R2)."""
    X = np.column_stack([np.ones(len(xs)), np.log(xs)]); yv = np.asarray(ys, float)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ coef
    r2 = 1 - float(res @ res) / float(((yv - yv.mean()) ** 2).sum()) if len(yv) > 1 else float('nan')
    return coef[0], coef[1], r2

def fit_2d(rows, key, xkey='T', vkey='V_real'):
    pts = [(r[xkey], r[vkey], r[key]) for r in rows if r.get(key) is not None and r[key] == r[key] and r[key] > 0]
    if len(pts) < 4:
        return None
    X = np.array([[1.0, math.log(a), math.log(b)] for a, b, c in pts]); yv = np.array([math.log(c) for a, b, c in pts])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ coef; dof = max(1, len(yv) - 3)
    cov = float(res @ res) / dof * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
    r2 = 1 - float(res @ res) / float(((yv - yv.mean()) ** 2).sum())
    return dict(bT=coef[1], seT=se[1], bV=coef[2], seV=se[2], r2=r2, n=len(yv))

def fmt2(f):
    return '(n/a)' if f is None else f"bT={f['bT']:+.2f}±{f['seT']:.2f} bV={f['bV']:+.2f}±{f['seV']:.2f} R²={f['r2']:.3f} n={f['n']}"

def cellmean(rows, key, cond):
    v = [r[key] for r in rows if cond(r) and r[key] == r[key]]
    return (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float('nan'), len(v)) if v else (float('nan'), float('nan'), 0)

# ----------------------------------------------------------------------------- reports
def xrows(rows, keys):
    """Return copies of rows where each key is replaced by its expected-regret version (key+'_x')."""
    out = []
    for r in rows:
        q = dict(r)
        for k in keys:
            if k + '_x' in r:
                q[k] = r[k + '_x']
        out.append(q)
    return out

def consistency_line(rows, keys):
    rat = [r[k] / r[k + '_x'] for r in rows for k in keys if k + '_x' in r and r[k + '_x'] == r[k + '_x'] and r[k + '_x'] > 0]
    return f'realized/expected regret ratio over all filters and cells: mean {np.mean(rat):.2f}, sd {np.std(rat):.2f}, n={len(rat)}'

def report_flh_bayes(rows_raw, Ts_s, Ts_d, Vs):
    KEYS = ['ewma_best', 'garch_best', 'fixed_best', 'fs_ewma', 'fs_cusum', 'meta2', 'flh_ftl', 'bayes']
    rows = xrows(rows_raw, KEYS)
    L = []; P = L.append
    P('=' * 110)
    P('PHASE 1, STEPS 1-3: the real construction (FLH-FTL) against the exact Bayes floor, lattice jump prior')
    P('regret = QLIKE dynamic regret vs the true path; bayes = exact Bayes-optimal causal filter for the DGP (a floor for ALL causal filters);')
    P('flh_ftl = Follow-the-Leading-History over follow-the-leader experts (the Theorem I.2 construction); fixed_best = hindsight-best EWMA/GARCH;')
    P('fs_ewma / fs_cusum = first-sandbox stand-ins (skipped at the largest T); meta2 = Fixed-Share over {flh_ftl, fs_ewma} ("gains x restarts").')
    P('METRIC: expected regret given the forecasts, sum_t d(theta_hat_t||theta*_t) -- the exact conditional expectation of realized QLIKE regret')
    P('(true path, no oracle; innovation noise removed from the scoring only).  ' + consistency_line(rows_raw, KEYS))
    P('=' * 110)
    for regime, Ts in (('sparse', Ts_s), ('dense', Ts_d)):
        rr = [r for r in rows if r['regime'] == regime]
        P(''); P(f'--- regime {regime}')
        keys = ['fixed_best', 'fs_ewma', 'fs_cusum', 'meta2', 'flh_ftl', 'bayes']
        P('mean expected regret (± s.e.) by cell:')
        P('filter'.ljust(12) + ''.join(f'{f"T={T//1000}k,V={V:g}":>17}' for T in Ts for V in Vs))
        for k in keys:
            P(k.ljust(12) + ''.join(('{:>12.1f}±{:<4.0f}'.format(*cellmean(rr, k, lambda r, T=T, V=V: r['T'] == T and r['V'] == V)[:2]) if cellmean(rr, k, lambda r, T=T, V=V: r['T'] == T and r['V'] == V)[2] else ' ' * 13 + 'skip') for T in Ts for V in Vs))
        P('realized K (mean): ' + ', '.join(f'T={T//1000}k,V={V:g}: {cellmean(rr, "K_real", lambda r, T=T, V=V: r["T"] == T and r["V"] == V)[0]:.1f}' for T in Ts for V in Vs))
        P('')
        P('2-d fits  log regret = a + bT log T + bV log V_real   (per-path realized V):')
        for k in keys:
            P(f'  {k:<12}{fmt2(fit_2d(rr, k))}')
        P('  ratios to the Bayes floor (mean over cells): ' + ', '.join(f'{k}: {np.nanmean([r[k] / r["bayes"] for r in rr if r[k] == r[k]]):.2f}' for k in keys[:-1]))
        if regime == 'sparse':
            P('')
            P('PER-JUMP COST vs T (regret / K_real, mean over seeds and V):  is it c1 + c2 log T (theorem) or a power law?')
            P('T'.rjust(8) + ''.join(f'{k:>14}' for k in keys))
            pj = {k: [] for k in keys}
            for T in Ts:
                vals = []
                for k in keys:
                    v = [r[k] / max(r['K_real'], 1) for r in rr if r['T'] == T and r[k] == r[k]]
                    m = float(np.mean(v)) if v else float('nan'); pj[k].append(m); vals.append(m)
                P(f'{T:>8}' + ''.join(f'{v:>14.2f}' for v in vals))
            for k in keys:
                xs = np.array([T for T, m in zip(Ts, pj[k]) if m == m and m > 0]); ys = np.array([m for m in pj[k] if m == m and m > 0])
                if len(xs) >= 3:
                    b, se, r2p = fit_power(xs, ys); c1, c2, r2l = fit_loglinear(xs, ys)
                    P(f'  {k:<12} power law: per-jump ~ T^{b:+.2f}±{se:.2f} (R²={r2p:.3f}) | log-linear: {c1:.1f} + {c2:.2f} log T (R²={r2l:.3f})')
            P('  reading: bayes and flh_ftl per-jump costs flat-or-logarithmic in T while fixed_best grows like T^0.5 is the theorem; a power law')
            P('  with exponent >= 0.25 persisting to T=256k for flh_ftl would mean the construction does not deliver log T at reachable sizes.')
    P('')
    P('GATE (from the plan): commit to Idea I if (i) flh_ftl tracks the Bayes floor up to a slowly varying factor on both regimes and')
    P('(ii) the Bayes floor itself scales like K log T (sparse) and T^(1/3) V^(2/3) (dense).  If the Bayes floor beats T^(1/3)V^(2/3) by a')
    P('polynomial factor on the dense prior, the minimax lower bound needs a different hard prior than the natural one.')
    return '\n'.join(L)

def report_heavy(rows_raw, Ts, innovs):
    KEYS = ['fixed_best', 'bayes_knows_nu', 'flh_raw', 'flh_clip9', 'flh_clip30', 'flh_clip100', 'flh_dcs_nu5', 'flh_dcs_true_nu', 'fs_ewma', 'meta2', 'fs_cusum_raw', 'fs_cusum_clip9']
    rows = xrows(rows_raw, KEYS)
    L = []; P = L.append
    P('=' * 110); P('PHASE 1, STEP 4: heavy tails -- FLH-FTL raw vs relative clipping vs bounded (DCS) scores; Bayes filter knows nu')
    P('metric = EXPECTED regret given the forecasts (see flh-bayes report); clipping at c means r^2 -> min(r^2, c*h_expert) inside the expert only')
    P(consistency_line(rows_raw, KEYS))
    P('=' * 110)
    keys = KEYS
    for regime in ('sparse', 'dense'):
        rr = [r for r in rows if r['regime'] == regime]
        P(''); P(f'--- regime {regime}: mean regret (± s.e.), rows = filter, cols = (T, innovations)')
        P('filter'.ljust(18) + ''.join(f'{f"T={T//1000}k,{inn}":>16}' for T in Ts for inn in innovs))
        for k in keys:
            P(k.ljust(18) + ''.join('{:>11.0f}±{:<4.0f}'.format(*cellmean(rr, k, lambda r, T=T, inn=inn: r['T'] == T and r['innov'] == inn)[:2]) for T in Ts for inn in innovs))
        P('ratio to Bayes floor (mean over seeds):')
        P('filter'.ljust(18) + ''.join(f'{f"T={T//1000}k,{inn}":>16}' for T in Ts for inn in innovs))
        for k in keys[2:]:
            P(k.ljust(18) + ''.join(f'{np.mean([r[k] / r["bayes_knows_nu"] for r in rr if r["T"] == T and r["innov"] == inn]):>16.2f}' for T in Ts for inn in innovs))
    P('')
    P('reading: (a) a fixed relative clip trades variance for a bias that grows linearly in T -- the best c should move UP with T and with nu;')
    P('(b) the DCS score with the true nu is the bias-free bounded update (the theorem-I.4 object); with a wrong working nu it is biased;')
    P('(c) raw FTL is unbiased but its per-jump cost inflates with the tail; the Bayes-floor ratio is the number to watch.')
    return '\n'.join(L)

def report_stoch(rows_ar_raw, rows_mt_raw, sigs, nus, ells=None):
    KA = ['fixed_best', 'fs_ewma', 'fs_cusum', 'meta2', 'flh_ftl', 'bayes_grid']
    KM = ['fixed_best', 'fs_ewma', 'fs_cusum', 'meta2', 'flh_ftl', 'pf_knows_model', 'linear_logspace']
    rows_ar = xrows(rows_ar_raw, KA); rows_mt = xrows(rows_mt_raw, KM)
    L = []; P = L.append
    if not rows_ar:
        P('(AR(1) part skipped: --matern-only)')
    else:
      P('=' * 110); P('PHASE 1, STEP 5(a): AR(1) log-variance (SV model), phi = 0.99 -- grid-HMM Bayes floor')
      P('theory: on random-walk-like paths a constant gain is near-optimal (Kalman); per-step error ~ increment scale^1 (H = 1/2 => 2/(2H+1) = 1)')
      P('metric = EXPECTED regret given the forecasts.  ' + consistency_line(rows_ar_raw, KA))
      P('=' * 110)
      keys = KA
      P('sig_w'.rjust(8) + ''.join(f'{k:>14}' for k in keys) + f'{"ewma_gamma":>12}{"V_real":>10}')
      means = {k: [] for k in keys}
      for sg in sigs:
          rr = [r for r in rows_ar if abs(r['sig_w'] - sg) < 1e-12]
          vals = [float(np.mean([r[k] for r in rr])) for k in keys]
          for k, v in zip(keys, vals):
              means[k].append(v)
          P(f'{sg:>8.3f}' + ''.join(f'{v:>14.1f}' for v in vals) + f'{np.mean([r["ewma_gamma"] for r in rr]):>12.4f}{np.mean([r["V_real"] for r in rr]):>10.0f}')
      for k in keys:
          b, se, r2 = fit_power(np.array(sigs), np.array(means[k]))
          P(f'  {k:<12} regret ~ sig_w^{b:+.2f}±{se:.2f} (R²={r2:.3f})   [prediction for the Bayes floor: exponent ~ +1.0]')
      P('  ratios to the Bayes floor (mean): ' + ', '.join(f'{k}: {np.mean([r[k] / r["bayes_grid"] for r in rows_ar]):.2f}' for k in keys[:-1]))
      P('')
    P('=' * 110); P('PHASE 1, STEP 5(b): stationary Matern(nu) log-variance -- references: PF that knows the model (near-Bayes), linear log-space filter')
    P('H = nu for nu < 1 (rough for nu < 1/2).  Theory (Thm I.5): per-step error ~ inc_sd^(2/(2H+1)) => slope of regret vs increment scale should be')
    P('2/(2H+1): nu=0.15 -> 1.54, nu=0.5 -> 1.00, nu=1.5 -> ~0.67.  Cells vary (ell, amp); inc_sd = sd of one-step increment of theta.')
    P('The log-space linear filter is information-inefficient (Fisher info 0.20 vs 0.50 for r^2-based filters) and is kept only as a sanity line.')
    P('=' * 110)
    keys = KM
    P(consistency_line(rows_mt_raw, KM))
    for nu in nus:
        P(f'--- nu = {nu:g}  (predicted slope 2/(2H+1) = {2.0/(2.0*min(nu,1.0)+1.0):.2f})')
        P('ell'.rjust(6) + 'amp'.rjust(6) + f'{"inc_sd":>8}' + ''.join(f'{k:>17}' for k in keys) + f'{"ewma_gamma":>11}')
        means = {k: [] for k in keys}; incs = []
        cells = sorted(set((r['ell'], r['amp']) for r in rows_mt if abs(r['nu'] - nu) < 1e-9))
        for ell, amp in cells:
            rr = [r for r in rows_mt if abs(r['nu'] - nu) < 1e-9 and abs(r['ell'] - ell) < 1e-9 and abs(r['amp'] - amp) < 1e-9]
            inc = float(np.mean([r['inc_sd'] for r in rr])); incs.append(inc)
            vals = [float(np.mean([r[k] for r in rr])) for k in keys]
            for k, v in zip(keys, vals):
                means[k].append(v)
            P(f'{ell:>6.0f}{amp:>6.2f}{inc:>8.3f}' + ''.join(f'{v:>17.1f}' for v in vals) + f'{np.mean([r["ewma_gamma"] for r in rr]):>11.4f}')
        for k in keys:
            if len(incs) >= 3 and all(m > 0 for m in means[k]):
                b, se, r2 = fit_power(np.array(incs), np.array(means[k]))
                P(f'  {k:<17} regret ~ inc_sd^{b:+.2f}±{se:.2f} (R²={r2:.3f})')
        P('  ratio to PF (mean over cells): ' + ', '.join(f'{k}: {np.mean([m/p for m, p in zip(means[k], means["pf_knows_model"])]):.2f}' for k in keys if k != 'pf_knows_model'))
    P('')
    P('reading: the exponent test is on slopes (the PF is a near-Bayes reference, not an exact floor); a no-hindsight filter whose slope and')
    P('PF-ratio stay stable across nu is "adaptive to roughness"; the hindsight-best EWMA is the constant-gain benchmark.')
    return '\n'.join(L)

# ----------------------------------------------------------------------------- main
def write_csv(path, rows):
    keys = sorted(set(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)

def run_pool(fn, jobs, procs, label):
    t0 = time.time(); rows = []
    print(f'[{label}] {len(jobs)} cells on {procs} process(es)', flush=True)
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f'  {i+1}/{len(jobs)} done, {time.time()-t0:.0f}s', flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['flh-bayes', 'heavy', 'heavy2', 'stoch'])
    ap.add_argument('--matern-only', action='store_true')
    ap.add_argument('--scale', choices=['small', 'full'], default='small')
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default='out_phase1')
    ap.add_argument('--seeds', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    full = a.scale == 'full'
    if a.cmd == 'flh-bayes':
        Vs = [8.0, 32.0]
        Ts_s = [2 ** k for k in range(11, 19)] if full else [2048, 8192, 32768]
        Ts_d = [2 ** k for k in range(11, 16)] if full else [2048, 8192, 32768]
        seeds = a.seeds or (5 if full else 3)
        jobs = [('sparse', T, V, s, T <= 65536) for T in Ts_s for V in Vs for s in range(seeds)]
        jobs += [('dense', T, V, s, True) for T in Ts_d for V in Vs for s in range(seeds)]
        jobs.sort(key=lambda j: -j[1])
        rows = run_pool(cell_flh_bayes, jobs, a.procs, 'flh-bayes')
        write_csv(os.path.join(a.out, 'flh_bayes_results.csv'), rows)
        rep = report_flh_bayes(rows, Ts_s, Ts_d, Vs)
    elif a.cmd == 'heavy2':
        Ts = [8192, 32768, 131072] if full else [8192]
        innovs = ['3', '4', '6', 'gaussian']
        seeds = a.seeds or (5 if full else 2)
        jobs = [(reg, T, 16.0, s, inn) for reg in ('sparse', 'dense') for T in Ts for inn in innovs for s in range(seeds)]
        jobs.sort(key=lambda j: -j[1])
        rows = run_pool(cell_heavy2, jobs, a.procs, 'heavy2')
        write_csv(os.path.join(a.out, 'heavy2_results.csv'), rows)
        rep = report_heavy2(rows, Ts, innovs)
    elif a.cmd == 'heavy':
        Ts = [8192, 32768] if full else [8192]
        innovs = ['3', '4', '6', 'gaussian']
        seeds = a.seeds or (5 if full else 2)
        jobs = [(reg, T, 16.0, s, inn) for reg in ('sparse', 'dense') for T in Ts for inn in innovs for s in range(seeds)]
        jobs.sort(key=lambda j: -j[1])
        rows = run_pool(cell_heavy, jobs, a.procs, 'heavy')
        write_csv(os.path.join(a.out, 'heavy_results.csv'), rows)
        rep = report_heavy(rows, Ts, innovs)
    else:
        T = 16384 if full else 8192
        seeds = a.seeds or (4 if full else 2)
        sigs = [0.02, 0.05, 0.1, 0.2]
        jobs = [(T, 0.99, sg, s) for sg in sigs for s in range(seeds)]
        rows_ar = run_pool(cell_stoch_ar1, jobs, a.procs, 'stoch-ar1') if not a.matern_only else []
        nus = [0.15, 0.5, 1.5]; ells = [30.0, 300.0]; W = 400 if full else 256
        amps_of = lambda nu: [0.1, 0.2, 0.4] if nu < 0.5 else [0.5, 1.0, 2.0]     # rough paths need SMALL increments to be in the scaling regime
        jobs = [(T, nu, ell, amp, s, W) for nu in nus for ell in ells for amp in amps_of(nu) for s in range(seeds)]
        rows_mt = run_pool(cell_stoch_matern, jobs, a.procs, 'stoch-matern')
        if rows_ar:
            write_csv(os.path.join(a.out, 'stoch_ar1_results.csv'), rows_ar)
        write_csv(os.path.join(a.out, 'stoch_matern_results.csv'), rows_mt)
        rep = report_stoch(rows_ar, rows_mt, sigs, nus, ells)
    with open(os.path.join(a.out, f'{a.cmd}_report.txt'), 'w') as f:
        f.write(rep)
    print(rep)

if __name__ == '__main__':
    main()
