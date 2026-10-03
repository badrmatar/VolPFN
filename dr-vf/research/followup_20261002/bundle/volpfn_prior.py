#!/usr/bin/env python3
"""
volpfn_prior.py -- the simulated universe ("prior") that VolPFN is trained on, and the exact Bayes filters of the
families that have one.

THE UNIVERSE.  A return series is drawn hierarchically:
  1. an innovation law: standardised (zero mean, unit variance) Fernandez-Steel skew-t with tail index nu
     (Gaussian with probability 1/4, otherwise nu log-uniform on [2.5, 50]) and skew gamma (1 with probability 3/4,
     otherwise log-uniform on [0.7, 1.4]);
  2. a variance dynamics family and its parameters (weights in PRIOR_WEIGHTS):
       garch    GARCH(1,1): 1-persistence log-uniform on [0.001, 0.15], ARCH share of the persistence U(0.02, 0.3)
       gjr      GJR-GARCH(1,1,1): asymmetric term, persistence alpha + kappa gamma + beta as above (kappa = E[e^2; e<0])
       egarch   EGARCH(1,1): log h_{t+1} = beta log h_t + a(|z_t| - E|z|) + g z_t
       sv       log-AR(1) stochastic volatility with (lagged) leverage: theta_{t+1} = phi theta_t + s_w (rho z_t + sqrt(1-rho^2) xi_{t+1})
       rough    stationary Gaussian log-variance with Matern covariance, smoothness H in {0.05, 0.1, 0.2, 0.35, 0.5}
                (H = 0.5 is the OU / log-AR(1) case, H < 0.5 rough), length scale in {30, 120, 500} steps
       hmm      Markov-switching log-variance with 2 or 3 levels and mean regime durations log-uniform on [20, 1500]
       lattice  the lattice-jump path of the theory notes: log-variance jumps by +-delta at rate K/T (reflected at +-3.5)
     the GARCH-type recursions start at a random multiple of the unconditional variance (or after a burn-in);
  3. contaminations, each switched on independently: return jumps (compound Poisson, intensity log-uniform on
     [0.002, 0.05], size 3-10 sigma), an AR(1) in returns with coefficient U(-0.15, 0.15) (bid-ask bounce or momentum),
     a drift N(0, 0.03^2) in units of sigma, and stale prices (a two-state Markov chain with unconditional zero fraction
     log-uniform on [0.005, 0.2] and mean run length U(1, 4); a stale day reports an exact zero and the next fresh day
     reports the accumulated return);
  4. the series is standardised by the root mean square of the non-zero returns among its first 100 observations --
     the same statistic VolPFN applies to real data (and that univol uses as its initial scale), so the network never
     sees an absolute scale.
Every series comes with the log of the conditional variance of the observed return given the latent state (the target
of an auxiliary loss) and a loss mask that is zero where the observed return is an exact (stale) zero.

THE FLOORS.  For garch/gjr/egarch (exact: the conditional variance is a deterministic function of the past given the
parameters), sv (a grid filter, exact up to discretisation, leverage included), hmm and lattice (forward algorithm,
exact), oracle_predictive() returns the known-parameter Bayes predictive: its variance forecast (posterior mean of the
variance, the Bayes act for QLIKE) and the log predictive density of the realised return.  VolPFN's excess log loss
over these is its measured distance from the exact filter -- a certificate no real-data foundation model can give.

usage (self-test):  python3 volpfn_prior.py            checks the law, the moments, the oracles and the timing
"""
import math, sys, time
import numpy as np
from scipy.special import betainc, gammainc, gammaln, ndtr

PRIOR_WEIGHTS = dict(garch=0.20, gjr=0.15, egarch=0.10, sv=0.20, rough=0.10, hmm=0.12, lattice=0.13)
OVERLAY_P = dict(jumps=0.30, ar=0.30, drift=0.50, stale=0.25)
NU_GAUSS = 1e6           # a t with this many degrees of freedom is treated as the Gaussian
FAMILIES = list(PRIOR_WEIGHTS)

# ----------------------------------------------------------------------------- the innovation law: standardised skew-t
def t_logpdf(x, nu):
    """log density of the standard Student-t(nu) (Gaussian for nu >= NU_GAUSS)."""
    x = np.asarray(x, dtype=float); nu = np.asarray(nu, dtype=float)
    return np.where(nu >= NU_GAUSS, -0.5 * x * x - 0.5 * math.log(2 * math.pi),
                    gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - 0.5 * (nu + 1) * np.log1p(x * x / nu))

def t_absmean(nu):
    """E|X| for the unit-variance Student-t(nu) (sqrt(2/pi) for the Gaussian)."""
    nu = np.asarray(nu, dtype=float)
    return np.where(nu >= NU_GAUSS, math.sqrt(2 / math.pi), 2 * np.sqrt(np.maximum(nu - 2, 1e-12)) * np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / (math.sqrt(math.pi) * (nu - 1)))

def skew_moments(nu, gamma):
    """mean m and variance v of the Fernandez-Steel skewed unit-variance t: Z = gamma |X| w.p. gamma^2/(1+gamma^2), else -|X|/gamma."""
    M1 = t_absmean(nu); g = np.asarray(gamma, dtype=float)
    m = M1 * (g - 1 / g); v = (g ** 3 + g ** -3) / (g + 1 / g) - m * m
    return m, v

def skew_negative_second_moment(nu, gamma):
    """E[e^2 1(e < 0)] for the standardized Fernandez-Steel innovation.

    Truncated absolute Student-t moments are incomplete-beta expressions;
    their Gaussian limits use incomplete-gamma expressions. Centering matters:
    the negative threshold in the uncentered skewed variable is its mean, not
    zero. Reflection supplies gamma < 1. No simulation or clipping is used.
    """
    nu, gamma = np.broadcast_arrays(np.asarray(nu, dtype=float), np.asarray(gamma, dtype=float))
    if not np.isfinite(nu).all() or not np.isfinite(gamma).all() or np.any(nu <= 2) or np.any(gamma <= 0):
        raise ValueError('finite nu > 2 and gamma > 0 required')
    g = np.where(gamma < 1, 1 / gamma, gamma)
    mean, variance = skew_moments(nu, g)
    absolute_mean = t_absmean(nu)
    threshold = mean / g
    gaussian = nu >= NU_GAUSS
    # Benign placeholder df avoids evaluating a huge-df beta expression on
    # Gaussian rows; those rows use the exact normal expressions below.
    df = np.where(gaussian, 3.0, nu)
    ratio = threshold * threshold / (df - 2)
    z = ratio / (1 + ratio)
    lower0 = betainc(0.5, df / 2, z)
    lower1 = absolute_mean * -np.expm1(-0.5 * (df - 1) * np.log1p(ratio))
    lower2 = betainc(1.5, (df - 2) / 2, z)
    normal_argument = threshold * threshold / 2
    lower0 = np.where(gaussian, gammainc(0.5, normal_argument), lower0)
    lower1 = np.where(gaussian, absolute_mean * -np.expm1(-normal_argument), lower1)
    lower2 = np.where(gaussian, gammainc(1.5, normal_argument), lower2)
    p = g * g / (1 + g * g)
    negative_branch = (1 / (g * g) + 2 * mean * absolute_mean / g + mean * mean) / (1 + g * g)
    positive_branch = p * (g * g * lower2 - 2 * g * mean * lower1 + mean * mean * lower0)
    right_skew = (negative_branch + positive_branch) / variance
    result = np.where(gamma < 1, 1 - right_skew, right_skew)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise FloatingPointError('invalid analytic negative second moment')
    return result

def law_logpdf(e, nu, gamma):
    """log density at e of the standardised (zero-mean, unit-variance) skew-t(nu, gamma) innovation. Broadcasts."""
    e = np.asarray(e, dtype=float); nu = np.asarray(nu, dtype=float); gamma = np.asarray(gamma, dtype=float)
    m, v = skew_moments(nu, gamma); sv = np.sqrt(v)
    z = m + sv * e                                             # the un-standardised skewed variable
    c = np.sqrt(np.where(nu >= NU_GAUSS, 1.0, nu / np.maximum(nu - 2, 1e-12)))   # X = T/c has unit variance
    x = np.where(z >= 0, z / gamma, z * gamma)                 # argument of the symmetric base density
    return np.log(2 / (gamma + 1 / gamma)) + np.log(c) + t_logpdf(x * c, nu) + np.log(sv)

def law_sample(rng, nu, gamma, size):
    """standardised skew-t innovations; nu, gamma broadcast against size (per-row values allowed)."""
    nu = np.broadcast_to(np.asarray(nu, dtype=float), size); gamma = np.broadcast_to(np.asarray(gamma, dtype=float), size)
    gauss = nu >= NU_GAUSS
    x = np.where(gauss, rng.standard_normal(size), rng.standard_t(np.where(gauss, 3.0, nu), size) * np.sqrt(np.maximum(nu - 2, 1e-12) / nu))
    x = np.abs(x); pos = rng.random(size) < gamma ** 2 / (1 + gamma ** 2)
    z = np.where(pos, gamma * x, -x / gamma)
    m, v = skew_moments(nu, gamma)
    return (z - m) / np.sqrt(v)

def draw_law(rng, n):
    nu = np.where(rng.random(n) < 0.25, NU_GAUSS, np.exp(rng.uniform(math.log(2.5), math.log(50.0), n)))
    gamma = np.where(rng.random(n) < 0.75, 1.0, np.exp(rng.uniform(math.log(0.7), math.log(1.4), n)))
    return nu, gamma

# ----------------------------------------------------------------------------- variance dynamics (vectorised over series)
def sim_garch_family(rng, kind, n, T, nu, gamma):
    """returns r (n,T), logv (n,T) = log conditional variance, params list. Unit unconditional variance."""
    burn = np.where(rng.random(n) < 0.5, 300, 0)
    Tb = T + int(burn.max())
    eps = law_sample(rng, nu[:, None], gamma[:, None], (n, Tb))
    P = 1 - np.exp(rng.uniform(math.log(0.001), math.log(0.15), n))
    params = []
    r = np.empty((n, Tb)); lv = np.empty((n, Tb))
    if kind in ('garch', 'gjr'):
        if kind == 'garch':
            alpha = P * rng.uniform(0.02, 0.3, n); gam = np.zeros(n)
        else:
            alpha = P * rng.uniform(0.0, 0.2, n); gam = P * rng.uniform(0.02, 0.35, n)
        # Preserve the historical RNG stream for unrelated parameters/families.
        # A finite heavy-tailed Monte Carlo sample can give kappa > 1 and hence
        # an inadmissible negative beta. GJR now uses its exact innovation moment.
        moment_draws = law_sample(rng, nu[:, None], gamma[:, None], (n, 20000))
        kappa = (skew_negative_second_moment(nu, gamma) if kind == 'gjr' else
                 np.array([np.mean(np.square(e) * (e < 0)) for e in moment_draws]))
        beta = P - alpha - gam * kappa; omega = 1 - P
        h = np.exp(rng.normal(0, 0.6, n))                       # start off-level; the burn-in (if any) removes it
        for t in range(Tb):
            lv[:, t] = np.log(h); e = eps[:, t]; r[:, t] = np.sqrt(h) * e
            h = omega + (alpha + gam * (e < 0)) * h * e * e + beta * h
        for i in range(n):
            params.append(dict(omega=float(omega[i]), alpha=float(alpha[i]), gamma=float(gam[i]), beta=float(beta[i])))
    else:                                                       # egarch
        beta = 1 - np.exp(rng.uniform(math.log(0.003), math.log(0.15), n)); a = rng.uniform(0.05, 0.3, n); g = rng.uniform(-0.25, 0.05, n)
        m1 = np.array([np.mean(np.abs(e)) for e in law_sample(rng, nu[:, None], gamma[:, None], (n, 20000))])   # E|z| of the (skewed) law
        th = rng.normal(0, 0.6, n)
        for t in range(Tb):
            lv[:, t] = th; e = eps[:, t]; r[:, t] = np.exp(th / 2) * e
            th = beta * th + a * (np.abs(e) - m1) + g * e
        for i in range(n):
            params.append(dict(beta=float(beta[i]), a=float(a[i]), g=float(g[i]), m1=float(m1[i])))
    out_r = np.empty((n, T)); out_lv = np.empty((n, T))
    for i in range(n):
        b = int(burn[i]); out_r[i] = r[i, b:b + T]; out_lv[i] = lv[i, b:b + T]
        params[i]['h0'] = float(np.exp(lv[i, b])); params[i]['burn'] = b
    return out_r, out_lv, params

def sim_sv(rng, n, T, nu, gamma):
    phi = 1 - np.exp(rng.uniform(math.log(0.002), math.log(0.1), n))
    sd = rng.uniform(0.3, 1.5, n); sw = sd * np.sqrt(1 - phi * phi)
    rho = np.where(rng.random(n) < 0.6, rng.uniform(-0.8, 0.2, n), 0.0)
    eps = law_sample(rng, nu[:, None], gamma[:, None], (n, T)); xi = rng.standard_normal((n, T))
    th = rng.normal(0, 1, n) * sd                               # stationary start
    r = np.empty((n, T)); lv = np.empty((n, T))
    for t in range(T):
        lv[:, t] = th; e = eps[:, t]; r[:, t] = np.exp(th / 2) * e
        th = phi * th + sw * (rho * e + np.sqrt(1 - rho * rho) * xi[:, t])
    params = [dict(phi=float(phi[i]), sw=float(sw[i]), rho=float(rho[i]), sd=float(sd[i])) for i in range(n)]
    return r, lv, params

def matern_cov(tau, H, ell):
    tau = np.abs(np.asarray(tau, dtype=float))
    if H == 0.5:
        return np.exp(-tau / ell)
    from scipy.special import kv, gamma as G
    a = math.sqrt(2.0 * H) * tau / ell; out = np.ones_like(a); az = a[a > 0]
    out[a > 0] = (2 ** (1 - H) / G(H)) * (az ** H) * kv(H, az)
    return out

_EMB = {}
def _embedding(T, H, ell):
    """circulant embedding (Davies-Harte) of the Matern covariance; negative eigenvalues (if any) are zeroed."""
    key = (T, H, ell)
    if key not in _EMB:
        M = 1
        while M < 2 * T:
            M *= 2
        row = matern_cov(np.concatenate([np.arange(M // 2 + 1), np.arange(M // 2 - 1, 0, -1)]), H, ell)
        eig = np.fft.fft(row).real; neg = -eig.min() if eig.min() < 0 else 0.0
        eig = np.maximum(eig, 0.0)
        _EMB[key] = (M, np.sqrt(eig / M), float(neg))
    return _EMB[key]

def sim_rough(rng, n, T, nu, gamma):
    Hs = np.array([0.05, 0.1, 0.2, 0.35, 0.5]); ells = np.array([30.0, 120.0, 500.0])
    H = Hs[rng.integers(len(Hs), size=n)]; ell = ells[rng.integers(len(ells), size=n)]; sd = rng.uniform(0.3, 1.5, n)
    lv = np.empty((n, T))
    for i in range(n):
        M, s, _ = _embedding(T, float(H[i]), float(ell[i]))
        z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        lv[i] = sd[i] * np.fft.fft(s * z).real[:T]
    eps = law_sample(rng, nu[:, None], gamma[:, None], (n, T)); r = np.exp(lv / 2) * eps
    params = [dict(H=float(H[i]), ell=float(ell[i]), sd=float(sd[i])) for i in range(n)]
    return r, lv, params

def sim_hmm(rng, n, T, nu, gamma):
    r = np.empty((n, T)); lv = np.empty((n, T)); params = []
    eps = law_sample(rng, nu[:, None], gamma[:, None], (n, T))
    for i in range(n):
        K = int(rng.integers(2, 4)); s = rng.uniform(0.4, 1.2)
        levels = np.sort(rng.normal(0, s, K)); levels -= levels.mean()
        D = np.exp(rng.uniform(math.log(20), math.log(1500), K))
        P = np.zeros((K, K))
        for k in range(K):
            P[k] = (1 / D[k]) / (K - 1); P[k, k] = 1 - 1 / D[k]
        st = int(rng.integers(K)); u = rng.random(T); cum = np.cumsum(P, axis=1)
        states = np.empty(T, dtype=int)
        for t in range(T):
            states[t] = st; st = min(int(np.searchsorted(cum[st], u[t])), K - 1)
        lv[i] = levels[states]; r[i] = np.exp(lv[i] / 2) * eps[i]
        params.append(dict(levels=levels.tolist(), P=P.tolist(), K=K))
    return r, lv, params

def sim_lattice(rng, n, T, nu, gamma):
    K = rng.uniform(2, 25, n); delta = rng.uniform(0.3, 1.5, n); lam = K / T
    jumps = rng.random((n, T)) < lam[:, None]; jumps[:, 0] = False
    signs = np.where(rng.random((n, T)) < 0.5, -1.0, 1.0)
    lv = np.zeros((n, T)); th = np.zeros(n)
    for t in range(1, T):                                       # random walk on the lattice, reflected at +-3.5
        step = np.where(jumps[:, t], signs[:, t] * delta, 0.0)
        step = np.where(np.abs(th + step) > 3.5, -step, step); th = th + step; lv[:, t] = th
    eps = law_sample(rng, nu[:, None], gamma[:, None], (n, T)); r = np.exp(lv / 2) * eps
    params = [dict(lam=float(lam[i]), delta=float(delta[i])) for i in range(n)]
    return r, lv, params

GARCH_FIXED = dict(omega=0.03, alpha=0.05, beta=0.92)     # persistence 0.97, unit unconditional variance

def sim_garch_fixed(rng, n, T, nu, gamma):
    """Ablation control (7 Sep 2026 amendment): GARCH(1,1) with the parameters of GARCH_FIXED and Gaussian
    innovations for every series; only the starting level is random. nu and gamma are ignored on purpose.
    Not in PRIOR_WEIGHTS or FAMILIES, so the default prior and every existing checkpoint are unaffected."""
    burn = np.where(rng.random(n) < 0.5, 300, 0); Tb = T + int(burn.max())
    eps = rng.standard_normal((n, Tb)); o, a, b = GARCH_FIXED['omega'], GARCH_FIXED['alpha'], GARCH_FIXED['beta']
    r = np.empty((n, Tb)); lv = np.empty((n, Tb)); h = np.exp(rng.normal(0, 0.6, n))
    for t in range(Tb):
        lv[:, t] = np.log(h); e = eps[:, t]; r[:, t] = np.sqrt(h) * e
        h = o + a * h * e * e + b * h
    out_r = np.empty((n, T)); out_lv = np.empty((n, T)); params = []
    for i in range(n):
        bi = int(burn[i]); out_r[i] = r[i, bi:bi + T]; out_lv[i] = lv[i, bi:bi + T]
        params.append(dict(omega=o, alpha=a, gamma=0.0, beta=b, h0=float(np.exp(lv[i, bi])), burn=bi, nu_used=NU_GAUSS, gamma_used=1.0))
    return out_r, out_lv, params

SIMULATORS = dict(garch=lambda rng, n, T, nu, g: sim_garch_family(rng, 'garch', n, T, nu, g),
                  garch_fixed=sim_garch_fixed,
                  gjr=lambda rng, n, T, nu, g: sim_garch_family(rng, 'gjr', n, T, nu, g),
                  egarch=lambda rng, n, T, nu, g: sim_garch_family(rng, 'egarch', n, T, nu, g),
                  sv=sim_sv, rough=sim_rough, hmm=sim_hmm, lattice=sim_lattice)

# ----------------------------------------------------------------------------- contaminations and standardisation
def _aggregate_stale_intervals(r, lv, fresh):
    """Aggregate each observed interval without subtracting global prefixes.

    A large earlier variance can erase a later positive increment in a running
    sum. Local reductions avoid that cancellation; logaddexp also avoids
    exponentiating the latent variance. Trailing, still-stale days are not
    included in the last observed interval. Unscored stale-day labels retain
    their previous convention.
    """
    starts = np.concatenate([[0], fresh[:-1] + 1])
    end = fresh[-1] + 1
    obs = np.zeros_like(r)
    obs[fresh] = np.add.reduceat(r[:end], starts)
    observed_lv = lv.copy()
    observed_lv[fresh] = np.logaddexp.reduceat(lv[:end], starts)
    return obs, observed_lv


def apply_overlays(rng, r, lv, overlay_p):
    """returns observed returns, log conditional variance of the observed return, mask (1 = score this target), descriptions."""
    n, T = r.shape; r = r.copy(); lv = lv.copy()
    on = {k: rng.random(n) < p for k, p in overlay_p.items()}
    # jumps: added to the innovation, so the conditional variance becomes h (1 + lam k^2)
    lamJ = np.exp(rng.uniform(math.log(0.002), math.log(0.05), n)); kJ = rng.uniform(3, 10, n)
    J = (rng.random((n, T)) < lamJ[:, None]) * rng.standard_normal((n, T)) * kJ[:, None] * on['jumps'][:, None]
    r = r + np.exp(lv / 2) * J
    lv = lv + np.where(on['jumps'], np.log1p(lamJ * kJ * kJ), 0.0)[:, None]
    # AR(1) filter on the observed returns (conditional variance unchanged)
    rho = rng.uniform(-0.15, 0.15, n) * on['ar']
    if on['ar'].any():
        prev = np.zeros(n)
        for t in range(T):
            r[:, t] = r[:, t] + rho * prev; prev = r[:, t]
    # drift in units of sigma
    mu = rng.normal(0, 0.03, n) * on['drift']; r = r + mu[:, None]
    # stale prices: zeros on stale days, the accumulated return (and variance) on the next fresh day
    mask = np.ones((n, T)); stale_frac = np.zeros(n)
    q = np.exp(rng.uniform(math.log(0.005), math.log(0.2), n)); m = rng.uniform(1, 4, n)
    b = 1 / m; a = q * b / (1 - q)
    for i in np.flatnonzero(on['stale']):
        u = rng.random(T); s = np.zeros(T, dtype=bool); st = False
        for t in range(T):
            st = (u[t] < (1 - b[i])) if st else (u[t] < a[i])
            s[t] = st
        fresh = np.flatnonzero(~s)
        if len(fresh) == 0:
            continue
        r[i], lv[i] = _aggregate_stale_intervals(r[i], lv[i], fresh)
        mask[i] = (~s).astype(float); stale_frac[i] = s.mean()
    desc = [dict(jumps=bool(on['jumps'][i]), lamJ=float(lamJ[i]) if on['jumps'][i] else 0.0, kJ=float(kJ[i]) if on['jumps'][i] else 0.0,
                 ar=float(rho[i]), drift=float(mu[i]), stale=float(stale_frac[i])) for i in range(n)]
    return r, lv, mask, desc

def initial_scale(r, k=100):
    """RMS of nonzero returns in the supplied prefix, never later observations.

    Callers must ensure that this prefix precedes the forecast. With no nonzero
    history the fallback is the fixed unit scale, not a statistic of future data.
    """
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')
    x = np.asarray(r, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('returns must be a finite one-dimensional array')
    x = x[:k]; x = x[x != 0]
    if not len(x):
        return 1.0
    largest = np.max(np.abs(x))
    return float(largest * np.sqrt(np.mean((x / largest) ** 2)))

def standardise(r):
    s = np.array([initial_scale(row) for row in r]); return r / s[:, None], s

# ----------------------------------------------------------------------------- a batch from the prior
def sample_batch(rng, n, T, weights=None, overlay_p=None, clean=False, family=None, nu=None, gamma=None):
    """n series of length T. Returns dict with r (n,T) standardised observed returns, logv (n,T) log conditional
    variance of the observed return in standardised units, mask (n,T), scale (n,), meta (list of dicts).
    clean=True or family=... switches the contaminations off so that the exact oracles apply; nu/gamma fix the law."""
    weights = dict(PRIOR_WEIGHTS if weights is None else weights); overlay_p = dict(OVERLAY_P if overlay_p is None else overlay_p)
    if clean or family is not None:
        overlay_p = {k: 0.0 for k in overlay_p}
    fams = list(weights); p = np.array([weights[f] for f in fams], dtype=float); p /= p.sum()
    fam_idx = rng.choice(len(fams), size=n, p=p) if family is None else np.full(n, fams.index(family))
    nus, gammas = draw_law(rng, n)
    if nu is not None:
        nus = np.full(n, float(nu))
    if gamma is not None:
        gammas = np.full(n, float(gamma))
    r = np.empty((n, T)); lv = np.empty((n, T)); meta = [None] * n
    for j, f in enumerate(fams):
        idx = np.flatnonzero(fam_idx == j)
        if len(idx) == 0:
            continue
        rr, ll, params = SIMULATORS[f](rng, len(idx), T, nus[idx], gammas[idx])
        r[idx] = rr; lv[idx] = ll
        for i, pr in zip(idx, params):
            meta[i] = dict(family=f, nu=float(nus[i]), gamma=float(gammas[i]), params=pr)
    r, lv, mask, desc = apply_overlays(rng, r, lv, overlay_p)
    r, s = standardise(r); lv = lv - 2 * np.log(s)[:, None]
    for i in range(n):
        meta[i]['overlay'] = desc[i]; meta[i]['scale'] = float(s[i])
        meta[i]['clean'] = not (desc[i]['jumps'] or desc[i]['ar'] != 0 or desc[i]['drift'] != 0 or desc[i]['stale'] > 0)
    return dict(r=r.astype(np.float32), logv=lv.astype(np.float32), mask=mask.astype(np.float32), scale=s, meta=meta)

# ----------------------------------------------------------------------------- exact known-parameter Bayes predictives
def _predictive_from_grid(p_pred, h_lev, r, nu, gamma):
    """p_pred: predictive state probabilities (G,), h_lev variances (G,), r realised return -> (E h, log p(r), log-lik per state)."""
    lp = law_logpdf(r / np.sqrt(h_lev), nu, gamma) - 0.5 * np.log(h_lev)
    m = lp.max(); return float(np.dot(p_pred, h_lev)), float(m + np.log(np.dot(p_pred, np.exp(lp - m)))), lp

def oracle_predictive(meta, r_std):
    """known-parameter Bayes predictive for a CLEAN series in standardised units: (h (T,), logdens (T,)) where h[t] is
    the predictive variance of r[t] made from r[:t] (posterior mean of the conditional variance) and logdens[t] the log
    predictive density of the realised r[t].  None for the rough family (no exact filter) and for contaminated series."""
    if not meta.get('clean', True):
        return None
    f = meta['family']; nu = meta['nu']; gamma = meta['gamma']; s = meta['scale']; pr = meta['params']
    r = np.asarray(r_std, dtype=float) * s; T = len(r)
    h = np.empty(T); ld = np.empty(T)
    if f in ('garch', 'gjr'):
        hh = pr['h0']; om, al, ga, be = pr['omega'], pr['alpha'], pr['gamma'], pr['beta']
        for t in range(T):
            h[t] = hh; e = r[t] / math.sqrt(hh)
            hh = om + (al + ga * (e < 0)) * hh * e * e + be * hh
        ld = law_logpdf(r / np.sqrt(h), nu, gamma) - 0.5 * np.log(h)
    elif f == 'egarch':
        th = math.log(pr['h0']); be, a, g, m1 = pr['beta'], pr['a'], pr['g'], pr['m1']
        for t in range(T):
            h[t] = math.exp(th); e = r[t] / math.sqrt(h[t])
            th = be * th + a * (abs(e) - m1) + g * e
        ld = law_logpdf(r / np.sqrt(h), nu, gamma) - 0.5 * np.log(h)
    elif f == 'sv':
        phi, sw, rho, sd = pr['phi'], pr['sw'], pr['rho'], pr['sd']
        G = 241; width = 4.5 * sd; grid = np.linspace(-width, width, G); dg = grid[1] - grid[0]; h_lev = np.exp(grid)
        sw2 = sw * math.sqrt(1 - rho * rho); edges = np.concatenate([[-np.inf], (grid[1:] + grid[:-1]) / 2, [np.inf]])
        p = np.exp(-0.5 * (grid / sd) ** 2); p /= p.sum()                       # stationary start, as simulated
        if rho == 0.0:                                                            # time-invariant integrated-Gaussian kernel
            Kmat = np.diff(ndtr((edges[None, :] - phi * grid[:, None]) / sw2), axis=1)
        for t in range(T):
            h[t], ld[t], lp = _predictive_from_grid(p, h_lev, r[t], nu, gamma)
            post = p * np.exp(lp - lp.max()); post /= post.sum()
            if rho == 0.0:
                p = post @ Kmat
            else:
                rows = np.flatnonzero(post > 1e-12 * post.max())
                mean_next = phi * grid[rows] + sw * rho * r[t] / np.sqrt(h_lev[rows])   # leverage shift depends on the state
                Kr = np.diff(ndtr((edges[None, :] - mean_next[:, None]) / sw2), axis=1)
                p = post[rows] @ Kr
            p /= p.sum()
    elif f == 'hmm':
        lev = np.array(pr['levels']); P = np.array(pr['P']); K = len(lev); h_lev = np.exp(lev)
        p = np.full(K, 1.0 / K)
        for t in range(T):
            h[t], ld[t], lp = _predictive_from_grid(p, h_lev, r[t], nu, gamma)
            post = p * np.exp(lp - lp.max()); post /= post.sum(); p = post @ P
    elif f == 'lattice':
        lam, delta = pr['lam'], pr['delta']; J = int(math.ceil(3.5 / delta)) + 1
        lev = delta * np.arange(-J, J + 1); h_lev = np.exp(lev); logp = np.full(2 * J + 1, -np.inf); logp[J] = 0.0
        up_ok = np.abs(lev + delta) <= 3.5 + 1e-9; dn_ok = np.abs(lev - delta) <= 3.5 + 1e-9     # the reflected walk: a blocked jump flips its sign
        l_up = np.log(lam / 2 * (up_ok + (~dn_ok)) + 1e-300); l_dn = np.log(lam / 2 * (dn_ok + (~up_ok)) + 1e-300); l_stay = math.log(1 - lam)
        for t in range(T):
            if t > 0:
                up = np.full_like(logp, -np.inf); up[1:] = logp[:-1] + l_up[:-1]
                dn = np.full_like(logp, -np.inf); dn[:-1] = logp[1:] + l_dn[1:]
                logp = np.logaddexp(logp + l_stay, np.logaddexp(up, dn))
            p = np.exp(logp - logp.max()); p /= p.sum()
            h[t], ld[t], lp = _predictive_from_grid(p, h_lev, r[t], nu, gamma)
            logp = logp + lp; logp -= logp.max()
    else:
        return None
    return h / (s * s), ld + math.log(s)

# ----------------------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.default_rng(0)
    print('law: sample moments against the density (mean, var) for (nu, gamma)')
    for nu, g in ((NU_GAUSS, 1.0), (3.0, 1.0), (5.0, 0.8), (30.0, 1.3)):
        e = law_sample(rng, nu, g, 400000)
        x = np.linspace(-14, 14, 280001); f = np.exp(law_logpdf(x, nu, g)); dx = x[1] - x[0]
        print(f'  ({nu:g}, {g:g}): sample mean {e.mean():+.4f} var {e.var():.4f} | density integral {np.sum(f) * dx:.5f} mean {np.sum(x * f) * dx:+.5f} var {np.sum(x * x * f) * dx:.4f} | mean log-lik of the sample {np.mean(law_logpdf(e, nu, g)):.4f}')
    for T in (512, 1024, 2048):
        t0 = time.time(); b = sample_batch(rng, 32, T); dt = time.time() - t0
        rms0 = [np.sqrt(np.mean(b['r'][i, :100][b['r'][i, :100] != 0] ** 2)) for i in range(32)]
        print(f'batch 32 x {T}: {dt:.2f}s; finite {np.isfinite(b["r"]).all() and np.isfinite(b["logv"]).all()}; first-100 rms in [{min(rms0):.3f}, {max(rms0):.3f}]; '
              f'overall rms {np.sqrt(np.mean(b["r"] ** 2)):.2f}; mask mean {b["mask"].mean():.3f}; clean share {np.mean([m["clean"] for m in b["meta"]]):.2f}; families {sorted(set(m["family"] for m in b["meta"]))}')
    for H, ell in ((0.05, 500.0), (0.1, 30.0), (0.5, 120.0)):
        M, s, neg = _embedding(1024, H, ell); z = rng.standard_normal((200, M)) + 1j * rng.standard_normal((200, M))
        x = np.fft.fft(s * z, axis=1).real[:, :1024]
        print(f'rough embedding H={H} ell={ell:g}: most negative eigenvalue zeroed {neg:.2e}; sample variance {x.var():.3f} (target 1), lag-1 corr {np.mean(x[:, 1:] * x[:, :-1]) / x.var():.3f} vs matern {matern_cov(1.0, H, ell):.3f}')
    print('oracle checks on clean series (T=1500, positions 200+): true-parameter filter minus a perturbed-parameter filter (must be > 0 by Gibbs),')
    print('  and the latent-variance plug-in minus the oracle (0 for GARCH-type where the variance is observable, > 0 for latent-state families)')
    for f in ('garch', 'gjr', 'egarch', 'sv', 'hmm', 'lattice'):
        bb = sample_batch(rng, 6, 1500, family=f); gaps = []; qgap = []; t0 = time.time()
        for i in range(6):
            m = bb['meta'][i]; r = bb['r'][i].astype(float)
            h, ld = oracle_predictive(m, r)
            m2 = dict(m); m2['params'] = dict(m['params'])
            for k, v in m2['params'].items():
                if isinstance(v, float) and k not in ('h0', 'lam', 'sd'):
                    m2['params'][k] = v * 0.9 if v != 0 else 0.05
                if k == 'lam':
                    m2['params'][k] = v * 2
            if f == 'hmm':
                m2['params']['levels'] = (np.array(m['params']['levels']) * 0.8).tolist()
            h2, ld2 = oracle_predictive(m2, r)
            plug = law_logpdf(r / np.exp(bb['logv'][i] / 2), m['nu'], m['gamma']) - 0.5 * bb['logv'][i]
            gaps.append(np.mean(ld[200:] - ld2[200:])); qgap.append(np.mean(plug[200:] - ld[200:]))
        print(f'  {f:8s}: true - perturbed {np.mean(gaps):+.4f} nats/step; plug-in - oracle {np.mean(qgap):+.4f}; QLIKE(oracle h) - QLIKE(true h) {np.mean([np.mean((r2 := bb["r"][i][200:].astype(float) ** 2) / (hh := oracle_predictive(bb["meta"][i], bb["r"][i].astype(float))[0][200:]) + np.log(hh) - r2 / np.exp(bb["logv"][i][200:]) - bb["logv"][i][200:]) for i in range(6)]):+.4f}; {(time.time() - t0) / 6:.2f}s per series')
    for rho_case in ('zero', 'lev'):
        while True:
            bb = sample_batch(rng, 1, 600, family='sv'); m = bb['meta'][0]; pr = m['params']
            if (pr['rho'] == 0.0) == (rho_case == 'zero'):
                break
        r = bb['r'][0].astype(float) * m['scale']; h, ld = oracle_predictive(m, bb['r'][0])
        N = 100000; th = rng.normal(0, pr['sd'], N); lpf = np.empty(600)
        for t in range(600):
            lw = law_logpdf(r[t] * np.exp(-th / 2), m['nu'], m['gamma']) - th / 2; mx = lw.max(); w = np.exp(lw - mx); lpf[t] = mx + math.log(w.mean())
            idx = rng.choice(N, N, p=w / w.sum()); th = th[idx]; z = r[t] * np.exp(-th / 2)
            th = pr['phi'] * th + pr['sw'] * (pr['rho'] * z + math.sqrt(1 - pr['rho'] ** 2) * rng.standard_normal(N))
        print(f'sv grid filter vs bootstrap PF (100k particles), rho = {pr["rho"]:+.2f}, sw = {pr["sw"]:.3f}, phi = {pr["phi"]:.4f}: mean log density {np.mean(ld):.5f} vs {np.mean(lpf) + math.log(m["scale"]):.5f}')

if __name__ == '__main__':
    _selftest()
