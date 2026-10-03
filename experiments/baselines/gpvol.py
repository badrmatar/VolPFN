#!/usr/bin/env python3
"""GP volatility baseline based on Wu, Hernandez-Lobato & Ghahramani (ICML 2014).

The observation model is x_t ~ N(0, exp(v_t)). A Gaussian process models the
transition from (v_{t-1}, x_{t-1}) to v_t, with linear mean a*v + b*x,
a squared-exponential kernel and Gaussian transition noise.

The auxiliary particle filter uses Liu-West parameter jitter and importance
weights. Each particle retains a bounded history of transitions: both the
particle count and history window default to 60. This is a finite-memory
adaptation of the paper's RAPCF, rather than a full-history implementation.

Reported log densities use 7-point Gauss-Hermite quadrature. VaR and ES use a
moment-matched Gaussian mixture, rather than the full variance mixture.
The initial variance uses the first 100 returns (or the available shorter
prefix). Early outputs therefore use future information through that scale.
The caller must select the scoring window; the burn argument does not mask outputs.

Example:
    from experiments.baselines.gpvol import run_gpvol
    out = run_gpvol(r, burn=500, levels=(0.99, 0.975))

Outputs h, logscore, var<level> and es<level> align with the input returns.
"""
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import norm

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(
    7
)  # E[g(v)], v ~ N(mu, s2), via Gauss-Hermite quadrature
_GH_W = _GH_W / math.sqrt(math.pi)


def _gh_expect(mu, s2, g):
    v = mu + math.sqrt(2.0 * max(s2, 1e-12)) * _GH_X
    return float(np.dot(_GH_W, g(v)))


class _Particle:
    __slots__ = ('theta', 'Zv', 'Zx', 'Y', 'v_last', 'x_last')

    def __init__(self, theta, v0):
        self.theta = theta  # (a, b, log_sigma_n, log_sigma_f, log_l)
        self.Zv = np.empty(0)
        self.Zx = np.empty(0)
        self.Y = np.empty(0)
        self.v_last = v0
        self.x_last = 0.0

    def copy(self):
        p = _Particle(self.theta.copy(), self.v_last)
        p.Zv = self.Zv.copy()
        p.Zx = self.Zx.copy()
        p.Y = self.Y.copy()
        p.x_last = self.x_last
        return p


def _kernel_vec(v1, x1, v2, x2, sf2, inv2l2):
    return sf2 * np.exp(-((v1 - v2) ** 2 + (x1 - x2) ** 2) * inv2l2)


def _gp_predict(part, theta, window):
    """posterior mean/var of f(v_last, x_last) given part's buffer (trimmed to `window`), under `theta` (which may
    differ from part.theta -- the shrunk point estimate uses a different theta on the same buffer).
    """
    a, b, log_sn, log_sf, log_l = theta
    sn2 = math.exp(2 * log_sn)
    sf2 = math.exp(2 * log_sf)
    inv2l2 = 1.0 / (2.0 * math.exp(2 * log_l))
    Zv, Zx, Y = part.Zv[-window:], part.Zx[-window:], part.Y[-window:]
    m_star = a * part.v_last + b * part.x_last
    if len(Zv) == 0:
        return m_star, sf2 + sn2
    mZ = a * Zv + b * Zx
    dvv = Zv[:, None] - Zv[None, :]
    dxx = Zx[:, None] - Zx[None, :]
    K = sf2 * np.exp(-(dvv**2 + dxx**2) * inv2l2) + (sn2 + 1e-6) * np.eye(len(Zv))
    k_star = _kernel_vec(Zv, Zx, part.v_last, part.x_last, sf2, inv2l2)
    try:
        c = cho_factor(K, lower=True)
        alpha = cho_solve(c, Y - mZ)
        beta = cho_solve(c, k_star)
        var_f = sf2 - float(k_star @ beta)
    except LinAlgError:
        return m_star, sf2 + sn2
    mu_f = m_star + float(k_star @ alpha)
    return mu_f, max(var_f, 1e-10) + sn2


def run_gpvol(r, burn=500, levels=(0.99, 0.975), N=60, window=60, rho=0.97, seed=0):
    """Run the finite-memory GP volatility filter on a return series.

    Return h, logscore, var<level> and es<level> arrays aligned with r.
    Initialisation uses up to the first 100 returns. burn is accepted for
    interface compatibility but does not mask outputs; select the scoring
    window in the caller.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float)
    T = len(r)
    s0 = float(np.mean(r[:100] ** 2)) if len(r) >= 5 else float(np.mean(r**2) + 1e-8)
    v0 = math.log(max(s0, 1e-8))
    # Parameter priors used for the daily-return experiments.
    a0 = rng.normal(0.85, 0.15, N)
    b0 = rng.normal(0.0, 0.3, N)
    log_sn0 = rng.normal(math.log(0.3), 0.3, N)
    log_sf0 = rng.normal(math.log(1.0), 0.3, N)
    log_l0 = rng.normal(math.log(2.0), 0.3, N)
    parts = [
        _Particle(np.array([a0[i], b0[i], log_sn0[i], log_sf0[i], log_l0[i]]), v0) for i in range(N)
    ]
    logw = np.full(N, -math.log(N))
    out = {k: np.full(T, np.nan) for k in ('h', 'logscore')}
    for lv in levels:
        out[f'var{lv:g}'] = np.full(T, np.nan)
        out[f'es{lv:g}'] = np.full(T, np.nan)
    for t in range(T):
        w = np.exp(logw - logw.max())
        w /= w.sum()
        mus = np.empty(N)
        s2s = np.empty(N)
        for i, p in enumerate(parts):
            mus[i], s2s[i] = _gp_predict(p, p.theta, window)
        rt2 = r[t] ** 2
        dens = np.array(
            [
                _gh_expect(mus[i], s2s[i], lambda v: np.exp(-0.5 * rt2 * np.exp(-v) - 0.5 * v))
                for i in range(N)
            ]
        ) / math.sqrt(2 * math.pi)
        out['logscore'][t] = math.log(max(float(np.dot(w, dens)), 1e-300))
        hcomp = np.exp(mus + 0.5 * s2s)
        out['h'][t] = float(np.dot(w, hcomp))
        sd = np.sqrt(hcomp)
        for lv in levels:
            a_ = 1 - lv
            lo, hi = -20.0 * math.sqrt(out['h'][t] + 1e-12), 0.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                cdf = float(np.dot(w, norm.cdf(mid / sd)))
                if cdf < a_:
                    lo = mid
                else:
                    hi = mid
            q = 0.5 * (lo + hi)
            z = q / sd
            es = float(np.dot(w, sd * norm.pdf(z))) / a_
            out[f'var{lv:g}'][t] = -q
            out[f'es{lv:g}'][t] = es
        theta_mat = np.array([p.theta for p in parts])
        theta_bar = w @ theta_mat
        m_shrunk = rho * theta_mat + (1 - rho) * theta_bar
        mu_pt = np.array([_gp_predict(parts[i], m_shrunk[i], window)[0] for i in range(N)])
        g = logw - 0.5 * rt2 * np.exp(-mu_pt) - 0.5 * mu_pt
        g -= g.max()
        gw = np.exp(g)
        gw /= gw.sum()
        idx = rng.choice(N, N, p=gw)
        chains = [parts[i].copy() for i in idx]
        m_sel = m_shrunk[idx]
        theta_sel = theta_mat[idx]
        Vcov = np.cov(theta_mat.T, aweights=np.maximum(w, 1e-12)) + 1e-8 * np.eye(5)
        jitter = rng.multivariate_normal(np.zeros(5), (1 - rho**2) * Vcov, size=N)
        theta_new = m_sel + jitter
        mu_prop = np.empty(N)
        s2_prop = np.empty(N)
        for i in range(N):
            mu_prop[i], s2_prop[i] = _gp_predict(chains[i], theta_new[i], window)
        v_new = rng.normal(mu_prop, np.sqrt(np.maximum(s2_prop, 1e-12)))
        ll_true = -0.5 * rt2 * np.exp(-v_new) - 0.5 * v_new
        ll_aux = -0.5 * rt2 * np.exp(-mu_pt[idx]) - 0.5 * mu_pt[idx]
        logw = ll_true - ll_aux
        logw -= logw.max()
        logw -= math.log(np.sum(np.exp(logw)))
        for i in range(N):
            c = chains[i]
            c.Zv = np.append(c.Zv, c.v_last)[-window:]
            c.Zx = np.append(c.Zx, c.x_last)[-window:]
            c.Y = np.append(c.Y, v_new[i])[-window:]
            c.v_last = v_new[i]
            c.x_last = r[t]
            c.theta = theta_new[i]
        parts = chains
    return out


if __name__ == '__main__':
    import time

    rng = np.random.default_rng(0)
    r = rng.standard_normal(300) * 0.01
    t0 = time.time()
    out = run_gpvol(r, burn=100)
    ok = ~np.isnan(out['h'])
    dt = time.time() - t0
    print(
        f'{dt:.1f}s for T=300, N=60, window=60 ({dt / 300 * 1000:.1f} ms/step); mean logscore {np.nanmean(out["logscore"]):.4f}; mean h {np.nanmean(out["h"][ok]):.5f}'
    )
    print(
        f'extrapolated: T=8314 (stocks) ~= {dt / 300 * 8314 / 60:.1f} min/series; T=5379 (fx) ~= {dt / 300 * 5379 / 60:.1f} min/series (serial; --procs parallelises across series)'
    )
