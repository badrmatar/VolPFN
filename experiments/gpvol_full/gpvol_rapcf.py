"""Full GP-Vol with RAPCF inference (Wu, Hernandez-Lobato & Ghahramani, NIPS 2014).

Model (paper eqs. 5-7, released code model/*.m):
    x_t ~ N(0, exp(v_t))
    v_t = f(v_{t-1}, x_{t-1}) + eps_t,   eps_t ~ N(0, sn^2)
    f ~ GP(m, k),  m(v, x) = a*v + b*x + c          (GPML meanSum{meanLinear, meanConst})
                   k(z, z') = sf^2 exp(-|z - z'|^2 / (2 ell^2))   (GPML covSEiso)
Hyperparameters theta = (a, b, c, log ell, log sf, log sn), i.e. the GPML hyp
vector [mean; cov; lik] used by the authors' code.

Inference is RAPCF (paper Algorithm 1; released racpf.m): N particles, each
carrying the FULL chain of past latent states (no window, no sparse
approximation), Liu-West kernel shrinkage/jitter on theta, an auxiliary
first-stage resampling step, proposal from the exact GP predictive, and
importance-weight correction.

Settings taken from the authors' released code (bitbucket
jmh233/code-gpvol-nips2014, commit ca98bae) unless stated:
  * prior on theta: N(init, I) with init = [0.98, -0.1, 0, 0, 0, log 0.1]
    (demo.m learning block; racpf.m: randn + initHypVec)
  * burn-in states: log of rolling 5-observation empirical variance plus
    N(0, 0.1^2) particle noise (demo.m, helpers/empiricalStd.m, racpf.m)
  * jitter: diagonal, unweighted per-parameter std of the previous particles
    times h = sqrt(1 - lambda^2) (racpf.m)
  * systematic resampling (resample/resample.m default)
  * lambda = 0.95 and burn-in 100 follow the paper's stated real-data settings
    (the released demo uses 0.99 and 50).

Deliberate differences from the released code, each documented:
  D1  Shrinkage target uses the NORMALISED weighted particle mean. racpf.m
      passes unnormalised log incremental weights to rapfPointEstimate.m, which
      computes sum(exp(logw) .* theta) / N, rescaling the target by the mean
      incremental weight.
  D2  The GP test input is (v_{t-1}, x_{t-1}), as in the paper's eq. 6 and the
      authors' own generator (gpVolGenerator.m). gpVolPredFun.m takes the test
      input after truncating the training inputs, so it uses (v_{t-2}, x_{t-2}),
      the last training input. Pass code_indexing=True to reproduce the released
      behaviour as a diagnostic.
  D3  The first-stage point estimate is the GP predictive MEAN (paper eq. 16);
      gpVolPredFun.m returns a random draw from the predictive.
  D4  The one-step predictive for x_t is formed before x_t is seen, from the
      time t-1 weights and the kernel-centre parameters m_t, integrating each
      particle's Gaussian predictive over v_t exactly (Gauss-Hermite). The
      released code uses one sampled v per particle under theta_{t-1}; both
      target the same predictive distribution.
  D5  Burn-in empirical variances use only the burn-in window (empiricalStd.m
      seeds its first values with the std of the whole series passed in).
"""
import math

import numpy as np
from scipy.linalg import solve_triangular

PARAM_NAMES = ("a_v", "b_x", "c", "log_ell", "log_sf", "log_sn")
DEFAULT_INIT = np.array([0.98, -0.1, 0.0, 0.0, 0.0, math.log(0.1)])
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(40)
_LOG_GH_W = np.log(_GH_W / math.sqrt(math.pi))
_LOG_2PI = math.log(2.0 * math.pi)


def empirical_log_variance(x, init_win=5, win=5):
    """log of helpers/empiricalStd.m (MATLAB nanstd, N-1), on the given window only."""
    x = np.asarray(x, float)
    s = np.empty(len(x))
    s[:init_win] = np.std(x, ddof=1)
    for i in range(init_win, len(x)):
        s[i] = np.std(x[max(0, i - win + 1): i + 1], ddof=1)
    return np.log(np.maximum(s, 1e-8) ** 2)


def systematic_resample(w, rng):
    n = len(w)
    q = np.cumsum(w)
    q[-1] = 1.0
    u = (np.arange(n) + rng.random()) / n
    return np.minimum(np.searchsorted(q, u, side="right"), n - 1)


def log_normal_obs(x, v):
    """log N(x; 0, exp(v)), vectorised over v (overflow-safe; -inf only for x != 0, v -> -inf)."""
    q = x * x * np.exp(np.minimum(-np.asarray(v, float), 709.0)) if x != 0 else 0.0
    return -0.5 * (_LOG_2PI + v + q)


def gp_predict(chains, x, t, params, chunk=32, code_indexing=False, jitter=0.0):
    """Exact GP predictive of v_t for every particle.

    chains[:, :t] hold v_0..v_{t-1}; x[:t] the observed x_0..x_{t-1}.
    Training pairs (v_s, x_s) -> v_{s+1}, s = 0..t-2 (all of them).
    Test input (v_{t-1}, x_{t-1}); with code_indexing, (v_{t-2}, x_{t-2}).
    Returns predictive mean and variance (including sn^2) and a failure mask.
    """
    n_part = chains.shape[0]
    a, b, c = params[:, 0], params[:, 1], params[:, 2]
    ell2 = np.exp(2.0 * params[:, 3])
    sf2 = np.exp(2.0 * params[:, 4])
    sn2 = np.exp(2.0 * params[:, 5])
    n = t - 1
    mu = np.empty(n_part)
    s2 = np.empty(n_part)
    failed = np.zeros(n_part, bool)
    if code_indexing:
        zv, zx = chains[:, t - 2], np.full(n_part, x[t - 2])
    else:
        zv, zx = chains[:, t - 1], np.full(n_part, x[t - 1])
    m_star = a * zv + b * zx + c
    if n <= 0:
        return m_star, sf2 + sn2, failed
    X = x[:n]
    dx2 = (X[:, None] - X[None, :]) ** 2
    idx = np.arange(n)
    for lo in range(0, n_part, chunk):
        hi = min(n_part, lo + chunk)
        Zv = chains[lo:hi, :n]
        Y = chains[lo:hi, 1:n + 1]
        K = (Zv[:, :, None] - Zv[:, None, :]) ** 2
        K += dx2[None]
        K *= (-0.5 / ell2[lo:hi])[:, None, None]
        np.exp(K, out=K)
        K *= sf2[lo:hi, None, None]
        K[:, idx, idx] += (sn2[lo:hi] + jitter)[:, None]
        kst = sf2[lo:hi, None] * np.exp(-0.5 * ((Zv - zv[lo:hi, None]) ** 2 + (X[None] - zx[lo:hi, None]) ** 2)
                                        / ell2[lo:hi, None])
        resid = Y - (a[lo:hi, None] * Zv + b[lo:hi, None] * X[None] + c[lo:hi, None])
        try:
            L = np.linalg.cholesky(K)
            batch_ok = True
        except np.linalg.LinAlgError:
            batch_ok = False
        for j in range(hi - lo):
            i = lo + j
            if batch_ok:
                Lj = L[j]
            else:
                try:
                    Lj = np.linalg.cholesky(K[j])
                except np.linalg.LinAlgError:
                    try:
                        Kj = K[j] + 1e-6 * np.eye(n)
                        Lj = np.linalg.cholesky(Kj)
                    except np.linalg.LinAlgError:
                        failed[i] = True
                        mu[i], s2[i] = m_star[i], sf2[i] + sn2[i]
                        continue
            q = solve_triangular(Lj, np.stack([resid[j], kst[j]], axis=1), lower=True, check_finite=False)
            mu[i] = m_star[i] + q[:, 1] @ q[:, 0]
            s2[i] = max(sf2[i] - q[:, 1] @ q[:, 1], 1e-12) + sn2[i]
    return mu, s2, failed


def predictive_mixture(w, mu, s2, x_obs):
    """Mixture over particles of N(x; 0, e^v), v ~ N(mu_i, s2_i).

    Returns E[x^2] = sum w_i exp(mu_i + s2_i/2) and log p(x_obs)."""
    h = float(np.dot(w, np.exp(mu + 0.5 * s2)))
    v = mu[:, None] + np.sqrt(2.0 * s2)[:, None] * _GH_X[None, :]
    lp = log_normal_obs(x_obs, v) + _LOG_GH_W[None, :]
    lp_part = np.logaddexp.reduce(lp, axis=1)
    with np.errstate(divide="ignore"):
        lw = np.log(w)
    logdens = float(np.logaddexp.reduce(lw + lp_part))
    return h, logdens


def normalise_log(lw):
    """Normalised weights from log weights; NaN treated as -inf. All -inf -> uniform (counted by caller)."""
    lw = np.where(np.isnan(lw), -np.inf, lw)
    m = np.max(lw)
    if not np.isfinite(m):
        return np.full(len(lw), 1.0 / len(lw))
    w = np.exp(lw - m)
    return w / w.sum()


class RAPCF:
    """Resumable RAPCF state machine; RAPCF(...).run() equals the one-shot filter exactly.

    Forecast arrays are aligned with x: entry t is the one-step predictive for
    x_t made with information x_0..x_{t-1}; entries t < burn_in are NaN.
    """
    OUT_KEYS = ("h", "logdens", "apf_loglik", "ess_first", "ess_second", "gp_failures", "weight_collapse", "v_mean")

    def __init__(self, x, n_particles=200, burn_in=100, lam=0.95, init=DEFAULT_INIT, prior_sd=1.0,
                 seed=0, learn_hyp=True, code_indexing=False, chunk=32, init_params=None):
        self.rng = np.random.default_rng(seed)
        self.x = np.asarray(x, float)
        T, N, B = len(self.x), int(n_particles), int(burn_in)
        if T <= B + 1:
            raise ValueError("Series shorter than burn-in")
        self.cfg = dict(n_particles=N, burn_in=B, lam=float(lam), learn_hyp=bool(learn_hyp),
                        code_indexing=bool(code_indexing), chunk=int(chunk))
        self.chains = np.zeros((N, T))
        init_v = empirical_log_variance(self.x[:B])
        self.chains[:, :B] = init_v[None, :] + 0.1 * self.rng.standard_normal((N, B))
        if init_params is not None:
            self.P = np.array(init_params, float).copy()
        elif learn_hyp:
            self.P = np.asarray(init, float)[None, :] + prior_sd * self.rng.standard_normal((N, len(init)))
        else:
            self.P = np.repeat(np.asarray(init, float)[None, :], N, axis=0)
        self.lw = np.zeros(N)
        self.t = B
        self.out = {k: np.full(T, np.nan) for k in self.OUT_KEYS}
        self.out["param_mean"] = np.full((T, self.P.shape[1]), np.nan)
        self.out["param_sd"] = np.full((T, self.P.shape[1]), np.nan)

    @property
    def done(self):
        return self.t >= len(self.x)

    def step(self):
        t, x, N, c = self.t, self.x, self.cfg["n_particles"], self.cfg
        lam, h_scale = c["lam"], math.sqrt(max(1.0 - c["lam"] ** 2, 0.0))
        P, lw, chains, rng, out = self.P, self.lw, self.chains, self.rng, self.out
        w = normalise_log(lw)
        if c["learn_hyp"]:
            theta_bar = w @ P
            m = lam * P + (1.0 - lam) * theta_bar[None, :]
        else:
            m = P
        mu, s2, fail1 = gp_predict(chains, x, t, m, chunk=c["chunk"], code_indexing=c["code_indexing"])
        # forecast for x_t, formed before x_t is used
        out["h"][t], out["logdens"][t] = predictive_mixture(w, mu, s2, x[t])
        # auxiliary first stage at the predictive mean (paper eq. 16-17)
        ll_mu = log_normal_obs(x[t], mu)
        with np.errstate(divide="ignore"):
            g = np.log(w) + ll_mu
        g[fail1] = -np.inf
        collapse = 0 if np.isfinite(np.max(np.where(np.isnan(g), -np.inf, g))) else 1
        gw = normalise_log(g)
        out["ess_first"][t] = 1.0 / np.sum(gw ** 2)
        idx = systematic_resample(gw, rng)
        chains[:, :t] = chains[idx, :t]
        if c["learn_hyp"]:
            sd = np.std(P, axis=0, ddof=1)
            P_new = m[idx] + rng.standard_normal(m.shape) * sd[None, :] * h_scale
        else:
            P_new = P[idx]
        mu2, s22, fail2 = gp_predict(chains, x, t, P_new, chunk=c["chunk"], code_indexing=c["code_indexing"])
        v_new = mu2 + np.sqrt(s22) * rng.standard_normal(N)
        lw_new = log_normal_obs(x[t], v_new) - ll_mu[idx]
        lw_new[fail2] = -np.inf
        chains[:, t] = v_new
        # APF estimate of log p(x_t | x_{0:t-1}) (Pitt 2002), diagnostic only
        finite_g = np.logaddexp.reduce(g[np.isfinite(g)]) if np.isfinite(g).any() else -np.inf
        finite_lw = lw_new[np.isfinite(lw_new)]
        out["apf_loglik"][t] = finite_g + (np.logaddexp.reduce(finite_lw) - math.log(N) if finite_lw.size else -np.inf)
        if not np.isfinite(np.max(np.where(np.isnan(lw_new), -np.inf, lw_new))):
            collapse += 2
        out["weight_collapse"][t] = collapse
        w2 = normalise_log(lw_new)
        out["ess_second"][t] = 1.0 / np.sum(w2 ** 2)
        out["gp_failures"][t] = int(fail1.sum() + fail2.sum())
        self.P, self.lw = P_new, lw_new
        out["param_mean"][t] = w2 @ P_new
        out["param_sd"][t] = np.sqrt(np.maximum(w2 @ (P_new - out["param_mean"][t]) ** 2, 0))
        out["v_mean"][t] = w2 @ v_new
        self.t = t + 1

    def run(self, budget_seconds=None, progress=None):
        import time as _time
        t0 = _time.time()
        while not self.done:
            self.step()
            if progress is not None:
                progress(self.t, len(self.x))
            if budget_seconds is not None and _time.time() - t0 > budget_seconds:
                break
        return self

    def result(self):
        out = dict(self.out)
        out["final_params"] = self.P
        out["final_log_weights"] = self.lw
        out["final_chains_mean"] = normalise_log(self.lw) @ self.chains
        return out

    def save(self, path):
        import json as _json
        path = str(path)
        arrays = dict(x=self.x, chains=self.chains, P=self.P, lw=self.lw, t=np.array(self.t),
                      **{"out__" + k: v for k, v in self.out.items()})
        meta = dict(cfg=self.cfg, rng=self.rng.bit_generator.state)
        tmp = path + ".tmp.npz"
        np.savez(tmp, meta=np.array(_json.dumps(meta)), **arrays)
        import os as _os
        _os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        import json as _json
        d = np.load(path, allow_pickle=False)
        self = cls.__new__(cls)
        meta = _json.loads(str(d["meta"]))
        self.cfg = meta["cfg"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = meta["rng"]
        self.x, self.chains, self.P, self.lw = d["x"], d["chains"].copy(), d["P"].copy(), d["lw"].copy()
        self.t = int(d["t"])
        self.out = {k[5:]: d[k].copy() for k in d.files if k.startswith("out__")}
        return self


def run_rapcf(x, n_particles=200, burn_in=100, lam=0.95, init=DEFAULT_INIT, prior_sd=1.0,
              seed=0, learn_hyp=True, code_indexing=False, chunk=32, record_params=True,
              init_params=None, progress=None):
    """One-shot RAPCF over standardised returns x (see RAPCF)."""
    f = RAPCF(x, n_particles=n_particles, burn_in=burn_in, lam=lam, init=init, prior_sd=prior_sd, seed=seed,
              learn_hyp=learn_hyp, code_indexing=code_indexing, chunk=chunk, init_params=init_params)
    return f.run(progress=progress).result()


def simulate_gpvol(T, params, rng, v1=0.0):
    """Authors' generator (gpVolGenerator.m): exact sequential GP draws."""
    a, b, c, log_ell, log_sf, log_sn = params
    ell2, sf2, sn2 = math.exp(2 * log_ell), math.exp(2 * log_sf), math.exp(2 * log_sn)
    v = np.zeros(T)
    x = np.zeros(T)
    v[0] = v1
    x[0] = rng.standard_normal() * math.exp(0.5 * v[0])
    for t in range(1, T):
        Zv, Zx, Y = v[:t - 1], x[:t - 1], v[1:t]
        zs = np.array([v[t - 1], x[t - 1]])
        m_star = a * zs[0] + b * zs[1] + c
        if t - 1 == 0:
            mu, s2 = m_star, sf2 + sn2
        else:
            D = (Zv[:, None] - Zv[None]) ** 2 + (Zx[:, None] - Zx[None]) ** 2
            K = sf2 * np.exp(-0.5 * D / ell2) + sn2 * np.eye(t - 1)
            k = sf2 * np.exp(-0.5 * ((Zv - zs[0]) ** 2 + (Zx - zs[1]) ** 2) / ell2)
            alpha = np.linalg.solve(K, Y - (a * Zv + b * Zx + c))
            mu = m_star + k @ alpha
            s2 = sf2 - k @ np.linalg.solve(K, k) + sn2
        v[t] = mu + math.sqrt(max(s2, 1e-12)) * rng.standard_normal()
        x[t] = rng.standard_normal() * math.exp(0.5 * v[t])
    return x, v
