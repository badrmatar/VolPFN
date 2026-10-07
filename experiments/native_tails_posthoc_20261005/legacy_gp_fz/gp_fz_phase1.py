"""
GP-FZ Phase 1: joint (VaR, ES) via a latent quantile state plus a gap parameter.

What changes from Phase 0: the observation model gains Taylor (2019)'s
multiplicative ES mapping,

    ES_t = (1 + exp(gamma)) * q_t,

which guarantees ES_t < q_t < 0 automatically (for q_t < 0). The AL scale is
no longer a fixed constant but derived from the state,

    sigma_t = -(1 - alpha) * exp(gamma) * q_t,

which is exactly the scale that makes the AL's theoretical ES equal the
mapping above. One likelihood therefore identifies BOTH quantities: the
location gives VaR, the scale gives ES.

DESIGN NOTE (learned the hard way -- keep for the write-up): gamma is a
STATIC parameter, as in Taylor (2019), not a second random walk. Letting both
the level q_t and the gap gamma_t evolve freely is unidentified under the AL
likelihood: a higher level with a wider gap produces nearly the same density
as a lower level with a narrower gap (a likelihood ridge), and the filtered
level drifts up until the violation rate roughly doubles. Pinning gamma
(filtered as a static parameter with tiny jitter) closes the ridge. This is
WHY Taylor's multiplicative spec has a constant gamma_0.

Run directly: simulates a KNOWN (q_t, gamma) path, filters it, and checks
(1) violation rate ~ alpha, (2) both VaR and ES paths recovered, (3) ES < VaR
everywhere, (4) FZ0 score of the filtered pair close to that of the true pair
(FZ0 is minimised by the truth in expectation, not on every finite sample,
so the filtered pair can legitimately edge it out on a single path).
"""

import numpy as np

from gp_fz_phase0 import systematic_resample


def al_log_density(r, q, sigma, alpha):
    """AL log-density, vectorised over particles with PER-PARTICLE scale."""
    u = r - q
    indicator = (r < q).astype(float)
    return (
        np.log(alpha * (1 - alpha) / sigma)
        - u * (alpha - indicator) / sigma
    )


def fz0_loss(r, q, e, alpha):
    """Patton-Ziegel-Chen (2019) FZ0 joint loss for (VaR, ES). Requires e < 0.

    Strictly consistent for the pair: uniquely minimised in expectation at the
    true (VaR, ES). Lower is better. Headline score for Section 4.
    """
    hit = (r <= q).astype(float)
    return -hit * (q - r) / (alpha * e) + q / e + np.log(-e) - 1.0


def particle_filter_var_es(returns, alpha, sigma_eta, sigma_nu=0.002,
                           n_particles=2000, seed=0,
                           q_init=None, gamma_init=np.log(0.4)):
    """Bootstrap particle filter: q_t is a random-walk state, gamma a static
    parameter carried by the particles (initial spread + tiny jitter sigma_nu;
    resampling concentrates it on values the data supports).

    Returns filtered VaR, filtered ES, and the ESS trace.
    """
    rng = np.random.default_rng(seed)
    T = len(returns)

    if q_init is None:
        q_init = min(np.quantile(returns[:20], alpha), -1e-2)
    q = np.full(n_particles, q_init)
    g = gamma_init + rng.normal(0.0, 0.3, n_particles)  # spread = prior over the gap
    weights = np.full(n_particles, 1.0 / n_particles)

    var_est = np.zeros(T)
    es_est = np.zeros(T)
    ess_trace = np.zeros(T)

    for t in range(T):
        # 1. propagate: level moves, gap only jitters
        q = q + rng.normal(0.0, sigma_eta, n_particles)
        g = g + rng.normal(0.0, sigma_nu, n_particles)

        # PREDICTIVE estimates: propagated particles, yesterday's weights
        var_est[t] = np.sum(weights * q)
        es_est[t] = np.sum(weights * (1 + np.exp(g)) * q)

        # 2. per-particle AL scale from the state; particles with q >= 0 invalid
        sigma = -(1 - alpha) * np.exp(g) * q
        valid = sigma > 0
        log_w = np.full(n_particles, -np.inf)
        log_w[valid] = al_log_density(returns[t], q[valid], sigma[valid], alpha)

        m = log_w.max()
        if not np.isfinite(m):
            weights = np.full(n_particles, 1.0 / n_particles)
        else:
            w = np.exp(log_w - m) * weights
            weights = w / w.sum()
        ess = 1.0 / np.sum(weights ** 2)
        ess_trace[t] = ess

        # 3. resample the joint state when ESS collapses
        if ess < n_particles / 2:
            idx = systematic_resample(weights, rng)
            q, g = q[idx], g[idx]
            weights = np.full(n_particles, 1.0 / n_particles)

    return var_est, es_est, ess_trace


def simulate_var_es_process(T, alpha, sigma_eta,
                            q0=-1.5, gamma0=np.log(0.4), seed=1):
    """Synthetic data with a KNOWN q_t path and STATIC gamma (Taylor's spec),
    r_t ~ AL(q_t, sigma_t, alpha). The AL's theoretical ES equals
    (1+exp(gamma0)) q_t, so the filter's ES estimate has a known target.
    """
    rng = np.random.default_rng(seed)
    true_q = np.zeros(T)
    true_es = np.zeros(T)
    returns = np.zeros(T)
    q, g = q0, gamma0
    for t in range(T):
        q = min(q + rng.normal(0.0, sigma_eta), -0.05)  # guard: stay negative
        sigma = -(1 - alpha) * np.exp(g) * q
        true_q[t] = q
        true_es[t] = (1 + np.exp(g)) * q
        u = rng.uniform()
        if u < alpha:
            r = q + sigma / (1 - alpha) * np.log(u / alpha)
        else:
            r = q - sigma / alpha * np.log((1 - u) / (1 - alpha))
        returns[t] = r
    return returns, true_q, true_es


if __name__ == "__main__":
    ALPHA = 0.05
    SIGMA_ETA = 0.02
    T = 1000
    N_PARTICLES = 2000

    returns, true_q, true_es = simulate_var_es_process(T, ALPHA, SIGMA_ETA)
    var_f, es_f, ess = particle_filter_var_es(returns, ALPHA, SIGMA_ETA,
                                              n_particles=N_PARTICLES)

    rmse_var = np.sqrt(np.mean((var_f - true_q) ** 2))
    rmse_es = np.sqrt(np.mean((es_f - true_es) ** 2))
    violations = np.mean(returns < var_f)
    fz0_filtered = np.mean(fz0_loss(returns, var_f, es_f, ALPHA))
    fz0_true = np.mean(fz0_loss(returns, true_q, true_es, ALPHA))

    print(f"RMSE filtered VaR vs true q_t   : {rmse_var:.4f}")
    print(f"RMSE filtered ES  vs true ES_t  : {rmse_es:.4f}")
    print(f"Target violation rate (alpha)   : {ALPHA:.3f}")
    print(f"Empirical violation rate        : {violations:.3f}")
    print(f"ES < VaR on every day           : {bool(np.all(es_f < var_f))}")
    print(f"Mean FZ0 score, filtered pair   : {fz0_filtered:.4f}")
    print(f"Mean FZ0 score, TRUE pair       : {fz0_true:.4f}   (in-expectation target)")
    print(f"Mean effective sample size      : {ess.mean():.1f} / {N_PARTICLES}")
