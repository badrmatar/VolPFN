"""
GP-FZ Phase 0: plain-transition Asymmetric Laplace particle filter for VaR.

This is the first rung of the agreed build order:
  Phase 0 -> plain (random-walk) transition, VaR only, bootstrap particle filter over q_t.
  Phase 1 -> add ES via Taylor's (2019) AL-scale mapping, same filter.
  Phase 2 -> replace the random-walk transition with a GP transition (GP-Vol / RAPCF style).

Run this file directly: it simulates returns from a KNOWN random-walk q_t path,
filters them, and reports whether the filter recovers the true quantile and
whether the empirical violation rate matches alpha. Those two checks are the
gate before this ever touches real index returns -- if they fail here, nothing
downstream (GP transition, ES, benchmarks) is worth building on top of it.
"""

import numpy as np


def al_log_density(r, q, sigma, alpha):
    """Log-density of the asymmetric Laplace observation model.

    Mode of this density is q -- the VaR quantile at level alpha. Vectorised
    over particles: r is a scalar (today's return), q is an array of
    per-particle quantile states.
    """
    u = r - q
    indicator = (r < q).astype(float)
    return (
        np.log(alpha * (1 - alpha) / sigma)
        - u * (alpha - indicator) / sigma
    )


def systematic_resample(weights, rng):
    """Standard systematic resampling. Returns indices into the particle array."""
    n = len(weights)
    positions = (rng.uniform() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against float drift
    return np.searchsorted(cumulative, positions)


def particle_filter_var(returns, alpha, sigma, sigma_eta, n_particles=1000, seed=0):
    """Bootstrap particle filter over a latent VaR state q_t, random-walk transition.

    returns    : array of daily returns, r_1 ... r_T
    alpha      : quantile level (e.g. 0.05 for 5% VaR)
    sigma      : AL scale (fixed here -- becomes state-dependent in Phase 1 for ES)
    sigma_eta  : std dev of the transition innovation
    Returns the filtered VaR estimate at each t (weighted particle mean) and the ESS trace.
    """
    rng = np.random.default_rng(seed)
    T = len(returns)

    particles = np.full(n_particles, np.quantile(returns[:20], alpha))
    weights = np.full(n_particles, 1.0 / n_particles)

    var_estimates = np.zeros(T)
    ess_trace = np.zeros(T)

    for t in range(T):
        # 1. propagate: q_t^(i) = q_{t-1}^(i) + eta_t^(i)          [your GP-Vol Eq 11]
        particles = particles + rng.normal(0.0, sigma_eta, n_particles)

        # PREDICTIVE estimate: propagated particles, yesterday's weights,
        # before today's return touches anything
        var_estimates[t] = np.sum(weights * particles)

        # 2. score: weight by the AL observation density at today's return  [Eq 12]
        log_w = al_log_density(returns[t], particles, sigma, alpha)
        log_w -= log_w.max()  # numerical stability
        w = np.exp(log_w) * weights
        weights = w / w.sum()  # [Eq 13, normalize]
        ess = 1.0 / np.sum(weights ** 2)
        ess_trace[t] = ess

        # 3. resample when the effective sample size collapses
        if ess < n_particles / 2:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)

    return var_estimates, ess_trace


def simulate_al_process(T, alpha, sigma, sigma_eta, q0=0.0, seed=1):
    """Generate synthetic returns with a KNOWN random-walk quantile path.

    Sanity-check data: if the filter can't recover q_t here, where you know
    the answer, it has no chance on messy real returns.
    """
    rng = np.random.default_rng(seed)
    true_q = np.zeros(T)
    returns = np.zeros(T)
    q = q0
    for t in range(T):
        q = q + rng.normal(0.0, sigma_eta)
        true_q[t] = q
        # inverse-CDF sampling from the AL density
        u = rng.uniform()
        if u < alpha:
            r = q + sigma / (1 - alpha) * np.log(u / alpha)
        else:
            r = q - sigma / alpha * np.log((1 - u) / (1 - alpha))
        returns[t] = r
    return returns, true_q


if __name__ == "__main__":
    ALPHA = 0.05
    SIGMA = 1.0
    SIGMA_ETA = 0.05
    T = 500
    N_PARTICLES = 1000

    returns, true_q = simulate_al_process(T, ALPHA, SIGMA, SIGMA_ETA)
    filtered_q, ess = particle_filter_var(returns, ALPHA, SIGMA, SIGMA_ETA, n_particles=N_PARTICLES)

    rmse = np.sqrt(np.mean((filtered_q - true_q) ** 2))
    violations = np.mean(returns < filtered_q)

    print(f"RMSE of filtered VaR vs true q_t : {rmse:.4f}")
    print(f"Target violation rate (alpha)    : {ALPHA:.3f}")
    print(f"Empirical violation rate         : {violations:.3f}")
    print(f"Mean effective sample size       : {ess.mean():.1f} / {N_PARTICLES}")