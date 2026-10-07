"""
Benchmark 2: Taylor (2019) ES-CAViaR (SAV quantile recursion, multiplicative ES).

    q_t  = b0 + b1*q_{t-1} + b2*|r_{t-1}|         (symmetric absolute value CAViaR)
    ES_t = (1 + exp(g)) * q_t                     (Taylor's multiplicative factor)

Estimated by maximizing the AL log-likelihood on the first n_fit days
(Taylor's own estimator; the state-dependent AL scale is safe in plain
optimization -- no competing particles to subsidize -- unlike in filtering,
which is exactly the Finding-3 distinction).
Protocol matched to v2: fit on first 1000 days, forward recursion after,
all forecasts one-step-ahead by construction.
"""
import numpy as np
from scipy.optimize import minimize

from gp_fz_phase1 import fz0_loss


def caviar_path(r, b0, b1, b2, q0):
    T = len(r)
    q = np.empty(T)
    q[0] = q0
    for t in range(1, T):
        q[t] = b0 + b1 * q[t-1] + b2 * abs(r[t-1])
    return q


def neg_al_loglik(params, r, alpha, q0):
    b0, b1, b2, g = params
    if not (0.0 <= b1 <= 0.999):
        return 1e10
    q = caviar_path(r, b0, b1, b2, q0)
    es = (1.0 + np.exp(g)) * q
    if np.any(q >= 0) or np.any(es >= q):
        return 1e10
    hit = (r <= q).astype(float)
    ll = np.log((1.0 - alpha) / (-es)) + (r - q) * (alpha - hit) / (alpha * es)
    val = -np.sum(ll)
    return val if np.isfinite(val) else 1e10


def fit_es_caviar(r_fit, alpha, seed=0):
    rng = np.random.default_rng(seed)
    q0 = np.quantile(r_fit[:100], alpha)
    best, best_val = None, np.inf
    for _ in range(30):                       # random restarts; CAViaR surfaces are lumpy
        x0 = np.array([
            rng.uniform(-0.10, 0.0),          # b0
            rng.uniform(0.75, 0.97),          # b1
            rng.uniform(-0.40, -0.02),        # b2  (negative: big |r| pushes q down)
            rng.uniform(-1.5, 0.0),           # g
        ])
        res = minimize(neg_al_loglik, x0, args=(r_fit, alpha, q0),
                       method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return best, q0


def run_es_caviar(returns, alpha=0.05, n_fit=1000, seed=0):
    r = np.asarray(returns, float)
    params, q0 = fit_es_caviar(r[:n_fit], alpha, seed)
    b0, b1, b2, g = params
    q = caviar_path(r, b0, b1, b2, q0)        # recursion uses only r_{t-1}: honest
    es = (1.0 + np.exp(g)) * q
    r_oos, v_oos, e_oos = r[n_fit:], q[n_fit:], es[n_fit:]
    viol = (r_oos < v_oos)
    return {
        "viol_rate": viol.mean(),
        "fz0": np.mean([fz0_loss(rr, vv, ee, alpha) for rr, vv, ee in zip(r_oos, v_oos, e_oos)]),
        "mean_ret_on_viol": r_oos[viol].mean() if viol.any() else np.nan,
        "mean_es_on_viol": e_oos[viol].mean() if viol.any() else np.nan,
        "p_viol_given_viol": (viol[1:] & viol[:-1]).sum() / max(viol[:-1].sum(), 1),
        "params": {"b0": b0, "b1": b1, "b2": b2, "es_mult": 1.0 + np.exp(g)},
        "var_line": v_oos, "es_line": e_oos, "viol": viol,
    }


if __name__ == "__main__":
    from gp_fz_hybrid import simulate_garch_t
    r_syn, true_var, true_es = simulate_garch_t(3000, 0.05, seed=7)
    out = run_es_caviar(r_syn, alpha=0.05, n_fit=1000)
    tv, te = true_var[1000:], true_es[1000:]
    print("=== GATE (synthetic GARCH-t, truth known) ===")
    print(f"viol rate : {out['viol_rate']:.3f}  (target 0.050)")
    print(f"rmse VaR  : {np.sqrt(np.mean((out['var_line']-tv)**2)):.4f}")
    print(f"rmse ES   : {np.sqrt(np.mean((out['es_line']-te)**2)):.4f}")
    print(f"FZ0       : {out['fz0']:.4f}")
    print(f"params    : {out['params']}")
