"""Versioned convergence repair of the historical tail specialists.

Model equations, training observations, starts and objectives are historical.
Only solver continuations and acceptance change. No future outcomes enter fits.
"""
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from numba import njit
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
SIBLING = ROOT.parent / "calibeat-var-es"
sys.path.insert(0, str(SIBLING))
from calibeat import market as legacy

# No fastmath: preserves the recurrence and float64 arithmetic conventions.
fast_gas_path = njit(cache=False)(legacy.gas_path)
POLICY = dict(version="tail_optimizer_repair_v2", gas_initial_maxfev=400,
              continuation_maxfev=4000, rescue_maxfev=20000,
              gas_xatol=1e-4, gas_fatol=1e-8, garch_gtol=1e-6,
              objective_slack=1e-7, failure="carry last accepted physical parameters; initial failure fails complete cell and coverage gate; never remove dates")


class FitFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("No converged, finite, feasible fit meets the objective gate")


def garch_objective(r):
    h0 = float(np.mean(r ** 2))
    def loss(th):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w, a, b, nu = legacy._garch_unpack(th, h0)
            h = legacy.garch_variance(r, w, a, b, h0)
            if not np.isfinite([w, a, b, nu]).all() or not np.all(h > 0) or nu > 300:
                return 1e6
            val = -float(np.mean(legacy.t_loglik(r[1:], h[1:], nu)))
        return val if np.isfinite(val) else 1e6
    return loss


def gas_objective(r, alpha):
    def loss(th):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            a, b, beta, gamma = legacy._gas_unpack(th)
            if not np.isfinite([a, b, beta, gamma]).all() or abs(gamma) > 1.0 / alpha:
                return 1e6
            v, e = fast_gas_path(r, a, b, beta, gamma, alpha)
            val = float(np.mean(legacy.fz0(r[50:], v[50:], e[50:], alpha)))
        return val if np.isfinite(val) else 1e6
    return loss


def starts_for(r, model, alpha=None, x0=None):
    warm = [np.asarray(x0, float)] if x0 is not None else []
    if model == "garch_t":
        return warm + [np.array([np.log(.05), np.log(.08/.52), np.log(.90/.0195), np.log(4.)]),
                       np.array([np.log(.02), np.log(.05/.55), np.log(.94/.0095), np.log(2.)])]
    q, es = legacy.empirical_var_es(r, alpha)
    base = [np.log(-q), np.log(max(q-es, 1e-3))]
    return warm + [np.array(base+[3.9, .001/alpha]), np.array(base+[2.2, .0003/alpha])]


def _attempt(loss, start, method, options, start_id, stage):
    clock = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = minimize(loss, start, method=method, options=options)
    rec = dict(start_id=start_id, stage=stage, method=method, options=options,
               success=bool(res.success), status=int(res.status), message=str(res.message),
               objective=float(res.fun), theta=np.asarray(res.x).tolist(),
               nfev=int(res.nfev), nit=int(res.nit), seconds=time.perf_counter()-clock,
               warnings=sorted(set(str(w.message) for w in caught)))
    return res, rec


def fit(r, model, alpha=None, x0=None, *, repaired=True, additional_starts=()):
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or len(r) < 51 or not np.isfinite(r).all() or np.mean(r*r) <= 0:
        raise ValueError("Finite nondegenerate training returns required")
    if model not in ("garch_t", "gas_fz0"):
        raise ValueError(model)
    loss = garch_objective(r) if model == "garch_t" else gas_objective(r, alpha)
    receipt = dict(model=model, alpha=alpha, training_rows=len(r), attempts=[])
    initial, accepted = [], []
    for j, start in enumerate(starts_for(r, model, alpha, x0)+list(additional_starts)):
        method = "BFGS" if model == "garch_t" else "Nelder-Mead"
        opts = dict(maxiter=300, gtol=1e-6) if model == "garch_t" else dict(maxfev=400, xatol=1e-4, fatol=1e-8)
        res, rec = _attempt(loss, start, method, opts, j, "historical_budget")
        receipt["attempts"].append(rec); initial.append(res)
        if res.success and np.isfinite(res.fun) and res.fun < 1e6 and np.isfinite(res.x).all():
            accepted.append((res, len(receipt["attempts"])-1))
        if repaired and not res.success and np.isfinite(res.fun) and res.fun < 1e6:
            for budget in (POLICY["continuation_maxfev"], POLICY["rescue_maxfev"]):
                res, rec = _attempt(loss, res.x, "Nelder-Mead",
                                    dict(maxfev=budget, xatol=1e-4, fatol=1e-8), j, "continuation")
                receipt["attempts"].append(rec)
                if res.success and np.isfinite(res.fun) and res.fun < 1e6 and np.isfinite(res.x).all():
                    accepted.append((res, len(receipt["attempts"])-1)); break
    old = min(initial, key=lambda x:x.fun)
    receipt.update(initial_best_objective=float(old.fun), initial_best_converged=bool(old.success))
    if not repaired:
        result, idx = old, next(i for i,a in enumerate(receipt["attempts"]) if a["theta"] == old.x.tolist())
    else:
        if not accepted:
            receipt.update(accepted=False, reason="no_converged_candidate")
            raise FitFailure(receipt)
        result, idx = min(accepted, key=lambda x:x[0].fun)
        if result.fun > old.fun + POLICY["objective_slack"]:
            receipt.update(accepted=False, reason="converged_candidate_worse_than_initial_best")
            raise FitFailure(receipt)
    params = legacy._garch_unpack(result.x, float(np.mean(r*r))) if model == "garch_t" else legacy._gas_unpack(result.x)
    if model == "garch_t":
        probe = legacy.garch_variance(np.append(r, 0.), *params[:3], float(np.mean(r*r)))[-1]
        feasible = np.isfinite(probe) and probe > 0 and params[3] > 2
    else:
        v, e = fast_gas_path(np.append(r, 0.), *params, alpha)
        probe = [v[-1], e[-1]]
        feasible = np.isfinite(probe).all() and e[-1] < v[-1] < 0
    receipt.update(accepted=bool(feasible), selected_attempt=idx, selected_converged=bool(result.success),
                   objective=float(result.fun), parameters=list(map(float, params)), next_forecast=np.asarray(probe).tolist(),
                   objective_improvement=float(old.fun-result.fun))
    if not feasible:
        raise FitFailure(receipt)
    return np.asarray(result.x), params, receipt
