"""Causal full-calendar paths with saved solver receipts and failure flags."""
import numpy as np
from .optimizer_core import fit, FitFailure, fast_gas_path, legacy


def specialist_paths(r, model, alpha=None):
    r = np.asarray(r, dtype=np.float64); T = len(r)
    dates = legacy.refit_schedule(T)
    if not dates: raise ValueError("At least 751 observations required")
    paths = [np.empty(T), np.empty(T)]
    failures = np.zeros(T, dtype=bool); receipts = []; th = None; physical = None; historical_th = None
    for k, t0 in enumerate(dates):
        lo = 0 if k == 0 else t0
        hi = dates[k+1] if k+1 < len(dates) else T
        h0 = float(np.mean(r[:t0]**2))
        failed = False
        # Preserve the complete original causal optimizer trajectory as an
        # additional search path. A repaired warm start alone can converge to a
        # worse basin than an earlier nonconverged historical trajectory.
        historical_th,_,historical=fit(r[:t0],model,alpha,historical_th,repaired=False)
        try:
            th, physical, receipt = fit(r[:t0], model, alpha, th,additional_starts=[historical_th])
            if receipt["objective"]>historical["objective"]+1e-7:
                receipt.update(accepted=False,reason="worse_than_causal_historical_trajectory")
                raise FitFailure(receipt)
        except FitFailure as exc:
            receipt = exc.receipt; failed = True
        receipt.update(historical_replay=historical,historical_replay_objective=historical["objective"])
        receipt.update(origin=t0, t=t0, forecast_start=lo, forecast_stop=hi, carried_parameters=failed and physical is not None,
                       initial_hist250=failed and physical is None)
        if physical is None:
            raise FitFailure(receipt)
        elif model == "garch_t":
            w,a,b,nu = physical
            # Accepted physical omega is retained if refitting fails; the
            # historical origin-specific initialization is still deterministic.
            h = legacy.garch_variance(r[:hi], w,a,b,h0)
            paths[0][lo:hi] = np.maximum(h[lo:hi],1e-6)
            paths[1][lo:hi] = nu
            receipt.update(omega=w, alpha=a, beta=b, nu=nu, nll=receipt.get("objective"))
        else:
            v,e = fast_gas_path(r[:hi], *physical, alpha)
            paths[0][lo:hi], paths[1][lo:hi] = v[lo:hi],e[lo:hi]
            a,b,beta,gamma = physical
            receipt.update(a=a,b=b,beta=beta,gamma=gamma,fz0=receipt.get("objective"))
        failures[lo:hi] = failed
        receipts.append(receipt)
    if not np.isfinite(paths).all(): raise FloatingPointError("Invalid specialist forecast path")
    return *paths, receipts, failures
