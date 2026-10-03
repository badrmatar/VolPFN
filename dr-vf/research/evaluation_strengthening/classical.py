"""Causal adapters for the archived and fixed expanded classical benchmarks.

All means are in percentage-return units and variances/second moments in their
squares. The log scores are densities in percentage-return units. The original
GARCH initialization, optimizer defaults, refit schedule and between-fit recursion
are preserved; a failed fit raises instead of silently removing observations.

UniVol is a mixture of centered, variance-standardized Student-t components.
Its density variance is the *arithmetic* mixture variance (``h_mean``); ``h`` is
only its geometric point forecast. Both are returned to keep this distinction
auditable. Its first-100-return initialization and the first-10-return EWMA
initialization are archived conventions: evaluation must start after this prefix.
"""

from __future__ import annotations

import warnings

import numpy as np

from research.prior_adaptation.model import log_law


CLASSICAL_MODELS = ("garch_t", "univol", "ewma_t8", "gjr_garch_t", "egarch_t")
ARCH_MODELS = ("garch_t", "gjr_garch_t", "egarch_t")
EGARCH_MAXITER = 2000
EGARCH_START_GRID = ((0.9, 0.0), (0.9, -0.1), (0.98, 0.0), (0.98, -0.1))


def _market_arrays(market, start_date):
    r = np.asarray(market["r"], dtype=float)
    z = np.asarray(market["z"], dtype=float)
    date = np.asarray(market["date"]).astype("U10")
    scale = float(market["scale"])
    if r.ndim != 1 or not len(r) or z.shape != r.shape or date.shape != r.shape:
        raise ValueError("market date, r and z must be nonempty aligned one-dimensional arrays")
    if not np.isfinite(r).all() or not np.isfinite(z).all():
        raise ValueError("market returns and normalized returns must be finite")
    if not np.isfinite(scale) or scale <= 0 or not np.allclose(z, r / scale, rtol=1e-12, atol=1e-12):
        raise ValueError("market scale must be positive and z must equal r / scale")
    if np.any(date[1:] <= date[:-1]):
        raise ValueError("market dates must be strictly increasing")
    origins = np.flatnonzero(date >= str(start_date))
    if not len(origins):
        raise ValueError("No observations at or after start_date")
    start = int(origins[0])
    if start < 100:
        raise ValueError("Evaluation must start after the first 100 initialization returns")
    return r, z, date, scale, start


def _empty_forecast(n):
    result = {key: np.full(n, np.nan) for key in
              ("mean", "variance", "second_moment", "point_variance", "logscore", "nu")}
    result["fit_origin"] = np.full(n, "", dtype="U10")
    result["fits"] = []
    return result


def _arch_spec(model_name):
    return {"mean": "Constant", "vol": "EGARCH" if model_name == "egarch_t" else "GARCH",
            "p": 1, "o": 0 if model_name == "garch_t" else 1, "q": 1,
            "dist": "t", "rescale": False}


def _next_variance(model_name, params, residual, previous_variance):
    """Observed-return one-step recursion; no held-out return enters a forecast.

    EGARCH uses arch's normal centering constant sqrt(2/pi), including with
    Student-t innovations. These fixed (1,1,1) specifications follow arch 7.2:
    https://arch.readthedocs.io/en/stable/univariate/generated/arch.univariate.EGARCH.html
    https://arch.readthedocs.io/en/stable/univariate/generated/arch.univariate.GARCH.html
    """
    if model_name == "egarch_t":
        standardized = residual / np.sqrt(previous_variance)
        log_h = (params["omega"] + params["alpha[1]"] * (abs(standardized) - np.sqrt(2.0 / np.pi))
                 + params["gamma[1]"] * standardized + params["beta[1]"] * np.log(previous_variance))
        # arch applies a machine-overflow cap, even for unconstrained forecasts.
        log_h = min(float(log_h), float(np.log(np.finfo(float).max) - 0.1))
        return float(np.exp(log_h))
    # Keep the archived GARCH operation order verbatim.
    h = float(params["omega"] + params["alpha[1]"] * residual ** 2
              + params["beta[1]"] * previous_variance)
    if model_name == "gjr_garch_t":
        h += float(params["gamma[1]"] * residual ** 2 * (residual < 0))
    return h


def _fit_egarch(history, label):
    """Fixed training-only multistart policy, identical at every EGARCH origin.

    arch's convergence flag alone can label a degenerate fit successful. Select
    the largest training likelihood among converged fits with finite parameters,
    finite positive historical variances and a finite positive next variance.
    That next forecast uses only the final *training* return. No later realized
    return, density score or variance loss participates in fitting/selection.
    """
    from arch import arch_model

    variance = float(np.var(history))
    if not np.isfinite(variance) or variance <= 0:
        raise RuntimeError(f"{label}: EGARCH requires positive finite training variance")
    starts = [("arch_default", None)]
    for beta, gamma in EGARCH_START_GRID:
        values = np.array([np.mean(history), (1.0 - beta) * np.log(variance),
                           0.1, gamma, beta, 8.0], dtype=float)
        starts.append((f"beta_{beta:g}_gamma_{gamma:g}", values))
    attempts = []
    eligible = []
    for identifier, starting_values in starts:
        attempt = {"id": identifier, "starting_values": None if starting_values is None else starting_values.tolist(),
                   "maxiter": EGARCH_MAXITER, "rejection_reasons": []}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                fit = arch_model(history, **_arch_spec("egarch_t")).fit(
                    disp="off", show_warning=False, starting_values=starting_values,
                    options={"maxiter": EGARCH_MAXITER})
                p = fit.params
                flag = int(fit.convergence_flag)
                ll = float(fit.loglikelihood)
                historical_h = np.asarray(fit.conditional_volatility, dtype=float) ** 2
                parameter_values = p.to_numpy(dtype=float)
                attempt.update({"flag": flag, "optimizer_message": str(fit.optimization_result.message),
                                "optimizer_iterations": int(fit.optimization_result.get("nit", -1)),
                                "loglikelihood": ll if np.isfinite(ll) else None,
                                "parameters": {str(k): float(v) if np.isfinite(v) else None for k, v in p.items()}})
                if flag != 0:
                    attempt["rejection_reasons"].append("optimizer_did_not_converge")
                if not np.isfinite(ll):
                    attempt["rejection_reasons"].append("nonfinite_training_likelihood")
                if not np.isfinite(parameter_values).all() or float(p["nu"]) <= 2:
                    attempt["rejection_reasons"].append("invalid_parameters")
                state_valid = (historical_h.shape == np.asarray(history).shape
                               and np.isfinite(historical_h).all() and np.all(historical_h > 0))
                if not state_valid:
                    attempt["rejection_reasons"].append("invalid_training_variances")
                if state_valid and np.isfinite(parameter_values).all():
                    next_h = _next_variance("egarch_t", p, history[-1] - p["mu"], float(historical_h[-1]))
                    attempt["last_training_variance"] = float(historical_h[-1])
                    attempt["next_variance"] = float(next_h) if np.isfinite(next_h) else None
                    if not np.isfinite(next_h) or next_h <= 0:
                        attempt["rejection_reasons"].append("invalid_next_variance")
                attempt["eligible"] = not attempt["rejection_reasons"]
                if attempt["eligible"]:
                    eligible.append((ll, fit, identifier))
            except Exception as exc:
                attempt["eligible"] = False
                attempt["rejection_reasons"].append("fit_or_validation_exception")
                attempt["exception"] = repr(exc)
        attempt["warnings"] = [str(item.message) for item in captured]
        attempts.append(attempt)
    if not eligible:
        details = "; ".join(f'{item["id"]}: {",".join(item["rejection_reasons"])}' for item in attempts)
        raise RuntimeError(f"{label}: no valid EGARCH fit across fixed training-only starts; {details}; no dates removed")
    # max preserves the earlier attempt when likelihood values are exactly tied.
    _, best_fit, selected = max(eligible, key=lambda item: item[0])
    return best_fit, {"selected_attempt": selected, "attempts": attempts,
                      "selection": "maximum training loglikelihood among valid converged candidates"}


def forecast_classical(market: dict, model_name: str, start_date="2015-01-01",
                       refit_every=250, history_window=1000) -> dict:
    """Return full-length predictive means, variances, moments and log scores.

    ``start_date`` fixes the first fitted-model origin; later origins are spaced
    by ``refit_every`` observations, using up to ``history_window`` prior returns.
    Each fitted block records its first forecast date in ``fit_origin`` (the fit
    data end on the preceding observation). Unavailable fitted forecasts are NaN.
    Fixed online filters run from series initialization, have empty fit origins
    and no fits. Their pre-initialization-prefix values must not be evaluated.
    No model selection, hyperparameter tuning, or output-file writes occur here.
    EGARCH uses the same fixed five-start optimizer policy at every origin, with
    selection by training likelihood. GARCH/GJR retain the original default fit.
    All failures are explicit and retain the offending origin.
    """
    if model_name not in CLASSICAL_MODELS:
        raise ValueError(f"Unknown classical model {model_name!r}; expected {CLASSICAL_MODELS}")
    for key, value in (("refit_every", refit_every), ("history_window", history_window)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    r, z, date, scale, start = _market_arrays(market, start_date)
    result = _empty_forecast(len(r))
    result["metadata"] = {"model": model_name, "return_units": "percent",
                          "variance_units": "percent_squared", "evaluation_start": str(date[start]),
                          "initialization_returns": 100, "normalization_scale": scale}

    if model_name == "univol":
        from univol import run_series

        filtered = run_series(r, levels=())
        result["variance"] = filtered["h_mean"].copy()
        result["second_moment"] = filtered["h_mean"].copy()
        result["point_variance"] = filtered["h"].copy()
        result["mean"][np.isfinite(filtered["h_mean"])] = 0.0
        result["logscore"] = filtered["logscore"].copy()
        result["metadata"].update({"density_mean": "zero", "variance": "arithmetic_mixture_variance",
                                   "point_variance": "geometric_mixture_variance",
                                   "initialization": "mean square of nonzero first 100 percentage returns"})
        return result

    if model_name == "ewma_t8":
        h = np.empty(len(r))
        h[0] = max(float(np.mean(r[:10] ** 2)), 1e-4)
        for t in range(1, len(r)):
            h[t] = 0.94 * h[t - 1] + (1 - 0.94) * r[t - 1] ** 2
        result["mean"].fill(0.0)
        result["variance"] = h
        result["second_moment"] = h.copy()
        result["point_variance"] = h.copy()
        result["nu"].fill(8.0)
        result["logscore"] = log_law(r, h, result["nu"])
        result["metadata"].update({"density_mean": "zero", "lambda": 0.94,
                                   "initialization_returns": 10,
                                   "initialization": "max(mean square of first 10 percentage returns, 1e-4)"})
        return result

    import arch
    from arch import arch_model

    spec = _arch_spec(model_name)
    result["metadata"].update({"arch_version": arch.__version__, "specification": spec,
                               "history_window": int(history_window), "refit_every": int(refit_every),
                               "density_mean": "fitted_constant",
                               "initialization": "last conditional variance from fitted historical series"})
    if model_name == "egarch_t":
        result["metadata"]["optimizer_policy"] = {
            "attempts": "arch default plus fixed beta/gamma starting-value grid",
            "beta_gamma_grid": [list(pair) for pair in EGARCH_START_GRID],
            "fixed_start_mu": "training sample mean", "fixed_start_omega": "(1-beta)*log(training sample variance)",
            "fixed_start_alpha": 0.1, "fixed_start_nu": 8.0, "maxiter": EGARCH_MAXITER,
            "selection": "largest training loglikelihood among converged candidates with finite parameters and positive finite training/next variances",
            "heldout_selection": False,
        }
    market_name = market.get("name", "market")
    for origin in range(start, len(r), refit_every):
        history_start = max(0, origin - history_window)
        history = z[history_start:origin]
        if len(history) < 2:
            raise RuntimeError(f"{market_name}: {model_name} has insufficient history at {date[origin]}")
        fit_selection = {}
        if model_name == "egarch_t":
            fit, fit_selection = _fit_egarch(history, f"{market_name}: {model_name} at {date[origin]}")
            fit_warnings = next(item["warnings"] for item in fit_selection["attempts"]
                                if item["id"] == fit_selection["selected_attempt"])
        else:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    fit = arch_model(history, **spec).fit(disp="off", show_warning=False)
                except Exception as exc:
                    raise RuntimeError(f"{market_name}: {model_name} fit failed at {date[origin]}: {exc}") from exc
            fit_warnings = [str(item.message) for item in captured]
        flag = int(fit.convergence_flag)
        if flag != 0:
            raise RuntimeError(f"{market_name}: {model_name} failed to converge at {date[origin]} "
                               f"(flag {flag}: {fit.optimization_result.message}); no dates removed")
        p = fit.params
        h = float(fit.conditional_volatility[-1] ** 2)
        if not np.isfinite(p.to_numpy()).all() or float(p["nu"]) <= 2 or not np.isfinite(h) or h <= 0:
            raise RuntimeError(f"{market_name}: {model_name} invalid fitted state at {date[origin]}")
        result["fits"].append({"origin": str(date[origin]), "history_n": len(history),
                               "history_start": str(date[history_start]), "history_end": str(date[origin - 1]),
                               "nu": float(p["nu"]), "converged": True, "flag": flag,
                               "parameters": {str(k): float(v) for k, v in p.items()},
                               "loglikelihood": float(fit.loglikelihood),
                               "optimizer_message": str(fit.optimization_result.message),
                               "optimizer_iterations": int(fit.optimization_result.get("nit", -1)),
                               "warnings": fit_warnings, **fit_selection})
        for t in range(origin, min(origin + refit_every, len(r))):
            h = _next_variance(model_name, p, z[t - 1] - p["mu"], h)
            if not np.isfinite(h) or h <= 0:
                raise RuntimeError(f"{market_name}: {model_name} invalid variance at {date[t]} "
                                   f"from fit {date[origin]}; no dates removed")
            mu = float(p["mu"]) * scale
            variance = h * scale ** 2
            result["mean"][t] = mu
            result["variance"][t] = variance
            result["second_moment"][t] = variance + mu ** 2
            result["point_variance"][t] = variance
            result["nu"][t] = float(p["nu"])
            result["fit_origin"][t] = date[origin]
            result["logscore"][t] = log_law(np.array([z[t] - p["mu"]]), np.array([h]),
                                            np.array([p["nu"]]))[0] - np.log(scale)
    return result
