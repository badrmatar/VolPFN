"""Frozen daily-return Tier A econometric forecasts; no scoring or data loading.

At forecast index t, fitting and filtering see only r[:t]. Parameters are fitted
on exactly the trailing 1000 (or prespecified 2000) observations every 20 days.
Between refits arch's public volatility API filters each newly observed return.
The fitted constant mean preserves the existing comparator convention. A
variance-standardized Student-t implies E[r**2] = Var(r) + fitted_mean**2.

There is one optimizer attempt per scheduled refit, with fixed settings. A failed
refit carries the previous accepted parameters and continues daily filtering.
An initial failure makes the model unavailable; it does not trigger later rescue
fits. Every attempt is retained, including exceptions and validation failures.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


MODELS = ("garch_t", "gjr_garch_t", "egarch_t")
HISTORY_WINDOWS = (1000, 2000)
REFIT_EVERY = 20
MAXITER = 1000
CONSTRAINT_TOLERANCE = 1e-8
OPTIMIZER_POLICY = {
    "attempts_per_refit": 1,
    "starting_values": "arch_default",
    "maxiter": MAXITER,
    "selection": "accept sole candidate only when all checks pass",
    "failed_refit": "carry last accepted parameters and continue daily filtering",
    "initial_failure": "model unavailable; stop without rescue fits",
}


def arch_spec(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown econometric model {model_name!r}; expected {MODELS}")
    return {"mean": "Constant", "vol": "EGARCH" if model_name == "egarch_t" else "GARCH",
            "p": 1, "o": 0 if model_name == "garch_t" else 1, "q": 1,
            "dist": "t", "rescale": False}


@dataclass
class _State:
    volatility: object
    volatility_parameters: np.ndarray
    parameters: dict
    backcast: float
    training_bounds: np.ndarray
    history_start: int
    fit_origin: int

    def forecast(self, observed_returns):
        """Public arch recursion with frozen fit initialization/bounds.

        New observations use unbounded variance bounds, matching arch's own
        one-step forecast convention. Re-estimating bounds from an appended
        block would allow later returns to alter earlier filtered variances.
        """
        residuals = np.asarray(observed_returns, dtype=float) - self.parameters["mu"]
        extra = len(residuals) - len(self.training_bounds)
        if extra < 0:
            raise ValueError("Observed history is shorter than fitted history")
        bounds = np.concatenate((self.training_bounds,
                                 np.tile([0.0, np.inf], (extra, 1))))
        forecast = self.volatility.forecast(
            self.volatility_parameters, residuals, self.backcast, bounds,
            start=len(residuals) - 1, horizon=1, method="analytic")
        value = float(forecast.forecasts[-1, 0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("nonpositive_or_nonfinite_forecast")
        return value


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _parameter_rejections(model, parameters, history):
    """Check the library's own named parameter, positivity and stationarity rules."""
    volatility_names = model.volatility.parameter_names()
    distribution_names = model.distribution.parameter_names()
    names = model.parameter_names() + volatility_names + distribution_names
    if set(parameters) != set(names):
        return ["unexpected_parameter_names"]
    if not np.isfinite([parameters[name] for name in names]).all():
        return ["nonfinite_parameters"]
    rejected = []
    if parameters["nu"] <= 2:
        rejected.append("student_t_has_no_finite_variance")
    for label, process, process_names in (
            ("volatility", model.volatility, volatility_names),
            ("distribution", model.distribution, distribution_names)):
        values = np.array([parameters[name] for name in process_names])
        a, b = process.constraints()
        if np.any(a @ values - b < -CONSTRAINT_TOLERANCE):
            rejected.append(f"{label}_parameter_constraints")
        bounds = np.asarray(process.bounds(history), dtype=float)
        if (np.any(values < bounds[:, 0] - CONSTRAINT_TOLERANCE)
                or np.any(values > bounds[:, 1] + CONSTRAINT_TOLERANCE)):
            rejected.append(f"{label}_parameter_bounds")
    return rejected


def _fit_one(history, model_name, history_start, origin):
    """Exactly one predetermined optimizer attempt; return state and full audit."""
    from arch import arch_model

    attempt = {"attempt_id": "arch_default", "maxiter": MAXITER,
               "starting_values": None, "accepted": False, "rejection_reasons": []}
    state = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            if not np.isfinite(np.var(history)) or np.var(history) <= 0:
                raise ValueError("Training returns need positive finite variance")
            model = arch_model(history, **arch_spec(model_name))
            fit = model.fit(disp="off", show_warning=False, update_freq=0,
                            starting_values=None, options={"maxiter": MAXITER})
            flag = int(fit.convergence_flag)
            attempt.update({"convergence_flag": flag,
                            "optimizer_message": str(fit.optimization_result.message),
                            "optimizer_iterations": int(fit.optimization_result.get("nit", -1))})
            if flag != 0:
                attempt["rejection_reasons"].append("optimizer_did_not_converge")
            # Retain rejected candidates too; they are never installed as state.
            parameters = {str(key): float(value) for key, value in fit.params.items()}
            attempt["parameters"] = {key: _finite_or_none(value)
                                     for key, value in parameters.items()}
            attempt["loglikelihood"] = _finite_or_none(fit.loglikelihood)
            if attempt["loglikelihood"] is None:
                attempt["rejection_reasons"].append("nonfinite_training_likelihood")
            initial_residuals = model.resids(model.starting_values())
            attempt["rejection_reasons"].extend(
                _parameter_rejections(model, parameters, initial_residuals))
            historical_h = np.asarray(fit.conditional_volatility, dtype=float) ** 2
            if (historical_h.shape != history.shape or not np.isfinite(historical_h).all()
                    or np.any(historical_h <= 0)):
                attempt["rejection_reasons"].append("invalid_training_variances")
            if not attempt["rejection_reasons"]:
                # Fit initializes using its starting mean, then filters using
                # the fitted mean. Reconstruct both through public methods;
                # recomputing backcast from fitted residuals would change state.
                candidate = _State(
                    volatility=model.volatility,
                    volatility_parameters=np.array([parameters[name] for name in
                                                    model.volatility.parameter_names()]),
                    parameters=parameters, backcast=float(model.volatility.backcast(initial_residuals)),
                    training_bounds=model.volatility.variance_bounds(initial_residuals),
                    history_start=history_start, fit_origin=origin)
                attempt["last_training_variance"] = float(historical_h[-1])
                attempt["next_variance"] = candidate.forecast(history)
                attempt["next_second_moment"] = (
                    attempt["next_variance"] + parameters["mu"] ** 2)
                if not np.isfinite(attempt["next_second_moment"]):
                    attempt["next_second_moment"] = None
                    raise ValueError("nonfinite_next_second_moment")
                state = candidate
                attempt["accepted"] = True
        except Exception as exc:
            attempt["rejection_reasons"].append("fit_or_validation_exception")
            attempt["exception"] = f"{type(exc).__name__}: {exc}"
    attempt["warnings"] = [str(item.message) for item in captured]
    return state, attempt


def forecast_econometric(market, model_name, start_date="2015-01-01", *,
                         history_window=1000, refit_every=REFIT_EVERY):
    """Forecast on an aligned daily calendar, retaining unavailable dates as NaN.

    ``market['r']`` must be finite daily returns, preferably percentage units;
    ``market['date']`` must be strictly increasing ISO calendar dates. The input
    is not standardized using future data. Output variances retain input units
    squared. Exactly ``history_window`` observations are required before the
    first origin. No target, loss, realized volatility or future outcome enters
    fitting, optimizer selection, state updates or fallback decisions.

    Status is ``complete`` even when some refits failed but valid old parameters
    carried the full path. Status ``unavailable`` means the caller must retain
    and report the failure, never silently drop affected dates from a comparison.
    """
    import arch

    spec = arch_spec(model_name)
    if (isinstance(history_window, bool) or not isinstance(history_window, (int, np.integer))
            or history_window not in HISTORY_WINDOWS):
        raise ValueError(f"history_window must be one of frozen windows {HISTORY_WINDOWS}")
    if (isinstance(refit_every, bool) or not isinstance(refit_every, (int, np.integer))
            or refit_every != REFIT_EVERY):
        raise ValueError("Tier A fixes refit_every=20")
    r = np.asarray(market["r"], dtype=float)
    dates_raw = np.asarray(market["date"])
    if r.ndim != 1 or not len(r) or dates_raw.shape != r.shape or not np.isfinite(r).all():
        raise ValueError("Daily returns and dates must be finite, nonempty, aligned vectors")
    try:
        days = dates_raw.astype("datetime64[D]")
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must be valid daily calendar dates") from exc
    if np.isnat(days).any() or np.any(days[1:] <= days[:-1]):
        raise ValueError("Dates must be valid and strictly increasing")
    dates = days.astype("U10")
    start = int(np.searchsorted(days, np.datetime64(start_date, "D")))
    if start >= len(r):
        raise ValueError("No daily observations on or after start_date")
    if start < history_window:
        raise ValueError(f"First origin requires {history_window} preceding returns")
    n = len(r)
    result = {key: np.full(n, np.nan) for key in
              ("h", "mean", "variance", "second_moment", "point_variance", "nu")}
    result.update({"date": dates.copy(), "fit_origin": np.full(n, "", dtype="U10"),
                   "forecast_status": np.full(n, "before_start", dtype="U32"),
                   "attempts": [], "fits": [], "failures": [], "status": "complete",
                   "metadata": {"model": model_name, "arch_version": arch.__version__,
                                "specification": spec, "history_window": int(history_window),
                                "refit_every": int(refit_every), "evaluation_start": str(dates[start]),
                                "return_units": market.get("return_units", "input_return_units"),
                                "density_mean": "fitted_constant", "variance_equals_second_moment": False,
                                "optimizer_policy": dict(OPTIMIZER_POLICY),
                                "parameter_constraint_tolerance": CONSTRAINT_TOLERANCE,
                                "filter": "arch public volatility.forecast; training-only backcast and bounds",
                                "daily_information_cutoff": "strictly before forecast date"}})
    state = None
    carrying_failed_refit = False
    for t in range(start, n):
        if (t - start) % refit_every == 0:
            history_start = t - history_window
            candidate, attempt = _fit_one(r[history_start:t].copy(), model_name, history_start, t)
            attempt.update({"origin": str(dates[t]), "origin_index": t,
                            "history_start": str(dates[history_start]),
                            "history_end": str(dates[t - 1]), "history_n": int(history_window),
                            "history_start_index": history_start, "history_end_index": t - 1})
            result["attempts"].append(attempt)
            if candidate is not None:
                state = candidate
                carrying_failed_refit = False
                attempt["action"] = "install_fitted_parameters"
                result["fits"].append(attempt.copy())
            elif state is None:
                attempt["action"] = "model_unavailable_initial_fit_failed"
                result["failures"].append(attempt.copy())
                result["status"] = "unavailable"
                result["unavailable_reason"] = "initial_fit_failed"
                result["forecast_status"][t:] = "unavailable_initial_fit"
                break
            else:
                carrying_failed_refit = True
                attempt["action"] = "carry_previous_parameters"
                attempt["carried_fit_origin"] = str(dates[state.fit_origin])
                result["failures"].append(attempt.copy())
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                h = state.forecast(r[state.history_start:t])
                second_moment = h + state.parameters["mu"] ** 2
                if not np.isfinite(second_moment):
                    raise ValueError("nonfinite_second_moment")
            if captured:
                result.setdefault("filter_warnings", []).append(
                    {"origin": str(dates[t]), "warnings": [str(item.message) for item in captured]})
        except Exception as exc:
            result["status"] = "unavailable"
            result["unavailable_reason"] = "daily_filter_failed"
            result["forecast_status"][t:] = "unavailable_filter"
            result["failures"].append({"origin": str(dates[t]), "origin_index": t,
                                       "action": "model_unavailable_daily_filter_failed",
                                       "fit_origin": str(dates[state.fit_origin]),
                                       "exception": f"{type(exc).__name__}: {exc}"})
            break
        for key in ("h", "variance", "point_variance"):
            result[key][t] = h
        result["mean"][t] = state.parameters["mu"]
        result["second_moment"][t] = second_moment
        result["nu"][t] = state.parameters["nu"]
        result["fit_origin"][t] = dates[state.fit_origin]
        result["forecast_status"][t] = "carried_parameters" if carrying_failed_refit else "ok"
    result["metadata"]["refit_attempt_count"] = len(result["attempts"])
    result["metadata"]["refit_failure_count"] = sum(not item["accepted"] for item in result["attempts"])
    return result
