"""Fixed, causal classical forecasts for the combined descriptive leaderboard.

Inputs are percentage returns and percent-squared realized kernels on the full
market-session calendar. An origin t predicts that session using only rows <t.
The caller supplies the sorted union of calibration and evaluation origins;
its first element anchors every 20-session refit. All intervening sessions are
filtered, including sessions omitted from either scoring/calibration panel.

This is a new fallback policy, not the archived Tier A forecast experiment.
Failed refits carry accepted parameters, or use zero-mean hist250 when none
exist. Invalid daily forecasts carry the last valid (mean, variance, M2) tuple;
without one, they use zero-mean hist250. No forecast date is removed. A carried
parameter block counts as substitution, separately from numerical replacement.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import warnings

import numpy as np

from research.tier_a_confirmation import econometric as tier_a

MODELS = ("hist250", "ewma", "garch_t", "gjr_garch_t", "egarch_t", "aparch_t",
          "compact9", "univol", "har", "loghar")
ARCH_MODELS = (*tier_a.MODELS, "aparch_t")
HISTORY_WINDOW = 1000
REFIT_EVERY = 20
HAR_MIN_ROWS = 50
HAR_FLOOR_RATIO = 1e-4
EWMA_DECAY = 0.94


def _arrays(market, origins, model):
    if model not in MODELS:
        raise ValueError(f"Unknown classical model {model!r}")
    r = np.asarray(market["r"], dtype=np.float64)
    rk = np.asarray(market["rk"], dtype=np.float64)
    date = np.asarray(market["date"]).astype("datetime64[D]")
    origins = np.asarray(origins)
    if r.ndim != 1 or not len(r) or date.shape != r.shape or rk.shape != r.shape:
        raise ValueError("date, r and rk must be nonempty aligned vectors")
    if not np.isfinite(r).all() or not np.isfinite(r * r).all():
        raise ValueError("Returns and their squares must be finite")
    if np.isnat(date).any() or np.any(date[1:] <= date[:-1]):
        raise ValueError("Dates must be valid and strictly increasing")
    if (origins.ndim != 1 or not len(origins) or origins.dtype.kind not in "iu"
            or np.any(origins[1:] <= origins[:-1]) or origins[0] < 250
            or origins[-1] >= len(r)):
        raise ValueError("Origins must be sorted unique integer indices with >=250 prior returns")
    if model in (*ARCH_MODELS, "har", "loghar") and origins[0] < HISTORY_WINDOW:
        raise ValueError("Fitted models require 1000 prior market sessions")
    return r, rk, date.astype("U10"), origins.astype(np.int64)


def _historical(r, t, *, zero_mean=False):
    history = r[t - 250:t]
    mu = 0.0 if zero_mean else float(history.mean())
    variance = float(np.var(history, ddof=1))
    return mu, variance, variance + mu * mu


def _valid_tuple(values):
    return bool(np.isfinite(values).all() and values[1] > 0 and values[2] > 0)


def _fit_one(history, model, history_start, origin):
    """Reuse Tier A for its three models; use identical safe fitting for APARCH."""
    if model in tier_a.MODELS:
        return tier_a._fit_one(history, model, history_start, origin)
    if model != "aparch_t":
        raise ValueError(model)
    from arch import arch_model

    attempt = {"attempt_id": "arch_default", "maxiter": tier_a.MAXITER,
               "starting_values": None, "accepted": False, "rejection_reasons": []}
    state = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            if not np.isfinite(np.var(history)) or np.var(history) <= 0:
                raise ValueError("Training returns need positive finite variance")
            fitted_model = arch_model(history, mean="Constant", vol="APARCH", p=1, o=1,
                                      q=1, dist="t", rescale=False)
            fit = fitted_model.fit(disp="off", show_warning=False, update_freq=0,
                                   starting_values=None, options={"maxiter": tier_a.MAXITER})
            flag = int(fit.convergence_flag)
            attempt.update(convergence_flag=flag,
                           optimizer_message=str(fit.optimization_result.message),
                           optimizer_iterations=int(fit.optimization_result.get("nit", -1)))
            if flag:
                attempt["rejection_reasons"].append("optimizer_did_not_converge")
            parameters = {str(k): float(v) for k, v in fit.params.items()}
            attempt["parameters"] = {k: tier_a._finite_or_none(v) for k, v in parameters.items()}
            attempt["loglikelihood"] = tier_a._finite_or_none(fit.loglikelihood)
            if attempt["loglikelihood"] is None:
                attempt["rejection_reasons"].append("nonfinite_training_likelihood")
            initial = fitted_model.resids(fitted_model.starting_values())
            attempt["rejection_reasons"].extend(
                tier_a._parameter_rejections(fitted_model, parameters, initial))
            historical_h = np.asarray(fit.conditional_volatility, dtype=float) ** 2
            if (historical_h.shape != history.shape or not np.isfinite(historical_h).all()
                    or np.any(historical_h <= 0)):
                attempt["rejection_reasons"].append("invalid_training_variances")
            if not attempt["rejection_reasons"]:
                candidate = tier_a._State(
                    volatility=fitted_model.volatility,
                    volatility_parameters=np.array([parameters[k] for k in
                                                     fitted_model.volatility.parameter_names()]),
                    parameters=parameters, backcast=float(fitted_model.volatility.backcast(initial)),
                    training_bounds=fitted_model.volatility.variance_bounds(initial),
                    history_start=history_start, fit_origin=origin)
                attempt["last_training_variance"] = float(historical_h[-1])
                attempt["next_variance"] = candidate.forecast(history)
                moment = attempt["next_variance"] + parameters["mu"] ** 2
                attempt["next_second_moment"] = tier_a._finite_or_none(moment)
                if attempt["next_second_moment"] is None:
                    raise ValueError("nonfinite_next_second_moment")
                state = candidate
                attempt["accepted"] = True
        except Exception as exc:
            attempt["rejection_reasons"].append("fit_or_validation_exception")
            attempt["exception"] = f"{type(exc).__name__}: {exc}"
    attempt["warnings"] = [str(item.message) for item in captured]
    return state, attempt


def _har_features(rk, log=False):
    """Missing/nonpositive RK is forward filled causally, never backfilled.

    Row t contains [1, RK[t-1], mean(RK[t-5:t]), mean(RK[t-22:t])].
    log-HAR logs the three arithmetic means, not their constituent observations.
    Observed positive RK, not the filled series, remains the fitting target.
    """
    filled = np.full(len(rk), np.nan)
    previous = np.nan
    for t, value in enumerate(rk):
        if np.isfinite(value) and value > 0:
            previous = float(value)
        filled[t] = previous
    x = np.full((len(rk), 4), np.nan)
    for t in range(22, len(rk)):
        lag = np.array([filled[t - 1], filled[t - 5:t].mean(), filled[t - 22:t].mean()])
        if np.isfinite(lag).all():
            x[t] = np.r_[1., np.log(lag) if log else lag]
    return x


@dataclass
class _HarState:
    beta: np.ndarray
    residual_variance: float
    floor: float
    log: bool
    fit_origin: int

    def forecast(self, x):
        if not np.isfinite(x).all():
            raise ValueError("missing_causal_RK_features")
        predicted = float(x @ self.beta)
        if self.log:
            with np.errstate(over="ignore", invalid="ignore"):
                h = float(np.exp(predicted + .5 * self.residual_variance))
            floored = False
        else:
            h = max(predicted, self.floor)
            floored = bool(predicted < self.floor)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("nonpositive_or_nonfinite_HAR_forecast")
        return h, floored


def _fit_har(rk, features, t, log):
    attempt = {"attempt_id": "ordinary_least_squares", "accepted": False,
               "rejection_reasons": [], "warnings": []}
    state = None
    x, y = features[t - HISTORY_WINDOW:t], rk[t - HISTORY_WINDOW:t]
    valid_target = np.isfinite(y) & (y > 0)
    complete = valid_target & np.isfinite(x).all(axis=1)
    attempt.update(history_n=HISTORY_WINDOW, positive_observed_targets=int(valid_target.sum()),
                   missing_or_nonpositive_targets=int((~valid_target).sum()),
                   incomplete_features_with_observed_target=int((valid_target & ~complete).sum()),
                   complete_rows=int(complete.sum()), columns=4)
    try:
        x, raw_y = x[complete], y[complete]
        if len(raw_y) < HAR_MIN_ROWS:
            raise ValueError("fewer_than_50_complete_training_rows")
        y = np.log(raw_y) if log else raw_y
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        attempt["rank"] = int(rank)
        if rank != x.shape[1]:
            raise ValueError("rank_deficient_training_design")
        residual_variance = float(np.sum((y - x @ beta) ** 2) / (len(y) - x.shape[1]))
        floor = HAR_FLOOR_RATIO * float(np.median(raw_y))
        if not np.isfinite(beta).all() or not np.isfinite(residual_variance):
            raise ValueError("nonfinite_HAR_parameters")
        candidate = _HarState(beta, residual_variance, floor, log, t)
        attempt.update(coefficients=beta.tolist(), residual_variance=residual_variance,
                       residual_variance_denominator=len(y) - x.shape[1],
                       training_median_RK=float(np.median(raw_y)), floor=floor)
        # A missing current feature is a daily forecast failure, not a reason to
        # reject an otherwise valid OLS fit.
        state = candidate
        attempt["accepted"] = True
    except Exception as exc:
        attempt["rejection_reasons"].append(str(exc))
        attempt["exception"] = f"{type(exc).__name__}: {exc}"
    return state, attempt


def _compact_path(market, r, dates, stop):
    """Run the archived compact recursion, retaining its exact provenance."""
    from research.evaluation_expansion import compact

    # Endpoint return is not observed and must never be needed by this adapter.
    prefix = {"r": np.r_[r[:stop], 0.], "date": dates[:stop + 1]}
    try:
        result = compact.predict(market["name"], prefix)
        return result["variance"], result["audit"]
    except RuntimeError as exc:
        # The original adapter rejects its entire path for any invalid value.
        # Preserve the same archived recursion but let this study replace only
        # the invalid dates, retaining earlier valid forecasts.
        if "Invalid frozen compact forecast" not in str(exc):
            raise
        group, fit, protocol, provenance = compact._saved_fit(market["name"])
        norm = fit["normalisation"]
        beta = np.array([fit["coefficients_likelihood"][k] for k in compact.PARAMETERS])
        valid = (np.arange(stop + 1) >= compact.BURN) & (prefix["date"] >= compact.FIT_CUTOFF)
        prepared = compact.original.prepare(prefix, norm, valid)
        _, _, r2, absr, signs, d1, d2, initial, _ = prepared
        h = compact.original.run_filter(r2, absr, signs, d1, d2, beta, norm["logh_mean"], initial)
        h[~valid] = np.nan
        return h, {"adapter_validation_exception": str(exc), "held_out_group": group,
                   "fit": fit, "protocol": protocol, "historical_inputs": provenance}


def forecast(market: dict, model: str, origins: np.ndarray) -> dict:
    """Return requested-origin means, variances, second moments and audit.

    hist250 uses exactly r[t-250:t], sample variance (ddof=1), sample mean and
    M2=variance+mean**2. Its use as fallback has mean=0 and M2=variance.
    EWMA emits h[100]=mean(r[:100]**2), then h[t]=.94*h[t-1]+.06*r[t-1]**2
    for t>100; it has zero mean and does not replay the initialization prefix.
    UniVol initializes from nonzero returns within the first 100 sessions and
    reuses its full-history recursion and defaults, including zero-return skipping. Its default geometric point variance
    is adapted to mean=0 and M2=variance for this point-volatility benchmark;
    these are proxy moments, not the native predictive mixture moments.
    """
    r, rk, dates, origins = _arrays(market, origins, model)
    start, stop = int(origins[0]), int(origins[-1])
    result = {key: np.empty(len(origins), dtype=float) for key in
              ("mean", "variance", "second_moment")}
    result["substituted"] = np.zeros(len(origins), dtype=bool)
    result["floor_applied"] = np.zeros(len(origins), dtype=bool)
    audit = {"model": model, "market": str(market.get("name", "market")),
             "units": {"mean": "percent", "variance": "percent_squared", "second_moment": "percent_squared"},
             "first_origin_index": start, "last_origin_index": stop,
             "refit_anchor_date": str(dates[start]), "requested_origins": len(origins),
             "processed_sessions": stop - start + 1, "attempts": [], "failures": [],
             "substitutions": [], "floors": [], "filter_warnings": [],
             "policy": {"history_window": HISTORY_WINDOW, "refit_every": REFIT_EVERY,
                        "optimizer_attempts": 1, "optimizer_start": "arch_default",
                        "maxiter": tier_a.MAXITER,
                        "initial_fit_failure": "zero_mean_hist250_until_next_scheduled_refit",
                        "refit_failure": "carry_previous_parameters_until_accepted_scheduled_refit",
                        "invalid_daily_forecast": "carry_last_valid_tuple_else_zero_mean_hist250",
                        "historical_variance": "250 prior returns; sample variance ddof=1",
                        "RK_missingness": "nonfinite or nonpositive; lag inputs causal forward fill; targets never imputed",
                        "HAR_training": "exactly 1000 prior target sessions; >=50 complete rows and rank=4",
                        "HAR_floor": "1e-4 times median observed RK among complete training rows; separate from substitution",
                        "logHAR_bias_correction": "exp(predicted_log_RK + 0.5*SSE/(n-4))"}}
    result["audit"] = audit
    if model in ARCH_MODELS:
        import arch
        audit["arch_version"] = arch.__version__
        audit["arch_specification"] = (tier_a.arch_spec(model) if model in tier_a.MODELS else
                                        {"mean": "Constant", "vol": "APARCH", "p": 1, "o": 1,
                                         "q": 1, "dist": "t", "rescale": False})
    state, last_valid = None, None
    carrying = False
    features = _har_features(rk[:stop + 1], model == "loghar") if model in ("har", "loghar") else None
    h = None
    compact_h = None
    online = None
    update_error = None
    if model == "ewma":
        h = float(np.mean(r[:100] ** 2))
        for t in range(101, start + 1):
            h = EWMA_DECAY * h + (1. - EWMA_DECAY) * r[t - 1] ** 2
        audit["initialization"] = {"h100": float(np.mean(r[:100] ** 2)), "decay": EWMA_DECAY,
                                   "initial_observations": 100, "first_recursive_observation_index": 100}
    elif model == "compact9":
        compact_h, audit["source_audit"] = _compact_path(market, r, dates, stop)
    elif model == "univol":
        from univol import UniVol, initial_scale
        scale = initial_scale(r[:100])
        online = UniVol(scale=scale)
        audit["initialization"] = {"initial_variance": scale, "observations": 100,
                                   "zero_returns_excluded_from_initial_mean_square": True,
                                   "variance": "default density-weighted geometric point forecast",
                                   "moment_semantics": "point-to-proxy: mean=0, variance=M2=h; not native mixture moments",
                                   "source": "univol.UniVol"}
        for i in range(start):
            try:
                online.update(float(r[i]))
            except Exception as exc:
                update_error = f"{type(exc).__name__}: {exc}"
                audit["failures"].append({"kind": "online_update", "observed_index": i,
                                          "exception": update_error})
    requested = {int(t): i for i, t in enumerate(origins)}
    for t in range(start, stop + 1):
        if model in (*ARCH_MODELS, "har", "loghar") and (t - start) % REFIT_EVERY == 0:
            if model in ARCH_MODELS:
                candidate, attempt = _fit_one(r[t - HISTORY_WINDOW:t].copy(), model, t - HISTORY_WINDOW, t)
            else:
                candidate, attempt = _fit_har(rk, features, t, model == "loghar")
            attempt.update(origin_index=t, origin=str(dates[t]), history_n=HISTORY_WINDOW,
                           history_start_index=t - HISTORY_WINDOW, history_end_index=t - 1,
                           history_start=str(dates[t - HISTORY_WINDOW]), history_end=str(dates[t - 1]))
            if candidate is None:
                carrying = True
                attempt["action"] = "carry_parameters" if state is not None else "zero_mean_hist250"
                audit["failures"].append({"kind": "fit_failure", **attempt})
            else:
                state, carrying = candidate, False
                attempt["action"] = "install_parameters"
            audit["attempts"].append(attempt)
        substituted, floored = False, False
        reason = None
        captured = []
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                if model == "hist250":
                    values = _historical(r, t)
                elif model == "ewma":
                    values = (0., float(h), float(h))
                elif model == "compact9":
                    value = float(compact_h[t])
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError("invalid_archived_compact_forecast")
                    values = (0., value, value)
                elif model == "univol":
                    if update_error is not None:
                        raise ValueError("preceding_online_update_failed: " + update_error)
                    value = float(online.predict()[1])
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError("invalid_UniVol_point_variance")
                    values = (0., value, value)
                elif state is None:
                    values = _historical(r, t, zero_mean=True)
                    reason = "no_accepted_parameters_hist250"
                elif model in ARCH_MODELS:
                    value = float(state.forecast(r[state.history_start:t]))
                    mu = float(state.parameters["mu"])
                    values = (mu, value, value + mu * mu)
                else:
                    value, floored = state.forecast(features[t])
                    values = (0., value, value)
                if not _valid_tuple(values):
                    raise ValueError("nonfinite_or_negative_forecast_tuple")
            if carrying and reason is None:
                reason = "carried_parameters_after_failed_refit"
            # Includes valid historical fallbacks; no current/future outcome is
            # involved in recognizing or choosing a valid forecast tuple.
            last_valid = values
        except Exception as exc:
            reason = "invalid_forecast_carry_last_valid" if last_valid is not None else "invalid_forecast_hist250"
            values = last_valid if last_valid is not None else _historical(r, t, zero_mean=True)
            if not _valid_tuple(values):
                raise RuntimeError(f"{model} at {dates[t]}: no finite positive forecast or historical fallback") from exc
            last_valid = values
            audit["failures"].append({"kind": "numeric_forecast", "origin_index": t,
                                      "origin": str(dates[t]), "exception": f"{type(exc).__name__}: {exc}",
                                      "action": reason})
        if captured:
            audit["filter_warnings"].append({"origin_index": t, "origin": str(dates[t]),
                                             "warnings": [str(item.message) for item in captured]})
        if reason is not None:
            substituted = True
            audit["substitutions"].append({"origin_index": t, "origin": str(dates[t]),
                                            "reason": reason, "requested": t in requested})
        if floored:
            audit["floors"].append({"origin_index": t, "origin": str(dates[t]),
                                    "floor": float(state.floor), "requested": t in requested})
        if t in requested:
            i = requested[t]
            for key, value in zip(("mean", "variance", "second_moment"), values):
                result[key][i] = value
            result["substituted"][i] = substituted
            result["floor_applied"][i] = floored
        if t < stop and model == "ewma":
            h = EWMA_DECAY * h + (1. - EWMA_DECAY) * r[t] ** 2
        elif t < stop and model == "univol":
            try:
                online.update(float(r[t]))
                update_error = None
            except Exception as exc:
                update_error = f"{type(exc).__name__}: {exc}"
                audit["failures"].append({"kind": "online_update", "observed_index": t,
                                          "exception": update_error})
    audit.update(fit_attempt_count=len(audit["attempts"]),
                 fit_failure_count=sum(not row["accepted"] for row in audit["attempts"]),
                 numerical_failure_count=sum(row["kind"] == "numeric_forecast" for row in audit["failures"]),
                 substitution_count=int(result["substituted"].sum()),
                 processed_substitution_count=len(audit["substitutions"]),
                 floor_count=int(result["floor_applied"].sum()), processed_floor_count=len(audit["floors"]))
    json.dumps(audit, allow_nan=False)
    return result
