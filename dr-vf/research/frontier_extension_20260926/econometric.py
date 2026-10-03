"""Causal, CPU-only realized-information controls for the dated extension.

An origin t predicts RK[t] from rows strictly before t.  RK is already in the
desired variance units.  These are benchmark adaptations, not paper replications:
1000 valid training targets, refits every 20 actual sessions, and the caller's
fixed origin panel replace the papers' samples and fitting schedules.

``mshar_sqrt`` fits the Fed paper's two-state Gaussian HAR form to sqrt(RK).
Only its intercept, daily slope and emission variance switch.  The output is
the regime-mixture E[sqrt(RK)**2], not the square of its mean.  This is a declared
variance-target adapter; it is not the Fed paper's volatility point forecast.

``logrhar`` is HAR on log RK with an MA(1) innovation.  Daily/weekly/monthly
regressors average log observations, rather than logging arithmetic RK means.
The likelihood is conditional on the observed HAR design, using a stationary
Gaussian MA(1) error state and Kalman filtering.  Its finite-sample initial
likelihood differs from a joint stationary restricted ARMA(22,1) likelihood.
``loghar_meanlog`` is its theta=0 control: OLS coefficients and Gaussian MLE
residual variance SSE/n.  Both log models use their model-implied one-step
forecast error variance for the lognormal mean and the same paper-style
training-range safeguard.  They do not replace the frozen older logHAR.

Sources: https://www.federalreserve.gov/econres/feds/files/2025061r1pap.pdf
         https://arxiv.org/html/2609.21587v1

Predictor gaps are forward-filled causally. Target gaps remain missing emissions
and retain the actual session clock. Failed fits carry the previous parameters
and keep updating states. With no fit, the fallback is a trailing positive-RK
mean (not a return-based historical variance: this API receives no returns).
All replacements are recorded and no requested date is silently discarded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
from scipy.special import logsumexp, ndtr


MODELS = ("mshar_sqrt", "logrhar", "loghar_meanlog")


@dataclass(frozen=True)
class EconometricConfig:
    model: str
    training_rows: int = 1000
    refit_every: int = 20
    maxiter: int = 300
    fallback_window: int = 250
    insanity_filter: bool = True
    ma_starts: tuple = (-0.5, -0.2, 0.0)
    # (regime persistence, intercept displacement in residual standard deviations)
    ms_starts: tuple = ((0.90, 0.5), (0.95, 1.0), (0.98, 1.5))
    variance_tolerance: float = 1e-10
    boundary_tolerance: float = 1e-6

    def validate(self):
        if self.model not in MODELS:
            raise ValueError("Unknown model: " + str(self.model))
        for name in ("training_rows", "refit_every", "maxiter", "fallback_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.training_rows < 8:
            raise ValueError("At least eight valid training rows are required")
        if not 0 < self.variance_tolerance < 1 or not 0 < self.boundary_tolerance < .01:
            raise ValueError("Invalid numerical acceptance tolerances")
        if not self.ma_starts or any(not -1 < float(v) < 1 for v in self.ma_starts):
            raise ValueError("MA starting values must be strictly invertible")
        if not self.ms_starts or any(not .5 < float(p) < 1 or float(d) <= 0
                                     for p, d in self.ms_starts):
            raise ValueError("Invalid deterministic MS starting values")


def _config(config):
    value = EconometricConfig(**config) if isinstance(config, dict) else config
    if not isinstance(value, EconometricConfig):
        raise TypeError("config must be EconometricConfig or a dictionary")
    value.validate()
    return value


def _features(rk, model):
    """x[t] uses exactly the previous 1/5/22 sessions, never RK[t]."""
    filled = np.full(len(rk), np.nan)
    previous = np.nan
    for i, value in enumerate(rk):
        if np.isfinite(value) and value > 0:
            previous = float(value)
        filled[i] = previous
    with np.errstate(invalid="ignore", divide="ignore"):
        transformed = np.sqrt(filled) if model == "mshar_sqrt" else np.log(filled)
    x = np.full((len(rk), 4), np.nan)
    for t in range(22, len(rk)):
        x[t] = [1., transformed[t - 1], transformed[t - 5:t].mean(),
                transformed[t - 22:t].mean()]
    return x


def _window(rk, x, t, config):
    valid = np.isfinite(rk[:t]) & (rk[:t] > 0) & np.isfinite(x[:t]).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < config.training_rows:
        raise ValueError("fewer_than_required_valid_training_rows")
    start = int(indices[-config.training_rows])
    observed = valid[start:t]
    design = x[start:t]
    # Missing target emissions are retained between these indices.
    if not np.isfinite(design).all():
        raise ValueError("nonfinite_training_predictors")
    if np.linalg.matrix_rank(design[observed]) != 4:
        raise ValueError("rank_deficient_training_design")
    return start, observed, design


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _positive_mean(rk, t, window):
    past = rk[max(0, t - window):t]
    valid = past[np.isfinite(past) & (past > 0)]
    if not len(valid):
        raise ValueError("no_positive_RK_in_causal_fallback_window")
    value = float(np.mean(valid))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("invalid_causal_RK_fallback")
    return value


@dataclass
class _LogState:
    beta: np.ndarray
    theta: float
    sigma2: float
    center: float
    innovation_mean: float
    innovation_variance: float
    lower: float
    upper: float
    training_mean: float
    fit_origin: int
    history_start: int
    training_rows: int

    def _design(self, x):
        return np.r_[1., np.asarray(x[1:]) - self.center]

    def log_moments(self, x):
        mu = self.center + float(self._design(x) @ self.beta) + self.theta * self.innovation_mean
        variance = self.sigma2 + self.theta ** 2 * self.innovation_variance
        return mu, variance

    def predict(self, x):
        mu, variance = self.log_moments(x)
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.exp(mu + .5 * variance))
        return value, {"predicted_log_mean": float(mu),
                       "predicted_log_variance": float(variance)}

    def observe(self, rk, x):
        # The observation is made after forecasting this session.
        if not np.isfinite(rk) or rk <= 0:
            self.innovation_mean = 0.
            self.innovation_variance = self.sigma2
            return
        mu, variance = self.log_moments(x)
        error = float(np.log(rk) - mu)
        gain = self.sigma2 / variance
        self.innovation_mean = gain * error
        self.innovation_variance = max(0., self.sigma2 * (1. - gain))


@dataclass
class _MSState:
    beta: np.ndarray             # (2,4), normalized standard-deviation units
    sigma2: np.ndarray
    transition: np.ndarray       # destination rows, source columns
    filtered: np.ndarray
    scale: float
    fit_origin: int
    history_start: int
    training_rows: int

    def _design(self, x):
        return np.r_[1., np.asarray(x[1:]) / self.scale]

    def components(self, x):
        probability = self.transition @ self.filtered
        probability = probability / probability.sum()
        mean = self.beta @ self._design(x)
        return probability, mean

    def predict(self, x):
        probability, mean = self.components(x)
        value = self.scale ** 2 * float(probability @ (mean ** 2 + self.sigma2))
        sd_mean = self.scale * float(probability @ mean)
        negative_mass = float(probability @ ndtr(-mean / np.sqrt(self.sigma2)))
        return value, {"regime_probabilities": probability.tolist(),
                       "predicted_sd_mean": sd_mean,
                       "predicted_sd_variance": max(0., value - sd_mean ** 2),
                       "negative_sd_mass": negative_mass}

    def observe(self, rk, x):
        probability, mean = self.components(x)
        if not np.isfinite(rk) or rk <= 0:
            self.filtered = probability
            return
        z = np.sqrt(rk) / self.scale
        log_weights = (np.log(probability) - .5 * np.log(2. * np.pi * self.sigma2)
                       - .5 * (z - mean) ** 2 / self.sigma2)
        normalizer = logsumexp(log_weights)
        if not np.isfinite(normalizer):
            raise ValueError("nonfinite_Hamilton_update")
        self.filtered = np.exp(log_weights - normalizer)


def _fit_log(rk, x, t, config):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    start, observed, raw_x = _window(rk, x, t, config)
    raw = rk[start:t]
    y = np.full(len(raw), np.nan)
    y[observed] = np.log(raw[observed])
    center = float(np.mean(y[observed]))
    y -= center
    design = raw_x.copy()
    design[:, 1:] -= center
    beta = np.linalg.lstsq(design[observed], y[observed], rcond=None)[0]
    sigma2 = float(np.mean((y[observed] - design[observed] @ beta) ** 2))
    if not np.isfinite(sigma2) or sigma2 <= config.variance_tolerance:
        raise ValueError("degenerate_training_residual_variance")
    attempts = []
    if config.model == "loghar_meanlog":
        best = np.r_[beta, 0., sigma2]
        attempts.append({"start": "nested_OLS_Gaussian_MLE", "accepted": True,
                         "sigma2_denominator": int(observed.sum())})
    else:
        model = SARIMAX(y, exog=design, order=(0, 0, 1), trend="n",
                        initialization="stationary", enforce_stationarity=True,
                        enforce_invertibility=True)
        best, best_ll = None, -np.inf
        for theta in config.ma_starts:
            audit = {"start": float(theta), "accepted": False}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fit = model.fit(start_params=np.r_[beta, theta, sigma2],
                                    transformed=True, method="lbfgs", maxiter=config.maxiter,
                                    disp=False, cov_type="none")
                    pars = np.asarray(fit.params, float)
                    converged = bool(fit.mle_retvals.get("converged", False))
                    good = (converged and np.isfinite(pars).all() and np.isfinite(fit.llf)
                            and pars[-1] > config.variance_tolerance
                            and abs(pars[-2]) < 1. - config.boundary_tolerance)
                    audit.update(accepted=bool(good), converged=converged,
                                 loglikelihood=_finite(fit.llf), theta=_finite(pars[-2]),
                                 iterations=int(fit.mle_retvals.get("iterations", -1)))
                    if good and fit.llf > best_ll:
                        best, best_ll = pars.copy(), float(fit.llf)
                except Exception as exc:
                    audit["exception"] = type(exc).__name__ + ": " + str(exc)
            audit["warnings"] = [str(w.message) for w in caught]
            attempts.append(audit)
        if best is None:
            return None, {"accepted": False, "reason": "no_accepted_MA1_start", "attempts": attempts}
    state = _LogState(best[:4], float(best[4]), float(best[5]), center, 0.,
                      float(best[5]), float(np.min(raw[observed])), float(np.max(raw[observed])),
                      float(np.mean(raw[observed])), t, start, int(observed.sum()))
    for value, row in zip(raw, raw_x):
        state.observe(value, row)
    return state, {"accepted": True, "attempts": attempts, "parameters": best.tolist(),
                   "history_start": start, "training_valid_rows": int(observed.sum()),
                   "training_calendar_rows": len(raw), "missing_emissions": int((~observed).sum()),
                   "likelihood": "conditional_on_HAR_design_stationary_MA1_Kalman",
                   "log_center": center}


def _fit_ms(rk, x, t, config):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    class MissingEmissionRegression(MarkovRegression):
        def _conditional_loglikelihoods(self, params):
            answer = super()._conditional_loglikelihoods(params)
            # A missing target has likelihood one in every state. Transition
            # evolution remains active; neither target nor calendar is imputed.
            answer[..., ~np.isfinite(self.endog)] = 0.
            return answer

    start, observed, raw_x = _window(rk, x, t, config)
    raw = rk[start:t]
    scale = float(np.median(np.sqrt(raw[observed])))
    y = np.full(len(raw), np.nan)
    y[observed] = np.sqrt(raw[observed]) / scale
    design = raw_x.copy()
    design[:, 1:] /= scale
    beta = np.linalg.lstsq(design[observed], y[observed], rcond=None)[0]
    sigma2 = float(np.mean((y[observed] - design[observed] @ beta) ** 2))
    if not np.isfinite(sigma2) or sigma2 <= config.variance_tolerance:
        raise ValueError("degenerate_training_residual_variance")
    model = MissingEmissionRegression(y, 2, trend="c", exog=design[:, 1:],
                                     switching_trend=True, switching_exog=[True, False, False],
                                     switching_variance=True, missing="none")
    attempts, best, best_ll = [], None, -np.inf
    for persistence, displacement in config.ms_starts:
        delta = float(displacement) * np.sqrt(sigma2)
        initial = np.array([persistence, 1. - persistence, beta[0] - delta, beta[0] + delta,
                            beta[1], beta[1], beta[2], beta[3], .5 * sigma2, 1.5 * sigma2])
        audit = {"start": [float(persistence), float(displacement)], "accepted": False}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                # No EM or stochastic searches: all starts and fit work are fixed.
                fit = model.fit(start_params=initial, transformed=True, method="bfgs",
                                maxiter=config.maxiter, disp=False, em_iter=0,
                                search_reps=0, cov_type="none")
                pars = np.asarray(fit.params, float)
                converged = bool(fit.mle_retvals.get("converged", False))
                eps = config.boundary_tolerance
                good = (converged and np.isfinite(pars).all() and np.isfinite(fit.llf)
                        and np.all(pars[-2:] > config.variance_tolerance)
                        and np.all(pars[:2] > eps) and np.all(pars[:2] < 1. - eps))
                audit.update(accepted=bool(good), converged=converged,
                             loglikelihood=_finite(fit.llf),
                             function_calls=int(fit.mle_retvals.get("fcalls", -1)))
                if good and fit.llf > best_ll:
                    best, best_ll = pars.copy(), float(fit.llf)
            except Exception as exc:
                audit["exception"] = type(exc).__name__ + ": " + str(exc)
        audit["warnings"] = [str(w.message) for w in caught]
        attempts.append(audit)
    if best is None:
        return None, {"accepted": False, "reason": "no_accepted_MS_start", "attempts": attempts}
    p00, p10 = best[:2]
    transition = np.array([[p00, p10], [1. - p00, 1. - p10]])
    stationary = np.array([p10, 1. - p00]) / (p10 + 1. - p00)
    coefficients = np.array([[best[2], best[4], best[6], best[7]],
                             [best[3], best[5], best[6], best[7]]])
    state = _MSState(coefficients, best[-2:].copy(), transition, stationary,
                     scale, t, start, int(observed.sum()))
    for value, row in zip(raw, raw_x):
        state.observe(value, row)
    return state, {"accepted": True, "attempts": attempts, "parameters": best.tolist(),
                   "parameter_names": list(model.param_names), "scale": scale,
                   "history_start": start, "training_valid_rows": int(observed.sum()),
                   "training_calendar_rows": len(raw), "missing_emissions": int((~observed).sum()),
                   "likelihood": "two_state_Gaussian_Hamilton_stationary_initial_probabilities"}


def _fit(rk, features, t, config):
    try:
        return (_fit_ms if config.model == "mshar_sqrt" else _fit_log)(rk, features, t, config)
    except Exception as exc:
        return None, {"accepted": False, "reason": type(exc).__name__ + ": " + str(exc),
                      "attempts": []}


def forecast_series(array_rk, origins, config):
    """Return requested rows ``{origin, variance, metadata}`` in input order.

    The smallest requested origin anchors the refit clock. Always supply the
    same full calibration/evaluation union for a market when resuming. No file
    is written, no loss is calculated, and no package is installed by this API.
    """
    config = _config(config)
    rk = np.asarray(array_rk, dtype=float)
    requested = np.asarray(origins)
    if rk.ndim != 1 or not len(rk):
        raise ValueError("array_rk must be a nonempty vector")
    if (requested.ndim != 1 or not len(requested) or requested.dtype.kind not in "iu"
            or requested[0] < 22 or requested[-1] >= len(rk)
            or np.any(requested[1:] <= requested[:-1])):
        raise ValueError("origins must be sorted unique integer indices inside the RK history")
    first, last = int(requested[0]), int(requested[-1])
    # The final target is never transformed or filtered; even a future infinity
    # cannot influence preparation of a preceding forecast.
    # Build the final design using a dummy missing final value, not its target.
    features = _features(np.r_[rk[:last], np.nan], config.model)
    requested_set = set(int(t) for t in requested)
    rows, state, last_valid = [], None, None
    fit_events, state_events = [], []
    carrying = False
    for t in range(first, last + 1):
        fit_audit = None
        if (t - first) % config.refit_every == 0:
            candidate, fit_audit = _fit(rk, features, t, config)
            fit_events.append({"origin": t, **fit_audit})
            if candidate is not None:
                state, carrying = candidate, False
            else:
                carrying = state is not None
        reasons, detail = [], {}
        if carrying:
            reasons.append("carried_parameters_after_failed_refit")
        if state is None:
            value = _positive_mean(rk, t, config.fallback_window)
            reasons.append("no_accepted_fit_causal_RK_mean")
        else:
            try:
                if not np.isfinite(features[t]).all():
                    raise ValueError("missing_causal_predictors")
                value, detail = state.predict(features[t])
                if isinstance(state, _LogState) and config.insanity_filter:
                    if not np.isfinite(value) or value < state.lower or value > state.upper:
                        value = state.training_mean
                        detail["range_replaced"] = True
                        reasons.append("outside_training_range_use_training_mean")
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("nonpositive_or_nonfinite_forecast")
            except Exception as exc:
                detail = {"forecast_failure": type(exc).__name__ + ": " + str(exc)}
                value = last_valid if last_valid is not None else _positive_mean(rk, t, config.fallback_window)
                reasons.append("invalid_forecast_carry_last" if last_valid is not None
                               else "invalid_forecast_causal_RK_mean")
        last_valid = float(value)
        if t in requested_set:
            metadata = {"model": config.model, "family": "R", "refit_anchor": first,
                        "fit_origin": None if state is None else int(state.fit_origin),
                        "history_start": None if state is None else int(state.history_start),
                        "training_valid_rows": None if state is None else int(state.training_rows),
                        "substituted": bool(reasons), "substitution_reasons": reasons,
                        "configuration": asdict(config), "fit_events": fit_events,
                        "state_events": state_events, **detail}
            if fit_audit is not None:
                metadata["fit_audit"] = fit_audit
            rows.append({"origin": t, "variance": float(value), "metadata": metadata})
            fit_events, state_events = [], []
        if t < last and state is not None:
            try:
                if not np.isfinite(rk[t]) or rk[t] <= 0:
                    state_events.append({"observed_index": t, "action": "missing_target_prediction_only"})
                state.observe(rk[t], features[t])
            except Exception as exc:
                # An invalid filtered state must not be carried as if sound.
                # Keep fixed refit scheduling and disclose the fallback next day.
                state_events.append({"observed_index": t, "action": "discard_invalid_state",
                                     "exception": type(exc).__name__ + ": " + str(exc)})
                state, carrying = None, False
    return rows
