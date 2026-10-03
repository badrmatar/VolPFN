"""Causal next-session GINN adaptation of Xu et al. (ICAIF 2024), Eqs. 13-17.

No latent variance labels are accepted.  The squared observed innovation is a
noisy conditional second-moment observation, not 'true historical variance'.
AR and constant-mean GARCH teacher parameters are fitted once on training
returns before any neural training, then frozen for validation/test filtering.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .garch_lstm import _finite, _history
except ImportError:  # direct execution from the programme directory
    from garch_lstm import _finite, _history


def _returns(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("returns must be a nonempty finite one-dimensional array")
    return array


@dataclass(frozen=True)
class FrozenTeacher:
    """Serializable train-only fits and causal filtered training diagnostics."""
    ar_intercept: float
    ar_coefficients: tuple[float, ...]  # newest lag first
    fallback_mean: float
    garch_mean: float
    omega: float
    alpha: float
    beta: float
    initial_variance: float
    train_count: int
    train_sha256: str
    train_residuals: tuple[float, ...]
    train_teacher_variances: tuple[float, ...]  # h_t before observing r_t
    next_teacher_variance: float
    fit_loglikelihood: float

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("train_residuals")
        result.pop("train_teacher_variances")
        result.update({"teacher_fit_partition": "training_only", "frozen": True,
                       "ar_order": len(self.ar_coefficients),
                       "garch_distribution": "normal", "garch_mean_type": "constant"})
        return result


def residual_history(history: Tensor, teacher: FrozenTeacher) -> Tensor:
    """Causal AR innovations: residual at j uses only returns before j.

    At the start of an isolated window, pre-window AR lags are unknown, so the
    frozen training mean initializes the first p means.  For exact continuous
    alignment, transform the complete available prefix once, then slice windows.
    """
    _history(history)
    order = len(teacher.ar_coefficients)
    means = torch.full_like(history, teacher.fallback_mean)
    coefficients = history.new_tensor(teacher.ar_coefficients)
    for j in range(order, history.shape[1]):
        means[:, j] = teacher.ar_intercept + (
            history[:, j - order:j].flip(dims=(1,)) * coefficients
        ).sum(dim=1)
    residuals = history - means
    _finite("AR residuals", residuals)
    return residuals


def next_ar_mean(history: Tensor, teacher: FrozenTeacher) -> Tensor:
    _history(history)
    order = len(teacher.ar_coefficients)
    if history.shape[1] < order:
        raise ValueError("history is shorter than the frozen AR order")
    return teacher.ar_intercept + (
        history[:, -order:].flip(dims=(1,)) * history.new_tensor(teacher.ar_coefficients)
    ).sum(dim=1)


def teacher_variances(history: Tensor, teacher: FrozenTeacher) -> Tensor:
    """Return [h_0,...,h_T], where h_j precedes observing history[:,j].

    Starts from the fixed training-fitted backcast.  Supplying the complete
    available return prefix gives the exact continuously filtered teacher path.
    The final column is a forecast for the next unseen return.
    """
    _history(history)
    variance = history.new_full((history.shape[0],), teacher.initial_variance)
    sequence = [variance]
    for observed in history.unbind(dim=1):
        innovation = observed - teacher.garch_mean
        variance = teacher.omega + teacher.alpha * innovation.square() + teacher.beta * variance
        sequence.append(variance)
    result = torch.stack(sequence, dim=1)
    _finite("teacher variances", result)
    if (result <= 0).any():
        raise FloatingPointError("teacher variance must be positive")
    return result


def prepare_history(history: Tensor, teacher: FrozenTeacher) -> dict[str, Tensor]:
    """Transform an observed prefix without reading a target or future return."""
    return {"squared_residuals": residual_history(history, teacher).square(),
            "mean": next_ar_mean(history, teacher),
            "teacher_variance": teacher_variances(history, teacher)[:, -1]}


def fit_teacher(train_returns: Any, *, ar_order: int = 1, maxiter: int = 1000) -> FrozenTeacher:
    """Fit AR(p) OLS and independent constant-mean GARCH(1,1) on TRAIN only.

    The caller supplies the explicit training slice, already standardized using
    training-only statistics if desired.  AR order=1 is predeclared because the
    paper does not specify AR order.  No validation/test data or variance labels
    are accepted.  arch is imported lazily so inference needs only numpy/torch.
    """
    returns = _returns(train_returns)
    if not isinstance(ar_order, int) or ar_order < 1 or len(returns) < max(30, ar_order + 3):
        raise ValueError("need AR order>=1 and at least max(30, order+3) training returns")
    if not isinstance(maxiter, int) or maxiter < 1 or returns.var() <= 0:
        raise ValueError("need nonconstant training returns and positive maxiter")
    from arch import arch_model

    target = returns[ar_order:]
    columns = [np.ones(len(target))]
    columns.extend(returns[ar_order - lag:len(returns) - lag]
                   for lag in range(1, ar_order + 1))
    coefficients = np.linalg.lstsq(np.column_stack(columns), target, rcond=None)[0]
    model = arch_model(returns, mean="Constant", vol="GARCH", p=1, o=0, q=1,
                       dist="normal", rescale=False)
    fitted = model.fit(disp="off", show_warning=False, options={"maxiter": maxiter})
    if fitted.convergence_flag != 0:
        raise RuntimeError(f"train-only GARCH teacher fit failed: {fitted.optimization_result}")
    parameters = fitted.params
    garch_mean, omega, alpha, beta = (float(parameters[name])
                                     for name in ("mu", "omega", "alpha[1]", "beta[1]"))
    conditional = np.asarray(fitted.conditional_volatility, dtype=np.float64) ** 2
    if (not np.isfinite([garch_mean, omega, alpha, beta]).all() or omega <= 0
            or alpha < 0 or beta < 0 or alpha + beta > 1 + 1e-8
            or not np.isfinite(coefficients).all()
            or not np.isfinite(conditional).all() or (conditional <= 0).any()):
        raise RuntimeError("GARCH teacher fit violates finite positive variance requirements")
    next_variance = omega + alpha * (returns[-1] - garch_mean) ** 2 + beta * conditional[-1]
    provisional = FrozenTeacher(
        ar_intercept=float(coefficients[0]),
        ar_coefficients=tuple(float(x) for x in coefficients[1:]),
        fallback_mean=float(returns.mean()), garch_mean=garch_mean,
        omega=omega, alpha=alpha, beta=beta, initial_variance=float(conditional[0]),
        train_count=len(returns),
        train_sha256=hashlib.sha256(np.ascontiguousarray(returns, dtype="<f8").tobytes()).hexdigest(),
        train_residuals=(), train_teacher_variances=tuple(float(x) for x in conditional),
        next_teacher_variance=float(next_variance), fit_loglikelihood=float(fitted.loglikelihood),
    )
    residuals = residual_history(torch.from_numpy(returns[None, :]), provisional)[0].numpy()
    return FrozenTeacher(**{**asdict(provisional), "train_residuals": tuple(float(x) for x in residuals)})


def ginn_loss(
    predicted_variance: Tensor, next_return: Tensor, next_mean: Tensor,
    teacher_variance: Tensor, *, lambda_observed: float = 0.01,
    reduction: str = "mean",
) -> Tensor:
    """Eq.17 loss, with squared observed innovation explicitly a noisy proxy.

    AR means and GARCH predictions are detached so teacher fitting cannot occur
    through the neural objective.  lambda=0.01 is predeclared from the paper,
    not reselected using concurrent market test observations.
    """
    if not math.isfinite(lambda_observed) or not 0 <= lambda_observed <= 1:
        raise ValueError("lambda_observed must lie in [0,1]")
    if any(value.shape != predicted_variance.shape
           for value in (next_return, next_mean, teacher_variance)):
        raise ValueError("loss inputs must have identical shapes; broadcasting labels is forbidden")
    for name, value in (("predicted_variance", predicted_variance),
                        ("next_return", next_return), ("next_mean", next_mean),
                        ("teacher_variance", teacher_variance)):
        _finite(name, value)
    if (predicted_variance <= 0).any() or (teacher_variance <= 0).any():
        raise ValueError("predicted and teacher variance must be positive")
    observed_squared_innovation = (next_return.detach() - next_mean.detach()).square()
    result = (lambda_observed * (predicted_variance - observed_squared_innovation).square()
              + (1 - lambda_observed) * (predicted_variance - teacher_variance.detach()).square())
    _finite("GINN loss", result)
    if reduction == "none":
        return result
    if reduction == "sum":
        return result.sum()
    if reduction == "mean":
        return result.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


class GINN(nn.Module):
    """Paper's three-layer LSTM on causal squared innovations, positive head.

    forward() receives squared AR residuals, not raw returns and not latent
    variance.  For raw returns use prepare_history and replace output['mean']
    with its frozen AR forecast.  Nu=5 is a scoring adapter, not paper Eq.17.
    """
    def __init__(self, *, hidden_size: int = 256, num_layers: int = 3,
                 dropout: float = 0.2, nu: float = 5.0,
                 variance_floor: float = 1e-8) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid LSTM dimensions or dropout")
        if not math.isfinite(nu) or nu <= 2 or not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("require finite nu>2 and positive variance_floor")
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.dropout, self.variance_floor = float(dropout), float(variance_floor)
        self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.readout = nn.Linear(hidden_size, 1)
        self.register_buffer("fixed_nu", torch.tensor(float(nu)))

    def forward(self, history: Tensor) -> dict[str, Tensor]:
        _history(history, nonnegative=True)
        sequence, _ = self.lstm(history.unsqueeze(-1))
        features = self.linear(sequence[:, -1, :])
        # Training batches must contain at least two independent windows for
        # the paper's BatchNorm.  Eval uses frozen running training statistics.
        features = F.relu(self.batch_norm(features))
        variance = F.softplus(self.readout(features).squeeze(-1)) + self.variance_floor
        _finite("forecast variance", variance)
        return {"mean": torch.zeros_like(variance), "variance": variance, "nu": self.fixed_nu}

    def metadata(self) -> dict[str, Any]:
        return {"model": "ginn_daily_adaptation", "paper": "arxiv:2410.00288v1",
                "target": "next observed squared AR innovation plus frozen GARCH regularizer",
                "hidden_size": self.hidden_size, "num_layers": self.num_layers,
                "dropout": self.dropout, "variance_floor": self.variance_floor,
                "nu": float(self.fixed_nu.detach().cpu()), "input": "causal squared AR innovations",
                "adaptation": "frozen train-only teachers; softplus positive variance head"}
