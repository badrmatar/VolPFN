"""Causal daily-variance adaptation of Zhao et al., AAAI 2024, Eqs. 1/6/7.

Clean-room equations implementation; no author code is vendored.  Inputs are
observed daily log returns in chronological order, ending at forecast origin t.
Every returned column j forecasts the return after history[:, j], never that
observed return.  This is not the paper's overlapping five-day target experiment.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _history(history: Tensor, *, nonnegative: bool = False) -> None:
    if history.ndim != 2 or history.shape[0] == 0 or history.shape[1] == 0:
        raise ValueError("history must be a nonempty floating tensor [batch, time]")
    if not history.is_floating_point():
        raise ValueError("history must be floating point")
    _finite("history", history)
    if nonnegative and (history < 0).any():
        raise ValueError("squared residual history must be nonnegative")


def student_t_nll(
    next_return: Tensor, mean: Tensor, variance: Tensor, nu: Tensor | float = 5.0,
    *, reduction: str = "mean",
) -> Tensor:
    """Complete Student-t NLL with *variance*, rather than t scale, as input.

    StudentT(df=nu, loc=mean, scale=sqrt(variance*(nu-2)/nu)) has the requested
    finite conditional variance.  Includes nu-dependent normalizing constants
    omitted by paper Eq. 6; they are required for a proper score or learned nu.
    """
    nu_t = torch.as_tensor(nu, dtype=variance.dtype, device=variance.device)
    if next_return.shape != variance.shape or mean.shape != variance.shape:
        raise ValueError("next_return, mean and variance must have identical shapes")
    if nu_t.numel() != 1 and nu_t.shape != variance.shape:
        raise ValueError("nu must be scalar or match the variance shape")
    for name, value in (("next_return", next_return), ("mean", mean),
                        ("variance", variance), ("nu", nu_t)):
        _finite(name, value)
    if (variance <= 0).any() or (nu_t <= 2).any():
        raise ValueError("variance must be positive and nu must exceed 2")
    error = next_return - mean
    result = (
        0.5 * (torch.log(variance) + torch.log(nu_t - 2) + math.log(math.pi))
        + torch.lgamma(nu_t / 2) - torch.lgamma((nu_t + 1) / 2)
        + 0.5 * (nu_t + 1)
        * torch.log1p(error.square() / ((nu_t - 2) * variance))
    )
    _finite("Student-t NLL", result)
    if reduction == "none":
        return result
    if reduction == "sum":
        return result.sum()
    if reduction == "mean":
        return result.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class GARCHLSTM(nn.Module):
    """Scalar Eq. 7 cell with a positive GARCH(1,1) output gate.

    omega>0, alpha/beta>=0, alpha+beta<1 are enforced by parameterization.
    |w|<1 keeps the Eq. 7 multiplicative controller strictly positive.  The
    kernel's stationarity constraint does not prove stationarity of the hybrid.
    Defaults presume training-only RMS standardized returns.  initial_variance
    is a frozen prior, never a statistic of the supplied forecast window.
    """
    def __init__(
        self, *, omega: float = 0.03, alpha: float = 0.07, beta: float = 0.90,
        initial_variance: float = 1.0, modulation: float = 0.10,
        nu: float = 5.0, learn_nu: bool = False, variance_floor: float = 1e-8,
    ) -> None:
        super().__init__()
        values = (omega, alpha, beta, initial_variance, modulation, nu, variance_floor)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("constructor parameters must be finite")
        if omega <= variance_floor or initial_variance <= 0 or variance_floor <= 0:
            raise ValueError("omega, initial variance and variance floor must be positive")
        if alpha <= 0 or beta <= 0 or alpha + beta >= 0.999:
            raise ValueError("require positive alpha/beta and alpha+beta<0.999")
        if abs(modulation) >= 0.99 or nu <= 2.01:
            raise ValueError("require |modulation|<0.99 and nu>2.01")
        self.variance_floor = float(variance_floor)
        self.raw_omega = nn.Parameter(torch.tensor(_inverse_softplus(omega - variance_floor)))
        self.raw_weights = nn.Parameter(torch.tensor([
            math.log(alpha), math.log(beta), math.log(0.999 - alpha - beta),
        ]))
        self.raw_modulation = nn.Parameter(torch.tensor(math.atanh(modulation / 0.99)))
        # Separate input/recurrent weights in Eq. 7 are a single affine map
        # over [epsilon_(t-1), variance_(t-1)] with three scalar outputs.
        self.gates = nn.Linear(2, 3)
        self.register_buffer("initial_variance", torch.tensor(float(initial_variance)))
        self.learn_nu = bool(learn_nu)
        if learn_nu:
            self.raw_nu = nn.Parameter(torch.tensor(_inverse_softplus(nu - 2.01)))
        else:
            self.register_buffer("fixed_nu", torch.tensor(float(nu)))

    def kernel_parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        weights = 0.999 * torch.softmax(self.raw_weights, dim=0)
        return F.softplus(self.raw_omega) + self.variance_floor, weights[0], weights[1]

    def degrees_of_freedom(self) -> Tensor:
        if self.learn_nu:
            return 2.01 + F.softplus(self.raw_nu)
        return self.fixed_nu

    def forward_sequence(self, history: Tensor) -> dict[str, Tensor]:
        """Return one-session-ahead forecasts after every observed prefix."""
        _history(history)
        omega, alpha, beta = self.kernel_parameters()
        variance = self.initial_variance.expand(history.shape[0])
        cell = torch.zeros_like(variance)
        modulation = 0.99 * torch.tanh(self.raw_modulation)
        variances = []
        for observed_return in history.unbind(dim=1):
            affine = self.gates(torch.stack((observed_return, variance), dim=-1))
            forget, include = torch.sigmoid(affine[:, 0]), torch.sigmoid(affine[:, 1])
            candidate = torch.tanh(affine[:, 2])
            cell = forget * cell + include * candidate
            kernel = omega + alpha * observed_return.square() + beta * variance
            variance = kernel * (1 + modulation * torch.tanh(cell))
            variances.append(variance)
        output = torch.stack(variances, dim=1)
        _finite("forecast variance", output)
        if (output <= 0).any():
            raise FloatingPointError("forecast variance is not positive")
        return {"mean": torch.zeros_like(output), "variance": output,
                "nu": self.degrees_of_freedom()}

    def forward(self, history: Tensor) -> dict[str, Tensor]:
        output = self.forward_sequence(history)
        return {"mean": output["mean"][:, -1], "variance": output["variance"][:, -1],
                "nu": output["nu"]}

    def metadata(self) -> dict[str, Any]:
        omega, alpha, beta = self.kernel_parameters()
        return {
            "model": "garch_lstm_daily_adaptation", "paper": "arxiv:2402.06642v1",
            "target": "next observed daily return likelihood; conditional variance",
            "kernel": "GARCH(1,1)", "zero_mean": True,
            "initial_variance": float(self.initial_variance.detach().cpu()),
            "omega": float(omega.detach().cpu()), "alpha": float(alpha.detach().cpu()),
            "beta": float(beta.detach().cpu()), "variance_floor": self.variance_floor,
            "nu": float(self.degrees_of_freedom().detach().cpu()), "learn_nu": self.learn_nu,
            "modulation": float((0.99 * torch.tanh(self.raw_modulation)).detach().cpu()),
            "adaptation": "single-session target, constrained positive Eq7 controller",
        }


# Name explicitly identifies the loss used by the shared experiment runner.
garch_lstm_loss = student_t_nll
