"""Author-code adaptation of ARCH-LSTM (Acevedo-Prins & Velasquez, 2026).

Port of the FIXED-nu ``build_lstm_t_student`` branch at author commit
9c8aef3506b1e676f181c5af5b1db7f1834212fb (MIT). This is not a reproduction
of the paper's complete experimental protocol; see ARCH_LSTM_SOURCE.md.

The runner supplies centered returns divided by a TRAIN-only RMS. Squaring
these returns reproduces the author's input eps2 / mean(eps2_train). The
model forecasts NEXT-period variance in those standardized return units.
It fits no scaler, uses no GARCH estimate, and keeps no state across calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


AUTHOR_COMMIT = "9c8aef3506b1e676f181c5af5b1db7f1834212fb"


class ARCHLSTM(nn.Module):
    """One-layer tanh LSTM on squared returns, with a positive variance head.

    ``context=1`` follows the paper's primary window. ``hidden_size=32``
    and zero dropout are declared experiment defaults within its search
    space, not claimed to be the paper's selected hyperparameters.
    Fixed nu=3 follows the requested sensitivity experiment, rather than
    the paper's inconsistently documented learned/tuned-nu final runs.
    The clipped exponential head follows executable AUTHOR CODE, while
    the paper describes a softplus head and batch normalization.
    """

    def __init__(
        self,
        context: int = 1,
        hidden_size: int = 32,
        dropout: float = 0.0,
        nu: float = 3.0,
        log_variance_clip: float = 20.0,
    ) -> None:
        super().__init__()
        if isinstance(context, bool) or int(context) != context or context < 1:
            raise ValueError("context must be a positive integer")
        if isinstance(hidden_size, bool) or int(hidden_size) != hidden_size or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(nu) or nu <= 2:
            raise ValueError("nu must be finite and greater than 2")
        if not math.isfinite(log_variance_clip) or log_variance_clip <= 0:
            raise ValueError("log_variance_clip must be finite and positive")
        self.context = int(context)
        self.hidden_size = int(hidden_size)
        self.input_dropout = float(dropout)
        self.log_variance_clip = float(log_variance_clip)
        # Keras has no recurrent dropout in this branch. Its input dropout
        # is applied below with a mask shared across timesteps.
        self.lstm = nn.LSTM(1, self.hidden_size, batch_first=True)
        self.output_dropout = nn.Dropout(dropout)
        self.log_variance_head = nn.Linear(self.hidden_size, 1)
        self.register_buffer("degrees_of_freedom", torch.tensor(float(nu)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Mirror Keras Glorot input/dense, orthogonal recurrent, forget=1."""
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        with torch.no_grad():
            # PyTorch/Keras gate order is input, forget, candidate, output.
            self.lstm.bias_ih_l0[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
        # PyTorch carries two biases; freezing the redundant one retains
        # the Keras single-bias parameterization during optimization.
        self.lstm.bias_hh_l0.requires_grad_(False)
        nn.init.xavier_uniform_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    def forward(self, history: Tensor) -> dict[str, Tensor]:
        """Forecast from only the last ``context`` observed returns [B,T]."""
        if history.ndim != 2 or history.shape[1] < self.context:
            raise ValueError("history must have shape [B,T] with T >= context")
        if not history.is_floating_point():
            raise TypeError("history must be floating point")
        x = history[:, -self.context :].square().unsqueeze(-1)
        if self.training and self.input_dropout:
            mask = x.new_empty((x.shape[0], 1, 1)).bernoulli_(1 - self.input_dropout)
            x = x * mask / (1 - self.input_dropout)
        _, (hidden, _) = self.lstm(x)
        log_variance = self.log_variance_head(self.output_dropout(hidden[-1])).squeeze(-1)
        variance = log_variance.clamp(
            -self.log_variance_clip, self.log_variance_clip
        ).exp()
        return {
            "mean": torch.zeros_like(variance),
            "variance": variance,
            "nu": self.degrees_of_freedom.to(variance).expand_as(variance),
        }


def _terms(params: Mapping[str, Tensor], next_return: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    variance = params["variance"]
    mean = params["mean"]
    if variance.ndim != 1 or mean.shape != variance.shape or next_return.shape != variance.shape:
        raise ValueError("mean, variance and next_return must all have shape [B]")
    nu = torch.as_tensor(params["nu"], dtype=variance.dtype, device=variance.device)
    if nu.ndim != 0 and nu.shape != variance.shape:
        raise ValueError("nu must be scalar or have shape [B]")
    if not bool(torch.isfinite(nu).all()) or bool((nu <= 2).any()):
        raise ValueError("nu must be finite and greater than 2")
    return variance.clamp_min(1e-8), (next_return - mean).square(), nu


def student_t_nll(
    params: Mapping[str, Tensor],
    next_return: Tensor,
    *,
    include_constants: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """NLL for a t distribution whose *variance* is params['variance'].

    Its torch.distributions.StudentT scale is sqrt(variance*(nu-2)/nu).
    ``include_constants=False`` matches the author's fixed-nu training
    convention; use the full default NLL for probabilistic evaluation.
    """
    variance, eps2, nu = _terms(params, next_return)
    terms = 0.5 * variance.log() + 0.5 * (nu + 1) * torch.log1p(
        eps2 / (variance * (nu - 2))
    )
    if include_constants:
        terms = terms + (
            torch.lgamma(nu / 2)
            - torch.lgamma((nu + 1) / 2)
            + 0.5 * (math.pi * (nu - 2)).log()
        )
    if reduction == "none":
        return terms
    if reduction == "sum":
        return terms.sum()
    if reduction == "mean":
        return terms.mean()
    raise ValueError("reduction must be 'none', 'sum' or 'mean'")


def loss(
    params: Mapping[str, Tensor],
    next_return: Tensor,
    weight: float = 0.5,
    *,
    sse_scale: float = 1.0,
    student_scale: float = 1.0,
) -> Tensor:
    """Author fixed-nu objective: (1-weight)*SSE/sse_scale + weight*NLL/student_scale.

    Inputs are next-period RETURNS, not variance or realized-volatility
    observations. Squaring the centered return produces the author's
    noisy daily variance proxy. Default scales of one reproduce the raw
    author objective. Optional scales must be frozen using TRAIN only.
    Do not compare different weights' hybrid validation-loss values:
    choose settings with a common predeclared validation score instead.
    """
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    if not math.isfinite(sse_scale) or sse_scale <= 0:
        raise ValueError("sse_scale must be finite and positive")
    if not math.isfinite(student_scale) or student_scale <= 0:
        raise ValueError("student_scale must be finite and positive")
    variance, eps2, _ = _terms(params, next_return)
    sse = (variance - eps2).square().mean() / sse_scale
    nll = student_t_nll(params, next_return, include_constants=False) / student_scale
    return (1 - weight) * sse + weight * nll


@dataclass(frozen=True)
class LossScales:
    sse_scale: float
    student_scale: float
    reference_variance: float
    reference_nu: float


@torch.no_grad()
def compute_loss_scales(training_returns: Tensor, reference_nu: float = 5.0) -> LossScales:
    """Author scale normalization fitted exclusively to a supplied TRAIN prefix.

    Returns must already use the runner's TRAIN mean/RMS. Caller owns the
    chronological cutoff. Scale floors for a constant/zero series are a
    declared numerical adaptation to the author's unguarded estimator.
    """
    if training_returns.ndim != 1 or training_returns.numel() < 2:
        raise ValueError("training_returns must be a 1D TRAIN vector with at least two entries")
    if not training_returns.is_floating_point() or not bool(torch.isfinite(training_returns).all()):
        raise ValueError("training_returns must be finite floating-point returns")
    if not math.isfinite(reference_nu) or reference_nu <= 2:
        raise ValueError("reference_nu must be greater than 2")
    eps2 = training_returns.detach().double().square()
    constant_variance = eps2.mean().clamp_min(1e-8)
    t_reference = 0.5 * constant_variance.log() + 0.5 * (reference_nu + 1) * torch.log1p(
        eps2 / (constant_variance * (reference_nu - 2))
    )
    return LossScales(
        float(eps2.var(unbiased=True).clamp_min(1e-8)),
        float(t_reference.mean().abs().clamp_min(1e-8)),
        float(constant_variance),
        float(reference_nu),
    )


# Convenient descriptive aliases for callers, without separate behavior.
ArchLSTM = ARCHLSTM
hybrid_loss = loss
