"""Lazy, pinned daily-return adapters; no model downloads on import.

All callers provide percentage-unit returns, not realised measures. Each forecast
uses exactly the most recent 512 returns. TTM predicts their square directly;
Sundial's point functional is its sample mean before the shared positive floor.
Native return densities and artificial prediction intervals are not supplied.
"""
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, Optional

import numpy as np


CONTEXT_LENGTH = 512
TTM_REPOSITORY = "ibm-granite/granite-timeseries-ttm-r2"
TTM_REVISION = "3442406f106122f414a11f72b4ebfe4878bec89d"
TTM_BRANCH = "512-96-ft-r2.1"
SUNDIAL_REPOSITORY = "thuml/sundial-base-128m"
SUNDIAL_REVISION = "3212e42564493f520593e5414af4367fc4b49226"


@dataclass(frozen=True)
class PointForecast:
    """Second moment in squared percentage-return units, with raw diagnostics."""

    h: float
    raw_h: float
    diagnostics: Dict[str, Any]


def squared_context(past_returns):
    returns = np.asarray(past_returns, dtype=np.float64)
    if returns.ndim != 1 or len(returns) < CONTEXT_LENGTH:
        raise ValueError("A one-dimensional history of at least 512 returns is required")
    returns = returns[-CONTEXT_LENGTH:]
    if not np.isfinite(returns).all():
        raise ValueError("Nonfinite return in available context")
    with np.errstate(over="ignore"):
        squares = np.square(returns)
    if not np.isfinite(squares).all():
        raise ValueError("Squared-return overflow")
    return squares


def floor_point(raw_h, context_r2, **diagnostics):
    """Floor the final point, never individual generative samples.

    The context mean is fixed at issuance. Nonfinite outputs fail instead of
    becoming forecasts. Retain the raw point and all floor events in reports.
    """
    raw_h = float(raw_h)
    context = np.asarray(context_r2, dtype=np.float64)
    if (context.ndim != 1 or not len(context) or
            not np.isfinite(context).all() or np.any(context < 0)):
        raise ValueError("Invalid nonnegative squared-return context")
    if not math.isfinite(raw_h):
        raise FloatingPointError("Nonfinite foundation forecast")
    floor = 1e-8 * max(float(context.mean()), 1e-12)
    return PointForecast(
        h=max(raw_h, floor), raw_h=raw_h,
        diagnostics=dict(diagnostics, output_floor=floor,
                         floor_activated=bool(raw_h < floor),
                         raw_nonpositive=bool(raw_h <= 0),
                         context_length=int(len(context)),
                         context_mean_r2=float(context.mean()),
                         units="percentage_return_squared"),
    )


def origin_seed(origin_id, base_seed=9242026):
    """Stable sample identity, independent of process hash randomisation."""
    if not isinstance(origin_id, str) or not origin_id:
        raise ValueError("Supply a stable nonempty market/date origin identifier")
    digest = hashlib.sha256(f"{int(base_seed)}:{origin_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class TTMAdapter:
    def __init__(self, device="cpu", local_files_only=True, model=None):
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.model = model

    def load(self):
        if self.model is None:
            from tsfm_public import TinyTimeMixerForPrediction
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                TTM_REPOSITORY, revision=TTM_REVISION,
                local_files_only=self.local_files_only,
            )
        self.model = self.model.to(self.device).float().eval()
        return self.model

    def predict(self, past_returns, origin_id: Optional[str] = None):
        import torch
        context = squared_context(past_returns)
        # External standard scaling, using this origin's observed context only.
        centre = float(context.mean())
        sd = float(context.std(ddof=0))
        scale = sd if sd > 1e-12 else 1.0
        values = torch.tensor((context - centre) / scale,
                              dtype=torch.float32, device=self.device)[None, :, None]
        token = torch.tensor([[8]], dtype=torch.long, device=self.device)
        model = self.load()
        with torch.inference_mode():
            output = model(past_values=values, freq_token=token).prediction_outputs
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < 1 or output.shape[2] != 1:
            raise RuntimeError(f"Unexpected TTM prediction shape: {tuple(output.shape)}")
        raw_h = float(output[0, 0, 0].detach().cpu()) * scale + centre
        return floor_point(raw_h, context, model="ttm", repository=TTM_REPOSITORY,
                          revision=TTM_REVISION, branch=TTM_BRANCH,
                          forecast_functional="MSE point output, first horizon",
                          scaling_centre=centre, scaling_scale=scale,
                          daily_frequency_token=8)


class SundialAdapter:
    def __init__(self, num_samples, device="cpu", base_seed=9242026,
                 local_files_only=True, model=None):
        if int(num_samples) != num_samples or num_samples < 1:
            raise ValueError("Choose and freeze a positive integer sample count")
        self.num_samples = int(num_samples)
        self.device = device
        self.base_seed = int(base_seed)
        self.local_files_only = bool(local_files_only)
        self.model = model

    def load(self):
        import torch
        if self.model is None:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                SUNDIAL_REPOSITORY, revision=SUNDIAL_REVISION,
                code_revision=SUNDIAL_REVISION, trust_remote_code=True,
                torch_dtype=torch.float32, local_files_only=self.local_files_only,
            )
        self.model = self.model.to(self.device).float().eval()
        return self.model

    def predict(self, past_returns, origin_id):
        import torch
        context = squared_context(past_returns)
        seed = origin_seed(origin_id, self.base_seed)
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("This seeded adapter supports tested RNG interfaces CPU/CUDA only")
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        model = self.load()
        values = torch.tensor(context, dtype=torch.float32, device=device)[None, :]
        # Preserve surrounding RNG state; per-origin identity permits resuming
        # without selecting or replacing favourable draws.
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            samples = model.generate(values, max_new_tokens=1,
                                     num_samples=self.num_samples)
        samples = samples.detach().cpu().numpy()
        expected = (1, self.num_samples, 1)
        if samples.shape != expected:
            raise RuntimeError(f"Unexpected Sundial samples: {samples.shape}, expected {expected}")
        draws = samples[0, :, 0].astype(np.float64)
        if not np.isfinite(draws).all():
            raise FloatingPointError("Nonfinite Sundial generated sample")
        raw_h = float(draws.mean())
        sample_se = float(draws.std(ddof=1) / np.sqrt(len(draws))) if len(draws) > 1 else None
        return floor_point(
            raw_h, context, model="sundial", repository=SUNDIAL_REPOSITORY,
            revision=SUNDIAL_REVISION, forecast_functional="raw sample mean, first horizon",
            num_samples=self.num_samples, sampling_seed=seed,
            negative_sample_count=int((draws < 0).sum()),
            nonpositive_sample_count=int((draws <= 0).sum()),
            sample_mean_standard_error=sample_se,
        )
