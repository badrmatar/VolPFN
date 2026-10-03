"""Pinned, causal realized-kernel point forecasts from IBM TTM R2.1.

The primary ``variance`` recipe feeds RK directly to the pretrained MSE point
head. This is a declared benchmark adaptation: Brini feeds sqrt(realized
variance). ``sqrt`` retains that recipe as a separately named diagnostic; its
squared point is not E[RK]. Neither recipe supplies native return moments.

No weights are downloaded and importing this module does not load a model.
Use one TTMRealizedAdapter per process, then forecast_context or forecast_series.
All results are RAW. The existing 72-origin calibration is fitted separately.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "ibm-granite/granite-timeseries-ttm-r2"
REVISION = "3442406f106122f414a11f72b4ebfe4878bec89d"
BRANCH = "512-96-ft-r2.1"
CONTEXT_LENGTH = 512
DEFAULT_SNAPSHOT = (ROOT / "research/outputs/return_only_benchmark_setup_v1"
                    / "foundation_cache/ttm" / REVISION)
ASSET_HASHES = {
    "README.md": "e20234ff1a9eec054fd643d5ef7a4683857c8b24c7b00b466a9f55b6e22728ff",
    "config.json": "9e437e55d74815fb77e2c89fd674c587224bab2156ae142bca64456217501ace",
    "generation_config.json": "ddcaccb39673f986431d6b9ea2064011e4d04784d7daf9d3aa9db93940bca960",
    "model.safetensors": "37fa184887f260cda022d3ae9f1c642fdba579d0aefeabdf751f83f45383a761",
}


def verify_snapshot(snapshot=DEFAULT_SNAPSHOT):
    """Check the exact local inference files before loading any checkpoint."""
    snapshot = Path(snapshot).resolve()
    for name, expected in ASSET_HASHES.items():
        actual = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned TTM asset mismatch: {name}")
    return dict(repository=REPOSITORY, revision=REVISION, branch=BRANCH,
                snapshot=str(snapshot), asset_sha256=dict(ASSET_HASHES))


def causal_fill_rk(rk):
    """Forward-fill invalid INPUTS only; leading missing values remain NaN.

    Every filled position depends exclusively on indices at or below itself.
    Original RK must be retained separately for the target-only scoring mask.
    """
    values = np.asarray(rk, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("RK history must be one-dimensional")
    valid = np.isfinite(values) & (values > 0)
    indices = np.maximum.accumulate(np.where(valid, np.arange(len(values)), -1))
    filled = np.full(len(values), np.nan, dtype=np.float64)
    observed = indices >= 0
    filled[observed] = values[indices[observed]]
    return filled, ~valid


def _context(past_rk):
    filled, missing = causal_fill_rk(past_rk)
    if len(filled) < CONTEXT_LENGTH:
        raise ValueError("At least 512 preceding observed market sessions are required")
    context = filled[-CONTEXT_LENGTH:]
    if not np.isfinite(context).all() or np.any(context <= 0):
        raise ValueError("No past-only positive RK fill is available for the context")
    return context, int(missing[-CONTEXT_LENGTH:].sum())


class TTMRealizedAdapter:
    """Univariate RK adapter; CPU float32 is the verified initial runtime.

    ``model`` is a dependency-injection hook for synthetic contract tests.
    Production callers omit it, so exact local asset hashes are mandatory.
    """
    def __init__(self, *, transform="variance", device="cpu",
                 snapshot=DEFAULT_SNAPSHOT, model=None):
        if transform not in ("variance", "sqrt"):
            raise ValueError("Choose the declared variance or sqrt recipe")
        self.transform = transform
        self.device = device
        self.snapshot = Path(snapshot)
        self.model = model
        self._loaded = False
        self.identity = None

    @property
    def model_id(self):
        return "ttm_rk" if self.transform == "variance" else "ttm_sqrtrk_diagnostic"

    def load(self):
        if self._loaded:
            return self.model
        if self.model is None:
            self.identity = verify_snapshot(self.snapshot)
            from tsfm_public import TinyTimeMixerForPrediction
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                str(self.snapshot.resolve()), local_files_only=True)
            cfg = self.model.config
            required = dict(context_length=512, prediction_length=96, loss="mse",
                            num_input_channels=1, resolution_prefix_tuning=True,
                            scaling="std")
            if any(getattr(cfg, k, None) != v for k, v in required.items()):
                raise ValueError("Unexpected configuration in pinned TTM checkpoint")
        else:
            self.identity = dict(source="injected_model_for_contract_testing")
        self.model = self.model.to(self.device).float().eval()
        self._loaded = True
        return self.model

    def forecast_batch(self, contexts, *, imputed_counts=None):
        """Forecast already causally filled [batch,512] RK contexts.

        Scaling is independently fitted to each row. No cross-series scaling,
        clipping, return squaring, stochastic sampling or calibration occurs.
        """
        import torch
        x = np.asarray(contexts, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != CONTEXT_LENGTH or len(x) < 1:
            raise ValueError("Expected a nonempty [batch,512] RK matrix")
        if not np.isfinite(x).all() or np.any(x <= 0):
            raise ValueError("Batch inputs must be finite positive causally filled RK")
        counts = np.zeros(len(x), dtype=int) if imputed_counts is None else np.asarray(imputed_counts)
        if (counts.shape != (len(x),) or not np.isfinite(counts).all()
                or np.any(counts < 0) or np.any(counts > 512)
                or np.any(counts != counts.astype(int))):
            raise ValueError("Invalid imputed-context counts")
        native = x if self.transform == "variance" else np.sqrt(x)
        centre = native.mean(axis=1)
        sd = native.std(axis=1, ddof=0)
        scale = np.where(sd > 1e-12, sd, 1.0)
        if not np.isfinite(centre).all() or not np.isfinite(scale).all():
            raise FloatingPointError("Nonfinite context scaling")
        values = torch.as_tensor((native - centre[:, None]) / scale[:, None],
                                 dtype=torch.float32, device=self.device).unsqueeze(-1)
        token = torch.full((len(x), 1), 8, dtype=torch.long, device=self.device)
        model = self.load()
        with torch.inference_mode():
            output = model(past_values=values, freq_token=token).prediction_outputs
        if tuple(output.shape) != (len(x), 96, 1):
            raise RuntimeError(f"Unexpected TTM output shape: {tuple(output.shape)}")
        points = output[:, 0, 0].detach().cpu().numpy().astype(np.float64) * scale + centre
        if not np.isfinite(points).all():
            raise FloatingPointError("Nonfinite TTM point; no fallback or dropped origin")
        rows = []
        for i, raw in enumerate(points):
            epsilon = 1e-8 * max(float(x[i].mean()), 1e-12)
            native_floor = epsilon if self.transform == "variance" else math.sqrt(epsilon)
            point = max(float(raw), native_floor)
            with np.errstate(over="ignore"):
                h = point if self.transform == "variance" else float(np.square(point))
                # An invalid negative SD is not transformed into positive variance.
                raw_h = float(raw) if self.transform == "variance" else (
                    float(np.square(raw)) if raw >= 0 else None)
            if not math.isfinite(h) or h <= 0:
                raise FloatingPointError("Invalid variance-scale TTM point")
            rows.append(dict(model=self.model_id, seed=None, mean=0.0,
                variance=h, second_moment=h, h=h, raw_h=raw_h,
                raw_native_point=float(raw), native_scale=self.transform,
                floor_applied=bool(raw < native_floor), output_floor=epsilon,
                raw_nonpositive=bool(raw <= 0), substituted=False,
                imputed_context_count=int(counts[i]),
                context_length=512, scaling_centre=float(centre[i]),
                scaling_scale=float(scale[i]), daily_frequency_token=8,
                repository=REPOSITORY, revision=REVISION, branch=BRANCH,
                checkpoint_verified="asset_sha256" in self.identity,
                units="percent_squared", information_group="R",
                moment_semantics="point-to-proxy: mean=0; variance=second_moment; no native return density",
                forecast_functional=("MSE point on RK, first horizon" if self.transform == "variance"
                    else "squared MSE point on sqrt(RK), first horizon; not E[RK]")))
        return rows

    def forecast_context(self, past_rk):
        """Forecast the next session from past RK only (percent-squared units)."""
        context, count = _context(past_rk)
        return self.forecast_batch(context[None, :], imputed_counts=[count])[0]

    def forecast_series(self, rk, origins, *, dates=None, market=None,
                        phase="evaluation", batch_size=64):
        """Return origin rows using RK[t-512:t], never target RK[t].

        ``origins`` contains integer target indices or mappings with ``index``
        and optional metadata (market/date/phase/targets). Targets are never
        modified. Invalid RK inputs are filled causally, with counts retained.
        """
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("Positive integer batch_size required")
        filled, missing = causal_fill_rk(rk)
        if dates is not None:
            dates = np.asarray(dates).astype("U10")
            if len(dates) != len(filled) or np.any(dates[1:] <= dates[:-1]):
                raise ValueError("Dates must be unique and chronological on the full history")
        descriptors = []
        for origin in origins:
            row = dict(origin) if isinstance(origin, Mapping) else {"index": origin}
            t = row["index"]
            if isinstance(t, bool) or int(t) != t or not 512 <= t < len(filled):
                raise ValueError("Each target index needs 512 past sessions and an observed target session")
            t = int(t)
            row["index"] = t
            if dates is not None:
                if "date" in row and row["date"] != str(dates[t]):
                    raise ValueError("Origin date and history index differ")
                row["date"] = str(dates[t])
            if market is not None:
                if "market" in row and row["market"] != market:
                    raise ValueError("Origin market differs")
                row["market"] = market
            row.setdefault("phase", phase)
            descriptors.append(row)
        if len({row["index"] for row in descriptors}) != len(descriptors):
            raise ValueError("Duplicate origin index")
        results = []
        for start in range(0, len(descriptors), int(batch_size)):
            chunk = descriptors[start:start + int(batch_size)]
            x = np.stack([filled[r["index"] - 512:r["index"]] for r in chunk])
            counts = [int(missing[r["index"] - 512:r["index"]].sum()) for r in chunk]
            predictions = self.forecast_batch(x, imputed_counts=counts)
            results.extend(dict(row, **prediction) for row, prediction in zip(chunk, predictions))
        return results


def forecast_context(past_rk, *, adapter=None, transform="variance"):
    """Convenience interface; pass a persistent adapter to reuse loaded weights."""
    return (adapter or TTMRealizedAdapter(transform=transform)).forecast_context(past_rk)


def calibration_factor(rows):
    """Existing past-only scalar: equal-market mean(R2 / raw M2), 72 origins.

    These must be the runner's exact frozen 72 calibration rows. This helper
    checks phase, date bounds, markets, count and uniqueness; the runner remains
    responsible for matching their keys to the hashed origin manifest.
    """
    rows = list(rows)
    counts = Counter(r["market"] for r in rows)
    if counts != Counter(dict(FTSE=24, N225=24, SPX=24)):
        raise ValueError("Require exactly 24 origins each in FTSE, N225 and SPX")
    if len({(r["market"], r["date"]) for r in rows}) != 72:
        raise ValueError("Duplicate calibration origin")
    ratios = {m: [] for m in counts}
    for row in rows:
        d = date.fromisoformat(row["date"])
        if row.get("phase") != "calibration" or not date(2012, 1, 4) <= d <= date(2013, 12, 2):
            raise ValueError("Calibration must use the frozen past-only development period")
        y, h = float(row["r2"]), float(row["second_moment"])
        if not math.isfinite(y) or y < 0 or not math.isfinite(h) or h <= 0:
            raise ValueError("Invalid calibration target or raw forecast")
        ratios[row["market"]].append(y / h)
    result = float(np.mean([np.mean(values) for values in ratios.values()]))
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Calibration scalar must be positive and finite")
    return result
