"""Pinned corrected VolPFN finals on strict, resetting 512-return contexts.

The adapter returns the predictive second moment, not centred variance, a
quantile, or an average of seed predictions. No training or market calibration
occurs here. The command-line entry point uses three synthetic contexts only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from experiments.seed_replication import causal


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_LENGTH = 512
SEEDS = (8101, 8102, 8103, 8104, 8105)
DEFAULT_REGISTRY = ROOT / "experiments/outputs/return_only_benchmark_setup_v1/checkpoints/CHECKPOINT_REGISTRY.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_checkpoint(seed: int, registry_path: Path = DEFAULT_REGISTRY) -> tuple[Path, dict]:
    registry_path = Path(registry_path)
    if sha256(registry_path) != registry_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Checkpoint registry hash mismatch")
    registry = json.loads(registry_path.read_text())
    if registry["seeds"] != list(SEEDS) or registry["arm"] != "broad":
        raise ValueError("Wrong fixed checkpoint allocation")
    matches = [entry for entry in registry["entries"]
               if entry["arm"] == "broad" and entry["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("Seed must identify exactly one of the five corrected broad finals")
    entry = matches[0]
    checkpoint = ROOT / entry["checkpoint"]
    if sha256(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("Pinned checkpoint hash mismatch")
    return checkpoint, entry


def configure_device(device: str = "cpu", threads: int = 2) -> torch.device:
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Specify cpu, cuda or mps; no automatic device substitution")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device is unavailable")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class VolPFNAdapter:
    """One corrected seed; input shape is N x 512 raw percentage returns."""

    def __init__(self, seed: int = 8101, device: str = "cpu", threads: int = 2,
                 registry_path: Path = DEFAULT_REGISTRY):
        self.device = configure_device(device, threads)
        self.checkpoint, self.entry = pinned_checkpoint(seed, registry_path)
        self.seed = seed
        self.model = causal.load_model(str(self.checkpoint), self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, contexts: np.ndarray, batch_size: int = 8) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=np.float64)
        if contexts.ndim != 2 or contexts.shape[1] != CONTEXT_LENGTH or not len(contexts):
            raise ValueError("contexts must be a nonempty N x 512 array")
        if not np.isfinite(contexts).all():
            raise ValueError("Nonfinite observed returns")
        if (not isinstance(batch_size, (int, np.integer))
                or isinstance(batch_size, (bool, np.bool_)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        forecasts = []
        for start in range(0, len(contexts), batch_size):
            # feature_batch resets scale, EWMA state and position independently
            # for each row, never borrowing observations from another context.
            features = causal.feature_batch(contexts[start:start + batch_size])
            output = self.model(torch.from_numpy(features["feats"]).to(self.device),
                                torch.from_numpy(features["base"]).to(self.device))
            final = {key: output[key][:, -1].float().cpu().numpy()
                     for key in ("logw", "loc", "logscale", "nu")}
            weights = np.exp(final["logw"])
            unit = features["output_scale"][:, -1]
            location = final["loc"].astype(np.float64) * unit[:, None]
            scale = np.exp(final["logscale"].astype(np.float64)) * unit[:, None]
            nu = final["nu"].astype(np.float64)
            if np.any(nu <= 2) or not np.isfinite(nu).all():
                raise FloatingPointError("Mixture component lacks a finite second moment")
            if not np.allclose(weights.sum(axis=1), 1, rtol=2e-6, atol=2e-7):
                raise FloatingPointError("Mixture weights do not sum to one")
            moment = np.sum(weights * (location**2 + scale**2 * nu / (nu - 2)), axis=1)
            if np.any(moment <= 0) or not np.isfinite(moment).all():
                raise FloatingPointError("Nonpositive or nonfinite predictive second moment")
            forecasts.append(moment)
        return np.concatenate(forecasts)

    def predict(self, past: np.ndarray) -> float:
        past = np.asarray(past, dtype=np.float64)
        if past.ndim != 1:
            raise ValueError("past must contain exactly 512 preceding returns")
        return float(self.predict_batch(past[None, :], batch_size=1)[0])


def synthetic_check(seed: int = 8101, device: str = "cpu", threads: int = 2) -> dict:
    """Three independent synthetic contexts; never reads market histories."""
    # Independent reference implementation also computes return quantiles.
    # Their computation is excluded from the adapter's measured batch time.
    from experiments.programme_closure.confirm_benchmark import native_prediction

    rng = np.random.default_rng(2026092401)
    contexts = rng.standard_t(8, size=(3, CONTEXT_LENGTH)) * np.array([0.4, 1.0, 2.0])[:, None]
    contexts[0, :15] = 0
    tick = time.perf_counter()
    adapter = VolPFNAdapter(seed=seed, device=device, threads=threads)
    _synchronize(device)
    load_seconds = time.perf_counter() - tick
    # Exclude the first kernel/allocation warmup from steady batch timing.
    adapter.predict(contexts[0])
    _synchronize(device)
    tick = time.perf_counter()
    batch = adapter.predict_batch(contexts, batch_size=3)
    _synchronize(device)
    batch_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    singles = adapter.predict_batch(contexts, batch_size=1)
    _synchronize(device)
    single_seconds = time.perf_counter() - tick
    reference = np.array([native_prediction(adapter.model, past, adapter.device)["h"] for past in contexts])
    # Float32 attention can differ in the last bits with batch shape; the final
    # moment reduction here is float64 while the historical reference is mixed.
    np.testing.assert_allclose(batch, singles, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(batch, reference, rtol=2e-5, atol=1e-7)
    permuted = adapter.predict_batch(contexts[[2, 0, 1]], batch_size=3)
    np.testing.assert_allclose(permuted, batch[[2, 0, 1]], rtol=2e-5, atol=1e-7)
    for invalid in (contexts[:, :-1], contexts[:, :0], np.full_like(contexts, np.nan)):
        try:
            adapter.predict_batch(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid context accepted")
    return dict(status="synthetic_check_passed", created_utc=datetime.now(timezone.utc).isoformat(),
                seed=seed, device=device, threads=threads, context_length=CONTEXT_LENGTH,
                contexts=3, market_data_used=False, training_performed=False,
                checkpoint_sha256=adapter.entry["checkpoint_sha256"],
                adapter_sha256=sha256(Path(__file__)), torch_version=torch.__version__,
                numpy_version=np.__version__, load_seconds=load_seconds,
                batch_three_seconds=batch_seconds, sequential_three_seconds=single_seconds,
                batch_seconds_per_context=batch_seconds / 3,
                max_batch_single_absolute_difference=float(np.max(np.abs(batch - singles))),
                max_batch_reference_absolute_difference=float(np.max(np.abs(batch - reference))),
                rtol=2e-5, atol=1e-7,
                checks=["native_reference_moment", "batch_single_agreement", "row_permutation_independence",
                        "strict_context_length", "reject_nonfinite"],
                timing_scope="Three synthetic contexts on one seed; loading excluded; not a full-run runtime guarantee")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic-only VolPFN 512-context timing and verification")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=8101)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise FileExistsError("Refusing to overwrite a timing attempt")
    result = synthetic_check(args.seed, args.device, args.threads)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
