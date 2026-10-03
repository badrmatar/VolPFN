"""Finite, causal per-market daily-return tuning/training/evaluation commands.

Run with ``python -m research.experiment_programme_20260930.daily_runner``.
All final data stages require a frozen programme manifest and new output folder.
``smoke`` uses synthetic data, small networks and two epochs; no market scores.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from . import arch_lstm, garch_lstm, ginn

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "research/data/oxfordman_barde/calendar_full_v1"
DEFAULT_ROSTER = ROOT / "research/outputs/matched_prior_design_inventory_v1/inventory.json"
MODELS = ("garch_lstm", "ginn", "arch_lstm")
CONTEXTS = {"garch_lstm": 126, "ginn": 90, "arch_lstm": 1}
FINAL_SEEDS = (8101, 8102, 8103, 8104, 8105)
TUNING_SEED = 717311
TRAIN_END, VALIDATION_END, EVALUATION_END = "2010-12-31", "2014-12-31", "2022-02-25"
MAX_EPOCHS, BATCH_SIZE, PATIENCE = 50, 128, 8
COMMON_NU = 8.0
EXPECTED_ROSTER = (
    "AEX", "AORD", "BFX", "BSESN", "BVSP", "DJI", "FCHI", "FTSE",
    "GDAXI", "GSPTSE", "HSI", "IBEX", "IXIC", "KS11", "KSE", "MXX",
    "N225", "NSEI", "OMXC20", "OMXHPI", "OMXSPI", "OSEAX", "RUT",
    "SMSI", "SPX", "SSEC", "SSMI", "STOXX50E",
)


def sha_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_values(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def hash_dates(dates: np.ndarray) -> str:
    return hashlib.sha256(("\n".join(map(str, dates)) + "\n").encode()).hexdigest()


def hash_dated_values(dates: np.ndarray, values: np.ndarray) -> str:
    return hashlib.sha256((hash_dates(dates) + ":" + hash_values(values)).encode()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def new_output(path: Path | str) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_receipt(device: str) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no silent hardware switch")
    packages = {}
    for name in ("arch", "scipy"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "requested_device": device,
            "packages": packages,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "autocast": torch.is_autocast_enabled(),
            "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn": torch.backends.cudnn.allow_tf32} if device == "cuda" else "not applicable"}


def configure_backend(device: str) -> None:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA unavailable; no silent hardware switch")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha_file(path) for path in (
        Path(__file__), Path(garch_lstm.__file__), Path(ginn.__file__), Path(arch_lstm.__file__),
    )}


def registered_grid(model: str) -> list[dict[str, Any]]:
    """Finite paper-aware grid, fixed before validation outcomes are read."""
    if model not in MODELS:
        raise ValueError("unknown model")
    if model == "ginn":
        settings = [{"lambda_observed": x} for x in (0., .01, .1)]
    elif model == "arch_lstm":
        settings = [{"weight": x} for x in (0., .5, 1.)]
    else:
        settings = [{}]  # Eq.7 has no paper regularization lambda to invent.
    constructors = {
        "arch_lstm": {"context": 1, "hidden_size": 32, "dropout": 0., "nu": 3., "log_variance_clip": 20.},
        "ginn": {"hidden_size": 256, "num_layers": 3, "dropout": .2, "nu": 5., "variance_floor": 1e-8},
        "garch_lstm": {"omega": .03, "alpha": .07, "beta": .90, "initial_variance": 1.,
                       "modulation": .1, "nu": 5., "learn_nu": False, "variance_floor": 1e-8},
    }
    return [{"model": model, "context": CONTEXTS[model], "learning_rate": lr,
             "model_kwargs": dict(constructors[model]),
             "optimizer": "AdamW" if model == "ginn" else "Adam",
             "optimizer_kwargs": {"betas": [.9, .999], "eps": 1e-8,
                "weight_decay": .01 if model == "ginn" else 0., "amsgrad": False,
                "maximize": False, "foreach": False, "capturable": False,
                "differentiable": False, "fused": False},
             "loss_kwargs": setting, "gradient_clip": 5.0}
            for lr in (0.0003, 0.001) for setting in settings]


@dataclass(frozen=True)
class TrainingScaler:
    center: float
    rms: float
    count: int
    training_sha256: str
    input_units: str = "percent_log_return"

    @classmethod
    def fit(cls, training_returns: np.ndarray) -> "TrainingScaler":
        values = np.asarray(training_returns, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("scaler requires at least two finite TRAIN returns")
        center = float(values.mean())
        rms = float(np.sqrt(np.mean((values - center) ** 2)))
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError("training RMS must be finite and positive")
        return cls(center, rms, len(values), hash_values(values))

    def transform(self, returns: np.ndarray) -> np.ndarray:
        return (np.asarray(returns, dtype=np.float64) - self.center) / self.rms

    def raw_params(self, mean: Tensor, variance: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw_mean = self.center + self.rms * mean
        raw_variance = self.rms ** 2 * variance
        return raw_mean, raw_variance, raw_variance + raw_mean.square()


@dataclass
class Series:
    market: str
    dates: np.ndarray
    returns: np.ndarray  # percent; input conversion occurs only in load_series
    raw_file_sha256: str
    inventory: dict[str, str]

    def indices(self, split: str, context: int) -> np.ndarray:
        if split == "train":
            mask = self.dates <= TRAIN_END
        elif split == "validation":
            mask = (self.dates > TRAIN_END) & (self.dates <= VALIDATION_END)
        elif split == "evaluation":
            mask = (self.dates > VALIDATION_END) & (self.dates <= EVALUATION_END)
        else:
            raise ValueError("unknown split")
        return np.flatnonzero(mask & (np.arange(len(self.dates)) >= context))

    def training_returns(self) -> np.ndarray:
        return self.returns[self.dates <= TRAIN_END]

    def lock(self) -> dict[str, Any]:
        return {"market": self.market, "raw_file_sha256": self.raw_file_sha256,
                "date_sha256": hash_dates(self.dates), "percent_return_sha256": hash_values(self.returns),
                "roster_and_data_inventory": self.inventory, "input_conversion": "decimal * 100 exactly once",
                "target": "next observed daily open-to-close log return in percent",
                "counts": {s: int(len(self.indices(s, 0)))
                           for s in ("train", "validation", "evaluation")}}


def validate_series(series: Series) -> None:
    if (series.dates.ndim != 1 or series.returns.shape != series.dates.shape
            or len(series.dates) == 0 or not np.isfinite(series.returns).all()):
        raise ValueError("dated return source must be finite and aligned")
    parsed = series.dates.astype("datetime64[D]")
    if np.isnat(parsed).any() or np.any(parsed[1:] <= parsed[:-1]):
        raise ValueError("source dates must be valid, unique and strictly increasing")
    if not np.array_equal(parsed.astype(str), series.dates.astype(str)):
        raise ValueError("source dates must be canonical YYYY-MM-DD")
    for split in ("train", "validation", "evaluation"):
        if len(series.indices(split, 0)) == 0:
            raise ValueError(f"source has no {split} observations; no market may be silently dropped")


def load_series(market: str, data_root: Path | str = DEFAULT_DATA,
                roster_path: Path | str = DEFAULT_ROSTER) -> Series:
    data_root, roster_path = Path(data_root).resolve(), Path(roster_path).resolve()
    roster = json.loads(roster_path.read_text())["retained_index_population"]
    if tuple(roster) != EXPECTED_ROSTER:
        raise ValueError("fixed 28-market roster changed")
    if market not in roster:
        raise ValueError("market is outside the fixed roster")
    inventory = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha_file(path) for path in (
        roster_path, data_root / "protocol.json", data_root / "audit.json")}
    protocol = json.loads((data_root / "protocol.json").read_text())
    if "open-to-close" not in protocol.get("scope", ""):
        raise ValueError("unexpected source return-session definition")
    path = data_root / (market + ".npz")
    with np.load(path, allow_pickle=False) as source:
        # Deliberately do not load rv5/rk_parzen. Input source returns are decimal.
        dates = np.asarray(source["date"], dtype="U10").copy()
        returns = np.asarray(source["returns"], dtype=np.float64).copy() * 100.0
    series = Series(market, dates, returns, sha_file(path), inventory)
    validate_series(series)
    return series


def manifest_lock(path: Path | str | None, *, smoke: bool = False) -> dict[str, str] | None:
    if path is None:
        if smoke:
            return None
        raise ValueError("a frozen --manifest is required for final market-data stages")
    manifest = Path(path).resolve()
    if manifest.is_dir():
        manifest = manifest / "manifest.json"
    value = json.loads(manifest.read_text())
    if value.get("name") == "volpfn_experiment_programme_20260930":
        from . import programme
        value = programme.verify(manifest, root=ROOT)
        if (value["seeds"] != list(FINAL_SEEDS) or value["tuning_seeds"] != [TUNING_SEED]
                or value["data"]["train_end"] != TRAIN_END
                or value["data"]["validation_end"] != VALIDATION_END
                or value["data"]["evaluation_end"] != EVALUATION_END):
            raise ValueError("runner seeds or chronology differ from the registered programme")
        for name in MODELS:
            registered = value["daily_training"]["grid"][name]
            configs = registered_grid(name)
            if sorted(set(c["learning_rate"] for c in configs)) != sorted(registered["learning_rate"]):
                raise ValueError("learning-rate grid differs from programme")
            if registered["context"] != [CONTEXTS[name]]:
                raise ValueError("context differs from programme")
            key = "lambda_observed" if name == "ginn" else "weight"
            if name != "garch_lstm" and [c["loss_kwargs"][key] for c in configs[:3]] != registered["weight"]:
                raise ValueError("loss-weight grid differs from programme")
    elif value.get("scope") != "unit_test_fixture" or not smoke:
        raise ValueError("final runs require a verified programme manifest")
    sidecar = manifest.with_suffix(".sha256")
    digest = sha_file(manifest)
    if sidecar.exists() and sidecar.read_text().strip() != digest:
        raise ValueError("manifest SHA256 sidecar mismatch")
    return {"path": str(manifest), "sha256": digest}


def verify_lock(lock: dict[str, Any], series: Series,
                manifest: Path | str | None, *, smoke: bool = False) -> None:
    if lock["series"] != series.lock():
        raise ValueError("dated data, units, roster or raw-source inventory changed")
    if lock["sources"] != source_hashes():
        raise ValueError("model or runner source changed since the tuning freeze")
    current_manifest = manifest_lock(manifest, smoke=smoke)
    expected_hash = lock["manifest"]["sha256"] if lock["manifest"] is not None else None
    current_hash = current_manifest["sha256"] if current_manifest is not None else None
    if expected_hash != current_hash:
        raise ValueError("programme manifest changed since the tuning freeze")


def verify_mask_labels(series: Series, locked_manifest: dict[str, str] | None) -> None:
    """Bind every market date/index/target to the programme's frozen masks."""
    if locked_manifest is None:
        return
    path = Path(locked_manifest["path"])
    manifest = json.loads(path.read_text())
    if manifest.get("scope") == "unit_test_fixture":
        return
    if series.market not in manifest["population"]:
        raise ValueError("market is outside the programme population")
    relative_source = str(Path(manifest["data"]["directory"]) / (series.market + ".npz"))
    if manifest["data_hashes"].get(relative_source) != series.raw_file_sha256:
        raise ValueError("raw market source differs from the programme's frozen file")
    for split in ("train", "validation", "evaluation"):
        with (path.parent / manifest["mask_files"][split]["path"]).open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["market"] == series.market]
        indices = series.indices(split, 0)
        if ([int(row["row_index"]) for row in rows] != indices.tolist()
                or [row["date"] for row in rows] != series.dates[indices].tolist()
                or not np.array_equal(np.asarray([float(row["return"]) for row in rows]), series.returns[indices])):
            raise ValueError("market target/date/index differs from frozen " + split + " mask")


def seal_result(path: Path, value: dict[str, Any]) -> None:
    save_json(path, value)
    path.with_suffix(".sha256").write_text(sha_file(path) + "\n")


def read_result(path: Path) -> dict[str, Any]:
    if path.with_suffix(".sha256").read_text().strip() != sha_file(path):
        raise ValueError("stage result changed: " + str(path))
    return json.loads(path.read_text())


def restored_teacher(path: Path) -> ginn.FrozenTeacher | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    for name in ("ar_coefficients", "train_residuals", "train_teacher_variances"):
        state[name] = tuple(state[name])
    return ginn.FrozenTeacher(**state)


@dataclass
class Windows:
    inputs: Tensor
    targets: Tensor
    mean: Tensor
    teacher_variance: Tensor | None
    indices: np.ndarray


def build_windows(series: Series, scaler: TrainingScaler, model: str, split: str,
                  teacher: ginn.FrozenTeacher | None = None) -> Windows:
    """Input ending at i-1 predicts return i; GINN AR filtering never resets."""
    indices = series.indices(split, CONTEXTS[model])
    if len(indices) < (2 if split == "train" else 1):
        raise ValueError(f"insufficient {split} targets for {model}; no silent omission")
    standardized = scaler.transform(series.returns)
    last_observed_count = int(indices[-1])
    prefix = torch.tensor(standardized[:last_observed_count][None, :], dtype=torch.float64)
    means = np.zeros(len(indices), dtype=np.float64)
    teacher_labels = None
    if model == "ginn":
        if teacher is None or teacher.train_count != scaler.count:
            raise ValueError("GINN requires its completed TRAIN-only teacher fit")
        if teacher.train_sha256 != hash_values(standardized[:scaler.count]):
            raise ValueError("teacher training-return hash differs from the scaler's TRAIN prefix")
        features = ginn.residual_history(prefix, teacher)[0].square().numpy()
        path = ginn.teacher_variances(prefix, teacher)[0].numpy()
        teacher_labels = torch.tensor(path[indices], dtype=torch.float32)
        p = len(teacher.ar_coefficients)
        means = teacher.ar_intercept + sum(
            coefficient * standardized[indices - lag]
            for lag, coefficient in enumerate(teacher.ar_coefficients, start=1))
        if (indices < p).any():
            raise ValueError("target precedes available AR lags")
    else:
        features = standardized[:last_observed_count]
    input_indices = indices[:, None] + np.arange(-CONTEXTS[model], 0)
    if (series.dates[input_indices[:, -1]] >= series.dates[indices]).any():
        raise ValueError("history includes its target date")
    return Windows(torch.tensor(features[input_indices], dtype=torch.float32),
                   torch.tensor(standardized[indices], dtype=torch.float32),
                   torch.tensor(means, dtype=torch.float32), teacher_labels, indices)


def make_model(config: dict[str, Any]) -> torch.nn.Module:
    constructors = {"garch_lstm": garch_lstm.GARCHLSTM, "ginn": ginn.GINN,
                    "arch_lstm": arch_lstm.ARCHLSTM}
    return constructors[config["model"]](**config["model_kwargs"])


def forecast_batch(model: torch.nn.Module, windows: Windows, positions: Tensor,
                   device: str, scaler: TrainingScaler) -> dict[str, Tensor]:
    output = model(windows.inputs[positions].to(device))
    if isinstance(model, ginn.GINN):
        output["mean"] = windows.mean[positions].to(device)
    mean, variance, second_moment = scaler.raw_params(output["mean"].double(), output["variance"].double())
    nu = torch.as_tensor(output["nu"], device=mean.device, dtype=mean.dtype).expand_as(mean)
    for name, value in (("mean", mean), ("variance", variance), ("second_moment", second_moment), ("nu", nu)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite {name}; no clipping or market removal")
    if (variance <= 0).any() or (second_moment <= 0).any() or (nu <= 2).any():
        raise FloatingPointError("invalid conditional density; no silent fallback")
    return {"standardized": output, "mean": mean, "variance": variance,
            "second_moment": second_moment, "nu": nu}


@torch.no_grad()
def validation_score(model: torch.nn.Module, windows: Windows, series: Series,
                     scaler: TrainingScaler, device: str) -> float:
    model.eval()
    totals = []
    for begin in range(0, len(windows.indices), BATCH_SIZE):
        positions = torch.arange(begin, min(begin + BATCH_SIZE, len(windows.indices)))
        output = forecast_batch(model, windows, positions, device, scaler)
        target = torch.tensor(series.returns[windows.indices[positions.numpy()]],
                              dtype=torch.float64, device=device)
        loss = target.square() / output["second_moment"] + output["second_moment"].log()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("validation QLIKE is not finite")
        totals.append(loss.cpu())
    return float(torch.cat(totals).mean())


def train_one(series: Series, scaler: TrainingScaler, config: dict[str, Any], seed: int,
              teacher: ginn.FrozenTeacher | None, *, device: str = "cpu",
              epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
              batch_size: int = BATCH_SIZE) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not 1 <= epochs <= MAX_EPOCHS or not 1 <= patience <= PATIENCE or not 2 <= batch_size <= BATCH_SIZE:
        raise ValueError("training budgets must be positive and within the declared caps")
    configure_backend(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    training = build_windows(series, scaler, config["model"], "train", teacher)
    validation = build_windows(series, scaler, config["model"], "validation", teacher)
    model = make_model(config).to(device)
    optimizer_class = torch.optim.AdamW if config["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer_options = dict(config["optimizer_kwargs"])
    optimizer_options["betas"] = tuple(optimizer_options["betas"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"], **optimizer_options)
    loss_scales = (arch_lstm.compute_loss_scales(
        torch.tensor(scaler.transform(series.training_returns())), reference_nu=3.)
        if config["model"] == "arch_lstm" else None)
    best_score, best_state, best_epoch, stale = math.inf, None, 0, 0
    events, updates, skipped = [], 0, 0
    numerical_events = {"arch_head_cap": 0, "arch_loss_variance_floor": 0, "gradient_clip": 0}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(training.indices))
        epoch_losses = []
        for begin in range(0, len(order), batch_size):
            positions = order[begin:begin + batch_size]
            if config["model"] == "ginn" and len(positions) == 1:
                skipped += 1
                continue
            targets = training.targets[positions].to(device)
            output = model(training.inputs[positions].to(device))
            if config["model"] == "ginn":
                loss = ginn.ginn_loss(output["variance"], targets,
                    training.mean[positions].to(device), training.teacher_variance[positions].to(device),
                    **config["loss_kwargs"])
            elif config["model"] == "arch_lstm":
                numerical_events["arch_loss_variance_floor"] += int((output["variance"] < 1e-8).sum())
                numerical_events["arch_head_cap"] += int(((output["variance"] <= math.exp(-20.) * 1.00001)
                                                         | (output["variance"] >= math.exp(20.) * .99999)).sum())
                loss = arch_lstm.loss(output, targets, **config["loss_kwargs"],
                    sse_scale=loss_scales.sse_scale, student_scale=loss_scales.student_scale)
            else:
                loss = garch_lstm.student_t_nll(targets, **output)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError("nonfinite gradient; no silent continuation")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient norm")
            numerical_events["gradient_clip"] += int(norm > config["gradient_clip"])
            optimizer.step()
            updates += 1
            epoch_losses.append(float(loss.detach().cpu()))
        score = validation_score(model, validation, series, scaler, device)
        improved = score < best_score
        events.append({"epoch": epoch, "training_loss": float(np.mean(epoch_losses)),
                       "validation_raw_r2_qlike": score, "selected_so_far": improved})
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("no accepted finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    status = {"status": "complete_frozen_after_validation", "seed": seed, "epochs_run": epoch,
              "epoch_cap": epochs, "patience": patience, "batch_size": batch_size,
              "best_epoch": best_epoch, "best_validation_raw_r2_qlike": best_score,
              "optimizer_updates": updates, "singleton_train_windows_skipped": skipped,
              "training_target_count": len(training.indices), "validation_target_count": len(validation.indices),
              "training_target_hash": hash_dated_values(series.dates[training.indices], series.returns[training.indices]),
              "validation_target_hash": hash_dated_values(series.dates[validation.indices], series.returns[validation.indices]),
              "elapsed_seconds": time.perf_counter() - started, "events": events,
              "optimizer": config["optimizer"], "optimizer_kwargs": {"lr": config["learning_rate"], **config["optimizer_kwargs"]},
              "runtime": runtime_receipt(device),
              "numerical_events": numerical_events,
              "loss_scales": asdict(loss_scales) if loss_scales is not None else None,
              "evaluation_predictions_or_scores_read": False}
    return model, status


def save_fit(output: Path, model: torch.nn.Module, config: dict[str, Any],
             scaler: TrainingScaler, teacher: ginn.FrozenTeacher | None, status: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": model.state_dict(), "config": config, "scaler": asdict(scaler)}, output / "checkpoint.pt")
    save_json(output / "config.json", config)
    save_json(output / "scaler.json", asdict(scaler))
    if teacher is not None:
        save_json(output / "teacher.json", asdict(teacher))
    metadata = (model.metadata() if hasattr(model, "metadata") else
                {"model": "arch_lstm_daily_adaptation", "author_commit": arch_lstm.AUTHOR_COMMIT,
                 "native_nu": float(model.degrees_of_freedom), "model_kwargs": config["model_kwargs"]})
    save_json(output / "model_metadata.json", metadata)
    save_json(output / "training_status.json", status)
    files = {path.name: sha_file(path) for path in output.iterdir() if path.is_file()}
    save_json(output / "artifact_hashes.json", files)
    return files


def stage_lock(series: Series, manifest: Path | str | None, *, smoke: bool = False) -> dict[str, Any]:
    locked_manifest = manifest_lock(manifest, smoke=smoke)
    verify_mask_labels(series, locked_manifest)
    return {"series": series.lock(), "sources": source_hashes(),
            "manifest": locked_manifest,
            "tuning_seed": TUNING_SEED, "final_seeds": list(FINAL_SEEDS),
            "selection": "per-market mean validation raw r² QLIKE only; equal-market final aggregation",
            "comparison": "per-market fitted baselines versus pooled frozen VolPFN; not architecture isolation",
            "common_adapter": "zero-mean variance-scaled t8",
            "split_dates": {"train_end": TRAIN_END, "validation_end": VALIDATION_END,
                            "evaluation_end": EVALUATION_END},
            "architecture_contexts": CONTEXTS, "historical_sample_previously_inspected": True}


def tune(series: Series, model_name: str, output: Path | str, *, manifest: Path | str | None,
         device: str = "cpu", smoke: bool = False, epochs: int = MAX_EPOCHS) -> dict[str, Any]:
    lock = stage_lock(series, manifest, smoke=smoke)
    output = new_output(output)
    save_json(output / "status.json", {"status": "running", "stage": "tune", "started": now()})
    try:
        grid = registered_grid(model_name)
        if smoke:
            grid = grid[:1]
            grid[0]["model_kwargs"].update({"hidden_size": 8}) if model_name != "garch_lstm" else None
            if model_name == "ginn":
                grid[0]["model_kwargs"].update({"num_layers": 1, "dropout": 0.})
        save_json(output / "freeze.json", {"lock": lock, "grid": grid,
            "budgets": {"epochs": epochs, "batch": BATCH_SIZE, "patience": PATIENCE}, "smoke": smoke})
        scaler = TrainingScaler.fit(series.training_returns())
        save_json(output / "scaler.json", asdict(scaler))
        teacher = ginn.fit_teacher(scaler.transform(series.training_returns())) if model_name == "ginn" else None
        if teacher is not None:
            save_json(output / "teacher.json", asdict(teacher))
        attempted = []
        for index, config in enumerate(grid):
            fitted, status = train_one(series, scaler, config, TUNING_SEED, teacher,
                                       device=device, epochs=epochs)
            hashes = save_fit(output / f"config_{index:02d}", fitted, config, scaler, teacher, status)
            attempted.append({"config_index": index, "config": config,
                "validation_raw_r2_qlike": status["best_validation_raw_r2_qlike"], "artifact_hashes": hashes})
            save_json(output / "status.json", {"status": "running", "stage": "tune",
                                               "completed_configurations": index + 1})
        selected = min(attempted, key=lambda row: (row["validation_raw_r2_qlike"], row["config_index"]))
        result = {"status": "complete", "stage": "tune", "model": model_name, "market": series.market,
                  "lock": lock, "grid": grid, "attempted": attempted, "selected": selected,
                  "scaler": asdict(scaler), "smoke": smoke, "epoch_cap": epochs,
                  "evaluation_scores": "not computed or consulted"}
        seal_result(output / "result.json", result)
        save_json(output / "status.json", {"status": "complete", "stage": "tune"})
        return result
    except Exception as error:
        save_json(output / "status.json", {"status": "failed", "stage": "tune", "error": repr(error),
                                           "market_retained_as_failure": series.market})
        raise


def verify_artifacts(directory: Path, sealed_hashes: dict[str, str] | None = None) -> None:
    hashes = json.loads((directory / "artifact_hashes.json").read_text())
    if sealed_hashes is not None and sealed_hashes != hashes:
        raise ValueError("artifact hash inventory differs from sealed stage result")
    for filename, expected in hashes.items():
        if sha_file(directory / filename) != expected:
            raise ValueError("fitted artifact changed: " + str(directory / filename))


def train(series: Series, tuning: Path | str, output: Path | str, *, manifest: Path | str | None,
          device: str = "cpu", smoke: bool = False) -> dict[str, Any]:
    tuning = Path(tuning).resolve()
    selected = read_result(tuning / "result.json")
    if selected["status"] != "complete" or bool(selected["smoke"]) != smoke:
        raise ValueError("tuning attempt is incomplete or has incompatible smoke scope")
    verify_lock(selected["lock"], series, manifest, smoke=smoke)
    configuration = selected["selected"]["config"]
    fitted_tuning = tuning / f"config_{selected['selected']['config_index']:02d}"
    verify_artifacts(fitted_tuning, selected["selected"]["artifact_hashes"])
    if configuration != json.loads((fitted_tuning / "config.json").read_text()):
        raise ValueError("selected configuration differs from accepted tuning artifact")
    teacher = restored_teacher(fitted_tuning / "teacher.json")
    scaler = TrainingScaler(**selected["scaler"])
    seeds = FINAL_SEEDS[:1] if smoke else FINAL_SEEDS
    output = new_output(output)
    save_json(output / "status.json", {"status": "running", "stage": "train"})
    try:
        save_json(output / "freeze.json", {"lock": selected["lock"], "config": configuration,
            "tuning_result_sha256": sha_file(tuning / "result.json"), "seeds": list(seeds), "smoke": smoke})
        attempts = []
        for seed in seeds:
            fitted, status = train_one(series, scaler, configuration, seed, teacher, device=device,
                                       epochs=selected["epoch_cap"])
            hashes = save_fit(output / f"seed_{seed}", fitted, configuration, scaler, teacher, status)
            attempts.append({"seed": seed, "status": status["status"], "artifact_hashes": hashes,
                             "selected_epoch": status["best_epoch"]})
            save_json(output / "status.json", {"status": "running", "stage": "train", "completed_seeds": [a["seed"] for a in attempts]})
        result = {"status": "complete", "stage": "train", "lock": selected["lock"],
                  "model": selected["model"], "market": series.market, "config": configuration,
                  "seeds": list(seeds), "attempts": attempts, "smoke": smoke,
                  "tuning_result_sha256": sha_file(tuning / "result.json"),
                  "evaluation_predictions_or_scores": "not computed or consulted"}
        seal_result(output / "result.json", result)
        save_json(output / "status.json", {"status": "complete", "stage": "train", "completed_seeds": list(seeds)})
        return result
    except Exception as error:
        save_json(output / "status.json", {"status": "failed", "stage": "train", "error": repr(error),
                                           "market_retained_as_failure": series.market})
        raise


@torch.no_grad()
def evaluate(series: Series, training: Path | str, output: Path | str, *, manifest: Path | str | None,
             device: str = "cpu", smoke: bool = False) -> dict[str, Any]:
    training = Path(training).resolve()
    trained = read_result(training / "result.json")
    if trained["status"] != "complete" or bool(trained["smoke"]) != smoke:
        raise ValueError("training attempt is incomplete or has incompatible smoke scope")
    if trained["seeds"] != list(FINAL_SEEDS[:1] if smoke else FINAL_SEEDS):
        raise ValueError("required seed roster is incomplete; no surviving-seed evaluation")
    verify_lock(trained["lock"], series, manifest, smoke=smoke)
    output = new_output(output)
    save_json(output / "status.json", {"status": "running", "stage": "evaluate"})
    try:
        configure_backend(device)
        runtime = runtime_receipt(device)
        summaries, forecast_hashes = [], {}
        for seed in trained["seeds"]:
            directory = training / f"seed_{seed}"
            expected_artifacts = next(row["artifact_hashes"] for row in trained["attempts"] if row["seed"] == seed)
            verify_artifacts(directory, expected_artifacts)
            checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
            model = make_model(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            scaler = TrainingScaler(**checkpoint["scaler"])
            teacher = restored_teacher(directory / "teacher.json")
            windows = build_windows(series, scaler, trained["model"], "evaluation", teacher)
            context = CONTEXTS[trained["model"]]
            rows = []
            for begin in range(0, len(windows.indices), BATCH_SIZE):
                positions = torch.arange(begin, min(begin + BATCH_SIZE, len(windows.indices)))
                prediction = forecast_batch(model, windows, positions, device, scaler)
                indices = windows.indices[positions.numpy()]
                target = torch.tensor(series.returns[indices], dtype=torch.float64, device=device)
                h = prediction["second_moment"]
                native = garch_lstm.student_t_nll(target, prediction["mean"], prediction["variance"],
                                                prediction["nu"], reduction="none")
                common = garch_lstm.student_t_nll(target, torch.zeros_like(target), h, COMMON_NU, reduction="none")
                qlike = target.square() / h + h.log()
                values = {key: prediction[key].cpu().numpy() for key in ("mean", "variance", "second_moment", "nu")}
                for j, index in enumerate(indices):
                    context_dates, context_returns = series.dates[index-context:index], series.returns[index-context:index]
                    feature = windows.inputs[int(positions[j])].numpy()
                    rows.append({"model": trained["model"], "market": series.market, "seed": seed,
                        "index": int(index), "date": str(series.dates[index]), "context_count": context,
                        "context_start": str(context_dates[0]), "context_end": str(context_dates[-1]),
                        "context_sha256": hash_dated_values(context_dates, context_returns),
                        "feature_sha256": hash_values(feature), "target_sha256": hash_dated_values(series.dates[index:index+1], series.returns[index:index+1]),
                        "raw_source_sha256": series.raw_file_sha256, "raw_return": float(series.returns[index]),
                        "return": float(series.returns[index]),
                        "raw_r2": float(series.returns[index] ** 2),
                        **{key: float(value[j]) for key, value in values.items()},
                        "qlike": float(qlike[j].cpu()), "native_nll": float(native[j].cpu()),
                        "common_t8_nll": float(common[j].cpu()),
                        "native_variance_cap_event": int(trained["model"] == "arch_lstm" and
                            (float(prediction["standardized"]["variance"][j]) <= math.exp(-20.) * 1.00001 or
                             float(prediction["standardized"]["variance"][j]) >= math.exp(20.) * .99999))})
            path = output / f"forecasts_{seed}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            forecast_hashes[path.name] = sha_file(path)
            summaries.append({"seed": seed, "count": len(rows),
                **{metric: float(np.mean([row[metric] for row in rows])) for metric in ("qlike", "native_nll", "common_t8_nll")},
                "native_variance_cap_events": sum(row["native_variance_cap_event"] for row in rows),
                "target_sha256": hash_dated_values(series.dates[windows.indices], series.returns[windows.indices])})
        result = {"status": "complete", "stage": "evaluate", "model": trained["model"],
                  "market": series.market, "lock": trained["lock"], "smoke": smoke, "seeds": trained["seeds"],
                  "summaries": summaries, "forecast_sha256": forecast_hashes,
                  "training_result_sha256": sha_file(training / "result.json"),
                  "runtime": runtime,
                  "score_interpretation": "per-seed scores; not an ensemble density",
                  "target_mask": "all finite source daily open-to-close returns, including zero",
                  "native_density": "raw fitted mean, central variance and native nu",
                  "common_density": "imposed zero-mean variance-scaled Student t8 using second moment"}
        seal_result(output / "result.json", result)
        save_json(output / "status.json", {"status": "complete", "stage": "evaluate"})
        return result
    except Exception as error:
        save_json(output / "status.json", {"status": "failed", "stage": "evaluate", "error": repr(error),
                                           "market_retained_as_failure": series.market})
        raise


def synthetic_smoke_series() -> Series:
    rng = np.random.default_rng(717399)
    dates = np.concatenate([np.datetime_as_string(np.datetime64(start) + np.arange(count))
                            for start, count in (("2008-01-01", 180), ("2011-01-01", 24), ("2015-01-01", 20))])
    returns, variance, previous = [], 1., 0.
    for _ in dates:
        variance = .06 + .1 * previous ** 2 + .84 * variance
        previous = math.sqrt(variance) * rng.normal()
        returns.append(.05 + previous)
    series = Series("SYNTHETIC", dates.astype("U10"), np.asarray(returns), "synthetic",
                    {"scope": "synthetic bounded engineering smoke only"})
    validate_series(series)
    return series


def smoke(output: Path | str, *, device: str = "cpu") -> dict[str, Any]:
    output = new_output(output)
    series = synthetic_smoke_series()
    attempts = []
    for model_name in MODELS:
        tune(series, model_name, output / f"{model_name}_tune", manifest=None, device=device, smoke=True, epochs=2)
        train(series, output / f"{model_name}_tune", output / f"{model_name}_train", manifest=None, device=device, smoke=True)
        result = evaluate(series, output / f"{model_name}_train", output / f"{model_name}_evaluate", manifest=None, device=device, smoke=True)
        attempts.append({"model": model_name, "status": result["status"], "count": result["summaries"][0]["count"]})
    result = {"status": "complete", "scope": "synthetic smoke, not forecasting evidence",
              "models": attempts, "epochs_per_fit": 2, "one_configuration_and_seed_per_model": True}
    save_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("tune", "train", "evaluate", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--threads", type=int, default=2)
        if name != "smoke":
            command.add_argument("--market", choices=EXPECTED_ROSTER, required=True)
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
            command.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
        if name == "tune":
            command.add_argument("--model", choices=MODELS, required=True)
        elif name == "train":
            command.add_argument("--tuning", type=Path, required=True)
        elif name == "evaluate":
            command.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "smoke":
        result = smoke(args.out, device=args.device)
    else:
        series = load_series(args.market, args.data_root, args.roster)
        if args.command == "tune":
            result = tune(series, args.model, args.out, manifest=args.manifest, device=args.device)
        elif args.command == "train":
            result = train(series, args.tuning, args.out, manifest=args.manifest, device=args.device)
        else:
            result = evaluate(series, args.training, args.out, manifest=args.manifest, device=args.device)
    print(json.dumps({"status": result["status"], "stage": args.command, "out": str(args.out)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
