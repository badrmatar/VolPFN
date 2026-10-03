"""Forecast observed returns with the preprocessing declared by a checkpoint.

This entry point checks checkpoint metadata and selects the declared
preprocessing. Legacy checkpoints require explicit opt-in.
It consumes returns, not prices: the caller must construct the intended target
(the dissertation uses percentage open-to-close log returns) before forecasting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import legacy as V
from . import inference as causal


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, device="cpu", *, allow_legacy=False):
    """Reject unknown preprocessing; legacy inference requires an explicit opt-in."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"cfg", "state_dict"}.issubset(checkpoint):
        raise ValueError("Not a VolPFN checkpoint with cfg and state_dict")
    if "causal_spec" in checkpoint or "normalization_spec" in checkpoint:
        if (checkpoint.get("causal_spec") != causal.CAUSAL_SPEC
                or checkpoint.get("normalization_spec") != causal.NORMALIZATION_SPEC):
            raise ValueError("Unsupported or inconsistent checkpoint preprocessing metadata")
        interface = causal.CAUSAL_SPEC
    else:
        if not allow_legacy:
            raise ValueError("Checkpoint has no corrected causal identity; use --allow-legacy only for retained legacy weights")
        interface = "legacy_observed_first_100_rms_window"
    cfg = checkpoint["cfg"]
    if (not isinstance(cfg, dict) or set(cfg) != {"d", "layers", "heads", "K"}
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in cfg.values())
            or cfg["d"] % cfg["heads"]):
        raise ValueError("Invalid VolPFN architecture configuration")
    if not checkpoint["state_dict"] or any(not torch.isfinite(v).all() for v in checkpoint["state_dict"].values()):
        raise ValueError("Checkpoint parameters must be finite")
    model = V.VolPFN(**cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(V.pick_device(device))
    if interface == causal.CAUSAL_SPEC:
        model.causal_spec = interface
    identity = dict(checkpoint=str(Path(path).resolve()), checkpoint_sha256=file_sha256(path),
                    preprocessing=interface, config=cfg, step=checkpoint.get("step"),
                    seed=checkpoint.get("seed"), device=str(next(model.parameters()).device))
    return model, identity


def forecast(path, returns, *, device="cpu", allow_legacy=False, L=1024,
             stride=None, min_history=100, batch_windows=4, levels=(.975, .99)):
    model, identity = load_checkpoint(path, device, allow_legacy=allow_legacy)
    predictor = causal.predict_series if identity["preprocessing"] == causal.CAUSAL_SPEC else V.predict_series
    result = predictor(model, returns, L=L, stride=stride, min_history=min_history,
                       batch_windows=batch_windows, levels=levels)
    identity.update(context=L, stride=L//2 if stride is None else stride,
                    min_history=min_history, batch_windows=batch_windows, levels=list(levels),
                    scale_output="second_moment = E[r^2|history]; predictive_variance = second_moment - predictive_mean^2",
                    target="Caller-supplied observed returns; no price conversion or future normalization")
    return result, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True, help="CSV with a return column in the desired forecast units")
    parser.add_argument("--col", default="return")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--min-history", type=int, default=100)
    parser.add_argument("--batch-windows", type=int, default=4)
    parser.add_argument("--levels", type=float, nargs="*", default=[.975, .99])
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    output = Path(args.out)
    provenance = output.with_suffix(output.suffix + ".provenance.json")
    if output.exists() or provenance.exists():
        raise FileExistsError("Refusing to overwrite forecast or provenance output")
    frame = pd.read_csv(args.data)
    if args.col not in frame or args.date_col not in frame:
        raise ValueError("CSV must contain the named date and return columns")
    dates = pd.to_datetime(frame[args.date_col], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be present, unique, and increasing")
    returns = pd.to_numeric(frame[args.col], errors="raise").to_numpy(float)
    if not np.isfinite(returns).all():
        raise ValueError("Returns must be finite; rows are never silently dropped")
    if any(not 0 < level < 1 for level in args.levels):
        raise ValueError("Risk levels must lie strictly between zero and one")
    prediction, identity = forecast(args.ckpt, returns, device=args.device,
        allow_legacy=args.allow_legacy, L=args.L, stride=args.stride,
        min_history=args.min_history, batch_windows=args.batch_windows, levels=tuple(args.levels))
    result = pd.DataFrame({"date": frame[args.date_col], "return": returns,
        "second_moment": prediction["h"], "predictive_mean": prediction["mean"],
        "predictive_variance": prediction["h"] - prediction["mean"]**2,
        "logscore": prediction["logscore"],
        **{key: value for key, value in prediction.items() if key.startswith(("var", "es"))}})
    identity.update(data=str(Path(args.data).resolve()), data_sha256=file_sha256(args.data),
                    return_column=args.col, date_column=args.date_col, input_rows=len(frame),
                    forecast_rows=int(np.isfinite(prediction["h"]).sum()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        result.to_csv(stream, index=False)
    with provenance.open("x") as stream:
        json.dump(identity, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {identity['forecast_rows']} forecasts using {identity['preprocessing']}: {output}")


if __name__ == "__main__":
    main()
