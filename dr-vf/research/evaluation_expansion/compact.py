"""Frozen nine-parameter likelihood recursion from the development attribution.

This adapter performs no estimation or model selection. Each named panel market uses
coefficients and transformations fitted before 2011 on its other geographic groups.
The archived scalar filter is reused so the extension preserves its actual clipping,
EWMA initialization, zero-return handling and output timing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research.attribution import gain_surgery as original
from research.prior_adaptation.experiment import GROUPS

ROOT = Path(__file__).resolve().parents[2]
FIT_ROOT = ROOT / "research/outputs/attribution_gain_v3"
SPECIFICATION = "state_dep_gain"
VARIANT = "likelihood"
FIT_CUTOFF = "2011-01-01"
BURN = 500
PARAMETERS = ("const", "log_h", "score", "absz", "signed", "d1", "d2",
              "score_x_d1", "score_x_d2")
NORMALIZATION = ("d1_mean", "d1_sd", "d2_mean", "d2_sd", "logh_mean")


def dependency_paths() -> tuple[Path, ...]:
    """Files the outer protocol should hash before any new evaluation."""
    return (
        Path(__file__).resolve(),
        ROOT / "research/attribution/gain_surgery.py",
        ROOT / "research/prior_adaptation/experiment.py",
        ROOT / "research/prior_adaptation/model.py",
        ROOT / "volpfn_prior.py",
        ROOT / "research/attribution/PROTOCOL.md",
        ROOT / "research/attribution/extract.py",
        ROOT / "research/outputs/prior_adaptation_mechanism_v1/scores.csv",
        FIT_ROOT / "folds.json", FIT_ROOT / "protocol.json", FIT_ROOT / "status.json",
    )


INPUT_FILES = list(dependency_paths())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_inputs(protocol: dict) -> list[dict]:
    """Resolve the archived absolute script path without depending on that host."""
    records = []
    for recorded, expected in protocol["inputs"].items():
        if Path(recorded).name == "gain_surgery.py":
            local = ROOT / "research/attribution/gain_surgery.py"
        elif Path(recorded).is_absolute():
            raise ValueError(f"Unexpected absolute input in compact protocol: {recorded}")
        else:
            local = ROOT / recorded
        current = _sha(local)
        records.append({"archived_path": recorded, "current_path": str(local.relative_to(ROOT)),
                        "archived_sha256": expected, "current_sha256": current,
                        "matches_archived": current == expected})
    # The archived predictor's executable definition must match. The extraction
    # driver is historical provenance only: this adapter never invokes it.
    executable = next(item for item in records if item["current_path"] ==
                      "research/attribution/gain_surgery.py")
    if not executable["matches_archived"]:
        raise ValueError("The original compact recursion source changed since the archived fit")
    return records


def _saved_fit(market_name: str) -> tuple[str, dict, dict, list[dict]]:
    groups = [group for group, names in GROUPS.items() if market_name in names]
    if len(groups) != 1:
        raise ValueError(f"No unique archived held-out fold for market {market_name!r}")
    group = groups[0]
    protocol = json.loads((FIT_ROOT / "protocol.json").read_text())
    folds = json.loads((FIT_ROOT / "folds.json").read_text())
    if protocol.get("fit_window") != (
        "pre-2011 history of the fitting groups; evaluation on held-out groups 2011-2014"
    ):
        raise ValueError("Unexpected compact fitting period")
    if (protocol.get("burn") != BURN or protocol.get("nu_score") != 8.0
            or tuple(protocol["specifications"][SPECIFICATION]) != PARAMETERS
            or tuple(original.SPECS[SPECIFICATION]) != PARAMETERS
            or original.BURN != BURN):
        raise ValueError("Compact specification differs from the archived protocol")
    if set(folds) != set(GROUPS):
        raise ValueError("Saved compact folds do not match the geographic groups")
    fit = folds[group][SPECIFICATION]
    if fit.get("free_parameters") != 9 or not fit.get("likelihood_converged"):
        raise ValueError(f"Missing converged nine-parameter likelihood fit for {group}")
    coefficients = fit["coefficients_likelihood"]
    norm = fit["normalisation"]
    if set(coefficients) != set(PARAMETERS) or set(norm) != set(NORMALIZATION):
        raise ValueError("Unexpected archived compact coefficient or transformation keys")
    if not all(np.isfinite(float(value)) for value in [*coefficients.values(), *norm.values()]):
        raise ValueError("Nonfinite archived compact fit")
    if norm["d1_sd"] <= 0 or norm["d2_sd"] <= 0:
        raise ValueError("Nonpositive archived transformation scale")
    return group, fit, protocol, _historical_inputs(protocol)


def predict(market_name: str, market_dict: dict) -> dict:
    """Return a full-length pre-observation variance path and serializable audit.

    ``r`` is the one-dimensional percentage-return history, ``date`` its strictly
    increasing ISO date array. At least 100 observations are needed to reproduce
    initialization. Entries before position 500 or before the fit cutoff are NaN;
    their retrospective initialization values must never be treated as forecasts.
    No forecast at a valid date t uses r[t] or any later return.
    """
    group, fit, protocol, provenance = _saved_fit(market_name)
    r = np.asarray(market_dict["r"], dtype=float)
    date = np.asarray(market_dict["date"])
    if r.ndim != 1 or date.ndim != 1 or len(r) != len(date) or len(r) < 100:
        raise ValueError("Compact forecast needs aligned one-dimensional histories of length >=100")
    date = date.astype("U10")
    if (not np.isfinite(r).all() or np.any(date[1:] <= date[:-1])
            or np.any(np.char.str_len(date) != 10)):
        raise ValueError("Compact forecast needs finite returns and strictly increasing ISO dates")
    try:
        if np.isnat(date.astype("datetime64[D]")).any():
            raise ValueError("NaT date")
    except ValueError as error:
        raise ValueError("Compact forecast dates must be valid ISO dates") from error
    if not np.isfinite(r * r).all():
        raise ValueError("Squared returns overflow")

    norm = fit["normalisation"]
    beta = np.asarray([fit["coefficients_likelihood"][key] for key in PARAMETERS], dtype=float)
    valid = (np.arange(len(r)) >= BURN) & (date >= FIT_CUTOFF)
    prepared = original.prepare({"r": r}, norm, valid)
    _, _, r2, absr, signs, d1, d2, initial_logh, _ = prepared
    h = original.run_filter(r2, absr, signs, d1, d2, beta, norm["logh_mean"], initial_logh)
    if not np.isfinite(h[valid]).all() or np.any(h[valid] <= 0):
        raise RuntimeError(f"Invalid frozen compact forecast for {market_name}")
    h[~valid] = np.nan
    first = np.flatnonzero(valid)
    training_groups = sorted(set(GROUPS) - {group})
    audit = {
        "model": "compact9_likelihood", "specification": SPECIFICATION, "variant": VARIANT,
        "parameters": dict(fit["coefficients_likelihood"]),
        "normalisation": dict(norm), "target_market": market_name, "held_out_group": group,
        "training_groups": training_groups,
        "training_markets": sorted(name for name_group in training_groups for name in GROUPS[name_group]),
        "fit_window": protocol["fit_window"], "fit_dates_strictly_before": FIT_CUTOFF,
        "refitting_in_this_evaluation": False,
        "likelihood_converged": bool(fit["likelihood_converged"]),
        "likelihood_evaluations": int(fit["likelihood_evaluations"]),
        "likelihood_mean_nll": float(fit["likelihood_mean_nll"]),
        "first_forecast_date": str(date[first[0]]) if len(first) else None,
        "forecast_observations": int(valid.sum()), "input_observations": len(r),
        "return_units": "percentage log returns", "mean": "zero working mean",
        "initialization": {"observations": 100, "start": str(date[0]), "end": str(date[99]),
                           "variance": float(np.exp(initial_logh)), "minimum_valid_position": BURN},
        "timing": "h[t] is emitted before consuming r[t]; all coefficients and transforms are frozen",
        "historical_inputs": provenance,
        "fit_artifact_sha256": {str((FIT_ROOT / name).relative_to(ROOT)): _sha(FIT_ROOT / name)
                                for name in ("folds.json", "protocol.json", "status.json")},
        "limitations": [
            "The formula and fitting design were developed using the broader development research sample.",
            "Excluded geographic groups were not used for coefficient fitting, but this is not a new-market research holdout.",
            "The later market period has prior exposure elsewhere in the research programme.",
            "Convergence flags document attained fits, not global likelihood optimality.",
            "Current historical-source mismatches are recorded; the historical extraction driver is not rerun.",
        ],
    }
    json.dumps(audit, allow_nan=False)
    return {"variance": h, "audit": audit}
