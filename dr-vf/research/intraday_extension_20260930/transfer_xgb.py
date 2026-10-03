"""Causal MTL-XGB-STD RK adaptation of Teller et al.; no target scoring.

Author core pinned at a-teller/vola-mtl 9afcd983e210a478527212d1833387bac5ad9f00.
The source-selection, tree settings and negative-prediction repair are retained.
Oxford-Man indexes, RK, full target histories and five seeds are adaptations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (8101, 8102, 8103, 8104, 8105)
AUTHOR_PIN = "9afcd983e210a478527212d1833387bac5ad9f00"
XGB_TARGET = ROOT / "research/environments/intraday_xgb_v1"
INTERNAL_SCALE = 10000.  # percent-squared RK -> author's decimal-RV * 1e8 units
DISTANCE_PERCENTILE = 50
RECENT_SESSIONS = 22
REFIT_SESSIONS = 5
AUTHOR_FILES = {
    "config.ini": "44d6b9ed6bb746dfe865b58942bab6278d4738d539b2824779b0d5738f35d64f",
    "src/models/data_handling.py": "ef8c507c30a0e3ceb95ea9a9338acc3874e480ba4255a8cb7d8c4d2397317e47",
    "src/models/model_factory.py": "16d9d74483a34aeb0f5b8816f3b19a29f590282d843f596cb980c15bb3e04fa3",
    "src/models/predictions.py": "36eab675939b3ee7d76b887ab94710be8d0f1da8873c498548f3329c5591c233",
    "src/data/raw_data_processing.py": "0f2c1f6490082d672d8f93853f8f70bbc6baf5d5d34e4653d819f6e9c15d796e",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xgb_class():
    """Use the isolated author-version package, never mutate existing runtimes."""
    if str(XGB_TARGET) not in sys.path:
        sys.path.insert(0, str(XGB_TARGET))
    import xgboost
    if xgboost.__version__ != "1.7.1":
        raise RuntimeError("The pinned XGBoost 1.7.1 dependency is required")
    return xgboost.XGBRegressor


def causal_features(risk):
    risk = np.asarray(risk, dtype=np.float64)
    valid = np.isfinite(risk) & (risk > 0)
    owner = np.maximum.accumulate(np.where(valid, np.arange(len(risk)), -1))
    filled = np.full(len(risk), np.nan)
    keep = owner >= 0
    filled[keep] = risk[owner[keep]]
    x = np.full((len(risk), 3), np.nan)
    imputations = np.zeros(len(risk), dtype=np.int64)
    if len(risk) > 22:
        window = np.lib.stride_tricks.sliding_window_view(filled[:-1], 22)
        x[22:] = np.column_stack([window[:, -1], window[:, -5:].mean(1), window.mean(1)])
        imputations[22:] = np.lib.stride_tricks.sliding_window_view(~valid[:-1], 22).sum(1)
    return x, valid, imputations


def make_item(dates, risk):
    dates, risk = np.asarray(dates, dtype="U10"), np.asarray(risk, dtype=np.float64)
    if len(dates) != len(risk) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("Source dates must be unique and increasing")
    x, valid, imputations = causal_features(risk)
    return dict(date=dates, rk=risk, x=x, valid=valid, imputations=imputations)


def load_panel(root=ROOT):
    root = Path(root).resolve()
    manifest_path = root / "research/outputs/experiment_programme_20260930/programme_design_v1/manifest.json"
    mask_path = root / "research/outputs/harmonized_leaderboard_20260930_v1/common_mask.csv"
    manifest = json.loads(manifest_path.read_text())
    mask = pd.read_csv(mask_path, dtype={"market": str, "date": str})
    if (len(mask) != 22026 or mask.market.nunique() != 28 or mask.date.nunique() != 827
            or mask.duplicated(["market", "date"]).any()
            or mask.date.min() != "2019-01-01" or mask.date.max() != "2022-02-25"):
        raise ValueError("The preserved common population changed")
    mask = mask.sort_values(["market", "date"]).reset_index(drop=True)
    hashes = {str(p): sha(p) for p in (manifest_path, mask_path)}
    panel = {}
    for market in sorted(mask.market.unique()):
        path = root / manifest["data"]["directory"] / (market + ".npz")
        digest = sha(path)
        if digest != manifest["data_hashes"][str(path.relative_to(root))]:
            raise ValueError("Authoritative archive hash drift: " + market)
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as data:
            panel[market] = make_item(data["date"], data["rk_parzen"].astype(float) * 10000.)
        wanted = mask.loc[mask.market == market]
        item = panel[market]
        indices = np.searchsorted(item["date"], wanted.date)
        if np.any(indices >= len(item["date"])) or not np.array_equal(item["date"][indices], wanted.date):
            raise ValueError("An original target key is absent: " + market)
        np.testing.assert_array_equal(indices, wanted["index"].to_numpy())
        np.testing.assert_allclose(item["rk"][indices], wanted.rk, rtol=1e-12, atol=1e-12)
    return panel, mask, hashes


def dtw_distances(candidates, recent):
    """Exact unconstrained multivariate DTW distance, vectorized across windows.

    tslearn's dtw_path minimizes the sum of squared Euclidean point distances
    over unrestricted monotone paths and returns its square root. Paths are not
    needed by the author selector, only distances. No approximation is used.
    """
    candidates, recent = np.asarray(candidates, float), np.asarray(recent, float)
    if (candidates.ndim != 3 or recent.ndim != 2
            or candidates.shape[1:] != recent.shape or not len(candidates)
            or not np.isfinite(candidates).all() or not np.isfinite(recent).all()):
        raise ValueError("Finite equal-length multivariate DTW windows required")
    n, width, _ = candidates.shape
    previous = np.full((n, width + 1), np.inf)
    previous[:, 0] = 0.
    for i in range(width):
        current = np.full_like(previous, np.inf)
        for j in range(width):
            diff = candidates[:, i, :] - recent[j]
            cost = np.einsum("ij,ij->i", diff, diff)
            current[:, j + 1] = cost + np.minimum(
                np.minimum(previous[:, j], previous[:, j + 1]), current[:, j])
        previous = current
    return np.sqrt(previous[:, -1])


def select_source_data(panel, target_market, target_index):
    own = panel[target_market]
    cutoff = str(own["date"][target_index])
    # Last 21 completed training feature rows plus the current forecast feature.
    recent = own["x"][target_index - 21:target_index + 1]
    if recent.shape != (22, 3) or not np.isfinite(recent).all():
        raise ValueError("Unfillable recent target history; do not omit target keys")
    windows, labels, source_names, label_ends, input_ends, counts = [], [], [], [], [], {}
    for market in sorted(panel):
        if market == target_market:
            continue
        item = panel[market]
        eligible = ((item["date"] < cutoff) & item["valid"]
                    & np.isfinite(item["x"]).all(1))
        rows = np.flatnonzero(eligible)
        rows = rows[len(rows) % RECENT_SESSIONS:]
        if not len(rows):
            continue
        blocks = rows.reshape(-1, RECENT_SESSIONS)
        windows.append(item["x"][blocks])
        labels.append(item["rk"][blocks])
        source_names.extend([market] * len(blocks))
        label_ends.extend(item["date"][blocks[:, -1]].tolist())
        input_ends.extend(item["date"][blocks[:, -1] - 1].tolist())
        counts[market] = len(blocks)
    if not windows:
        raise ValueError("No complete source subsequences; no substitute is permitted")
    windows, labels = np.concatenate(windows), np.concatenate(labels)
    # Common author scaling cannot change the ordering of exact DTW distances.
    distances = dtw_distances(windows, recent)
    threshold = float(np.percentile(distances, DISTANCE_PERCENTILE))
    selected = distances <= threshold
    label_ends, input_ends = np.asarray(label_ends), np.asarray(input_ends)
    source_names = np.asarray(source_names)
    if np.any(label_ends >= cutoff) or np.any(input_ends >= cutoff):
        raise AssertionError("A source label or input sees the forecast date")
    selected_counts = {m: int(np.count_nonzero(selected & (source_names == m))) for m in counts}
    receipt = {
        "cutoff_exclusive": cutoff, "recent_target_sessions": 22,
        "recent_latest_input_date": str(own["date"][target_index - 1]),
        "distance_percentile": DISTANCE_PERCENTILE, "distance_threshold_native_rk": threshold,
        "candidate_subsequences": len(distances), "selected_subsequences": int(selected.sum()),
        "candidate_counts": counts, "selected_counts": selected_counts,
        "latest_source_label_date": str(max(label_ends[selected])),
        "latest_source_input_date": str(max(input_ends[selected])),
    }
    return windows[selected].reshape(-1, 3), labels[selected].reshape(-1), receipt


def repair_predictions(prediction, historical_min):
    """Author negative repair; zero/nonfinite forecasts are rejected explicitly."""
    prediction = np.asarray(prediction, float).copy()
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite issued forecast; no repair is permitted")
    if not np.isfinite(historical_min) or historical_min <= 0:
        raise ValueError("Negative repair requires a strictly positive historical minimum")
    negative = prediction < 0
    prediction[negative] = historical_min
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise RuntimeError("Nonfinite or zero issued forecast; no clipping or dropping")
    return prediction, int(negative.sum())


def fit_models(x, y, seeds):
    if x.shape != (len(y), 3) or len(y) < 100 or not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Invalid or insufficient training labels")
    cls = xgb_class()
    x_scaled = np.ascontiguousarray(x * INTERNAL_SCALE, dtype=np.float32)
    y_scaled = np.ascontiguousarray(y * INTERNAL_SCALE, dtype=np.float32)
    models, diagnostics = {}, {}
    baseline = float(np.mean((y - y.mean()) ** 2))
    for seed in seeds:
        model = cls(verbosity=0, max_depth=5, learning_rate=0.1, subsample=0.75,
                    gamma=0.1, n_estimators=40, seed_per_iteration=True,
                    seed=int(seed), n_jobs=1)
        model.fit(x_scaled, y_scaled)
        fitted = model.predict(x_scaled).astype(np.float64) / INTERNAL_SCALE
        if not np.isfinite(fitted).all():
            raise RuntimeError("Nonfinite training prediction")
        train_mse = float(np.mean((fitted - y) ** 2))
        configuration = json.loads(model.get_booster().save_config())
        diagnostics[str(seed)] = dict(training_mse_native_rk_squared=train_mse,
            constant_mean_mse_native_rk_squared=baseline,
            ratio_to_constant_mean=train_mse / baseline if baseline > 0 else None,
            tree_count=len(model.get_booster().get_dump()),
            objective=configuration["learner"]["objective"]["name"],
            actual_seed=int(model.get_xgb_params()["seed"]), seed_per_iteration=True)
        models[int(seed)] = model
    return models, float(x_scaled[:, 0].min()), diagnostics


def forecast_market(panel, market, wanted_dates, *, seeds=SEEDS, pilot_refits=None,
                    checkpoint=None, checkpoint_refits=20):
    item = panel[market]
    wanted_dates = np.asarray(wanted_dates, dtype="U10")
    indices = np.searchsorted(item["date"], wanted_dates)
    if (np.any(indices >= len(item["date"]))
            or not np.array_equal(item["date"][indices], wanted_dates)
            or np.any(indices[1:] <= indices[:-1])):
        raise ValueError("Missing, duplicated or unordered original target keys")
    rows, fits, models, anchor, active = [], [], None, None, None
    completed = 0
    accumulated_seconds = 0.
    identity = dict(market=market, seeds=list(seeds), author_pin=AUTHOR_PIN,
                    code_sha256=sha(__file__), target_dates_sha256=hashlib.sha256(wanted_dates.tobytes()).hexdigest())
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            xgb_class()  # Register the pinned XGBoost classes before deserialization.
            state = pickle.loads(checkpoint.read_bytes())
            if state["identity"] != identity:
                raise ValueError("Checkpoint identity differs from this frozen recipe")
            rows, fits, models, anchor = (state[k] for k in ("rows", "fits", "models", "anchor"))
            completed, accumulated_seconds = state["completed"], state["elapsed_seconds"]
            historical_min = state["historical_min"]
            active = fits[-1] if fits else None
    started = time.perf_counter()
    def save_checkpoint():
        if checkpoint is None:
            return
        state = dict(identity=identity, rows=rows, fits=fits, models=models, anchor=anchor,
                     completed=completed, historical_min=historical_min,
                     elapsed_seconds=accumulated_seconds + time.perf_counter() - started)
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temporary, checkpoint)
    for position, i in enumerate(indices[completed:], start=completed):
        if anchor is None or i >= anchor + REFIT_SESSIONS:
            if pilot_refits is not None and len(fits) >= pilot_refits:
                break
            fit_started = time.perf_counter()
            sx, sy, selection = select_source_data(panel, market, int(i))
            available = ((item["date"] < item["date"][i]) & item["valid"]
                         & np.isfinite(item["x"]).all(1))
            own_rows = np.flatnonzero(available)
            x = np.concatenate([item["x"][own_rows], sx])
            y = np.concatenate([item["rk"][own_rows], sy])
            models, historical_min, diagnostics = fit_models(x, y, seeds)
            fit_end = max(str(item["date"][own_rows[-1]]), selection["latest_source_label_date"])
            if fit_end >= str(item["date"][i]):
                raise AssertionError("Training labels cross the forecast date")
            active = dict(selection, forecast_block_start=str(item["date"][i]),
                          fit_end=fit_end, own_fit_start=str(item["date"][own_rows[0]]),
                          own_fit_count=len(own_rows), source_fit_count=len(sy),
                          historical_min_native_rk=historical_min / INTERNAL_SCALE,
                          training_diagnostics=diagnostics,
                          fit_seconds=time.perf_counter() - fit_started,
                          negative_prediction_repairs=0)
            fits.append(active)
            anchor = int(i)
        if not np.isfinite(item["x"][i]).all():
            raise ValueError("Unfillable original forecast history: " + market)
        feature = np.ascontiguousarray(item["x"][i:i+1] * INTERNAL_SCALE, dtype=np.float32)
        for seed, model in models.items():
            prediction, repaired = repair_predictions(model.predict(feature), historical_min)
            active["negative_prediction_repairs"] += repaired
            latest = max(str(item["date"][i-1]), active["latest_source_input_date"])
            rows.append(dict(market=market, date=str(item["date"][i]), seed=seed,
                rk_forecast=float(prediction[0] / INTERNAL_SCALE),
                origin_date=str(item["date"][i-1]), latest_input_date=latest,
                fit_end=active["fit_end"], fit_block_start=active["forecast_block_start"],
                latest_source_label_date=active["latest_source_label_date"],
                input_imputations=int(item["imputations"][i]), repaired_negative=bool(repaired)))
        completed = position + 1
        if len(fits) % checkpoint_refits == 0 and i == anchor:
            save_checkpoint()
            print(f"{market}: refits={len(fits)} keys={completed} elapsed={accumulated_seconds + time.perf_counter() - started:.1f}s", flush=True)
    save_checkpoint()
    return dict(market=market, rows=rows, fits=fits,
                elapsed_seconds=accumulated_seconds + time.perf_counter() - started)


_PANEL = None


def initialize_worker(root):
    global _PANEL
    _PANEL, _, _ = load_panel(root)


def worker(market, dates, pilot_refits, checkpoint):
    return forecast_market(_PANEL, market, dates, pilot_refits=pilot_refits, checkpoint=checkpoint)


def run(out, *, workers=1, pilot_refits=None, pilot_markets=("SPX",), resume=False):
    if pilot_refits is not None and pilot_refits < 1:
        raise ValueError("Pilot refit count must be positive")
    panel, mask, hashes = load_panel()
    del panel
    markets = sorted(mask.market.unique()) if pilot_refits is None else list(pilot_markets)
    if not set(markets).issubset(set(mask.market)):
        raise ValueError("Unknown pilot market")
    out = Path(out)
    if resume:
        previous = json.loads((out / "DESIGN.json").read_text())
        if previous["code_sha256"] != sha(__file__) or previous["source_hashes"] != hashes:
            raise ValueError("Resume requires identical frozen code and original sources")
        if previous["markets"] != markets or previous["pilot_refits"] != pilot_refits:
            raise ValueError("Resume population differs")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "markets").mkdir()
        (out / "source").mkdir()
        for name in ("transfer_xgb.py", "test_transfer_xgb.py", "TRANSFER_XGB_ADAPTATION.md"):
            shutil.copyfile(Path(__file__).parent / name, out / "source" / name)
    design = dict(model="MTL-XGB-STD RK adaptation", author_pin=AUTHOR_PIN,
        author_file_sha256=AUTHOR_FILES, target="next own-session RK in percent-squared units",
        source_markets="all other27indexes; label date strictly before forecast block start",
        features="own prior1/5/22-session RK means; causal carry only in inputs",
        target_training="full observed positive completed history, never imputed labels",
        dtw="exact unconstrained multivariate Euclidean DTW; nonoverlapping22-row subsequences",
        distance_percentile=50, seeds=list(SEEDS), author_seed=256,
        refit_sessions=5, internal_scale=INTERNAL_SCALE, xgboost_version="1.7.1",
        tree_settings=dict(max_depth=5, learning_rate=.1, subsample=.75, gamma=.1,
                           n_estimators=40, seed_per_iteration=True, n_jobs=1),
        prediction_policy="negative=>positive training dailyRKminimum; zero/nonfinite=>rejectrun",
        pilot_refits=pilot_refits, markets=markets, workers=workers, source_hashes=hashes,
        code_sha256=sha(__file__), python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        scoring="none; no evaluation outcomes used for tuning or stopping")
    design["source_freeze_sha256"] = {p.name: sha(p) for p in sorted((out / "source").iterdir())}
    design["checkpoint"] = "atomic per20refits and at completion; pinned models, selection and issued rows"
    if not resume:
        (out / "DESIGN.json").write_text(json.dumps(design, indent=2, allow_nan=False))
    started = time.perf_counter()
    all_rows, receipts = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(ROOT,)) as pool:
        jobs = {pool.submit(worker, m, mask.loc[mask.market == m, "date"].to_numpy(), pilot_refits,
                            out / "markets" / (m + "_checkpoint.pkl")): m for m in markets}
        for future in as_completed(jobs):
            result = future.result()
            rows = pd.DataFrame(result.pop("rows"))
            all_rows.append(rows)
            rows.to_csv(out / "markets" / (result["market"] + ".csv"), index=False)
            fits = result.pop("fits")
            (out / "markets" / (result["market"] + "_fits.json")).write_text(json.dumps(fits, indent=2, allow_nan=False))
            result["refits"] = len(fits)
            result["rows"] = len(rows)
            receipts.append(result)
            print(result["market"], result["elapsed_seconds"], flush=True)
    forecasts = pd.concat(all_rows, ignore_index=True).sort_values(["seed", "market", "date"]).reset_index(drop=True)
    if (forecasts.duplicated(["seed", "market", "date"]).any()
            or not np.isfinite(forecasts.rk_forecast).all() or (forecasts.rk_forecast <= 0).any()
            or not (forecasts.latest_input_date < forecasts.date).all()
            or not (forecasts.fit_end < forecasts.fit_block_start).all()):
        raise ValueError("Forecast coverage, positivity or chronology failed")
    if pilot_refits is None:
        for seed in SEEDS:
            actual = forecasts.loc[forecasts.seed == seed, ["market", "date"]].reset_index(drop=True)
            if not actual.equals(mask[["market", "date"]]):
                raise ValueError("Exact22,026 common-key coverage failed for seed" + str(seed))
    forecasts.to_csv(out / "predictions.csv", index=False)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError("Original source changed during run: " + path)
    if sha(__file__) != design["code_sha256"]:
        raise ValueError("Frozen adapter changed during fitting")
    receipt = dict(status="pilot_complete" if pilot_refits else "complete", rows=len(forecasts),
        unique_keys=len(forecasts[["market", "date"]].drop_duplicates()), markets=len(markets),
        elapsed_seconds=time.perf_counter() - started, market_receipts=receipts,
        predictions_sha256=sha(out / "predictions.csv"), originals_preserved=True, scoring="not performed")
    (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pilot-refits", type=int)
    parser.add_argument("--pilot-markets", nargs="+", default=["SPX"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.out, workers=args.workers, pilot_refits=args.pilot_refits,
                         pilot_markets=args.pilot_markets, resume=args.resume), indent=2))
