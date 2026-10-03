"""Causal, fixed-fit GNNHAR1L RK adaptation; original archives are read-only.

Architecture adapted from chaozhang-ox/GNNHAR, commit
98a85b0802bf9336bb68196013c7bfbfa92592f8, GNNHAR.py.

MIT License
Copyright (c) 2024 Chao Zhang
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (8101, 8102, 8103, 8104, 8105)
AUTHOR_COMMIT = "98a85b0802bf9336bb68196013c7bfbfa92592f8"
AUTHOR_FILE_SHA256 = "840f728ca2365329cd67fa7c54b233cbfa941db2ca1f6861f50de62bc5c097b5"
EPSILON = 1e-4  # Native author's QLIKE denominator stabilizer, percent squared.
TRAIN_GATE_TOLERANCE = 1e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prior_features(values):
    """At observed session k, include k and its own preceding sessions only.

    No calendar expansion occurs. Missing input RK is carried from a strictly
    earlier observed session; missing targets remain missing in the caller.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    owner = np.maximum.accumulate(np.where(valid, np.arange(len(values)), -1))
    filled = np.full(len(values), np.nan)
    keep = owner >= 0
    filled[keep] = values[owner[keep]]
    features = np.full((len(values), 3), np.nan)
    imputations = np.zeros(len(values), dtype=np.int64)
    if len(values) >= 22:
        windows = np.lib.stride_tricks.sliding_window_view(filled, 22)
        features[21:, 0] = filled[21:]
        features[21:, 1] = windows[:, -5:].mean(1)
        features[21:, 2] = windows.mean(1)
        imputations[21:] = np.lib.stride_tricks.sliding_window_view(~valid, 22).sum(1)
    return features, imputations


def make_features(panel, markets, adjacency, target_market, target_indices):
    """One target-specific as-of snapshot; never pair rows by ordinal index.

    Own lags end at the previous own session. Neighbour lags end strictly before
    that origin's date, conservatively avoiding same-date exchange availability.
    """
    own = panel[target_market]
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if np.any(target_indices < 1):
        raise ValueError("A target must have an observed predecessor")
    origins = own["date"][target_indices - 1]
    own_features = own["features"][target_indices - 1].copy()
    aggregated = np.zeros_like(own_features)
    latest = np.full(len(origins), "0000-00-00", dtype="U10")
    available = np.isfinite(own_features).all(1)
    missing_inputs = own["imputations"][target_indices - 1].copy()
    i = markets.index(target_market)
    for j, market in enumerate(markets):
        if i == j or adjacency[i, j] == 0:
            continue
        neighbor = panel[market]
        positions = np.searchsorted(neighbor["date"], origins, side="left") - 1
        found = positions >= 0
        clipped = np.maximum(positions, 0)
        feature = neighbor["features"][clipped]
        available &= found & np.isfinite(feature).all(1)
        # Invalid early snapshots are explicitly excluded from TRAIN only.
        aggregated += adjacency[i, j] * np.nan_to_num(feature, nan=0.)
        dates = neighbor["date"][clipped]
        latest = np.where(found & (dates > latest), dates, latest)
        missing_inputs += np.where(found, neighbor["imputations"][clipped], 0)
        if np.any(found & (dates >= origins)):
            raise AssertionError("A neighbour input crosses the origin date")
    return own_features, aggregated, origins, latest, available, missing_inputs


def training_graph(panel, markets, train_end):
    """Author GLASSO rule on complete *training-return* dates only.

    Complete dates are a graph estimator input, never an evaluation row filter.
    No validation/test return enters its CV, covariance, or normalization.
    """
    from sklearn.covariance import GraphicalLassoCV

    columns = []
    for market in markets:
        item = panel[market]
        mask = item["date"] <= train_end
        columns.append(pd.Series(item["returns"][mask], index=item["date"][mask], name=market))
    frame = pd.concat(columns, axis=1, join="inner").sort_index()
    if len(frame) < 100 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Insufficient finite complete training-return vectors")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = GraphicalLassoCV().fit(frame.to_numpy(dtype=np.float64))
    precision = fitted.precision_
    minimum_eigenvalue = float(np.linalg.eigvalsh(precision).min())
    final_iterations = int(getattr(fitted, "n_iter_", 0))
    final_costs = getattr(fitted, "costs_", [])
    final_dual_gap = float(final_costs[-1][1]) if len(final_costs) else None
    binary = (precision != 0).astype(np.float64)
    np.fill_diagonal(binary, 0.)
    degree_inverse = np.sqrt(1. / (binary.sum(1) + 1e-8))
    adjacency = binary * degree_inverse[:, None] * degree_inverse[None, :]
    if not np.isfinite(precision).all() or not np.isfinite(adjacency).all():
        raise ValueError("Nonfinite GLASSO fit; no graph replacement is permitted")
    if minimum_eigenvalue <= 0 or final_iterations >= fitted.max_iter:
        raise ValueError("Final GLASSO precision is not positive/converged")
    if final_dual_gap is not None and (not np.isfinite(final_dual_gap) or abs(final_dual_gap) > fitted.tol):
        raise ValueError("Final GLASSO dual gap exceeds its registered tolerance")
    np.testing.assert_allclose(adjacency, adjacency.T, rtol=1e-10, atol=1e-12)
    return adjacency, {
        "method": "GraphicalLassoCV defaults; precision != 0; no self-loops",
        "training_return_rows": len(frame), "first_date": str(frame.index[0]),
        "last_date": str(frame.index[-1]), "train_end": train_end,
        "alpha": float(fitted.alpha_), "undirected_edges": int(binary.sum() / 2),
        "final_iterations": final_iterations, "maximum_iterations": fitted.max_iter,
        "final_dual_gap": final_dual_gap, "tolerance": fitted.tol,
        "precision_minimum_eigenvalue": minimum_eigenvalue,
        "isolated_nodes": [markets[i] for i in np.flatnonzero(binary.sum(1) == 0)],
        "seconds": time.perf_counter() - started, "warning_count": len(caught),
        "warnings": sorted(set(str(w.message) for w in caught))[:20],
    }


@dataclass
class Prepared:
    markets: list[str]
    adjacency: np.ndarray
    own: np.ndarray
    neighbor: np.ndarray
    target: np.ndarray
    records: pd.DataFrame
    evaluation_positions: np.ndarray
    mask: pd.DataFrame
    hashes: dict[str, str]
    graph_receipt: dict
    train_end: str
    validation_end: str
    preparation_receipt: dict


def prepare_dataset(root=ROOT, train_end="2017-12-31", validation_end="2018-12-31"):
    root = Path(root).resolve()
    if not train_end < validation_end < "2019-01-01":
        raise ValueError("TRAIN and validation must be chronological and strictly pre2019")
    bundle_path = root / "experiments/outputs/frontier_extension_20260926_v1/bundle.json"
    manifest_path = root / "experiments/outputs/experiment_programme_20260930/programme_design_v1/manifest.json"
    common_path = root / "experiments/outputs/harmonized_leaderboard_20260930_v1/common_mask.csv"
    hashes = {str(p): sha(p) for p in (bundle_path, manifest_path, common_path)}
    bundle, manifest = json.loads(bundle_path.read_text()), json.loads(manifest_path.read_text())
    mask = pd.DataFrame([r for r in bundle["rows"] if r["phase"] == "evaluation" and r["primary_eligible"]])
    mask = mask.sort_values(["market", "date"]).reset_index(drop=True)
    current = pd.read_csv(common_path)
    key = ["market", "date"]
    current = current.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(mask[key], current[key], check_dtype=False)
    for field in ("index", "r2", "rk"):
        np.testing.assert_allclose(mask[field], current[field], rtol=1e-12, atol=1e-12)
    markets = sorted(bundle["market_paths"])
    if len(mask) != 22026 or len(markets) != 28 or mask.date.nunique() != 827 or mask.duplicated(key).any():
        raise ValueError("The preserved primary score population changed")
    panel = {}
    for market in markets:
        source = root / manifest["data"]["directory"] / (market + ".npz")
        digest = sha(source)
        if digest != manifest["data_hashes"][str(source.relative_to(root))]:
            raise ValueError("Authoritative archive hash drift: " + market)
        hashes[str(source)] = digest
        with np.load(source, allow_pickle=False) as raw:
            dates = raw["date"].astype("U10")
            rk = raw["rk_parzen"].astype(np.float64) * 10000.
            returns = raw["returns"].astype(np.float64) * 100.
        if np.any(dates[1:] <= dates[:-1]) or not np.isfinite(returns).all():
            raise ValueError("Invalid source dates or returns: " + market)
        features, imputations = prior_features(rk)
        panel[market] = dict(date=dates, rk=rk, returns=returns, features=features, imputations=imputations)
        rows = mask.loc[mask.market == market]
        positions = np.searchsorted(dates, rows.date)
        np.testing.assert_array_equal(dates[positions], rows.date)
        np.testing.assert_array_equal(positions, rows["index"])
        np.testing.assert_allclose(rk[positions], rows.rk, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(returns[positions] ** 2, rows.r2, rtol=1e-12, atol=1e-12)
    adjacency, graph_receipt = training_graph(panel, markets, train_end)
    all_own, all_neighbor, targets, records, preparation = [], [], [], [], {}
    for market in markets:
        item = panel[market]
        # Positive observed TRAIN/validation targets only; evaluation mask is authoritative.
        prior = np.flatnonzero((item["date"] <= validation_end) & np.isfinite(item["rk"]) & (item["rk"] > 0))
        prior = prior[prior >= 22]
        evaluation = mask.loc[mask.market == market, "index"].to_numpy(dtype=np.int64)
        indices = np.concatenate([prior, evaluation])
        own, neighbor, origins, latest, available, imputations = make_features(panel, markets, adjacency, market, indices)
        is_evaluation = np.arange(len(indices)) >= len(prior)
        if not available[is_evaluation].all():
            raise ValueError("Unfillable evaluation history: " + market)
        # Missing historical target or early graph context cannot create synthetic target values.
        keep = available
        preparation[market] = dict(
            eligible_historical_target_candidates=len(prior),
            early_unavailable_historical_contexts=int(np.sum(~available & ~is_evaluation)),
            invalid_historical_targets=int(np.sum((item["date"] <= validation_end) &
                (~np.isfinite(item["rk"]) | (item["rk"] <= 0)))),
            evaluation_rows=len(evaluation), target_imputations=0,
            training_feature_imputations=int(imputations[(item["date"][indices] <= train_end) & keep].sum()),
            evaluation_feature_imputations=int(imputations[is_evaluation].sum()))
        all_own.append(own[keep]); all_neighbor.append(neighbor[keep]); targets.append(item["rk"][indices[keep]])
        for k in np.flatnonzero(keep):
            date = str(item["date"][indices[k]])
            split = "evaluation" if is_evaluation[k] else "train" if date <= train_end else "validation"
            records.append(dict(market=market, date=date, index=int(indices[k]), split=split,
                origin_date=str(origins[k]), latest_input_date=str(origins[k]),
                latest_neighbor_input_date=str(latest[k]), input_imputations=int(imputations[k])))
    frame = pd.DataFrame(records)
    evaluation_positions = np.flatnonzero(frame.split.to_numpy() == "evaluation")
    pd.testing.assert_frame_equal(frame.iloc[evaluation_positions][key].reset_index(drop=True), mask[key], check_dtype=False)
    if np.any(frame.latest_input_date >= frame.date) or np.any(frame.latest_neighbor_input_date >= frame.origin_date):
        raise AssertionError("Input chronology failed")
    return Prepared(markets, adjacency, np.concatenate(all_own).astype(np.float32),
        np.concatenate(all_neighbor).astype(np.float32), np.concatenate(targets).astype(np.float32),
        frame, evaluation_positions, mask, hashes, graph_receipt, train_end, validation_end, preparation)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, node_feature, adj):
        return torch.matmul(adj, torch.matmul(node_feature, self.weight))


class GNNHAR1L(nn.Module):
    """Exact author neural architecture; selected-node route is algebraically equal."""
    def __init__(self, n_hid=9):
        super().__init__()
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        own = self.linear1(node_feat)
        graph = self.mlp1(self.relu(self.gcn1(node_feat, adj)))
        return self.relu(own + graph).squeeze(-1)

    def forward_selected(self, own_feat, neighbor_aggregate):
        own = self.linear1(own_feat)
        graph = self.mlp1(self.relu(neighbor_aggregate @ self.gcn1.weight))
        return self.relu(own + graph).squeeze(-1)


def author_qlike(target, native_raw_forecast):
    ratio = target / (native_raw_forecast + EPSILON)
    return torch.mean(ratio - torch.log(ratio))


def _subset(indices, limit):
    indices = np.asarray(indices, dtype=np.int64)
    if limit and len(indices) > limit:
        return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]
    return indices


def checkpoint_quality(target, native_raw_forecast, constant_qlike):
    """An evaluated TRAIN-only gate; never use epoch minibatch-average loss."""
    target = np.asarray(target, dtype=np.float64)
    raw = np.asarray(native_raw_forecast, dtype=np.float64)
    finite = bool(np.isfinite(raw).all() and np.all(raw >= 0) and np.all(raw + EPSILON > 0))
    if finite:
        ratio = target / (raw + EPSILON)
        loss = float(np.mean(ratio - np.log(ratio)))
    else:
        loss = None
    threshold = constant_qlike + TRAIN_GATE_TOLERANCE
    passed = finite and loss is not None and np.isfinite(loss) and loss <= threshold
    return dict(full_train_qlike=loss, train_constant_qlike=constant_qlike,
        acceptance_threshold=threshold, finite_train_outputs=finite,
        train_gate_pass=bool(passed), zero_raw_train_outputs=int(np.sum(raw == 0)),
        train_gate_failure=None if passed else "nonfinite_or_invalid_train_output" if not finite
            else "full_train_qlike_exceeds_constant_baseline")


class QualifiedCheckpoint:
    """Validation selection and patience operate only after TRAIN qualification."""
    def __init__(self, min_delta=1e-6):
        self.best = float("inf")
        self.significant_best = float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.best_quality = None
        self.first_qualified_epoch = None
        self.stale_epochs = 0
        self.min_delta = min_delta

    def consider(self, epoch, quality, validation_loss, state_getter):
        selected = False
        if quality["train_gate_pass"] and np.isfinite(validation_loss):
            if self.first_qualified_epoch is None:
                self.first_qualified_epoch = epoch
            if validation_loss < self.best:
                self.best, self.best_epoch = validation_loss, epoch
                self.best_state = copy.deepcopy(state_getter())
                self.best_quality = dict(quality)
                selected = True
            if validation_loss < self.significant_best - self.min_delta:
                self.significant_best, self.stale_epochs = validation_loss, 0
            else:
                self.stale_epochs += 1
        elif self.first_qualified_epoch is not None:
            self.stale_epochs += 1
        return selected


class UnqualifiedFitError(ValueError):
    def __init__(self, message, trial):
        super().__init__(message)
        self.trial = trial


def train_seed(data, seed, epochs=5000, batch_size=128, hidden=9, device="cpu", pilot=False,
               patience=20, min_delta=1e-6, pilot_train_rows=1024, history_callback=None):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = GNNHAR1L(hidden).to(device)
    full_train = np.flatnonzero(data.records.split.to_numpy() == "train")
    train = _subset(full_train, pilot_train_rows if pilot else 0)
    validation = _subset(np.flatnonzero(data.records.split.to_numpy() == "validation"), 512 if pilot else 0)
    if not len(train) or not len(validation):
        raise ValueError("Empty chronological TRAIN or validation")
    tensors = [torch.from_numpy(a) for a in (data.own, data.neighbor, data.target)]
    loader = DataLoader(TensorDataset(*(a[train] for a in tensors)), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    target_train = data.target[full_train].astype(np.float64)
    constant_prediction = float(target_train.mean())
    constant_ratio = target_train / (constant_prediction + EPSILON)
    constant_qlike = float(np.mean(constant_ratio - np.log(constant_ratio)))
    selected, history = QualifiedCheckpoint(min_delta), []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train(); total = 0.; count = 0
        for own, neighbor, target in loader:
            own, neighbor, target = own.to(device), neighbor.to(device), target.to(device)
            optimizer.zero_grad()
            loss = author_qlike(target, model.forward_selected(own, neighbor))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; retain the failed seed attempt")
            loss.backward(); optimizer.step()
            total += loss.item() * len(target); count += len(target)
        model.eval()
        with torch.no_grad():
            train_raw = []
            for start in range(0, len(full_train), 4096):
                positions = full_train[start:start + 4096]
                raw = model.forward_selected(tensors[0][positions].to(device), tensors[1][positions].to(device))
                train_raw.append(raw.cpu().numpy().astype(np.float64))
            quality = checkpoint_quality(target_train, np.concatenate(train_raw), constant_qlike)
            own, neighbor, target = (a[validation].to(device) for a in tensors)
            validation_loss = author_qlike(target, model.forward_selected(own, neighbor)).item()
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        accepted = selected.consider(epoch + 1, quality, validation_loss, model.state_dict)
        record = dict(epoch=epoch + 1, train_loss=total / count, validation_loss=validation_loss,
            **quality, checkpoint_selected=accepted, early_stopping_active=selected.best_state is not None,
            stale_qualified_epochs=selected.stale_epochs)
        history.append(record)
        if history_callback is not None:
            history_callback(record)
        if patience and selected.best_state is not None and selected.stale_epochs >= patience:
            break
    seconds = time.perf_counter() - started
    gate_failures = int(sum(not row["train_gate_pass"] for row in history))
    if selected.best_state is None:
        raise UnqualifiedFitError("No TRAIN-qualified checkpoint within the registered budget", dict(
            seed=seed, status="unqualified", maximum_epochs=epochs, epochs_completed=len(history),
            train_constant_prediction=constant_prediction, train_constant_qlike=constant_qlike,
            train_gate_tolerance=TRAIN_GATE_TOLERANCE, failed_candidate_gates=gate_failures,
            history=history, training_seconds=seconds, evaluation_rows=0,
            evaluation_loss_computed=False, seed_screening=False))
    model.load_state_dict(selected.best_state)
    evaluation = _subset(data.evaluation_positions, 512 if pilot else 0)
    model.eval(); chunks = []; infer_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(evaluation), 4096):
            positions = evaluation[start:start + 4096]
            raw = model.forward_selected(tensors[0][positions].to(device), tensors[1][positions].to(device))
            chunks.append(raw.cpu().numpy().astype(np.float64))
    raw = np.concatenate(chunks)
    prediction = raw + EPSILON
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError("Invalid issued RK prediction; no row exclusion allowed")
    fields = ["market", "date", "index", "origin_date", "latest_input_date", "latest_neighbor_input_date", "input_imputations"]
    result = data.records.iloc[evaluation][fields].reset_index(drop=True).copy()
    result["seed"] = seed; result["rk_forecast"] = prediction; result["native_raw_forecast"] = raw
    return model, result, dict(seed=seed, maximum_epochs=epochs, epochs_completed=len(history), best_epoch=selected.best_epoch,
        best_validation_loss=selected.best, selected_checkpoint_quality=selected.best_quality,
        train_constant_prediction=constant_prediction, train_constant_qlike=constant_qlike,
        train_gate_tolerance=TRAIN_GATE_TOLERANCE, failed_candidate_gates=gate_failures,
        passed_candidate_gates=len(history) - gate_failures, first_qualified_epoch=selected.first_qualified_epoch,
        training_rows=len(train), full_training_quality_rows=len(full_train), validation_rows=len(validation),
        evaluation_rows=len(evaluation), training_seconds=seconds,
        inference_seconds=time.perf_counter() - infer_started, zero_raw_outputs=int(np.sum(raw == 0)),
        history=history, patience=patience, min_delta=min_delta,
        stopped_early=len(history) < epochs, seed_screening=False, test_loss_computed=False)


def verify_coverage(frame, mask, seeds=SEEDS):
    if set(frame.seed) != set(seeds) or frame.duplicated(["market", "date", "seed"]).any():
        raise ValueError("Missing seeds or duplicate forecasts")
    expected = set(map(tuple, mask[["market", "date"]].to_numpy()))
    for seed in seeds:
        current = frame.loc[frame.seed == seed]
        if len(current) != len(mask) or set(map(tuple, current[["market", "date"]].to_numpy())) != expected:
            raise ValueError("Primary coverage differs for seed " + str(seed))
    if not np.isfinite(frame.rk_forecast).all() or (frame.rk_forecast <= 0).any():
        raise ValueError("Invalid RK forecasts")


def fit_predict(root, out, seeds=SEEDS, epochs=5000, batch_size=128, hidden=9,
                device="cpu", pilot=False, threads=2, patience=20, min_delta=1e-6,
                pilot_train_rows=1024):
    if epochs < 1 or batch_size < 1 or hidden < 1 or threads < 1 or patience < 0 or min_delta < 0 or pilot_train_rows < 0:
        raise ValueError("Positive training budgets are required")
    if not pilot and tuple(seeds) != SEEDS:
        raise ValueError("Every full run requires the five locked seeds")
    if pilot and epochs > 5:
        raise ValueError("Local timing pilot is capped at five epochs")
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    started = time.perf_counter()
    data = prepare_dataset(root)
    receipt = dict(status="pilot_only" if pilot else "running", author_commit=AUTHOR_COMMIT,
        author_file_sha256=AUTHOR_FILE_SHA256, target="Observed daily RK, percent squared",
        adaptation="Fixed pre2019 GNNHAR1L; asynchronous prior-only graph features",
        train_end=data.train_end, validation_end=data.validation_end, seeds=list(seeds),
        epochs=epochs, batch_size=batch_size, hidden=hidden, learning_rate=.001,
        patience=patience, min_delta=min_delta, pilot_train_rows=pilot_train_rows if pilot else None,
        checkpoint_acceptance={"criterion": "Full TRAIN native QLIKE <= TRAIN mean constant native QLIKE + tolerance",
            "tolerance": TRAIN_GATE_TOLERANCE, "finite_outputs_required": True,
            "selection": "Best whole2018 validation only among TRAIN-qualified candidates",
            "patience_starts": "First TRAIN-qualified candidate"},
        weight_decay=1e-5, epsilon=EPSILON, device=device, threads=threads,
        graph=data.graph_receipt, input_hashes=data.hashes,
        source_hashes={str(Path(__file__).resolve()): sha(__file__)},
        split_counts=data.records.split.value_counts().to_dict(),
        preparation=data.preparation_receipt,
        primary_required_rows_per_seed=len(data.mask),
        runtime={name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scikit-learn")},
        trials=[], failed_seeds=[], test_loss_computed=False, seed_screening=False)
    (out / "design.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    np.save(out / "adjacency.npy", data.adjacency, allow_pickle=False)
    data.mask.to_csv(out / "common_mask.csv", index=False)
    results = []
    try:
        for seed in seeds:
            receipt["active_seed"] = seed
            (out / "progress.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            def record_candidate(stream, record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
            try:
                with (out / f"seed_{seed}_candidate_history.jsonl").open("x") as stream:
                    model, predictions, trial = train_seed(data, seed, epochs, batch_size, hidden, device, pilot,
                        patience, min_delta, pilot_train_rows, lambda row: record_candidate(stream, row))
            except UnqualifiedFitError as exc:
                receipt["trials"].append(exc.trial)
                receipt["failed_seeds"].append(seed)
                (out / "progress.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
                continue
            results.append(predictions); receipt["trials"].append(trial)
            torch.save(model.state_dict(), out / f"seed_{seed}.pt")
            (out / "progress.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        if receipt["failed_seeds"]:
            raise ValueError("Required seeds lack TRAIN-qualified models: " + str(receipt["failed_seeds"]))
        frame = pd.concat(results, ignore_index=True)
        if not pilot:
            verify_coverage(frame, data.mask, seeds)
        for path, digest in data.hashes.items():
            if sha(path) != digest:
                raise ValueError("Input changed during run: " + path)
        frame.to_csv(out / ("pilot_predictions.csv" if pilot else "predictions.csv"), index=False, float_format="%.17g")
        receipt.update(status="pilot_complete_not_final" if pilot else "complete",
            output_rows=len(frame), wall_seconds=time.perf_counter() - started,
            full_coverage_verified=not pilot, active_seed=None,
            output_sha256=sha(out / ("pilot_predictions.csv" if pilot else "predictions.csv")))
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", wall_seconds=time.perf_counter() - started)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        raise
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return frame, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=9)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--pilot-train-rows", type=int, default=1024,
        help="Pilot only: zero times one pass over all TRAIN rows; never increases test predictions")
    args = parser.parse_args()
    _, receipt = fit_predict(args.root, args.out, seeds=SEEDS[:1] if args.pilot else SEEDS,
        epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden,
        device=args.device, pilot=args.pilot, threads=args.threads, patience=args.patience,
        min_delta=args.min_delta, pilot_train_rows=args.pilot_train_rows)
    print(json.dumps({k: receipt[k] for k in ("status", "output_rows", "wall_seconds", "split_counts", "graph")}, indent=2))


if __name__ == "__main__":
    main()
