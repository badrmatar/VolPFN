"""Frozen four-arm VolPFN training, with an explicitly unpromotable CPU smoke.

Full/noaux/direct share every synthetic draw and the same initial VolPFN.
Historical changes the training distribution and uses only observed returns.
Direct changes both head and objective, so it is NOT a pure auxiliary-loss
ablation. No validation score selects a checkpoint; only the final step does.

Historical sources are Oxford-Man NPZ files: date and decimal returns. Other
keys, including realized variance, are never read. Returns are multiplied by
100 exactly once, and the TRAIN cutoff is applied BEFORE sampling windows.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import volpfn as V
from research.matched_prior import run as M, simulators as S
from research.programme_closure import gpu_training as G
from research.seed_replication import causal, training as T


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("full", "noaux", "historical", "direct")
SEEDS = (8101, 8102, 8103, 8104, 8105)
KIND = "volpfn_four_arm_20260930_v1"
HISTORICAL_ROOT = ROOT / "research/data/oxfordman_barde/calendar_full_v1"
ROSTER_PATH = ROOT / "research/outputs/matched_prior_design_inventory_v1/inventory.json"
CUTOFF = "2010-12-31"


def specification(smoke=False, device="cuda"):
    if device not in ("cpu", "cuda") or (device == "cpu" and not smoke):
        raise ValueError("Final runs require CUDA; CPU is allowed only for explicit smoke")
    s = G.specification(device)
    s.update(kind="ablations_smoke" if smoke else "ablations_final", seeds=list(SEEDS),
             arms=list(ARMS), smoke=bool(smoke), historical_cutoff=CUTOFF,
             auxiliary_weight={"full": .1, "noaux": 0., "historical": 0., "direct": 0.},
             synthetic_arm="broad", diagnostic_validation_selects=False,
             checkpoint_selection="fixed final update only; diagnostic validation never selects",
             historical_sampling="uniform market then uniform contiguous TRAIN window, with replacement",
             direct_semantics="positive point forecast of next standardized squared return; head+loss confound",
             auxiliary_semantics="Gaussian NLL for simulated logv (without its constant), not MSE; overlay labels inherit simulator limitations")
    if smoke:
        s.update(size="unpromotable_tiny_smoke", cfg=dict(d=16, layers=1, heads=2, K=2),
                 steps=2, tokens=32, lengths={"8": .4, "16": .6}, warmup=1,
                 eval_every=2, checkpoint_every=1, log_every=1,
                 val_T=16, val_n=2, validation_burn=2, workers=0,
                 prefetch_depth=1, threads=1)
        s["backend"] = "Explicit CPU/CUDA tiny smoke; unpromotable"
    return s


def source_paths():
    names = ("volpfn.py", "volpfn_prior.py", "research/matched_prior/run.py",
             "research/matched_prior/simulators.py", "research/matched_prior/diagnostics.py",
             "research/programme_closure/gpu_training.py",
             "research/seed_replication/training.py", "research/seed_replication/causal.py",
             "research/experiment_programme_20260930/ablations.py",
             "research/experiment_programme_20260930/test_ablations.py")
    return [ROOT / name for name in names]


def _manifest_path(path):
    path = Path(path).resolve()
    return path / "manifest.json" if path.is_dir() else path


def _hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _read_npz(path, cutoff=CUTOFF):
    with np.load(path, allow_pickle=False) as source:
        # Deliberately never access rv5/rk_parzen/split, even if present.
        dates = np.asarray(source["date"]).astype("U10")
        returns = np.asarray(source["returns"], dtype=np.float64)
    if dates.ndim != 1 or returns.shape != dates.shape:
        raise ValueError("NPZ date/returns must be aligned one-dimensional arrays")
    parsed = dates.astype("datetime64[D]")
    if np.isnat(parsed).any() or not np.array_equal(parsed.astype("U10"), dates):
        raise ValueError("Dates must be valid ISO YYYY-MM-DD")
    if len(dates) < 2 or np.any(parsed[1:] <= parsed[:-1]):
        raise ValueError("Historical dates must be strictly increasing")
    keep = dates <= cutoff
    selected = returns[keep].copy()
    if not np.isfinite(selected).all():
        raise ValueError("TRAIN returns must be finite")
    # Exactly one decimal-to-percentage conversion, after the time gate.
    return dict(date=dates[keep].copy(), returns=selected * 100.)


def freeze(out, *, smoke=False, device="cuda", historical_root=HISTORICAL_ROOT,
           roster_path=ROSTER_PATH, historical_files=None, runtime_receipt=None):
    """Copy/hash whole original NPZs and source before any training is allowed."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing design: " + str(out))
    s = specification(smoke, device)
    roster_path = Path(roster_path).resolve()
    inventory = json.loads(roster_path.read_text())
    markets = inventory["retained_index_population"]
    if not isinstance(markets, list) or not markets or len(set(markets)) != len(markets):
        raise ValueError("Roster must contain distinct retained_index_population market names")
    if any(not isinstance(m, str) or Path(m).name != m or m in (".", "..") for m in markets):
        raise ValueError("Market names must be safe NPZ file stems")
    if not smoke and len(markets) != 28:
        raise ValueError("Final battery requires the retained 28-market roster")
    if historical_files is not None and set(historical_files) != set(markets):
        raise ValueError("Explicit NPZ mapping must match the retained roster exactly")
    originals = {market: Path(historical_files[market] if historical_files is not None
                              else Path(historical_root) / (market + ".npz")).resolve()
                 for market in markets}
    historical = {market: _read_npz(path) for market, path in originals.items()}
    for length in map(int, s["lengths"]):
        if any(len(value["returns"]) < length for value in historical.values()):
            raise ValueError("Every retained market must support every TRAIN sequence length")
    out.mkdir(parents=True)
    hashes = {}
    for source in source_paths():
        relative = str(source.relative_to(ROOT))
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = T.sha(source)
    shutil.copy2(roster_path, out / "inventory.json")
    files = {}
    for market, source in originals.items():
        relative = "data/" + market + ".npz"
        target = out / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        try:
            original = str(source.relative_to(ROOT))
        except ValueError:
            original = str(source)
        files[market] = dict(file=relative, sha256=T.sha(source), original=original,
            train_rows=len(historical[market]["returns"]),
            train_first_date=str(historical[market]["date"][0]),
            train_last_date=str(historical[market]["date"][-1]))
    expected_runtime = None
    if runtime_receipt is not None:
        receipt = json.loads(Path(runtime_receipt).read_text())
        expected_runtime = receipt.get("runtime", receipt)
        T.save_json(out / "runtime_receipt.json", expected_runtime)
    manifest = dict(kind=KIND, created_utc=T.now(), specification=s, markets=markets,
        source_hashes=hashes, config_sha256=_hash_object(s), historical_files=files,
        inventory_sha256=T.sha(roster_path), expected_runtime=expected_runtime,
        runtime_receipt_sha256=T.sha(out / "runtime_receipt.json") if expected_runtime else None,
        input_contract=dict(npz_keys_read=["date", "returns"], original_returns="decimal",
                            conversion="multiply by 100 exactly once", cutoff=CUTOFF,
                            target="next observed return / RMS of strictly preceding row prefix",
                            latent_labels="absent for historical arm; never invented"),
        initialization="same seeded initial VolPFN including unused auxiliary head; direct head is independently seeded after trunk copy",
        historical_repetition="sampling with replacement; log distinct/repeated drawn windows; budget does not imply unique historical information",
        arm_interpretations=dict(full="broad synthetic mixture NLL + .1 Gaussian latent logv NLL",
            noaux="same synthetic batches and initial model, auxiliary coefficient zero",
            historical="same mixture model/noaux objective, balanced contiguous historical TRAIN draws",
            direct="same synthetic batches and initial trunk; new positive point head plus QLIKE; head+loss confounded"),
        normalization_spec=causal.NORMALIZATION_SPEC, causal_spec=causal.CAUSAL_SPEC,
        test_access="no scoring, drift/test targets, or real-market validation is used during training",
        failure_policy="preserve all failure receipts; explicit resume only with matching manifest/runtime/checkpoint; never replace seeds",
        smoke_policy="smoke checkpoints are unpromotable and never satisfy a final run")
    T.save_json(out / "manifest.json", manifest)
    (out / "manifest.sha256").write_text(T.sha(out / "manifest.json") + "\n")
    verify(out / "manifest.json")
    return manifest


def verify(path):
    path = _manifest_path(path)
    if T.sha(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("Manifest seal changed")
    m = json.loads(path.read_text())
    if m.get("kind") != KIND or _hash_object(m["specification"]) != m["config_sha256"]:
        raise RuntimeError("Wrong design or changed configuration")
    for relative, expected in m["source_hashes"].items():
        if T.sha(ROOT / relative) != expected or T.sha(path.parent / "source" / relative) != expected:
            raise RuntimeError("Source drift: " + relative)
    if T.sha(path.parent / "inventory.json") != m["inventory_sha256"]:
        raise RuntimeError("Roster snapshot changed")
    for market, entry in m["historical_files"].items():
        copied = (path.parent / entry["file"]).resolve()
        if not copied.is_relative_to(path.parent) or T.sha(copied) != entry["sha256"]:
            raise RuntimeError("Historical source changed: " + market)
        original = Path(entry["original"])
        if not original.is_absolute():
            original = ROOT / original
        # The snapshot permits relocation; existing originals must still match.
        if original.exists() and T.sha(original) != entry["sha256"]:
            raise RuntimeError("Original historical NPZ changed: " + market)
    if m["expected_runtime"] is not None:
        if T.sha(path.parent / "runtime_receipt.json") != m["runtime_receipt_sha256"]:
            raise RuntimeError("Runtime receipt changed")
    return m


def load_historical(manifest_path, manifest=None):
    path = _manifest_path(manifest_path)
    m = verify(path) if manifest is None else manifest
    return {market: _read_npz(path.parent / entry["file"], m["specification"]["historical_cutoff"])
            for market, entry in m["historical_files"].items()}


def historical_batch(data, data_seed, batch_size, length):
    """Observed-return-only causal batch; no dummy target_logv is manufactured."""
    markets = sorted(data)
    if not markets or any(len(data[m]["returns"]) < length for m in markets):
        raise ValueError("Every market must support the requested contiguous sequence")
    if any(str(data[m]["date"][-1]) > CUTOFF for m in markets):
        raise ValueError("Apply the TRAIN cutoff before historical window sampling")
    rng = np.random.default_rng(data_seed)
    rows, draws = [], []
    for _ in range(batch_size):
        market = markets[int(rng.integers(len(markets)))]
        start = int(rng.integers(len(data[market]["returns"]) - length + 1))
        stop = start + length
        rows.append(data[market]["returns"][start:stop])
        dates = data[market]["date"]
        if str(dates[stop - 1]) > CUTOFF:
            raise ValueError("Historical context/next target exceeds TRAIN cutoff")
        draws.append(dict(market=market, start=start, stop=stop, length=length,
                          first_date=str(dates[start]), last_target_date=str(dates[stop - 1])))
    raw = np.stack(rows)
    f = causal.feature_batch(raw)
    scale = f["output_scale"][:, :-1]
    return dict(feats=f["feats"], base=f["base"],
                target_r=(raw[:, 1:] / scale).astype(np.float32),
                target_mask=np.ones((batch_size, length - 1), dtype=np.float32),
                output_scale=scale, output_log_scale=np.log(scale).astype(np.float32),
                historical_draws=draws)


def synthetic_batch(data_seed, batch_size, length):
    # The corrected prior and causal normalization are shared with GPU training.
    return G.generated(("broad", data_seed, batch_size, length))


def batch_hash(batch):
    digest = hashlib.sha256()
    for key in sorted(k for k, v in batch.items() if isinstance(v, np.ndarray)):
        value = np.ascontiguousarray(batch[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def trunk_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(("inp.", "blocks.", "ln_f.")):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class DirectVolPFN(V.VolPFN):
    """Same causal trunk, new positive point head; unused original heads retained."""
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.head_direct = nn.Linear(cfg["d"], 1)
        nn.init.normal_(self.head_direct.weight, std=.02)
        nn.init.constant_(self.head_direct.bias, math.log(math.expm1(1.)))

    def forward(self, feats, base):
        x = self.inp(feats)
        bias = self.bias(x.shape[1], x.device, x.dtype)
        if self.use_sdpa is None:
            self.use_sdpa = hasattr(F, "scaled_dot_product_attention")
        for block in self.blocks:
            x = block(x, bias, self.use_sdpa)
        x = self.ln_f(x)
        h = torch.exp(base) * F.softplus(self.head_direct(x).squeeze(-1)).clamp_min(1e-8)
        return {"h": h}


def initialize(arm, seed, s):
    if arm not in ARMS:
        raise ValueError("Unknown ablation arm")
    torch.manual_seed(seed)
    original = V.VolPFN(**s["cfg"])
    initial = M.state_hash(original)
    shared_trunk = trunk_hash(original)
    model = original
    if arm == "direct":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 7_100_019)
            model = DirectVolPFN(**s["cfg"])
        # Copy every original parameter exactly; only the new head differs.
        missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
        if set(missing) != {"head_direct.weight", "head_direct.bias"} or unexpected:
            raise AssertionError("Unexpected direct-model state mapping")
    if trunk_hash(model) != shared_trunk:
        raise AssertionError("Initial trunk differs by arm")
    model.to(s["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["lr"],
        betas=tuple(s["optimizer"]["betas"]), weight_decay=s["optimizer"]["weight_decay"],
        eps=s["optimizer"]["eps"])
    return model, optimizer, dict(shared_initial_state_sha256=initial,
        initial_trunk_sha256=shared_trunk, initial_state_sha256=M.state_hash(model))


def _validate(batch, full=False):
    keys = ["feats", "base", "target_r", "target_mask", "output_log_scale"]
    if full:
        keys.append("target_logv")
    for key in keys:
        if not np.isfinite(batch[key]).all():
            raise FloatingPointError("nonfinite_generated_" + key)
    if np.any(batch["target_mask"] < 0) or batch["target_mask"].sum() <= 0:
        raise FloatingPointError("No valid training targets")


def objective(arm, prediction, batch):
    w = batch["target_mask"]
    denominator = w.sum().clamp_min(1.)
    if arm == "direct":
        h = prediction["h"][:, :-1].clamp_min(1e-8)
        qlike = (h.log() + batch["target_r"].square() / h)
        value = (qlike * w).sum() / denominator
        return value, {"qlike": value}
    pred = {key: value[:, :-1] for key, value in prediction.items()}
    nll = -V.mixture_logpdf(pred, batch["target_r"]) + batch["output_log_scale"]
    nll = (nll * w).sum() / denominator
    if arm == "full":
        # Gaussian latent log-variance NLL, including learned log-sd, NOT MSE.
        mean, logsd = pred["lat_mean"], pred["lat_logsd"]
        gaussian = .5 * ((batch["target_logv"] - mean) * torch.exp(-logsd)).square() + logsd
        aux = (gaussian * w).sum() / denominator
        return nll + .1 * aux, {"nll": nll, "aux_gaussian_nll": aux}
    return nll, {"nll": nll}


def update(arm, model, optimizer, batch, step, s):
    _validate(batch, full=arm == "full")
    bt = T.to_device(batch, s["device"])
    for group in optimizer.param_groups:
        group["lr"] = T.learning_rate(step, s)
    model.train()
    prediction = model(bt["feats"], bt["base"])
    for key, value in prediction.items():
        T.finite_tensor(value, "output_" + key)
    value, components = objective(arm, prediction, bt)
    T.finite_tensor(value, "loss")
    optimizer.zero_grad(set_to_none=True)
    value.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), s["gradient_norm_clip"],
                                             error_if_nonfinite=True)
    T.finite_tensor(gradient, "gradient")
    optimizer.step()
    T.finite_parameters(model)
    return dict(loss=float(value.detach().cpu()), gradient=float(gradient.detach().cpu()),
                **{key: float(v.detach().cpu()) for key, v in components.items()})


@torch.no_grad()
def diagnostic(arm, model, panel, s):
    """One fixed synthetic panel, never used for stopping or selection."""
    model.eval()
    values = []
    for i in range(len(panel["feats"])):
        batch = {key: value[i:i+1].copy() for key, value in panel.items()
                 if isinstance(value, np.ndarray)}
        batch["target_mask"][:, :max(s["validation_burn"] - 1, 0)] = 0
        _validate(batch, full=arm == "full")
        bt = T.to_device(batch, s["device"])
        prediction = model(bt["feats"], bt["base"])
        _, components = objective(arm, prediction, bt)
        values.append({key: float(v.cpu()) for key, v in components.items()})
    return {key: float(np.mean([row[key] for row in values])) for key in values[0]}


def _runtime_identity(receipt):
    return {key: receipt.get(key) for key in ("python", "numpy", "scipy", "torch", "cuda", "gpu")}


def _bind_json(path, value):
    """Atomic create-once binding, safe for independently launched arm workers."""
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise RuntimeError("Run binding mismatch: " + str(path))
    finally:
        temp.unlink(missing_ok=True)


def _checkpoint(path, model, optimizer, step, history, s, arm, seed, manifest_hash,
                initial, runtime, batch_chain, historical_seen, historical_observed, counts):
    value = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        cfg=model.cfg, arm=arm, seed=seed, step=step, history=history, training_spec=s,
        manifest_sha256=manifest_hash, initial=initial, runtime=runtime,
        optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if s["device"] == "cuda" else [],
        batch_chain_sha256=batch_chain, historical_seen=sorted(historical_seen),
        historical_observed=sorted(historical_observed), counts=counts,
        normalization_spec=causal.NORMALIZATION_SPEC, causal_spec=causal.CAUSAL_SPEC,
        smoke=s["smoke"], selection="fixed final step only")
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(manifest_path, out, arm, seed, *, resume=False):
    """Train one registered arm/seed; no paid launch or automatic retry occurs."""
    path = _manifest_path(manifest_path)
    m = verify(path)
    s = m["specification"]
    if arm not in ARMS or seed not in s["seeds"]:
        raise ValueError("Arm or seed outside frozen design")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = T.sha(path)
    run = out / "runs" / (arm + "_" + str(seed))
    if run.exists() and not resume:
        raise FileExistsError("Existing seed retained; explicit matching resume required")
    if resume and not run.exists():
        raise FileNotFoundError("No existing run to resume")
    if resume and (run / "status.json").exists():
        if json.loads((run / "status.json").read_text())["status"] == "complete":
            raise RuntimeError("Completed final seed cannot be retrained")
    run.mkdir(parents=True, exist_ok=True)
    # A live owner excludes concurrent writers to one seed.
    lock = run / "active.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    stream = None
    start = time.perf_counter()
    attempt = len(list(run.glob("attempt_*.json"))) + 1
    record = dict(status="running", arm=arm, seed=seed, attempt=attempt,
        started_utc=T.now(), manifest_sha256=manifest_hash, runtime=None,
        completed_updates=0, attempted_updates=0, input_positions=0, scored_targets=0,
        historical_draws=0, historical_unique_windows=0, historical_repeat_draws=0,
        historical_unique_observations=0, historical_repeated_input_positions=0,
        prior_failure_receipts=sorted(p.name for p in run.glob("failure_*.txt")))
    history, seen, observed = [], set(), set()
    chain = hashlib.sha256(b"ablation-synthetic-batch-chain-v1").hexdigest()
    model = optimizer = None
    try:
        G.configure(s)
        runtime = G.runtime()
        identity = _runtime_identity(runtime)
        record["runtime"] = identity
        if m["expected_runtime"] is not None and identity != _runtime_identity(m["expected_runtime"]):
            raise RuntimeError("Runtime/GPU type differs from frozen pilot receipt")
        _bind_json(out / "BINDING.json", dict(manifest_sha256=manifest_hash, runtime=identity))
        model, optimizer, initial = initialize(arm, seed, s)
        step0 = 0
        if resume:
            old = json.loads((run / "status.json").read_text())
            if old["status"] == "complete":
                raise RuntimeError("Completed final seed cannot be retrained")
            T.save_json(run / ("preserved_status_attempt_%04d.json" % attempt), old)
            ck = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
            if T.sha(run / "resume.pt") != old.get("resume_checkpoint_sha256"):
                raise RuntimeError("Resume checkpoint hash changed or is unsealed")
            if (ck["manifest_sha256"] != manifest_hash or ck["arm"] != arm or ck["seed"] != seed
                    or ck["training_spec"] != s or ck["runtime"] != identity or ck["initial"] != initial):
                raise RuntimeError("Checkpoint/manifest/runtime/initialization mismatch")
            model.load_state_dict(ck["state_dict"])
            optimizer.load_state_dict(ck["optimizer"])
            torch.set_rng_state(ck["torch_rng"])
            if s["device"] == "cuda":
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            step0, history, chain = ck["step"], ck["history"], ck["batch_chain_sha256"]
            seen = set(ck["historical_seen"])
            observed = set(ck["historical_observed"])
            record.update(ck["counts"])
            record["resumed_from_update"] = step0
        else:
            torch.save(dict(state_dict=model.state_dict(), initial=initial,
                            manifest_sha256=manifest_hash), run / "initial.pt")
        record.update(initial)
        T.save_json(run / "status.json", record)
        data = load_historical(path, m) if arm == "historical" else None
        panel = synthetic_batch(99101, s["val_n"], s["val_T"])
        if not history:
            history.append(dict(step=0, diagnostic=diagnostic(arm, model, panel, s)))
        entries = list(M.schedule(seed, s))
        if arm != "historical":
            stream = G.OrderedBatches("broad", seed, s, start=step0)
        for step in range(step0, s["steps"]):
            record["attempted_updates"] = step + 1
            data_seed, batch_size, length = entries[step]
            batch = (historical_batch(data, data_seed, batch_size, length) if arm == "historical"
                     else stream.next())
            digest = batch_hash(batch)
            chain = hashlib.sha256(bytes.fromhex(chain) + bytes.fromhex(digest)).hexdigest()
            components = update(arm, model, optimizer, batch, step, s)
            record.update(completed_updates=step + 1,
                input_positions=(step + 1) * s["tokens"],
                scored_targets=record["scored_targets"] + int(batch["target_mask"].sum()))
            if arm == "historical":
                for draw in batch["historical_draws"]:
                    seen.add(draw["market"] + ":" + str(draw["start"]) + ":" + str(length))
                    observed.update(draw["market"] + ":" + str(position)
                                    for position in range(draw["start"], draw["stop"]))
                record["historical_draws"] += len(batch["historical_draws"])
                record["historical_unique_windows"] = len(seen)
                record["historical_repeat_draws"] = record["historical_draws"] - len(seen)
                record["historical_unique_observations"] = len(observed)
                record["historical_repeated_input_positions"] = record["input_positions"] - len(observed)
            event = dict(attempt=attempt, step=step + 1, data_seed=data_seed,
                batch_size=batch_size, sequence_length=length, batch_sha256=digest,
                batch_chain_sha256=chain, input_positions=record["input_positions"],
                seconds=time.perf_counter() - start, **components)
            if arm == "historical":
                event["historical_draws"] = batch["historical_draws"]
            # Hash every batch; training schedules can be audited arm by arm.
            with (run / "training.jsonl").open("a") as log:
                log.write(json.dumps(event, allow_nan=False) + "\n")
            if (step + 1) % s["eval_every"] == 0 or step + 1 == s["steps"]:
                history.append(dict(step=step + 1, diagnostic=diagnostic(arm, model, panel, s)))
                T.save_json(run / "diagnostics.json", history)
            if (step + 1) % s["checkpoint_every"] == 0 or step + 1 == s["steps"]:
                counts = {key: record[key] for key in ("completed_updates", "input_positions",
                    "scored_targets", "historical_draws", "historical_unique_windows", "historical_repeat_draws",
                    "historical_unique_observations", "historical_repeated_input_positions")}
                _checkpoint(run / "resume.pt", model, optimizer, step + 1, history, s,
                            arm, seed, manifest_hash, initial, identity, chain, seen, observed, counts)
                record["resume_checkpoint_sha256"] = T.sha(run / "resume.pt")
                T.save_json(run / "status.json", record)
            if (step + 1) % s["log_every"] == 0:
                T.save_json(run / "status.json", record)
                print("ABLATION", arm, seed, step + 1, components, flush=True)
        if record["input_positions"] != s["steps"] * s["tokens"]:
            raise AssertionError("Input-position budget mismatch")
        verify(path)
        shutil.copy2(run / "resume.pt", run / "final.pt")
        record.update(status="complete", final_step=s["steps"],
            checkpoint_sha256=T.sha(run / "final.pt"), batch_chain_sha256=chain,
            smoke_unpromotable=s["smoke"])
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=repr(error), failure_receipt="failure_%04d.txt" % attempt)
        (run / record["failure_receipt"]).write_text(traceback.format_exc())
        raise
    finally:
        if stream is not None:
            stream.close()
        record.update(ended_utc=T.now(), elapsed_seconds=time.perf_counter() - start)
        T.save_json(run / "status.json", record)
        T.save_json(run / ("attempt_%04d.json" % attempt), record)
        lock.unlink(missing_ok=True)
    return record


def load_model(checkpoint, manifest_path, device="cuda"):
    m = verify(manifest_path)
    s = m["specification"]
    if device not in ("cpu", "cuda") or (device == "cpu" and not s["smoke"]):
        raise ValueError("CPU loading allowed only for an explicit smoke manifest")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (ck["manifest_sha256"] != T.sha(_manifest_path(manifest_path)) or ck["training_spec"] != s
            or ck["step"] != s["steps"] or ck["selection"] != "fixed final step only"
            or ck["normalization_spec"] != causal.NORMALIZATION_SPEC or ck["causal_spec"] != causal.CAUSAL_SPEC):
        raise RuntimeError("Checkpoint is not this design's fixed final model")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no backend substitution")
    model = DirectVolPFN(**ck["cfg"]) if ck["arm"] == "direct" else V.VolPFN(**ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    model.ablation_arm = ck["arm"]
    model.smoke_unpromotable = ck["smoke"]
    model.eval()
    return model.to(device)


@torch.no_grad()
def predict_windows(model, histories, batch_size=8, device=None, targets=None):
    """Per-origin last-1024 context; raw mixture moments or direct point h.

    Variable-length prefixes are grouped without padding. No future return
    enters a feature/scaler. This does not use legacy stride ownership.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device if device is None else torch.device(device)
    if torch.device(device) != next(model.parameters()).device:
        raise ValueError("Prediction device must match model device")
    rows = [np.asarray(row, dtype=np.float64)[-1024:] for row in histories]
    if not rows or any(row.ndim != 1 or not len(row) or not np.isfinite(row).all() for row in rows):
        raise ValueError("Histories must be nonempty finite one-dimensional past returns")
    if targets is not None:
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (len(rows),) or not np.isfinite(targets).all():
            raise ValueError("Optional raw targets must be finite and align one-to-one with histories")
    direct = isinstance(model, DirectVolPFN)
    keys = ("h",) if direct else ("mean", "variance", "second_moment", "h")
    result = {key: np.empty(len(rows), dtype=np.float64) for key in keys}
    if targets is not None and not direct:
        result["native_nll"] = np.empty(len(rows), dtype=np.float64)
    was_training = model.training
    model.eval()
    try:
        for length in sorted(set(map(len, rows))):
            members = [i for i, row in enumerate(rows) if len(row) == length]
            for begin in range(0, len(members), batch_size):
                indices = members[begin:begin + batch_size]
                features = causal.feature_batch(np.stack([rows[i] for i in indices]))
                output = model(torch.as_tensor(features["feats"], device=device),
                               torch.as_tensor(features["base"], device=device))
                unit = features["output_scale"][:, -1]
                if direct:
                    result["h"][indices] = output["h"][:, -1].double().cpu().numpy() * unit ** 2
                else:
                    final = {key: value[:, -1].double().cpu().numpy() for key, value in output.items()}
                    weights = np.exp(final["logw"])
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    location = final["loc"] * unit[:, None]
                    scale = np.exp(final["logscale"]) * unit[:, None]
                    mean = np.sum(weights * location, axis=1)
                    component_variance = scale ** 2 * final["nu"] / (final["nu"] - 2)
                    # Stable total-variance formula avoids subtracting nearly
                    # equal large moments for narrow mixtures with large means.
                    central = np.sum(weights * (component_variance + (location - mean[:, None]) ** 2), axis=1)
                    second = central + mean ** 2
                    result["mean"][indices], result["variance"][indices] = mean, central
                    result["second_moment"][indices] = result["h"][indices] = second
                    if targets is not None:
                        # Scoring occurs after the complete history-only forward.
                        # Parameters came from float32 inference; density arithmetic
                        # uses float64 and adds the raw-unit Jacobian exactly once.
                        p = {key: torch.as_tensor(final[key]) for key in ("logw", "loc", "logscale", "nu")}
                        standardized_target = torch.as_tensor(targets[indices] / unit)
                        result["native_nll"][indices] = (
                            -V.mixture_logpdf(p, standardized_target).numpy() + np.log(unit))
        for key, value in result.items():
            if not np.isfinite(value).all() or (key in ("variance", "second_moment", "h") and np.any(value <= 0)):
                raise FloatingPointError("Invalid raw prediction: " + key)
    finally:
        model.train(was_training)
    if "native_nll" not in result:
        result["native_nll"] = None
    return result


def predict(model, past, device=None, target=None):
    targets = None if target is None else [target]
    return {key: None if value is None else float(value[0])
            for key, value in predict_windows(model, [past], device=device, targets=targets).items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "train", "predict"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--historical-root", type=Path, default=HISTORICAL_ROOT)
    p.add_argument("--roster", type=Path, default=ROSTER_PATH)
    p.add_argument("--runtime-receipt", type=Path)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--histories", type=Path, help="NPY array of past-return windows in percentage units")
    p.add_argument("--targets", type=Path, help="Optional NPY raw next returns, used only after inference")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    if a.action == "freeze":
        result = freeze(a.out, smoke=a.smoke, device=a.device, historical_root=a.historical_root,
                        roster_path=a.roster, runtime_receipt=a.runtime_receipt)
        print(json.dumps({"manifest": str(a.out / "manifest.json"), "kind": result["kind"]}))
        return
    if a.manifest is None:
        p.error("--manifest is required for every execution")
    m = verify(a.manifest)
    if bool(a.smoke) != bool(m["specification"]["smoke"]) or a.device != m["specification"]["device"]:
        p.error("Explicit --smoke/--device must match the frozen manifest")
    if a.action == "train":
        if a.arm is None or a.seed is None:
            p.error("--arm and --seed are required")
        print(json.dumps(train(a.manifest, a.out, a.arm, a.seed, resume=a.resume)))
    else:
        if a.checkpoint is None or a.histories is None:
            p.error("--checkpoint and --histories are required")
        if a.out.exists():
            raise FileExistsError("Refusing to overwrite predictions")
        model = load_model(a.checkpoint, a.manifest, a.device)
        targets = np.load(a.targets, allow_pickle=False) if a.targets is not None else None
        result = predict_windows(model, np.load(a.histories, allow_pickle=False), a.batch_size, targets=targets)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, **{key: value for key, value in result.items() if value is not None},
                 native_nll_available=np.asarray(result["native_nll"] is not None))


if __name__ == "__main__":
    main()
