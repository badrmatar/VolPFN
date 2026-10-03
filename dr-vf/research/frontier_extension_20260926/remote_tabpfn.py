"""Portable, squared-return-only TabPFN worker; no losses or cloud operations.

Transfer only this file, the frozen bundle, referenced market NPZs, the original
standalone adapter/checkpoint, a remote manifest, and all 72 calibration references.
All manifest paths are relative to --root. The manifest itself is SHA-pinned by
--manifest-sha256; it must contain kind, frozen, bundle, worker, adapter,
checkpoint, files, packages, bridge_references and fixed bridge_tolerances.
Full runs additionally require approved_runtime copied from the accepted pilot.
Completed chunks and terminal attempts are immutable. Resume only the same
manifest, selection, runtime and unfinished attempt. Failed attempts require a
new output directory. A pilot accepts exactly the first 16 or 32 calibration
origins, never evaluation observations.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
for _key in ('TABPFN_DISABLE_TELEMETRY', 'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK'):
    os.environ[_key] = '1'

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import time
import traceback
import uuid
import numpy as np

KIND = 'frontier_extension_tabpfn_remote_v1'
SEED = 9182026
ADAPTER_SHA256 = '7f3973e0c0a3e555b4cfb91efef238744736508299421f2d0ffe3ff13801850d'
PACKAGES = {'torch': '2.14.0', 'numpy': '2.5.3', 'scipy': '1.18.1',
            'pandas': '2.3.3', 'tabpfn': '9.0.0', 'tabpfn-time-series': '1.3.0',
            'huggingface-hub': '1.32.0'}
BRIDGE_TOLERANCES = {'rtol': .005, 'atol': .0002}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def chunk_path(out, index):
    return Path(out) / 'chunks' / ('chunk-%06d' % index) / 'forecast.json'


def publish_chunk(path, block):
    """One rename publishes data and checksum together; interrupted work is private."""
    path = Path(path)
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent.with_name('.staging-' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        atomic(staging / path.name, block, immutable=True)
        atomic(staging / path.with_suffix('.sha256.json').name,
               {'sha256': sha(staging / path.name)}, immutable=True)
        if path.parent.exists():
            raise FileExistsError('Completed chunk is immutable')
        os.rename(staging, path.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def atomic(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(tmp, path)  # fails rather than overwriting any completed file
        else:
            os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def safe_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError('Manifest path escapes root')
    return path


def calibration_references(root, manifest, bundle):
    """Require the exact canonical 72 keys, never an optional/empty bridge."""
    name = manifest.get('bridge_references')
    if not name or name not in manifest['files']:
        raise ValueError('All 72 calibration bridge references must be supplied and hashed')
    path = safe_path(root, name)
    if sha(path) != manifest['files'][name]:
        raise ValueError('Calibration bridge reference hash mismatch')
    if manifest.get('bridge_tolerances') != BRIDGE_TOLERANCES:
        raise ValueError('Calibration bridge tolerances differ from the fixed contract')
    expected = {(r['phase'], r['market'], r['date']) for r in bundle['rows']
                if r['phase'] == 'calibration'}
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != 72 or len(expected) != 72:
        raise ValueError('Exactly 72 canonical calibration references are required')
    references = {(r['phase'], r['market'], r['date']): r for r in rows}
    if len(references) != 72 or set(references) != expected:
        raise ValueError('Calibration bridge keys differ from the canonical 72 origins')
    for row in rows:
        h = row.get('h', row.get('second_moment'))
        if not isinstance(h, (int, float)) or not np.isfinite(h) or h <= 0:
            raise ValueError('Calibration reference h must be finite and positive')
    return references


def approved_runtime(manifest, actual=None):
    """Full forecast/import identity includes hardware and every pilot setting."""
    approved = manifest.get('approved_runtime')
    required = {'packages', 'python', 'cuda', 'gpu', 'precision', 'tf32',
                'random_state', 'n_estimators', 'thread_limit'}
    if not isinstance(approved, dict) or not required <= set(approved):
        raise ValueError('Full run requires the complete approved pilot runtime')
    if (set(approved['packages']) != set(PACKAGES)
            or any(approved['packages'][k].split('+')[0] != v for k, v in PACKAGES.items())
            or any(not isinstance(approved[k], str) or not approved[k] for k in ('python', 'cuda', 'gpu'))
            or approved['precision'] != 'float32' or approved['tf32'] is not False
            or approved['random_state'] != SEED or approved['n_estimators'] != 8
            or approved['thread_limit'] != 1):
        raise ValueError('Approved pilot runtime violates the frozen adapter contract')
    if actual is not None and actual != approved:
        raise ValueError('Full runtime differs from the approved pilot runtime')
    return approved


def verify(root, manifest_path, expected_sha, evaluation=False):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    if sha(manifest_path) != expected_sha:
        raise ValueError('Remote manifest SHA mismatch')
    manifest = read(manifest_path)
    if manifest['kind'] != KIND or (evaluation and manifest.get('frozen') is not True):
        raise ValueError('Evaluation requires an explicitly frozen remote manifest')
    files = manifest['files']
    for name, digest in files.items():
        if sha(safe_path(root, name)) != digest:
            raise ValueError('Input/source SHA mismatch: ' + name)
    for key in ('bundle', 'worker', 'adapter', 'checkpoint'):
        if manifest[key] not in files:
            raise ValueError('Required source/input absent from hash manifest: ' + key)
    if sha(__file__) != files[manifest['worker']]:
        raise ValueError('Executed worker source differs from frozen source')
    if files[manifest['adapter']] != ADAPTER_SHA256 or manifest['packages'] != PACKAGES:
        raise ValueError('Original standalone adapter or pinned package configuration changed')
    bundle = read(safe_path(root, manifest['bundle']))
    if files[manifest['bundle']] != manifest['bundle_sha256']:
        raise ValueError('Frozen bundle identity mismatch')
    if manifest['checkpoint'] not in bundle['inputs']:
        raise ValueError('Checkpoint absent from frozen bundle')
    if bundle['inputs'][manifest['checkpoint']] != files[manifest['checkpoint']]:
        raise ValueError('Checkpoint differs from original frozen bundle')
    for name, rel in bundle['market_paths'].items():
        if rel not in files or files[rel] != bundle['inputs'][rel]:
            raise ValueError('Market not covered by original frozen hashes: ' + name)
    calibration_references(root, manifest, bundle)
    if evaluation:
        approved_runtime(manifest)
    return manifest, bundle


def import_attempts(local_out, remote_root, manifest_path, manifest_digest, attempts):
    """Validate complete coverage before publishing any canonical market cells."""
    from research.frontier_extension_20260926 import run as local
    local_out = Path(local_out)
    bundle, sealed = local.verify(local_out)
    freeze_digest = sha(local_out / 'FREEZE.json')
    manifest, remote_bundle = verify(remote_root, manifest_path, manifest_digest, evaluation=True)
    if (manifest.get('extension_freeze_sha256') != freeze_digest
            or manifest['bundle_sha256'] != sha(local_out / 'bundle.json')
            or remote_bundle != bundle):
        raise ValueError('Remote attempt differs from local frozen bundle/source identity')
    all_rows, receipts, shared_runtime, seen_keys = [], [], None, set()
    market_cache = MarketCache(remote_root, bundle)
    references = calibration_references(remote_root, manifest, bundle)
    all_bridge = []
    for directory in map(Path, attempts):
        terminal, identity = read(directory / 'terminal.json'), read(directory / 'attempt.json')
        if terminal['status'] != 'complete' or terminal['manifest_sha256'] != manifest_digest:
            raise ValueError('Incomplete attempt or different remote manifest')
        if identity['manifest_sha256'] != manifest_digest or identity['bundle_sha256'] != manifest['bundle_sha256']:
            raise ValueError('Attempt identity mismatch')
        selection = identity['selection']
        if selection['pilot'] is not None:
            raise ValueError('Pilot output cannot replace the full frozen attempt')
        approved_runtime(manifest, identity['runtime'])
        if shared_runtime is None:
            shared_runtime = identity['runtime']
        if identity['runtime'] != shared_runtime:
            raise ValueError('Cannot mix different remote runtimes')
        selected = select_rows(bundle, None, selection['shard_index'], selection['shard_count'])
        expected_keys = [[r['phase'], r['market'], r['date']] for r in selected]
        if identity['selected_keys'] != expected_keys:
            raise ValueError('Attempt selection changed')
        attempt_rows, chunk_hashes, attempt_bridge = [], {}, []
        count = identity['chunk_size']
        for start in range(0, len(selected), count):
            path = chunk_path(directory, start // count)
            if read(path.with_suffix('.sha256.json'))['sha256'] != sha(path):
                raise ValueError('Remote chunk checksum mismatch')
            chunk = read(path)
            if (chunk['manifest_sha256'] != manifest_digest
                    or chunk['bundle_sha256'] != manifest['bundle_sha256']
                    or chunk['origins'] != selected[start:start + count]):
                raise ValueError('Remote chunk identity/target mismatch')
            local.validate_rows(chunk['records'], chunk['origins'], 'tabpfn_ts', None)
            for row in chunk['records']:
                k = (row['phase'], row['market'], row['date'])
                if k in seen_keys:
                    raise ValueError('Overlapping remote attempts')
                seen_keys.add(k)
                if (row['mean'] != 0 or row['variance'] != row['second_moment']
                        or row['substituted'] is not False or row['track'] != 'squared'):
                    raise ValueError('Remote squared-return semantics changed')
                past = market_cache.context(row)
                expected_floor = max(1e-12, 1e-8 * float(np.mean(past ** 2)))
                if (not np.isfinite(row['raw_point']) or row['positive_floor'] != expected_floor
                        or row['variance'] != max(row['raw_point'], expected_floor)
                        or row['floor_applied'] != (row['raw_point'] < expected_floor)):
                    raise ValueError('Frozen positivity floor changed')
            checks = [bridge_check(r, references, BRIDGE_TOLERANCES) for r in chunk['records']
                      if r['phase'] == 'calibration']
            if chunk['bridge'] != checks:
                raise ValueError('Missing or changed calibration bridge diagnostics')
            if not all(x['passed'] for x in checks):
                raise ValueError('Failed remote calibration bridge')
            attempt_bridge.extend(checks)
            attempt_rows.extend(chunk['records'])
            chunk_hashes[str(path.relative_to(directory))] = sha(path)
        if len(attempt_rows) != len(selected) or terminal['completed'] != len(selected):
            raise ValueError('Remote attempt has incomplete coverage')
        if terminal.get('bridge_checked') != len(attempt_bridge) or terminal.get('bridge_passed') is not True:
            raise ValueError('Remote terminal bridge coverage changed')
        all_rows.extend(attempt_rows)
        all_bridge.extend(attempt_bridge)
        receipts.append(dict(attempt_sha256=sha(directory / 'attempt.json'),
                             terminal_sha256=sha(directory / 'terminal.json'), chunks=chunk_hashes))
    local.validate_rows(all_rows, bundle['rows'], 'tabpfn_ts', None)
    if len(all_bridge) != 72 or {(r['phase'], r['market'], r['date']) for r in all_bridge} != set(references):
        raise ValueError('Import requires agreement on all 72 calibration origins')
    audit = dict(model='tabpfn_ts', manifest_sha256=manifest_digest, runtime=shared_runtime,
                 source='validated_remote_squared_return_attempt', attempts=receipts,
                 calibration_bridge=all_bridge, bridge_tolerances=BRIDGE_TOLERANCES)
    groups = {m: sorted((r for r in all_rows if r['market'] == m), key=lambda r: r['index'])
              for m in bundle['markets']}
    # Preflight every existing cell before the first publication; allow exact resume only.
    for market, rows in groups.items():
        existing = local.cell(local_out, local.relative_cell('tabpfn_ts', None, market), freeze_digest)
        if existing is not None and (existing['rows'] != rows or existing['audit'] != audit):
            raise ValueError('Existing TabPFN cell belongs to a different attempt')
    for market, rows in groups.items():
        rel = local.relative_cell('tabpfn_ts', None, market)
        if local.cell(local_out, rel, freeze_digest) is None:
            local.cell(local_out, rel, freeze_digest, dict(rows=rows, audit=audit))
    return dict(imported_records=len(all_rows), markets=len(groups), manifest_sha256=manifest_digest)


def select_rows(bundle, pilot=None, shard_index=0, shard_count=1):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('Invalid shard index/count')
    rows = bundle['rows']
    if pilot is not None:
        if pilot not in (16, 32) or shard_index != 0 or shard_count != 1:
            raise ValueError('Pilot requires 16 or 32 calibration rows without sharding')
        rows = [r for r in rows if r['phase'] == 'calibration'][:pilot]
        if len(rows) != pilot:
            raise ValueError('Insufficient calibration origins for pilot')
    else:
        rows = [r for i, r in enumerate(rows) if i % shard_count == shard_index]
    if not rows:
        raise ValueError('Empty selection')
    keys = [(r['phase'], r['market'], r['date']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate selected origins')
    return rows


def covariates(past):
    """Exact NumPy feature arithmetic copied from causal.feature_batch.

    Eight float32 causal covariates reset at each 1024-return context; target
    rows are past[1:] and pair with features[:-1], never their own returns.
    """
    r = np.asarray(past, dtype=np.float64)[None]
    if r.shape != (1, 1024) or not np.isfinite(r).all():
        raise ValueError('Exactly 1024 finite prior returns are required')
    B, T = r.shape
    nz = r != 0
    logabs = np.full(r.shape, -np.inf)
    np.log(np.abs(r), out=logabs, where=nz)
    logsum = np.logaddexp.accumulate(2 * logabs, axis=1)
    count = np.cumsum(nz, axis=1)
    log_scale = np.zeros(r.shape)
    np.subtract(logsum, np.log(np.maximum(count, 1)), out=log_scale, where=count > 0)
    log_scale *= 0.5
    standardized = np.zeros(r.shape)
    standardized[nz] = np.sign(r[nz]) * np.exp(logabs[nz] - log_scale[nz])
    feats = np.empty((B, T, 8), dtype=np.float32)
    feats[..., 0] = np.arcsinh(standardized)
    feats[..., 1] = np.log(standardized * standardized + 1e-6) / 4
    feats[..., 2] = ~nz
    feats[..., 3] = np.sign(r)
    for j, half_life in enumerate((5.0, 20.0, 100.0)):
        lam = 0.5 ** (1.0 / half_life)
        log_lam, log_one_minus_lam = math.log(lam), math.log1p(-lam)
        log_e = np.zeros(B)
        seen = np.zeros(B, dtype=bool)
        col = np.empty((B, T), dtype=np.float64)
        for t in range(T):
            first = nz[:, t] & ~seen
            update = nz[:, t] & seen
            log_e[first] = 2 * logabs[first, t]
            log_e[update] = np.logaddexp(log_lam + log_e[update],
                                        log_one_minus_lam + 2 * logabs[update, t])
            seen |= nz[:, t]
            col[:, t] = log_e - 2 * log_scale[:, t]
        feats[..., 4 + j] = col / 4
    feats[..., 7] = np.log1p(np.arange(T))[None, :] / 8
    return feats[0, :-1].copy(), feats[0, -1:].copy()


class MarketCache:
    def __init__(self, root, bundle):
        self.root, self.bundle, self.cache = Path(root), bundle, {}

    def context(self, row):
        name = row['market']
        if name not in self.cache:
            with np.load(safe_path(self.root, self.bundle['market_paths'][name]), allow_pickle=False) as z:
                self.cache[name] = {key: z[key].copy() for key in ('date', 'r', 'rk')}
        market = self.cache[name]
        t = row['index']
        if not isinstance(t, int) or t < 1024 or t >= len(market['r']):
            raise ValueError('Origin outside full causal history')
        if str(market['date'][t])[:10] != row['date']:
            raise ValueError('Market date/index mismatch')
        if float(market['r'][t]) ** 2 != row['r2']:
            raise ValueError('Frozen squared-return target mismatch')
        rk = float(market['rk'][t])
        target = rk if np.isfinite(rk) and rk > 0 else None
        if target != row['rk']:
            raise ValueError('Frozen RK target mismatch')
        return np.asarray(market['r'][t - 1024:t], float).copy()


def record(origin, prediction, past, duration):
    point = float(prediction['point'])
    quantiles = np.asarray(prediction['quantiles'], float)
    if not np.isfinite(point) or quantiles.shape != (9,) or not np.isfinite(quantiles).all():
        raise ValueError('Invalid TabPFN squared-return forecast')
    floor = max(1e-12, 1e-8 * float(np.mean(past ** 2)))
    h = max(point, floor)
    return dict(origin, model='tabpfn_ts', seed=None, mean=0.0, variance=h,
                second_moment=h, floor_applied=bool(point < floor), substituted=False,
                raw_point=point, squared_quantiles=quantiles.tolist(), positive_floor=floor,
                quantiles_rearranged=bool(prediction.get('rearranged', False)),
                point_semantics=prediction['point_semantics'], inference_seconds=float(duration),
                context_count=1024, track='squared')


def bridge_check(row, references, tolerances):
    key = (row['phase'], row['market'], row['date'])
    ref = references.get(key)
    if ref is None:
        return None
    expected = float(ref.get('h', ref.get('second_moment')))
    good = bool(np.isclose(row['second_moment'], expected, **tolerances))
    return dict(phase=key[0], market=key[1], date=key[2], passed=good,
                reference_h=expected, output_h=row['second_moment'],
                absolute_delta=abs(row['second_moment'] - expected), tolerances=tolerances)


def runtime_and_adapter(root, manifest, evaluation=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    packages = {name: importlib.metadata.version(name) for name in manifest['packages']}
    for name, expected in manifest['packages'].items():
        if packages[name].split('+')[0] != expected:
            raise RuntimeError('Frozen package mismatch: ' + name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = dict(packages=packages, python=platform.python_version(), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), precision='float32', tf32=False,
                   random_state=SEED, n_estimators=8, thread_limit=1)
    if evaluation:
        approved_runtime(manifest, runtime)
    spec = importlib.util.spec_from_file_location('frozen_tabpfn_adapter', safe_path(root, manifest['adapter']))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return runtime, module.TabPFN(safe_path(root, manifest['checkpoint'])), torch.cuda.synchronize


def run(root, out, manifest, bundle, manifest_digest, rows, selection,
        adapter, runtime, synchronize=lambda: None, chunk_size=16):
    """Injectable worker core for tests; actual CLI always verifies hashes first."""
    if chunk_size < 1:
        raise ValueError('Chunk size must be positive')
    references = calibration_references(root, manifest, bundle)
    if selection.get('pilot') is None:
        if manifest.get('frozen') is not True:
            raise ValueError('Full run requires a frozen manifest')
        approved_runtime(manifest, runtime)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(kind=KIND, manifest_sha256=manifest_digest,
                    bundle_sha256=manifest['bundle_sha256'], selection=selection,
                    chunk_size=chunk_size, selected_keys=[(r['phase'], r['market'], r['date']) for r in rows],
                    runtime=runtime)
    identity = json.loads(json.dumps(identity))
    if (out / 'terminal.json').exists():
        raise FileExistsError('Completed/failed attempt is immutable; choose a new output directory')
    if (out / 'attempt.json').exists():
        if read(out / 'attempt.json') != identity:
            raise ValueError('Resume identity/runtime mismatch')
    else:
        if list(out.iterdir()):
            raise ValueError('Unidentified nonempty output directory')
        atomic(out / 'attempt.json', identity, immutable=True)
    lock = (out / 'worker.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise RuntimeError('Another process owns this remote attempt')
    started = time.perf_counter()
    completed, timings, bridge_results = 0, [], []
    cache = MarketCache(root, bundle)
    tolerances = BRIDGE_TOLERANCES
    try:
        for start in range(0, len(rows), chunk_size):
            selected = rows[start:start + chunk_size]
            path = chunk_path(out, start // chunk_size)
            digest_path = path.with_suffix('.sha256.json')
            if path.exists():
                if not digest_path.exists() or read(digest_path)['sha256'] != sha(path):
                    raise ValueError('Completed chunk digest missing/mismatch')
                block = read(path)
                if block['manifest_sha256'] != manifest_digest or block['origins'] != selected:
                    raise ValueError('Completed chunk identity/selection mismatch')
            else:
                if digest_path.exists():
                    raise ValueError('Orphan chunk digest')
                predictions, checks = [], []
                for origin in selected:
                    past = cache.context(origin)
                    x, xn = covariates(past)
                    before = time.perf_counter()
                    prediction = adapter.predict(past, 'squared', x, xn)
                    synchronize()
                    result = record(origin, prediction, past, time.perf_counter() - before)
                    check = bridge_check(result, references, tolerances)
                    if check is not None:
                        checks.append(check)
                        if not check['passed']:
                            atomic(out / 'failed_bridge.json', check, immutable=True)
                            raise RuntimeError('Frozen CPU/CUDA calibration bridge failed')
                    predictions.append(result)
                block = dict(manifest_sha256=manifest_digest, bundle_sha256=manifest['bundle_sha256'],
                             origins=selected, records=predictions, bridge=checks)
                publish_chunk(path, block)
            completed += len(block['records'])
            timings.extend(r['inference_seconds'] for r in block['records'])
            bridge_results.extend(block['bridge'])
            atomic(out / 'progress.json', dict(completed=completed, total=len(rows),
                   elapsed_seconds=time.perf_counter() - started,
                   median_inference_seconds=float(np.median(timings))))
            print('CHUNK', start // chunk_size, completed, len(rows), flush=True)
        terminal = dict(status='complete', completed=completed, total=len(rows),
                        elapsed_seconds=time.perf_counter() - started,
                        inference_seconds_total=float(sum(timings)),
                        inference_seconds_mean=float(np.mean(timings)),
                        inference_seconds_median=float(np.median(timings)),
                        bridge_checked=len(bridge_results), bridge_passed=all(r['passed'] for r in bridge_results),
                        bridge_status='passed' if bridge_results else 'no_matching_reference',
                        manifest_sha256=manifest_digest)
        atomic(out / 'terminal.json', terminal, immutable=True)
        return terminal
    except BaseException as exc:
        atomic(out / 'terminal.json', dict(status='failed', completed=completed,
               error=repr(exc), traceback=traceback.format_exc(), manifest_sha256=manifest_digest), immutable=True)
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--pilot', type=int, choices=(16, 32))
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--import-out', type=Path)
    parser.add_argument('--attempts', type=Path, nargs='+')
    args = parser.parse_args()
    if args.import_out is not None:
        if not args.attempts or args.out is not None or args.pilot is not None:
            parser.error('--import-out requires --attempts, without --out/--pilot')
        print(json.dumps(import_attempts(args.import_out, args.root, args.manifest,
                                        args.manifest_sha256, args.attempts)))
        return
    if args.out is None:
        parser.error('--out is required for inference')
    manifest, bundle = verify(args.root, args.manifest, args.manifest_sha256, evaluation=args.pilot is None)
    rows = select_rows(bundle, args.pilot, args.shard_index, args.shard_count)
    runtime, adapter, synchronize = runtime_and_adapter(args.root, manifest, evaluation=args.pilot is None)
    run(args.root, args.out, manifest, bundle, args.manifest_sha256, rows,
        dict(pilot=args.pilot, shard_index=args.shard_index, shard_count=args.shard_count),
        adapter, runtime, synchronize, args.chunk_size)


if __name__ == '__main__':
    main()
