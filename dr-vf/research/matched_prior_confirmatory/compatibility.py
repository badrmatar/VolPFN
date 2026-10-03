"""Read-only synthetic compatibility audit; never loads market observations/scores.

Run as a module so the historical spawned-worker generator can safely import it.
The audit records failures and exits unsuccessfully rather than silently relaxing
any equality. Initial states are reconstructed unless a step-zero file exists.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
import volpfn
from research.seed_replication import training as T, causal
from research.matched_prior import run as M, simulators as S

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLICATION = ROOT/'research/outputs/seed_replication_session_recovery_20260918'
DEFAULT_OUTPUT = ROOT/'research/outputs/matched_prior_compatibility_v1'
ARRAY_KEYS = ('feats', 'base', 'target_r', 'target_logv', 'target_mask',
              'output_scale', 'output_log_scale')
CORE_KEYS = ('size', 'steps', 'tokens', 'lengths', 'lr', 'warmup', 'aux',
             'optimizer', 'gradient_norm_clip', 'precision', 'normalization_spec')


def compare_arrays(left, right):
    """Require exact key, dtype, shape and byte equality, not numeric closeness."""
    left_keys = {k for k, v in left.items() if isinstance(v, np.ndarray)}
    right_keys = {k for k, v in right.items() if isinstance(v, np.ndarray)}
    if left_keys != right_keys or left_keys != set(ARRAY_KEYS):
        raise AssertionError('array key mismatch')
    hashes = {}
    for key in sorted(left_keys):
        a, b = left[key], right[key]
        if a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError('dtype/shape mismatch: ' + key)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('nonfinite array: ' + key)
        ba, bb = a.tobytes(order='C'), b.tobytes(order='C')
        if ba != bb:
            raise AssertionError('byte mismatch: ' + key)
        hashes[key] = hashlib.sha256(ba).hexdigest()
    return hashes


def check_schedule(seed, original, new):
    """All updates, independently of the expensive 300-batch tensor replay."""
    old = T.Batches(seed, dict(original, workers=0))
    pairs = []
    try:
        schedule = list(M.schedule(seed, new))
        if len(schedule) != original['steps']:
            raise AssertionError('schedule length mismatch')
        for i, candidate in enumerate(schedule):
            expected = old.args()[:3]
            if tuple(candidate) != tuple(expected):
                raise AssertionError('schedule mismatch at update ' + str(i + 1))
            pairs.append(list(candidate))
    finally:
        old.close()
    return dict(updates=len(pairs), status='passed',
                sha256=hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
                first=pairs[0], last=pairs[-1])


def frozen_source_audit(manifest):
    """Hash original source/protocol files only, excluding market artifacts."""
    rows = []
    for text, expected in manifest['hashes'].items():
        path = Path(text)
        if path.suffix not in ('.py', '.sh', '.md') or 'outputs' in path.parts:
            continue
        actual = T.sha(path)
        rows.append(dict(path=str(path), expected=expected, actual=actual,
                         matches=actual == expected))
        if actual != expected:
            raise AssertionError('original frozen source changed: ' + str(path))
    required = [ROOT/'volpfn.py', ROOT/'volpfn_prior.py',
                ROOT/'research/seed_replication/causal.py',
                ROOT/'research/seed_replication/training.py']
    audited = {row['path'] for row in rows}
    if any(str(path) not in audited for path in required):
        raise AssertionError('essential training source absent from frozen manifest')
    return rows


def core_configuration(original, new):
    rows = {}
    for key in CORE_KEYS:
        rows[key] = dict(original=original[key], new=new[key],
                         equal=original[key] == new[key])
        if not rows[key]['equal']:
            raise AssertionError('training configuration mismatch: ' + key)
    if original['prior'] != S.SPECIFICATION['broad_weights']:
        raise AssertionError('broad prior weights differ')
    if original['overlay'] != S.SPECIFICATION['overlay_probabilities']:
        raise AssertionError('overlay probabilities differ')
    if new['cfg'] != volpfn.SIZES[original['size']]:
        raise AssertionError('model configuration differs')
    return rows


def checkpoint_audit(replication, seed, original, manifest_sha, new):
    folder = replication/'base'/str(seed)
    status = json.loads((folder/'status.json').read_text())
    final = folder/'final.pt'
    final_hash = T.sha(final)
    if status.get('status') != 'complete' or status.get('completed_updates') != original['steps']:
        raise AssertionError('seed is not a completed full-budget training run')
    if status.get('final_checkpoint_sha256') != final_hash:
        raise AssertionError('final checkpoint differs from completed training record')
    value = torch.load(final, map_location='cpu', weights_only=False)
    if value['step'] != original['steps'] or value['seed'] != seed:
        raise AssertionError('checkpoint step or seed mismatch')
    if value['training_spec'] != original or value['manifest_sha256'] != manifest_sha:
        raise AssertionError('checkpoint does not match original frozen specification')
    if value['cfg'] != new['cfg'] or value['normalization_spec'] != causal.NORMALIZATION_SPEC:
        raise AssertionError('checkpoint model/normalization mismatch')
    if value['causal_spec'] != causal.CAUSAL_SPEC:
        raise AssertionError('checkpoint causal specification mismatch')
    checkpoint_rows = []
    archived_initial = None
    for path in sorted(folder.glob('*.pt')):
        candidate = value if path == final else torch.load(path, map_location='cpu', weights_only=False)
        checkpoint_rows.append(dict(file=path.name, step=int(candidate['step']), sha256=T.sha(path)))
        if int(candidate['step']) == 0:
            archived_initial = candidate['state_dict']
    torch.manual_seed(seed)
    reconstructed_old = volpfn.VolPFN(**volpfn.SIZES[original['size']])
    torch.manual_seed(seed)
    reconstructed_new = volpfn.VolPFN(**new['cfg'])
    old_hash, new_hash = M.state_hash(reconstructed_old), M.state_hash(reconstructed_new)
    if old_hash != new_hash:
        raise AssertionError('reconstructed initial states differ')
    archived_match = None
    if archived_initial is not None:
        reconstructed_old.load_state_dict(archived_initial)
        archived_match = M.state_hash(reconstructed_old) == new_hash
        if not archived_match:
            raise AssertionError('archived initial state differs from reconstruction')
    return dict(path=str(final), sha256=final_hash, step=value['step'], seed=seed,
                final_selected_by='fixed 20000th update; synthetic-validation best.pt is not reused',
                archived_checkpoints=checkpoint_rows,
                reconstructed_old_initial_sha256=old_hash,
                reconstructed_new_initial_sha256=new_hash,
                archived_initial_available=archived_initial is not None,
                archived_initial_matches=archived_match,
                initialization_evidence=('archived step-zero comparison' if archived_initial is not None else
                    'seed-labelled pairs with initial-state matching reconstructed from frozen source/configuration; '
                    'no archived step-zero comparison is possible; replay is not an observation of historical initialization'))


def run(output=DEFAULT_OUTPUT, replication=DEFAULT_REPLICATION, batches=300, seed=8101):
    output, replication = Path(output).resolve(), Path(replication).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', started_utc=T.now(), seed=seed,
                  requested_batches=batches, market_data_or_scores_loaded=False,
                  thread_limit=1, environment=dict(python=platform.python_version(),
                      numpy=np.__version__, torch=str(torch.__version__)))
    started = time.perf_counter()
    data = None
    try:
        if batches < 300:
            raise ValueError('reuse gate requires at least 300 full-size batches')
        torch.set_num_threads(1)
        manifest_path = replication/'manifest.json'
        manifest_sha = T.sha(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        record['original_manifest_sha256'] = manifest_sha
        original = manifest['base']
        if original['workers'] != 2 or original['prefetch_depth'] != 4:
            raise AssertionError('not the declared two-worker/four-prefetch procedure')
        record['frozen_sources'] = frozen_source_audit(manifest)
        new = M.specification('full', device=original['device'])
        record['core_configuration'] = core_configuration(original, new)
        record['configuration_differences'] = {k: dict(old=original.get(k), new=new.get(k))
            for k in sorted(set(original) | set(new)) if original.get(k) != new.get(k)}
        record['schedule'] = check_schedule(seed, original, new)
        record['checkpoint'] = checkpoint_audit(replication, seed, original, manifest_sha, new)
        data = T.Batches(seed, original)
        row_digest = hashlib.sha256()
        checked = 0
        with (output/'batch_checks.jsonl').open('x') as stream:
            for i, (data_seed, B, L) in enumerate(M.schedule(seed, new)):
                if i >= batches:
                    break
                old_batch = data.next()
                new_batch = S.prepared_batch('broad', data_seed, B, L)
                hashes = compare_arrays(old_batch, new_batch)
                row = dict(update=i+1, data_seed=data_seed, batch_size=B, length=L,
                           status='bitwise_identical', array_sha256=hashes)
                text = json.dumps(row, sort_keys=True)
                stream.write(text + '\n')
                row_digest.update(text.encode())
                checked += 1
                if checked % 25 == 0:
                    print('COMPATIBILITY', checked, '/', batches, 'batches passed', flush=True)
        if checked != batches:
            raise AssertionError('insufficient replayed batches')
        record['tensor_replay'] = dict(status='passed', batches=checked,
            arrays_per_batch=len(ARRAY_KEYS), array_keys=list(ARRAY_KEYS),
            input_positions=checked*original['tokens'], row_sha256=row_digest.hexdigest(),
            metadata='prepared batches contain no metadata; all seven array outputs compared including output_scale')
        if frozen_source_audit(manifest) != record['frozen_sources']:
            raise AssertionError('source hashes changed during audit')
        record['audit_source_hashes'] = {str(path.relative_to(ROOT)): T.sha(path) for path in
            [Path(__file__), ROOT/'research/matched_prior/run.py', ROOT/'research/matched_prior/simulators.py']}
        record.update(status='passed', completed_utc=T.now(),
            scope='configuration/source and synthetic sequence compatibility, not bitwise full optimization replay; '
                  'only seed 8101 completed-final provenance verified; validate each future final before reuse')
    except BaseException as exc:
        record.update(status='failed', failure=type(exc).__name__ + ': ' + str(exc), failed_utc=T.now())
        (output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if data is not None:
            data.close()
        record['elapsed_seconds'] = time.perf_counter() - started
        T.save_json(output/'audit.json', record)
    print(json.dumps({k:record[k] for k in ('status', 'elapsed_seconds', 'tensor_replay')}), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--replication', type=Path, default=DEFAULT_REPLICATION)
    parser.add_argument('--batches', type=int, default=300)
    parser.add_argument('--seed', type=int, default=8101)
    args = parser.parse_args()
    run(args.output, args.replication, args.batches, args.seed)
