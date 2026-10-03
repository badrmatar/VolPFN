"""Replay the recorded simulator failure and audit its numerical repair.

No market data, model weights, fitting, or remote resources are used here.
"""
import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

import volpfn_prior as VP
from research.matched_prior import simulators as S
from research.seed_replication import causal, training as T
from research.programme_closure import test_simulator_correction as tests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'research/outputs/programme_closure_20260919_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_failure():
    old = tests.archived_prior()
    args = tests.failing_arguments()
    rows, saved, metadata = [], {}, []

    def trace(frame, event, arg):
        if frame.f_code is old.apply_overlays.__code__ and event == 'line' and frame.f_lineno == 260:
            d = frame.f_locals
            i, fresh = int(d['i']), d['fresh']
            lv, r, cumv = d['lv'][i].copy(), d['r'][i].copy(), d['cumv']
            previous = np.r_[0., cumv[fresh[:-1]]]
            differences = cumv[fresh] - previous
            bad = np.flatnonzero(differences <= 0)
            if len(bad):
                saved.update(r=r, logv=lv, fresh=fresh.copy())
                events = []
                for j in bad:
                    start, end = (int(fresh[j-1]) + 1 if j else 0), int(fresh[j])
                    events.append(dict(start=start, end=end, cumulative_variance=float(cumv[end]),
                        previous_cumulative_variance=float(previous[j]), rounded_difference=float(differences[j]),
                        direct_segment_variance=float(np.exp(lv[start:end+1]).sum()),
                        direct_segment_log_variance=float(np.logaddexp.reduce(lv[start:end+1]))))
                rows.append(dict(row=i, finite_input_log_variance_range=[float(lv.min()), float(lv.max())],
                                 invalid_fresh_targets=len(bad), events=events))
        if frame.f_code is causal.raw_batch.__code__ and event == 'line' and frame.f_lineno == 93:
            metadata[:] = frame.f_locals['meta']
        return trace

    error = None
    with patch.object(causal, 'VP', old), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sys.settrace(trace)
        try:
            T.generated_batch(*args)
        except FloatingPointError as exc:
            error = str(exc)
        finally:
            sys.settrace(None)
    assert error == 'raw simulator produced nonfinite observations or latent targets'
    assert len(rows) == 1 and rows[0]['invalid_fresh_targets'] == 62
    fixture = OUTPUT / 'failed_stale_interval.npz'
    np.savez(fixture, **saved)
    fixed = T.generated_batch(*args)
    T.validate_batch(fixed)
    for row in rows:
        row['family'] = metadata[row['row']]['family']
    return dict(seed=8103, attempted_update=4575, data_seed=args[0], batch_size=args[1], length=args[2],
                original_failure=error, source_hash=sha(tests.ARCHIVE), rows=rows,
                corrected_prepared_arrays_all_finite=all(np.isfinite(v).all() for v in fixed.values()
                                                         if isinstance(v, np.ndarray)),
                fixture=str(fixture.relative_to(ROOT)), fixture_sha256=sha(fixture))


def stress():
    result, total = [], 0
    for arm in ('broad', 'matern'):
        for seed in range(8101, 8106):
            spec = T.spec()
            spec['workers'] = 0
            schedule = T.Batches(seed, spec)
            lengths, checked_targets = {}, 0
            begun = time.perf_counter()
            for _ in range(128):
                data_seed, B, L, _, _ = schedule.args()
                batch = S.prepared_batch(arm, data_seed, B, L)
                T.validate_batch(batch)
                assert all(np.isfinite(v).all() for v in batch.values() if isinstance(v, np.ndarray))
                lengths[str(L)] = lengths.get(str(L), 0) + 1
                checked_targets += int(batch['target_mask'].sum())
            result.append(dict(arm=arm, training_seed=seed, updates=128, lengths=lengths,
                               scored_targets=checked_targets, elapsed_seconds=time.perf_counter()-begun,
                               status='passed'))
            total += 128
            print('SIMULATOR_STRESS', arm, seed, total, 'batches passed', flush=True)
    spec = T.spec()
    spec['workers'] = 0
    schedule = T.Batches(8103, spec)
    for update in range(1, 4596):
        args = schedule.args()
        if update < 4555:
            continue
        batch = T.generated_batch(*args)
        T.validate_batch(batch)
        assert all(np.isfinite(v).all() for v in batch.values() if isinstance(v, np.ndarray))
    return dict(batches=total+41, training_positions=(total+41)*8192, arms=result,
                failure_neighborhood=dict(seed=8103, inclusive_updates=[4555, 4595], batches=41, status='passed'),
                scope='Finite-draw stress screen, not a proof that all future simulator draws are safe.')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=platform.python_version(), numpy=np.__version__, market_data_loaded=False,
                  trained_models=False, paid_compute_used=False,
                  original_sha256=sha(tests.ARCHIVE), corrected_sha256=sha(ROOT/'volpfn_prior.py'))
    record['replay'] = replay_failure()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (OUTPUT/'SIMULATOR_TESTS.log').write_text(stream.getvalue())
    assert outcome.wasSuccessful(), stream.getvalue()
    record['regression_tests'] = dict(tests=outcome.testsRun, failures=len(outcome.failures),
                                     errors=len(outcome.errors), status='passed')
    with threadpool_limits(limits=2):
        record['stress'] = stress()
    record['source_files'] = {str(p.relative_to(ROOT)):sha(p) for p in
        (ROOT/'volpfn_prior.py', Path(__file__), Path(tests.__file__))}
    record['status'] = 'passed'
    (OUTPUT/'SIMULATOR_CORRECTION.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':record['status'], 'stress_batches':record['stress']['batches'],
                      'corrected_sha256':record['corrected_sha256']}), flush=True)


if __name__ == '__main__':
    main()
