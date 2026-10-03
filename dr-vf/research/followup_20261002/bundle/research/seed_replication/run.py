"""Freeze, run the engineering pilot, then train and evaluate the seed battery."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from . import causal, evaluation, statistics, training

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT/'research/outputs/seed_replication_v1'


def verified_manifest(out):
    path = Path(out)/'manifest.json'
    if training.sha(path) != path.with_suffix('.json.sha256').read_text().strip():
        raise RuntimeError('Replication manifest changed')
    value = json.loads(path.read_text())
    for filename, digest in value['hashes'].items():
        if training.sha(filename) != digest:
            raise RuntimeError('Frozen input changed: '+filename)
    return value


def freeze(out=DEFAULT_OUT, device='mps'):
    out = Path(out).resolve()
    if (out/'manifest.json').exists():
        return verified_manifest(out)
    if (out/'pilot').exists() or (out/'base').exists():
        raise RuntimeError('Cannot freeze after a named seed has started')
    out.mkdir(parents=True, exist_ok=True)
    inventory = evaluation.inventory_inputs()
    hashes = dict(inventory['hashes'])
    sources = list(PACKAGE.glob('*.py')) + [PACKAGE/'PROTOCOL.md', ROOT/'volpfn.py', ROOT/'volpfn_prior.py']
    for source in sources:
        hashes[str(source.resolve())] = training.sha(source)
    masks = evaluation.expected_rows(inventory)
    mask_paths = {}
    for key, frame in masks.items():
        path = out/'masks'/(key.replace('::', '__')+'.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[['market','date']].to_csv(path, index=False)
        hashes[str(path)] = training.sha(path)
        mask_paths[key] = str(path)
    value = dict(created_utc=training.now(), status='frozen_before_named_training',
                 hashes=hashes, evaluation_inventory=inventory, expected_masks=mask_paths,
                 pilot=training.spec('pilot', device), base=training.spec('base', device),
                 inference=dict(L=1024, stride=512, min_history=100, batch_windows=4, device='cpu', threads=2),
                 statistics=dict(reps=2000, block=60, seed=927601, margin=.003),
                 headline='arithmetic mean performance of all five new base seeds; retained checkpoint separate',
                 normalization_spec=causal.NORMALIZATION_SPEC)
    training.save_json(out/'manifest.json', value)
    (out/'manifest.json.sha256').write_text(training.sha(out/'manifest.json')+'\n')
    print('FROZEN', training.sha(out/'manifest.json'), 'inputs',len(hashes), flush=True)
    return verified_manifest(out)


def checkpoint_probe(path):
    model = causal.load_model(path, 'cpu')
    torch.set_num_threads(2)
    rng = np.random.default_rng(27183)
    r = rng.standard_normal(1100)
    r[:12] = 0.
    reference = causal.predict_series(model, r, levels=(), min_history=100)
    maximum = 0.
    for t in (100, 511, 512, 1023, 1024):
        modified = r.copy()
        modified[t:] = rng.standard_normal(len(r)-t)*20
        a = causal.predict_series(model, modified, levels=(), min_history=100)
        b = causal.predict_series(model, r[:t+1], levels=(), min_history=100)
        for field in ('h','mean','logw','loc','scale','nu'):
            for other in (a,b):
                np.testing.assert_allclose(reference[field][100:t+1], other[field][100:t+1], rtol=1e-6, atol=1e-6)
                maximum = max(maximum, float(np.max(np.abs(reference[field][100:t+1]-other[field][100:t+1]))))
    return dict(status='passed', seed_checkpoint_sha256=training.sha(path), max_absolute_delta=maximum,
                note='Predictive parameters invariant; realized log scores are not expected to be target-invariant.')


def existing_seed(path, *, manifest_sha=None, specification=None, seed=None):
    path = Path(path)
    if not path.exists():
        return None
    file = path/'status.json'
    if not file.exists():
        raise RuntimeError('Existing seed directory lacks a status record: '+str(path))
    record = json.loads(file.read_text())
    if manifest_sha is not None and record.get('manifest_sha256') != manifest_sha:
        raise RuntimeError('Seed belongs to a different manifest')
    if specification is not None and record.get('specification') != specification:
        raise RuntimeError('Seed training specification changed')
    if seed is not None and record.get('seed') != seed:
        raise RuntimeError('Seed identity changed')
    if record['status'] == 'complete':
        for name in ('best', 'final'):
            if training.sha(path/(name+'.pt')) != record[name+'_checkpoint_sha256']:
                raise RuntimeError('Completed checkpoint changed: '+str(path))
    if record['status'] == 'running':
        record.update(status='interrupted', interruption_observed_utc=training.now(),
                      failure_code='unfinished_prior_invocation_no_restart')
        training.save_json(file, record)
    return record


def pilot(out=DEFAULT_OUT):
    out = Path(out).resolve()
    manifest = verified_manifest(out)
    digest = training.sha(out/'manifest.json')
    if (out/'pilot_summary.json').exists():
        summary = json.loads((out/'pilot_summary.json').read_text())
        if summary['manifest_sha256'] != digest:
            raise RuntimeError('Pilot summary belongs to another manifest')
        for filename, expected in summary.get('hashes', {}).items():
            if training.sha(filename) != expected:
                raise RuntimeError('Completed pilot artifact changed')
        if summary['status'] == 'passed':
            for seed in manifest['pilot']['seeds']:
                existing_seed(out/'pilot'/str(seed), manifest_sha=digest, specification=manifest['pilot'], seed=seed)
        return summary
    rows = []
    gate_failures = []
    for seed in manifest['pilot']['seeds']:
        verified_manifest(out)
        path = out/'pilot'/str(seed)
        previous = existing_seed(path, manifest_sha=digest, specification=manifest['pilot'], seed=seed)
        record = previous or training.train_seed(seed, manifest['pilot'], path, digest)
        rows.append(record)
        if record['status'] != 'complete':
            break
        gate_status = path/'gate_status.json'
        try:
            if gate_status.exists():
                previous_gate = json.loads(gate_status.read_text())
                if previous_gate['status'] != 'passed':
                    raise RuntimeError('Prior failed/interrupted pilot gate cannot be silently retried')
            else:
                training.save_json(gate_status, dict(status='running', started_utc=training.now()))
                probe = checkpoint_probe(path/'best.pt')
                training.save_json(path/'causality_probe.json', probe)
                model = causal.load_model(path/'best.pt', 'cpu')
                evaluation.synthetic_integration(path/'synthetic_evaluation', predictor=causal.predict_series, model=model)
                training.save_json(gate_status, dict(status='passed', completed_utc=training.now()))
        except BaseException as error:
            failure = dict(seed=seed, status='failed', error=str(error), traceback=traceback.format_exc())
            training.save_json(gate_status, failure)
            gate_failures.append(failure)
            if not isinstance(error, Exception):
                raise
            break
    passed = (not gate_failures and len(rows) == len(manifest['pilot']['seeds'])
              and all(r['status']=='complete' for r in rows))
    value = dict(status='passed' if passed else 'failed', kind='synthetic_engineering_only',
                 manifest_sha256=digest, seeds=rows, completed_utc=training.now(),
                 market_outcomes_evaluated=False, gate_failures=gate_failures,
                 hashes={str(p.resolve()):training.sha(p) for p in (out/'pilot').rglob('*') if p.is_file()})
    training.save_json(out/'pilot_summary.json', value)
    return value


def aggregate(out, manifest, evaluated, failed):
    registry = evaluation.comparison_registry()
    families = {}
    for row in registry:
        key = row['family']+'::'+row['comparison']
        families.setdefault(row['family'], []).append(key)
    masks = {key: pd.read_csv(path, dtype=str) for key,path in manifest['expected_masks'].items()}
    pieces = []
    for seed in evaluated:
        frame = pd.read_csv(out/'evaluation'/str(seed)/'loss_differences.csv.gz', dtype={'market':str,'date':str})
        frame['seed'] = seed
        frame['contrast'] = frame['family']+'::'+frame['comparison']
        pieces.append(frame[['seed','market','date','contrast','gain']])
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=['seed','market','date','contrast','gain'])
    result = statistics.analyze_seed_contrasts(combined, families=families,
        expected_seeds=manifest['base']['seeds'], expected_rows=masks,
        model_failures=[{'seed':s,'contrasts':[c for slots in families.values() for c in slots],
                         'reason':'training_or_evaluation_failed'} for s in failed],
        equivalence_contrast='transfer_stocks::VolPFN__R_class', **manifest['statistics'])
    training.save_json(out/'seed_mean_results.json', result)
    lines = ['# Five-seed causal VolPFN replication', '',
             'The headline averages individual-seed score gains, not predictive densities. The retained checkpoint remains a separate historical comparator.', '',
             'Every declared seed is retained. Crossed seed/date intervals are approximate with only five seeds. Older panels were previously inspected.', '',
             '| Family | Contrast | Mean gain | Crossed 95% interval | Holm p | All five positive | Survival |',
             '|---|---|---:|---|---:|---|---|']
    # Raw structured results are authoritative; tolerate unavailable family rows.
    for family, values in result['families'].items():
        for key in values['registered_slots']:
            c = result['contrasts'][key]
            row = c['holm_by_family'][family]
            crossed = c.get('crossed', {})
            mean = crossed.get('mean_gain')
            interval = crossed.get('interval95')
            lines.append('| '+family+' | '+key.split('::',1)[-1]+' | '+('unavailable' if mean is None else f'{mean:+.6f}')+
                         ' | '+('unavailable' if interval is None else f'[{interval[0]:+.6f}, {interval[1]:+.6f}]')+
                         ' | '+str(row['crossed_p_holm'])+' | '+str(c.get('all_seeds_positive',False))+
                         ' | '+str(row['crossed_claim_survives'])+' |')
    lines += ['', 'Stock equivalence and complete per-seed values are recorded in seed_mean_results.json.',
              'Failures: '+str(failed), '', 'No subsequent level-only comparator, extended dates or stabilized per-market fits are included.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def run_all(out=DEFAULT_OUT):
    out = Path(out).resolve()
    manifest = verified_manifest(out)
    digest = training.sha(out/'manifest.json')
    status_path = out/'status.json'
    if status_path.exists():
        previous_status = json.loads(status_path.read_text())
        previous_pid = previous_status.get('pid')
        if previous_pid and previous_pid != os.getpid() and previous_status.get('status') in ('pilot','training','evaluating'):
            try:
                os.kill(previous_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Another replication process is active: '+str(previous_pid))
    if status_path.exists() and json.loads(status_path.read_text()).get('status') in ('complete','complete_with_failures'):
        raise RuntimeError('Completed battery is immutable')
    training.save_json(status_path, dict(status='pilot', pid=os.getpid(), started_utc=training.now(), manifest_sha256=digest))
    check = pilot(out)
    if check['status'] != 'passed':
        training.save_json(status_path, dict(status='pilot_failed_base_not_started', manifest_sha256=digest,
                                            completed_utc=training.now(), reason='Keep failed pilot; separate repair protocol required'))
        return
    records = []
    for seed in manifest['base']['seeds']:
        verified_manifest(out)
        training.save_json(status_path, dict(status='training', current_seed=seed, pid=os.getpid(), manifest_sha256=digest,
                                             terminal_seeds=[r['seed'] for r in records], updated_utc=training.now()))
        path = out/'base'/str(seed)
        record = existing_seed(path, manifest_sha=digest, specification=manifest['base'], seed=seed) or training.train_seed(seed, manifest['base'], path, digest)
        records.append(record)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    # No real-market scoring until every base seed is terminal.
    torch.set_num_threads(manifest['inference']['threads'])
    evaluated, failed = [], [r['seed'] for r in records if r['status'] != 'complete']
    for record in records:
        seed = record['seed']
        if record['status'] != 'complete':
            continue
        path = out/'evaluation'/str(seed)
        training.save_json(status_path, dict(status='evaluating', current_seed=seed, pid=os.getpid(), manifest_sha256=digest,
                                             updated_utc=training.now()))
        try:
            verified_manifest(out)
            selected_checkpoint = out/'base'/str(seed)/'best.pt'
            if training.sha(selected_checkpoint) != record['best_checkpoint_sha256']:
                raise RuntimeError('Selected base checkpoint changed')
            if (path/'completed_artifacts.json').exists():
                completion = json.loads((path/'completed_artifacts.json').read_text())
                if completion['checkpoint_sha256'] != record['best_checkpoint_sha256']:
                    raise RuntimeError('Evaluation belongs to a different checkpoint')
                for filename, file_sha in completion['hashes'].items():
                    if training.sha(filename) != file_sha:
                        raise RuntimeError('Completed seed evaluation changed')
            else:
                evaluation.evaluate_checkpoint(out/'base'/str(seed)/'best.pt', path,
                    input_inventory=manifest['evaluation_inventory'], device=manifest['inference']['device'],
                    reps=manifest['statistics']['reps'], block=manifest['statistics']['block'], seed=manifest['statistics']['seed'])
            evaluated.append(seed)
        except Exception as error:
            failed.append(seed)
            training.save_json(out/'evaluation_failures'/('seed'+str(seed)+'.json'),
                               dict(seed=seed,status='failed',error=str(error),traceback=traceback.format_exc()))
    aggregate(out, manifest, evaluated, failed)
    verified_manifest(out)
    artifacts = {str(p.resolve()):training.sha(p) for p in out.rglob('*') if p.is_file() and
                 p.name not in ('status.json','run.log','completed_artifacts.json') and not p.name.endswith('.tmp')}
    training.save_json(out/'completed_artifacts.json', dict(manifest_sha256=digest, hashes=artifacts,
                                                          completed_utc=training.now()))
    training.save_json(status_path, dict(status='complete_with_failures' if failed else 'complete',
        completed_utc=training.now(), manifest_sha256=digest, evaluated_seeds=evaluated, failed_seeds=failed,
        report=str(out/'REPORT.md')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze','pilot','run','status'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--device', default='mps', choices=('mps','cpu'))
    args = parser.parse_args()
    if args.command=='freeze':
        freeze(args.out,args.device)
    elif args.command=='pilot':
        print(json.dumps(pilot(args.out), indent=2))
    elif args.command=='run':
        run_all(args.out)
    else:
        print((args.out/'status.json').read_text() if (args.out/'status.json').exists() else 'Not started')


if __name__ == '__main__':
    main()
