"""Portable finite queue/bundle; execution never creates or charges a GPU.

Infrastructure is controlled separately using the user's authorized Runpod
account. The queue retains failed attempts; it never changes a scientific
budget or fills a missing method with an unnamed surrogate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.experiment_programme_20260930 import programme as P


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def bundle(manifest_path, out, ablation_manifest=None):
    manifest = P.verify(manifest_path)
    path = Path(manifest_path).resolve()
    if path.is_dir():
        path = path / 'manifest.json'
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Bundle output already exists')
    # Preserve the repository-relative manifest path so runtime ROOT rebases.
    relative = path.parent.relative_to(P.ROOT)
    files = {P.ROOT / value for key in ['source_hashes', 'data_hashes'] for value in manifest[key]}
    files.update(file for file in path.parent.rglob('*') if file.is_file())
    component_receipt = None
    if ablation_manifest is not None:
        from research.experiment_programme_20260930 import ablations as A
        A.verify(ablation_manifest)
        component_path = A._manifest_path(ablation_manifest)
        component_receipt = {'path': str(component_path.relative_to(P.ROOT)), 'sha256': P.sha(component_path)}
        if manifest.get('component_design') != component_receipt:
            raise ValueError('Bundle component differs from the programme binding')
        files.update(file for file in component_path.parent.rglob('*') if file.is_file())
    elif 'component_design' in manifest:
        raise ValueError('Bound component design must be included using --ablation-manifest')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, 'w:gz') as archive:
        for file in sorted(files):
            archive.add(file, arcname=str(file.relative_to(P.ROOT)), recursive=False)
    receipt = {'status': 'complete', 'created_utc': timestamp(), 'bundle_sha256': P.sha(out),
               'bytes': out.stat().st_size, 'files': len(files),
               'manifest_relative': str(relative / 'manifest.json'),
               'manifest_sha256': P.sha(path), 'contents': 'Registered sources, data and freeze only; no credentials/checkpoints/old outputs'}
    receipt['component_design'] = component_receipt
    P.write_json(out.with_suffix(out.suffix + '.json'), receipt)
    return receipt


def plan(manifest_path, ablation_manifest, out, result_root):
    manifest = P.verify(manifest_path)
    from research.experiment_programme_20260930 import ablations as A
    component = A.verify(ablation_manifest)
    if component['specification']['smoke']:
        raise ValueError('Cannot plan a final programme using a smoke design')
    primary = Path(manifest_path).resolve()
    if primary.is_dir():
        primary = primary / 'manifest.json'
    ablation = A._manifest_path(ablation_manifest)
    if manifest.get('component_design') != {'path': str(ablation.relative_to(P.ROOT)), 'sha256': P.sha(ablation)}:
        raise ValueError('Queue component design differs from programme binding')
    root = Path(result_root).resolve()
    # Paths within the checkout are recorded relative for portable relocation.
    primary_rel = str(primary.relative_to(P.ROOT))
    ablation_rel = str(ablation.relative_to(P.ROOT))
    root_rel = str(root.relative_to(P.ROOT))
    entries = []

    def entry(name, module, arguments, completion):
        entries.append({'id': name, 'module': module, 'arguments': list(map(str, arguments)),
                        'completion': str(completion), 'status': 'pending'})

    for arm in manifest['volpfn_arms']:
        for seed in manifest['seeds']:
            entry('train_' + arm + '_' + str(seed), 'research.experiment_programme_20260930.ablations',
                  ['train', '--manifest', ablation_rel, '--out', root_rel + '/component_training',
                   '--arm', arm, '--seed', seed], root_rel + '/component_training/runs/' + arm + '_' + str(seed) + '/status.json')
        entry('evaluate_' + arm, 'research.experiment_programme_20260930.evaluation',
              ['--manifest', primary_rel, '--ablation-manifest', ablation_rel,
               '--training-root', root_rel + '/component_training', '--arm', arm,
               '--out', root_rel + '/component_evaluation/' + arm, '--device', 'cuda'],
              root_rel + '/component_evaluation/' + arm + '/result.json')
    for model in manifest['daily_models']:
        for market in manifest['population']:
            prefix = root_rel + '/daily/' + model + '/' + market
            common = ['--manifest', primary_rel, '--market', market, '--device', 'cuda']
            entry('tune_' + model + '_' + market, 'research.experiment_programme_20260930.daily_runner',
                  ['tune', *common, '--model', model, '--out', prefix + '/tuning'], prefix + '/tuning/result.json')
            entry('train_' + model + '_' + market, 'research.experiment_programme_20260930.daily_runner',
                  ['train', *common, '--tuning', prefix + '/tuning', '--out', prefix + '/training'], prefix + '/training/result.json')
            entry('evaluate_' + model + '_' + market, 'research.experiment_programme_20260930.daily_runner',
                  ['evaluate', *common, '--training', prefix + '/training', '--out', prefix + '/evaluation'], prefix + '/evaluation/result.json')
    result = {'created_utc': timestamp(), 'scope': 'Finite final queue; no infrastructure provisioning',
              'manifest': primary_rel, 'manifest_sha256': P.sha(primary),
              'ablation_manifest': ablation_rel, 'ablation_manifest_sha256': P.sha(ablation),
              'results': root_rel, 'entries': entries,
              'failure_policy': 'Stop on first failed dependency; independent work can be dispatched explicitly without replacing failed seeds',
              'cleanup': 'Controller must download/verify outputs and stop GPU after finite queue; container process exit alone is not a billing guard'}
    out = Path(out)
    if out.exists():
        raise FileExistsError('Queue already exists')
    P.write_json(out, result)
    return result


def execute(queue_path, out):
    queue_path = Path(queue_path).resolve()
    queue = json.loads(queue_path.read_text())
    P.verify(P.ROOT / queue['manifest'])
    from research.experiment_programme_20260930 import ablations as A
    A.verify(P.ROOT / queue['ablation_manifest'])
    if P.sha(P.ROOT / queue['manifest']) != queue['manifest_sha256'] or P.sha(P.ROOT / queue['ablation_manifest']) != queue['ablation_manifest_sha256']:
        raise ValueError('Queue design hash changed')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    P.write_json(out / 'status.json', {'status': 'running', 'created_utc': timestamp(), 'queue_sha256': P.sha(queue_path)})
    completed = []
    for entry in queue['entries']:
        name = entry['id']
        command = [sys.executable, '-m', entry['module'], *entry['arguments']]
        record = {'id': name, 'started_utc': timestamp(), 'command': command, 'status': 'running'}
        P.write_json(out / 'status.json', record)
        with (out / (name + '.log')).open('w') as log:
            process = subprocess.run(command, cwd=P.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        record.update(ended_utc=timestamp(), returncode=process.returncode)
        if process.returncode:
            record.update(status='failed', log=str(out / (name + '.log')), completed=completed)
            P.write_json(out / (name + '.json'), record)
            P.write_json(out / 'status.json', record)
            raise RuntimeError('Finite queue halted at ' + name)
        try:
            receipt = json.loads((P.ROOT / entry['completion']).read_text())
            if receipt['status'] != 'complete':
                raise RuntimeError('Child exited without complete result: ' + name)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            record.update(status='failed', error=repr(error), completed=completed)
            P.write_json(out / (name + '.json'), record)
            P.write_json(out / 'status.json', record)
            raise
        record.update(status='complete', receipt_sha256=P.sha(P.ROOT / entry['completion']))
        P.write_json(out / (name + '.json'), record)
        completed.append(name)
    P.write_json(out / 'status.json', {'status': 'complete', 'ended_utc': timestamp(),
                                       'completed': completed, 'queue_sha256': P.sha(queue_path)})


def collect(manifest_path, result_root, out):
    manifest = P.verify(manifest_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(result_root)
    forecasts, unavailable = {}, {}
    for arm in manifest['volpfn_arms']:
        source = root / 'component_evaluation' / arm / 'forecasts.csv'
        receipt = source.parent / 'result.json'
        if source.exists() and receipt.exists() and json.loads(receipt.read_text())['status'] == 'complete':
            record = json.loads(receipt.read_text())
            primary = Path(manifest_path)
            if primary.is_dir():
                primary = primary / 'manifest.json'
            if record['forecast_sha256'] != P.sha(source) or record['manifest_sha256'] != P.sha(primary):
                raise ValueError('Component forecast or design hash mismatch: ' + arm)
            if record['ablation_manifest_sha256'] != manifest['component_design']['sha256']:
                raise ValueError('Component forecasts use a different ablation design')
            forecasts[arm] = str(source.resolve())
        else:
            unavailable[arm] = 'Fixed-final all-seed forecast table incomplete'
    for model in manifest['daily_models']:
        parts = []
        try:
            for market in manifest['population']:
                directory = root / 'daily' / model / market / 'evaluation'
                receipt = json.loads((directory / 'result.json').read_text())
                if receipt['status'] != 'complete' or receipt['smoke'] or receipt['seeds'] != manifest['seeds']:
                    raise ValueError('Market evaluation is not complete/final')
                primary = Path(manifest_path)
                if primary.is_dir():
                    primary = primary / 'manifest.json'
                if receipt['lock']['manifest']['sha256'] != P.sha(primary):
                    raise ValueError('Daily forecast uses a different programme design')
                for seed in manifest['seeds']:
                    file = directory / ('forecasts_' + str(seed) + '.csv')
                    if P.sha(file) != receipt['forecast_sha256'][file.name]:
                        raise ValueError('Forecast hash mismatch')
                    parts.append(pd.read_csv(file, dtype={'date': str, 'market': str}, float_precision='round_trip'))
            table = pd.concat(parts, ignore_index=True)
            path = out / (model + '.csv')
            table.to_csv(path, index=False, float_format='%.17g')
            forecasts[model] = str(path.resolve())
        except (FileNotFoundError, ValueError, KeyError) as error:
            unavailable[model] = str(error)
    P.write_json(out / 'forecasts.json', forecasts)
    P.write_json(out / 'unavailable.json', dict(unavailable, **manifest['external_unavailable_slots']))
    return forecasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['bundle', 'plan', 'execute', 'collect'])
    parser.add_argument('--manifest')
    parser.add_argument('--ablation-manifest')
    parser.add_argument('--queue')
    parser.add_argument('--result-root')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.action == 'bundle':
        result = bundle(args.manifest, args.out, args.ablation_manifest)
    elif args.action == 'plan':
        result = plan(args.manifest, args.ablation_manifest, args.out, args.result_root)
    elif args.action == 'execute':
        execute(args.queue, args.out)
        result = {'status': 'complete'}
    else:
        result = collect(args.manifest, args.result_root, args.out)
    print(json.dumps(result if args.action != 'plan' else {'entries': len(result['entries']), 'out': args.out}))


if __name__ == '__main__':
    main()
