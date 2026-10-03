"""Versioned numerical repair: eight fresh trainings plus two certified Matérn finals.

Never changes or relabels an archived checkpoint. No survivor-only evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from experiments.programme_closure import gpu_training as G, evaluate as E
from experiments.matched_prior import run as M
from experiments.seed_replication import causal, training as T

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/outputs/programme_closure_20260919_v1'


def freeze(out, whitelist):
    out = Path(out).resolve()
    manifest = G.freeze(out)
    entries = E.read(whitelist)
    if isinstance(entries, dict):
        entries = entries['reused_checkpoints']
    E.require(len(entries)==2 and {(x['arm'], x['seed']) for x in entries} == {('matern', 8101), ('matern', 8102)},
              'Only the two predeclared Matérn finals may be reused')
    old = BASE / 'gpu_design_v1/manifest.json'
    for item in entries:
        E.require(item['source_manifest_sha256'] == E.sha(old), 'Wrong ancestor design')
        source = ROOT / item['source_directory']
        for name, key in [('status.json', 'status_sha256'), ('final.pt', 'checkpoint_sha256'),
                          ('initial.pt', 'initial_file_sha256')]:
            E.require(E.sha(source/name) == item[key], 'Reuse input drift: '+name)
    for path in [Path(__file__), Path(E.__file__),
                 ROOT/'experiments/programme_closure/scan_broad_inputs.py',
                 ROOT/'experiments/programme_closure/run_pipeline.py',
                 ROOT/'experiments/programme_closure/GJR_REPAIR_AMENDMENT.md']:
        relative = path.relative_to(ROOT)
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest['source_hashes'][str(relative)] = E.sha(path)
    manifest.update(revision='exact_gjr_semivariance_v2', previous_design_sha256=E.sha(old),
        reused_checkpoints=entries,
        reuse='Only Matérn8101/8102, whose training dependency path is unchanged. Original files and original manifest identities retained. All five broad models restart from initialization.',
        continuation='User explicitly requested fix and continue after the recorded failure; no market outcomes from this battery were available.',
        fresh_training_count=8)
    E.save(out/'manifest.json', manifest)
    (out/'manifest.sha256').write_text(E.sha(out/'manifest.json')+'\n')
    G.verify(out)
    return manifest


def authorize(frozen, timing_path, out, pod_id):
    """Record the already requested continuation after its short runtime check."""
    manifest = G.verify(frozen)
    timing = E.read(timing_path)
    E.require(timing['status'] == 'complete' and timing['manifest_sha256'] == E.sha(Path(frozen)/'manifest.json'),
              'No completed repair-version timing gate')
    deadline = datetime.fromisoformat('2026-09-19T15:56:38+00:00')
    projected = sum((5 if r['arm']=='broad' else 3)*r['full_run_projected_seconds'] for r in timing['records'])
    E.require(datetime.now(timezone.utc) < deadline, 'Original stop deadline expired')
    remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    E.require(projected+1800 < remaining, 'Measured continuation plus30-minute analysis/export allowance does not fit original deadline')
    E.require((projected/3600+.5)*.74 <= 3.75, 'Projected continuation exceeds remaining compute allowance')
    receipt = dict(approved_full_battery=True,
        user_authorization_reference='User explicitly requested: i want to contnue fix snd continue. Original $5 total compute allowance and stop deadline remain unchanged.',
        approved_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=E.sha(Path(frozen)/'manifest.json'),
        timing_path=str(Path(timing_path).resolve()), timing_sha256=E.sha(timing_path), runtime=timing['runtime'],
        total_programme_compute_allowance_usd=5, prior_compute_allowance_reserved_usd=1.25,
        cost_ceiling_usd=3.75, stop_deadline_utc='2026-09-19T15:56:38Z', pod_id=pod_id,
        stop_enforcement='Worker timeout stops computation; five-minute monitor remains PAUSED at user request. Pod stopping requires the active task or user. No claim of cloud-side billing cap.',
        eight_fresh_training_projected_seconds=projected, reused_checkpoints=manifest['reused_checkpoints'])
    with Path(out).open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def run(frozen, out, authorization):
    manifest = G.verify(frozen)
    receipt = E.read(authorization)
    digest = E.sha(Path(frozen)/'manifest.json')
    E.require(receipt['approved_full_battery'] is True and receipt['manifest_sha256'] == digest,
              'Wrong continuation authorization')
    E.require(E.sha(receipt['timing_path']) == receipt['timing_sha256'], 'Changed timing evidence')
    timing = E.read(receipt['timing_path'])
    E.require(timing['status']=='complete' and timing['manifest_sha256']==digest,
              'Timing evidence not bound to corrected design')
    E.require(timing['runtime']==receipt['runtime'], 'Timing runtime mismatch')
    E.require(0 < receipt['cost_ceiling_usd'] <= 3.75 and receipt['user_authorization_reference'],
              'Continuation exceeds original programme allowance')
    deadline = datetime.fromisoformat(receipt['stop_deadline_utc'].replace('Z','+00:00'))
    E.require(deadline<=datetime.fromisoformat('2026-09-19T15:56:38+00:00'), 'Original stop deadline cannot be extended')
    E.require(datetime.now(timezone.utc)<deadline, 'Deadline expired')
    scan=Path(frozen).parent/'broad_input_preflight_v2/status.json'
    screen=E.read(scan)
    E.require(screen['status']=='complete' and screen['completed']==100000 and screen['total']==100000
              and screen['manifest_sha256']==digest, 'Complete corrected broad-input preflight required')
    E.require(E.sha(scan.parent/'batch_hashes.jsonl')==screen['batch_hashes_sha256'], 'Input-screen evidence changed')
    actual = G.runtime()
    for key in ('python','numpy','scipy','torch','cuda','gpu'):
        E.require(actual[key]==receipt['runtime'][key], 'Runtime changed: '+key)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    s=manifest['training'];G.configure(s)
    panels=M.shared_panels(s)
    np.savez_compressed(out/'shared_validation.npz',**{arm+'__'+k:v for arm,p in panels.items() for k,v in p.items() if isinstance(v,np.ndarray)})
    E.save(out/'runtime.json',actual)
    reused={(e['arm'],e['seed']):e for e in manifest['reused_checkpoints']}
    records=[];reuse_validations=[]
    try:
        for item in manifest['order']:
            E.require(datetime.now(timezone.utc)<deadline, 'Deadline reached')
            G.verify(frozen)
            arm,seed=item['arm'],item['seed']
            directory=out/'runs'/('%s_%d'%(arm,seed))
            E.save(out/'status.json',dict(status='training',current=item,completed=len(records)))
            if (arm,seed) in reused:
                evidence=reused[(arm,seed)];source=ROOT/evidence['source_directory']
                directory.mkdir(parents=True,exist_ok=False)
                for name,key in [('status.json','status_sha256'),('final.pt','checkpoint_sha256'),('initial.pt','initial_file_sha256')]:
                    E.require(E.sha(source/name)==evidence[key],'Archived reuse input changed')
                    shutil.copy2(source/name,directory/name)
                record=E.read(directory/'status.json')
                E.save(directory/'REUSE_PROVENANCE.json',dict(evidence,new_comparison_design_sha256=digest,
                    note='Original status, checkpoint and initialization files copied byte-for-byte; no relabeling.'))
                model=causal.load_model(directory/'final.pt',s['device'])
                validation=M.evaluate_panels(model,panels,s)
                reuse_validations.append(dict(arm=arm,seed=seed,validation=validation,
                    note='Final checkpoint rescored on corrected common synthetic panels; no selection or training.'))
                E.save(out/'REUSED_FINAL_VALIDATION.json',reuse_validations)
                del model
                print('REUSED',arm,seed,'20000',flush=True)
            else:
                record=G.train_one(arm,seed,s,directory,digest,panels)
            records.append(record)
        for seed in G.SEEDS:
            pair=[r for r in records if r['seed']==seed]
            E.require(len(pair)==2 and len({r['initial_state_sha256'] for r in pair})==1,
                      'Mismatched initial states')
        E.save(out/'status.json',dict(status='complete',records=records))
        E.verify_checkpoints(frozen,out)
    except BaseException as exc:
        E.save(out/'status.json',dict(status='incomplete_or_failed',records=records,error=repr(exc)))
        (out/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','authorize','run'])
    p.add_argument('--frozen',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--whitelist',type=Path);p.add_argument('--timing',type=Path)
    p.add_argument('--authorization',type=Path);p.add_argument('--pod-id')
    a=p.parse_args()
    if a.action=='freeze':freeze(a.out,a.whitelist)
    elif a.action=='authorize':print(json.dumps(authorize(a.frozen,a.timing,a.out,a.pod_id),indent=2))
    else:run(a.frozen,a.out,a.authorization)
