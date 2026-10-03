"""Lock the five-pair design, then queue five synthetic Matérn runs.

This command never imports or executes market scoring. Original replication
market artifacts are checked for existence at freeze, never read as outcomes.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
import numpy as np
import scipy
import torch
from research.matched_prior import run as M, simulators as S
from research.seed_replication import training as T, causal
from . import compatibility as C, data_gate as D, statistics as ST

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
REPLICATION = ROOT/'research/outputs/seed_replication_session_recovery_20260918'
AUDIT = ROOT/'research/outputs/matched_prior_compatibility_v1/audit.json'
INVENTORY = ROOT/'research/outputs/matched_prior_design_inventory_v1/inventory.json'
COVERAGE = ROOT/'research/outputs/matched_prior_coverage_v1/report.json'
DEFAULT_OUT = ROOT/'research/outputs/matched_prior_five_pairs_v1'
SEEDS = [8101, 8102, 8103, 8104, 8105]


def no_replication_market_outputs(replication):
    """Absence check only; do not open forecast/loss/report content."""
    replication = Path(replication)
    paths = [replication/'REPORT.md', replication/'seed_mean_results.json',
             replication/'evaluation', replication/'evaluation_failures']
    contacts = [str(p) for p in paths if p.exists()]
    status_path = replication/'status.json'
    phase = json.loads(status_path.read_text()).get('status') if status_path.exists() else None
    if phase in ('evaluating','complete','complete_with_failures'):
        contacts.append('replication phase: ' + phase)
    if contacts:
        raise RuntimeError('Pre-exposure freeze window not established; artifacts already exist: ' + str(contacts))
    return dict(checked_utc=T.now(), status='no_designated_base_market_artifacts_exist',
                inspected_paths=[str(p) for p in paths], replication_phase=phase, outcomes_opened=False,
                scope='designated original replication only; not a global claim about outside access')


def analysis_specification():
    return dict(primary=dict(contrast=ST.PRIMARY, seeds=SEEDS, score='common_t8_second_moment_log_density',
                             mean='paired daily gains, equal markets and seeds; not ensemble density'),
                secondary=dict(contrast=ST.SECONDARY, seeds=[8101,8102,8103], launch=False, unavailable_p=1.),
                superiority=dict(slots=[ST.PRIMARY,ST.SECONDARY], method='Holm', alpha=.05,
                                  one_sided=True, point_materiality_nats=.005, positive_interval95_required=True),
                equivalence=dict(contrast=ST.PRIMARY, margin_nats=.003, alpha=.05,
                                 method='TOST and basic90 interval', multiplicity='separate single endpoint; no joint FWER'),
                bootstrap=dict(reps=ST.DEFAULT_REPS, block=ST.DEFAULT_BLOCK, seed=ST.DEFAULT_BOOTSTRAP_SEED,
                               method='synchronized circular calendar, centered/basic', resample_training_seeds=False),
                data=D.specification(),
                intermediate_language='Statistically supported but below the declared materiality bar',
                synthetic_breadth_costs=['NLL[broad,garch]-NLL[garch,garch]', 'NLL[broad,matern]-NLL[matern,matern]'])


def source_paths():
    return sorted(set(M.source_paths() + list(PACKAGE.glob('*.py')) + [PACKAGE/'PROTOCOL.md']))


def verify(out):
    out = Path(out).resolve()
    if (out/'freeze_failure.json').exists():
        raise RuntimeError('Design freeze was invalidated; cannot launch it')
    path = out/'manifest.json'
    if T.sha(path) != (out/'manifest.sha256').read_text().strip():
        raise RuntimeError('Design manifest changed')
    manifest = json.loads(path.read_text())
    for relative, expected in manifest['sources'].items():
        if T.sha(ROOT/relative) != expected or T.sha(out/'source'/relative) != expected:
            raise RuntimeError('Frozen code/protocol drift: ' + relative)
    receipt = out/'post_freeze_exposure_guard.json'
    if T.sha(receipt) != (out/'post_freeze_exposure_guard.sha256').read_text().strip():
        raise RuntimeError('Post-freeze exposure receipt changed')
    current_versions = dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=str(torch.__version__))
    if any(manifest['environment'][key] != value for key,value in current_versions.items()):
        raise RuntimeError('Frozen Python/numerical runtime changed')
    for relative, expected in manifest['fixed_artifacts'].items():
        if T.sha(out/relative) != expected:
            raise RuntimeError('Frozen artifact changed: ' + relative)
    replication = Path(manifest['replication']['path'])
    if T.sha(replication/'manifest.json') != manifest['replication']['manifest_sha256']:
        raise RuntimeError('Original replication manifest changed')
    original = json.loads((replication/'manifest.json').read_text())
    C.frozen_source_audit(original)
    C.core_configuration(original['base'], manifest['training'])
    return manifest


def freeze(out=DEFAULT_OUT, replication=REPLICATION, audit=AUDIT, inventory=INVENTORY, coverage=COVERAGE):
    out, replication, audit, inventory, coverage = map(lambda p: Path(p).resolve(), (out,replication,audit,inventory,coverage))
    before = no_replication_market_outputs(replication)
    original_path = replication/'manifest.json'
    original_sha = T.sha(original_path)
    if original_sha != original_path.with_suffix('.json.sha256').read_text().strip():
        raise RuntimeError('Original manifest checksum mismatch')
    original = json.loads(original_path.read_text())
    a, inv = json.loads(audit.read_text()), json.loads(inventory.read_text())
    if a['status'] != 'passed' or a['tensor_replay']['batches'] < 300 or a['original_manifest_sha256'] != original_sha:
        raise RuntimeError('Required compatibility gate is absent or incompatible')
    for relative, expected in a['audit_source_hashes'].items():
        if T.sha(ROOT/relative) != expected:
            raise RuntimeError('Compatibility-audited implementation changed')
    candidates = inv['retained_index_population']
    if len(candidates) != 28 or len(set(candidates)) != 28:
        raise RuntimeError('Expected exactly28 fixed candidate index identities')
    s = M.specification('full', device='mps')
    s['seeds'] = SEEDS.copy()
    s['kind'] = 'five_paired_matern_counterparts'
    C.core_configuration(original['base'], s)
    C.frozen_source_audit(original)
    if out.exists():
        raise FileExistsError('Refusing to overwrite design lock: ' + str(out))
    out.mkdir(parents=True)
    fixed = {}
    for name, source in [('compatibility_audit.json',audit), ('exposure_inventory.json',inventory),
                         ('synthetic_coverage.json',coverage)]:
        shutil.copy2(source,out/name)
        fixed[name] = T.sha(out/name)
    sources = {}
    for source in source_paths():
        relative = str(source.relative_to(ROOT))
        target = out/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        sources[relative] = T.sha(source)
    # Synthetic only. These inputs cannot select checkpoints or alter allocation.
    torch.set_num_threads(1)
    panels = M.shared_panels(s)
    np.savez_compressed(out/'shared_validation.npz', **{
        arm+'__'+key:value for arm,panel in panels.items() for key,value in panel.items()})
    fixed['shared_validation.npz'] = T.sha(out/'shared_validation.npz')
    # Preserve the exact historical mask as exposed-data secondary metadata.
    mask = Path(original['expected_masks']['index_density::C4'])
    if T.sha(mask) != original['hashes'].get(str(mask.resolve())):
        raise RuntimeError('Historical mask no longer matches original frozen manifest')
    shutil.copy2(mask,out/'historical_secondary_mask.csv')
    fixed['historical_secondary_mask.csv'] = T.sha(out/'historical_secondary_mask.csv')
    after = no_replication_market_outputs(replication)
    manifest = dict(created_utc=T.now(), status='prospective_design_locked_before_designated_replication_market_output',
                    sources=sources, fixed_artifacts=fixed, training=s, analysis=analysis_specification(),
                    candidate_population=candidates,
                    replication=dict(path=str(replication),manifest_sha256=original_sha, original_base_spec=original['base']),
                    exposure_guard_before=before,exposure_guard_after=after,
                    reused_broad_weights='all five complete final.pt at20000; no best.pt or seed selection',
                    new_runs=[dict(arm='matern',seed=seed) for seed in SEEDS],
                    chronology='original broad battery and original evaluation finish before new Matérn battery; not interleaved',
                    prospective_data_lock='not yet present; must hash mapping, raw data, QC, population, context, masks before scoring',
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     torch=str(torch.__version__),platform=platform.platform()))
    T.save_json(out/'manifest.json',manifest)
    (out/'manifest.sha256').write_text(T.sha(out/'manifest.json')+'\n')
    try:
        final_guard = no_replication_market_outputs(replication)
    except BaseException:
        T.save_json(out/'freeze_failure.json',dict(status='invalid_exposure_guard',failed_utc=T.now()))
        raise
    T.save_json(out/'post_freeze_exposure_guard.json',final_guard)
    (out/'post_freeze_exposure_guard.sha256').write_text(T.sha(out/'post_freeze_exposure_guard.json')+'\n')
    verify(out)
    T.save_json(out/'status.json',dict(status='frozen_not_started',created_utc=T.now()))
    return manifest


def load_panels(out):
    with np.load(Path(out)/'shared_validation.npz',allow_pickle=False) as archive:
        return {arm:{key.split('__',1)[1]:archive[key].copy() for key in archive.files if key.startswith(arm+'__')}
                for arm in S.ARMS}


def wait_for_replication(out, manifest, interval=30.):
    replication = Path(manifest['replication']['path'])
    while True:
        verify(out)
        status = json.loads((replication/'status.json').read_text())
        if status['status'] in ('complete','complete_with_failures'):
            return
        if status['status'] not in ('pilot','training','evaluating'):
            raise RuntimeError('Original replication needs intervention: ' + status['status'])
        pid = status.get('pid')
        if pid:
            try:
                os.kill(pid,0)
            except ProcessLookupError as exc:
                raise RuntimeError('Original replication process is absent; refusing a stale wait') from exc
        T.save_json(Path(out)/'status.json', dict(status='waiting_for_replication',pid=os.getpid(),
                   checked_utc=T.now(),replication_phase=status['status'],replication_seed=status.get('current_seed'),
                   note='No Matérn training or market scoring yet; original results not opened'))
        time.sleep(interval)


def run(out=DEFAULT_OUT):
    out = Path(out).resolve()
    manifest = verify(out)
    with (out/'started.json').open('x') as stream:
        json.dump(dict(started_utc=T.now(),pid=os.getpid()),stream)
    records = []
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS backend unavailable; no silent CPU substitution')
        if float(torch.ones(2,device='mps').sum().cpu()) != 2.:
            raise RuntimeError('MPS preflight failed')
        print('QUEUED: waiting for original replication; no market scores will be opened.',flush=True)
        wait_for_replication(out,manifest)
        s = manifest['training']
        torch.set_num_threads(s['threads'])
        replication = Path(manifest['replication']['path'])
        original = manifest['replication']['original_base_spec']
        broad = {}
        for seed in SEEDS:
            verify(out)
            broad[str(seed)] = C.checkpoint_audit(replication,seed,original,
                manifest['replication']['manifest_sha256'],s)
        T.save_json(out/'broad_final_provenance.json',broad)
        panels = load_panels(out)
        # Broad scores here are SYNTHETIC only and cannot change allocation.
        synthetic = {}
        for seed in SEEDS:
            model = causal.load_model(broad[str(seed)]['path'],'mps')
            synthetic[str(seed)] = M.evaluate_panels(model,panels,s)
            del model
            torch.mps.empty_cache()
        T.save_json(out/'broad_synthetic_matrix.json',synthetic)
        for seed in SEEDS:
            verify(out)  # includes prior/overlay/normalization/auxiliary-label source paths
            entry = broad[str(seed)]
            if T.sha(entry['path']) != entry['sha256']:
                raise RuntimeError('Broad final changed after eligibility audit')
            launch = dict(verified_utc=T.now(),manifest_sha256=T.sha(out/'manifest.json'),
                          source_hashes=manifest['sources'],broad_final_sha256=entry['sha256'],
                          reconstructed_initial_sha256=entry['reconstructed_new_initial_sha256'])
            T.save_json(out/'launch_checks'/f'{seed}.json',launch)
            T.save_json(out/'status.json',dict(status='training_matern',seed=seed,pid=os.getpid(),
                                              completed_runs=len(records),updated_utc=T.now()))
            r = M.train_arm('matern',seed,s,out/'matern'/str(seed),T.sha(out/'manifest.json'),panels)
            if r.get('initial_state_sha256') != entry['reconstructed_new_initial_sha256']:
                r.update(status='failed',failure_code='initial_state_does_not_match_reconstructed_seed_state')
                T.save_json(out/'matern'/str(seed)/'status.json',r)
            verify(out)
            records.append(r)
            T.save_json(out/'training_records.json',records)
            print('MATERN',seed,r['status'],flush=True)
            torch.mps.empty_cache()
        complete = len(records)==5 and all(r['status']=='complete' for r in records)
        pairs = [dict(seed=seed,broad_path=broad[str(seed)]['path'],broad_sha256=broad[str(seed)]['sha256'],
                      matern_path=str(out/'matern'/str(seed)/'final.pt'),
                      matern_sha256=r.get('checkpoint_sha256'),status=r['status'])
                 for seed,r in zip(SEEDS,records)]
        T.save_json(out/'paired_checkpoints.json',dict(complete_primary_battery=complete,pairs=pairs,
                                                     market_scoring_performed=False))
        T.save_json(out/'status.json',dict(status='trained_waiting_for_data_lock' if complete else 'training_incomplete',
                   completed_utc=T.now(),complete_primary_battery=complete,
                   completed_runs=sum(r['status']=='complete' for r in records),market_scoring_performed=False))
        return complete
    except BaseException as exc:
        T.save_json(out/'status.json',dict(status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',
                   failure_code=type(exc).__name__+': '+str(exc),updated_utc=T.now(),pid=os.getpid(),
                   completed_runs=len(records),market_scoring_performed=False))
        (out/'failure.txt').write_text(traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','verify','run','status'))
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    a=p.parse_args()
    if a.command=='freeze':
        m=freeze(a.out)
        print('FROZEN',m['created_utc'],T.sha(a.out/'manifest.json'))
    elif a.command=='verify':
        verify(a.out)
        print('VERIFIED',a.out)
    elif a.command=='status':
        print((a.out/'status.json').read_text())
    elif not run(a.out):
        raise SystemExit(1)


if __name__=='__main__':
    main()
