"""Freeze the exposed historical confirmation programme before new scoring.

The freeze records code and every data file; it is not a retrospective claim
that the empirical periods were untouched. No paid resources are provisioned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODULE = Path(__file__).resolve().parent
INVENTORY = Path('experiments/outputs/matched_prior_design_inventory_v1/inventory.json')
DATA = Path('experiments/data/oxfordman_barde/calendar_full_v1')
SEEDS = [8101, 8102, 8103, 8104, 8105]
ARMS = ['full', 'noaux', 'historical', 'direct']
DAILY_MODELS = ['arch_lstm', 'garch_lstm', 'ginn']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def population(root=ROOT):
    roster = json.loads((Path(root) / INVENTORY).read_text())['retained_index_population']
    if len(roster) != 28 or len(set(roster)) != 28:
        raise ValueError('Expected the locked 28 distinct Oxford–Man markets')
    return list(roster)


def read_market(market, root=ROOT):
    if market not in population(root):
        raise ValueError('Market is outside the locked population')
    path = Path(root) / DATA / (market + '.npz')
    with np.load(path, allow_pickle=False) as raw:
        dates = raw['date'].astype('U10')
        returns = raw['returns'].astype(np.float64) * 100.
        # Source realised kernel is in decimal-return-squared units.
        kernel = raw['rk_parzen'].astype(np.float64) * 10000.
    if dates.ndim != 1 or dates.shape != returns.shape or kernel.shape != dates.shape:
        raise ValueError('Unexpected market-array shapes')
    if not len(dates) or np.any(dates[1:] <= dates[:-1]) or not np.isfinite(returns).all():
        raise ValueError('Market dates must be strictly increasing; all returns finite')
    # Parse every date rather than relying solely on string comparisons.
    parsed = pd.to_datetime(dates, format='%Y-%m-%d', errors='raise')
    if not np.array_equal(parsed.strftime('%Y-%m-%d').to_numpy(), dates):
        raise ValueError('Noncanonical date labels')
    return dates, returns, kernel


def masks(root=ROOT):
    records = {'train': [], 'validation': [], 'evaluation': []}
    for market in population(root):
        dates, returns, _ = read_market(market, root)
        for index, (date, value) in enumerate(zip(dates, returns)):
            split = ('train' if date <= '2010-12-31' else
                     'validation' if date <= '2014-12-31' else 'evaluation')
            if date > '2022-02-25':
                raise ValueError('Historical source exceeds the registered end date')
            records[split].append((market, str(date), index, float(value)))
    result = {key: pd.DataFrame(rows, columns=['market', 'date', 'row_index', 'return'])
              for key, rows in records.items()}
    expected = {'train': 70199, 'validation': 28086, 'evaluation': 50073}
    if {key: len(value) for key, value in result.items()} != expected:
        raise ValueError('The audited data population/counts changed')
    for key, frame in result.items():
        if frame.duplicated(['market', 'date']).any():
            raise ValueError('Duplicate frozen score labels: ' + key)
    return result


def specification():
    return {
        'name': 'volpfn_experiment_programme_20260930',
        'status': 'registered execution design; not completed results',
        'empirical_exposure': 'Every existing empirical panel was previously inspected. '
            'These are retrospective historical confirmations, not fresh holdouts.',
        'target': 'Next observed session open-to-close percent log return; no missing-session imputation',
        'seeds': SEEDS, 'tuning_seeds': [717311], 'volpfn_arms': ARMS,
        'daily_models': DAILY_MODELS,
        'external_unavailable_slots': {'rech': 'Official source is a later MATLAB/Bayesian model; '
            'original-2022 replication and finite-variance bridge not verified'},
        'additional_information_track': {'dcc_gngarch': 'Separate feasibility/input-information track; '
            'not in the returns-only family. Requires separately locked point-in-time graph data.'},
        'data': {'directory': str(DATA), 'inventory': str(INVENTORY),
                 'train_end': '2010-12-31', 'validation_start': '2011-01-01',
                 'validation_end': '2014-12-31', 'evaluation_start': '2015-01-01',
                 'evaluation_end': '2022-02-25', 'source_return_conversion': 'decimal * 100 once'},
        'component_training': {'updates': 20000, 'positions_per_update': 8192,
             'length_probability': {'512': .4, '1024': .6}, 'fixed_final_checkpoint': True,
             'device': 'CUDA float32; deterministic algorithms; no TF32/autocast',
             'historical_label': 'No latent target is invented; compare with synthetic noaux',
             'direct_output': 'Positive next-return second moment trained with r² QLIKE; '
                 'changes both output and objective, not a pure auxiliary-head ablation'},
        'daily_training': {'max_epochs': 50, 'patience': 8, 'batch_size': 128,
             'selection': 'Separate per-market development raw r² QLIKE; parameters frozen before evaluation; '
                 'final scores aggregate equally across markets and seeds',
             'scale': 'Per-market TRAIN-only centered-return RMS and TRAIN-only mean',
             'grid': {'arch_lstm': {'learning_rate': [.0003, .001], 'weight': [0., .5, 1.],
                                  'context': [1], 'nu': [3.]},
                      'garch_lstm': {'learning_rate': [.0003, .001], 'context': [126], 'nu': [5.]},
                      'ginn': {'learning_rate': [.0003, .001], 'weight': [0., .01, .1], 'context': [90]}},
             'failure': 'Strict coverage; retain fit/training failures and do not silently omit rows or seeds'},
        'inference': {'volpfn': 'Per origin, last 1024 completed returns; causal features reset in window',
                      'external': 'Native declared contexts; GINN teacher filters continuous completed prefix',
                      'h': 'Raw second moment = central variance + mean²; direct head outputs h',
                      'common_density': 'Imposed zero-mean Student-t8 with scale sqrt(6*h/8)'},
        'primary_metric': 'r²/h + log(h); finite h>0, all frozen return rows including zeros',
        'secondary_metrics': ['native return NLL where defined', 'common-t8 NLL',
                              'RK QLIKE on independently frozen positive finite kernel mask'],
        'statistical_families': {
            'components': ['full_vs_noaux', 'noaux_vs_historical', 'noaux_vs_direct'],
            'external': ['full_vs_arch_lstm', 'full_vs_garch_lstm', 'full_vs_ginn', 'full_vs_rech']},
        'uncertainty': {'resamples': 20000, 'calendar_block': 60, 'sensitivity_blocks': [20, 120],
                       'seed': 2026093001, 'tests': 'Two-sided centered circular-calendar bootstrap; '
                           'basic95% intervals; Holm within each family; unavailable slots p=1',
                       'scope': 'Calendar uncertainty conditional on five trained models, with separate '
                           'exploratory crossed-seed/calendar uncertainty; no prospective-selection correction'},
        'scientific_limits': ['Published external methods are adapted to the common open-to-close target.',
            'No auxiliary benefit or superiority is presumed before results.',
            'Historical-return training differs in task diversity, exposure and repeated observations.',
            'No globally best, universally superior or first-ever claim follows from this programme.'],
        'spending': {'authorization': 'User explicitly authorized paid compute with open budget',
                     'first_deposit_discussion_usd': 20,
                     'policy': 'Finite timing pilots; record rate and duration; do not leave idle GPUs running; '
                         'preserve/download verified outputs before resource cleanup.'},
    }


def source_files(root=ROOT):
    root = Path(root)
    explicit = ['src/volpfn/model.py', 'src/volpfn/prior.py',
                'experiments/matched_prior/run.py', 'experiments/matched_prior/simulators.py',
                'experiments/matched_prior/diagnostics.py', 'experiments/matched_prior/__init__.py',
                'experiments/seed_replication/training.py', 'experiments/seed_replication/causal.py',
                'experiments/seed_replication/__init__.py',
                'experiments/programme_closure/gpu_training.py',
                'experiments/matched_prior_confirmatory/statistics.py']
    paths = {root / value for value in explicit}
    paths.update((root / 'src/volpfn').rglob('*.py'))
    paths.update(path for path in (root / 'experiments/experiment_programme_20260930').rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and
                 'external_sources' not in path.parts and path.suffix in {'.py', '.md', '.json', '.txt'})
    return sorted(paths)


def freeze(out, root=ROOT, ablation_manifest=None):
    out, root = Path(out).resolve(), Path(root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tables = masks(root)
    source_hashes = {}
    for path in source_files(root):
        relative = path.relative_to(root)
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = sha(path)
    data_paths = [root / INVENTORY, root / DATA / 'protocol.json']
    data_paths += [root / DATA / (market + '.npz') for market in population(root)]
    data_hashes = {str(path.relative_to(root)): sha(path) for path in data_paths}
    mask_files = {}
    for key, frame in tables.items():
        path = out / 'masks' / (key + '.csv')
        path.parent.mkdir(exist_ok=True)
        frame.to_csv(path, index=False, float_format='%.17g')
        mask_files[key] = {'path': str(path.relative_to(out)), 'sha256': sha(path), 'rows': len(frame)}
    kernel_rows = []
    for market in population(root):
        dates, _, kernel = read_market(market, root)
        keep = (dates >= '2015-01-01') & (dates <= '2022-02-25') & np.isfinite(kernel) & (kernel > 0)
        kernel_rows.extend((market, str(dates[index]), int(index), float(kernel[index]))
                           for index in np.flatnonzero(keep))
    rk = out / 'masks' / 'evaluation_kernel.csv'
    pd.DataFrame(kernel_rows, columns=['market', 'date', 'row_index', 'kernel']).to_csv(rk, index=False, float_format='%.17g')
    mask_files['evaluation_kernel'] = {'path': str(rk.relative_to(out)), 'sha256': sha(rk), 'rows': len(kernel_rows),
        'units': 'Percent-return-squared; source rk_parzen multiplied by10000 exactly once',
        'interpretation': 'Secondary realised-variability proxy; equivalence to the open-to-close conditional second moment unverified'}
    manifest = dict(specification(), created_utc=datetime.now(timezone.utc).isoformat(),
                    population=population(root), source_hashes=source_hashes,
                    data_hashes=data_hashes, mask_files=mask_files)
    if ablation_manifest is not None:
        from experiments.experiment_programme_20260930 import ablations as A
        component = A.verify(ablation_manifest)
        if component['specification']['smoke']:
            raise ValueError('Final programme cannot bind a smoke component design')
        component_path = A._manifest_path(ablation_manifest)
        manifest['component_design'] = {'path': str(component_path.relative_to(root)),
                                        'sha256': sha(component_path)}
    write_json(out / 'manifest.json', manifest)
    (out / 'manifest.sha256').write_text(sha(out / 'manifest.json') + '\n')
    return manifest


def verify(manifest_path, root=ROOT):
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / 'manifest.json'
    expected = manifest_path.with_suffix('.sha256').read_text().strip()
    if sha(manifest_path) != expected:
        raise ValueError('Programme manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    for key in ['source_hashes', 'data_hashes']:
        for relative, digest in manifest[key].items():
            if sha(Path(root) / relative) != digest:
                raise ValueError(key + ' drift: ' + relative)
            if key == 'source_hashes' and sha(manifest_path.parent / 'source' / relative) != digest:
                raise ValueError('Frozen source drift: ' + relative)
    for entry in manifest['mask_files'].values():
        if sha(manifest_path.parent / entry['path']) != entry['sha256']:
            raise ValueError('Frozen scoring mask drift')
    if 'component_design' in manifest:
        from experiments.experiment_programme_20260930 import ablations as A
        component_path = Path(root) / manifest['component_design']['path']
        if sha(component_path) != manifest['component_design']['sha256']:
            raise ValueError('Component design binding changed')
        A.verify(component_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'verify', 'describe'])
    parser.add_argument('--out')
    parser.add_argument('--manifest')
    parser.add_argument('--ablation-manifest')
    args = parser.parse_args()
    if args.action == 'describe':
        print(json.dumps(specification(), indent=2))
    elif args.action == 'freeze':
        if not args.out:
            parser.error('--out is required')
        manifest = freeze(args.out, ablation_manifest=args.ablation_manifest)
        print(json.dumps({'status': 'frozen', 'source_files': len(manifest['source_hashes']),
                          'rows': {key: entry['rows'] for key, entry in manifest['mask_files'].items()}}))
    else:
        if not args.manifest:
            parser.error('--manifest is required')
        verify(args.manifest)
        print('Verified programme sources, data and masks')


if __name__ == '__main__':
    main()
