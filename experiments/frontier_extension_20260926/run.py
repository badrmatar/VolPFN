"""Frozen, resumable historical extension. No training or infrastructure creation.

Run different model families with their existing compatible Python environments.
Every market/seed cell contains the complete calibration/evaluation union, so
resuming never resets an econometric refit clock. TabPFN is imported by the
separate checked remote workflow; this module never starts paid compute.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np

from experiments.combined_leaderboard.common import ROOT, key, load_bundle, now, read, require, save, sha
from .prepare import OUT, CLASSICAL, SEEDS, model_specs

HERE = Path(__file__).resolve().parent
ECONOMETRIC = {'ms_har': 'mshar_sqrt', 'log_rhar': 'logrhar', 'loghar_meanlog': 'loghar_meanlog'}
LOCAL_FOUNDATION = ('chronos2', 'moirai2')


def atomic(path, value):
    """Publish a complete JSON file without replacing an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        save(temporary, value)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_paths():
    """Pin project imports, including lazy adapter imports, without installed packages."""
    paths = set(ROOT.glob('*.py'))
    for directory, children, files in os.walk(ROOT / 'research'):
        children[:] = [c for c in children if c not in
                       ('outputs', 'models', 'environments', '__pycache__', '.git', '.venv', 'venv')]
        paths.update(Path(directory) / f for f in files if f.endswith('.py'))
    paths.update(HERE.glob('*.md'))
    return sorted(paths)


def freeze(out):
    """Only explicit recorded approval of this exact bundle/protocol can freeze."""
    out = Path(out)
    require(not (out / 'FREEZE.json').exists(), 'Already frozen; never overwrite a freeze')
    bundle = load_bundle(out)
    source_path = bundle.get('source_bundle_path')
    require(source_path in bundle['inputs'] and bundle['inputs'][source_path] == bundle.get('source_bundle_sha256'),
            'Canonical source bundle must be hash-bound')
    source = read(ROOT / source_path)
    expected_cal = {key(r): r for r in source['rows'] if r['phase'] == 'calibration'}
    actual_cal = {key(r): r for r in bundle['rows'] if r['phase'] == 'calibration'}
    require(set(actual_cal) == set(expected_cal), 'Calibration keys differ from canonical source')
    require(all(all(actual_cal[k].get(f) == original.get(f) for f in
                    ('market', 'date', 'phase', 'index', 'r2', 'rk')) for k, original in expected_cal.items()),
            'Calibration metadata differ from canonical source')
    require(bundle['model_specs'] == model_specs(), 'Bundle roster differs from executable roster')
    from .statistics import coverage_audit
    coverage_audit([], bundle['model_specs'], bundle)
    require(sum(s['primary'] for m, s in bundle['model_specs'].items() if m != 'volpfn') * 2 == 32,
            'Expected the declared 32-comparison primary family')
    protocol, approval, tests = out / 'PROTOCOL.md', out / 'APPROVAL.json', out / 'TESTS.json'
    require(all(p.is_file() for p in (protocol, approval, tests)), 'Protocol, approval and passing tests required')
    decision, evidence = read(approval), read(tests)
    require(decision.get('approved') is True, 'Explicit protocol approval required')
    require(decision.get('bundle_sha256') == sha(out / 'bundle.json'), 'Approval predates bundle')
    require(decision.get('protocol_sha256') == sha(protocol), 'Approval predates protocol')
    require(evidence.get('passed') is True and evidence.get('source_hashes'), 'Passing source-bound tests required')
    core = {str((HERE / n).relative_to(ROOT)) for n in
            ('run.py', 'prepare.py', 'econometric.py', 'ttm.py', 'statistics.py', 'supervise_local.py')}
    require(core <= set(evidence['source_hashes']), 'Tests must bind every extension implementation')
    for name, digest in evidence['source_hashes'].items():
        require(sha(ROOT / name) == digest, 'Tests predate executable change: ' + name)
    paths = source_paths() + [out / 'bundle.json', protocol, approval, tests]
    # Absolute paths are allowed for output-directory receipts; repository sources remain portable.
    sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in paths}
    value = dict(kind='frontier_extension_20260926', created_utc=now(), inputs=bundle['inputs'],
        sources=sources, settings=bundle['settings'], model_specs=bundle['model_specs'],
        runtime_policy='CPU only locally; VolPFN/Chronos/Moirai batch1; immutable per-model runtime receipt; remote TabPFN separately validated',
        approved_runtimes=decision.get('runtimes', {}),
        disclosure=bundle['exposure'], calibration='Unchanged canonical 72 dates; one scalar per model/seed after complete forecasts')
    atomic(out / 'FREEZE.json', value)
    with (out / 'FREEZE.sha256').open('x') as stream:
        stream.write(sha(out / 'FREEZE.json') + '\n')
    return value


def verify(out):
    out = Path(out)
    require((out / 'FREEZE.json').is_file() and (out / 'FREEZE.sha256').is_file(),
            'Execution requires an approved frozen protocol; no forecasts made')
    require(sha(out / 'FREEZE.json') == (out / 'FREEZE.sha256').read_text().strip(), 'Freeze hash changed')
    sealed = read(out / 'FREEZE.json')
    for group in ('sources', 'inputs'):
        for name, digest in sealed[group].items():
            require(sha(ROOT / name) == digest, 'Frozen content changed: ' + name)
    return read(out / 'bundle.json'), sealed


def relative_cell(model, seed, market):
    for value in (model, market):
        require(isinstance(value, str) and value and '/' not in value and value not in ('.', '..'), 'Unsafe cell name')
    return Path(model) / ('none' if seed is None else str(int(seed))) / market


def cell(out, relative, sealed_id, data=None):
    """Atomic forecast+checksum directory; existing data is never overwritten."""
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe relative cell path')
    out = Path(out)
    path = out / 'cells' / relative
    if path.exists():
        require(data is None, 'Attempted overwrite of committed cell: ' + str(relative))
        require((path / 'forecast.json').is_file() and (path / 'forecast.sha256').is_file(), 'Unsealed cell; explicit recovery required')
        require(sha(path / 'forecast.json') == (path / 'forecast.sha256').read_text().strip(), 'Cell checksum changed')
        found = read(path / 'forecast.json')
        require(found['freeze_sha256'] == sealed_id, 'Cell belongs to another freeze')
        return found
    if data is None:
        return None
    require('freeze_sha256' not in data and 'created_utc' not in data, 'Reserved cell metadata')
    value = dict(freeze_sha256=sealed_id, created_utc=now(), **data)
    staging = out / 'work' / ('cell_' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    save(staging / 'forecast.json', value)
    (staging / 'forecast.sha256').write_text(sha(staging / 'forecast.json') + '\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Another process committed this cell')
    os.rename(staging, path)
    return value


def market_data(bundle, name):
    """Full canonical history: deliberately does not use the old 2016 cutoff."""
    require(name in bundle['markets'], 'Undeclared market')
    with np.load(ROOT / bundle['market_paths'][name], allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in ('date', 'r', 'rk')}
    data['date'] = data['date'].astype('U10')
    require(all(x.ndim == 1 and len(x) == len(data['date']) for x in data.values()), 'Invalid market arrays')
    require(np.isfinite(data['r']).all() and np.all(data['date'][1:] > data['date'][:-1]), 'Invalid canonical return history')
    return dict(name=name, **data)


def origins_for(bundle, name, data):
    origins = sorted((dict(r) for r in bundle['rows'] if r['market'] == name), key=lambda r: r['index'])
    require(origins and len({r['index'] for r in origins}) == len(origins), 'Missing/duplicate origins')
    for row in origins:
        i = row['index']
        require(isinstance(i, int) and 1024 <= i < len(data['r']), 'Insufficient canonical causal history')
        require(row['date'] == data['date'][i], 'Origin date/index mismatch')
        require(row['r2'] == float(data['r'][i]) ** 2, 'Canonical return target mismatch')
        expected_rk = float(data['rk'][i]) if np.isfinite(data['rk'][i]) and data['rk'][i] > 0 else None
        require(row['rk'] == expected_rk, 'Canonical RK target mismatch')
    return origins


def record(base, model, seed, values, substituted=False, floor=False, **metadata):
    mean, variance, second = map(float, values)
    require(np.isfinite([mean, variance, second]).all() and variance > 0 and second > 0, 'Invalid moments; no silent fallback')
    require(np.isclose(second, variance + mean ** 2, rtol=1e-7, atol=1e-10), 'Incoherent second moment')
    row = dict(base)
    row.update(model=model, seed=seed, mean=mean, variance=variance, second_moment=second,
               substituted=bool(substituted), floor_applied=bool(floor), forecast_failed=False)
    require(not set(metadata) & set(row), 'Adapter metadata would overwrite canonical record')
    row.update(metadata)
    return row


def validate_rows(rows, origins, model, seed):
    require(len(rows) == len(origins), 'Cell must retain every requested origin')
    expected = {key(r): r for r in origins}
    require(len({key(r) for r in rows}) == len(rows), 'Duplicate forecast row')
    for row in rows:
        require(key(row) in expected and row['model'] == model and row['seed'] == seed, 'Wrong cell identity')
        base = expected[key(row)]
        for field in ('index', 'r2', 'rk', 'primary_eligible'):
            if field in base:
                require(row.get(field) == base[field], 'Changed canonical metadata: ' + field)
        require(isinstance(row.get('substituted'), bool) and isinstance(row.get('floor_applied'), bool), 'Explicit forecast flags required')
        require(row.get('forecast_failed', False) is False, 'Failed forecast may not enter a committed cell')
        record(base, model, seed, [row[k] for k in ('mean', 'variance', 'second_moment')])
    require({key(r) for r in rows} == set(expected), 'Missing origin in cell')


def runtime_receipt(model, args):
    packages = {}
    for name in ('numpy', 'scipy', 'pandas', 'arch', 'statsmodels', 'torch', 'transformers',
                 'chronos-forecasting', 'uni2ts', 'granite-tsfm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), executable=str(Path(sys.executable).absolute()),
                packages=packages, device=args.device, threads=args.threads,
                batch_size=args.batch_size if model.startswith('ttm_') else 1)


def runtime_matches(actual, expected):
    """The frozen relevant package subset may omit unrelated installed packages."""
    if not isinstance(actual, dict):
        return False
    return all(runtime_matches(actual.get(k, {}), v) if isinstance(v, dict)
               else actual.get(k) == v for k, v in expected.items())


def forecast_market(bundle, model, seed, name, args, predictors):
    """Pure adapter dispatch; runner's freeze gate precedes every production call."""
    data = market_data(bundle, name)
    origins = origins_for(bundle, name, data)
    indices = np.array([r['index'] for r in origins], dtype=int)
    audit = dict(model=model, seed=seed, market=name, origin_count=len(origins),
                 refit_anchor=int(indices[0]), input_history='full canonical history; inputs strictly before target index')
    if model in CLASSICAL:
        from experiments.combined_leaderboard.classical import forecast
        pred = forecast(data, model, indices)
        rows = [record(r, model, seed, [pred[k][i] for k in ('mean', 'variance', 'second_moment')],
                       pred['substituted'][i], pred['floor_applied'][i]) for i, r in enumerate(origins)]
        audit['adapter'] = pred['audit']
    elif model in ECONOMETRIC:
        from .econometric import EconometricConfig, forecast_series
        pred = forecast_series(data['rk'], indices, EconometricConfig(ECONOMETRIC[model]))
        require([r['origin'] for r in pred] == indices.tolist(), 'Econometric origin order changed')
        rows = [record(r, model, seed, (0., p['variance'], p['variance']),
                       p['metadata']['substituted'], False, adapter=p['metadata']) for r, p in zip(origins, pred)]
    elif model in ('ttm_rk', 'ttm_sqrt_rk'):
        from .ttm import TTMRealizedAdapter
        if model not in predictors:
            predictors[model] = TTMRealizedAdapter(transform='variance' if model == 'ttm_rk' else 'sqrt',
                device='cpu', snapshot=ROOT / bundle['ttm_path'])
        pred = predictors[model].forecast_series(data['rk'], origins, batch_size=args.batch_size)
        rows = [record(r, model, seed, [p[k] for k in ('mean', 'variance', 'second_moment')],
                       p['substituted'], p['floor_applied'],
                       adapter={k: v for k, v in p.items() if k not in r and k not in
                                ('model', 'seed', 'mean', 'variance', 'second_moment', 'substituted', 'floor_applied')})
                for r, p in zip(origins, pred)]
    elif model == 'volpfn':
        from experiments.combined_leaderboard.neural import Predictor
        if (model, seed) not in predictors:
            predictors[model, seed] = Predictor(seed, 'cpu', args.threads)
        predictor = predictors[model, seed]
        checkpoint = ROOT / bundle['checkpoints'][str(seed)]
        require(predictor.entry['checkpoint_sha256'] == sha(checkpoint), 'Predictor checkpoint differs from bundle')
        rows = []
        for r in origins:
            i = r['index']
            values = predictor.predict(data['r'][i - 1024:i][None])[0]
            rows.append(record(r, model, seed, values))
        audit.update(context_length=1024, checkpoint_sha256=sha(checkpoint), batch_size=1)
    elif model in LOCAL_FOUNDATION:
        from experiments.frontier_benchmark.adapters import make
        from experiments.frontier_benchmark.metrics import positive_point
        if model not in predictors:
            predictors[model] = make(model, ROOT / bundle['foundation_models'][model]['path'])
        rows = []
        for r in origins:
            i = r['index']; past = data['r'][i - 1024:i]
            native = predictors[model].predict(past, 'squared')
            h, floored, lower = positive_point(native['point'], past)
            rows.append(record(r, model, seed, (0., h, h), floor=floored,
                raw_native_point=float(native['point']), output_floor=lower,
                point_semantics=native['point_semantics']))
        audit.update(context_length=1024, track='squared', batch_size=1,
                     revision=bundle['foundation_models'][model]['provenance']['revision'])
    else:
        raise ValueError('Model requires the separately validated remote import: ' + model)
    validate_rows(rows, origins, model, seed)
    return rows, audit


def run(out, args):
    out = Path(out)
    bundle, sealed = verify(out)
    identity = sha(out / 'FREEZE.json')
    require(args.device == 'cpu', 'Local extension adapters are frozen for CPU only')
    require(args.batch_size > 0 and args.threads > 0, 'Positive batch size and thread count required')
    models = args.models or [m for m in bundle['model_specs'] if m != 'tabpfn_ts']
    markets = args.markets or bundle['markets']
    require(len(set(models)) == len(models) and set(models) <= set(bundle['model_specs']), 'Undeclared/duplicate models')
    require(len(set(markets)) == len(markets) and set(markets) <= set(bundle['markets']), 'Undeclared/duplicate markets')
    require('tabpfn_ts' not in models, 'Use the separately checked TabPFN remote importer')
    if args.seeds is not None:
        require('volpfn' in models and set(args.seeds) <= set(SEEDS) and len(set(args.seeds)) == len(args.seeds), 'Undeclared/duplicate training seeds')
    if any(m == 'volpfn' or m in LOCAL_FOUNDATION for m in models):
        require(args.batch_size == 1, 'VolPFN/Chronos/Moirai require their frozen batch1 path; run TTM separately for batching')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('HF_HOME', str(ROOT / 'experiments/outputs/return_only_benchmark_setup_v1/foundation_cache/hf_home'))
    if any(m == 'volpfn' or m in LOCAL_FOUNDATION or m.startswith('ttm_') for m in models):
        import torch
        torch.set_num_threads(args.threads)
    completed = 0
    for model in models:
        runtime = runtime_receipt(model, args)
        required_runtime = bundle.get('runtime_requirements', {}).get(model)
        require(required_runtime is not None, 'No prepared runtime requirements for model')
        require(runtime_matches(runtime, required_runtime), 'Runtime differs from prepared model requirements: ' + model)
        approved = sealed.get('approved_runtimes', {}).get(model)
        if approved is not None:
            require(runtime_matches(runtime, approved), 'Runtime differs from approved model receipt')
        runtime_path = out / 'runtimes' / (model + '.json')
        if runtime_path.exists():
            require(read(runtime_path) == runtime, 'Cannot mix runtimes for one model')
        else:
            try:
                atomic(runtime_path, runtime)
            except FileExistsError:
                require(read(runtime_path) == runtime, 'Concurrent incompatible runtime')
        seeds = (args.seeds or SEEDS) if model == 'volpfn' else [None]
        for seed in seeds:
            predictors = {}
            for name in markets:
                relative = relative_cell(model, seed, name)
                lock_path = out / 'locks' / relative.with_suffix('.lock')
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                with lock_path.open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    prior = cell(out, relative, identity)
                    if prior is not None:
                        require(prior['audit']['runtime'] == runtime, 'Saved cell runtime differs')
                        validate_rows(prior['rows'], origins_for(bundle, name, market_data(bundle, name)), model, seed)
                        print('RESUME', model, seed, name, len(prior['rows']), flush=True)
                        continue
                    tick = time.perf_counter()
                    attempt_id = uuid.uuid4().hex
                    receipt = dict(model=model, seed=seed, market=name, freeze_sha256=identity,
                                   started_utc=now(), pid=os.getpid(), runtime=runtime)
                    atomic(out / 'attempts' / (attempt_id + '_start.json'), receipt)
                    try:
                        rows, audit = forecast_market(bundle, model, seed, name, args, predictors)
                        audit['runtime'] = runtime
                        cell(out, relative, identity, dict(rows=rows, audit=audit,
                             elapsed_seconds=time.perf_counter() - tick, attempt_id=attempt_id))
                    except BaseException as exc:
                        atomic(out / 'attempts' / (attempt_id + '_failed.json'),
                               dict(**receipt, ended_utc=now(), exception=f'{type(exc).__name__}: {exc}'))
                        raise
                    completed += 1
                    print('COMPLETE', model, seed, name, len(rows), round(time.perf_counter() - tick, 3), flush=True)
    return dict(new_cells=completed, status='requested_scope_complete')


def load_records(out, identity):
    rows, audits = [], []
    for path in sorted((Path(out) / 'cells').rglob('forecast.json')):
        relative = path.parent.relative_to(Path(out) / 'cells')
        value = cell(out, relative, identity)
        rows.extend(value['rows'])
        audits.append(dict(cell=str(relative), **value.get('audit', {})))
    return rows, audits


def status(out):
    out = Path(out)
    bundle = read(out / 'bundle.json')
    frozen = (out / 'FREEZE.sha256').is_file()
    identity = sha(out / 'FREEZE.json') if frozen else None
    records, audits = load_records(out, identity)
    from .statistics import coverage_audit
    supervisors = []
    for path in sorted((out / 'supervisor').glob('*/plan.json')):
        plan = read(path); terminal = path.parent / 'terminal.json'
        events = path.parent / 'events.jsonl'
        supervisors.append(dict(id=path.parent.name, workers=plan['workers'],
            deadline_timestamp=plan['deadline_timestamp'],
            terminal=read(terminal) if terminal.is_file() else None,
            recent_events=[json.loads(line) for line in events.read_text().splitlines()[-6:]] if events.is_file() else []))
    return dict(created_utc=now(), frozen=frozen, cells=len(audits), supervisors=supervisors,
                coverage=coverage_audit(records, bundle['model_specs'], bundle),
                failed_attempts=len(list((out / 'attempts').glob('*_failed.json'))),
                analysis_complete=(out / 'analysis' / 'verification.json').is_file())


def write_report(result, out, bundle):
    """One all-model table, with information group and diagnostic scope visible."""
    out = Path(out)
    r2 = {r['model']: r for r in result['leaderboards']['primary']['r2']}
    rk = {r['model']: r for r in result['leaderboards']['primary']['rk']}
    raw = {proxy: {r['model']: r for r in result['leaderboards']['raw'][proxy]} for proxy in ('r2', 'rk')}
    effects = {(r['model'], r['proxy']): r for r in result.get('calibration_effects', [])}
    main = {(r['model'], r['proxy']): r for r in result['contrasts'] if r['block_role'] == 'main'}
    table = []
    for model, spec in result['models'].items():
        table.append(dict(model=model, family=spec['family'], primary=spec['primary'], seeds=len(spec['seeds']),
            r2_loss=r2[model]['loss'], rk_loss=rk[model]['loss'],
            r2_raw_loss=raw['r2'][model]['loss'], rk_raw_loss=raw['rk'][model]['loss'],
            r2_rank=r2[model]['rank'], rk_rank=rk[model]['rank'],
            r2_raw_gain=effects.get((model, 'r2'), {}).get('raw_gain'),
            rk_raw_gain=effects.get((model, 'rk'), {}).get('raw_gain'),
            r2_gain=effects.get((model, 'r2'), {}).get('calibrated_gain'),
            rk_gain=effects.get((model, 'rk'), {}).get('calibrated_gain'),
            r2_direction_changed=effects.get((model, 'r2'), {}).get('gain_direction_changed'),
            rk_direction_changed=effects.get((model, 'rk'), {}).get('gain_direction_changed'),
            r2_verdict=main.get((model, 'r2'), {}).get('verdict'),
            rk_verdict=main.get((model, 'rk'), {}).get('verdict'),
            r2_holm=main.get((model, 'r2'), {}).get('p_holm'),
            rk_holm=main.get((model, 'rk'), {}).get('p_holm'),
            volpfn_supported_both_proxies=result['comparison_support'].get(model),
            output_adapter=spec['output_adapter']))
    atomic(out / 'leaderboard.json', table)
    with (out / 'leaderboard.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    lines = [f'# Historical extension: {bundle["start"]} to {bundle["end"]}', '',
        bundle['exposure'], '', result['scope'], '',
        f'Primary common sample: {result["samples"]["primary"]["rows"]:,} rows across {result["samples"]["primary"]["markets"]} markets. '
        f'The joint family contains {result["settings"]["primary_family_slots"]} tests; main calendar block {result["settings"]["main_block"]}.', '',
        result['calibration_rule'] + '. Raw and all-return-date results remain secondary in results.json.', '',
        'V/E/F ranks and richer-information R ranks are separate. All five VolPFN seeds are calibrated and scored individually; their losses are averaged. '
        'Diagnostic rows are excluded from the primary testing family.', '',
        '| Model | Group | Scope | Seeds | R² calibrated / raw | RK calibrated / raw | R² rank | RK rank | VolPFN supported on both |',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
    for row in table:
        lines.append('| ' + ' | '.join([row['model'], row['family'], 'primary' if row['primary'] else 'diagnostic',
            str(row['seeds']), f'{row["r2_loss"]:.6f} / {row["r2_raw_loss"]:.6f}',
            f'{row["rk_loss"]:.6f} / {row["rk_raw_loss"]:.6f}',
            str(row['r2_rank']) if row['r2_rank'] is not None else '—',
            str(row['rk_rank']) if row['rk_rank'] is not None else '—',
            '—' if row['volpfn_supported_both_proxies'] is None else str(row['volpfn_supported_both_proxies'])]) + ' |')
    lines.extend(['', '| Comparator | Proxy | Calibrated gain | Raw gain | Direction changed | Main-block verdict |',
                  '|---|---|---:|---:|---|---|'])
    for row in table:
        if row['model'] == 'volpfn':
            continue
        for proxy in ('r2', 'rk'):
            effect = effects.get((row['model'], proxy), {})
            if effect:
                lines.append(f'| {row["model"]} | {proxy} | {effect["calibrated_gain"]:.6f} | '
                             f'{effect["raw_gain"]:.6f} | {effect["gain_direction_changed"]} | '
                             f'{row[proxy+"_verdict"] or "diagnostic only"} |')
    lines.extend(['', 'Positive gains mean comparator loss minus VolPFN loss. Support requires positive gains for every retained VolPFN seed, '
        'a positive family interval lower bound and Holm p<0.05 on both proxies. Inconclusive does not establish equivalence.', '',
        'TTM direct-RK is a declared benchmark adaptation. TTM sqrt-RK and mean-log-HAR are diagnostics. '
        'New econometric controls are declared adaptations, with substitutions and fit events retained in each cell.', ''])
    (out / 'REPORT.md').write_text('\n'.join(lines))


def analyse(out):
    out = Path(out)
    bundle, _ = verify(out)
    require(not (out / 'analysis').exists(), 'Analysis already exists; never overwrite')
    identity = sha(out / 'FREEZE.json')
    records, audits = load_records(out, identity)
    from .statistics import analyse as score
    archive = {}
    result = score(records, bundle['model_specs'], bundle=bundle, archive=archive)
    require(result['settings']['primary_family_slots'] == 32, 'Unexpected primary family')
    result['period'] = dict(start=bundle['start'], end=bundle['end'])
    result['cell_audits'] = audits
    destination = out / 'work' / ('analysis_' + uuid.uuid4().hex)
    destination.mkdir(parents=True)
    atomic(destination / 'results.json', result)
    np.savez_compressed(destination / 'bootstrap_archive.npz', **archive)
    with (destination / 'records.jsonl').open('x') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_report(result, destination, bundle)
    atomic(destination / 'verification.json', dict(freeze_sha256=identity, records=len(records),
        coverage=result['coverage'], outputs={p.name: sha(p) for p in destination.iterdir() if p.is_file()}))
    require(not (out / 'analysis').exists(), 'Concurrent analysis already published')
    os.rename(destination, out / 'analysis')
    return dict(status='complete', report=str(out / 'analysis' / 'REPORT.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'status', 'analyse'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--markets', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    value = run(args.out, args) if args.command == 'run' else globals()[args.command](args.out)
    print(json.dumps(value, default=str), flush=True)


if __name__ == '__main__':
    main()
