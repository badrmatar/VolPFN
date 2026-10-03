"""Read-only study adapters for the predeclared seed replication.

No market fit or market score is computed by ``inventory_inputs``. Empirical
evaluation requires that inventory from the experiment's frozen manifest.
Existing checkpoints, fits, forecasts, protocols and results are never edited.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluation_strengthening.statistics import qlike, paired_summary, holm
from experiments.population_transfer.statistics import common_t8_log_score, analyze_population

ROOT = Path(__file__).resolve().parents[2]
STRENGTH = Path('experiments/outputs/evaluation_strengthening_v1')
CALENDAR = Path('experiments/outputs/session_calendar_screen_v1')
BASE = Path('experiments/outputs/session_information_pilot_v1')
TRANSFER = Path('experiments/outputs/population_transfer_v1')
INDEX_DATA = Path('experiments/data/oxfordman_barde/calendar_full_v1')
CONTROLS = ('garch_t', 'univol', 'univol_point', 'ewma_t8', 'gjr_garch_t', 'egarch_t')
PARAMS = ('logw', 'loc', 'scale', 'nu')
SCENARIOS = ('nominal', 'conservative')
TRANSFER_CONTRASTS = (('R_class', 'R_eq'), ('VolPFN', 'R_eq'),
                      ('VolPFN', 'R_class'), ('VolPFN', 'R_market'))
BOOTSTRAP = dict(reps=2000, block=60, seed=927601)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _json(path):
    return json.loads(Path(path).read_text())


def _resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def comparison_registry():
    """All hypotheses, including unavailable/reserved slots, before training."""
    rows = []
    def add(family, name, candidate, control, metric, **extra):
        rows.append(dict(family=family, comparison=name, candidate=candidate,
                         control=control, metric=metric, **extra))
    for number, control in enumerate(('garch_t', 'univol', 'ewma_t8'), 1):
        add('index_density', 'C' + str(number), 'network', control, 'native_log_score')
    add('index_density', 'C4', 'network', 'garch_t', 'common_t8_log_score')
    for target in ('r2', 'rk'):
        for control in CONTROLS:
            add('index_qlike', target + '__' + control, 'network', control,
                'qlike', target=target)
    for scenario in SCENARIOS:
        for model in ('network', 'compact', 'gjr'):
            for target in ('r2', 'rk'):
                add('calendar_a', scenario + '__' + model + '__' + target,
                    model + '_latest', model + '_base', 'qlike',
                    target=target, scenario=scenario)
        for candidate, control in (('network_latest', 'network_lagged'),
                                   ('network_adaptive', 'network_latest')):
            for target in ('r2', 'rk'):
                add('calendar_b', scenario + '__' + candidate + '__' + target,
                    candidate, control, 'qlike', target=target, scenario=scenario,
                    reserved=candidate == 'network_adaptive')
    for population in ('stocks', 'fx'):
        for candidate, control in TRANSFER_CONTRASTS:
            add('transfer_' + population, candidate + '__' + control,
                candidate, control, 'common_t8_log_score', population=population)
    return rows


def inventory_inputs(root=ROOT):
    """Hash explicit existing inputs and masks without generating predictions.

    Includes the old hash chains and cached comparator forecasts; no downloaded
    cookie/cache directory is traversed. This inventory must enter the new
    protocol before any training or empirical evaluation.
    """
    root = Path(root).resolve()
    hashes = {}
    def include(path, expected=None):
        path = _resolve(path, root).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise RuntimeError('Archived input hash mismatch: ' + str(path))
        hashes[str(path)] = actual
        return path
    def chain(path):
        path = include(path)
        value = _json(path)
        for name, digest in value.get('hashes', {}).items():
            include(name, digest)
        return value

    strength = chain(STRENGTH / 'protocol.json')
    include(STRENGTH / 'protocol.sha256')
    if sha(root / STRENGTH / 'protocol.json') != (root / STRENGTH / 'protocol.sha256').read_text().strip():
        raise RuntimeError('Strengthening protocol signature mismatch')
    include(STRENGTH / 'scoring_mask.csv')
    for market in strength['markets']:
        include(INDEX_DATA / (market + '.npz'))
        audit_path = include(STRENGTH / 'cells' / (market + '_audit.json'))
        audit = _json(audit_path)
        if audit['protocol_sha256'] != sha(root / STRENGTH / 'protocol.json'):
            raise RuntimeError('Stale audited index comparator')
        include(STRENGTH / 'cells' / (market + '.csv.gz'), audit['forecast_sha256'])

    calendar = chain(CALENDAR / 'screen_manifest.json')
    chain(BASE / 'preparation.json')
    for market in calendar['design']['markets']:
        include(BASE / 'forecasts' / (market + '.npz'))
        include(BASE / 'forecasts' / (market + '.json'))

    transfer = chain(TRANSFER / 'manifest.json')
    chain(TRANSFER / 'completed_artifacts.json')
    include(TRANSFER / 'manifest.json.sha256')
    if sha(root / TRANSFER / 'manifest.json') != (root / TRANSFER / 'manifest.json.sha256').read_text().strip():
        raise RuntimeError('Transfer protocol signature mismatch')
    # All reused computational paths are explicitly pinned, beyond the old chains.
    for name in ('experiments/evaluation_strengthening/statistics.py',
                 'experiments/population_transfer/statistics.py',
                 'experiments/session_information_pilot/run.py',
                 'experiments/session_information_pilot/scaling.py',
                 'experiments/session_calendar_screen/run.py'):
        include(name)
    registry = comparison_registry()
    return dict(schema_version=1, root=str(root), hashes=hashes,
                studies=dict(index_markets=strength['markets'], index_rows=50073,
                             calendar_markets=calendar['design']['markets'],
                             calendar_evaluation_rows=calendar['design']['evaluation_rows'],
                             transfer_expected=transfer['expected']),
                registry=registry,
                family_sizes={name: sum(r['family'] == name for r in registry)
                              for name in sorted(set(r['family'] for r in registry))},
                index_density_note='C1-C4 use Holm4 in this new replication; historical C1-C4 were unadjusted. C4 uses matched second moments about zero from audited caches.',
                legacy_scope='Three previously inspected panels; no fresh-data or whole expanded-comparator replication claim.')


def verify_inventory(inventory):
    if inventory.get('registry') != comparison_registry():
        raise RuntimeError('Evaluation comparison registry changed')
    for path, digest in inventory['hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen evaluation input changed: ' + path)


def expected_rows(inventory):
    """Return exact pre-training masks keyed by globally unique contrast ID.

    Reads only dates, declared masks and realised-proxy eligibility, never any
    model loss. Includes masks for reserved/unavailable slots so an omitted
    hypothesis cannot disappear from the aggregation family.
    """
    root = Path(inventory['root'])
    index = pd.read_csv(root / STRENGTH / 'scoring_mask.csv', dtype=str)
    rk_parts = []
    for name in inventory['studies']['index_markets']:
        with np.load(root / INDEX_DATA / (name + '.npz'), allow_pickle=False) as z:
            dates = z['date'].astype('U10')
            valid = (z['split'] == 2) & np.isfinite(z['rk_parzen']) & (z['rk_parzen'] > 0)
        rk_parts.append(pd.DataFrame(dict(market=name, date=dates[valid])))
    index_rk = pd.concat(rk_parts, ignore_index=True)
    calendar = pd.read_csv(root / CALENDAR / 'scoring_mask.csv', dtype=str)
    calendar = calendar.loc[calendar.date >= '2015-01-01', ['market', 'date']].reset_index(drop=True)
    rk_parts = []
    for name in inventory['studies']['calendar_markets']:
        with np.load(root / BASE / 'forecasts' / (name + '.npz'), allow_pickle=False) as z:
            dates = z['date'].astype('U10'); rk = z['rk']
        sub = calendar.loc[calendar.market == name]
        idx = pd.Index(dates).get_indexer(sub.date)
        if np.any(idx < 0):
            raise RuntimeError('Missing calendar mask input')
        rk_parts.append(sub.loc[np.isfinite(rk[idx]) & (rk[idx] > 0)])
    calendar_rk = pd.concat(rk_parts, ignore_index=True)
    populations = {}
    for population in ('stocks', 'fx'):
        f = pd.read_csv(root / TRANSFER / (population + '_mask.csv'), dtype=str)
        populations[population] = f.loc[f.date >= '2015-01-01', ['market', 'date']].reset_index(drop=True)
    result = {}
    for spec in comparison_registry():
        if spec['family'].startswith('index_'):
            frame = index_rk if spec.get('target') == 'rk' else index
        elif spec['family'].startswith('calendar_'):
            frame = calendar_rk if spec.get('target') == 'rk' else calendar
        else:
            frame = populations[spec['population']]
        result[spec['family'] + '::' + spec['comparison']] = frame[['market', 'date']].copy()
    return result


def _assert_mask(frame, expected, label):
    columns = ['market', 'date']
    if frame.duplicated(columns).any() or expected.duplicated(columns).any():
        raise ValueError(label + ': duplicate scoring rows')
    left = frame[columns].astype(str).sort_values(columns).reset_index(drop=True)
    right = expected[columns].astype(str).sort_values(columns).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(label + ': frozen scoring mask changed')


def _gain_rows(spec, frame, gain):
    gain = np.asarray(gain, float)
    if len(gain) != len(frame) or not np.isfinite(gain).all():
        raise ValueError('Invalid paired gains: ' + spec['comparison'])
    result = frame[['market', 'date']].copy()
    result['gain'] = gain
    result['family'] = spec['family']
    result['comparison'] = spec['comparison']
    return result[['family', 'comparison', 'market', 'date', 'gain']]


def _finish(spec, frame=None, gain=None, reason=None, **bootstrap):
    if reason is not None:
        return {**spec, 'status': 'unavailable_reserved_slot', 'reason': reason,
                'p_null': 1.0}, None
    rows = _gain_rows(spec, frame, gain)
    summary = paired_summary(rows.date, rows.market, rows.gain, **bootstrap)
    return {**spec, 'status': 'available', **summary}, rows


def _adjust(results):
    for family in sorted(set(r['family'] for r in results)):
        selected = [r for r in results if r['family'] == family]
        for row, p in zip(selected, holm([r['p_null'] for r in selected])):
            row['holm_p'] = float(p)
            row['supported_positive'] = bool(row['status'] == 'available' and
                row['mean_gain'] > 0 and row['interval95'][0] > 0 and p <= .05)


def score_index_frame(frame, **bootstrap):
    """Score audited comparator arrays plus the newly generated network arrays."""
    bootstrap = {**BOOTSTRAP, **bootstrap}
    results, gains = [], []
    for spec in comparison_registry():
        if not spec['family'].startswith('index_'):
            continue
        control = spec['control']
        if not frame[control + '_valid'].all():
            result, rows = _finish(spec, reason='Archived comparator unavailable on full frozen panel')
        elif spec['metric'] == 'native_log_score':
            result, rows = _finish(spec, frame, frame.network_logscore - frame[control + '_logscore'], **bootstrap)
        elif spec['metric'] == 'common_t8_log_score':
            r = frame['return'].to_numpy()
            gain = common_t8_log_score(r, frame.network_second_moment) - common_t8_log_score(r, frame[control + '_second_moment'])
            result, rows = _finish(spec, frame, gain, **bootstrap)
        else:
            target = spec['target']
            y = frame['return'].to_numpy() ** 2 if target == 'r2' else frame.rk.to_numpy()
            valid = np.isfinite(y) & (y >= 0 if target == 'r2' else y > 0)
            sub = frame.loc[valid]
            if set(sub.market) != set(frame.market):
                result, rows = _finish(spec, reason='Proxy missing for an entire market')
            else:
                field = 'second_moment' if target == 'r2' else 'variance'
                gain = qlike(y[valid], sub[control + '_' + field]) - qlike(y[valid], sub['network_' + field])
                result, rows = _finish(spec, sub, gain, **bootstrap)
        results.append(result)
        if rows is not None:
            gains.append(rows)
    _adjust(results)
    return results, gains


def score_calendar_arrays(panel, distributions, features, **bootstrap):
    """Exactly the archived correction fitter, masks, features and family sizes."""
    from experiments.session_information_pilot.run import evaluate
    from experiments.session_calendar_screen.run import combine_results
    bootstrap = {**BOOTSTRAP, **bootstrap}
    if set(features) != set(SCENARIOS):
        raise ValueError('Both timing scenarios must be supplied')
    scenarios, details, saved, results, gains = {}, {}, {}, [], []
    for scenario in SCENARIOS:
        result, test_panel, forecasts = evaluate(panel, distributions, features[scenario], **bootstrap)
        scenarios[scenario] = result
        details[scenario] = {k: v for k, v in result.items() if k not in
                            ('family_a', 'family_b', 'channel_gate_passed', 'timing_gate_passed', 'interpretation')}
        saved[scenario] = (test_panel, forecasts)
        for spec in comparison_registry():
            if not spec['family'].startswith('calendar_') or spec['scenario'] != scenario:
                continue
            candidate, control = spec['candidate'], spec['control']
            if candidate not in forecasts or control not in forecasts:
                summary, rows = _finish(spec, reason='Unimplemented adaptive slot retained')
            else:
                target = spec['target']
                y = test_panel.r.to_numpy() ** 2 if target == 'r2' else test_panel.rk.to_numpy()
                valid = np.isfinite(y) & (y >= 0 if target == 'r2' else y > 0)
                field = 'second_moment' if target == 'r2' else 'variance'
                if set(test_panel.loc[valid, 'market']) != set(test_panel.market):
                    summary, rows = _finish(spec, reason='Proxy missing for an entire market')
                else:
                    gain = qlike(y[valid], forecasts[control][field][valid]) - qlike(y[valid], forecasts[candidate][field][valid])
                    summary, rows = _finish(spec, test_panel.loc[valid], gain, **bootstrap)
            results.append(summary)
            if rows is not None:
                gains.append(rows)
    _adjust(results)
    return results, gains, dict(joint=combine_results(scenarios), details=details), saved


def score_transfer_frames(frames, expected, **bootstrap):
    bootstrap = {**BOOTSTRAP, **bootstrap}
    results, gains, legacy = [], [], {}
    for population in ('stocks', 'fx'):
        frame = frames[population]
        legacy[population] = analyze_population(frame, population, expected[population],
            ref_frame=frames['indices'], margin=.003, **bootstrap)
        for spec in comparison_registry():
            if spec['family'] != 'transfer_' + population:
                continue
            candidate, control = spec['candidate'], spec['control']
            complete = (set(frame.market) == set(expected[population]) and
                np.isfinite(frame[[candidate, control]].to_numpy()).all() and
                (frame[[candidate, control]].to_numpy() > 0).all())
            if complete:
                gain = common_t8_log_score(frame.r, frame[candidate]) - common_t8_log_score(frame.r, frame[control])
                # The frozen transfer study uses basic rather than percentile CIs.
                old = next(r for r in legacy[population]['log_score_contrasts']
                           if r['candidate'] == candidate and r['control'] == control)
                summary = {**spec, **old, 'status': 'available'}
                rows = _gain_rows(spec, frame, gain)
            else:
                summary, rows = _finish(spec, reason='Archived fit failure or incomplete frozen population; no complete-case superiority claim')
            results.append(summary)
            if rows is not None:
                gains.append(rows)
    _adjust(results)
    return results, gains, legacy


def _prediction(predictor, model, r):
    prediction = predictor(model, np.asarray(r, float), L=1024, stride=512,
                           min_history=100, batch_windows=4, levels=())
    n = len(r)
    for key in ('h', 'mean', 'logscore', *PARAMS):
        if key not in prediction or np.asarray(prediction[key]).shape[0] != n:
            raise ValueError('Prediction omitted aligned field: ' + key)
        if not np.isfinite(np.asarray(prediction[key])[100:]).all():
            raise ValueError('Nonfinite issued prediction: ' + key)
    variance = np.asarray(prediction['h']) - np.asarray(prediction['mean']) ** 2
    if np.any(variance[100:] <= 0) or np.any(np.asarray(prediction['h'])[100:] <= 0):
        raise ValueError('Invalid issued neural variance/second moment')
    return prediction


def _save_prediction(out, key, date, r, prediction):
    path = Path(out) / 'forecasts' / (key + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, date=np.asarray(date, dtype='U10'), r=np.asarray(r, float),
                        **{k: np.asarray(prediction[k]) for k in ('h', 'mean', 'logscore', *PARAMS)})
    return path


def evaluate_checkpoint(ckpt, out, *, input_inventory, predictor=None, model=None,
                        device='cpu', reps=2000, block=60, seed=927601):
    """Evaluate one synthetically selected checkpoint after ALL base training.

    The caller enforces the all-seeds-trained barrier and freezes this adapter,
    the supplied inventory and training configuration first. This routine only
    fits each seed's one-dimensional 2011–2014 calendar corrections.
    """
    verify_inventory(input_inventory)
    root, out = Path(input_inventory['root']), Path(out).resolve()
    protected = [root / p for p in (STRENGTH, CALENDAR, BASE, TRANSFER)]
    if any(out == p or p in out.parents for p in protected):
        raise ValueError('Evaluation output cannot alter an archived study')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Seed evaluation output exists; refusing overwrite')
    out.mkdir(parents=True, exist_ok=True)
    if predictor is None or model is None:
        from . import causal
        if predictor is None:
            predictor = causal.predict_series
        if model is None:
            model = causal.load_model(str(ckpt), device)
    kwargs = dict(reps=reps, block=block, seed=seed)
    save(out / 'identity.json', dict(checkpoint=str(Path(ckpt).resolve()), checkpoint_sha256=sha(ckpt),
         input_hashes=input_inventory['hashes'], status='evaluating', bootstrap=kwargs))
    all_results, all_gains, legacy = [], [], {}

    # Reserved confirmation: full original return history, exact archived dates.
    index_frames = []
    for name in input_inventory['studies']['index_markets']:
        with np.load(root / INDEX_DATA / (name + '.npz'), allow_pickle=False) as z:
            dates = z['date'].astype('U10'); r = z['returns'].astype(float) * 100.
        prediction = _prediction(predictor, model, r)
        _save_prediction(out, 'index_' + name, dates, r, prediction)
        frame = pd.read_csv(root / STRENGTH / 'cells' / (name + '.csv.gz'), dtype={'market': str, 'date': str})
        idx = pd.Index(dates).get_indexer(frame.date)
        if np.any(idx < 100):
            raise ValueError('Index scoring date missing/inside warmup')
        np.testing.assert_allclose(r[idx], frame['return'], rtol=1e-12, atol=1e-12)
        for field, values in dict(mean=prediction['mean'], second_moment=prediction['h'],
                                  variance=prediction['h'] - prediction['mean'] ** 2,
                                  logscore=prediction['logscore']).items():
            frame['network_' + field] = values[idx]
        index_frames.append(frame)
    frame = pd.concat(index_frames, ignore_index=True)
    _assert_mask(frame, pd.read_csv(root / STRENGTH / 'scoring_mask.csv', dtype=str), 'indices')
    frame.to_csv(out / 'index_evaluated.csv.gz', index=False)
    results, gains = score_index_frame(frame, **kwargs)
    all_results.extend(results); all_gains.extend(gains)

    # Calendar screen: retain exact features and comparator mixtures, change only
    # the new network distribution and its development-only scale coefficient.
    from experiments.session_calendar_screen.run import load_inputs
    panel, distributions, _, _, _ = load_inputs(root / CALENDAR, root / BASE)
    _assert_mask(panel, pd.read_csv(root / CALENDAR / 'scoring_mask.csv', dtype=str), 'calendar')
    replacement = {k: [] for k in PARAMS}
    for name in panel.market.drop_duplicates():
        with np.load(root / BASE / 'forecasts' / (name + '.npz'), allow_pickle=False) as z:
            dates, r = z['date'].astype('U10'), z['r'].astype(float)
        prediction = _prediction(predictor, model, r)
        _save_prediction(out, 'calendar_' + name, dates, r, prediction)
        selected = panel.loc[panel.market == name]
        idx = pd.Index(dates).get_indexer(selected.date)
        if np.any(idx < 100):
            raise ValueError('Calendar scoring date missing/inside warmup')
        np.testing.assert_allclose(r[idx], selected.r, rtol=1e-12, atol=1e-12)
        for key in PARAMS:
            replacement[key].append(prediction[key][idx])
    distributions['network'] = {k: np.concatenate(v) for k, v in replacement.items()}
    features = {scenario: pd.read_csv(root / CALENDAR / (scenario + '_features.csv.gz'),
                                    dtype={'market': str, 'date': str}) for scenario in SCENARIOS}
    results, gains, legacy['calendar'], saved = score_calendar_arrays(panel, distributions, features, **kwargs)
    all_results.extend(results); all_gains.extend(gains)
    for scenario, (test_panel, forecasts) in saved.items():
        for name, values in forecasts.items():
            np.savez_compressed(out / ('calendar_' + scenario + '_' + name + '.npz'),
                market=test_panel.market.to_numpy(dtype='U20'), date=test_panel.date.to_numpy(dtype='U10'), **values)

    # Transfer: reuse ALL fitted compact forecasts exactly, including failed fits
    # and unstable paths. Histories start in 2005 exactly as in the frozen study.
    frames = {}
    for population in ('indices', 'stocks', 'fx'):
        blocks = []
        expected = input_inventory['studies']['transfer_expected'][population]
        for name in expected:
            with np.load(root / TRANSFER / 'forecasts' / (population + '_' + name + '.npz'), allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            prediction = _prediction(predictor, model, arrays['r'])
            _save_prediction(out, 'transfer_' + population + '_' + name, arrays['date'], arrays['r'], prediction)
            arrays['VolPFN'] = prediction['h']
            selected = arrays['date'] >= '2015-01-01'
            blocks.append(pd.DataFrame(dict(market=name, date=arrays['date'][selected], r=arrays['r'][selected],
                **{k: arrays[k][selected] for k in ('R_eq', 'R_class', 'R_market', 'R_archive', 'VolPFN')})))
        frames[population] = pd.concat(blocks, ignore_index=True)
        mask = pd.read_csv(root / TRANSFER / (population + '_mask.csv'), dtype=str)
        _assert_mask(frames[population], mask.loc[mask.date >= '2015-01-01'], 'transfer_' + population)
        frames[population].to_csv(out / ('transfer_' + population + '_evaluated.csv.gz'), index=False)
    results, gains, legacy['transfer'] = score_transfer_frames(frames,
        input_inventory['studies']['transfer_expected'], **kwargs)
    all_results.extend(results); all_gains.extend(gains)

    verify_inventory(input_inventory)
    gain_frame = pd.concat(all_gains, ignore_index=True)
    gain_frame.to_csv(out / 'loss_differences.csv.gz', index=False)
    save(out / 'comparison_registry.json', all_results)
    save(out / 'individual_results.json', legacy)
    artifacts = {str(p.resolve()): sha(p) for p in out.rglob('*') if p.is_file()}
    result = dict(status='complete', checkpoint_sha256=sha(ckpt), comparisons=len(all_results),
                  loss_rows=len(gain_frame), hashes=artifacts)
    save(out / 'completed_artifacts.json', result)
    return result


def synthetic_integration(out, *, predictor=None, model=None, reps=39):
    """Exercise all three scoring routes with generated data and no file reads.

    Supplying the smoke checkpoint's predictor/model additionally exercises
    actual neural inference. This does not call inventory_inputs or any market
    loader, and cannot be used to view a preliminary empirical outcome.
    """
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Synthetic integration output exists')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(51702)
    dates = pd.bdate_range('2013-01-01', periods=780).strftime('%Y-%m-%d').to_numpy()
    panels, distributions, index_frames, transfer = [], [], [], []
    for number, name in enumerate(('FAKE_A', 'FAKE_B')):
        r = rng.standard_t(8, len(dates)) * np.sqrt(.75) * (1.0 + .1 * number)
        if predictor is None:
            h = np.full(len(r), 1. + .2 * number)
            scale = np.sqrt(.75 * h)[:, None]
            pred = dict(h=h, mean=np.zeros(len(r)), logscore=common_t8_log_score(r, h),
                        logw=np.zeros((len(r), 1)), loc=np.zeros((len(r), 1)),
                        scale=scale, nu=np.full((len(r), 1), 8.))
        else:
            pred = _prediction(predictor, model, r)
        choose = np.arange(len(r)) >= 100
        panels.append(pd.DataFrame(dict(market=name, date=dates[choose], r=r[choose], rk=.2 + r[choose] ** 2)))
        distributions.append({k: pred[k][choose] for k in PARAMS})
        test = choose & (dates >= '2015-01-01')
        f = pd.DataFrame(dict(market=name, date=dates[test], return_=r[test], rk=.2 + r[test] ** 2)).rename(columns={'return_': 'return'})
        f['network_mean'] = pred['mean'][test]
        f['network_second_moment'] = pred['h'][test]
        f['network_variance'] = pred['h'][test] - pred['mean'][test] ** 2
        f['network_logscore'] = pred['logscore'][test]
        for control in CONTROLS:
            f[control + '_valid'] = control != 'egarch_t'
            for field in ('mean', 'variance', 'second_moment', 'logscore'):
                f[control + '_' + field] = f['network_' + field]
        index_frames.append(f)
        transfer.append(pd.DataFrame(dict(market=name, date=dates[test], r=r[test],
            VolPFN=pred['h'][test], R_eq=np.full(test.sum(), 1.4),
            R_class=np.full(test.sum(), 1.2), R_market=np.full(test.sum(), 1.3), R_archive=np.full(test.sum(), 1.4))))
    panel = pd.concat(panels, ignore_index=True)
    dist = {k: np.concatenate([d[k] for d in distributions]) for k in PARAMS}
    features = panel[['market', 'date']].copy()
    features['lagged_x'] = 0.; features['latest_x'] = 0.
    kwargs = dict(reps=reps, block=20, seed=927601)
    results, gains = score_index_frame(pd.concat(index_frames, ignore_index=True), **kwargs)
    c, cg, cal, _ = score_calendar_arrays(panel, {k: dist for k in ('network', 'compact', 'gjr')},
                                          {k: features.copy() for k in SCENARIOS}, **kwargs)
    tf = pd.concat(transfer, ignore_index=True)
    frames = {k: tf.copy() for k in ('indices', 'stocks', 'fx')}
    expected = {k: ['FAKE_A', 'FAKE_B'] for k in frames}
    t, tg, tr = score_transfer_frames(frames, expected, **kwargs)
    results += c + t; gains += cg + tg
    pd.concat(gains, ignore_index=True).to_csv(out / 'loss_differences.csv.gz', index=False)
    save(out / 'comparison_registry.json', results)
    value = dict(status='passed', kind='synthetic_only_no_market_input_access', comparisons=len(results),
                 calendar=cal, transfer=tr)
    save(out / 'synthetic_integration.json', value)
    return value
