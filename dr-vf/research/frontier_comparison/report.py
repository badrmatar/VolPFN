"""Fail-closed summaries for a bounded, development-only forecast comparison."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from research.frontier_benchmark import metrics


SCORES = ('mean_pinball', 'qlike_uncalibrated', 'qlike_calibrated')
COMPLETE = 'complete_engineering_only'
PHASE_PERIODS = {
    'calibration': {'start': '2012-01-01', 'end': '2013-12-31'},
    'evaluation': {'start': '2014-01-01', 'end': '2014-12-31'},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row):
    return (row['market'], row['date'], int(row['index']), row['phase'])


def _equal_market(values, rows, markets):
    per_market = {
        market: float(np.mean([value for value, row in zip(values, rows)
                               if row['market'] == market]))
        for market in markets
    }
    return per_market, float(np.mean(list(per_market.values())))


def _validate_manifest(manifest):
    markets = manifest['selected_markets']
    if not markets or len(set(markets)) != len(markets):
        raise ValueError('Selected markets must be nonempty and unique')
    if len(set(manifest['order'])) != len(manifest['order']):
        raise ValueError('Duplicate model identity')
    if manifest.get('final_period_scoring_authorized') is not False:
        raise ValueError('Report only supports development scoring')
    for phase, period in PHASE_PERIODS.items():
        if manifest.get(phase + '_period') != [period['start'], period['end']]:
            raise ValueError('Unexpected development calibration/evaluation periods')
    origins = manifest['origins']
    keys = [_key(row) for row in origins]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Duplicate or empty manifest origins')
    market_dates = [(row['market'], row['date']) for row in origins]
    if len(market_dates) != len(set(market_dates)):
        raise ValueError('Duplicate market-date in manifest')
    for row in origins:
        period = PHASE_PERIODS.get(row['phase'])
        if (row['market'] not in markets or period is None
                or not period['start'] <= row['date'] <= period['end']):
            raise ValueError('Unexpected market, phase or date in manifest')
    for market in markets:
        for phase in ('calibration', 'evaluation'):
            if not any(row['market'] == market and row['phase'] == phase for row in origins):
                raise ValueError('Missing market-phase in manifest')
    levels = np.asarray(manifest['quantiles'], float)
    if levels.shape != metrics.QUANTILES.shape or not np.allclose(levels, metrics.QUANTILES, rtol=0, atol=1e-15):
        raise ValueError('Unexpected quantile grid')
    return markets, origins, keys, levels


def _load_complete_rows(out, name, status, origins, expected_keys, levels, targets):
    path = out / 'models' / name / 'predictions.json'
    if not path.exists() or sha(path) != status.get('forecast_sha256'):
        raise ValueError(name + ': prediction file absent or hash does not match completed status')
    rows = json.loads(path.read_text())
    actual_keys = [_key(row) for row in rows]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError(name + ': duplicate forecast rows')
    if len(rows) != len(origins) or set(actual_keys) != set(expected_keys):
        raise ValueError(name + ': incomplete or unexpected forecast rows; no row dropping allowed')
    by_key = dict(zip(actual_keys, rows))
    rows = [by_key[key] for key in expected_keys]
    for key, row in zip(expected_keys, rows):
        if not np.isfinite(row['target']) or row['target'] != targets[key]:
            raise ValueError(name + ': forecast target does not match frozen shared target')
        q = np.asarray(row['raw_quantiles'], float)
        if q.shape != levels.shape or not np.isfinite(q).all() or np.any(np.diff(q) < 0):
            raise ValueError(name + ': invalid raw quantiles')
        metrics.qlike(row['target'] ** 2, row['h'])
        for flag in ('floor_activated', 'quantiles_rearranged'):
            if not isinstance(row[flag], bool):
                raise ValueError(name + ': missing or invalid audit flag ' + flag)
        if row.get('inference_seconds', 0) < 0 or not np.isfinite(row.get('inference_seconds', 0)):
            raise ValueError(name + ': invalid inference duration')
    return rows, path


def build_report(out):
    """Read only complete model cells. A malformed completed cell aborts reporting."""
    out = Path(out).resolve()
    manifest_path = out / 'manifest.json'
    manifest_digest = sha(manifest_path)
    if (out / 'manifest.sha256').read_text().strip() != manifest_digest:
        raise ValueError('Manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    markets, origins, expected_keys, levels = _validate_manifest(manifest)
    snapshot = out / 'development_data.npz'
    if sha(snapshot) != manifest['development_snapshot_sha256']:
        raise ValueError('Development snapshot hash mismatch')
    targets = {}
    with np.load(snapshot, allow_pickle=False) as data:
        for key, row in zip(expected_keys, origins):
            dates, returns = data[row['market'] + '__date'], data[row['market'] + '__r']
            if row['index'] < 0 or row['index'] >= len(dates) or str(dates[row['index']]) != row['date']:
                raise ValueError('Frozen origin does not match development snapshot')
            targets[key] = float(returns[row['index']])
    inputs = {str(manifest_path): manifest_digest, str(out / 'manifest.sha256'): sha(out / 'manifest.sha256'),
              str(snapshot): sha(snapshot), str(Path(__file__).resolve()): sha(__file__)}
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'bounded_development_comparison',
        'selected_markets': markets,
        'model_order': manifest['order'],
        'expected_origins': len(origins),
        'calibration_origins': sum(row['phase'] == 'calibration' for row in origins),
        'evaluation_origins': sum(row['phase'] == 'evaluation' for row in origins),
        'weighting': 'mean within each market, then equal mean across all selected markets',
        'phase_periods': PHASE_PERIODS,
        'calibration': 'one positive QLIKE scale per model fitted only on sampled 2012–2013 rows; evaluation is sampled 2014 rows',
        'difference_convention': 'comparator loss minus VolPFN loss; positive favours VolPFN on these observations',
        'inferential_claims': False,
        'final_period_scoring_performed': False,
        'models': {}, 'unavailable': {}, 'input_hashes': inputs,
    }
    for name in manifest['order']:
        status_path = out / 'models' / name / 'status.json'
        if not status_path.exists():
            report['unavailable'][name] = {'status': 'not_started', 'comparison': 'not ranked or scored'}
            continue
        inputs[str(status_path)] = sha(status_path)
        status = json.loads(status_path.read_text())
        if status.get('status') != COMPLETE:
            report['unavailable'][name] = {
                'status': status.get('status', 'unknown'), 'comparison': 'not ranked or scored',
                'error': status.get('error', status.get('reason')),
            }
            continue
        rows, prediction_path = _load_complete_rows(
            out, name, status, origins, expected_keys, levels, targets)
        inputs[str(prediction_path)] = sha(prediction_path)
        calibration = [row for row in rows if row['phase'] == 'calibration']
        evaluation = [row for row in rows if row['phase'] == 'evaluation']
        scale = metrics.fit_scale(
            [row['target'] ** 2 for row in calibration], [row['h'] for row in calibration],
            [row['market'] for row in calibration], [row['date'] for row in calibration])
        y = np.asarray([row['target'] for row in evaluation])
        h = np.asarray([row['h'] for row in evaluation])
        scores = {
            'mean_pinball': metrics.pinball(y, [row['raw_quantiles'] for row in evaluation], levels),
            'qlike_uncalibrated': metrics.qlike(y * y, h),
            'qlike_calibrated': metrics.qlike(y * y, scale * h),
        }
        result = {
            'status': COMPLETE, 'scale': scale,
            'calibration_rows': len(calibration), 'evaluation_rows': len(evaluation),
            'per_market': {market: {'calibration_rows': sum(r['market'] == market for r in calibration),
                                    'evaluation_rows': sum(r['market'] == market for r in evaluation)}
                           for market in markets},
            'equal_market_mean': {},
            'point_semantics': sorted(set(row['h_semantics'] for row in rows)),
            'audit_counts': {
                phase: {flag: sum(row[flag] for row in rows if row['phase'] == phase)
                        for flag in ('floor_activated', 'quantiles_rearranged')}
                for phase in ('calibration', 'evaluation')
            },
            'mean_inference_seconds': float(np.mean([row.get('inference_seconds', 0) for row in rows])),
        }
        for metric, values in scores.items():
            per_market, mean = _equal_market(values, evaluation, markets)
            result['equal_market_mean'][metric] = mean
            for market, value in per_market.items():
                result['per_market'][market][metric] = value
        report['models'][name] = result
    reference = report['models'].get('volpfn')
    for name, model in report['models'].items():
        model['difference_vs_volpfn'] = (
            {metric: model['equal_market_mean'][metric] - reference['equal_market_mean'][metric]
             for metric in SCORES} if reference is not None else None)
        for market in markets:
            model['per_market'][market]['difference_vs_volpfn'] = (
                {metric: model['per_market'][market][metric] - reference['per_market'][market][metric]
                 for metric in SCORES} if reference is not None else None)
    report['status'] = 'complete' if not report['unavailable'] else 'partial'
    return report


def _readout(report):
    lines = ['# Bounded development comparison', '',
             'Exploratory screening on fixed monthly samples: the first common available session of each month '
             'across the selected markets in 2012–2013 calibrates one scale per model, and the corresponding sampled sessions in 2014 '
             'are evaluated. All models use the same frozen origins and targets. No rows are dropped.', '',
             f"Report status: **{report['status']}**. Markets: {', '.join(report['selected_markets'])}. "
             f"Calibration rows: {report['calibration_origins']}; evaluation rows: {report['evaluation_origins']}.", '',
             'These are development observations that have already been available to the project. '
             'The final evaluation period remains unscored. There are no p-values, significance claims '
             'or population-level rankings from this small screen. VolPFN uses fixed-final seed 8101; '
             'this is not a seed-mean headline result. The 32-update LSTM is excluded because it is an '
             'integration check rather than a competitively trained comparator.', '',
             'Losses below are averaged within each market and then equally across markets. '
             'Lower losses are better on this sample. The calibrated QLIKE column uses only the '
             'sampled 2012–2013 outcomes to fit its multiplier, with no 2014 outcomes in calibration. '
             'This is a screening subset of the 2011–2014 development period, not the final calibration protocol.', '',
             '| Model | Scale | Raw quantile pinball | Native-point QLIKE | Calibrated QLIKE |',
             '|---|---:|---:|---:|---:|']
    for name in report['model_order']:
        if name in report['models']:
            model = report['models'][name]; means = model['equal_market_mean']
            lines.append(f"| {name} | {model['scale']:.6f} | {means['mean_pinball']:.6f} | "
                         f"{means['qlike_uncalibrated']:.6f} | {means['qlike_calibrated']:.6f} |")
        else:
            lines.append(f"| {name} ({report['unavailable'][name]['status']}; unavailable) | — | — | — | — |")
    lines += ['', 'Differences are comparator loss minus VolPFN loss. Positive values favour VolPFN '
              'on these observations; they are descriptive, with no inference beyond the sampled dates.', '',
              '| Model | Pinball difference | Native QLIKE difference | Calibrated QLIKE difference |',
              '|---|---:|---:|---:|']
    for name in report['model_order']:
        if name not in report['models']:
            continue
        difference = report['models'][name]['difference_vs_volpfn']
        if difference is not None:
            lines.append('| ' + name + ' | ' + ' | '.join(f'{difference[m]:+.6f}' for m in SCORES) + ' |')
    if 'volpfn' not in report['models']:
        lines.append('VolPFN is unavailable; no differences are calculated.')
    lines += ['', '## Per-market evaluation losses', '',
              '| Market | Model | Pinball | Native QLIKE | Calibrated QLIKE | Δ pinball | Δ native QLIKE | Δ calibrated QLIKE |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for market in report['selected_markets']:
        for name in report['model_order']:
            if name in report['models']:
                values = report['models'][name]['per_market'][market]
                differences = values['difference_vs_volpfn']
                delta_cells = [f'{differences[m]:+.6f}' for m in SCORES] if differences is not None else ['—'] * 3
                lines.append('| ' + market + ' | ' + name + ' | ' +
                             ' | '.join(f'{values[m]:.6f}' for m in SCORES) + ' | ' +
                             ' | '.join(delta_cells) + ' |')
    lines += ['', '## Interpretation and audit', '',
              'Raw-return quantiles measure distributional calibration and sharpness through pinball loss; '
              'they are not a full-density comparison. The squared-return points retain their native semantics: '
              'some are medians, whereas VolPFN supplies a mixture second moment. Scale adjustment improves '
              'comparability for QLIKE but does not turn a median into a conditional expectation. '
              'No cross-model native log-density ranking is made. All models have access to the same '
              'past raw returns, but use different input representations: Chronos and Moirai receive '
              'squared history alone on the squared-return track, whereas TabPFN also receives eight '
              'causal features derived from raw returns. This is not an identical-input-tensor experiment.', '',
              '| Model | Calibration floors / rearrangements | Evaluation floors / rearrangements | Mean origin seconds |',
              '|---|---:|---:|---:|']
    for name in report['model_order']:
        if name in report['models']:
            model = report['models'][name]; counts = model['audit_counts']
            cells = [f"{counts[phase]['floor_activated']} / {counts[phase]['quantiles_rearranged']}"
                     for phase in ('calibration', 'evaluation')]
            lines.append(f"| {name} | {cells[0]} | {cells[1]} | {model['mean_inference_seconds']:.3f} |")
    lines += ['', 'The JSON report retains per-market differences, point semantics and input hashes. '
              'The separate hash record records the report outputs. Incomplete or failed models remain '
              'explicitly unavailable; malformed completed predictions stop reporting rather than changing the sample.', '']
    return '\n'.join(lines)


def write_report(out):
    out = Path(out).resolve()
    report = build_report(out)
    report_path, readout_path = out / 'comparison.json', out / 'READOUT.md'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    readout_path.write_text(_readout(report))
    hashes = {'inputs': report['input_hashes'], 'outputs': {
        str(report_path): sha(report_path), str(readout_path): sha(readout_path)}}
    (out / 'comparison_hashes.json').write_text(json.dumps(hashes, indent=2, allow_nan=False) + '\n')
    return report
