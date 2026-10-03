"""Separate exploratory calendar screen; never satisfies the strict pilot gate."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.session_information_pilot.run import (
    ROOT, MODELS, PARAMS, DEFAULT_OUT as BASE_OUT, PREVIOUS,
    evaluate, locked_manifest, sha, save, now,
)
from research.session_information_pilot.sessions import build_session_features
from research.evaluation_strengthening.statistics import holm

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT/'research/outputs/session_calendar_screen_v1'
SCENARIOS = {
    'nominal': {'lead_minutes': 0, 'buffer_minutes': 15},
    'conservative': {'lead_minutes': 30, 'buffer_minutes': 60},
}


def shared_eligibility(sessions):
    """Only timing/coverage determines exclusion, never return or model outcomes.

    The predecessor is the immediately preceding *observed* row in the cache,
    not the preceding retained calendar row. No gap may be silently bridged.
    """
    s = sessions.copy()
    required = {'market', 'date', 'open_utc', 'close_utc',
                'predecessor_unmatched', 'predecessor_close_utc'}
    if not required.issubset(s.columns):
        raise ValueError('Missing calendar/predecessor columns')
    if s.duplicated(['market', 'date']).any():
        raise ValueError('Duplicate sessions')
    for c in ('open_utc', 'close_utc', 'predecessor_close_utc'):
        s[c] = pd.to_datetime(s[c], utc=True, errors='raise')
    if s[['open_utc', 'close_utc']].isna().any().any() or (s.close_utc <= s.open_utc).any():
        raise ValueError('Invalid modeled session endpoints')
    missing = s.predecessor_unmatched.astype(int).ne(0) | s.predecessor_close_utc.isna()
    s['eligible'] = ~missing
    s['reason'] = np.where(missing, 'preceding_observed_session_unmapped', '')
    for spec in SCENARIOS.values():
        unavailable = (s.predecessor_close_utc + pd.Timedelta(minutes=spec['buffer_minutes']) >=
                       s.open_utc - pd.Timedelta(minutes=spec['lead_minutes']))
        s.loc[~missing & unavailable, 'reason'] = 'preceding_observed_session_unavailable'
        s['eligible'] &= ~unavailable
    return s


def combine_results(scenarios):
    """Joint families across BOTH scenarios, with missing/adaptive slots retained."""
    if set(scenarios) != set(SCENARIOS):
        raise ValueError('Both registered scenarios required')
    result = {'strict_gate_passed': False,
              'strict_gate_status': 'not_tested_unverified_source_session_endpoints',
              'status': 'exploratory_calendar_screen_complete',
              'scenario_descriptive_hurdles': {}}
    for family, expected in (('family_a', 6), ('family_b', 4)):
        rows = []
        for scenario in SCENARIOS:
            source = scenarios[scenario][family]
            if len(source) != expected:
                raise ValueError('Reserved comparison slots missing')
            rows.extend(dict(copy.deepcopy(r), scenario=scenario) for r in source)
        for row, adjusted in zip(rows, holm([r['p_null'] for r in rows])):
            row['holm_p'] = float(adjusted)
            row['supported_positive'] = bool(
                row['status'] == 'scored' and row['mean_gain'] > 0 and
                row['interval95'][0] > 0 and adjusted <= .05)
        result[family] = rows
    for scenario in SCENARIOS:
        channel = [r for r in result['family_a'] if r['scenario'] == scenario and r['candidate'] == 'network_latest']
        timing = [r for r in result['family_b'] if r['scenario'] == scenario and r['candidate'] == 'network_latest']
        if len(channel) != 2 or len(timing) != 2:
            raise ValueError('Network comparison slots missing')
        result['scenario_descriptive_hurdles'][scenario] = {
            'channel_numeric_hurdle': all(r['supported_positive'] and r.get('mean_gain', 0) >= .010 for r in channel),
            'timing_supported_on_both_proxies': all(r['supported_positive'] for r in timing),
        }
    result['both_scenarios_channel_numeric_hurdle'] = all(
        r['channel_numeric_hurdle'] for r in result['scenario_descriptive_hurdles'].values())
    result['interpretation'] = (
        'Previously inspected retrospective panel with modeled exchange sessions. '
        'Joint Holm across 12 channel and 8 timing/adaptive slots; intervals are ordinary. '
        'Not the original verified-session test; no new-method, causality or no-harm claim.')
    return result


def load_inputs(out, base):
    locked_manifest(base/'preparation.json')
    provenance = json.loads((out/'provenance.json').read_text())
    if provenance.get('package_version') != '4.11.3' or provenance.get('timestamp_basis') != 'pinned_exchange_calendar_model':
        raise ValueError('Unexpected calendar reconstruction')
    for filename, digest in provenance['outputs_sha256'].items():
        if sha(out/filename) != digest: raise RuntimeError('Calendar artifact changed: '+filename)
    calendar = pd.read_csv(out/'modeled_sessions.csv.gz', dtype={'market': str, 'date': str})
    if (calendar.date > '2022-02-25').any(): raise ValueError('Unexpected post-period calendar dates')
    eligible = shared_eligibility(calendar)
    frames, donors, hashes = [], [], {}
    distributions = {m: {k: [] for k in PARAMS} for m in MODELS}
    for name, rows in eligible.groupby('market', sort=True):
        path, meta_path = base/'forecasts'/(name+'.npz'), base/'forecasts'/(name+'.json')
        meta = json.loads(meta_path.read_text())
        if meta['cache_sha256'] != sha(path) or meta['preparation_sha256'] != sha(base/'preparation.json'):
            raise RuntimeError('Invalid frozen base cache: '+name)
        if provenance['source_cache_sha256'].get(str(path.resolve())) != sha(path):
            raise RuntimeError('Calendar built from a different forecast cache: '+name)
        for p in (path, meta_path): hashes[str(p.resolve())] = sha(p)
        with np.load(path, allow_pickle=False) as z:
            rows = rows.loc[rows.eligible].sort_values('date').copy()
            indices = rows.source_row_index.to_numpy(int)
            if np.any(indices < 100): raise ValueError('Warmup row in calendar input')
            np.testing.assert_array_equal(z['date'][indices].astype('U10'), rows.date.to_numpy())
            rows['surprise'] = z['surprise'][indices]
            if not np.isfinite(rows.surprise).all(): raise ValueError('Missing causal donor surprise')
            donors.append(rows)
            retained = rows.date >= '2011-01-01'
            chosen = indices[retained.to_numpy()]
            dates = rows.loc[retained, 'date'].to_numpy()
            if not np.any(dates < '2015-01-01') or not np.any(dates >= '2015-01-01'):
                raise ValueError('Market lacks development/evaluation coverage: '+name)
            frames.append(pd.DataFrame({'market': name, 'date': dates,
                                        'r': z['r'][chosen], 'rk': z['rk'][chosen]}))
            for m in MODELS:
                if meta['models'][m]['status'] != 'valid':
                    raise RuntimeError('Unavailable baseline: '+name+' '+m)
                for k in PARAMS:
                    values = z[m+'_'+k][chosen]
                    if not np.isfinite(values).all(): raise ValueError('Invalid baseline row: '+name+' '+m)
                    distributions[m][k].append(values)
    if not frames: raise ValueError('No mapped markets')
    return (pd.concat(frames, ignore_index=True),
            {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in distributions.items()},
            pd.concat(donors, ignore_index=True), eligible, hashes)


def prepare(out=DEFAULT_OUT, base=BASE_OUT):
    """Build features and freeze every input before any coefficient fit."""
    out, base = Path(out).resolve(), Path(base).resolve()
    if (out/'screen_manifest.json').exists():
        return locked_manifest(out/'screen_manifest.json')
    if (out/'results.json').exists(): raise FileExistsError('Existing results without manifest')
    panel, distributions, donors, eligible, hashes = load_inputs(out, base)
    eligible[['market','date','eligible','reason']].to_csv(out/'eligibility.csv.gz', index=False)
    panel[['market','date']].to_csv(out/'scoring_mask.csv', index=False)
    old = pd.read_csv(PREVIOUS/'scoring_mask.csv', dtype=str)[['market','date']]
    actual = panel.loc[panel.date >= '2015-01-01', ['market','date']]
    diff = old.merge(actual, on=['market','date'], how='outer', indicator=True)
    if (diff._merge == 'right_only').any(): raise ValueError('Calendar screen added unseen evaluation dates')
    diff.loc[diff._merge != 'both'].to_csv(out/'original_mask_exclusions.csv.gz', index=False)
    summary = {'status': 'prepared_no_correction_fitted', 'created_utc': now(),
               'markets': sorted(panel.market.unique().tolist()),
               'market_count': int(panel.market.nunique()),
               'development_rows': int((panel.date < '2015-01-01').sum()),
               'evaluation_rows': len(actual), 'original_evaluation_rows': len(old),
               'original_rows_excluded': int((diff._merge == 'left_only').sum()),
               'predecessor_exclusions': int((~eligible.eligible).sum()),
               'calendar_rows': len(eligible), 'scenarios': {}}
    for name, spec in SCENARIOS.items():
        print('BUILD FEATURES', name, len(panel), 'target rows', flush=True)
        sessions = donors.copy()
        sessions['available_utc'] = sessions.close_utc + pd.Timedelta(minutes=spec['buffer_minutes'])
        targets = panel[['market','date']].merge(
            sessions[['market','date','open_utc']], on=['market','date'], validate='one_to_one')
        targets['open_utc'] -= pd.Timedelta(minutes=spec['lead_minutes'])
        features = build_session_features(sessions, targets=targets, max_age_hours=120., clip=3.)
        if not panel[['market','date']].equals(features[['market','date']]): raise ValueError('Feature order mismatch')
        if not (features.new_session_count > 0).any(): raise ValueError('No timing increment: failed positive control')
        features.to_csv(out/(name+'_features.csv.gz'), index=False)
        summary['scenarios'][name] = {
            **spec, 'target_rows': len(features),
            'rows_with_new_sessions': int((features.new_session_count > 0).sum()),
            'zero_donor_rows': int((features.donor_count == 0).sum()),
            'mean_matched_donors': float(features.donor_count.mean()),
            'mean_absolute_timing_increment': float(np.abs(features.timing_increment).mean()),
        }
        print('FEATURES READY', name, summary['scenarios'][name], flush=True)
    save(out/'design_audit.json', summary)
    files = list(PACKAGE.glob('*.py')) + [PACKAGE/'PROTOCOL.md', base/'preparation.json',
        out/'modeled_sessions.csv.gz', out/'coverage.json', out/'excluded_rows.csv.gz',
        out/'eligibility.csv.gz', out/'scoring_mask.csv', out/'original_mask_exclusions.csv.gz',
        out/'design_audit.json']
    files += list(out.glob('*provenance*.json')) + list(out.glob('*.zip'))
    files += [out/(n+'_features.csv.gz') for n in SCENARIOS]
    for p in files: hashes[str(p.resolve())] = sha(p)
    import importlib.metadata
    spec = {'created_utc': now(), 'kind': 'calendar_exploratory_screen', 'hashes': hashes,
            'base_preparation': str(base/'preparation.json'), 'scenarios': SCENARIOS,
            'bootstrap': {'reps': 2000, 'block': 60, 'seed': 927601},
            'environment': {p: importlib.metadata.version(p) for p in ('numpy','pandas','scipy')},
            'family_sizes': {'a': 12, 'b': 8},
            'strict_gate_eligible': False, 'design': summary}
    manifest = locked_manifest(out/'screen_manifest.json', spec)
    print('FROZEN BEFORE FITTING', out/'screen_manifest.json', flush=True)
    return manifest


def write_report(out, result, details, design):
    lines = ['# Calendar-based cross-market screen', '',
        '**Exploratory, previously inspected data; modeled exchange times.** '
        'The original verified-session gate remains untested because source observation endpoints are unresolved.', '',
        f"Markets: {design['market_count']}. Development rows: {design['development_rows']:,}; "
        f"evaluation rows: {design['evaluation_rows']:,} (original panel: {design['original_evaluation_rows']:,}).", '',
        'Both timing assumptions use the same observations. Nominal: modeled open and close +15 minutes. '
        'Conservative: modeled open −30 minutes and close +60 minutes. These are assumptions, not certified latency bounds.', '',
        '| Scenario | Family | Candidate / control | Proxy | QLIKE gain | Ordinary 95% interval | Joint Holm p |',
        '|---|---|---|---|---:|---|---:|']
    for fam in ('a', 'b'):
        for r in result['family_'+fam]:
            if r['status'] == 'scored':
                gain = f"{r['mean_gain']:+.6f}"
                interval = f"[{r['interval95'][0]:+.6f}, {r['interval95'][1]:+.6f}]"
            else: gain, interval = r['status'], '—'
            lines.append(f"| {r['scenario']} | {fam.upper()} | {r['candidate']} / {r['control']} | "
                         f"{r['target']} | {gain} | {interval} | {r['holm_p']:.4f} |")
    lines += ['', 'Positive gain favors the candidate. Family A includes all 12 model/proxy/scenario comparisons; '
              'family B includes eight timing/adaptation slots, with unimplemented adaptation assigned p=1. '
              'Intervals do not have simultaneous familywise coverage.', '',
              '| Scenario | VolPFN channel numerical hurdle, both proxies | Timing supported, both proxies |',
              '|---|---|---|']
    for name, x in result['scenario_descriptive_hurdles'].items():
        lines.append(f"| {name} | {x['channel_numeric_hurdle']} | {x['timing_supported_on_both_proxies']} |")
    lines += ['', '**These descriptive hurdles never constitute a pass of the original strict pilot.**', '',
              '## Development-fitted scale coefficients', '',
              '| Scenario | Model | Constant | Calendar-lagged | Latest available |', '|---|---|---:|---:|---:|']
    for name, d in details.items():
        for model, fits in d['coefficients'].items():
            lines.append(f"| {name} | {model} | {fits['constant']['coefficient']:+.6f} | "
                         f"{fits['lagged']['coefficient']:+.6f} | {fits['latest']['coefficient']:+.6f} |")
    lines += ['', '## Global calibration diagnostic', '',
              'This exploratory mean compares latest information with a development-fitted constant rescaling. '
              'It helps distinguish foreign information from a generic change in predictive scale; no significance claim.', '',
              '| Scenario | Model | Proxy | Latest versus constant QLIKE gain |', '|---|---|---|---:|']
    for name, d in details.items():
        for r in d['constant_calibration_diagnostics']:
            value = r['latest_minus_constant_qlike_gain']
            display = 'unavailable' if value is None else f'{value:+.6f}'
            lines.append(f"| {name} | {r['model']} | {r['target']} | {display} |")
    lines += ['', 'The frozen checkpoint was not retrained. One pooled correction coefficient per feature and model '
              'was fitted only on 2011–2014; comparison dates are 2015–February 2022. The compact-rule selection and '
              'retained network already have a research history, so this is not a fresh holdout. '
              'The baseline distributions, coherent tails, log scores and all correction forecasts are retained.', '',
              'Limits: incomplete calendar/source endpoint alignment; regular-session models rather than actual vendor '
              'release histories; excluded markets and dates; a fixed pooled coefficient; no test of online adaptation, '
              'guard protection, independent training seeds or a new asynchronous inference mechanism. '
              'Bootstrap uncertainty is conditional on the selected forecasts and markets.', '',
              'The predecessor check establishes modeled availability of the immediately preceding observed session. '
              'The cached base forecasts retain their original full histories, including older observations whose '
              'calendar mapping may be absent; this screen does not independently certify every historical endpoint.', '',
              'See coverage.json and excluded_rows.csv.gz for calendar exclusions; eligibility.csv.gz for predecessor '
              'checks; design_audit.json for matched coverage; both *_features.csv.gz files for every donor used; '
              'screen_manifest.json for inputs frozen before fitting; results.json for joint inference; '
              'scenario_details.json for exploratory density/tail diagnostics.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(out=DEFAULT_OUT, base=BASE_OUT):
    out, base = Path(out).resolve(), Path(base).resolve()
    manifest = locked_manifest(out/'screen_manifest.json')
    if str(base/'preparation.json') != manifest['base_preparation']: raise ValueError('Changed base directory')
    if (out/'results.json').exists(): raise FileExistsError('Completed screen preserved; refusing overwrite')
    panel, distributions, _, _, _ = load_inputs(out, base)
    mask = pd.read_csv(out/'scoring_mask.csv', dtype=str)
    if not panel[['market','date']].equals(mask): raise ValueError('Frozen scoring mask changed')
    scenario_results, details, artifacts = {}, {}, []
    for name in SCENARIOS:
        print('FIT AND SCORE', name, '2011–2014 development only; evaluation follows', flush=True)
        features = pd.read_csv(out/(name+'_features.csv.gz'), dtype={'market': str, 'date': str})
        result, test_panel, forecasts = evaluate(panel, distributions, features, **manifest['bootstrap'])
        scenario_results[name] = result
        # Do not expose the obsolete per-scenario multiplicity or gate labels.
        details[name] = {k: v for k, v in result.items() if k not in
                         ('family_a','family_b','channel_gate_passed','timing_gate_passed','interpretation')}
        for key, values in forecasts.items():
            path = out/(name+'_'+key+'_evaluated.npz')
            np.savez_compressed(path, market=test_panel.market.to_numpy(dtype='U20'),
                                date=test_panel.date.to_numpy(dtype='U10'), **values)
            artifacts.append(path)
        print('SCENARIO COMPLETE', name, len(test_panel), 'evaluation rows', flush=True)
    result = combine_results(scenario_results)
    result['completed_utc'] = now()
    locked_manifest(out/'screen_manifest.json'); locked_manifest(base/'preparation.json')
    save(out/'scenario_details.json', details); save(out/'results.json', result)
    design = manifest['design']
    write_report(out, result, details, design)
    artifacts += [out/'results.json', out/'scenario_details.json', out/'REPORT.md']
    locked_manifest(out/'completed_artifacts.json', {'created_utc': now(),
                    'screen_manifest_sha256': sha(out/'screen_manifest.json'),
                    'hashes': {str(p): sha(p) for p in artifacts}})
    print('SCREEN COMPLETE', out/'REPORT.md', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'run'])
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--base', type=Path, default=BASE_OUT)
    args = ap.parse_args()
    (prepare if args.action == 'prepare' else run)(args.out, args.base)


if __name__ == '__main__':
    main()
