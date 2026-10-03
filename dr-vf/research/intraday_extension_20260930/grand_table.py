"""One descriptive saved-forecast table; no fitting, tuning or new inference.

CLI: python -m research.intraday_extension_20260930.grand_table
       --source FINAL_INTRADAY_DIRECTORY --out NEW_DIRECTORY
The explicit --allow-incomplete switch is for marked interim verification only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY_SOURCE = ROOT / 'research/outputs/harmonized_leaderboard_20260930_v1'
SEEDS = ('8101', '8102', '8103', '8104', '8105')
DAILY = ('volpfn_20260930_full', 'volpfn_20260926', 'arch_lstm',
         'garch_lstm', 'ginn', 'hist250', 'ewma', 'garch_t', 'gjr_garch_t',
         'egarch_t', 'aparch_t', 'univol', 'chronos2', 'moirai2', 'tabpfn_ts')
PAIRS = (('volpfn_20260930_full', 'ttm_rk'),
         ('volpfn_20260926', 'ttm_rk'),
         ('volpfn_20260930_full', 'loghar'),
         ('volpfn_20260930_full', 'ms_har'),
         ('gjr_garch_t', 'ttm_rk'), ('gjr_garch_t', 'loghar'),
         ('ms_har', 'loghar'))
RK_BASES = ('har', 'loghar', 'ttm_rk', 'ms_har', 'log_rhar')
OVERLAPS = ('volpfn_20260930_full', 'volpfn_20260926', 'gjr_garch_t')
FRONTIER = ('gnnhar', 'qlike_har', 'transfer_xgboost', 'm2vn', 'deep_rgarch')
INTRADAY = tuple(a + '+' + b for a, b in PAIRS) + OVERLAPS + RK_BASES + FRONTIER
ROSTER = DAILY + tuple(n for n in INTRADAY if n not in DAILY)
UNAVAILABLE = ('m2vn', 'deep_rgarch')
DIAGNOSTICS = ('compact9', 'loghar_meanlog', 'ttm_sqrt_rk',
               'volpfn_20260930_noaux', 'volpfn_20260930_direct',
               'volpfn_20260930_historical')
METRICS = ('r2_qlike', 'rk_qlike')
assert len(DAILY) == 15 and len(INTRADAY) == 20 and len(set(ROSTER)) == 32
assert set(DAILY) & set(INTRADAY) == set(OVERLAPS)

LABELS = {
    'volpfn_20260930_full': 'VolPFN (Sep 30 full)',
    'volpfn_20260926': 'VolPFN (Sep 26)', 'arch_lstm': 'ARCH-LSTM',
    'garch_lstm': 'GARCH-LSTM', 'ginn': 'GINN',
    'hist250': 'Historical variance (250)', 'ewma': 'EWMA',
    'garch_t': 'GARCH-t', 'gjr_garch_t': 'GJR-GARCH-t',
    'egarch_t': 'EGARCH-t', 'aparch_t': 'APARCH-t', 'univol': 'UniVol',
    'chronos2': 'Chronos-2', 'moirai2': 'Moirai-2', 'tabpfn_ts': 'TabPFN-TS',
    'har': 'HAR', 'loghar': 'Log-HAR', 'ttm_rk': 'TTM-RK',
    'ms_har': 'MS-HAR', 'log_rhar': 'Log-Rough-HAR',
    'gnnhar': 'GNNHAR (fixed-fit architecture adaptation)',
    'qlike_har': 'QLIKE-HAR (rolling adaptation)',
    'transfer_xgboost': 'Transfer-XGBoost (causal adaptation)',
    'm2vn': 'M2VN (unavailable)', 'deep_rgarch': 'DeepRGARCH (unavailable)',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


class Pins:
    """Check registered hashes and seal exactly the files read by this builder."""
    def __init__(self):
        self.hashes = {}
        self.amendments = []

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Registered source hash changed: ' + str(path))
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError('Source changed during read: ' + str(path))
        self.hashes[str(path)] = digest
        return path

    def read_json(self, path):
        return json.loads(self.pin(path).read_text())

    def csv(self, path):
        return pd.read_csv(self.pin(path), dtype={'market': str, 'date': str, 'seed': str},
                           float_precision='round_trip')

    def receipt(self, directory, name):
        receipt = self.read_json(Path(directory) / name)
        files = receipt.get('files', receipt)
        for relative, digest in files.items():
            self.pin(Path(directory) / relative, digest)
        return len(files)

    def registered(self, hashes, directory):
        for name, digest in hashes.items():
            path = Path(name).resolve()
            if sha(path) == digest:
                self.pin(path, digest)
                continue
            # A verified byte-identical snapshot preserves later amended prose/code.
            snapshot = Path(directory) / 'frozen_sources' / path.name
            self.pin(snapshot, digest)
            self.amendments.append({'registered_source': str(path),
                                    'registered_sha256': digest,
                                    'verified_snapshot': str(snapshot.resolve()),
                                    'active_sha256': sha(path)})

    def recheck(self):
        for name, digest in self.hashes.items():
            if sha(name) != digest:
                raise ValueError('Source modified during analysis: ' + name)


def expected_seeds(model):
    if model not in ROSTER:
        raise ValueError('Undeclared main procedure: ' + model)
    if model.startswith('volpfn') or model in ('arch_lstm', 'garch_lstm', 'ginn',
                                              'gnnhar', 'transfer_xgboost', 'm2vn', 'deep_rgarch'):
        return SEEDS
    return ('none',)


def loss(y, h):
    y, h = np.asarray(y, dtype=float), np.asarray(h, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(h).all() or (y < 0).any() or (h <= 0).any():
        raise ValueError('Nonfinite/invalid issued targets or forecasts; no clipping/dropping')
    values = y / h + np.log(h)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite QLIKE')
    return values


def align(frame, mask, model, seeds):
    """Require exact keys per seed; never intersect successful forecasts."""
    f = frame.copy()
    needed = {'market', 'date', 'seed', 'r2', 'rk'}
    if not needed.issubset(f) or f[list(needed)].isna().any().any():
        raise ValueError('Missing saved identifiers/targets')
    f['seed'] = f.seed.astype(str)
    if set(f.seed) != set(seeds) or f.duplicated(['seed', 'market', 'date']).any():
        raise ValueError('Incorrect seed coverage/duplicate keys: ' + model)
    if 'model' in f and set(f.model) != {model}:
        raise ValueError('Incorrect row model identity: ' + model)
    keys = pd.MultiIndex.from_frame(mask[['market', 'date']])
    built = []
    for seed in seeds:
        rows = f.loc[f.seed == seed].set_index(['market', 'date'])
        if len(rows) != len(keys) or set(rows.index) != set(keys):
            raise ValueError('Missing/extra exact scoring keys: ' + model + '|' + seed)
        rows = rows.reindex(keys).reset_index()
        for field in ('r2', 'rk'):
            np.testing.assert_allclose(rows[field], mask[field], rtol=1e-12, atol=1e-12,
                                       err_msg='Target/unit mismatch: ' + model + '|' + field)
        if 'index' in rows:
            np.testing.assert_array_equal(rows['index'], mask['index'])
        built.append(rows)
    return pd.concat(built, ignore_index=True)


def native_rows(frame, mask, model):
    f = align(frame, mask, model, expected_seeds(model))
    moments = f[['mean', 'variance', 'second_moment']].to_numpy(float)
    if not np.isfinite(moments).all() or (moments[:, 1:] <= 0).any():
        raise ValueError('Invalid saved native moments: ' + model)
    np.testing.assert_allclose(f.second_moment, f.variance + f['mean'] ** 2,
                               rtol=1e-7, atol=1e-10, err_msg='M2 != variance + mean squared')
    for field in ('context_end', 'origin_date', 'latest_input_date'):
        if field in f and (f[field].isna().any() or (f[field] >= f.date).any()):
            raise ValueError('Noncausal saved context: ' + model)
    for proxy, forecast in (('r2', 'second_moment'), ('rk', 'variance')):
        values = loss(f[proxy], f[forecast])
        np.testing.assert_allclose(f[proxy + '_qlike'], values, rtol=2e-10, atol=2e-10,
                                   err_msg='Saved native row score differs: ' + model)
        f[proxy + '_forecast'] = f[forecast]
        f[proxy + '_qlike'] = values
    return f


def risk_rows(frame, mask, model):
    f = align(frame, mask, model, expected_seeds(model))
    for proxy in ('r2', 'rk'):
        values = loss(f[proxy], f[proxy + '_forecast'])
        np.testing.assert_allclose(f[proxy + '_qlike'], values, rtol=2e-10, atol=2e-10,
                                   err_msg='Saved issued-risk row score differs: ' + model)
        f[proxy + '_qlike'] = values
    if model in ('gnnhar', 'qlike_har', 'transfer_xgboost'):
        np.testing.assert_array_equal(f.r2_forecast, f.rk_forecast,
                                      err_msg='Native RK point changed between proxy scores')
    return f


def compare_overlap(first, second, model):
    """One shared procedure must have identical issued h, targets and row losses."""
    a = first.sort_values(['seed', 'market', 'date']).reset_index(drop=True)
    b = second.sort_values(['seed', 'market', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[['seed', 'market', 'date']], b[['seed', 'market', 'date']])
    for field in ('r2', 'rk', 'r2_forecast', 'rk_forecast', *METRICS):
        np.testing.assert_allclose(a[field], b[field], rtol=2e-12, atol=2e-12,
                                   err_msg='Overlap differs: ' + model + '|' + field)


def summarize(frame):
    per = frame.groupby(['seed', 'market'], sort=True)[list(METRICS)].mean().reset_index()
    seed = per.groupby('seed', sort=True)[list(METRICS)].mean().reset_index()
    result = {'seeds': len(seed), 'markets': frame.market.nunique(),
              'rows_per_seed': len(frame) // len(seed)}
    for metric in METRICS:
        result[metric] = float(seed[metric].mean())
        result[metric + '_seed_min'] = float(seed[metric].min())
        result[metric + '_seed_max'] = float(seed[metric].max())
    return result, per, seed


def verify_saved_summary(frame, saved, model):
    actual = summarize(frame)[0]
    for metric in METRICS:
        np.testing.assert_allclose(actual[metric], float(saved[metric]), rtol=2e-12, atol=2e-12,
                                   err_msg='Saved leaderboard summary differs: ' + model)
    if 'seeds' in saved and actual['seeds'] != int(saved['seeds']):
        raise ValueError('Saved leaderboard seed count differs: ' + model)


def verify_seed_market(frame, saved, model):
    actual = summarize(frame)[1].sort_values(['seed', 'market']).reset_index(drop=True)
    expected = saved.copy()
    expected['seed'] = expected.seed.astype(str)
    if 'model' in expected:
        expected = expected.loc[expected.model == model]
    expected = expected.sort_values(['seed', 'market']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[['seed', 'market']], expected[['seed', 'market']])
    for metric in METRICS:
        np.testing.assert_allclose(actual[metric], expected[metric], rtol=2e-12, atol=2e-12,
                                   err_msg='Saved seed/market summary differs: ' + model)


def verify_native_point(issued, mask, model, source_hashes, pins):
    patterns = {'gnnhar': 'gnnhar', 'qlike_har': 'qhar', 'transfer_xgboost': 'transfer_xgb'}
    candidates = [Path(p) for p in source_hashes
                  if Path(p).name == 'predictions.csv' and patterns[model] in str(p).lower()]
    if len(candidates) != 1:
        raise ValueError('One registered native point source required: ' + model)
    native = pins.csv(candidates[0])
    native['seed'] = native.seed.astype(str)
    if (set(native.seed) != set(expected_seeds(model))
            or native.duplicated(['seed', 'market', 'date']).any()):
        raise ValueError('Native point source seed/duplicate mismatch: ' + model)
    keys = pd.MultiIndex.from_frame(mask[['market', 'date']])
    selected = pd.MultiIndex.from_frame(native[['market', 'date']]).isin(keys)
    native = native.loc[selected].copy()
    for field in ('origin_date', 'latest_input_date'):
        if field not in native or native[field].isna().any() or (native[field] >= native.date).any():
            raise ValueError('Missing/noncausal native point input: ' + model)
    for field in ('latest_neighbor_input_date', 'latest_source_label_date', 'fit_end'):
        if field in native and (native[field].isna().any() or (native[field] >= native.date).any()):
            raise ValueError('Noncausal native point cutoff: ' + model)
    if model == 'gnnhar':
        if not (native.latest_neighbor_input_date < native.origin_date).all():
            raise ValueError('GNN neighbor input must precede own origin')
        np.testing.assert_allclose(native.rk_forecast, native.native_raw_forecast + 0.0001,
                                   rtol=2e-12, atol=2e-12)
    if 'fit_block_start' in native:
        if ('fit_end' not in native or not (native.fit_end < native.fit_block_start).all()
                or not (native.fit_block_start <= native.date).all()):
            raise ValueError('Native fit does not precede forecast block: ' + model)
    a = issued.sort_values(['seed', 'market', 'date']).reset_index(drop=True)
    b = native.sort_values(['seed', 'market', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[['seed', 'market', 'date']], b[['seed', 'market', 'date']])
    # CSV parser/serialization can differ by one floating-point unit; do not
    # confuse that with a changed issued forecast or allow material rescaling.
    np.testing.assert_allclose(a.rk_forecast, b.rk_forecast, rtol=2e-12, atol=2e-12,
                               err_msg='Native RK point differs from scored issued forecast')


def verify_mask(mask, other, pins, daily_provenance):
    pd.testing.assert_frame_equal(mask, other)
    if (len(mask) != 22026 or mask.market.nunique() != 28 or mask.date.nunique() != 827
            or mask.duplicated(['market', 'date']).any()
            or not mask.equals(mask.sort_values(['market', 'date']).reset_index(drop=True))):
        raise ValueError('Original exact common mask required')
    if (not np.isfinite(mask[['r2', 'rk']]).all().all()
            or (mask.r2 < 0).any() or (mask.rk <= 0).any()):
        raise ValueError('Invalid locked target population')
    bundle_path = ROOT / 'research/outputs/frontier_extension_20260926_v1/bundle.json'
    bundle = pins.read_json(bundle_path)
    original = pd.DataFrame([r for r in bundle['rows']
                             if r['phase'] == 'evaluation' and r['primary_eligible']])
    original = original.sort_values(['market', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(mask, original)
    raw_dir = ROOT / 'research/data/oxfordman_barde/calendar_full_v1'
    for market in sorted(mask.market.unique()):
        selected = mask.loc[mask.market == market]
        raw_path = raw_dir / (market + '.npz')
        old_path = ROOT / bundle['market_paths'][market]
        pins.pin(raw_path, daily_provenance['source_sha256'][str(raw_path.resolve())])
        pins.pin(old_path, bundle['inputs'][str(old_path.relative_to(ROOT))])
        with np.load(raw_path, allow_pickle=False) as raw, np.load(old_path, allow_pickle=False) as old:
            raw_dates, old_dates = raw['date'].astype('U10'), old['date'].astype('U10')
            i, j = np.searchsorted(raw_dates, selected.date), np.searchsorted(old_dates, selected.date)
            np.testing.assert_array_equal(raw_dates[i], selected.date)
            np.testing.assert_array_equal(old_dates[j], selected.date)
            np.testing.assert_array_equal(j, selected['index'])
            returns, rk = raw['returns'][i].astype(float) * 100., raw['rk_parzen'][i].astype(float) * 10000.
            np.testing.assert_allclose(selected.r2, returns ** 2, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(selected.rk, rk, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(old['r'][j], returns, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(old['rk'][j], rk, rtol=1e-12, atol=1e-12)


def metadata(model, daily_provenance):
    label = LABELS.get(model, ' + '.join(LABELS[n] for n in model.split('+')))
    if model in DAILY:
        info = 'Daily returns'
        functional = ('Saved issued second moment', 'Saved issued central variance')
        fit = ('Original frozen daily procedure; fitting/context/pretraining opportunities differ')
        adapter = daily_provenance['models'][model]['output_adapter']
    elif '+' in model:
        info = ('Daily returns + own lagged RK; pooled online blend calibration'
                if model.split('+')[0] in DAILY else 'Own lagged RK; pooled online blend calibration')
        functional = ('Proxy-specific issued risk level', 'Proxy-specific issued risk level')
        fit = 'Expanding prior evaluation labels; 60 union dates burn-in; refit every 20 union dates; fitted intercept'
        adapter = 'Separate proxy/seed log-linear risk blends; no jointly coherent return density'
    elif model in UNAVAILABLE:
        info, functional, fit, adapter = ('Unavailable', ('Unavailable', 'Unavailable'),
                                          'Not reproduced on exact population', 'No forecast or defeat assigned')
    elif model in RK_BASES:
        info = 'Own lagged RK'
        functional = ('Saved RK proxy risk level', 'Saved RK proxy risk level')
        fit = ('Frozen foundation/context adapter' if model == 'ttm_rk' else
               'Original rolling own-market fitting; earlier evaluation labels can enter refits')
        adapter = daily_provenance['models'][model]['output_adapter']
    else:
        info = 'Own and pooled neighbor lagged RK; return-based graph' if model == 'gnnhar' else (
            'Own and pooled source-market lagged RK' if model == 'transfer_xgboost' else 'Own lagged RK')
        functional = ('Native RK point used as r² risk proxy', 'Native RK point')
        fit = {'gnnhar': 'TRAIN through 2017; whole 2018 validation; fixed graph/weights during 2019–2022',
               'qlike_har': 'Rolling 1000 earlier own observed sessions; every-session refit',
               'transfer_xgboost': 'Saved causal transfer fits; source labels must precede forecast block'}[model]
        adapter = 'Architecture/method adaptation; RK point is not a return-density M2 estimator'
    pathology = {'har': 'Preserved OLS nonpositive-forecast floor events drive extreme losses; no tuning/removal',
                 'moirai2': 'Preserved squared-return point/floor events drive extreme losses; not universal family defeat',
                 'egarch_t': 'Preserved fit carries and extreme issued forecasts; inspect original coverage diagnostics',
                 'aparch_t': 'Preserved fit carries/extreme forecasts; inspect original diagnostics'}.get(model, '')
    return {'label': label, 'information_group': info,
            'r2_forecast_functional': functional[0], 'rk_forecast_functional': functional[1],
            'fitting_opportunity': fit, 'output_adapter': adapter, 'pathology_notes': pathology}


def rank_table(records):
    table = pd.DataFrame(records)
    for proxy in ('r2', 'rk'):
        metric = proxy + '_qlike'
        table[proxy + '_rank'] = table[metric].rank(method='min')
        if table.loc[table.status != 'complete', proxy + '_rank'].notna().any():
            raise ValueError('Unavailable method acquired rank')
    return table.sort_values(['rk_rank', 'r2_rank', 'model'], na_position='last').reset_index(drop=True)


def run(source, out, *, daily_source=DAILY_SOURCE, allow_incomplete=False):
    source, daily_source, out = Path(source).resolve(), Path(daily_source).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins = Pins()
    pins.pin(Path(__file__))
    pins.pin(Path(__file__).with_name('test_grand_table.py'))
    daily_receipts = pins.receipt(daily_source, 'artifact_receipt.json')
    intraday_receipts = pins.receipt(source, 'ARTIFACT_RECEIPT.json')
    daily_prov, intraday_prov = pins.read_json(daily_source / 'provenance.json'), pins.read_json(source / 'provenance.json')
    pins.registered(daily_prov['source_sha256'], daily_source)
    pins.registered(intraday_prov['source_hashes'], source)
    design = pins.read_json(source / 'REGISTERED_DESIGN.json')
    if set(design['roster']) != set(INTRADAY) or len(design['roster']) != len(INTRADAY):
        raise ValueError('Intraday registered roster differs')
    mask = pins.csv(daily_source / 'common_mask.csv')
    verify_mask(mask, pins.csv(source / 'common_mask.csv'), pins, daily_prov)
    daily_table, intraday_table = pins.csv(daily_source / 'daily_leaderboard.csv'), pins.csv(source / 'leaderboard.csv')
    if len(daily_table) != len(DAILY) or set(daily_table.model) != set(DAILY):
        raise ValueError('Fixed 15 daily primary procedures required')
    if len(intraday_table) != len(INTRADAY) or set(intraday_table.model) != set(INTRADAY):
        raise ValueError('Fixed 20 intraday procedures required')
    complete = set(intraday_table.loc[intraday_table.status == 'complete', 'model'])
    if not allow_incomplete and complete != set(INTRADAY) - set(UNAVAILABLE):
        raise ValueError('Full generation requires 18 complete intraday procedures, including transfer-XGBoost')
    if not set(OVERLAPS + RK_BASES).issubset(complete):
        raise ValueError('All preserved base forecasts required')
    native, frames = {}, {}
    daily_per = pins.csv(daily_source / 'per_seed_market.csv')
    for model in DAILY + RK_BASES:
        native[model] = native_rows(pins.csv(daily_source / 'aligned_forecasts' / (model + '.csv.gz')), mask, model)
        if model in DAILY:
            verify_saved_summary(native[model], daily_table.set_index('model').loc[model], model)
            verify_seed_market(native[model], daily_per, model)
            frames[model] = native[model]
    overlap_checks = []
    for model in INTRADAY:
        row = intraday_table.set_index('model').loc[model]
        if model not in complete:
            if row[list(METRICS)].notna().any():
                raise ValueError('Unavailable slot has invented losses')
            continue
        issued = risk_rows(pins.csv(source / 'row_forecasts' / (model + '.csv.gz')), mask, model)
        verify_saved_summary(issued, row, model)
        verify_seed_market(issued, pins.csv(source / (model + '_per_seed_market.csv')), model)
        if model in ('gnnhar', 'qlike_har', 'transfer_xgboost'):
            verify_native_point(issued, mask, model, intraday_prov['source_hashes'], pins)
        if model in native:
            compare_overlap(native[model], issued, model)
            if model in OVERLAPS:
                overlap_checks.append(model)
            else:
                frames[model] = native[model]
        else:
            frames[model] = issued
    records, per_frames, seed_frames, market_frames = [], [], [], []
    (out / 'row_forecasts').mkdir()
    for model in ROSTER:
        spec = metadata(model, daily_prov)
        base = {'model': model, **spec,
                'daily_archive': str(daily_source) if model in DAILY else '',
                'intraday_archive': str(source) if model in INTRADAY else ''}
        if model not in frames:
            records.append({**base, 'status': 'unavailable/not reproduced',
                            'seeds': 0, 'markets': 0, 'rows_per_seed': 0,
                            **{m: None for m in METRICS}})
            continue
        f = frames[model]
        values, per, seed = summarize(f)
        per_frames.append(per.assign(model=model))
        seed_frames.append(seed.assign(model=model))
        market_frames.append(per.groupby('market')[list(METRICS)].mean().reset_index().assign(model=model))
        columns = ['market', 'date', 'seed', 'r2', 'rk', 'r2_forecast', 'rk_forecast', *METRICS]
        columns += [c for c in ('mean', 'variance', 'second_moment', 'substituted', 'floor_applied',
                                'forecast_failed', 'native_variance_cap_event', 'context_end') if c in f]
        saved = f[columns].copy().assign(model=model)
        saved.to_csv(out / 'row_forecasts' / (model + '.csv.gz'), index=False,
                     compression={'method': 'gzip', 'mtime': 0})
        records.append({**base, 'status': 'complete', **values})
    table = rank_table(records)
    table.to_csv(out / 'leaderboard.csv', index=False)
    pd.concat(per_frames, ignore_index=True).to_csv(out / 'per_seed_market.csv', index=False)
    pd.concat(seed_frames, ignore_index=True).to_csv(out / 'per_seed.csv', index=False)
    pd.concat(market_frames, ignore_index=True).to_csv(out / 'per_market.csv', index=False)
    mask.to_csv(out / 'common_mask.csv', index=False)
    diagnostic_paths = ('diagnostics_unranked.csv', 'coverage.csv', 'REPORT.md', 'statistics.json',
                        'original_calibrated_leaderboard.csv', 'interpretation_checks.json')
    diagnostics = {n: str(pins.pin(daily_source / n)) for n in diagnostic_paths}
    write_json(out / 'REGISTERED_DESIGN.json', {
        'status': 'interim_test_only' if allow_incomplete else 'complete_descriptive',
        'daily_primary_roster': DAILY, 'intraday_roster': INTRADAY, 'union_roster': ROSTER,
        'exact_overlaps': OVERLAPS, 'unranked_diagnostic_models': DIAGNOSTICS,
        'expected_complete_procedures': 30, 'unavailable_slots': UNAVAILABLE,
        'seed_ids_by_model': {m: expected_seeds(m) for m in ROSTER},
        'loss': 'y/h + log(h)', 'units': 'percent-return-squared',
        'aggregation': 'Time within market/seed, then equal markets, then equal retained seeds',
        'statistics': 'No new global inference family; all ranks descriptive',
        'forecast_ensembling': False, 'fitting_or_tuning': False,
        'diagnostic_and_original_inference_pointers': diagnostics,
        'original_intraday_inference_pointer': str(source),
        'proxy_functionals': 'Daily saved M2/central variance; native RK points explicitly risk proxies; blend outputs proxy specific'})
    write_json(out / 'verification.json', {
        'status': 'passed', 'exact_common_keys': len(mask), 'markets': mask.market.nunique(),
        'union_dates': mask.date.nunique(), 'target_based_RK_exclusions_retained': 17,
        'all_targets_checked_against_two_authoritative_raw_archives': True,
        'all_rows_losses_recomputed': True, 'native_M2_and_central_variance_preserved': True,
        'all_declared_seeds_required': True, 'overlap_row_equality': overlap_checks,
        'overlaps_counted_once': True, 'equal_market_seed_weighting': True,
        'registered_slots': len(ROSTER), 'completed_procedures': len(frames),
        'unavailable_or_pending_procedures': [n for n in ROSTER if n not in frames],
        'daily_artifact_entries_verified': daily_receipts,
        'intraday_artifact_entries_verified': intraday_receipts,
        'historical_outputs_overwritten': False,
        'no_global_significance_flags_or_tests': True})
    lines = ['# Combined descriptive saved-forecast table', '',
             'Exact 22,026 keys; 28 markets; 827 union dates; 2019-01-01–2022-02-25.',
             'Targets/forecasts use percent-return-squared. Score y/h+log(h); retain zero squared returns.',
             'Average dates within market/seed, then markets equally, then retained seeds equally.',
             'The three identical daily/intraday overlaps are verified row by row and counted once.', '',
             '| Procedure | Information | r² QLIKE | RK QLIKE | r² rank | RK rank | Seeds |',
             '|---|---|---:|---:|---:|---:|---:|']
    for _, row in table.loc[table.status == 'complete'].iterrows():
        lines.append(f'| {row.label} | {row.information_group} | {row.r2_qlike:.6f} | {row.rk_qlike:.6f} | {int(row.r2_rank)} | {int(row.rk_rank)} | {row.seeds} |')
    lines += ['', 'Unavailable/pending: ' + ', '.join(n for n in ROSTER if n not in frames) + '.', '',
              'Ranks describe issued pipelines with different information. Daily-return models, own-RK models, pooled neighbor/source-RK models and online blends do not have equal information or fitting opportunities.',
              'Daily sources retain saved M2 for r² and saved central variance for RK, including their original documented proxy adapters. Native RK methods use the same issued RK point for both scores; the r² score does not establish a return-density M2 estimate. Blends issue separate risk forecasts for the two proxies and are not jointly coherent densities.',
              'GNNHAR fits through 2017, selects on whole 2018 validation and freezes graph/weights in evaluation. Other models have original rolling fits, foundation pretraining or different training windows. Blends gain online fitted-intercept calibration from prior evaluation labels. Causal transfer source-label cutoffs differ from fixed GNN fitting. Original adapters, floors and carries remain; no model is retrained here.',
              'HAR floor events, Moirai point/floor pathologies and EGARCH extreme forecasts/carries materially affect losses. These ranks cannot establish paper-family defeat or isolate architecture quality. See the preserved coverage/diagnostic sources.',
              'No new global inference family is created, and original daily/intraday significance results are not transferred to this enlarged roster. The exposed panel is not a fresh holdout. No universal superiority or equal-information fairness claim is supported.',
              'Daily publication dates establish source-date conventions; exchange/vendor intraday publication timestamps and foundation-pretraining overlap are not certified.',
              'Compact9, mean-log/sqrt-RK variants and VolPFN noaux/direct/historical arms remain outside main ranks. Their preserved diagnostic pointers are recorded in REGISTERED_DESIGN.json.',
              'An --allow-incomplete run is a marked interim verification artifact, not the requested full final table.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (out / 'frozen_sources').mkdir()
    for name in ('grand_table.py', 'test_grand_table.py'):
        shutil.copyfile(Path(__file__).with_name(name), out / 'frozen_sources' / name)
    pins.recheck()
    write_json(out / 'provenance.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'daily_source': str(daily_source),
        'source_hashes': pins.hashes, 'registered_source_amendments': pins.amendments,
        'originals_preserved': True, 'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__})
    write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
               for p in sorted(out.rglob('*')) if p.is_file()})
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Final intraday comparison directory')
    parser.add_argument('--out', required=True, help='Exclusive new output directory')
    parser.add_argument('--daily-source', default=str(DAILY_SOURCE))
    parser.add_argument('--allow-incomplete', action='store_true', help='Mark interim verification only')
    args = parser.parse_args()
    print(run(args.source, args.out, daily_source=args.daily_source,
              allow_incomplete=args.allow_incomplete).to_string(index=False))
