"""Independent per-market gamma-XGBoost RK helper, frozen before 2019.

84 predictors: own HAR1/5/22 and individual lexical foreign HAR1/5/22 lags.
Own-target TRAIN labels only; no graph, pooled labels, date/category predictor,
VolPFN training or evaluation refits. Missing early foreign lag cells stay NaN.
Full fitting is explicit via --out; importing/preparing never fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.intraday_extension_20260930.transfer_xgb import load_panel, xgb_class

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (8101, 8102, 8103, 8104, 8105)
TRAIN_END = '2017-12-31'
VALIDATION_END = '2018-12-31'
MIN_FOREIGN = 10
TRAIN_GATE_TOLERANCE = 1e-10
FEATURE_LAGS = ('har1', 'har5', 'har22')
PILOT_MARKETS = ('AEX', 'KSE')
PARAMETERS = dict(objective='reg:gamma', n_estimators=40, max_depth=3,
                  learning_rate=0.1, subsample=0.75, colsample_bytree=1.,
                  reg_lambda=1., min_child_weight=1., gamma=0., base_score=1.,
                  n_jobs=1, tree_method='exact', seed_per_iteration=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def positive(values, name):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Nonpositive/nonfinite ' + name + '; no clipping or repair')
    return values


def qlike(target, forecast):
    target, forecast = positive(target, 'RK target'), positive(forecast, 'forecast')
    result = target / forecast + np.log(forecast)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite QLIKE')
    return result


def session_har(item):
    """Snapshots end at each observed session; carry inputs forward only.

    Track actual positive RK owners, rather than calling a carried observation
    a newly available source value. Targets always remain the original item.rk.
    """
    risk = np.asarray(item['rk'], dtype=np.float64)
    valid = np.isfinite(risk) & (risk > 0)
    owner = np.maximum.accumulate(np.where(valid, np.arange(len(risk)), -1))
    filled = np.full(len(risk), np.nan)
    found = owner >= 0
    filled[found] = risk[owner[found]]
    har = np.full((len(risk), 3), np.nan)
    imputations = np.zeros(len(risk), dtype=np.int64)
    har[:, 0] = filled
    prefix = np.concatenate(([0], np.cumsum(~valid)))
    for column, width in ((1, 5), (2, 22)):
        if len(risk) >= width:
            har[width - 1:, column] = np.lib.stride_tricks.sliding_window_view(filled, width).mean(1)
    # Count carried RK entries in the largest available lag window; overlapping
    # HAR1/5/22 windows are not counted three times.
    for i in range(len(risk)):
        width = 22 if np.isfinite(har[i, 2]) else 5 if np.isfinite(har[i, 1]) else 1
        imputations[i] = prefix[i + 1] - prefix[max(0, i + 1 - width)]
    return dict(har=har, owner=owner, imputations=imputations, valid=valid)


def feature_names(target_market, markets):
    return tuple('own_' + lag for lag in FEATURE_LAGS) + tuple(
        'foreign_' + market + '_' + lag
        for market in sorted(markets) if market != target_market for lag in FEATURE_LAGS)


def training_scales(panel, train_end=TRAIN_END):
    scales, receipts = {}, {}
    for market in sorted(panel):
        item = panel[market]
        available = ((item['date'] <= train_end) & np.isfinite(item['rk']) & (item['rk'] > 0))
        values = np.asarray(item['rk'][available], dtype=np.float64)
        if not len(values):
            raise ValueError('No positive observed TRAIN risk for normalization: ' + market)
        scale = float(values.mean())
        positive([scale], 'TRAIN mean')
        scales[market] = scale
        receipts[market] = dict(mean_rk_percent_squared=scale, positive_observed_rows=len(values),
                                first_date=str(item['date'][available][0]),
                                last_date=str(item['date'][available][-1]),
                                fit_cutoff=train_end, target_imputations=0,
                                population='All positive observed TRAIN labels, including early feature-ineligible sessions')
    return scales, receipts


def feature_rows(panel, snapshots, scales, target_market, indices, min_foreign=MIN_FOREIGN):
    """Own previous session and foreign sessions strictly before own origin.

    Individual foreign lag columns retain lexical source identity. Missing cells
    stay NaN for the native tree missing-value route. Eligibility counts complete
    three-lag source groups; there is no complete-date intersection or padding.
    """
    item = panel[target_market]
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < 1) or np.any(indices >= len(item['date'])):
        raise ValueError('Target must have an observed predecessor')
    origins = item['date'][indices - 1]
    own = snapshots[target_market]
    own_x = own['har'][indices - 1] / scales[target_market]
    own_owner = own['owner'][indices - 1]
    latest_own = np.full(len(indices), '', dtype='U10')
    found_own = own_owner >= 0
    latest_own[found_own] = item['date'][own_owner[found_own]]
    own_imputations = own['imputations'][indices - 1]
    counts = np.zeros(len(indices), dtype=np.int64)
    partial_counts = np.zeros(len(indices), dtype=np.int64)
    foreign_imputations = np.zeros(len(indices), dtype=np.int64)
    latest_foreign = np.full(len(indices), '', dtype='U10')
    columns = [own_x]
    for market in sorted(panel):
        if market == target_market:
            continue
        other, snapshot = panel[market], snapshots[market]
        positions = np.searchsorted(other['date'], origins, side='left') - 1
        selected = np.maximum(positions, 0)
        har = snapshot['har'][selected]
        owner = snapshot['owner'][selected]
        source_available = (positions >= 0) & (owner >= 0)
        values = har / scales[market]
        values[~source_available] = np.nan
        complete = source_available & np.isfinite(values).all(1)
        contributing = source_available & np.isfinite(values).any(1)
        counts += complete
        partial_counts += contributing & ~complete
        columns.append(values)
        foreign_imputations[contributing] += snapshot['imputations'][selected[contributing]]
        actual_dates = np.full(len(indices), '', dtype='U10')
        actual_dates[contributing] = other['date'][owner[contributing]]
        latest_foreign = np.where(actual_dates > latest_foreign, actual_dates, latest_foreign)
        if np.any(actual_dates[contributing] >= origins[contributing]):
            raise AssertionError('Foreign input is not strictly earlier than own origin')
    x = np.column_stack(columns)
    if np.isinf(x).any():
        raise ValueError('Infinite normalized feature')
    eligible = np.isfinite(own_x).all(1) & (counts >= min_foreign)
    metadata = pd.DataFrame(dict(market=target_market, date=item['date'][indices], index=indices,
                                 origin_date=origins, latest_input_date=latest_own,
                                 latest_own_input_date=latest_own,
                                 latest_foreign_input_date=latest_foreign,
                                 foreign_count=counts, foreign_partial_count=partial_counts,
                                 own_input_imputations=own_imputations,
                                 foreign_input_imputations=foreign_imputations,
                                 market_scale=scales[target_market]))
    if np.any(latest_own[eligible] > origins[eligible]):
        raise AssertionError('Own actual RK owner crosses its origin')
    return x, eligible, metadata


@dataclass
class Prepared:
    x: np.ndarray
    y: np.ndarray
    records: pd.DataFrame
    mask: pd.DataFrame
    scales: dict
    scale_receipts: dict
    feature_names: dict
    preparation: dict
    hashes: dict


def prepare_panel(panel, mask, *, train_end=TRAIN_END, validation_end=VALIDATION_END,
                  min_foreign=MIN_FOREIGN, hashes=None):
    """Pure data preparation, also used by small synthetic causality tests."""
    if not train_end < validation_end < '2019-01-01':
        raise ValueError('Frozen TRAIN/validation must be chronological and pre2019')
    if mask.duplicated(['market', 'date']).any():
        raise ValueError('Duplicate target mask')
    scales, scale_receipts = training_scales(panel, train_end)
    snapshots = {m: session_har(panel[m]) for m in panel}
    parts, labels, records, receipts = [], [], [], {}
    for market in sorted(panel):
        item = panel[market]
        historical = np.flatnonzero((item['date'] <= validation_end) & snapshots[market]['valid'])
        historical = historical[historical >= 1]
        wanted = mask.loc[mask.market == market].sort_values('date')
        evaluation = np.searchsorted(item['date'], wanted.date)
        if np.any(evaluation >= len(item['date'])) or not np.array_equal(item['date'][evaluation], wanted.date):
            raise ValueError('Missing exact evaluation key: ' + market)
        if 'index' in wanted:
            np.testing.assert_array_equal(evaluation, wanted['index'])
        positive(item['rk'][evaluation], 'original evaluation RK')
        if 'rk' in wanted:
            np.testing.assert_allclose(item['rk'][evaluation], wanted.rk, rtol=1e-12, atol=1e-12)
        indices = np.concatenate((historical, evaluation))
        x, eligible, meta = feature_rows(panel, snapshots, scales, market, indices, min_foreign)
        split = np.where(np.arange(len(indices)) >= len(historical), 'evaluation',
                         np.where(item['date'][indices] <= train_end, 'train', 'validation'))
        if not eligible[split == 'evaluation'].all():
            raise ValueError('Incomplete evaluation inputs: ' + market)
        meta['split'] = split
        meta['normalized_target'] = item['rk'][indices] / scales[market]
        meta['observed_rk'] = item['rk'][indices]
        meta = meta.loc[eligible].reset_index(drop=True)
        parts.append(x[eligible]); labels.append(meta.normalized_target.to_numpy(float)); records.append(meta)
        receipts[market] = dict(historical_positive_candidates=len(historical),
                                early_unavailable_feature_candidates=int((~eligible & (split != 'evaluation')).sum()),
                                invalid_historical_labels=int(((item['date'] <= validation_end) & ~snapshots[market]['valid']).sum()),
                                target_imputations=0, evaluation_rows=len(evaluation))
    frame = pd.concat(records, ignore_index=True)
    evaluation = frame.loc[frame.split == 'evaluation', ['market', 'date']].reset_index(drop=True)
    pd.testing.assert_frame_equal(evaluation, mask[['market', 'date']].sort_values(['market', 'date']).reset_index(drop=True), check_dtype=False)
    train = frame.split == 'train'
    if set(frame.loc[train, 'market']) != set(panel) or not (frame.split == 'validation').any():
        raise ValueError('Every market needs eligible TRAIN rows and whole-panel validation')
    return Prepared(np.ascontiguousarray(np.concatenate(parts), dtype=np.float32),
                    np.concatenate(labels), frame, mask.copy(), scales, scale_receipts,
                    {m: feature_names(m, panel) for m in panel}, receipts, dict(hashes or {}))


def prepare_dataset(root=ROOT):
    started = time.perf_counter()
    panel, mask, hashes = load_panel(root)
    if len(panel) != 28 or len(mask) != 22026:
        raise ValueError('Exact authoritative 28-market population required')
    data = prepare_panel(panel, mask, hashes=hashes)
    evaluation = data.records.split == 'evaluation'
    if not (data.records.loc[evaluation, 'foreign_count'] == 27).all():
        raise ValueError('Expected all 27 foreign markets on every primary evaluation key')
    data.preparation['total'] = dict(seconds=time.perf_counter() - started,
                                     split_counts=data.records.split.value_counts().to_dict(),
                                     minimum_train_foreign_count=int(data.records.loc[data.records.split == 'train', 'foreign_count'].min()),
                                     evaluation_foreign_count=27, target_imputations=0)
    return data


def model(seed):
    if seed not in SEEDS:
        raise ValueError('Undeclared seed')
    fitted = xgb_class()(**PARAMETERS, random_state=int(seed))
    # Compatibility metadata only: pinned XGBoost1.7.1 JSON serialization reads
    # this sklearn mixin attribute, removed in the installed sklearn version.
    # It does not change the objective, predictors or any tree hyperparameter.
    fitted._estimator_type = 'regressor'
    return fitted


def train_quality(data, market, forecast):
    train = ((data.records.split == 'train') & (data.records.market == market)).to_numpy()
    prediction = positive(forecast, 'TRAIN normalized prediction')
    if prediction.shape != data.y[train].shape:
        raise ValueError('Wrong TRAIN forecast shape')
    target = data.y[train]
    constant = float(target.mean())
    normalized_loss = qlike(target, prediction)
    constant_loss = qlike(target, np.full_like(target, constant))
    fitted, baseline = float(normalized_loss.mean()), float(constant_loss.mean())
    summary = data.records.loc[train, ['market', 'observed_rk', 'market_scale']].copy()
    summary['normalized_qlike'], summary['normalized_constant_qlike'] = normalized_loss, constant_loss
    summary['rk_qlike'] = qlike(summary.observed_rk, prediction * summary.market_scale)
    summary['constant_rk_qlike'] = qlike(summary.observed_rk, constant * summary.market_scale)
    return dict(passed=bool(fitted <= baseline + TRAIN_GATE_TOLERANCE),
                market=market, rows=len(target), eligible_train_constant_normalized=constant,
                normalized_train_qlike=fitted, normalized_constant_qlike=baseline,
                normalized_fixed1_qlike=float(qlike(target, np.ones_like(target)).mean()),
                rk_train_qlike=float(summary.rk_qlike.mean()),
                rk_constant_qlike=float(summary.constant_rk_qlike.mean()),
                tolerance=TRAIN_GATE_TOLERANCE, finite_positive_outputs=True,
                aggregation='Uniform own-market eligible TRAIN labels',
                constant_fitted_on='Eligible own-market TRAIN normalized targets only')


def validation_diagnostics(data, market, forecast):
    selected = ((data.records.split == 'validation') & (data.records.market == market)).to_numpy()
    train = ((data.records.split == 'train') & (data.records.market == market)).to_numpy()
    constant = float(data.y[train].mean())
    rows = data.records.loc[selected, ['market', 'observed_rk', 'market_scale']].copy()
    target = data.y[selected]
    prediction = positive(forecast, 'validation normalized prediction')
    rows['normalized_qlike'] = qlike(target, prediction)
    rows['normalized_constant_qlike'] = qlike(target, np.full_like(target, constant))
    rows['normalized_fixed1_qlike'] = qlike(target, np.ones_like(target))
    rows['rk_qlike'] = qlike(rows.observed_rk, prediction * rows.market_scale)
    rows['constant_rk_qlike'] = qlike(rows.observed_rk, constant * rows.market_scale)
    fields = ['normalized_qlike', 'normalized_constant_qlike', 'normalized_fixed1_qlike', 'rk_qlike', 'constant_rk_qlike']
    return dict(selection_or_tuning=False, market=market, rows=len(rows),
                train_fitted_constant_normalized=constant, means=rows[fields].mean().to_dict())


def predictor_matrix(data, selected, market):
    return pd.DataFrame(data.x[selected], columns=data.feature_names[market])


def feature_usage(fitted, names):
    split = fitted.get_booster().get_score(importance_type='weight')
    importance = dict(zip(names, map(float, fitted.feature_importances_)))
    own = [n for n in names if n.startswith('own_')]
    foreign = [n for n in names if n.startswith('foreign_')]
    return dict(diagnostic_only=True, selection_or_gate=False,
                own_split_count=float(sum(split.get(n, 0.) for n in own)),
                foreign_split_count=float(sum(split.get(n, 0.) for n in foreign)),
                own_normalized_gain_importance=float(sum(importance[n] for n in own)),
                foreign_normalized_gain_importance=float(sum(importance[n] for n in foreign)),
                split_counts_by_feature={n: float(split.get(n, 0.)) for n in names},
                normalized_gain_importance_by_feature=importance)


def dependency_hashes():
    xgb_class()
    import xgboost
    import scipy
    import sklearn
    package = Path(xgboost.__file__).resolve().parent
    hashes = {str(p): sha(p) for p in sorted(package.rglob('*.py'))}
    for p in sorted((package / 'lib').glob('*xgboost*')):
        if p.is_file():
            hashes[str(p)] = sha(p)
    metadata = package.parent / 'xgboost-1.7.1.dist-info' / 'METADATA'
    if not metadata.exists():
        raise ValueError('Pinned XGBoost metadata missing')
    hashes[str(metadata)] = sha(metadata)
    # Seal numerical dependency identities and native kernels used by the
    # existing local runtime; this never installs or modifies a package.
    for name, module in (('numpy', np), ('pandas', pd), ('scipy', scipy), ('scikit-learn', sklearn)):
        package_root = Path(module.__file__).resolve().parent
        hashes[str(Path(module.__file__).resolve())] = sha(module.__file__)
        distribution = importlib.metadata.distribution(name)
        for entry in distribution.files or []:
            if entry.name in ('METADATA', 'RECORD') and entry.parent.name.endswith('.dist-info'):
                path = Path(distribution.locate_file(entry)).resolve()
                hashes[str(path)] = sha(path)
        for suffix in ('*.so', '*.dylib'):
            for path in sorted(package_root.rglob(suffix)):
                if path.is_file():
                    hashes[str(path.resolve())] = sha(path)
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__,
                xgboost=xgboost.__version__, hashes=hashes)


def validate_predictions(frame, mask):
    dates = ('date', 'origin_date', 'latest_input_date',
             'latest_own_input_date', 'latest_foreign_input_date')
    required = {'market', 'seed', 'rk_forecast', 'market_scale',
                'normalized_prediction', 'foreign_count', *dates}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError('Required output identifiers/forecasts/availability metadata missing')
    for field in dates:
        if not frame[field].astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            raise ValueError('Noncanonical output date: ' + field)
        try:
            pd.to_datetime(frame[field], format='%Y-%m-%d', errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid actual output date: ' + field) from exc
    if set(frame.seed) != set(SEEDS) or frame.duplicated(['seed', 'market', 'date']).any():
        raise ValueError('Exact five seeds and unique keys required')
    keys = set(zip(mask.market, mask.date))
    for seed in SEEDS:
        rows = frame.loc[frame.seed == seed]
        if len(rows) != len(mask) or set(zip(rows.market, rows.date)) != keys:
            raise ValueError('Missing/extra exact primary keys')
    positive(frame.rk_forecast, 'issued RK forecast')
    positive(frame.market_scale, 'frozen market normalization')
    positive(frame.normalized_prediction, 'normalized issued forecast')
    np.testing.assert_allclose(frame.rk_forecast, frame.normalized_prediction * frame.market_scale,
                               rtol=1e-12, atol=1e-12, err_msg='Forecast scaling must occur exactly once')
    if (not (frame.latest_input_date == frame.latest_own_input_date).all()
            or not (frame.latest_own_input_date <= frame.origin_date).all()
            or not (frame.origin_date < frame.date).all()
            or not (frame.latest_foreign_input_date < frame.origin_date).all()
            or not (frame.foreign_count == 27).all()):
        raise ValueError('Invalid issued availability metadata')


def run(out, root=ROOT, *, pilot=False):
    """Root invokes this only after protocol/source review and freeze."""
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = Path(__file__).with_name('PROTOCOL.md')
    if not protocol.exists():
        raise ValueError('Frozen protocol required before full fitting')
    started = time.perf_counter()
    data = prepare_dataset(root)
    loader = root / 'experiments/intraday_extension_20260930/transfer_xgb.py'
    sources = {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('test_helper.py'), protocol, loader)}
    dependencies = dependency_hashes()
    markets = PILOT_MARKETS if pilot else tuple(sorted(data.scales))
    seeds = (8101,) if pilot else SEEDS
    design = dict(model='Independent own-target cross-market gamma-XGBoost RK helper',
                  paper_replication=False, seed_ids=SEEDS, parameters=PARAMETERS,
                  runtime_compatibility='Instance _estimator_type=regressor restores pinned XGBoost1.7.1 JSON save/load metadata; no method/hyperparameter change',
                  actual_run_markets=markets, actual_run_seeds=seeds,
                  timing_pilot_only=pilot, pilot_selection_or_budget_tuning=False,
                  train_end=TRAIN_END, validation_end=VALIDATION_END,
                  validation_selection=False, evaluation_refits=False,
                  feature_names_by_target_market=data.feature_names,
                  predictors_per_market=84, foreign_order='Lexical market identifiers, then HAR1/5/22',
                  minimum_fully_available_foreign_markets=MIN_FOREIGN,
                  missing_foreign_lag_cells='Native XGBoost NaN routing; no backward fill or invented labels',
                  normalization='Each market positive observed TRAIN mean, including feature-ineligible early sessions',
                  sample_weights='Uniform within own-market TRAIN; no pooled target labels',
                  checkpoint_acceptance='Per market/seed: finite positive full eligible TRAIN predictions and QLIKE <= eligible TRAIN empirical-mean constant',
                  fixed_constant1='Logged diagnostic only; not the optimal eligible-TRAIN baseline',
                  train_gate_tolerance=TRAIN_GATE_TOLERANCE, prediction_scaling='Normalized gamma prediction * own frozen TRAIN mean once',
                  output_repair_or_floor=False, targets_imputed=False,
                  source_hashes=sources, input_hashes=data.hashes, dependencies=dependencies)
    write_json(out / 'DESIGN.json', design)
    write_json(out / 'NORMALIZATION.json', data.scale_receipts)
    write_json(out / 'PREPARATION.json', data.preparation)
    data.records.loc[data.records.split != 'evaluation'].to_csv(out / 'historical_metadata.csv.gz', index=False, compression='gzip')
    (out / 'frozen_sources').mkdir()
    for path in (Path(__file__), Path(__file__).with_name('test_helper.py'), protocol):
        shutil.copyfile(path, out / 'frozen_sources' / path.name)
    diagnostics, predictions, failed, fitted_models = [], [], [], {}
    for market in markets:
        train = ((data.records.split == 'train') & (data.records.market == market)).to_numpy()
        validation = ((data.records.split == 'validation') & (data.records.market == market)).to_numpy()
        folder = out / 'markets' / market
        folder.mkdir(parents=True)
        for seed in seeds:
            fitted = model(seed)
            seed_started = time.perf_counter()
            x_train, x_validation = predictor_matrix(data, train, market), predictor_matrix(data, validation, market)
            checkpoint = folder / ('seed_' + str(seed) + '.json')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fitted.fit(x_train, data.y[train])
                actual_parameters = {name: fitted.get_params()[name]
                                     for name in (*PARAMETERS, 'random_state')}
                pre_save_booster_config = json.loads(fitted.get_booster().save_config())
                fitted.save_model(checkpoint)
                quality = train_quality(data, market, fitted.predict(x_train))
                val = validation_diagnostics(data, market, fitted.predict(x_validation))
            entry = dict(market=market, seed=seed, train_quality=quality, validation=val,
                         actual_parameters=actual_parameters,
                         pre_save_booster_config=pre_save_booster_config,
                         feature_usage=feature_usage(fitted, data.feature_names[market]),
                         warning_count=len(caught), warnings=sorted(set(str(w.message) for w in caught)),
                         checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                         seconds=time.perf_counter() - seed_started)
            diagnostics.append(entry)
            write_json(folder / ('seed_' + str(seed) + '_diagnostics.json'), entry)
            fitted_models[market, seed] = fitted
            if not quality['passed']:
                failed.append(dict(market=market, seed=seed))
    receipt = dict(status='failed_TRAIN_gate' if failed else 'pilot_complete' if pilot else 'complete', design=design,
                   preparation=data.preparation['total'], trials=diagnostics, failed_market_seeds=failed,
                   seed_substitution=False, evaluation_loss_computed=False,
                   new_volpfn_training=False, wall_seconds=time.perf_counter() - started)
    if failed:
        write_json(out / 'RECEIPT.json', receipt)
        write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
                   for p in sorted(out.rglob('*')) if p.is_file()})
        raise RuntimeError('Required market/seed fits failed TRAIN gates; no substitutions or evaluation forecasts: ' + str(failed))
    if not pilot:
        for market in markets:
            evaluation = ((data.records.split == 'evaluation') & (data.records.market == market)).to_numpy()
            for seed in seeds:
                fitted = fitted_models[market, seed]
                forecast = positive(fitted.predict(predictor_matrix(data, evaluation, market)), 'evaluation normalized prediction')
                rows = data.records.loc[evaluation].drop(columns=['normalized_target', 'observed_rk']).copy()
                rows['seed'] = seed
                rows['normalized_prediction'] = forecast
                rows['rk_forecast'] = positive(forecast * rows.market_scale, 'issued percent-squared RK forecast')
                rows['train_end'], rows['validation_end'] = TRAIN_END, VALIDATION_END
                predictions.append(rows)
        frame = pd.concat(predictions, ignore_index=True)
        validate_predictions(frame, data.mask)
        frame.to_csv(out / 'predictions.csv', index=False)
        data.mask.to_csv(out / 'common_mask.csv', index=False)
        receipt.update(output_rows=len(frame), output_sha256=sha(out / 'predictions.csv'),
                       exact_primary_keys_per_seed=len(data.mask))
    for path, digest in {**sources, **data.hashes, **dependencies['hashes']}.items():
        if sha(path) != digest:
            raise ValueError('Read-only input/source/dependency changed: ' + path)
    receipt.update(source_inputs_unchanged=True,
                   wall_seconds=time.perf_counter() - started,
                   fit_models=len(diagnostics), evaluation_forecasts_computed=not pilot)
    write_json(out / 'RECEIPT.json', receipt)
    write_json(out / 'ARTIFACT_RECEIPT.json', {str(p.relative_to(out)): sha(p)
               for p in sorted(out.rglob('*')) if p.is_file()})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='Exclusive new full-fit output directory')
    parser.add_argument('--pilot', action='store_true', help='Only AEX/KSE seed8101 TRAIN/2018 diagnostics, no evaluation forecasts')
    args = parser.parse_args()
    receipt = run(args.out, pilot=args.pilot)
    print(json.dumps({k: receipt[k] for k in ('status', 'fit_models', 'wall_seconds', 'evaluation_loss_computed')}, indent=2))
