"""Replay every frozen helper checkpoint without fitting or altering forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cross_market_blend_20260930 import helper as H


def run(source):
    source = Path(source).resolve()
    quality_path = source / 'QUALITY_VERIFICATION.json'
    if quality_path.exists():
        raise FileExistsError('Do not replace a previous verification')
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = H.sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Frozen receipt mismatch: ' + str(path))
        pins[str(path)] = digest
        return path
    artifacts = json.loads(pin(source / 'ARTIFACT_RECEIPT.json').read_text())
    for path, expected in artifacts.items():
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid artifact path')
        pin(source / relative, expected)
    design = json.loads(pin(source / 'DESIGN.json').read_text())
    receipt = json.loads(pin(source / 'RECEIPT.json').read_text())
    if (receipt['status'] != 'complete' or design['timing_pilot_only']
            or tuple(design['actual_run_seeds']) != H.SEEDS
            or design['parameters'] != H.PARAMETERS
            or design['predictors_per_market'] != 84
            or len(design['actual_run_markets']) != 28
            or receipt['fit_models'] != 140):
        raise ValueError('Complete unchanged registered helper design required')
    for section in ('source_hashes', 'input_hashes'):
        for path, digest in design[section].items():
            pin(path, digest)
    for path, digest in design['dependencies']['hashes'].items():
        pin(path, digest)
    pin(Path(__file__))
    data = H.prepare_dataset()
    if design['feature_names_by_target_market'] != {k: list(v) for k, v in data.feature_names.items()}:
        raise ValueError('Feature names/order changed')
    for path, digest in data.hashes.items():
        pin(path, digest)
    frame = pd.read_csv(pin(source / 'predictions.csv', receipt['output_sha256']),
                        dtype={'market': str, 'date': str, 'seed': int})
    H.validate_predictions(frame, data.mask)
    mask = pd.read_csv(pin(source / 'common_mask.csv'), dtype={'market': str, 'date': str})
    pd.testing.assert_frame_equal(mask, data.mask, check_dtype=False, rtol=1e-12, atol=1e-12)
    normalizers = json.loads(pin(source / 'NORMALIZATION.json').read_text())
    if normalizers != data.scale_receipts:
        raise ValueError('TRAIN-only scaling changed')
    trials = {(t['market'], t['seed']): t for t in receipt['trials']}
    expected_trials = {(m, s) for m in sorted(data.scales) for s in H.SEEDS}
    if len(trials) != 140 or set(trials) != expected_trials or len(receipt['trials']) != 140:
        raise ValueError('Missing/duplicate helper checkpoints')
    models_verified, rows_verified, train_rows_verified = 0, 0, 0
    own_splits, foreign_splits = [], []
    for market in sorted(data.scales):
        train = ((data.records.split == 'train') & (data.records.market == market)).to_numpy()
        evaluation = ((data.records.split == 'evaluation') & (data.records.market == market)).to_numpy()
        expected = data.records.loc[evaluation].drop(columns=['normalized_target', 'observed_rk']).reset_index(drop=True)
        for seed in H.SEEDS:
            trial = trials[market, seed]
            checkpoint = pin(trial['checkpoint'], trial['checkpoint_sha256'])
            fitted = H.model(seed)
            fitted.load_model(checkpoint)
            config = json.loads(fitted.get_booster().save_config())
            if (config['learner']['objective']['name'] != 'reg:gamma'
                    or len(fitted.get_booster().get_dump()) != 40
                    or fitted.get_booster().feature_names != list(data.feature_names[market])):
                raise ValueError('Unexpected saved model/feature identity')
            expected_parameters = {**H.PARAMETERS, 'random_state': seed}
            if (trial['actual_parameters'] != expected_parameters
                    or {name: fitted.get_params()[name] for name in expected_parameters} != expected_parameters):
                raise ValueError('Saved wrapper settings/seed mismatch')
            native = trial['pre_save_booster_config']['learner']
            generic, booster = native['generic_param'], native['gradient_booster']
            if (int(generic['seed']) != seed or generic['seed_per_iteration'] != '1'
                    or int(generic['nthread']) != 1
                    or native['objective']['name'] != 'reg:gamma'
                    or booster['gbtree_train_param']['tree_method'] != 'exact'
                    or int(booster['gbtree_model_param']['num_trees']) != 40):
                raise ValueError('Actual pre-save native TRAIN settings mismatch')
            train_params = booster['updater']['grow_colmaker']['train_param']
            for name, value in dict(max_depth=3, learning_rate=.1, subsample=.75,
                                    colsample_bytree=1., reg_lambda=1., min_child_weight=1., gamma=0.).items():
                np.testing.assert_allclose(float(train_params[name]), value, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(float(native['learner_model_param']['base_score']), 1., rtol=0, atol=0)
            actual = frame.loc[(frame.market == market) & (frame.seed == seed)].sort_values('date').reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[list(expected.columns)], expected, check_dtype=False,
                                          rtol=1e-12, atol=1e-12)
            norm = fitted.predict(H.predictor_matrix(data, evaluation, market)).astype(float)
            np.testing.assert_allclose(actual.normalized_prediction, norm, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.rk_forecast, norm * data.scales[market], rtol=1e-12, atol=1e-12)
            quality = H.train_quality(data, market, fitted.predict(H.predictor_matrix(data, train, market)))
            if not quality['passed']:
                raise ValueError('Saved checkpoint fails registered TRAIN gate')
            for key, value in quality.items():
                original = trial['train_quality'][key]
                if isinstance(value, float):
                    np.testing.assert_allclose(value, original, rtol=1e-12, atol=1e-12)
                elif value != original:
                    raise ValueError('TRAIN diagnostics changed: ' + key)
            usage = H.feature_usage(fitted, data.feature_names[market])
            if usage != trial['feature_usage']:
                raise ValueError('Feature-usage diagnostics changed')
            own_splits.append(usage['own_split_count'])
            foreign_splits.append(usage['foreign_split_count'])
            models_verified += 1; rows_verified += len(actual); train_rows_verified += int(train.sum())
    for path, digest in pins.items():
        if H.sha(path) != digest:
            raise ValueError('Changed input during checkpoint replay: ' + path)
    result = dict(status='passed', method='All checkpoints replayed; no refitting',
                  checkpoints=models_verified, evaluation_rows=rows_verified,
                  train_rows_replayed=train_rows_verified, exact_keys_per_seed=22026,
                  all_28_markets=True, all_five_seeds=True,
                  once_only_percent_squared_scaling=True, original_inputs_unchanged=True,
                  causal_metadata_rebuilt=True, own_feature_splits_total=sum(own_splits),
                  foreign_feature_splits_total=sum(foreign_splits),
                  models_using_foreign_features=int(np.count_nonzero(foreign_splits)),
                  feature_usage_is_diagnostic_not_a_selection_gate=True, source_hashes=pins)
    H.write_json(quality_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    run(parser.parse_args().source)
