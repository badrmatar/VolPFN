"""Causality/normalization/gamma safeguards; synthetic fits only in tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.cross_market_blend_20260930 import helper as H


def fixture():
    dates = pd.bdate_range('2017-09-01', '2019-02-01').strftime('%Y-%m-%d').to_numpy(dtype='U10')
    panel = {}
    for i, market in enumerate(('A', 'B', 'C')):
        risk = (i + 1.) * (1. + np.arange(len(dates)) / 100. + 0.1 * np.sin(np.arange(len(dates))))
        panel[market] = dict(date=dates.copy(), rk=risk)
    rows = []
    for market, item in panel.items():
        for j in np.flatnonzero(item['date'] >= '2019-01-01')[:4]:
            rows.append(dict(market=market, date=str(item['date'][j]), index=int(j), rk=float(item['rk'][j])))
    return panel, pd.DataFrame(rows).sort_values(['market', 'date']).reset_index(drop=True)


def snapshots(panel):
    return {m: H.session_har(item) for m, item in panel.items()}


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.mask = fixture()

    def test_individual_foreign_columns_are_lexical_and_full84(self):
        markets = ['A'] + ['M' + str(i).zfill(2) for i in range(27)]
        names = H.feature_names('A', reversed(markets))
        self.assertEqual(len(names), 84)
        self.assertEqual(names[:6], ('own_har1', 'own_har5', 'own_har22',
                                     'foreign_M00_har1', 'foreign_M00_har5', 'foreign_M00_har22'))
        self.assertFalse(any('foreign_A_' in n for n in names))

    def test_normalizers_use_all_positive_train_only(self):
        scales, receipts = H.training_scales(self.panel)
        changed = copy.deepcopy(self.panel)
        for item in changed.values():
            item['rk'][item['date'] > H.TRAIN_END] *= 1000
        self.assertEqual(scales, H.training_scales(changed)[0])
        for market, item in self.panel.items():
            original = item['rk'][item['date'] <= H.TRAIN_END]
            self.assertEqual(scales[market], original.mean())
            self.assertEqual(receipts[market]['positive_observed_rows'], len(original))
            self.assertEqual(receipts[market]['first_date'], item['date'][0])

    def test_future_target_shocks_do_not_change_current_features(self):
        scales = H.training_scales(self.panel)[0]
        own = self.panel['A'];i = int(np.searchsorted(own['date'], '2019-01-10'))
        a, eligible_a, meta_a = H.feature_rows(self.panel, snapshots(self.panel), scales, 'A', [i], 1)
        changed = copy.deepcopy(self.panel)
        for item in changed.values():
            item['rk'][item['date'] >= own['date'][i]] *= 2000
        b, eligible_b, meta_b = H.feature_rows(changed, snapshots(changed), H.training_scales(changed)[0], 'A', [i], 1)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(eligible_a, eligible_b)
        pd.testing.assert_frame_equal(meta_a, meta_b)

    def test_same_day_foreign_shock_excluded_but_later_can_enter(self):
        scales = H.training_scales(self.panel)[0]
        own = self.panel['A'];i = int(np.searchsorted(own['date'], '2019-01-11'))
        shock_day = own['date'][i - 1]
        original, _, _ = H.feature_rows(self.panel, snapshots(self.panel), scales, 'A', [i, i + 1], 1)
        changed = copy.deepcopy(self.panel)
        changed['B']['rk'][changed['B']['date'] == shock_day] *= 1000
        other, _, meta = H.feature_rows(changed, snapshots(changed), scales, 'A', [i, i + 1], 1)
        np.testing.assert_array_equal(original[0], other[0])
        self.assertFalse(np.array_equal(original[1], other[1]))
        self.assertTrue((meta.latest_foreign_input_date < meta.origin_date).all())

    def test_holidays_use_source_dates_not_ordinal_matching(self):
        panel = copy.deepcopy(self.panel)
        keep = panel['B']['date'] != '2019-01-09'
        panel['B'] = {k: v[keep] for k, v in panel['B'].items()}
        scales = H.training_scales(panel)[0]
        own = panel['A'];i = int(np.searchsorted(own['date'], '2019-01-11'))
        x, eligible, meta = H.feature_rows(panel, snapshots(panel), scales, 'A', [i], 1)
        source = panel['B'];p = int(np.searchsorted(source['date'], '2019-01-10', side='left') - 1)
        self.assertEqual(source['date'][p], '2019-01-08')
        self.assertTrue(eligible[0]);self.assertEqual(meta.origin_date.iloc[0], '2019-01-10')
        self.assertAlmostEqual(float(x[0, 3]), float(source['rk'][p] / scales['B']))

    def test_partial_foreign_lags_stay_nan_and_full_count_controls_eligibility(self):
        panel = copy.deepcopy(self.panel)
        panel['C'] = {k: v[-35:] for k, v in panel['C'].items()}
        # Means were fitted from a different historical TRAIN archive; this test
        # isolates feature availability rather than refitting an empty scaler.
        scales = H.training_scales(self.panel)[0]
        i = int(np.searchsorted(panel['A']['date'], panel['C']['date'][7]))
        x, eligible, meta = H.feature_rows(panel, snapshots(panel), scales, 'A', [i], 1)
        self.assertTrue(eligible[0]);self.assertEqual(int(meta.foreign_count.iloc[0]), 1)
        self.assertEqual(int(meta.foreign_partial_count.iloc[0]), 1)
        self.assertTrue(np.isfinite(x[0, 6:8]).all());self.assertTrue(np.isnan(x[0, 8]))
        self.assertFalse(H.feature_rows(panel, snapshots(panel), scales, 'A', [i], 2)[1][0])

    def test_labels_are_never_carried_and_actual_input_owner_is_recorded(self):
        panel = copy.deepcopy(self.panel);invalid = 50;day = panel['A']['date'][invalid]
        panel['A']['rk'][invalid] = np.nan
        data = H.prepare_panel(panel, self.mask, min_foreign=1)
        self.assertFalse(((data.records.market == 'A') & (data.records.date == day)).any())
        next_row = data.records[(data.records.market == 'A') & (data.records['index'] == invalid + 1)].iloc[0]
        self.assertEqual(next_row.origin_date, day)
        self.assertEqual(next_row.latest_own_input_date, panel['A']['date'][invalid - 1])
        self.assertGreater(next_row.own_input_imputations, 0)
        self.assertTrue((data.records.observed_rk > 0).all())
        panel['A']['rk'][int(self.mask[self.mask.market == 'A']['index'].iloc[0])] = np.nan
        with self.assertRaises(ValueError):
            H.prepare_panel(panel, self.mask, min_foreign=1)

    def test_gate_uses_eligible_train_mean_and_validation_has_no_selection(self):
        data = H.prepare_panel(self.panel, self.mask, min_foreign=1)
        train = ((data.records.split == 'train') & (data.records.market == 'A')).to_numpy()
        c = float(data.y[train].mean())
        optimal = H.train_quality(data, 'A', np.full(train.sum(), c))
        self.assertTrue(optimal['passed']);self.assertAlmostEqual(optimal['eligible_train_constant_normalized'], c)
        self.assertGreater(optimal['normalized_fixed1_qlike'], optimal['normalized_constant_qlike'])
        self.assertFalse(H.train_quality(data, 'A', np.full(train.sum(), 2*c))['passed'])
        self.assertFalse(H.train_quality(data, 'A', np.ones(train.sum()))['passed'])
        val = ((data.records.split == 'validation') & (data.records.market == 'A')).to_numpy()
        diagnostic = H.validation_diagnostics(data, 'A', np.full(val.sum(), c))
        self.assertFalse(diagnostic['selection_or_tuning'])
        self.assertEqual(diagnostic['train_fitted_constant_normalized'], c)

    def test_gamma_model_accepts_missing_features_and_stays_positive(self):
        rng = np.random.default_rng(42)
        names = H.feature_names('A', ['A'] + ['M' + str(i).zfill(2) for i in range(27)])
        values = rng.uniform(.2, 2., size=(64, 84));values[:20, 6] = np.nan
        x = pd.DataFrame(values, columns=names)
        y = np.exp(.2*x.own_har1.to_numpy() + .1*x.foreign_M00_har1.to_numpy())
        fitted = H.model(8101);fitted.fit(x, y)
        forecast = fitted.predict(x)
        self.assertTrue(np.isfinite(forecast).all());self.assertTrue((forecast > 0).all())
        self.assertEqual(fitted.get_params()['objective'], 'reg:gamma')
        self.assertEqual(fitted.get_params()['n_estimators'], 40)
        self.assertTrue(fitted.get_params()['seed_per_iteration'])
        config = json.loads(fitted.get_booster().save_config())
        self.assertEqual(config['learner']['generic_param']['seed'], '8101')
        use = H.feature_usage(fitted, names)
        self.assertEqual(len(use['split_counts_by_feature']), 84)
        self.assertFalse(use['selection_or_gate'])
        self.assertAlmostEqual(use['own_split_count'] + use['foreign_split_count'],
                               sum(use['split_counts_by_feature'].values()))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'synthetic_gamma.json'
            fitted.save_model(checkpoint)
            replay = H.model(8101);replay.load_model(checkpoint)
            np.testing.assert_array_equal(replay.predict(x), forecast)
            self.assertEqual(replay.get_booster().feature_names, list(names))
            self.assertEqual(replay._estimator_type, 'regressor')
            self.assertEqual(len(replay.get_booster().get_dump()), 40)
            self.assertEqual(replay.get_params()['random_state'], 8101)
            for parameter, value in H.PARAMETERS.items():
                self.assertEqual(replay.get_params()[parameter], value, parameter)
        with self.assertRaises(ValueError):
            H.model(9000)

    def test_exact_mask_seed_set_positive_outputs_and_no_overwrite(self):
        mask = pd.DataFrame({'market': ['A', 'B'], 'date': ['2020-01-02', '2020-01-03']})
        f = pd.concat([mask.assign(seed=s, rk_forecast=1., latest_input_date='2019-12-31',
                                  latest_own_input_date='2019-12-31', market_scale=1., normalized_prediction=1.,
                                  origin_date='2020-01-01', latest_foreign_input_date='2019-12-31',
                                  foreign_count=27) for s in H.SEEDS], ignore_index=True)
        H.validate_predictions(f, mask)
        for broken in (f.iloc[:-1], f[f.seed != 8101], f.assign(rk_forecast=0.),
                       f.assign(latest_foreign_input_date=f.origin_date),
                       f.assign(latest_own_input_date=None),
                       f.assign(latest_own_input_date='2020-01-01'),
                       f.assign(latest_foreign_input_date='0000-00-00'),
                       f.assign(origin_date='2020-1-01')):
            with self.assertRaises(ValueError):
                H.validate_predictions(broken, mask)
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'original.txt';marker.write_text('original')
            with self.assertRaises(FileExistsError):
                H.run(folder)
            self.assertEqual(marker.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
