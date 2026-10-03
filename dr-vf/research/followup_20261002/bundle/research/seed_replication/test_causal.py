"""Synthetic causality, simulator provenance and unit-conversion checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import volpfn as V
import volpfn_prior as VP
from research.seed_replication import causal as C


def simple_raw(r):
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    return dict(r=r, logv=np.zeros_like(r), mask=np.ones_like(r))


class CausalPreparationTests(unittest.TestCase):
    def test_raw_sampler_matches_before_legacy_normalization(self):
        captured = {}
        original = VP.apply_overlays

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured['r'], captured['logv'], captured['mask'] = [x.copy() for x in result[:3]]
            return result

        with patch.object(VP, 'apply_overlays', side_effect=capture):
            VP.sample_batch(np.random.default_rng(9871), 32, 192)
        with patch.object(VP, 'standardise', side_effect=AssertionError('legacy normalization called')):
            actual = C.raw_batch(9871, 32, 192)
        for key in ('r', 'logv', 'mask'):
            np.testing.assert_array_equal(actual[key], captured[key])
        self.assertEqual(actual['r'].dtype, np.float64)
        self.assertEqual(actual['logv'].dtype, np.float64)
        self.assertTrue(all(m['scale'] == 1.0 for m in actual['meta']))

    def test_raw_sampler_reproducible_and_does_not_mutate_weights(self):
        weights = {'garch': 1.0}
        first = C.raw_batch(71, 2, 128, weights=weights, family='garch')
        second = C.raw_batch(71, 2, 128, weights=weights, family='garch')
        self.assertEqual(weights, {'garch': 1.0})
        np.testing.assert_array_equal(first['r'], second['r'])
        self.assertTrue(all(m['clean'] for m in first['meta']))

    def test_running_nonzero_rms_and_zero_fallback(self):
        f = C.feature_batch(np.array([[0., 0., -2., 0., 4.]]))
        np.testing.assert_allclose(f['output_scale'], [[1., 1., 2., 2., np.sqrt(10.)]], rtol=1e-14)
        np.testing.assert_allclose(f['base'][0, :4], 0, atol=1e-7)
        lam = 0.5 ** (1 / 20.)
        self.assertAlmostEqual(float(f['base'][0, 4]), np.log((lam * 4 + (1-lam) * 16) / 10), places=7)
        self.assertTrue(np.isfinite(f['feats']).all())

    def test_prefix_and_suffix_invariance_at_early_and_window_boundaries(self):
        r = np.random.default_rng(14).normal(size=(2, 1100))
        r[0, :5] = 0
        whole = C.feature_batch(r)
        for length in (1, 2, 99, 100, 101, 511, 512, 513, 1023, 1024):
            prefix = C.feature_batch(r[:, :length])
            changed = r.copy()
            changed[:, length:] = 1e20
            suffix = C.feature_batch(changed)
            for key in whole:
                np.testing.assert_array_equal(prefix[key], whole[key][:, :length], err_msg=(key, length))
                np.testing.assert_array_equal(suffix[key][:, :length], whole[key][:, :length], err_msg=(key, length))

    def test_target_units_exclude_target_itself(self):
        raw = simple_raw([2., 4., 6., 8.])
        raw['logv'][0] = np.log([1., 9., 25., 49.])
        b = C.prepare_batch(raw)
        sc = np.array([2., np.sqrt(10.), np.sqrt(56/3)])
        np.testing.assert_allclose(b['target_r'][0], [4., 6., 8.] / sc, rtol=1e-7)
        np.testing.assert_allclose(b['target_logv'][0], np.log([9., 25., 49.]) - 2*np.log(sc), rtol=1e-6)
        other = simple_raw([2., 400000., 6., 8.])
        p = C.prepare_batch(other)
        self.assertEqual(b['output_scale'][0, 0], p['output_scale'][0, 0])
        self.assertNotEqual(b['target_r'][0, 0], p['target_r'][0, 0])
        np.testing.assert_array_equal(b['feats'][:, 0], p['feats'][:, 0])

    def test_target_masks_shift_with_target(self):
        raw = simple_raw([1., 0., 2., 3.])
        raw['mask'][0] = [1, 0, 1, 0]
        b = C.prepare_batch(raw)
        np.testing.assert_array_equal(b['target_mask'], [[0, 1, 0]])
        self.assertEqual(b['feats'].shape, (1, 4, 8))
        self.assertEqual(b['target_r'].shape, (1, 3))

    def test_extreme_finite_scale_stability(self):
        r = np.array([[0., 1e-250, -2e-250, 0.], [0., 1e250, -2e250, 0.]])
        f = C.feature_batch(r)
        for key, value in f.items():
            self.assertTrue(np.isfinite(value).all(), key)
        np.testing.assert_allclose(f['feats'][0], f['feats'][1], atol=1e-6)
        np.testing.assert_allclose(f['base'][0], f['base'][1], atol=1e-6)

    def test_preparation_and_inference_feature_function_identical(self):
        raw = C.raw_batch(76, 2, 192)
        b = C.prepare_batch(raw)
        f = C.feature_batch(raw['r'])
        np.testing.assert_array_equal(b['feats'], f['feats'])
        np.testing.assert_array_equal(b['base'], f['base'])
        np.testing.assert_array_equal(b['output_scale'], f['output_scale'][:, :-1])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            C.feature_batch(np.array([[np.nan, 2.]]))
        with self.assertRaises(ValueError):
            C.prepare_batch(simple_raw([1.]))
        with self.assertRaises(ValueError):
            C.raw_batch(2, 0, 100)
        with self.assertRaises(ValueError):
            C.raw_batch(2, 1, 100, weights={'garch': -1})


class ModelUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(77)
        cls.model = V.VolPFN(d=16, layers=1, heads=2, K=2).eval()

    def test_density_jacobian_and_latent_units(self):
        raw = simple_raw([1.7, -2.1, 0., 7.5, 2.1])
        b = C.to_dev(C.prepare_batch(raw), 'cpu')
        out = self.model(b['feats'], b['base'])
        nll, aux = C.losses(out, b)
        pred = {k: v[:, :-1].double() for k, v in out.items()}
        sc = b['output_scale']
        raw_pred = dict(pred)
        raw_pred['loc'] = pred['loc'] * sc[..., None]
        raw_pred['logscale'] = pred['logscale'] + sc.log()[..., None]
        expected = -V.mixture_logpdf(raw_pred, torch.as_tensor(raw['r'][:, 1:])).mean()
        self.assertAlmostEqual(float(nll.detach()), float(expected.detach()), places=6)
        raw_lat_mean = pred['lat_mean'] + 2*sc.log()
        expected_aux = (0.5*((torch.as_tensor(raw['logv'][:, 1:])-raw_lat_mean)
                            * torch.exp(-pred['lat_logsd']))**2 + pred['lat_logsd']).mean()
        self.assertAlmostEqual(float(aux.detach()), float(expected_aux.detach()), places=6)
        self.assertTrue(torch.isfinite(nll + aux))

    def test_model_matches_prepared_one_window_predictions(self):
        raw = C.raw_batch(123, 1, 160, family='garch')
        b = C.to_dev(C.prepare_batch(raw), 'cpu')
        with torch.no_grad():
            out = self.model(b['feats'], b['base'])
            expected = V.mixture_second_moment(out)[:, :-1].numpy() * b['output_scale'].numpy()**2
        got = C.predict_series(self.model, raw['r'][0], L=160, min_history=1, levels=())
        np.testing.assert_allclose(got['h'][1:], expected[0], rtol=2e-7)
        self.assertTrue(np.isnan(got['h'][0]))

    def test_inference_prefix_and_suffix_invariance_for_owned_outputs(self):
        r = np.random.default_rng(522).normal(size=1100)
        r[:5] = 0
        full = C.predict_series(self.model, r, L=1024, stride=512, min_history=1, levels=())
        for target in (1, 2, 99, 100, 511, 512, 513, 1023, 1024, 1025):
            prefix = C.predict_series(self.model, r[:target+1], L=1024, stride=512, min_history=1, levels=())
            changed = r.copy()
            changed[target:] = np.random.default_rng(target).normal(size=len(r)-target) * 1e3
            suffix = C.predict_series(self.model, changed, L=1024, stride=512, min_history=1, levels=())
            for key in ('logw', 'loc', 'scale', 'nu', 'h', 'mean'):
                np.testing.assert_allclose(prefix[key][1:target+1], full[key][1:target+1], rtol=2e-6, atol=2e-7, err_msg=(key, target))
                np.testing.assert_allclose(suffix[key][1:target+1], full[key][1:target+1], rtol=2e-6, atol=2e-7, err_msg=(key, target))

    def test_window_reset_is_same_as_fresh_training_task(self):
        r = np.random.default_rng(235).normal(size=1100)
        full = C.predict_series(self.model, r, L=1024, stride=512, min_history=1, levels=())
        segment = C.predict_series(self.model, r[512:], L=1024, stride=512, min_history=1, levels=())
        for key in ('h', 'loc', 'scale', 'logw'):
            np.testing.assert_allclose(full[key][1024:], segment[key][512:], rtol=2e-6, atol=2e-7)

    def test_warmup_batchsize_and_training_state_do_not_change_predictions(self):
        r = np.random.default_rng(34).normal(size=360)
        self.model.train()
        a = C.predict_series(self.model, r, L=128, stride=64, min_history=1, batch_windows=1, levels=())
        self.assertTrue(self.model.training)
        b = C.predict_series(self.model, r, L=128, stride=64, min_history=100, batch_windows=4, levels=())
        self.assertTrue(self.model.training)
        self.model.eval()
        np.testing.assert_allclose(a['h'][100:], b['h'][100:], rtol=2e-6)
        self.assertTrue(np.isnan(b['h'][:100]).all())

    def test_checkpoint_requires_new_normalization_specification(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.pt'
            ck = dict(cfg=self.model.cfg, state_dict=self.model.state_dict())
            torch.save(ck, path)
            with self.assertRaises(ValueError):
                C.load_model(path)
            ck['causal_spec'] = C.CAUSAL_SPEC
            torch.save(ck, path)
            with self.assertRaises(ValueError):
                C.load_model(path)
            ck['normalization_spec'] = C.NORMALIZATION_SPEC
            torch.save(ck, path)
            loaded = C.load_model(path)
            self.assertEqual(loaded.causal_spec, C.CAUSAL_SPEC)
            self.assertFalse(loaded.training)


if __name__ == '__main__':
    unittest.main()
