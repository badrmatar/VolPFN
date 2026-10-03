"""Diagnostic tests; no historical forecast or result archive is written."""
import decimal
import hashlib
import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from research.gpvol_full import gpvol_rapcf as original
from research.scientific_followup_20260930.gp import likelihood_v1 as repair


def oracle(x, v):
    with decimal.localcontext() as c:
        c.prec = 100
        c.Emax = 999999999
        # High precision pi; binary64 x/v are the exact inputs being evaluated.
        pi = decimal.Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679")
        xx, vv = decimal.Decimal.from_float(float(x)), decimal.Decimal.from_float(float(v))
        return -((2 * pi).ln() + vv + xx * xx * (-vv).exp()) / 2


class LikelihoodTests(unittest.TestCase):
    def test_ordinary_range(self):
        rng = np.random.default_rng(20260930)
        for x in rng.normal(size=20):
            v = np.linspace(-20, 20, 41)
            np.testing.assert_allclose(repair.log_normal_obs(x, v), original.log_normal_obs(x, v), rtol=5e-15, atol=1e-13)

    def test_decimal_oracle_counterexamples(self):
        cases = [(1e-100, -1000), (1e-200, -1000), (1.0, -710),
                 (float(np.nextafter(0.0, 1.0)), -1500), (1e200, 1000), (0.0, -1000)]
        for x, v in cases:
            with self.subTest(x=x, v=v):
                expected = float(oracle(x, v))
                self.assertTrue(np.isfinite(expected))
                np.testing.assert_allclose(repair.log_normal_obs(x, v), expected, rtol=5e-13, atol=1e-10)

    def test_representable_cancellation(self):
        v = -1400.0
        x = math.exp(0.5 * (v + math.log(-v - math.log(2 * math.pi))))
        expected = float(oracle(x, v))
        self.assertLess(abs(expected), 1e-9)
        self.assertLess(abs(float(repair.log_normal_obs(x, v)) - expected), 2e-10)

    def test_true_overflow_and_invalid_inputs(self):
        self.assertTrue(np.isneginf(repair.log_normal_obs(0.1, -1000.0)))
        self.assertTrue(np.isfinite(repair.log_normal_obs(0.0, -1e308)))
        for x, v in [(float("nan"), 0), (float("inf"), 0), (1, float("nan")), (1, float("inf")), ([1, 2], 0)]:
            with self.assertRaises(ValueError):
                repair.log_normal_obs(x, v)

    def test_zero_scalar_vector_extreme_log_variance(self):
        values = np.array([-1e308, -1500., -1000., 0., 1000., 1e308])
        expected = -0.5 * values - 0.5 * math.log(2 * math.pi)
        np.testing.assert_array_equal(repair.log_normal_obs(0., values), expected)
        for v, target in zip(values, expected):
            self.assertEqual(float(repair.log_normal_obs(0., float(v))), float(target))
        nonzero = repair.log_normal_obs(1., values)
        self.assertTrue(np.isneginf(nonzero[0]))
        self.assertTrue(np.isneginf(nonzero[1]))
        self.assertTrue(np.isneginf(nonzero[2]))
        self.assertTrue(np.isfinite(nonzero[3:]).all())

    def test_private_binding_and_short_filter(self):
        original_hash = hashlib.sha256(repair.ORIGINAL_PATH.read_bytes()).hexdigest()
        old_function = original.log_normal_obs
        fixed = repair.load_filter_module()
        self.assertIs(original.log_normal_obs, old_function)
        self.assertIs(fixed.log_normal_obs, repair.log_normal_obs)
        x = np.random.default_rng(31).normal(size=22)
        settings = dict(n_particles=12, burn_in=10, seed=4, learn_hyp=False)
        before = original.RAPCF(x, **settings).run()
        after = fixed.RAPCF(x, **settings).run()
        np.testing.assert_allclose(after.out["h"][10:], before.out["h"][10:], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(after.out["logdens"][10:], before.out["logdens"][10:], rtol=1e-10, atol=1e-12)
        self.assertEqual(after.rng.bit_generator.state, before.rng.bit_generator.state)
        self.assertIs(original.log_normal_obs, old_function)
        self.assertEqual(hashlib.sha256(repair.ORIGINAL_PATH.read_bytes()).hexdigest(), original_hash)

    def test_actual_filter_causality(self):
        fixed = repair.load_filter_module()
        x = np.random.default_rng(7).normal(size=20)
        y = x.copy()
        y[15:] += 0.8
        settings = dict(n_particles=12, burn_in=10, seed=4, learn_hyp=False)
        a = fixed.RAPCF(x, **settings).run()
        b = fixed.RAPCF(y, **settings).run()
        # h[15] is issued before x[15]. Its retrospective density is not causal output.
        np.testing.assert_array_equal(a.out["h"][10:16], b.out["h"][10:16])
        np.testing.assert_array_equal(a.out["param_mean"][10:15], b.out["param_mean"][10:15])

    def test_save_resume_exact(self):
        fixed = repair.load_filter_module()
        x = np.random.default_rng(9).normal(size=20)
        settings = dict(n_particles=12, burn_in=10, seed=4, learn_hyp=False)
        a = fixed.RAPCF(x, **settings).run()
        b = fixed.RAPCF(x, **settings)
        for _ in range(4):
            b.step()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state.npz"
            b.save(p)
            c = fixed.RAPCF.load(p).run()
        for key in a.out:
            np.testing.assert_array_equal(a.out[key], c.out[key])
        self.assertEqual(a.rng.bit_generator.state, c.rng.bit_generator.state)

    def test_versioned_runner_actual_200_particle_prefix(self):
        from research.gpvol_full import run_frontier_gpvol as runner
        from research.scientific_followup_20260930.gp import run_same_protocol_v1 as supported
        from types import SimpleNamespace
        bundle, rows = runner.load_plan()
        settings = runner.settings_from(SimpleNamespace(seed=2,n_particles=200,burn_in=100,
            lam=.95,standardise_window=1000,code_indexing=False,chunk=32))
        full = runner.prepare_segments('GSPTSE', rows['GSPTSE'], bundle['market_paths']['GSPTSE'], settings)[0]
        short = dict(full, x=full['x'][:102], prow=full['prow'][:2])
        old_function = original.log_normal_obs
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(runner, 'prepare_segments', return_value=[short]):
                result = supported.full_cell('GSPTSE',2,bundle,rows,Path(d))
            self.assertFalse(result['failed'])
            payload = json.loads((Path(d)/'cells/gpvol_likelihood_v1/2/GSPTSE/forecast.json').read_text())
            self.assertEqual(len(payload['rows']), 2)
            self.assertEqual(payload['audit']['settings']['n_particles'],200)
            self.assertFalse(payload['audit']['full_ranking_established'] if 'full_ranking_established' in payload['audit'] else payload['audit']['provenance']['full_primary_replay_complete'])
        self.assertIs(original.log_normal_obs,old_function)


if __name__ == "__main__":
    unittest.main()
