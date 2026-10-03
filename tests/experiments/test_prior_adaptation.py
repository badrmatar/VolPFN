"""Independent mathematical and information-set checks for prior fitting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import t as student

from experiments.grid_reference import reference
from experiments.prior_adaptation.model import (
    component_paths, log_law, make_bank, marginal_fit, predict,
    predictive_fit, predictive_objective)
from experiments.prior_adaptation.experiment import (
    BURN, GROUPS, load_market, make_blocks, training_arrays)


class PriorAdaptationTests(unittest.TestCase):
    def test_gaussian_bank_matches_independent_existing_reference(self):
        bank, fixed = make_bank()
        r = np.random.default_rng(3).normal(size=75)
        mask = fixed['coarse'] > 0
        lp, h = component_paths(r, bank[mask])
        logdensity, weight = predict(lp, np.full(mask.sum(), 1/mask.sum()))
        old = reference(r[None], grid=bank[mask, :3])
        np.testing.assert_allclose(h, old['variance'][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(weight, old['weights'][0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(logdensity, old['logdensity'][0], rtol=1e-12, atol=1e-12)

    def test_student_density_normalization_and_variance(self):
        for nu in (5., 8.):
            h = 2.7
            def density(x):
                return np.exp(log_law(np.array([x]), np.array([h]), np.array([nu]))[0])
            area = quad(density, -np.inf, np.inf, epsabs=1e-9)[0]
            moment = quad(lambda x: x*x*density(x), -np.inf, np.inf, epsabs=1e-9)[0]
            self.assertAlmostEqual(area, 1., places=8)
            self.assertAlmostEqual(moment, h, places=8)
            x = np.array([-10., -1., 0., 1., 10.])
            expected = student.logpdf(x, nu, scale=np.sqrt(h*(nu-2)/nu))
            np.testing.assert_allclose(log_law(x, np.full(5, h), np.full(5, nu)), expected)

    def test_reweighting_and_zero_mass_match_direct_enumeration(self):
        bank, _ = make_bank(); bank = bank[[0, 46, 80]]
        r = np.random.default_rng(4).normal(size=10)
        lp, _ = component_paths(r, bank)
        for prior in (np.array([1., 0., 0.]), np.array([.1, .2, .7])):
            score, weights = predict(lp, prior)
            for t in range(len(r)):
                evidence = np.exp(lp[:t].sum(0))*prior
                evidence /= evidence.sum()
                expected = np.log(np.sum(evidence*np.exp(lp[t])))
                self.assertAlmostEqual(score[t], expected, places=11)
                np.testing.assert_allclose(weights[t], evidence, atol=1e-12)
        np.testing.assert_allclose(predict(lp, np.array([1., 0., 0.]))[0], lp[:, 0])

    def test_predictive_objective_gradient(self):
        rng = np.random.default_rng(5)
        before = rng.normal(size=(12, 4))*2
        end = before+rng.normal(size=(12, 4))*3-15
        logits = rng.normal(size=4); weight = rng.uniform(.1, 1., size=12)
        _, analytic = predictive_objective(logits, end, before, weight)
        numerical = np.empty(4)
        for j in range(4):
            dx = np.zeros(4); dx[j] = 1e-5
            numerical[j] = (predictive_objective(logits+dx, end, before, weight)[0]
                            - predictive_objective(logits-dx, end, before, weight)[0])/2e-5
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)

    def test_estimators_improve_their_declared_objectives(self):
        rng = np.random.default_rng(6)
        end = rng.normal(size=(60, 4)); end[:, 2] += 3.
        before = np.zeros_like(end); weight = np.full(60, 1/10)
        eb, info = marginal_fit(end, weight)
        self.assertTrue(info['converged'])
        self.assertGreater(eb[2], .8)
        pp, diagnostic = predictive_fit(end, before, weight,
                                       {'uniform': np.full(4, .25), 'eb': eb})
        self.assertTrue(diagnostic['converged'])
        # With no conditioning prefix, both optimize the same smoothed likelihood.
        np.testing.assert_allclose(eb, pp, atol=2e-5)

    def test_history_suffix_does_not_change_component_predictions(self):
        bank, _ = make_bank(); r = np.random.default_rng(7).normal(size=90)
        lp, h = component_paths(r, bank)
        altered = r.copy(); altered[50:] *= 100
        other_lp, other_h = component_paths(altered, bank)
        np.testing.assert_array_equal(h[:51], other_h[:51])
        np.testing.assert_array_equal(predict(lp, np.ones(len(bank)))[1][:51],
                                      predict(other_lp, np.ones(len(bank)))[1][:51])

    def test_load_rejects_reserved_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.npz'
            np.savez(path, date=np.array(['2014-12-31', '2015-01-02']),
                     returns=np.array([.01, .02]))
            with self.assertRaises(ValueError):
                load_market(path)

    def test_population_fit_excludes_target_markets_and_later_blocks(self):
        def block(role, v):
            return {'role': role, 'cumulative': np.full((BURN+4, 3), v)}
        records = {'source': [block('fit', 1.), block('select', 2.), block('evaluate', 3.)],
                   'target': [block('fit', 100.)]}
        a = training_arrays(records, ['source'])
        records['target'][0]['cumulative'][:] = -10000
        records['source'][2]['cumulative'][:] = 10000
        b = training_arrays(records, ['source'])
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(a[2].sum(), 1.)

    def test_blocks_do_not_cross_calendar_roles(self):
        dates = np.r_[np.full(130, '2007-12-31'), np.full(80, '2010-12-31'),
                      np.full(80, '2011-01-03')]
        market = {'date': dates, 'z': np.ones(len(dates))}
        bank, _ = make_bank(); blocks, _ = make_blocks(market, bank[:1], 128)
        for b in blocks:
            self.assertEqual(len(np.unique(dates[b['index']])), 1)
        all_names = [name for group in GROUPS.values() for name in group]
        self.assertEqual(len(set(all_names)), 28)
        self.assertEqual(len(all_names), 28)


if __name__ == '__main__':
    unittest.main()
