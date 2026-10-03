"""Analytic moment, admissibility and unchanged-path checks; no training."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.integrate import quad

import volpfn_prior as VP
from research.matched_prior import simulators as S
from research.seed_replication import causal, training

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'research/outputs/programme_closure_20260919_v1/original_sources/volpfn_prior_before_gjr_correction.py'


def archived_prior():
    spec=importlib.util.spec_from_file_location('before_gjr_correction',ARCHIVE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def numerical_negative_moment(nu,gamma):
    # Independent integration of the standardized density, not partial-moment
    # expressions. Split where the skewed density changes branches.
    mean,var=VP.skew_moments(nu,gamma)
    split=-float(mean)/np.sqrt(float(var))
    points=[-np.inf]+([split] if split<0 else [])+[0.]
    total=0.
    for lower,upper in zip(points[:-1],points[1:]):
        value,error=quad(lambda x:x*x*np.exp(VP.law_logpdf(x,nu,gamma)),lower,upper,
                         epsabs=2e-10,epsrel=2e-10,limit=300)
        total+=value
    return total


class GJRMomentTests(unittest.TestCase):
    def test_density_quadrature_across_prior_and_gaussian_limit(self):
        for nu in (2.5,2.541481637298658,3.,8.,50.,VP.NU_GAUSS):
            for gamma in (.7,.85,1.,1.15,1.4):
                with self.subTest(nu=nu,gamma=gamma):
                    actual=float(VP.skew_negative_second_moment(nu,gamma))
                    self.assertAlmostEqual(actual,numerical_negative_moment(nu,gamma),delta=2e-8)

    def test_symmetry_reflection_vectorization_and_admissibility(self):
        nu=np.array([2.0001,2.5,3.,50.,VP.NU_GAUSS])[:,None]
        gamma=np.linspace(.7,1.4,101)[None,:]
        k=VP.skew_negative_second_moment(nu,gamma)
        np.testing.assert_allclose(k+VP.skew_negative_second_moment(nu,1/gamma),1.,rtol=0,atol=3e-15)
        np.testing.assert_array_equal(VP.skew_negative_second_moment(nu,1.),np.full((5,1),.5))
        self.assertTrue(np.all((k>=0)&(k<=1)))
        persistence=.999
        # Worst allowed coefficient combination; all other sampled choices give
        # an equal or larger nonnegative beta for the same kappa.
        beta=persistence-.2*persistence-.35*persistence*k
        self.assertTrue(np.all(beta>=.45*persistence-1e-15))
        np.testing.assert_allclose(.2*persistence+.35*persistence*k+beta,persistence,atol=2e-16)

    def test_invalid_moment_requests_fail_without_clipping(self):
        for nu,gamma in ((2.,1.),(np.nan,1.),(3.,0.),(3.,np.inf)):
            with self.assertRaises(ValueError):VP.skew_negative_second_moment(nu,gamma)

    def test_failed_draw_reproduces_before_and_is_admissible_after(self):
        with patch.object(causal,'VP',archived_prior()),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            with self.assertRaisesRegex(FloatingPointError,'raw simulator'):
                causal.raw_batch(810333125,8,1024)
        raw=causal.raw_batch(810333125,8,1024)
        prepared=causal.prepare_batch(raw)
        training.validate_batch(prepared)
        for meta in raw['meta']:
            if meta['family']=='gjr':
                p=meta['params'];k=float(VP.skew_negative_second_moment(meta['nu'],meta['gamma']))
                self.assertGreater(p['beta'],0)
                self.assertAlmostEqual(p['alpha']+p['gamma']*k+p['beta'],1-p['omega'],places=13)
        self.assertAlmostEqual(raw['meta'][7]['params']['beta'],.8580225316725713,places=14)

    def test_rng_consumption_unchanged_and_non_gjr_family_paths_identical(self):
        old=archived_prior()
        nu=np.array([2.6,3.,8.,VP.NU_GAUSS]);gamma=np.array([.7,1.,1.2,1.4])
        for kind in ('garch','gjr','egarch'):
            before_rng=np.random.default_rng(177);after_rng=np.random.default_rng(177)
            before=old.sim_garch_family(before_rng,kind,4,128,nu,gamma)
            after=VP.sim_garch_family(after_rng,kind,4,128,nu,gamma)
            self.assertEqual(before_rng.bit_generator.state,after_rng.bit_generator.state)
            if kind!='gjr':
                np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
                self.assertEqual(before[2],after[2])

    def test_matern_arrays_metadata_and_causal_preparation_bitwise_unchanged(self):
        old=archived_prior()
        for seed in (810124304,810224307,810333125,810328884):
            with patch.object(S,'VP',old):before=S.raw_batch('matern',seed,8,128)
            after=S.raw_batch('matern',seed,8,128)
            self.assertEqual(before['meta'],after['meta'])
            for key in ('r','logv','mask'):np.testing.assert_array_equal(before[key],after[key])
            prepared_before,prepared_after=causal.prepare_batch(before),causal.prepare_batch(after)
            for key in prepared_before:np.testing.assert_array_equal(prepared_before[key],prepared_after[key])


if __name__=='__main__':unittest.main()
