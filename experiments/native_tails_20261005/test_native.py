import unittest
import numpy as np
import torch
from scipy import stats
from volpfn import VolPFN
from volpfn.inference import predict_series
from mixtures import predict_mixture,combine
from tails import var_es,numerical_es


class TailTests(unittest.TestCase):
    def test_single_gaussian(self):
        for alpha in (.05,.025,.01):
            q,e,_=var_es([1.],[.2],[1.7],[np.inf],alpha)
            z=stats.norm.ppf(alpha)
            np.testing.assert_allclose(q,.2+1.7*z,atol=1e-10)
            np.testing.assert_allclose(e,.2-1.7*stats.norm.pdf(z)/alpha,atol=1e-10)

    def test_single_student(self):
        for df in (2.1,5.,8.,30.):
            for alpha in (.05,.025,.01):
                q,e,_=var_es([1.],[-.1],[.8],[df],alpha)
                z=stats.t.ppf(alpha,df)
                np.testing.assert_allclose(q,-.1+.8*z,atol=1e-10)
                np.testing.assert_allclose(e,-.1-.8*(df+z*z)/(df-1)*stats.t.pdf(z,df)/alpha,atol=1e-9)

    def test_heterogeneous_mixture_quadrature(self):
        w=np.array([.2,.3,.5]);mu=np.array([-.3,.5,2.]);s=np.array([.2,1.3,.7]);df=np.array([2.1,8.,np.inf])
        for a in (.05,.025,.01):
            q,e,_=var_es(w,mu,s,df,a)
            num,err=numerical_es(w,mu,s,df,a,q[0])
            np.testing.assert_allclose(num,e[0],rtol=1e-8,atol=1e-8)

    def test_equal_density_combination(self):
        seeds=[dict(weight=np.array([[.25,.75]]),location=np.array([[i,i+1.]]),scale=np.ones((1,2)),df=np.ones((1,2))*5) for i in range(5)]
        mixed=combine(seeds)
        np.testing.assert_allclose(mixed['weight'],np.tile([[.05,.15]],(1,5)))
        np.testing.assert_allclose(mixed['weight'].sum(),1)

    def test_causal_and_original_inference_parity(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        model=VolPFN(d=16,layers=1,heads=2,K=2).eval()
        r=np.random.default_rng(3).normal(size=81)
        m=predict_mixture(model,r,L=32,stride=16,min_history=1)
        original=predict_series(model,r,L=32,stride=16,min_history=1,levels=())
        w=m['weight'];mean=np.sum(w*m['location'],axis=1)
        h=np.sum(w*(m['location']**2+m['scale']**2*m['df']/(m['df']-2)),axis=1)
        np.testing.assert_allclose(mean,original['mean'],rtol=2e-6,atol=1e-7,equal_nan=True)
        np.testing.assert_allclose(h,original['h'],rtol=2e-6,atol=1e-7,equal_nan=True)
        for cut in (16,32,40,64):
            changed=r.copy();changed[cut:]+=100
            altered=predict_mixture(model,changed,L=32,stride=16,min_history=1)
            for key in m:
                np.testing.assert_allclose(m[key][:cut+1],altered[key][:cut+1],rtol=1e-6,atol=1e-7,equal_nan=True)


class DecisionTests(unittest.TestCase):
    def test_holm_six_test_family(self):
        from diagnostics import holm
        np.testing.assert_allclose(holm([.001,.01,.02,.04,.2,.9]),[.006,.05,.08,.12,.4,.9])

    def test_coverage_boundary_and_clustered_hits(self):
        from diagnostics import coverage
        zero=coverage(np.zeros(1000,dtype=bool),.01)
        self.assertEqual(zero['p_ind'],1.)
        self.assertLess(zero['p_cc'],.05)
        hits=np.zeros(1000,dtype=bool);hits[100:150]=True
        clustered=coverage(hits,.05)
        self.assertLess(clustered['p_ind'],.05)
        self.assertLess(clustered['p_cc'],.05)

    def test_guard_uses_only_past_outcomes(self):
        from calibeat import market as M
        from calibeat.fz import fz0
        rng=np.random.default_rng(18);y=rng.standard_t(5,2400)
        v=np.full(len(y),-2.);e=np.full(len(y),-3.)
        def guard(values):
            vw,ew=M.calibeat('ratio_win1000',values,v,e,.05)
            return M.switch(fz0(values,v,e,.05),fz0(values,vw,ew,.05),v,e,vw,ew,
                            window=M.GUARD_WINDOW,margin=M.GUARD_MARGIN[.05])
        changed=y.copy();changed[1800:]-=100
        first,second=guard(y),guard(changed)
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a[:1801],b[:1801])


if __name__=='__main__':
    unittest.main()
