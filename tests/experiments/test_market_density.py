import copy
import unittest
import numpy as np
from arch.univariate import LS, GARCH, Normal
from experiments.market_density.models import fit, advance, simulate, design, score, garch_next, HORIZONS, PROBS, METHODS


class DensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(90721)
        z = np.zeros(390)
        for t in range(22, len(z)):
            z[t] = .45*z[t-1]+.25*z[t-5:t].mean()+.2*z[t-22:t].mean()+.2*rng.normal()
        cls.v = np.exp(z-9)
        cls.s = fit(cls.v[:350], window=250)

    def test_features_have_no_future_response(self):
        z = np.arange(30.)
        x,y = design(z)
        np.testing.assert_equal(x[0], [1,21,19,10.5])
        self.assertEqual(y[0],22)
        z[22:] = 999
        np.testing.assert_equal(design(z)[0][0], x[0])

    def test_arch_one_step_mean_variance(self):
        # Independent library forecast, not another invocation of our simulator.
        z = np.log(self.v[:350][-272:])-self.s['center']
        x,y = design(z)
        g = self.s['log_garch']
        result = LS(y,x[:,1:],volatility=GARCH(1,0,1),distribution=Normal(),rescale=False).fix(np.r_[g['b'],g['theta']])
        nextx=np.array([z[-1],z[-5:].mean(),z[-22:].mean()])
        forecast=result.forecast(horizon=1,x=nextx.reshape(3,1,1),reindex=False)
        expected=float(forecast.residual_variance.values[-1,0])
        self.assertAlmostEqual(expected,garch_next(g['theta'],g['residual'],g['h']),places=10)
        innovations=np.zeros((3,22));innovations[:,0]=[-1,0,1]
        paths,_=simulate(self.s,self.v[:350],'gaussian_loghar_garch',np.random.default_rng(1),n=3,innovations=innovations)
        expected_logs=float(forecast.mean.values[-1,0])+np.sqrt(expected)*innovations[:,0]+self.s['center']
        np.testing.assert_allclose(np.log(paths[:,0]),expected_logs,atol=1e-10)

    def test_causal_state_updates_match_manual_recursion(self):
        s=copy.deepcopy(self.s);g=s['log_garch'];h=g['h'];e=g['residual']
        for t in range(350,355):
            h=garch_next(g['theta'],e,h)
            z=np.log(self.v[t-22:t+1])-s['center']
            x=np.array([1,z[-2],z[-6:-1].mean(),z[:-1].mean()])
            e=z[-1]-x@g['b']
        advance(s,self.v[:355]);self.assertAlmostEqual(s['log_garch']['h'],h,places=12)
        self.assertAlmostEqual(s['log_garch']['residual'],e,places=12)
        before=copy.deepcopy(s);advance(s,self.v[:355]);self.assertEqual(s['log_garch']['h'],before['log_garch']['h'])

    def test_horizon_aggregation_and_positive_paths(self):
        for method in METHODS:
            a,_=simulate(self.s,self.v[:350],method,np.random.default_rng(34),n=128)
            daily,_=simulate(self.s,self.v[:350],method,np.random.default_rng(34),n=128,return_daily=True)
            np.testing.assert_allclose(a,np.cumsum(daily,axis=1)[:,HORIZONS-1])
            self.assertTrue(np.all(daily>0))

    def test_units_change_only_output_scale(self):
        scaled=fit(self.v[:350]*10000,window=250)
        for method in METHODS:
            a,_=simulate(self.s,self.v[:350],method,np.random.default_rng(55),n=128)
            b,_=simulate(scaled,self.v[:350]*10000,method,np.random.default_rng(55),n=128)
            np.testing.assert_allclose(a,b/10000,rtol=.002,atol=1e-9)

    def test_bayesian_parameters_persist_for_entire_path(self):
        seed=245;n=64;rng=np.random.default_rng(seed)
        var=self.s['rate']/rng.gamma(self.s['shape'],size=n)
        b=self.s['bn']+(rng.normal(size=(n,4))@self.s['chol'].T)*np.sqrt(var[:,None])
        histories=np.tile(np.log(self.v[328:350])-self.s['center'],(n,1))
        for _ in range(22):
            x=np.c_[np.ones(n),histories[:,-1],histories[:,-5:].mean(1),histories[:,-22:].mean(1)]
            histories=np.c_[histories,np.sum(x*b,axis=1)]
        got,_=simulate(self.s,self.v[:350],'bayes_loghar',np.random.default_rng(seed),n=n,innovations=np.zeros((n,22)),return_daily=True)
        np.testing.assert_allclose(np.log(got)-self.s['center'],histories[:,-22:],atol=1e-12)

    def test_known_pinball_and_coverage(self):
        q=np.tile(np.array([1,2,3,4,5]),(3,1))*HORIZONS[:,None]
        scores=score(q,np.array([3,15,66]),2)
        errors=np.array([2,1,0,-1,-2]);expected=np.maximum(PROBS*errors,(PROBS-1)*errors).mean()/2
        np.testing.assert_allclose(scores[:,0],expected)
        np.testing.assert_equal(scores[:,1:4],[[1,0,0]]*3)

    def test_date_blocks_resample_market_totals_together(self):
        from experiments.market_density.analyze import block_weights, panel_summary
        values=np.array([1.,3.,10.]).reshape(3,1,1,1)
        # Two contemporaneous markets on date 0 and one on date 1.
        weights=np.array([[2.,0.],[0.,2.],[1.,1.]])
        point,boot=panel_summary(values,np.array([0,0,1]),2,weights)
        np.testing.assert_allclose(point,14/3)
        np.testing.assert_allclose(boot.ravel(),[2,10,14/3])
        w=block_weights(25,7,replicates=10)
        np.testing.assert_equal(w.sum(1),np.full(10,25))


if __name__=='__main__':unittest.main()
