import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from scipy.special import logsumexp
import torch
import volpfn.legacy as V
from experiments.grid_reference import parameter_grid, simulate, reference

class ReferenceTests(unittest.TestCase):
    def test_sequential_posterior_matches_batch_likelihood(self):
        r,_,_=simulate(21,3,19); grid=parameter_grid(); ref=reference(r)
        # Independent per-hypothesis full-prefix likelihood calculation.
        for b in range(3):
            for t in (0,1,7,18):
                ll=[]
                for om,a,be in grid:
                    h=1.; likelihood=0.
                    for x in r[b,:t]:
                        likelihood += -0.5*(np.log(2*np.pi*h)+x*x/h)
                        h=om+a*x*x+be*h
                    ll.append(likelihood)
                w=np.exp(ll-logsumexp(ll))
                np.testing.assert_allclose(ref['weights'][b,t],w,rtol=1e-12,atol=1e-12)

    def test_single_parameter_matches_simulator(self):
        grid=parameter_grid()[2:3]; r,h,_=simulate(7,8,100,grid)
        out=reference(r,grid)
        np.testing.assert_allclose(out['mean_variance'],h)
        np.testing.assert_allclose(out['logdensity'],-.5*(np.log(2*np.pi*h)+r*r/h))

    def test_reference_causality_and_prefix(self):
        r,_,_=simulate(3,4,50); changed=r.copy(); changed[:,23:]+=30
        a,b=reference(r),reference(changed)
        for key in ('weights','variance','mean_variance'):
            np.testing.assert_array_equal(a[key][:,:24],b[key][:,:24])
        np.testing.assert_array_equal(a['logdensity'][:,:23],b['logdensity'][:,:23])
        np.testing.assert_array_equal(a['weights'][:,:23],reference(r[:,:23])['weights'])
        self.assertGreater(np.max(abs(a['mean_variance'][:,24:]-b['mean_variance'][:,24:])),1.)

    def test_mixture_moments_and_normalization(self):
        r,_,_=simulate(4,4,40); out=reference(r)
        np.testing.assert_allclose(out['weights'].sum(-1),1.)
        self.assertTrue(np.all(out['mean_variance']>0))
        # Predictive draws independently validate mixture second moments.
        rng=np.random.default_rng(55); w=out['weights'][0,-1]; h=out['variance'][0,-1]
        ix=rng.choice(len(w),size=200000,p=w); x=np.sqrt(h[ix])*rng.standard_normal(len(ix))
        self.assertLess(abs(np.mean(x*x)/np.dot(w,h)-1),0.02)

    def test_network_and_features_causal(self):
        torch.manual_seed(5); model=V.VolPFN(**V.SIZES['tiny']).eval()
        r,_,_=simulate(5,2,50); changed=r.copy(); changed[:,23:]+=20
        f,b=V.make_features(r); ff,bb=V.make_features(changed)
        with torch.no_grad():
            a=model(torch.tensor(f),torch.tensor(b)); c=model(torch.tensor(ff),torch.tensor(bb))
        for k in a:
            torch.testing.assert_close(a[k][:,:23],c[k][:,:23],rtol=0,atol=0)
        self.assertGreater(float((a['loc'][:,23:]-c['loc'][:,23:]).abs().max()),.01)

    def test_target_alignment(self):
        # The output at t must score return t+1, not the already observed return t.
        out={'logw':torch.zeros(1,3,1),'loc':torch.tensor([[[1.],[2.],[3.]]]),
             'logscale':torch.zeros(1,3,1),'nu':torch.full((1,3,1),5.),
             'lat_mean':torch.zeros(1,3),'lat_logsd':torch.zeros(1,3)}
        r=torch.tensor([[90.,1.,2.]])
        loss,_=V.losses(out,r,torch.zeros_like(r),torch.ones_like(r))
        expected=-V.t_logpdf_torch(torch.tensor(0.),torch.tensor(5.))
        torch.testing.assert_close(loss,expected)

    def test_conditional_loss_single_draw_matches_standard(self):
        from experiments.pilot import conditional_loss
        torch.manual_seed(8)
        model=V.VolPFN(**V.SIZES['tiny'])
        r,_,_=simulate(22,3,12)
        f,b=V.make_features(r)
        out={k:v[:,:-1] for k,v in model(torch.tensor(f),torch.tensor(b)).items()}
        x=torch.tensor(r[:,1:],dtype=torch.float32)
        torch.testing.assert_close(conditional_loss(out,x[:,:,None]),-V.mixture_logpdf(out,x).mean())

    def test_conditional_gradient_matches_gaussian_integral(self):
        from scipy.integrate import quad
        from experiments.pilot import conditional_loss
        # Independent quadrature of d[-log Student-t(x; location)] / d location.
        mu=.7; variance=1.4; nu=5.
        integrand=lambda x: (nu+1)*(mu-x)/(nu+(x-mu)**2)*np.exp(-x*x/(2*variance))/np.sqrt(2*np.pi*variance)
        exact=quad(integrand,-np.inf,np.inf,epsabs=1e-10)[0]
        samples=np.sqrt(variance)*np.random.default_rng(991).standard_normal(100000)
        location=torch.tensor([[[mu]]],dtype=torch.float64,requires_grad=True)
        out={'loc':location,'logscale':torch.zeros_like(location),'logw':torch.zeros_like(location),'nu':torch.full_like(location,nu)}
        conditional_loss(out,torch.tensor(samples[None,None,:])).backward()
        contributions=(nu+1)*(mu-samples)/(nu+(samples-mu)**2)
        se=contributions.std(ddof=1)/np.sqrt(len(samples))
        self.assertLess(abs(float(location.grad)-exact),5*se)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
