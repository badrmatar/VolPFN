"""Finite-prior Gaussian GARCH laboratory; no fitted normalization or hidden burn-in.

h[0]=1 is known to all predictors. At t, weights and h depend only on r[:t].
The Bayesian predictor sums over ALL grid parameters, not the sampled identity.
"""
import itertools
import numpy as np
from scipy.special import logsumexp


def parameter_grid(heldout=False):
    values = ((0.75, 1.5), (0.10,), (0.75,)) if heldout else ((0.5, 1., 2.), (0.05, 0.15), (0.70, 0.80))
    return np.array([((1-a-b)*v, a, b) for v,a,b in itertools.product(*values)], dtype=float)


def simulate(seed, batch, length, grid=None):
    grid = parameter_grid() if grid is None else np.asarray(grid)
    if np.any(grid[:,0] <= 0) or np.any(grid[:,1:] < 0) or np.any(grid[:,1:].sum(1) >= 1):
        raise ValueError("Invalid stationary GARCH grid")
    rng = np.random.default_rng(seed)
    ids = rng.integers(len(grid), size=batch)
    omega, alpha, beta = grid[ids].T
    h = np.ones(batch); r = np.empty((batch,length)); variance = np.empty_like(r)
    for t in range(length):
        variance[:,t] = h
        r[:,t] = np.sqrt(h)*rng.standard_normal(batch)
        h = omega + alpha*r[:,t]**2 + beta*h
    return r, variance, ids


def reference(r, grid=None, prior=None):
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or not np.isfinite(r).all():
        raise ValueError('r must be a finite (batch,time) array')
    grid = parameter_grid() if grid is None else np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 3 or np.any(grid[:,0]<=0) or np.any(grid[:,1:]<0) or np.any(grid[:,1:].sum(1)>=1):
        raise ValueError('Expected stationary GARCH grid with positive omega')
    prior = np.ones(len(grid))/len(grid) if prior is None else np.asarray(prior, dtype=float)
    if prior.shape != (len(grid),) or np.any(prior<=0) or not np.isfinite(prior).all():
        raise ValueError('Expected strictly positive prior probabilities')
    prior = prior/prior.sum()
    B,T = r.shape; J = len(grid)
    h = np.ones((B,J)); lw = np.broadcast_to(np.log(prior),(B,J)).copy()
    var = np.empty((B,T,J)); weights = np.empty_like(var); ld = np.empty((B,T))
    omega,alpha,beta = grid.T
    for t in range(T):
        var[:,t] = h; weights[:,t] = np.exp(lw)
        lp = -0.5*(np.log(2*np.pi*h)+r[:,t,None]**2/h)
        ld[:,t] = logsumexp(lw+lp,axis=1)
        lw += lp; lw -= logsumexp(lw,axis=1,keepdims=True)
        h = omega + alpha*r[:,t,None]**2 + beta*h
    return {'variance':var,'weights':weights,'logdensity':ld,
            'mean_variance':np.sum(var*weights,axis=-1)}
