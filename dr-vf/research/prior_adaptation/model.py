"""Finite GARCH likelihood bank and conventional population-prior estimators."""
import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, softmax


def make_bank():
    # nu=0 denotes Gaussian; positive nu denotes a unit-variance Student law.
    old = [((1-a-b)*v, a, b, 0.) for v, a, b in itertools.product(
        (.5, .75, 1., 1.5, 2.), (.05, .10, .15), (.70, .75, .80))]
    extra = [((1-p)*v, a, p-a, nu) for v, a, p, nu in itertools.product(
        (.5, 1., 2.), (.03, .10, .20), (.80, .94, .985, .995), (5., 8., 0.))]
    rows = list(old)
    for row in extra:
        if not any(np.allclose(row, x, rtol=0, atol=1e-12) for x in rows):
            rows.append(row)
    bank = np.asarray(rows)
    coarse = np.array([((1-a-b)*v, a, b, 0.) for v, a, b in itertools.product(
        (.5, 1., 2.), (.05, .15), (.70, .80))])
    anchor = np.any(np.all(np.isclose(bank[:, None], coarse[None], rtol=0,
                                    atol=1e-12), axis=-1), axis=1)
    dense = np.arange(len(bank)) < len(old)
    weights = {'coarse': anchor.astype(float)/anchor.sum(),
               'dense': dense.astype(float)/dense.sum(),
               'historical_90_10': np.where(anchor, .9/12,
                                           np.where(dense, .1/33, 0.)),
               'uniform_expanded': np.full(len(bank), 1/len(bank))}
    for mass in (.25, .5, .75):
        weights[f'blend_{mass:g}'] = ((1-mass)*weights['historical_90_10']
                                     + mass*weights['uniform_expanded'])
    return bank, weights


def validate_bank(bank):
    b = np.asarray(bank, float)
    if (b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all()
            or np.any(b[:, 0] <= 0) or np.any(b[:, 1:3] < 0)
            or np.any(b[:, 1:3].sum(1) >= 1)
            or np.any((b[:, 3] != 0) & (b[:, 3] <= 2))):
        raise ValueError('Expected stationary finite-variance GARCH atoms')
    return b


def log_law(r, h, nu):
    """Log density at r for zero mean and variance h; nu=0 is Gaussian."""
    h, nu = np.broadcast_arrays(np.asarray(h, float), np.asarray(nu, float))
    if np.any(h <= 0) or not np.isfinite(h).all():
        raise ValueError('Invalid predictive variance')
    out = -.5*(np.log(2*np.pi*h) + np.asarray(r)**2/h)
    student = nu != 0
    if np.any(student):
        n = nu[student]; v = h[student]; y = np.broadcast_to(r, h.shape)[student]
        out = np.array(out, copy=True)
        out[student] = (gammaln((n+1)/2)-gammaln(n/2)
                        -.5*np.log((n-2)*np.pi*v)
                        -(n+1)/2*np.log1p(y*y/((n-2)*v)))
    return out


def component_paths(returns, bank, h0=1.):
    b = validate_bank(bank); r = np.asarray(returns, float)
    if r.ndim != 1 or not np.isfinite(r).all() or h0 <= 0:
        raise ValueError('Invalid returns or initialization')
    h = np.full(len(b), h0, dtype=float)
    lp = np.empty((len(r), len(b))); variance = np.empty_like(lp)
    for t, y in enumerate(r):
        variance[t] = h
        lp[t] = log_law(y, h, b[:, 3])
        h = b[:, 0]+b[:, 1]*y*y+b[:, 2]*h
    return lp, variance


def log_prior(prior):
    w = np.asarray(prior, float)
    if w.ndim != 1 or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('Invalid prior')
    w = w/w.sum()
    with np.errstate(divide='ignore'):
        return np.log(w)


def predict(lp, prior):
    """Exactly marginalized one-step density and pre-observation atom weights."""
    before = np.vstack([np.zeros(lp.shape[1]), np.cumsum(lp, axis=0)[:-1]])
    lw = before + log_prior(prior)
    lw -= logsumexp(lw, axis=1, keepdims=True)
    return logsumexp(lw+lp, axis=1), np.exp(lw)


def marginal_fit(end, weight, strength=1., maxiter=3000):
    """Concave finite-mixture ML with Dirichlet pseudo-count regularization."""
    end, weight = np.asarray(end), np.asarray(weight)
    j = end.shape[1]; uniform = np.full(j, 1/j); w = uniform.copy()
    old = -np.inf
    for iteration in range(maxiter):
        resp = softmax(end+np.log(w), axis=1)
        nxt = (weight @ resp + strength*uniform)/(weight.sum()+strength)
        value = float(weight @ logsumexp(end+np.log(nxt), axis=1)
                      + strength*np.mean(np.log(nxt)))
        change = float(np.max(np.abs(nxt-w)))
        if value < old-1e-7:
            raise ArithmeticError('Marginal-likelihood EM decreased its objective')
        w = nxt
        if change < 1e-7:
            break
        old = value
    # Derivative with respect to simplex weights: equal to the Lagrange multiplier.
    resp = softmax(end+np.log(w), axis=1)
    derivative = (weight @ resp + strength*uniform)/w
    residual = float(np.max(np.abs(derivative-(weight.sum()+strength))))
    return w, {'converged': bool(change < 1e-7 and residual < .01),
               'iterations': iteration+1, 'objective': value,
               'max_weight_change': change, 'simplex_stationarity_residual': residual}


def predictive_objective(logits, end, before, weight, strength=1.):
    lw = logits-logsumexp(logits); w = np.exp(lw)
    a, b = end+lw, before+lw
    value = weight @ (logsumexp(a, axis=1)-logsumexp(b, axis=1))
    value += strength*np.mean(lw)
    grad = weight @ (softmax(a, axis=1)-softmax(b, axis=1))
    grad += strength*(1/len(w)-w)
    scale = max(1., weight.sum())
    return -float(value)/scale, -grad/scale


def predictive_fit(end, before, weight, starts, strength=1.):
    runs = []
    for name, w in starts.items():
        result = minimize(predictive_objective, np.log(w),
                          args=(end, before, weight, strength), method='L-BFGS-B',
                          jac=True, options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-7})
        stationary = float(np.max(np.abs(result.jac)))
        runs.append((softmax(result.x), {'start': name, 'objective': float(result.fun),
                    'converged': bool(result.success and stationary < 1e-4),
                    'iterations': int(result.nit), 'gradient_max': stationary,
                    'message': str(result.message)}))
    valid = [x for x in runs if x[1]['converged']]
    chosen = min(valid or runs, key=lambda x: x[1]['objective'])
    return chosen[0], {'converged': chosen[1]['converged'], 'selected_start': chosen[1]['start'],
                       'runs': [x[1] for x in runs]}
