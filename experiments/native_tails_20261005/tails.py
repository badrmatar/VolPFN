"""Lower-tail quantiles and partial expectations of a density mixture."""
import numpy as np
from scipy import integrate, special, stats


def validate(weights, location, scale, df):
    arrays = [np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in (weights, location, scale, df)]
    w, mu, s, nu = arrays
    if any(x.shape != w.shape for x in arrays):
        raise ValueError('Mixture arrays must have equal shape')
    if not all(np.isfinite(x).all() for x in (w, mu, s)) or np.isnan(nu).any():
        raise ValueError('Invalid mixture parameters')
    if (w < 0).any() or (s <= 0).any() or (nu <= 1).any():
        raise ValueError('Weights, scales or first moments invalid')
    if not np.allclose(w.sum(axis=1), 1, atol=1e-12, rtol=1e-12):
        raise ValueError('Weights must sum to one')
    return arrays


def component_cdf(z, nu):
    return np.where(np.isinf(nu), special.ndtr(z), special.stdtr(nu, z))


def component_pdf(z, nu):
    finite = np.isfinite(nu)
    safe = np.where(finite, nu, 5.)
    log_t = special.gammaln((safe+1)/2)-special.gammaln(safe/2)-.5*np.log(safe*np.pi)-.5*(safe+1)*np.log1p(z*z/safe)
    return np.where(finite, np.exp(log_t), np.exp(-z*z/2)/np.sqrt(2*np.pi))


def var_es(weights, location, scale, df, alpha):
    w, mu, s, nu = validate(weights, location, scale, df)
    if not 0 < alpha < .5:
        raise ValueError('Expected a lower-tail probability')
    component_q = mu + s * np.where(np.isinf(nu), stats.norm.ppf(alpha), stats.t.ppf(alpha, nu))
    lo, hi = component_q.min(axis=1), component_q.max(axis=1)
    for _ in range(160):
        mid = (lo+hi)/2
        cdf = np.sum(w*component_cdf((mid[:,None]-mu)/s, nu), axis=1)
        lo = np.where(cdf < alpha, mid, lo)
        hi = np.where(cdf >= alpha, mid, hi)
        if np.max(hi-lo) < 1e-11:
            break
    else:
        raise ArithmeticError('Mixture quantile did not converge')
    q = (lo+hi)/2
    z = (q[:,None]-mu)/s
    cdf = component_cdf(z, nu)
    safe = np.where(np.isinf(nu), 5., nu)
    factor = np.where(np.isinf(nu), 1., (safe+z*z)/(safe-1))
    es = np.sum(w*(mu*cdf - s*factor*component_pdf(z,nu)),axis=1)/alpha
    residual = np.abs(np.sum(w*cdf,axis=1)-alpha)
    if np.max(residual) > 1e-9 or not np.isfinite(es).all() or (es > q+1e-9).any():
        raise ArithmeticError('Invalid quantile or partial expectation')
    return q, es, residual


def numerical_es(w, mu, s, nu, alpha, q):
    """Independent quadrature in each component's standardised coordinates."""
    total, error = 0., 0.
    for weight, location, scale, df in zip(w, mu, s, nu):
        zq = (q-location)/scale
        if np.isinf(df):
            def pdf(z):
                return np.exp(-z*z/2)/np.sqrt(2*np.pi)
        else:
            normalizer = np.exp(special.gammaln((df+1)/2)-special.gammaln(df/2)) / np.sqrt(df*np.pi)
            def pdf(z):
                return normalizer*(1+z*z/df)**(-(df+1)/2)
        value, err = integrate.quad(lambda z:(location+scale*z)*pdf(z), -np.inf, zq,
                                    epsabs=1e-10, epsrel=1e-10, limit=250)
        total += weight*value
        error += weight*err
    return total/alpha, error/alpha
