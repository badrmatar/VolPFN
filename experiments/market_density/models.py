"""Causal HAR path controls. Inputs contain observations strictly before origin."""
import numpy as np
from arch.univariate import LS, GARCH, Normal

METHODS = ('bayes_loghar', 'bootstrap_loghar', 'gaussian_loghar_garch',
           'bootstrap_loghar_garch', 'gaussian_sqrt_har_garch')
PROBS = np.array([.01, .025, .5, .975, .99])
HORIZONS = np.array([1, 5, 22])


def design(z):
    w = np.lib.stride_tricks.sliding_window_view(z, 23)
    return np.column_stack([np.ones(len(w)), w[:, 21],
                            w[:, 17:22].mean(1), w[:, :22].mean(1)]), w[:, 22]


def feature(z):
    return np.array([1., z[-1], np.mean(z[-5:]), np.mean(z[-22:])])


def garch_next(theta, residual, variance):
    return theta[0] + theta[1] * residual ** 2 + theta[2] * variance


def fit_garch(z):
    x, y = design(z)
    model = LS(y, x[:, 1:], volatility=GARCH(1, 0, 1),
               distribution=Normal(), rescale=False)
    result = model.fit(disp='off', show_warning=False, options={'maxiter': 500})
    retried = False
    if result.convergence_flag:
        retried = True
        result = model.fit(disp='off', show_warning=False,
                           starting_values=result.params, options={'maxiter': 2000})
    if result.convergence_flag:
        raise RuntimeError('GARCH optimization failed: ' + str(result.optimization_result))
    params = np.asarray(result.params)
    resids = np.asarray(result.resid)
    h = np.asarray(result.conditional_volatility) ** 2
    standardized = resids / np.sqrt(h)
    standardized -= standardized.mean()
    standardized /= np.sqrt(np.mean(standardized ** 2))
    return dict(b=params[:4], theta=params[4:7], h=float(h[-1]),
                residual=float(resids[-1]), standardized=standardized,
                convergence=int(result.convergence_flag), retried=retried,
                loglikelihood=float(result.loglikelihood))


def fit(history, window=1000):
    values = np.asarray(history[-window-22:], dtype=float)
    if len(values) != window + 22 or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError('Need window + 22 positive finite historical observations')
    logs = np.log(values)
    center = float(logs.mean())
    z = logs - center
    x, y = design(z)
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ b
    residuals -= residuals.mean()
    p0 = .001 * np.eye(4)
    b0 = np.array([0., .3, .3, .3])
    pn = p0 + x.T @ x
    bn = np.linalg.solve(pn, p0 @ b0 + x.T @ y)
    rate = .0625 + .5 * (np.sum((y - x @ bn) ** 2) + (bn-b0) @ p0 @ (bn-b0))
    roots = np.sqrt(values)
    root_center, root_scale = float(roots.mean()), float(roots.std())
    if root_scale <= 0:
        raise ValueError('Constant series')
    return dict(center=center, b=b, residuals=residuals, bn=bn,
                chol=np.linalg.cholesky(np.linalg.inv(pn)),
                shape=2 + len(y)/2, rate=rate, log_garch=fit_garch(z),
                root_center=root_center, root_scale=root_scale,
                root_garch=fit_garch((roots-root_center)/root_scale),
                scale=float(values[-window:].mean()),
                stress_threshold=float(np.quantile(
                    np.lib.stride_tricks.sliding_window_view(values[-window:], 5).mean(1), .9)),
                observed=len(history))


def advance(state, history):
    """Update residual variance for newly observed records without refitting."""
    for t in range(state['observed'], len(history)):
        for key, z in [('log_garch', np.log(history[t-22:t+1])-state['center']),
                       ('root_garch', (np.sqrt(history[t-22:t+1])-state['root_center'])/state['root_scale'])]:
            g = state[key]
            g['h'] = float(garch_next(g['theta'], g['residual'], g['h']))
            g['residual'] = float(z[-1] - feature(z[:-1]) @ g['b'])
    state['observed'] = len(history)


def simulate(state, history, method, rng, n=4096, innovations=None, return_daily=False):
    """One parameter draw per path; rolling feature sums avoid repeated reductions."""
    if method not in METHODS:
        raise ValueError(method)
    root = method == 'gaussian_sqrt_har_garch'
    z0 = ((np.sqrt(history[-22:])-state['root_center'])/state['root_scale']
          if root else np.log(history[-22:])-state['center'])
    z = np.empty((n, 44))
    z[:, :22] = z0
    total22 = np.full(n, z0.sum())
    total5 = np.full(n, z0[-5:].sum())
    g = None
    if method == 'bayes_loghar':
        var = state['rate'] / rng.gamma(state['shape'], size=n)
        b = state['bn'] + (rng.normal(size=(n, 4)) @ state['chol'].T) * np.sqrt(var[:, None])
    elif method == 'bootstrap_loghar':
        b = state['b']
        var = None
    else:
        g = state['root_garch' if root else 'log_garch']
        b = g['b']
        var = np.full(n, garch_next(g['theta'], g['residual'], g['h']))
    for j in range(22):
        t = 22+j
        if b.ndim == 2:
            mu = b[:, 0]+b[:, 1]*z[:, t-1]+b[:, 2]*total5/5+b[:, 3]*total22/22
        else:
            mu = b[0]+b[1]*z[:, t-1]+b[2]*total5/5+b[3]*total22/22
        if innovations is not None:
            residual = innovations[:, j] if var is None else np.sqrt(var)*innovations[:, j]
        elif method == 'bootstrap_loghar':
            residual = rng.choice(state['residuals'], n)
        elif method == 'bootstrap_loghar_garch':
            residual = np.sqrt(var)*rng.choice(g['standardized'], n)
        else:
            residual = np.sqrt(var)*rng.normal(size=n)
        nxt = mu+residual
        total22 += nxt-z[:, t-22]
        total5 += nxt-z[:, t-5]
        z[:, t] = nxt
        if g is not None:
            var = garch_next(g['theta'], residual, var)
    if root:
        signed = z[:, 22:]*state['root_scale']+state['root_center']
        negative = float(np.mean(np.any(signed < 0, axis=1)))
        daily = signed**2
    else:
        negative = 0.
        with np.errstate(over='raise', under='raise', invalid='raise'):
            daily = np.exp(z[:, 22:]+state['center'])
    if not np.all(np.isfinite(daily) & (daily > 0)):
        raise FloatingPointError('Invalid simulated variance; no clipping permitted')
    totals = np.cumsum(daily, axis=1)[:, HORIZONS-1]
    return (daily if return_daily else totals), negative


def score(quantiles, truth, scale):
    error = truth[:, None]-quantiles
    pin = np.maximum(PROBS*error, (PROBS-1)*error).mean(1)/(HORIZONS*scale)
    return np.column_stack([pin, (truth >= quantiles[:, 1]) & (truth <= quantiles[:, 3]),
                            truth < quantiles[:, 0], truth > quantiles[:, 4],
                            (quantiles[:, 3]-quantiles[:, 1])/(HORIZONS*scale)])
