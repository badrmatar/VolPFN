"""Numerical checks for Theorem B (THEOREM_B.md): the exact excess-risk identity of FZ0, the
constant c_alpha of the O(log T) regret, Monte Carlo of the empirical pair's excess risk, and the
comparison with the sandbox's oracle cells (which are exactly the theorem's i.i.d. setting).

    python3 -m calibeat.theory_check
"""
import json
import numpy as np
from scipy import stats, integrate
from .fz import empirical_var_es, fz0


class Law:
    """Standardised-outcome law with the objects of the theorem: A(q) = E(q - z)^+, eps(q), R(q, eps)."""

    def __init__(self, name, nu=None):
        self.name = name; self.nu = nu
        self.dist = stats.t(nu) if nu else stats.norm()

    def A(self, q):
        if self.nu:                                   # partial expectation of Student-t: E z 1{z<=q} = -(nu+q^2)/(nu-1) f(q)
            return q * self.dist.cdf(q) + (self.nu + q * q) / (self.nu - 1) * self.dist.pdf(q)
        return q * self.dist.cdf(q) + self.dist.pdf(q)

    def eps(self, q, alpha):
        return q - self.A(q) / alpha

    def R(self, q, e, alpha):
        return (q - self.A(q) / alpha) / e + np.log(-e) - 1.0

    def optimum(self, alpha):
        q = self.dist.ppf(alpha); return q, self.eps(q, alpha)

    def c_alpha(self, alpha):
        q, e = self.optimum(alpha); f = self.dist.pdf(q)
        m2 = integrate.quad(lambda u: (q - u) ** 2 * self.dist.pdf(u), -np.inf, q)[0]
        var_plus = m2 - self.A(q) ** 2
        return (1 - alpha) / (2 * f * abs(e)) + var_plus / (2 * alpha ** 2 * e ** 2), (1 - alpha) / (2 * f * abs(e)), var_plus / (2 * alpha ** 2 * e ** 2)


def phi(x):
    return x - 1.0 - np.log(x)


def identity_check(law, alpha, rng, n=2000):
    """R(q, e) - R(q*, e*) == phi(eps(q) / e) + log(eps(q) / e*) for every (q, e) with e < 0."""
    qs, es = law.optimum(alpha); worst = 0.0
    for _ in range(n):
        q = qs + rng.normal(0, 0.5); e = min(law.eps(q, alpha) * np.exp(rng.normal(0, 0.4)), -1e-6)
        lhs = law.R(q, e, alpha) - law.R(qs, es, alpha); rhs = phi(law.eps(q, alpha) / e) + np.log(law.eps(q, alpha) / es)
        worst = max(worst, abs(lhs - rhs))
    return worst


def monte_carlo(law, alpha, ns, reps, rng):
    qs, es = law.optimum(alpha); Rstar = law.R(qs, es, alpha); out = {}
    for n in ns:
        z = law.dist.rvs(size=(reps, n), random_state=rng); ex = np.empty(reps)
        for i in range(reps):
            q, e = empirical_var_es(z[i], alpha); ex[i] = law.R(q, e, alpha) - Rstar
        out[n] = (float(n * ex.mean()), float(n * ex.std() / np.sqrt(reps)))
    return out


def sandbox_comparison(law, alpha, path='outputs/sandbox_v1/cells.jsonl', warm=250):
    c = law.c_alpha(alpha)[0]; rows = [json.loads(l) for l in open(path)]; out = []
    for T in sorted({r['T'] for r in rows}):
        pred = c * np.sum(1.0 / np.arange(warm, T)) / (T - warm)           # round t uses t past outcomes
        meas = {}
        for dgp in ['garch_t', 'regime_t']:
            v = [r['ratio_ftl'] - r['oracle'] for r in rows if r['dgp'] == dgp and r['forecaster'] == 'oracle' and r['alpha'] == alpha and r['T'] == T]
            meas[dgp] = (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) if v else (np.nan, np.nan)
        out.append((T, pred, meas))
    return out


def main():
    rng = np.random.default_rng(7); law = Law('t5', 5.0)
    print('Exact identity R(q,e) - R* = phi(eps(q)/e) + log(eps(q)/e*), worst abs error over random (q, e):')
    for alpha in (0.05, 0.025, 0.01):
        print(f'  t5 alpha={alpha}: {identity_check(law, alpha, rng):.2e}   normal: {identity_check(Law("normal"), alpha, rng):.2e}')
    print('\nConstant c_alpha = (1-alpha)/(2 f(q*)|e*|) + Var((q*-z)^+)/(2 alpha^2 e*^2)  [VaR term + ES term]:')
    for alpha in (0.05, 0.025, 0.01):
        c, cv, ce = law.c_alpha(alpha); cn = Law('normal').c_alpha(alpha)
        print(f'  alpha={alpha}: t5  c={c:.3f} (VaR {cv:.3f} + ES {ce:.3f})   normal c={cn[0]:.3f} (VaR {cn[1]:.3f} + ES {cn[2]:.3f})')
    print('\nMonte Carlo, i.i.d. t5: n x E[excess risk of the empirical pair] -> c_alpha  (mean +- se)')
    for alpha in (0.05, 0.025, 0.01):
        mc = monte_carlo(law, alpha, [250, 1000, 4000, 16000], 3000, rng); c = law.c_alpha(alpha)[0]
        print(f'  alpha={alpha}: c={c:.3f} | ' + ' | '.join(f'n={n}: {m:.3f}+-{s:.3f}' for n, (m, s) in mc.items()))
    print('\nSandbox oracle cells (ratio_ftl minus oracle, scored from t=250): predicted c_alpha * mean(1/t) vs measured')
    for alpha in (0.05, 0.025, 0.01):
        print(f'  alpha={alpha}:')
        for T, pred, meas in sandbox_comparison(law, alpha):
            print(f'    T={T:>6}: predicted {pred:.4f}   garch_t {meas["garch_t"][0]:+.4f}+-{meas["garch_t"][1]:.4f}   regime_t {meas["regime_t"][0]:+.4f}+-{meas["regime_t"][1]:.4f}')


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------- Theorem D: the price of binning
def group_resolution_loss(law, alpha, s):
    """min over a constant pair (v, e) of mean_t R(v / s_t, e / s_t) minus R*: the excess a single pair pays
    for the scale variation inside one group (Lemma 1 applied inside the group)."""
    from scipy.optimize import minimize
    qs, es = law.optimum(alpha); Rstar = law.R(qs, es, alpha); m = float(np.mean(s))

    def loss(th):
        v = -np.exp(th[0]); e = v - np.exp(th[1])
        return float(np.mean(law.R(v / s, e / s, alpha)))
    best = None
    for start in ([np.log(-qs * m), np.log((qs - es) * m)], [np.log(-qs * np.median(s)), np.log((qs - es) * np.median(s))]):
        r = minimize(loss, np.array(start), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-10, maxfev=2000))
        if best is None or r.fun < best.fun:
            best = r
    return float(best.fun - Rstar)


def rho_alpha(law, alpha):
    """Second-order coefficient: resolution loss ~ rho_alpha * Var_g(1/s) / E_g(1/s^2)."""
    q, e = law.optimum(alpha); return law.dist.pdf(q) * q * q / (2 * alpha * abs(e)) + 0.5


def binning_check(law, alpha, T=32000, seeds=(1, 2, 3, 4, 5), Ks=(1, 5, 10), warm=250, dgp='garch_t', path='outputs/sandbox_v1/cells.jsonl'):
    from . import sandbox as S
    rows = [json.loads(l) for l in open(path)]; c = law.c_alpha(alpha)[0]; rho = rho_alpha(law, alpha); out = {}
    for K in Ks:
        pred_h = []; pred_2 = []; meas_h = []; pred_on = []; meas_on = []
        for seed in seeds:
            y, sigma, qv, qe = S.simulate(dgp, T, seed, alpha); v = qv; s = np.abs(v); sc = np.arange(warm, T)
            edges = np.quantile(v[:warm], np.linspace(0, 1, K + 1)[1:-1]) if K > 1 else np.array([])
            g = np.searchsorted(edges, v); loss_h = 0.0; loss_2 = 0.0; est = 0.0
            for k in np.unique(g[sc]):
                idx = sc[g[sc] == k]; sk = s[idx]
                loss_h += len(idx) * group_resolution_loss(law, alpha, sk)
                u = 1.0 / sk; loss_2 += len(idx) * rho * (np.var(u) / np.mean(u * u))
                counts = np.cumsum(g == k)                      # past members of the group at each round
                est += np.sum(c / np.maximum(counts[idx] - 1, 1))
            pred_h.append(loss_h / len(sc)); pred_2.append(loss_2 / len(sc)); pred_on.append(est / len(sc))
            cell = [r for r in rows if r['forecaster'] == 'oracle' and r['dgp'] == dgp and r['alpha'] == alpha and r['T'] == T and r['seed'] == seed]
            if cell:
                meas_h.append(cell[0][f'hindsight_bins{K}'] - cell[0]['oracle']); meas_on.append(cell[0][f'bins{K}'] - cell[0][f'hindsight_bins{K}'])
        out[K] = dict(pred_hindsight=float(np.mean(pred_h)), pred_second_order=float(np.mean(pred_2)), meas_hindsight=float(np.mean(meas_h)) if meas_h else np.nan,
                      pred_online_minus_hindsight=float(np.mean(pred_on)), meas_online_minus_hindsight=float(np.mean(meas_on)) if meas_on else np.nan)
    return out


def main_binning():
    law = Law('t5', 5.0)
    print('Theorem D check on the sandbox oracle cells (garch_t, T = 32k, 5 seeds; groups built as the sandbox built them):')
    print('  hindsight K-group recalibration minus oracle: predicted from Lemma 1 on the realised scales vs measured;')
    print('  online minus hindsight: predicted sum_t c_alpha / n_g(t) vs measured')
    for alpha in (0.05, 0.025, 0.01):
        res = binning_check(law, alpha); print(f'  alpha={alpha}  (rho_alpha={rho_alpha(law, alpha):.3f}, c_alpha={law.c_alpha(alpha)[0]:.2f})')
        for K, r in res.items():
            print(f"    K={K:>2}: hindsight excess predicted {r['pred_hindsight']:.4f} (2nd order {r['pred_second_order']:.4f}) measured {r['meas_hindsight']:.4f}"
                  f" | online-hindsight predicted {r['pred_online_minus_hindsight']:.4f} measured {r['meas_online_minus_hindsight']:.4f}")


if __name__ == '__main__' and len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == 'binning':
    main_binning()
