#!/usr/bin/env python3
"""Online volatility forecasts from a mixture of score-driven filters.

UniVol combines a grid of EWMA experts over gain, Student-t degrees of freedom
and leverage. Each expert updates its log-variance using a bounded Student-t
score and a sign term for asymmetric responses. Follow-the-Leading-History
restart experts (Hazan & Seshadhri, 2009) start from the current ensemble forecast
and use a decreasing gain. Their lifetimes depend on the trailing zeros of the
birth time, keeping O(log t) restart experts active.

Fixed-Share weights combine the predictive densities. By default, the point
forecast is the density-weighted geometric mean of the component variances.
VaR is found numerically from the Student-t mixture CDF; ES uses the components'
truncated moments. Both are returned as positive losses.

Library use:
    f = UniVol()
    for r in returns:
        h = f.update(r)  # Forecast variance for the next period.
    q, es = f.tail(0.975)

Command-line use:
    python3 univol.py --data returns.csv --col EURUSD --out eurusd_forecasts.csv

The CSV contains variance, sigma, VaR, ES, effective nu and effective half-life.
The command also reports QLIKE, log score and VaR coverage. run_series() sets the
initial scale from the first 100 returns, so scoring needs an appropriate burn-in.
"""
import argparse, math, os, sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from scipy.special import gammaln, stdtr
from scipy.optimize import brentq


def logsumexp(x):
    x = np.asarray(x, dtype=float)
    m = x.max()
    return float(m + np.log(np.sum(np.exp(x - m)))) if np.isfinite(m) else float(m)


def t_logpdf(z, nu, _unused=None):
    """log density of the standard Student-t(nu) at z."""
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    )


def t_score(u, nu):
    """d/dtheta of the standardized Student-t(nu) log-likelihood, u = r^2 e^{-theta}; bounded in u."""
    return 0.5 * ((nu + 1.0) * u / (nu - 2.0 + u) - 1.0)


def t_nll2(u, h, nu, lg):
    """-2 log p(r | h, nu) for the unit-variance Student-t(nu) scale family, u = r^2/h, lg = normalising constant."""
    return np.log(h) + (nu + 1.0) * np.log1p(u / (nu - 2.0)) - 2.0 * lg


def lgam(nu):
    return gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi)


class UniVol:
    def __init__(
        self,
        gains=None,
        nus=(3.0, 4.5, 7.0, 12.0, 40.0),
        leverages=(0.0, -0.5, -1.0, -2.0),
        switch=None,
        use_flh=True,
        n0=2.0,
        eta=0.5,
        scale=None,
        skip_zero=True,
        floor_ratio=1e-4,
        point='density',
        clip_c=9.0,
    ):
        """gains: EWMA gains of the bank experts (default 12 log-spaced in [0.001, 0.5]); nus: Student-t laws;
        leverages: coefficients of the bounded sign term lam * b(z), b(z) = (nu+1) z / (nu - 2 + z^2), added to the
        log-variance update (negative = a negative return raises variance more than a positive one of the same size;
        0 = symmetric); switch: Fixed-Share switching rate (None = 1/(t+1)); use_flh: restart experts on/off;
        scale: initial variance (None = mean square of the first 10 returns); skip_zero: an exactly-zero return is
        treated as a missing observation (stale price) and does not update anything; floor_ratio: variance forecasts
        are floored at floor_ratio x a slow long-run variance (half-life 1000 steps), which stops collapse on runs of
        tiny returns; point: which weights build the point variance forecast -- 'density' (default: the
        predictive-density weights, i.e. the posterior of the switching mixture, applied as a geometric mean of the
        experts' variances), 'qlike' (exponential weights on the QLIKE loss r^2/h + log h with r^2 clipped at
        clip_c * h) or 'mean' (the mixture-mean variance).
        The predictive distribution, VaR and ES always use the density weights.
        """
        self.point = point
        self.clip_c = clip_c
        self.gains = (
            np.geomspace(1e-3, 0.5, 12) if gains is None else np.asarray(gains, dtype=float)
        )
        self.nus = np.asarray(nus, dtype=float)
        self.levs = np.asarray(leverages, dtype=float)
        G, N, Lv = np.meshgrid(self.gains, self.nus, self.levs, indexing='ij')
        self.bg = G.ravel()
        self.bnu = N.ravel()
        self.blev = Lv.ravel()
        self.blg = lgam(self.bnu)
        self.bnewton = 2.0 * (self.bnu + 3.0) / self.bnu
        self.nb = len(self.bg)
        self.switch = switch  # None -> 1/(t+1) (Fixed-Share with vanishing switching rate)
        self.use_flh = use_flh
        self.n0 = n0
        self.eta = eta
        self.skip_zero = skip_zero
        self.floor_ratio = floor_ratio
        self.t = 0
        self.scale = scale  # scale: initial variance guess; None -> set from the first 10 returns
        self.h_floor = None
        self.lr_var = None
        self.lr_gain = 1.0 - 0.5 ** (1.0 / 1000.0)
        self.theta = None
        self.logw = None  # bank log-variances and log-weights (predictive density)
        self.logq = None
        self.flogq = np.zeros(0)
        self.topq = np.log(np.array([0.5, 0.5]))  # QLIKE weights for the point forecast
        self.fth = np.zeros(0)
        self.fn = np.zeros(0)
        self.fnu = np.zeros(0)
        self.flg = np.zeros(0)  # FLH restart experts
        self.fdeath = np.zeros(0, dtype=np.int64)
        self.flogv = np.zeros(0)
        self.top = np.log(np.array([0.5, 0.5]))  # top-level weights over {bank, FLH}
        self.warm = []  # first returns used to set the scale
        self._mix = None  # cached predictive mixture (weights, variances, nus)

    # predictive mixture and its summaries
    def _mixture(self):
        """weights, variances and nus of all components of the current predictive distribution (cached per step)."""
        if self._mix is None:
            wb = np.exp(self.logw - logsumexp(self.logw))
            hb = np.exp(self.theta)
            if self.use_flh and len(self.fth):
                wf = np.exp(self.flogv - logsumexp(self.flogv))
                tw = np.exp(self.top - logsumexp(self.top))
                self._mix = (
                    np.concatenate([tw[0] * wb, tw[1] * wf]),
                    np.concatenate([hb, np.exp(self.fth)]),
                    np.concatenate([self.bnu, self.fnu]),
                )
            else:
                self._mix = (wb, hb, self.bnu)
        return self._mix

    def predict(self):
        """Return mixture variance, point variance, effective nu and half-life.

        The default point forecast is the density-weighted geometric mean of the
        component variances. point='qlike' uses QLIKE weights; point='mean' returns
        the arithmetic mixture variance. update() returns the point forecast.
        """
        w, h, nu = self._mixture()
        h_mean = float(np.dot(w, h))
        if self.point == 'mean':
            h_pt = h_mean
        else:
            wq = self._point_weights() if self.point == 'qlike' else w
            h_pt = float(np.exp(np.dot(wq, np.log(h))))
        wb = np.exp(self.logw - logsumexp(self.logw))
        nu_eff = float(np.exp(np.dot(w, np.log(nu))))
        g_eff = float(np.exp(np.dot(wb, np.log(self.bg))))
        self.lev_eff = float(np.dot(wb, self.blev))
        return h_mean, h_pt, nu_eff, math.log(0.5) / math.log(1.0 - g_eff)

    def _pt_h_bank(self):
        qb = np.exp(self.logq - logsumexp(self.logq))
        return float(np.exp(np.dot(qb, self.theta)))

    def _point_weights(self):
        """weights over all components built from the clipped-QLIKE exponential weights (bank, FLH, top level)."""
        qb = np.exp(self.logq - logsumexp(self.logq))
        if self.use_flh and len(self.fth):
            qf = np.exp(self.flogq - logsumexp(self.flogq))
            tq = np.exp(self.topq - logsumexp(self.topq))
            return np.concatenate([tq[0] * qb, tq[1] * qf])
        return qb

    def tail(self, level=0.975):
        """one-step VaR and ES at `level` of the predictive mixture, as positive loss numbers (loss = -return)."""
        w, h, nu = self._mixture()
        s = np.sqrt(h * (nu - 2.0) / nu)  # scale of each t component (unit-variance convention)
        a = 1.0 - level
        cdf = lambda x: float(np.dot(w, stdtr(nu, x / s)))
        lo = -20.0 * math.sqrt(float(np.dot(w, h)))
        hi = 0.0
        while cdf(lo) > a:
            lo *= 2.0
        q = brentq(lambda x: cdf(x) - a, lo, hi, xtol=1e-10)  # lower-tail quantile of the return
        z = q / s
        partial = (
            -s * (nu + z * z) / (nu - 1.0) * np.exp(t_logpdf(z, nu, None))
        )  # int_{-inf}^q x f(x) dx per component
        es = float(np.dot(w, partial)) / a  # E[r | r < q]
        return -q, -es

    def log_score(self, r):
        """log predictive density of a return r under the current mixture (before updating)."""
        w, h, nu = self._mixture()
        s = np.sqrt(h * (nu - 2.0) / nu)
        return logsumexp(np.log(np.maximum(w, 1e-300)) + t_logpdf(r / s, nu, None) - np.log(s))

    # the update
    def _start(self):
        v = (
            self.scale
            if self.scale is not None
            else (float(np.mean(np.square(self.warm))) or 1e-12)
        )
        self.lr_var = v
        self.h_floor = self.floor_ratio * v
        self.theta = np.full(self.nb, math.log(v))
        self.logw = np.full(self.nb, -math.log(self.nb))
        self.logq = self.logw.copy()

    def update(self, r):
        """feed the return of the period just ended; returns the variance forecast for the next period."""
        if self.theta is None:
            self.warm.append(float(r))
            if self.scale is None and len(self.warm) < 10:
                return float(np.mean(np.square(self.warm))) if self.warm else float('nan')
            self._start()
            for x in self.warm[:-1]:
                self._step(x)
            self.warm = []
        self._step(float(r))
        return self.predict()[1]

    def _step(self, r):
        if self.skip_zero and r == 0.0:
            return  # stale price: no information, no update
        self.t += 1
        t = self.t
        x2 = r * r
        self.lr_var += self.lr_gain * (x2 - self.lr_var)
        self.h_floor = self.floor_ratio * self.lr_var
        # bank: predictive log-losses of r (computed BEFORE updating)
        wb_log = self.logw - logsumexp(self.logw)
        hb = np.exp(self.theta)
        ub = x2 / hb
        loss_b = t_nll2(ub, hb, self.bnu, self.blg)
        ll_bank = logsumexp(
            wb_log - 0.5 * loss_b
        )  # log predictive density of the bank mixture (up to a constant)
        bt = 1.0 / (t + 1.0) if self.switch is None else self.switch
        ql_b = np.minimum(ub, self.clip_c) + np.log(hb)  # clipped QLIKE loss of each bank expert
        qb_log = self.logq - logsumexp(self.logq)
        ql_bank = float(
            np.minimum(x2 / self._pt_h_bank(), self.clip_c) + math.log(self._pt_h_bank())
        )  # clipped QLIKE of the bank's point forecast
        if self.use_flh:
            # birth of a restart expert at the ensemble's current (geometric) forecast, then deaths
            h_ens = self.predict()[1] if t > 1 else float(hb.mean())
            k = (t & -t).bit_length() - 1
            nu_new = float(self.nus[(t - 1) % len(self.nus)])
            if len(self.fth):
                self.flogv = self.flogv + math.log(1.0 - 1.0 / t)
                self.flogq = self.flogq + math.log(1.0 - 1.0 / t)
                lv_new = math.log(1.0 / t)
            else:
                lv_new = 0.0
            self.fth = np.append(self.fth, math.log(max(h_ens, self.h_floor)))
            self.fn = np.append(self.fn, 0.0)
            self.fnu = np.append(self.fnu, nu_new)
            self.flg = np.append(self.flg, float(lgam(nu_new)))
            self.fdeath = np.append(self.fdeath, t + (1 << (k + 2)) + 1)
            self.flogv = np.append(self.flogv, lv_new)
            self.flogq = np.append(self.flogq, lv_new)
            keep = self.fdeath > t
            if not keep.all():
                self.fth, self.fn, self.fnu, self.flg, self.fdeath = (
                    self.fth[keep],
                    self.fn[keep],
                    self.fnu[keep],
                    self.flg[keep],
                    self.fdeath[keep],
                )
                self.flogv, self.flogq = self.flogv[keep], self.flogq[keep]
            self.flogv = self.flogv - logsumexp(self.flogv)
            self.flogq = self.flogq - logsumexp(self.flogq)
            hf = np.exp(self.fth)
            uf = x2 / hf
            loss_f = t_nll2(uf, hf, self.fnu, self.flg)
            ll_flh = logsumexp(self.flogv - 0.5 * loss_f)
            ql_f = np.minimum(uf, self.clip_c) + np.log(hf)
            hq_f = float(np.exp(np.dot(np.exp(self.flogq), np.log(hf))))
            ql_flh = float(min(x2 / hq_f, self.clip_c) + math.log(hq_f))
            # top-level Fixed-Share over {bank, FLH}: density weights with the two mixtures' log-densities, QLIKE weights with the two point forecasts
            self.top = self.top + np.array([ll_bank, ll_flh])
            self.top -= logsumexp(self.top)
            self.top = np.logaddexp(math.log(1.0 - bt) + self.top, math.log(bt / 2.0))
            self.topq = self.topq - self.eta * np.array([ql_bank, ql_flh])
            self.topq -= logsumexp(self.topq)
            self.topq = np.logaddexp(math.log(1.0 - bt) + self.topq, math.log(bt / 2.0))
            # FLH weights and states
            self.flogv = self.flogv - self.eta * loss_f
            self.flogv -= logsumexp(self.flogv)
            self.flogq = self.flogq - self.eta * ql_f
            self.flogq -= logsumexp(self.flogq)
            self.flogq = np.logaddexp(
                math.log(1.0 - bt) + self.flogq, math.log(bt / len(self.flogq))
            )
            self.fth = np.maximum(
                self.fth
                + (2.0 * (self.fnu + 3.0) / self.fnu)
                * t_score(uf, self.fnu)
                / (self.fn + 1.0 + self.n0),
                math.log(self.h_floor),
            )
            self.fn += 1.0
        # bank weights: Fixed-Share with the predictive log-loss; bank states: score-driven EWMA in log-variance
        self.logw = wb_log - self.eta * loss_b
        self.logw -= logsumexp(self.logw)
        self.logw = np.logaddexp(math.log(1.0 - bt) + self.logw, math.log(bt / self.nb))
        self.logq = qb_log - self.eta * ql_b
        self.logq -= logsumexp(self.logq)
        self.logq = np.logaddexp(math.log(1.0 - bt) + self.logq, math.log(bt / self.nb))
        z = np.sign(r) * np.sqrt(ub)
        lev = self.blev * (self.bnu + 1.0) * z / (self.bnu - 2.0 + z * z)
        self.theta = np.maximum(
            self.theta + self.bg * (self.bnewton * t_score(ub, self.bnu) + lev),
            math.log(self.h_floor),
        )
        self._mix = None


# batch helpers
def initial_scale(r, k=100):
    """variance scale from the first k returns (mean square of the non-zero ones), floored at 1e-12."""
    x = np.asarray(r[:k], dtype=float)
    x = x[x != 0]
    return float(np.mean(x * x)) if len(x) else 1e-12


def run_series(r, levels=(0.975,), **kw):
    """Run the filter over a whole series. Returns dict of arrays aligned with r: h (point forecast = geometric mean,
    made before r_t), h_mean (mixture mean), nu_eff, halflife, logscore, and VaR/ES per level.  The initial variance
    scale is taken from the first 100 returns (a look-ahead of the scale only; evaluate from a burn-in).
    """
    r = np.asarray(r, dtype=float)
    T = len(r)
    kw.setdefault('scale', initial_scale(r))
    f = UniVol(**kw)
    out = {k: np.full(T, np.nan) for k in ('h', 'h_mean', 'nu_eff', 'halflife', 'logscore')}
    for lv in levels:
        out[f'var{lv:g}'] = np.full(T, np.nan)
        out[f'es{lv:g}'] = np.full(T, np.nan)
    for t in range(T):
        if f.theta is not None:
            hm, hg, nu, hl = f.predict()
            out['h'][t], out['h_mean'][t], out['nu_eff'][t], out['halflife'][t] = hg, hm, nu, hl
            out['logscore'][t] = f.log_score(r[t])
            for lv in levels:
                q, es = f.tail(lv)
                out[f'var{lv:g}'][t], out[f'es{lv:g}'][t] = q, es
        f.update(r[t])
    return out


def qlike(h, r2):
    return r2 / h + np.log(h)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--data', required=True)
    ap.add_argument('--col', default=None)
    ap.add_argument('--prices', action='store_true')
    ap.add_argument('--levels', type=float, nargs='*', default=[0.975])
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-flh', action='store_true')
    ap.add_argument(
        '--demean', action='store_true', help='subtract a slow EWMA mean (half-life 250) causally'
    )
    a = ap.parse_args()
    import pandas as pd

    df = pd.read_csv(a.data)
    df = df.set_index(df.columns[0])
    col = a.col or df.columns[0]
    x = pd.to_numeric(df[col], errors='coerce').dropna()
    r = (x.pct_change().dropna() if a.prices else x).to_numpy(dtype=float)
    idx = x.index[1:] if a.prices else x.index
    if a.demean:
        m = 0.0
        g = 1 - 0.5 ** (1 / 250)
        rd = np.empty_like(r)
        for t in range(len(r)):
            rd[t] = r[t] - m
            m = m + g * (r[t] - m)
        r = rd
    out = run_series(r, levels=a.levels, use_flh=not a.no_flh)
    ok = ~np.isnan(out['h'])
    r2 = r * r
    print(
        f'{col}: {len(r)} returns; scale from the first 100 returns; realised QLIKE {np.mean(qlike(out["h"][ok], r2[ok])):.4f} (point forecast), '
        f'{np.mean(qlike(out["h_mean"][ok], r2[ok])):.4f} (mixture mean); mean log score {np.nanmean(out["logscore"]):.4f}'
    )
    for lv in a.levels:
        hit = -r[ok] > out[f'var{lv:g}'][ok]
        print(
            f'  VaR {lv:g}: violation rate {hit.mean():.4f} (nominal {1-lv:.4f}); mean ES forecast {np.nanmean(out[f"es{lv:g}"]):.4g}'
        )
    print(
        f'  effective nu (median) {np.nanmedian(out["nu_eff"]):.1f}; effective half-life (median) {np.nanmedian(out["halflife"]):.1f} steps'
    )
    if a.out:
        res = pd.DataFrame(
            {
                'return': r,
                'variance': out['h'],
                'sigma': np.sqrt(out['h']),
                **{k: v for k, v in out.items() if k.startswith(('var', 'es'))},
                'nu_eff': out['nu_eff'],
                'halflife': out['halflife'],
            },
            index=idx,
        )
        res.to_csv(a.out)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
