"""Online, deterministic calibeaters for the (VaR, ES) pair.

Both wrappers take a black-box forecaster's (v_t, e_t) and past outcomes only, and emit a new pair
before y_t is observed. Nothing here uses the future.

RatioCalibeater   one group, in the forecaster's own units: standardise outcomes by |v_t|, track the
                  empirical (VaR, ES) of the standardised sample (follow-the-leader) or a
                  stochastic-approximation tracker of the identification function (adaptive), map
                  back through |v_t|. Corrects a mis-specified tail law and a scale bias with two
                  numbers; this is the practical wrapper for finance.
BinCalibeater     Foster-Hart style: K groups defined by the level of the forecaster's v_t, each
                  holding the empirical (VaR, ES) of the raw outcomes seen in that group. Model-free,
                  needs alpha * T / K tail events per group to say anything.
"""
import bisect
import numpy as np
from .fz import empirical_var_es


class _Sample:
    """A sorted sample with O(log n) insertion, optional sliding window, and empirical (VaR, ES)
    queries computed on the sorted list without re-sorting."""

    def __init__(self, window=None):
        self.z = []; self.window = window; self.order = []

    def add(self, x):
        x = float(x); bisect.insort(self.z, x)
        if self.window is not None:
            self.order.append(x)
            if len(self.order) > self.window:
                old = self.order.pop(0); del self.z[bisect.bisect_left(self.z, old)]

    def __len__(self):
        return len(self.z)

    def var_es(self, alpha):
        n = len(self.z); k = alpha * n; m = int(k)
        if m == 0:
            return self.z[0], self.z[0]
        v = self.z[m - 1] if k == m else self.z[m]
        es = (sum(self.z[:m]) + (k - m) * (self.z[m] if m < n else self.z[m - 1])) / k
        return v, min(es, v)


class _Tracker:
    """Stochastic approximation on the pair's identification function: v moves against the hit
    indicator (a quantile tracker, as in adaptive conformal inference), e moves toward the
    hit-weighted tail contribution. Step eta sets the memory (~1/eta observations)."""

    def __init__(self, alpha, eta, v0, e0):
        self.alpha, self.eta, self.v, self.e = alpha, eta, float(v0), float(e0)

    def update(self, z):
        hit = 1.0 if z <= self.v else 0.0
        target_e = self.v + hit * (z - self.v) / self.alpha
        self.v += self.eta * (self.alpha - hit) * (-self.v if self.v < 0 else 1.0)   # scale-aware quantile step
        self.e += self.eta * (target_e - self.e)
        self.e = min(self.e, self.v - 1e-12)


class RatioCalibeater:
    """mode 'ftl': empirical pair of all past standardised outcomes; 'window': the same over the last
    `window` outcomes (a drifting-tail variant); 'sa': the stochastic-approximation tracker."""

    def __init__(self, alpha, warm=250, mode='ftl', eta=0.002, window=None):
        self.alpha, self.warm, self.mode, self.eta, self.window = alpha, warm, mode, eta, window

    def run(self, y, v, e):
        y = np.asarray(y, float); v = np.asarray(v, float); e = np.asarray(e, float); T = len(y)
        vh = v.copy(); eh = e.copy(); sample = _Sample(self.window if self.mode == 'window' else None); tracker = None
        for t in range(T):
            s = abs(v[t]) if v[t] < 0 else max(abs(v[t]), 1e-8)
            if len(sample) >= self.warm:
                if self.mode in ('ftl', 'window'):
                    q, es = sample.var_es(self.alpha)
                else:
                    if tracker is None:
                        q0, es0 = sample.var_es(self.alpha); tracker = _Tracker(self.alpha, self.eta, q0, es0)
                    q, es = tracker.v, tracker.e
                vh[t] = q * s; eh[t] = min(es * s, vh[t] - 1e-12)
            z = y[t] / s
            sample.add(z)
            if tracker is not None:
                tracker.update(z)
        return vh, eh


class BinCalibeater:
    def __init__(self, alpha, K=5, warm=250, min_points=None):
        self.alpha, self.K, self.warm = alpha, K, warm
        self.min_points = min_points if min_points is not None else max(int(2.0 / alpha), 20)

    def run(self, y, v, e):
        y = np.asarray(y, float); v = np.asarray(v, float); e = np.asarray(e, float); T = len(y)
        vh = v.copy(); eh = e.copy(); edges = None; samples = None; groups = np.zeros(T, int)
        for t in range(T):
            if edges is None and t >= self.warm:
                edges = np.quantile(v[:t], np.linspace(0, 1, self.K + 1)[1:-1]) if self.K > 1 else np.array([])
                samples = [_Sample() for _ in range(self.K)]
                for u in range(t):                          # back-fill the groups with the warm-up outcomes
                    g = int(np.searchsorted(edges, v[u])); samples[g].add(y[u]); groups[u] = g
            if edges is not None:
                g = int(np.searchsorted(edges, v[t])); groups[t] = g
                if len(samples[g]) >= self.min_points:
                    q, es = samples[g].var_es(self.alpha); vh[t] = q; eh[t] = min(es, q - 1e-12)
                samples[g].add(y[t])
        self.groups = groups
        return vh, eh
