"""The adaptive guard (Theorem E): follow-the-leader over a grid of windows, compared over the last
W days, with the original as the default.

Experts: the original pair and the ratio wrapper with windows in WINDOWS (None = full history), all
from the same standardised history. Each day the guard plays the expert whose cumulative FZ0 over
the last W days is lowest, provided it beats the original by more than `margin` per day; otherwise
the original. The selection is predictable (Theorem B(iii) applies) and uses unclipped FZ0
differences, which by homogeneity are scale-free.

Why not exponential weights: FZ0 loss differences are heavy-tailed and the information about
calibration sits in the tail days; clipping them (which the fixed-share regret bound needs) removes
exactly that information, and in the sandbox the clipped exponential-weights guard stayed with a
mis-specified RiskMetrics box (+0.142 at 1%) where the FTL-window guard captured the gain
(+0.037 vs the best expert's +0.025). `fixed_share_select` is kept for reference.
"""
import numpy as np
from .calibeater import RatioCalibeater
from .fz import fz0

WINDOWS = (500, 1000, 2000, 4000, None)
REGRET_BUDGET = {0.05: 0.0035, 0.025: 0.006, 0.01: 0.013}


def expert_name(W):
    return 'ftl' if W is None else f'win{W}'


def expert_paths(y, v, e, alpha, warm=250, windows=WINDOWS):
    paths = {'original': (np.asarray(v, float), np.asarray(e, float))}
    for W in windows:
        cb = RatioCalibeater(alpha, warm=warm, mode='ftl' if W is None else 'window', window=W)
        paths[expert_name(W)] = cb.run(y, v, e)
    return paths


def ftl_window_select(L, start, W=2000, margin=0.0):
    """L: (T, K) daily losses, column 0 the original. Day t's choice uses days max(0, t-W)..t-1 only."""
    L = np.asarray(L, float); T, K = L.shape
    cs = np.vstack([np.zeros((1, K)), np.cumsum(L, axis=0)]); choice = np.zeros(T, int)
    for t in range(start, T):
        lo = max(0, t - W); n = t - lo
        if n <= 0:
            continue
        gap = (cs[t] - cs[lo] - (cs[t, 0] - cs[lo, 0])) / n          # per-day excess over the original
        k = int(np.argmin(gap)); choice[t] = k if gap[k] < -margin else 0
    return choice


def fixed_share_select(L, start, switch=None, eta=1.0, clip=1.0):
    """Fixed-share exponential weights on clipped relative losses (kept for reference; see module doc)."""
    L = np.asarray(L, float); T, K = L.shape
    D = np.clip(L - L[:, :1], -clip, clip) / (2.0 * clip) + 0.5
    w = np.full(K, 1.0 / K); choice = np.zeros(T, int); n = 0
    for t in range(start, T):
        choice[t] = int(np.argmax(w)); n += 1
        eta_t = eta if eta is not None else np.sqrt(np.log(K) / n); a = switch if switch is not None else 1.0 / (n + 1)
        u = w * np.exp(-eta_t * D[t]); u /= u.sum(); w = (1.0 - a) * u + a / K
    return choice


def adaptive_guard(y, v, e, alpha, warm=250, windows=WINDOWS, min_days=250, W=2000, margin=None):
    """Returns (v_hat, e_hat, chosen expert name per day)."""
    y = np.asarray(y, float); paths = expert_paths(y, v, e, alpha, warm, windows); names = list(paths)
    L = np.stack([fz0(y, *paths[k], alpha) for k in names], axis=1)
    m = 0.5 * REGRET_BUDGET.get(alpha, 0.0) if margin is None else margin
    choice = ftl_window_select(L, start=warm + min_days, W=W, margin=m)
    vh = np.empty(len(y)); eh = np.empty(len(y))
    for k, name in enumerate(names):
        idx = choice == k; vh[idx] = paths[name][0][idx]; eh[idx] = paths[name][1][idx]
    return vh, eh, np.array(names)[choice]
