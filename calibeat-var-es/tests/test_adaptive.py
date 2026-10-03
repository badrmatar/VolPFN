import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibeat import adaptive as A
from calibeat.fz import fz0


def test_selection_is_predictable_and_tracks_a_switch():
    T = 6000; rng = np.random.default_rng(1)
    good = rng.normal(-0.05, 0.3, T); bad = good + 0.10                  # good beats the original by 0.05, bad loses by 0.05
    L1 = np.column_stack([np.zeros(T), good, bad]); L2 = np.column_stack([np.zeros(T), bad, good])
    L = np.where(np.arange(T)[:, None] < T // 2, L1, L2)                # the better expert switches at T/2
    c = A.ftl_window_select(L, start=500, W=1000, margin=0.01)
    assert np.mean(c[1500:T // 2] == 1) > 0.9 and np.mean(c[T // 2 + 1500:] == 2) > 0.9
    L_alt = L.copy(); L_alt[4000:] = rng.normal(size=L_alt[4000:].shape)   # changing the future changes nothing before it
    assert np.array_equal(A.ftl_window_select(L, start=500, W=1000)[:4001], A.ftl_window_select(L_alt, start=500, W=1000)[:4001])


def test_adaptive_guard_is_never_much_worse_than_the_original_on_a_calibrated_box():
    from calibeat import sandbox as S
    y, sigma, qv, qe = S.simulate('garch_t', 8000, 3, 0.05); v, e = qv, qe
    vh, eh, names = A.adaptive_guard(y, v, e, 0.05)
    sc = slice(500, None); excess = np.mean(fz0(y[sc], vh[sc], eh[sc], .05)) - np.mean(fz0(y[sc], v[sc], e[sc], .05))
    assert excess < 0.01, excess
    assert set(names[:500]) == {'original'}
