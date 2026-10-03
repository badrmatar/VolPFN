"""The online wrappers against planted answers: they must correct a wrong tail law and a scale
bias, must not use the future, and must be near the hindsight benchmark on stationary data."""
import unittest
import numpy as np
from calibeat.fz import fz0, student_t_var_es, normal_var_es, hindsight_ratio_recalibration
from calibeat.calibeater import RatioCalibeater, BinCalibeater


def world(T=20000, seed=0, nu=5.0, alpha=0.05):
    rng = np.random.default_rng(seed)
    sig = np.exp(0.3 * np.sin(np.arange(T) / 300.0))                   # slowly varying scale, known
    y = sig * rng.standard_t(nu, size=T) / np.sqrt(nu / (nu - 2.0))
    q, es = student_t_var_es(alpha, nu)
    return y, sig, sig * q, sig * es


class TestCalibeaters(unittest.TestCase):
    def test_ratio_corrects_wrong_tail_law(self):
        alpha = 0.05; y, sig, qv, qe = world(alpha=alpha)
        qn, esn = normal_var_es(alpha); v, e = sig * qn, sig * esn        # right scale, Gaussian tail
        vh, eh = RatioCalibeater(alpha, warm=250).run(y, v, e)
        sc = slice(2000, None)
        self.assertLess(np.mean(fz0(y[sc], vh[sc], eh[sc], alpha)), np.mean(fz0(y[sc], v[sc], e[sc], alpha)))
        self.assertLess(abs(np.mean(fz0(y[sc], vh[sc], eh[sc], alpha)) - np.mean(fz0(y[sc], qv[sc], qe[sc], alpha))), 0.01)
        self.assertAlmostEqual(float(np.mean(y[sc] <= vh[sc])), alpha, delta=0.006)

    def test_ratio_corrects_scale_bias_and_is_near_hindsight(self):
        alpha = 0.025; y, sig, qv, qe = world(alpha=alpha, seed=1)
        v, e = 0.7 * qv, 0.7 * qe
        vh, eh = RatioCalibeater(alpha, warm=250).run(y, v, e); sc = slice(2000, None)
        vr, er = hindsight_ratio_recalibration(y[sc], v[sc], alpha)
        online = np.mean(fz0(y[sc], vh[sc], eh[sc], alpha)); hind = np.mean(fz0(y[sc], vr, er, alpha)); orig = np.mean(fz0(y[sc], v[sc], e[sc], alpha))
        self.assertLess(online, orig); self.assertLess(online - hind, 0.1 * (orig - hind))

    def test_no_future_leak(self):
        alpha = 0.05; y, sig, qv, qe = world(T=3000)
        v, e = sig * normal_var_es(alpha)[0], sig * normal_var_es(alpha)[1]
        a = RatioCalibeater(alpha, warm=250).run(y, v, e)
        y2 = y.copy(); y2[2000:] = -y2[2000:]                            # change the future only
        b = RatioCalibeater(alpha, warm=250).run(y2, v, e)
        np.testing.assert_allclose(a[0][:2001], b[0][:2001]); np.testing.assert_allclose(a[1][:2001], b[1][:2001])
        c = BinCalibeater(alpha, K=5, warm=250).run(y, v, e); d = BinCalibeater(alpha, K=5, warm=250).run(y2, v, e)
        np.testing.assert_allclose(c[0][:2001], d[0][:2001])

    def test_bins_improve_when_forecasts_are_constant_within_bins(self):
        """Foster-Hart binning retains the forecaster's information only at the bins' resolution. With a
        constant scale (one bin is exact) the bin calibeater must fix a wrong tail law; with a varying
        scale its resolution loss can exceed the tail gain, and the ratio calibeater, which keeps the
        forecaster's dynamics, must beat it. Both facts are part of the theorem's scope."""
        alpha = 0.05; rng = np.random.default_rng(2); T = 20000
        y = rng.standard_t(5.0, size=T) / np.sqrt(5.0 / 3.0)                # constant scale
        v = np.full(T, normal_var_es(alpha)[0]); e = np.full(T, normal_var_es(alpha)[1])
        vh, eh = BinCalibeater(alpha, K=1, warm=250).run(y, v, e); sc = slice(2000, None)
        self.assertLess(np.mean(fz0(y[sc], vh[sc], eh[sc], alpha)), np.mean(fz0(y[sc], v[sc], e[sc], alpha)))
        y2, sig, qv, qe = world(alpha=alpha, seed=2)                         # varying scale
        v2, e2 = sig * normal_var_es(alpha)[0], sig * normal_var_es(alpha)[1]
        vb, eb = BinCalibeater(alpha, K=5, warm=250).run(y2, v2, e2); vr, er = RatioCalibeater(alpha, warm=250).run(y2, v2, e2)
        self.assertLess(np.mean(fz0(y2[sc], vr[sc], er[sc], alpha)), np.mean(fz0(y2[sc], vb[sc], eb[sc], alpha)))

    def test_sa_tracker_moves_with_a_break(self):
        alpha = 0.05; T = 20000; rng = np.random.default_rng(5)
        nu = np.where(np.arange(T) < T // 2, 12.0, 3.0)
        y = np.array([rng.standard_t(n) / np.sqrt(n / (n - 2.0)) for n in nu]); v = np.full(T, normal_var_es(alpha)[0]); e = np.full(T, normal_var_es(alpha)[1])
        ftl = RatioCalibeater(alpha, warm=250, mode='ftl').run(y, v, e); sa = RatioCalibeater(alpha, warm=250, mode='sa', eta=0.01).run(y, v, e)
        late = slice(T - 4000, None); q3, es3 = student_t_var_es(alpha, 3.0)
        self.assertLess(abs(np.mean(sa[1][late]) - es3), abs(np.mean(ftl[1][late]) - es3))


if __name__ == '__main__':
    unittest.main()
