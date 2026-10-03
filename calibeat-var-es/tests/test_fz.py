"""The scoring machinery against known answers."""
import unittest
import numpy as np
from scipy import stats
from calibeat.fz import fz0, identification, student_t_var_es, normal_var_es, empirical_var_es, hindsight_recalibration


class TestFZ(unittest.TestCase):
    def test_closed_form_tails_match_simulation(self):
        rng = np.random.default_rng(0); n = 2_000_000
        for nu in (3.0, 5.0, 12.0):
            z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2.0))
            for alpha in (0.05, 0.01):
                q, es = student_t_var_es(alpha, nu)
                qe, ese = empirical_var_es(z, alpha)
                self.assertAlmostEqual(q, qe, delta=0.03, msg=f'VaR nu={nu} alpha={alpha}')
                self.assertAlmostEqual(es, ese, delta=0.06 if nu > 3 else 0.15, msg=f'ES nu={nu} alpha={alpha}')
        q, es = normal_var_es(0.05); self.assertAlmostEqual(q, -1.6449, places=3); self.assertAlmostEqual(es, -2.0627, places=3)

    def test_fz0_minimised_at_truth(self):
        """The expected FZ0 over a large sample is smallest at the true (VaR, ES) among a grid of alternatives."""
        rng = np.random.default_rng(1); z = rng.standard_t(5.0, size=400_000) / np.sqrt(5.0 / 3.0); alpha = 0.05
        q, es = student_t_var_es(alpha, 5.0); best = np.mean(fz0(z, q, es, alpha))
        for dq in (-0.2, 0.2):
            for de in (-0.2, 0.2):
                self.assertGreater(np.mean(fz0(z, q + dq, es + de, alpha)), best)
        self.assertGreater(np.mean(fz0(z, q, es * 1.15, alpha)), best)

    def test_identification_mean_zero_at_truth(self):
        rng = np.random.default_rng(2); z = rng.standard_t(5.0, size=400_000) / np.sqrt(5.0 / 3.0); alpha = 0.025
        q, es = student_t_var_es(alpha, 5.0); a, b = identification(z, q, es, alpha)
        self.assertLess(abs(a.mean()), 0.001); self.assertLess(abs(b.mean()), 0.02)

    def test_empirical_pair_minimises_empirical_fz0(self):
        rng = np.random.default_rng(3); z = rng.standard_t(4.0, size=3000); alpha = 0.05
        v, e = empirical_var_es(z, alpha); base = np.sum(fz0(z, v, e, alpha))
        for dv in (-0.05, 0.05):
            for de in (-0.05, 0.05):
                self.assertGreaterEqual(np.sum(fz0(z, v + dv, e + de, alpha)) + 1e-9, base)

    def test_hindsight_recalibration_never_hurts(self):
        rng = np.random.default_rng(4); y = rng.standard_t(5.0, size=5000) * (1 + 0.5 * (rng.random(5000) > 0.5)); alpha = 0.05
        v = np.full(5000, -1.0); e = np.full(5000, -1.5); groups = (np.abs(y) > 0).astype(int) * (rng.random(5000) > 0.5)
        vh, eh = hindsight_recalibration(y, v, e, alpha, groups)
        self.assertLessEqual(np.mean(fz0(y, vh, eh, alpha)), np.mean(fz0(y, v, e, alpha)) + 1e-9)


if __name__ == '__main__':
    unittest.main()
