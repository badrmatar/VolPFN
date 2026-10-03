"""Small local checks for the ARCH-LSTM adaptation; no market-data training."""

import math
import unittest

import torch

from experiments.experiment_programme_20260930.arch_lstm import ARCHLSTM, compute_loss_scales, loss, student_t_nll


class ARCHLSTMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)

    def test_nll_matches_variance_standardized_student_t_distribution(self):
        variance = torch.tensor([0.1, 1.0, 7.0, 1e-8], dtype=torch.float64)
        mean = torch.tensor([0.2, -0.3, 0.0, 0.0], dtype=torch.float64)
        nu = torch.tensor([3.0, 5.0, 10.0, 30.0], dtype=torch.float64)
        target = torch.tensor([-2.0, 0.2, 5.0, 1e-4], dtype=torch.float64)
        params = dict(mean=mean, variance=variance, nu=nu)
        distribution = torch.distributions.StudentT(
            nu, mean, (variance * (nu - 2) / nu).sqrt()
        )
        torch.testing.assert_close(
            student_t_nll(params, target, reduction="none"),
            -distribution.log_prob(target),
            rtol=1e-12, atol=1e-12,
        )

    def test_hybrid_loss_matches_author_fixed_nu_formula(self):
        params = {
            "mean": torch.zeros(3, dtype=torch.float64),
            "variance": torch.tensor([0.5, 1.2, 3.0], dtype=torch.float64),
            "nu": torch.tensor(3.0, dtype=torch.float64),
        }
        target = torch.tensor([-0.1, 0.4, -2.0], dtype=torch.float64)
        # Independent scalar reference to the MIT author's stated formula.
        sse = sum((v - r * r) ** 2 for v, r in zip([0.5, 1.2, 3.0], [-0.1, 0.4, -2.0])) / 3
        t_term = sum(0.5 * math.log(v) + 2 * math.log1p(r * r / v)
                     for v, r in zip([0.5, 1.2, 3.0], [-0.1, 0.4, -2.0])) / 3
        for weight in [0.0, 0.1, 0.5, 1.0]:
            value = loss(params, target, weight, sse_scale=2, student_scale=3)
            self.assertAlmostEqual(float(value), (1 - weight) * sse / 2 + weight * t_term / 3, 12)

    def test_only_squared_returns_in_declared_context_affect_forecast(self):
        model = ARCHLSTM(context=2, hidden_size=4).double().eval()
        history = torch.tensor([[9.0, -3.0, 0.7, -1.2], [4.0, 6.0, -0.2, 0.4]], dtype=torch.float64)
        original = model(history)["variance"]
        changed_prefix = history.clone()
        changed_prefix[:, :2] = 1000
        torch.testing.assert_close(original, model(changed_prefix)["variance"])
        torch.testing.assert_close(original, model(-history)["variance"])
        changed_last = history.clone()
        changed_last[:, -1] *= 2
        self.assertFalse(torch.allclose(original, model(changed_last)["variance"]))

    def test_eval_is_stateless_and_independent_of_other_batch_members(self):
        model = ARCHLSTM(context=3, hidden_size=4).double().eval()
        history = torch.randn(7, 3, dtype=torch.float64)
        together = model(history)
        separately = torch.cat([model(row[None, :])["variance"] for row in history])
        torch.testing.assert_close(together["variance"], separately)
        torch.testing.assert_close(model(history)["variance"], together["variance"])
        self.assertEqual(together["mean"].shape, (7,))
        self.assertTrue(bool((together["mean"] == 0).all()))
        self.assertTrue(bool((together["nu"] == 3).all()))

    def test_small_training_step_has_finite_gradients_and_changes_forecasts(self):
        model = ARCHLSTM(context=2, hidden_size=4, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = torch.randn(12, 2)
        target = torch.randn(12)
        model.eval()
        before = model(history)["variance"].detach().clone()
        model.train()
        optimizer.zero_grad()
        objective = loss(model(history), target, weight=0.5)
        objective.backward()
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        optimizer.step()
        model.eval()
        after = model(history)["variance"].detach()
        self.assertTrue(bool(torch.isfinite(after).all()))
        self.assertTrue(bool((after > 0).all()))
        self.assertFalse(torch.allclose(before, after))

    def test_scale_estimator_matches_author_training_only_reference(self):
        returns = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
        scales = compute_loss_scales(returns)
        self.assertAlmostEqual(scales.sse_scale, 3.0)
        self.assertAlmostEqual(scales.reference_variance, 1.5)
        expected_t = sum(0.5 * math.log(1.5) + 3 * math.log1p(x / 4.5)
                         for x in [1.0, 0.0, 1.0, 4.0]) / 4
        self.assertAlmostEqual(scales.student_scale, abs(expected_t))
        zero_scales = compute_loss_scales(torch.zeros(5))
        self.assertGreater(zero_scales.sse_scale, 0)
        self.assertTrue(math.isfinite(zero_scales.student_scale))

    def test_variance_head_does_not_overflow_on_extreme_pre_activation(self):
        model = ARCHLSTM(hidden_size=4).eval()
        with torch.no_grad():
            model.log_variance_head.weight.zero_()
            model.log_variance_head.bias.fill_(1000)
        result = model(torch.ones(2, 1))["variance"]
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(result, torch.full_like(result, math.exp(20)))

    def test_shape_and_tail_parameter_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            ARCHLSTM(nu=2)
        with self.assertRaises(ValueError):
            ARCHLSTM(context=0)
        model = ARCHLSTM(context=2, hidden_size=4)
        with self.assertRaises(ValueError):
            model(torch.ones(2, 1))
        params = model(torch.ones(2, 2))
        with self.assertRaises(ValueError):
            loss(params, torch.ones(2, 1))
        with self.assertRaises(ValueError):
            loss(params, torch.ones(2), weight=1.1)
        with self.assertRaises(ValueError):
            student_t_nll({**params, "nu": torch.tensor(2.0)}, torch.ones(2))


if __name__ == "__main__":
    unittest.main()
