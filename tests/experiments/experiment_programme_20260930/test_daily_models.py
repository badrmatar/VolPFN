"""Prefix-causality, density semantics and train-only-teacher contract tests."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import unittest

import numpy as np
import torch

try:
    from .garch_lstm import GARCHLSTM, student_t_nll
    from .ginn import (GINN, FrozenTeacher, fit_teacher, ginn_loss, next_ar_mean,
                       prepare_history, residual_history, teacher_variances)
except ImportError:
    from garch_lstm import GARCHLSTM, student_t_nll
    from ginn import (GINN, FrozenTeacher, fit_teacher, ginn_loss, next_ar_mean,
                      prepare_history, residual_history, teacher_variances)


class DailyGARCHLSTMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(701)
        torch.set_num_threads(1)

    def test_daily_student_t_is_variance_parameterized_complete_density(self):
        variance = torch.tensor([0.2, 1., 4.], dtype=torch.float64)
        mean = torch.tensor([0., .2, -.3], dtype=torch.float64)
        target = torch.tensor([-.8, .5, 2.], dtype=torch.float64)
        nu = torch.tensor([3., 5., 12.], dtype=torch.float64)
        distribution = torch.distributions.StudentT(
            nu, loc=mean, scale=torch.sqrt(variance * (nu - 2) / nu))
        torch.testing.assert_close(distribution.variance, variance)
        torch.testing.assert_close(student_t_nll(target, mean, variance, nu, reduction="none"),
                                   -distribution.log_prob(target), rtol=1e-12, atol=1e-12)

    def test_daily_garch_lstm_zero_modulation_recovers_garch_kernel(self):
        model = GARCHLSTM(modulation=0).double()
        history = torch.tensor([[.1, -2., .7, 0.]], dtype=torch.float64)
        omega, alpha, beta = model.kernel_parameters()
        variance = model.initial_variance
        manual = []
        for observation in history[0]:
            variance = omega + alpha * observation.square() + beta * variance
            manual.append(variance)
        torch.testing.assert_close(model.forward_sequence(history)["variance"][0], torch.stack(manual))

    def test_daily_garch_lstm_prefix_forecasts_do_not_see_future(self):
        model = GARCHLSTM().double().eval()
        history = torch.randn(2, 126, dtype=torch.float64)
        changed_future = history.clone()
        changed_future[:, 20:] = 50 * torch.randn_like(changed_future[:, 20:])
        full = model.forward_sequence(history)["variance"]
        changed = model.forward_sequence(changed_future)["variance"]
        torch.testing.assert_close(full[:, :20], changed[:, :20], rtol=0, atol=0)
        torch.testing.assert_close(full[:, 19], model(history[:, :20])["variance"], rtol=0, atol=0)

    def test_daily_garch_lstm_positive_finite_variance_and_nu(self):
        model = GARCHLSTM(learn_nu=True)
        history = torch.zeros(3, 126)
        history[1, -1] = 30
        history[2, 10] = -30
        output = model(history)
        self.assertTrue(torch.isfinite(output["variance"]).all())
        self.assertTrue((output["variance"] > 0).all())
        self.assertGreater(output["nu"].item(), 2)
        omega, alpha, beta = model.kernel_parameters()
        self.assertGreater(omega.item(), 0)
        self.assertGreaterEqual(alpha.item(), 0)
        self.assertLess((alpha + beta).item(), 1)
        loss = student_t_nll(torch.tensor([0., -5., 4.]), **output)
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_daily_garch_lstm_state_and_metadata_roundtrip(self):
        model = GARCHLSTM(nu=7.).eval()
        restored = GARCHLSTM().eval()
        restored.load_state_dict(model.state_dict())
        history = torch.randn(3, 40)
        for name, value in model(history).items():
            torch.testing.assert_close(value, restored(history)[name])
        json.dumps(restored.metadata(), allow_nan=False)

    def test_daily_losses_reject_invalid_density_and_cross_broadcast(self):
        returns = torch.ones(3)
        for variance, nu in ((torch.zeros(3), 5.), (torch.ones(3), 2.),
                             (torch.ones(3), math.inf)):
            with self.assertRaises(ValueError):
                student_t_nll(returns, torch.zeros(3), variance, nu)
        with self.assertRaises(ValueError):
            student_t_nll(returns[:, None], torch.zeros(3), torch.ones(3))
        with self.assertRaises(ValueError):
            GARCHLSTM()(torch.tensor([[math.nan]]))


class DailyGINNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1920)
        returns = np.empty(700, dtype=np.float64)
        variance, last = 1., 0.
        for j in range(len(returns)):
            variance = .05 + .10 * last ** 2 + .85 * variance
            returns[j] = .02 + .15 * (returns[j - 1] if j else 0.) + math.sqrt(variance) * rng.normal()
            last = returns[j] - .02
        cls.train = returns[:600].copy()
        cls.test = returns[600:].copy()
        cls.teacher = fit_teacher(cls.train)

    def test_daily_teacher_fit_contains_training_slice_only(self):
        teacher = self.teacher
        expected_hash = hashlib.sha256(self.train.astype("<f8").tobytes()).hexdigest()
        self.assertEqual(teacher.train_count, len(self.train))
        self.assertEqual(teacher.train_sha256, expected_hash)
        self.assertEqual(len(teacher.train_residuals), len(self.train))
        self.assertEqual(len(teacher.train_teacher_variances), len(self.train))
        self.assertTrue(teacher.metadata()["frozen"])
        self.assertEqual(teacher.metadata()["teacher_fit_partition"], "training_only")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            teacher.alpha = 0.9
        json.dumps(teacher.metadata(), allow_nan=False)

    def test_daily_teacher_is_filtered_before_target_and_matches_arch_fit(self):
        history = torch.tensor(self.train[None, :], dtype=torch.float64)
        path = teacher_variances(history, self.teacher)[0]
        self.assertEqual(len(path), len(self.train) + 1)
        torch.testing.assert_close(path[:-1], torch.tensor(self.teacher.train_teacher_variances,
                                                         dtype=path.dtype),
                                   rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(path[-1].item(), self.teacher.next_teacher_variance, places=10)
        self.assertGreater(path[-1].item(), 0)

    def test_daily_ar_and_teacher_paths_are_prefix_causal(self):
        history = torch.tensor(np.concatenate((self.train, self.test))[None, :])
        changed = history.clone()
        changed[:, 600:] = 1000.
        residuals = residual_history(history, self.teacher)
        altered_residuals = residual_history(changed, self.teacher)
        torch.testing.assert_close(residuals[:, :600], altered_residuals[:, :600], rtol=0, atol=0)
        torch.testing.assert_close(teacher_variances(history, self.teacher)[:, :601],
                                   teacher_variances(changed, self.teacher)[:, :601], rtol=0, atol=0)
        prefix = prepare_history(history[:, :600], self.teacher)
        self.assertAlmostEqual(prefix["teacher_variance"].item(), self.teacher.next_teacher_variance, places=10)
        # The AR next mean at t uses r_t, never r_(t+1).
        expected = self.teacher.ar_intercept + self.teacher.ar_coefficients[0] * self.train[-1]
        self.assertAlmostEqual(next_ar_mean(history[:, :600], self.teacher).item(), expected, places=12)

    def test_daily_ginn_loss_is_observed_innovation_plus_frozen_teacher(self):
        predicted = torch.tensor([1., 4.], requires_grad=True)
        observation = torch.tensor([3., -1.], requires_grad=True)
        ar_mean = torch.tensor([1., 1.], requires_grad=True)
        teacher_variance = torch.tensor([2., 3.], requires_grad=True)
        loss = ginn_loss(predicted, observation, ar_mean, teacher_variance,
                         lambda_observed=.25, reduction="none")
        expected = .25 * (predicted - torch.tensor([4., 4.])).square() \
            + .75 * (predicted - torch.tensor([2., 3.])).square()
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        self.assertIsNotNone(predicted.grad)
        self.assertIsNone(observation.grad)
        self.assertIsNone(ar_mean.grad)
        self.assertIsNone(teacher_variance.grad)

    def test_daily_ginn_zero_lambda_is_teacher_distillation(self):
        prediction = torch.tensor([.5, 1.])
        teacher = torch.tensor([.7, .9])
        loss = ginn_loss(prediction, torch.tensor([200., -200.]), torch.zeros(2), teacher,
                         lambda_observed=0., reduction="none")
        torch.testing.assert_close(loss, (prediction - teacher).square())

    def test_daily_ginn_variance_positive_and_model_train_gradients(self):
        torch.manual_seed(7)
        model = GINN(hidden_size=12, num_layers=3)
        history = torch.zeros(4, 90)
        history[1, -1] = 100.
        output = model(history)
        self.assertTrue(torch.isfinite(output["variance"]).all())
        self.assertTrue((output["variance"] > 0).all())
        self.assertEqual(output["mean"].shape, (4,))
        self.assertGreater(output["nu"].item(), 2)
        loss = ginn_loss(output["variance"], torch.tensor([0., 2., -1., .3]),
                         torch.zeros(4), torch.ones(4))
        loss.backward()
        self.assertTrue(any(p.grad is not None and (p.grad != 0).any() for p in model.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_daily_ginn_eval_freezes_batch_norm_statistics(self):
        torch.manual_seed(8)
        model = GINN(hidden_size=8, num_layers=2).eval()
        history = torch.randn(3, 90).square()
        forecast = model(history[:1])["variance"]
        torch.testing.assert_close(forecast, model(history)["variance"][:1], rtol=1e-5, atol=1e-6)
        restored = GINN(hidden_size=8, num_layers=2).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(history)["variance"], restored(history)["variance"])
        json.dumps(restored.metadata(), allow_nan=False)

    def test_daily_ginn_rejects_invalid_proxy_and_broadcast_labels(self):
        model = GINN(hidden_size=8, num_layers=1).eval()
        with self.assertRaises(ValueError):
            model(torch.tensor([[1., -1.]]))
        with self.assertRaises(ValueError):
            ginn_loss(torch.ones(2), torch.ones(2, 1), torch.zeros(2), torch.ones(2))
        with self.assertRaises(ValueError):
            fit_teacher(np.ones(40))
        with self.assertRaises(ValueError):
            fit_teacher(np.ones(10))


if __name__ == "__main__":
    unittest.main()
