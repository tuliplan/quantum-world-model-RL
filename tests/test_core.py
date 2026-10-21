"""Numerical contracts for FRDN probabilities, HMM recurrence and statistics."""

import unittest
from unittest.mock import patch
import numpy as np
import torch

from frdn_hmm.config import BASELINE_WORLD, EvaluationConfig, TrainingConfig, WorldConfig
from frdn_hmm.frdn import compute_true_frdn_analytic, tick_probabilities, true_curve
from frdn_hmm.baselines import build_phase_cycle_hmm, check_model, S_age_counter, S_cycle_formula, S_cycle_from_hmm
from frdn_hmm.hmm import compute_predicted_quantities_from_DT, iter_probability_chunks
from frdn_hmm.evaluation import action_from_gap, decision_metrics, evaluate_decisions_chunked
from frdn_hmm.training import (project_simplex, project_weighted_simplex,
                               _fit_restart, _predict_hazards, _prepare_gradients,
                               train_classical_hidden_model_DT, train_classical_hidden_model_DT_sampled)


class NumericalContracts(unittest.TestCase):
    def setUp(self):
        self.transition = np.array([[0.4, 0.1], [0.2, 0.5]])
        self.belief = np.array([0.7, 0.3])

    def test_analytic_curve_and_reset(self):
        target = compute_true_frdn_analytic(age_max=100)
        probabilities = tick_probabilities(np.arange(101))
        np.testing.assert_allclose(probabilities, target["S_cond"], atol=1e-14, rtol=0)
        self.assertLess(probabilities[0], 1)
        self.assertEqual(true_curve(10)[0], BASELINE_WORLD.reset_probability)

    def test_true_curve_custom_world(self):
        world = WorldConfig(lambda_val=0.6, alpha=1.3)
        target = compute_true_frdn_analytic(world.lambda_val, world.alpha, 50)
        np.testing.assert_allclose(tick_probabilities(np.arange(51), world=world), target["S_cond"], atol=1e-14)

    def test_conditional_prediction_matches_matrix_powers(self):
        prediction = compute_predicted_quantities_from_DT(self.transition, self.belief, 20)
        survival = np.array([np.linalg.matrix_power(self.transition, t).dot(self.belief).sum() for t in range(22)])
        np.testing.assert_allclose(prediction["S_pred"], survival[1:] / survival[:-1], atol=1e-14)
        np.testing.assert_allclose(prediction["P_C"], survival, atol=1e-14)

    def test_underflow_does_not_damage_conditional_probability(self):
        prediction = compute_predicted_quantities_from_DT(np.array([[0.4]]), np.array([1.0]), 5000, store_history=False)
        self.assertEqual(prediction["P_C"][-1], 0)
        self.assertTrue(np.isfinite(prediction["S_pred"]).all())
        np.testing.assert_allclose(prediction["S_pred"], 0.4)
        self.assertIsNone(prediction["eta_history"])

    def test_chunked_prediction_matches_full_curve(self):
        expected = compute_predicted_quantities_from_DT(self.transition, self.belief, 101)["S_pred"]
        for size in (1, 7, 1000):
            chunks = list(iter_probability_chunks(self.transition, self.belief, 101, size))
            self.assertEqual(chunks[0][0], 0)
            np.testing.assert_allclose(np.concatenate([curve for _, curve in chunks]), expected, atol=1e-14)

    def test_zero_tick_probability_policy(self):
        prediction = compute_predicted_quantities_from_DT(np.zeros((1, 1)), np.ones(1), 4)
        self.assertEqual(prediction["valid_until"], 0)
        self.assertEqual(prediction["S_pred"][0], 0)
        self.assertTrue(np.isnan(prediction["S_pred"][1:]).all())
        with self.assertRaises(ValueError):
            list(iter_probability_chunks(np.zeros((1, 1)), np.ones(1), 4))

    def test_invalid_transition_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_predicted_quantities_from_DT(np.array([[1.1]]), np.ones(1), 10)

    def test_phase_cycle_formula_matches_shared_recurrence(self):
        for p, q in ((1, 3), (6, 17)):
            cycle = build_phase_cycle_hmm(p, q)
            self.assertTrue(check_model(cycle))
            np.testing.assert_allclose(S_cycle_from_hmm(cycle, 200), S_cycle_formula(cycle, 200), atol=1e-14)

    def test_age_counter_retains_final_probability(self):
        reference = true_curve(100)
        for N in (1, 4, 18):
            np.testing.assert_array_equal(S_age_counter(N, 100), reference[np.minimum(np.arange(101), N - 1)])

    def test_ties_and_index_zero(self):
        np.testing.assert_array_equal(action_from_gap(np.array([-1e-13, -0.2, 0, 0.2, 1e-13])), [0, -1, 0, 1, 0])
        metrics = decision_metrics(np.array([0.9, 0.8, 0.2, 0.4]), np.array([0.1, 0.1, 0.8, 0.9]), 0.4)
        np.testing.assert_allclose(metrics["cum_wrong"], [0, 1, 1, 2 / 3])
        np.testing.assert_allclose(metrics["cum_loss"], [0, 0.4, 0.3, 0.2])

    def test_streamed_decisions_match_full_statistics(self):
        cycle = build_phase_cycle_hmm(1, 3)
        model = {"D_T": self.transition, "b0": self.belief}
        horizon = 137
        reference = true_curve(horizon)
        predictions = {"Age counter": S_age_counter(4, horizon), "Rational cycle": S_cycle_formula(cycle, horizon),
                       "Best-found HMM": compute_predicted_quantities_from_DT(self.transition, self.belief, horizon)["S_pred"]}
        for size in (1, 13, 1000):
            result = evaluate_decisions_chunked(model, cycle, horizon, chunk_size=size)
            for method, prediction in predictions.items():
                expected = decision_metrics(reference, prediction, 0.4)
                for metric in ("cum_wrong", "cum_loss"):
                    np.testing.assert_allclose(result["methods"][method][metric], expected[metric][result["durations"]], atol=1e-14)

    def test_fast_mode_records_approximation_and_respects_threshold(self):
        diagnostics = {}
        list(iter_probability_chunks(np.array([[0.7]]), np.ones(1), 50, mode="fast", convergence_window=4, diagnostics=diagnostics))
        self.assertEqual(diagnostics["exact_steps"], 4)
        self.assertEqual(diagnostics["approximated_steps"], 47)
        self.assertEqual(diagnostics["tail_start_age"], 4)
        diagnostics = {}
        list(iter_probability_chunks(np.array([[0.4]]), np.ones(1), 50, mode="fast", convergence_window=4, diagnostics=diagnostics))
        self.assertEqual(diagnostics["approximated_steps"], 0)

    def test_equal_metric_projection_recovers_euclidean_projection(self):
        values = torch.tensor([[-0.4, 0.7, 1.3], [0.2, 0.2, 0.2]], dtype=torch.float64)
        np.testing.assert_allclose(project_weighted_simplex(values, torch.ones_like(values)).numpy(), project_simplex(values).numpy(), atol=1e-14)

    def test_weighted_projection_simplex_and_stationarity(self):
        values = torch.tensor([0.7, 0.6, -0.5], dtype=torch.float64)
        metric = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)
        projection = project_weighted_simplex(values, metric)
        self.assertAlmostEqual(float(projection.sum()), 1)
        self.assertTrue(bool(torch.all(projection >= 0)))
        residual = (metric * (values - projection))[projection > 0]
        np.testing.assert_allclose(residual.numpy(), float(residual[0]), atol=1e-14)

    def test_gradient_clipping_prevents_adam_second_moment_overflow(self):
        parameter = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([parameter], betas=(0.9, 0.5))
        parameter.grad = torch.tensor([2.16e159, -0.25], dtype=torch.float64)
        self.assertTrue(_prepare_gradients((parameter,), 1e4))
        torch.testing.assert_close(parameter.grad, torch.tensor([1e4, -0.25], dtype=torch.float64))
        optimizer.step()
        self.assertTrue(bool(torch.isfinite(optimizer.state[parameter]["exp_avg_sq"]).all()))
        parameter.grad = torch.tensor([float("nan"), 0.0], dtype=torch.float64)
        with self.assertRaisesRegex(FloatingPointError, "non-finite gradient"):
            _prepare_gradients((parameter,), 1e4)

    def test_long_recurrence_overflow_recovers_best_finite_parameters(self):
        # An unreachable, longer-lived state makes backward unstable although
        # every forward hazard remains 0.7 and the loss stays finite.
        initial = torch.tensor([[0.7, 0.1], [0.3, 0.0], [0.0, 0.9]], dtype=torch.float64)
        belief = torch.tensor([1.0, 0.0], dtype=torch.float64)
        diagnostics = {}
        with patch("frdn_hmm.training._initialize_parameters",
                   return_value=(initial.clone().requires_grad_(True), belief.clone().requires_grad_(True))):
            with self.assertWarnsRegex(RuntimeWarning, "non-finite gradient"):
                theta, beta = _fit_restart(2, torch.full((1001,), 0.6, dtype=torch.float64),
                    torch.full((1001,), 1 / 1001, dtype=torch.float64), age_max=1000, ages=None,
                    steps=3, lr=1e-3, seed=0, device="cpu", eps=1e-9,
                    optimizer_name="adam-metric", adam_betas=(0.9, 0.5), grad_clip_value=None,
                    diagnostics=diagnostics)
        torch.testing.assert_close(theta, initial)
        torch.testing.assert_close(beta, belief)
        self.assertTrue(diagnostics["stopped_early"])
        self.assertEqual(diagnostics["completed_steps"], 0)

    def test_recurrence_gradient_clipping_keeps_long_backward_finite(self):
        initial = torch.tensor([[0.7, 0.1], [0.3, 0.0], [0.0, 0.9]], dtype=torch.float64)
        belief = torch.tensor([1.0, 0.0], dtype=torch.float64)
        theta = initial.clone().requires_grad_(True)
        beta = belief.clone().requires_grad_(True)
        plain = _predict_hazards(theta, beta, 1000, 1e-9)
        protected = _predict_hazards(theta, beta, 1000, 1e-9, grad_clip_value=1e4)
        torch.testing.assert_close(plain, protected, rtol=0, atol=0)
        loss = -(0.6 * protected.log() + 0.4 * (1 - protected).log()).mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(theta.grad).all()))
        self.assertTrue(bool(torch.isfinite(beta.grad).all()))
        diagnostics = {}
        with patch("frdn_hmm.training._initialize_parameters",
                   return_value=(initial.clone().requires_grad_(True), belief.clone().requires_grad_(True))):
            _fit_restart(2, torch.full((1001,), 0.6, dtype=torch.float64),
                torch.full((1001,), 1 / 1001, dtype=torch.float64), age_max=1000, ages=None,
                steps=3, lr=1e-3, seed=0, device="cpu", eps=1e-9,
                optimizer_name="adam-metric", adam_betas=(0.9, 0.5), diagnostics=diagnostics)
        self.assertEqual(diagnostics["completed_steps"], 3)
        self.assertFalse(diagnostics["stopped_early"])

    def test_sampled_training_matches_dense_training_on_the_same_ages(self):
        target = compute_true_frdn_analytic(age_max=8)
        ages = np.arange(9)
        sparse = dict(Phi=None, S_cond=target["S_cond"], q=target["q"], log_Phi=target["log_Phi"][:9])
        for weight in ("balanced", "natural"):
            options = dict(K=3, steps=4, num_restarts=1, weight_mode=weight)
            dense = train_classical_hidden_model_DT(target, **options)
            sampled = train_classical_hidden_model_DT_sampled(sparse, training_ages=ages, **options)
            np.testing.assert_allclose(dense["D_T"], sampled["D_T"], atol=1e-12)
            np.testing.assert_allclose(dense["b0"], sampled["b0"], atol=1e-12)

    def test_configuration_validation_and_optimizer_presets(self):
        self.assertEqual(TrainingConfig().optimizer, "adam-metric")
        self.assertEqual(TrainingConfig(weight_mode="natural").betas, (0.9, 0.999))
        self.assertEqual(TrainingConfig(weight_mode="balanced", optimizer_name="adam").betas, (0.9, 0.999))
        with self.assertRaises(ValueError):
            EvaluationConfig(horizon=0)
        with self.assertRaises(ValueError):
            TrainingConfig(steps=0)
        for clip in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TrainingConfig(grad_clip_value=clip)


if __name__ == "__main__":
    unittest.main()
