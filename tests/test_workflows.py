"""Persistence, import behavior and small real-training workflow checks."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

from frdn_hmm.config import TrainingConfig, WorldConfig
from frdn_hmm.model_io import save_optimized_model, load_optimized_model, find_matching_model, selected_model_path
from frdn_hmm.plotting import plot_fitted_hmm_decision, plot_decision_figure
from frdn_hmm.evaluation import decision_metrics
from frdn_hmm.frdn import true_curve
from run import _different_K, main as run_main
from compare_baselines import run_comparison

ROOT = Path(__file__).resolve().parents[1]


class WorkflowContracts(unittest.TestCase):
    def test_imports_do_not_run_experiments_or_import_torch(self):
        code = "import sys; import run, compare_baselines; from frdn_hmm.model_io import load_optimized_curve; assert 'torch' not in sys.modules"
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
            result = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_metadata_matching_handles_suffix_and_rounded_name_collision(self):
        model = dict(K=2, D_T=np.eye(2) * 0.6, b0=np.ones(2) / 2, optimizer="adam-metric", adam_betas=(0.9, 0.5))
        world = WorldConfig()
        config = TrainingConfig(age_max=8, steps=4, num_restarts=1)
        with tempfile.TemporaryDirectory() as tmp:
            expected = save_optimized_model(model, lambda_val=world.lambda_val, alpha_val=world.alpha, age_max=8,
                                             weight_mode="balanced", data_dir=tmp, training_config=config)
            save_optimized_model(model, lambda_val=0.40001, alpha_val=world.alpha, age_max=8,
                                 weight_mode="balanced", data_dir=tmp, training_config=config)
            found = find_matching_model(2, data_dir=tmp, world=world, age_max=8, weight_mode="balanced", training=config)
            self.assertEqual(expected, found)
            self.assertEqual(expected, selected_model_path(2, data_dir=tmp, world=world, age_max=8))
            with self.assertRaises(FileNotFoundError):
                find_matching_model(2, data_dir=tmp, world=world, age_max=8,
                                    training=TrainingConfig(age_max=8, steps=5, num_restarts=1))

    def test_npz_round_trip(self):
        model = dict(K=2, D_T=np.eye(2) * 0.6, b0=np.array([0.2, 0.8]))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_optimized_model(model, lambda_val=0.4, alpha_val=WorldConfig().alpha,
                                        age_max=8, weight_mode="natural", data_dir=tmp)
            restored = load_optimized_model(2, path)
            np.testing.assert_array_equal(restored["D_T"], model["D_T"])
            np.testing.assert_array_equal(restored["b0"], model["b0"])

    def test_failed_restart_preserves_model_and_records_incomplete_training(self):
        from frdn_hmm import training as trainer
        target = trainer.train_classical_hidden_model_DT
        original = trainer._fit_restart

        def fail_second_restart(*args, **kwargs):
            if kwargs["seed"] == 1:
                raise FloatingPointError("test restart failure")
            return original(*args, **kwargs)

        config = TrainingConfig(age_max=8, steps=3, num_restarts=2)
        world = WorldConfig()
        from frdn_hmm.frdn import compute_true_frdn_analytic
        with patch("frdn_hmm.training._fit_restart", side_effect=fail_second_restart):
            with self.assertWarnsRegex(RuntimeWarning, "skipping this restart"):
                model = target(compute_true_frdn_analytic(age_max=8), 2, **config.trainer_kwargs())
        self.assertFalse(model["training_complete"])
        self.assertEqual(model["restart"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_optimized_model(model, lambda_val=world.lambda_val, alpha_val=world.alpha,
                age_max=8, weight_mode="balanced", data_dir=tmp, training_config=config)
            restored = load_optimized_model(2, path)
            diagnostics = json.loads(str(restored["training_diagnostics_json"]))
            self.assertEqual(diagnostics[0]["completed_steps"], 3)
            self.assertTrue(diagnostics[1]["stopped_early"])
            # A valid partial model remains usable for comparison, while a full
            # training request cannot mistake it for a completed cached run.
            self.assertEqual(path, selected_model_path(2, data_dir=tmp, world=world, age_max=8))
            with self.assertRaises(FileNotFoundError):
                find_matching_model(2, data_dir=tmp, world=world, age_max=8, training=config)

    def test_cache_distinguishes_gradient_clipping_settings(self):
        model = dict(K=2, D_T=np.eye(2) * 0.6, b0=np.ones(2) / 2,
                     optimizer="adam-metric", adam_betas=(0.9, 0.5))
        config = TrainingConfig(age_max=8)
        world = WorldConfig()
        with tempfile.TemporaryDirectory() as tmp:
            save_optimized_model(model, lambda_val=world.lambda_val, alpha_val=world.alpha,
                age_max=8, weight_mode="balanced", data_dir=tmp, training_config=config)
            with self.assertRaises(FileNotFoundError):
                find_matching_model(2, data_dir=tmp, world=world, age_max=8,
                                    training=TrainingConfig(age_max=8, grad_clip_value=100))

    def test_training_cache_force_retrain_and_baseline_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for weight in ("natural", "balanced"):
                training = TrainingConfig(age_max=8, steps=4, num_restarts=1, weight_mode=weight)
                options = dict(K_list=(18,), T_eval=40, data_dir=directory / "data", figure_dir=directory / "training", training=training)
                with contextlib.redirect_stdout(io.StringIO()):
                    first = _different_K(**options)
                    with patch("frdn_hmm.training.train_classical_hidden_model_DT", side_effect=AssertionError("Cached model should avoid training")):
                        self.assertEqual(first, _different_K(**options))
                    third = _different_K(**options, force_retrain=True)
                self.assertNotEqual(first[18], third[18])
            with contextlib.redirect_stdout(io.StringIO()):
                figures, statistics = run_comparison(dimensions=(18,), model_age_max=8, data_dir=directory / "data",
                    output_dir=directory / "comparison", short_horizon=25, error_horizon=50, decision_horizon=70,
                    mode="exact", chunk_size=11)
            self.assertEqual(len(figures), 6)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in figures.values()))
            self.assertEqual(statistics[18]["hmm_evaluation"]["exact_steps"], 71)
            self.assertEqual(len(list((directory / "training").glob("*.pdf"))), 8)

    def test_evaluate_only_exports_legacy_models_without_training(self):
        model = dict(K=2, D_T=np.eye(2) * 0.6, b0=np.ones(2) / 2)
        world = WorldConfig()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = save_optimized_model(model, lambda_val=world.lambda_val, alpha_val=world.alpha,
                age_max=8, weight_mode="balanced", data_dir=directory / "data")
            with patch("frdn_hmm.training.train_classical_hidden_model_DT",
                       side_effect=AssertionError("Evaluation-only must not train")):
                with contextlib.redirect_stdout(io.StringIO()):
                    paths = run_main(["--evaluate-only", "--dimensions", "2", "--train-age", "8",
                        "--eval-horizon", "20", "--data-dir", str(directory / "data"),
                        "--figure-dir", str(directory / "figures")])
            self.assertEqual(paths, {2: path})
            self.assertEqual(len(list((directory / "data").glob("*.npz"))), 1)
            self.assertEqual(len(list((directory / "figures").glob("*_existing.pdf"))), 4)

    def test_evaluate_only_checks_all_models_before_starting(self):
        world = WorldConfig()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_optimized_model(dict(K=2, D_T=np.eye(2) * 0.6, b0=np.ones(2) / 2),
                lambda_val=world.lambda_val, alpha_val=world.alpha, age_max=8,
                weight_mode="balanced", data_dir=directory / "data")
            with patch("run.evaluate_model", side_effect=AssertionError("Missing model must fail first")):
                with self.assertRaises(FileNotFoundError):
                    _different_K(K_list=(2, 3), age_max=8, data_dir=directory / "data",
                                 figure_dir=directory / "figures", evaluate_only=True)
            self.assertFalse((directory / "figures").exists())
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                _different_K(K_list=(2,), evaluate_only=True, force_retrain=True)

    def test_decision_overview_uses_precomputed_metrics(self):
        reference = true_curve(20)
        curves = {"Example HMM": np.full(21, 0.6)}
        metrics = {name: decision_metrics(reference, curve, 0.4) for name, curve in curves.items()}
        fig, supplied = plot_decision_figure(20, 0.4, curves, S_true_curve=reference, metrics_dict=metrics)
        self.assertIs(supplied, metrics)
        self.assertEqual(len(fig.axes), 4)
        plt.close(fig)

    def test_plotting_preserves_global_style_and_closes_figures(self):
        before = dict(plt.rcParams)
        data = dict(K_list=[2], T_vals=np.arange(1, 5), weight_model="natural",
                    result_decision={2: {"cum_wrong": np.zeros(5), "cum_loss": np.zeros(5)}})
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_fitted_hmm_decision(data, output_dir=tmp)
            self.assertEqual(set(paths), {"cum_wrong", "cum_loss"})
        self.assertEqual(before, dict(plt.rcParams))
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
