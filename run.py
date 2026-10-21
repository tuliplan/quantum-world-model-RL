"""Train contiguous-age HMMs and export FRDN benchmark figures."""

import argparse
import numpy as np

from frdn_hmm.config import DATA_DIR, FIGURE_DIR, EvaluationConfig, TrainingConfig, WorldConfig
from frdn_hmm.evaluation import evaluate_model
from frdn_hmm.frdn import compute_theoretical_frdn_gap, compute_true_frdn_analytic
from frdn_hmm.model_io import find_matching_model, load_optimized_model, save_optimized_model
from frdn_hmm.plotting import plot_different_K, plot_fitted_hmm_decision


def _different_K(age_max=1000, T_eval=100000, K_list=(18,100,479),
                 data_dir=DATA_DIR, *, force_retrain=False, weight_mode="balanced",
                 world=None, training=None, evaluation=None, figure_dir=FIGURE_DIR,
                 evaluate_only=False):
    """Train on every age 0..age_max, save before evaluating and plot the results.

    Config objects expose experiment settings without adding positional arguments.
    The return value remains a mapping from model dimension to saved NPZ path.
    """
    world = world or WorldConfig()
    training = training or TrainingConfig(age_max=age_max, weight_mode=weight_mode)
    evaluation = evaluation or EvaluationConfig(horizon=T_eval, threshold=world.lambda_val)
    dimensions = tuple(K_list)
    if not dimensions or any(isinstance(N, bool) or not isinstance(N, int) or N < 1 for N in dimensions):
        raise ValueError("K_list must contain positive integer dimensions")
    if evaluate_only and force_retrain:
        raise ValueError("evaluate_only and force_retrain cannot be combined")
    # Resolve every existing model first; evaluation-only never starts training.
    existing_paths = ({N: find_matching_model(N, data_dir=data_dir, world=world,
                        age_max=training.age_max, weight_mode=training.weight_mode)
                       for N in dimensions} if evaluate_only else {})
    target = compute_true_frdn_analytic(world.lambda_val, world.alpha, training.age_max)
    if world.reset_probability is not None:
        raise ValueError("Training uses the analytic reset probability; use reset overrides for baseline comparison")
    saved_paths, results = {}, {}
    for N in dimensions:
        try:
            if force_retrain:
                raise FileNotFoundError
            path = existing_paths[N] if evaluate_only else find_matching_model(N, data_dir=data_dir, world=world,
                                       age_max=training.age_max, weight_mode=training.weight_mode,
                                       training=training)
            model = load_optimized_model(N, path)
            print(f"Model N={N} loaded from {path}")
        except FileNotFoundError:
            if evaluate_only:
                raise
            from frdn_hmm.training import train_classical_hidden_model_DT
            model = train_classical_hidden_model_DT(target, N, **training.trainer_kwargs(), verbose=True)
            path = save_optimized_model(model, lambda_val=world.lambda_val, alpha_val=world.alpha,
                                        age_max=training.age_max, weight_mode=training.weight_mode,
                                        data_dir=data_dir, training_config=training)
            print(f"Model N={N} saved to {path}")
        saved_paths[N] = path
        results[N] = evaluate_model(model, plot_age_max=training.age_max, world=world, config=evaluation)
        tail = results[N]["tail"]
        print(f"N={N}: S(T={evaluation.horizon})={tail['endpoint']:.12f}; "
              f"tail spread={tail['spread']:.3e}; stable={tail['stable']}")

    suffix = f"_{training.weight_mode}"
    if evaluate_only:
        suffix += "_existing"
    elif training.optimizer == "adam-metric":
        suffix += f"_adam-metric_b2-{training.betas[1]:g}"
    bound = compute_theoretical_frdn_gap(world.lambda_val, world.alpha)["kappa_FRDN"]
    data = dict(result_prob={0: target["S_cond"], **{N: value["probability"] for N, value in results.items()}},
                result_value={N: value["value_gap"] for N, value in results.items()},
                result_decision={N: value["decisions"] for N, value in results.items()},
                K_list=list(dimensions), T_vals=np.arange(1, evaluation.horizon + 1),
                value_bound=evaluation.reward_scale * bound,
                age_max=training.age_max, T_eval=evaluation.horizon, alpha_val=world.alpha,
                lambda_val=world.lambda_val, weight_model=training.weight_mode,
                optimizer=training.optimizer, adam_betas=training.betas, artifact_suffix=suffix)
    paths = plot_different_K(data, output_dir=figure_dir)
    paths.update(plot_fitted_hmm_decision(data, output_dir=figure_dir))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return saved_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[479,])
    parser.add_argument("--train-age", type=int, default=10000)
    parser.add_argument("--eval-horizon", type=int, default=100000)
    parser.add_argument("--weight", choices=["balanced", "natural"], default="balanced")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer", choices=["adam", "adam-metric"])
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--grad-clip-value", type=float, default=1e4,
                        help="Limit recurrence and parameter gradients before Adam (default: 10000)")
    parser.add_argument("--lambda-val", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=WorldConfig().alpha)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force-retrain", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true",
                      help="Evaluate existing models matched by world, age and weight; never train")
    args = parser.parse_args(argv)
    training = TrainingConfig(age_max=args.train_age, weight_mode=args.weight,
                              steps=args.steps, num_restarts=args.restarts, lr=args.learning_rate,
                              seed=args.seed, device=args.device, optimizer_name=args.optimizer,
                              adam_betas=None if args.beta2 is None else (0.9, args.beta2),
                              grad_clip_value=args.grad_clip_value)
    return _different_K(K_list=args.dimensions, data_dir=args.data_dir,
                        force_retrain=args.force_retrain, evaluate_only=args.evaluate_only,
                        world=WorldConfig(args.lambda_val, args.alpha),
                        training=training, evaluation=EvaluationConfig(horizon=args.eval_horizon,
                        threshold=args.lambda_val), figure_dir=args.figure_dir)


if __name__ == "__main__":
    main()
