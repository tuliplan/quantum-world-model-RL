"""Compare age-counter, rational-cycle and fitted HMMs without import-time work."""

import argparse
from pathlib import Path
import numpy as np

from frdn_hmm.baselines import build_phase_cycle_hmm, check_model, S_age_counter, S_cycle_formula, S_cycle_from_hmm
from frdn_hmm.config import BASELINE_WORLD, DATA_DIR, BASELINE_FIGURE_DIR, PHASE_CYCLES, WorldConfig
from frdn_hmm.evaluation import error_metrics, evaluate_decisions_chunked, print_fitted_hmm_convergence
from frdn_hmm.frdn import compute_theoretical_frdn_gap, true_curve
from frdn_hmm.hmm import compute_predicted_quantities_from_DT
from frdn_hmm.model_io import load_optimized_model, selected_model_path
from frdn_hmm.plotting import (plot_short_horizon, plot_cumulative_errors, save_figure,
                              plot_wrong_action_frequency_by_N, plot_wrong_action_frequency_by_method,
                              plot_decision_loss_by_N, plot_decision_loss_by_method)


def build_model_curves(T, classical_models_dict, *, world=BASELINE_WORLD,
                       data_dir=DATA_DIR, model_age_max=1000, weight_mode="balanced"):
    """Resolve configured models and produce their curves for an explicit experiment."""
    curves = {}
    for name, spec in classical_models_dict.items():
        kind = spec["type"]
        if kind == "age_counter":
            curves[name] = S_age_counter(spec["N"], T, world=world)
        elif kind == "phase_cycle":
            curves[name] = S_cycle_from_hmm(build_phase_cycle_hmm(spec["p"], spec["q"], world=world), T)
        elif kind == "hmm":
            model = spec["model"]
            if "D_T_W" in model:
                curves[name] = S_cycle_from_hmm(model, T)
            else:
                curves[name] = compute_predicted_quantities_from_DT(model["D_T"], model["b0"], T,
                                                                   store_history=False)["S_pred"]
        elif kind == "optimized":
            path = selected_model_path(spec["N"], spec.get("data_path"), data_dir=data_dir,
                                       world=world, age_max=model_age_max, weight_mode=weight_mode)
            model = load_optimized_model(spec["N"], path)
            curves[name] = compute_predicted_quantities_from_DT(model["D_T"], model["b0"], T,
                                                               store_history=False)["S_pred"]
        else:
            raise ValueError(f"Unknown classical model type: {kind}")
    return curves


def sampled_decision_metrics(N, horizon, threshold=0.4, *, data_path=None,
                             world=BASELINE_WORLD, data_dir=DATA_DIR, model_age_max=1000,
                             weight_mode="balanced", **options):
    """Compatibility entry: all ages contribute; only plot coordinates are sampled."""
    p, q = next((p, q) for p, q in PHASE_CYCLES if q + 1 == N)
    path = selected_model_path(N, data_path, data_dir=data_dir, world=world,
                               age_max=model_age_max, weight_mode=weight_mode)
    return evaluate_decisions_chunked(load_optimized_model(N, path), build_phase_cycle_hmm(p, q, world=world),
                                      horizon, threshold, world=world, **options)


def run_comparison(*, dimensions=(18, 100, 479), world=BASELINE_WORLD,
                   data_dir=DATA_DIR, output_dir=BASELINE_FIGURE_DIR, model_age_max=1000,
                   weight_mode="balanced", short_horizon=150, error_horizon=1000000,
                   decision_horizon=10000000, mode="fast", chunk_size=100000,
                   convergence_tol=1e-10, convergence_window=1000):
    """Compute each benchmark once, then export the six comparison figures."""
    if min(short_horizon, error_horizon, decision_horizon) < 1 or not dimensions:
        raise ValueError("Positive horizons and at least one model dimension are required")
    cycles_by_dimension = {q + 1: (p, q) for p, q in PHASE_CYCLES}
    if any(N not in cycles_by_dimension for N in dimensions):
        raise ValueError(f"Available rational-cycle dimensions: {tuple(cycles_by_dimension)}")
    # Resolve all model files before starting a long computation.
    fitted = {N: load_optimized_model(N, selected_model_path(N, data_dir=data_dir, world=world,
                        age_max=model_age_max, weight_mode=weight_mode)) for N in dimensions}
    cycles = {N: build_phase_cycle_hmm(*cycles_by_dimension[N], world=world) for N in dimensions}
    for N, cycle in cycles.items():
        check_model(cycle)
        if not np.allclose(S_cycle_from_hmm(cycle, 1000), S_cycle_formula(cycle, 1000), atol=1e-12):
            raise AssertionError(f"Phase-cycle formula and HMM disagree for N={N}")
    output_dir = Path(output_dir)
    output_paths = {}
    short_dimension = 100 if 100 in dimensions else dimensions[0]
    model = fitted[short_dimension]
    short_prediction = compute_predicted_quantities_from_DT(model["D_T"], model["b0"], short_horizon,
                                                            store_history=False)["S_pred"]
    print_fitted_hmm_convergence(short_dimension, short_horizon, short_prediction)
    short_curves = {"Rational cycle": S_cycle_formula(cycles[short_dimension], short_horizon),
                    "Best-found HMM": short_prediction,
                    "Age counter": S_age_counter(short_dimension, short_horizon, world=world)}
    fig = plot_short_horizon(np.arange(short_horizon + 1), true_curve(short_horizon, world=world), short_curves)
    output_paths["short_prediction"] = save_figure(fig, output_dir / "short_horizon_prediction.pdf")

    reference = true_curve(error_horizon, world=world)
    cumulative_errors = {}
    for N in dimensions:
        predictions = {"Age counter": S_age_counter(N, error_horizon, world=world),
                       "Rational cycle": S_cycle_formula(cycles[N], error_horizon),
                       "Best-found HMM": compute_predicted_quantities_from_DT(
                           fitted[N]["D_T"], fitted[N]["b0"], error_horizon, store_history=False)["S_pred"]}
        cumulative_errors[N] = {method: error_metrics(reference, prediction)["cumulative_mean_error"][1:]
                                for method, prediction in predictions.items()}
    bound = compute_theoretical_frdn_gap(world.lambda_val, world.alpha)["kappa_FRDN"]
    fig = plot_cumulative_errors(np.arange(1, error_horizon + 1), cumulative_errors, bound=bound)
    output_paths["prediction_error"] = save_figure(fig, output_dir / "cumulative_mean_error.pdf")
    statistics = {N: evaluate_decisions_chunked(fitted[N], cycles[N], decision_horizon,
                    world.lambda_val, world=world, mode=mode, chunk_size=chunk_size,
                    convergence_tol=convergence_tol, convergence_window=convergence_window,
                    verbose=True) for N in dimensions}
    suffix = "_fast" if mode == "fast" else ""
    for name, plot in (("wrong_action_frequency", plot_wrong_action_frequency_by_N),
                       ("wrong_action_frequency_by_method", plot_wrong_action_frequency_by_method),
                       ("decision_loss", plot_decision_loss_by_N),
                       ("decision_loss_by_method", plot_decision_loss_by_method)):
        output_paths[name] = save_figure(plot(statistics), output_dir / f"{name}{suffix}.pdf", dpi=200)
    return output_paths, statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[5, 50, 500])
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(BASELINE_FIGURE_DIR))
    parser.add_argument("--model-age", type=int, default=1000)
    parser.add_argument("--weight", choices=["balanced", "natural"], default="balanced")
    parser.add_argument("--short-horizon", type=int, default=150)
    parser.add_argument("--error-horizon", type=int, default=1000000)
    parser.add_argument("--decision-horizon", type=int, default=10000000)
    parser.add_argument("--mode", choices=["exact", "fast"], default="fast")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--convergence-tol", type=float, default=1e-10)
    parser.add_argument("--convergence-window", type=int, default=1000)
    parser.add_argument("--lambda-val", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=BASELINE_WORLD.alpha)
    parser.add_argument("--analytic-reset", action="store_true",
                        help="Use the analytic reset probability instead of the historical six-decimal value")
    args = parser.parse_args(argv)
    world = WorldConfig(args.lambda_val, args.alpha,
                        None if args.analytic_reset else BASELINE_WORLD.reset_probability)
    return run_comparison(dimensions=tuple(args.dimensions), world=world, data_dir=args.data_dir,
                          output_dir=args.output_dir, model_age_max=args.model_age, weight_mode=args.weight,
                          short_horizon=args.short_horizon, error_horizon=args.error_horizon,
                          decision_horizon=args.decision_horizon, mode=args.mode, chunk_size=args.chunk_size,
                          convergence_tol=args.convergence_tol, convergence_window=args.convergence_window)


if __name__ == "__main__":
    main()
