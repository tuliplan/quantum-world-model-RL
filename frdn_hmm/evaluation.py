"""Prediction, action-value and decision statistics, independent of plotting."""

import numpy as np

from .config import BASELINE_WORLD, EvaluationConfig, WorldConfig
from .frdn import tick_probabilities, f_alpha
from .hmm import compute_predicted_quantities_from_DT, iter_probability_chunks


def bernoulli_cross_entropy(q_true, q_pred, weights, eps=1e-12):
    q_pred = np.clip(q_pred, eps, 1.0 - eps)
    q_true = np.clip(q_true, eps, 1.0 - eps)

    ce = -np.sum(
        weights * (
            q_true * np.log(q_pred)
            + (1.0 - q_true) * np.log(1.0 - q_pred)
        )
    )
    return ce


def bayes_cross_entropy(q_true, weights, eps=1e-12):
    return bernoulli_cross_entropy(q_true, q_true, weights, eps=eps)


def action_from_gap(delta, tie_tol=1e-12):
    """
    Convert decision gap delta = Q_P - Q_M into action label.

    Returns
    -------
    action : array of same shape as delta
        +1 : choose Probe (P)
         0 : tie / undecided
        -1 : choose Maintain (M)
    """
    delta = np.asarray(delta)
    a = np.zeros_like(delta, dtype=int)
    a[delta > tie_tol] = 1
    a[delta < -tie_tol] = -1
    return a


def decision_metrics(S_true_curve, S_model_curve, lam, tie_tol=1e-12):
    """
    Given true S(t) and model S_C(t), compute the decision metrics
    for the Probe-vs-Maintain benchmark.

    Because of the reward construction:
        Delta_true(t) = S_true(t) - lam
        Delta_model(t) = S_model(t) - lam

    Returns
    -------
    dict with keys:
        delta_true
        delta_model
        a_true
        a_model
        wrong
        loss
        cum_wrong
        cum_loss
    """
    delta_true = S_true_curve - lam
    delta_model = S_model_curve - lam

    a_true = action_from_gap(delta_true, tie_tol=tie_tol)
    a_model = action_from_gap(delta_model, tie_tol=tie_tol)

    # wrong only if both sides are decisive and opposite
    wrong = (a_true != 0) & (a_model != 0) & (a_true != a_model)

    # For this theorem-backed P/M benchmark:
    # if the model picks the wrong side, the true decision loss is |S(t)-lam|
    loss = np.where(wrong, np.abs(delta_true), 0.0)

    # Uniform averages over ages 1,...,T; index zero is a placeholder.
    t_idx = np.arange(1, len(S_true_curve))
    cum_wrong = np.zeros_like(delta_true, dtype=float)
    cum_loss = np.zeros_like(delta_true, dtype=float)
    cum_wrong[1:] = np.cumsum(wrong[1:]) / t_idx
    cum_loss[1:] = np.cumsum(loss[1:]) / t_idx

    return {
        "delta_true": delta_true,
        "delta_model": delta_model,
        "a_true": a_true,
        "a_model": a_model,
        "wrong": wrong,
        "loss": loss,
        "cum_wrong": cum_wrong,
        "cum_loss": cum_loss,
    }


def error_metrics(S, S_C):
    """
    Pointwise and cumulative errors.

    cumulative_mean_error[T]
        = (1/T) sum_{t=1}^T |S(t)-S_C(t)|

    This matches the quantity used in the collaborator memo.
    """

    abs_error = np.abs(S - S_C)

    cumulative_mean_error = np.zeros_like(abs_error)

    cumulative_mean_error[0] = 0.0

    cumulative_mean_error[1:] = (
        np.cumsum(abs_error[1:])
        / np.arange(1, len(abs_error))
    )

    return {
        "abs_error": abs_error,
        "max_error": np.max(abs_error),
        "mean_error": np.mean(abs_error),
        "cumulative_mean_error":
            cumulative_mean_error,
    }


def tail_diagnostics(prediction, *, window=1000, tolerance=1e-10):
    """Report numerical tail stability; this is an estimate, not a convergence proof."""
    tail = prediction[-min(window, len(prediction)):]
    spread = float(np.ptp(tail))
    return dict(endpoint=float(prediction[-1]), window=len(tail), spread=spread,
                stable=len(tail) >= 2 and spread <= tolerance)


def print_fitted_hmm_convergence(N, T, prediction):
    tail = tail_diagnostics(prediction)
    status = "numerically stable (convergence estimate)" if tail["stable"] else "convergence not confirmed"
    print(f"Fitted HMM N={N}: S(t={T}) = {tail['endpoint']:.12f}; "
          f"last {tail['window']} ages: max-min = {tail['spread']:.3e}; {status}", flush=True)


def evaluate_model(model, *, plot_age_max, world=WorldConfig(), config=EvaluationConfig()):
    """Compute fitted-HMM curves and all training-experiment benchmark metrics."""
    if config.mode != "exact":
        raise ValueError("Use evaluate_decisions_chunked for fast tail approximation")
    horizon = max(plot_age_max, config.horizon)
    prediction = compute_predicted_quantities_from_DT(
        model["D_T"], model["b0"], age_max=horizon, store_history=False,
    )["S_pred"]
    predicted_eval = prediction[:config.horizon + 1]
    reference = tick_probabilities(np.arange(config.horizon + 1), world=world)
    # Preserve the gamma=0 Wait reward construction and its arithmetic.
    q_true = config.reward_scale * reference - config.breakdown_cost
    q_model = config.reward_scale * predicted_eval - config.breakdown_cost
    value_gap = np.cumsum(np.abs(q_true - q_model)[1:]) / np.arange(1, config.horizon + 1)
    return dict(probability=prediction[:plot_age_max + 1], value_gap=value_gap,
                decisions=decision_metrics(reference, predicted_eval, config.threshold, config.tie_tol),
                tail=tail_diagnostics(predicted_eval))


def evaluate_decisions_chunked(model, cycle, horizon, threshold=0.4, *, mode="exact",
                             convergence_tol=1e-10, convergence_window=1000,
                             chunk_size=100000, world=BASELINE_WORLD,
                             tie_tol=1e-12, verbose=False):
    """S_C(t) -> action -> wrong/loss, evaluated at every integer age.

    Only the returned plotting coordinates are sampled. Both cumulative
    statistics include all ages 1,...,T. mode="exact" uses direct HMM matrix
    recurrence throughout. mode="fast" approximates a numerically stable HMM
    tail by a constant probability/action, falling back to direct recurrence
    when stability or threshold separation is insufficient. Age counter,
    Rational cycle, and the true process are evaluated at every age in both
    modes. The returned hmm_evaluation records whether a tail was frozen.
    """
    N = cycle["N"]
    if (isinstance(horizon, (bool, np.bool_))
            or not isinstance(horizon, (int, np.integer)) or horizon < 1):
        raise ValueError("horizon must be a positive integer")
    if mode not in ("exact", "fast"):
        raise ValueError("mode must be 'exact' or 'fast'")
    durations = np.unique(np.concatenate((
        np.arange(1, min(1000, horizon) + 1),
        np.geomspace(1, horizon, 6000).astype(np.int64),
        10 ** np.arange(int(np.log10(horizon)) + 1),
        [horizon],
    )))
    methods = ("Age counter", "Rational cycle", "Best-found HMM")
    sampled = {method: {key: np.empty(len(durations))
                        for key in ("cum_wrong", "cum_loss")} for method in methods}
    wrong_totals = dict.fromkeys(methods, 0)
    loss_totals = dict.fromkeys(methods, 0.0)
    ties = dict.fromkeys(methods, 0)
    true_ties = 0
    hmm_evaluation = {}
    tail_reported = False
    for start, hmm_prediction in iter_probability_chunks(
        model["D_T"], model["b0"], horizon, chunk_size=chunk_size, mode=mode, threshold=threshold,
        convergence_tol=convergence_tol, convergence_window=convergence_window,
        diagnostics=hmm_evaluation,
    ):
        if verbose and hmm_evaluation["tail_start_age"] is not None and not tail_reported:
            print(
                f"N={N}, fast: approximate HMM tail from "
                f"t={hmm_evaluation['tail_start_age']:,}, "
                f"S_C={hmm_evaluation['tail_probability']:.12f}; "
                f"stable window={convergence_window}, tol={convergence_tol:g}",
                flush=True,
            )
            tail_reported = True
        ages = np.arange(start, start + len(hmm_prediction))
        # All metrics exclude age zero, including across chunk boundaries.
        if start == 0:
            ages = ages[1:]
            hmm_prediction = hmm_prediction[1:]
        if not len(ages):
            continue
        reference = tick_probabilities(ages, world=world)
        predictions = {
            "Age counter": tick_probabilities(np.minimum(ages, N - 1), world=world),
            "Rational cycle": f_alpha(ages * cycle["beta"], world=world),
            "Best-found HMM": hmm_prediction,
        }
        left, right = np.searchsorted(durations, [ages[0], ages[-1] + 1])
        displayed_ages = durations[left:right]
        true_ties += np.count_nonzero(action_from_gap(reference - threshold, tie_tol) == 0)
        for method, prediction in predictions.items():
            # Reuse the common decision definition. Index-zero placeholders
            # let decision_metrics treat this chunk as ages 1,...,chunk_length.
            result = decision_metrics(np.r_[threshold, reference],
                                      np.r_[threshold, prediction], threshold, tie_tol)
            wrong_sum = np.cumsum(result["wrong"][1:], dtype=np.int64)
            loss_sum = np.cumsum(result["loss"][1:])
            offsets = displayed_ages - ages[0]
            sampled[method]["cum_wrong"][left:right] = (
                wrong_totals[method] + wrong_sum[offsets]) / displayed_ages
            sampled[method]["cum_loss"][left:right] = (
                loss_totals[method] + loss_sum[offsets]) / displayed_ages
            wrong_totals[method] += int(wrong_sum[-1])
            loss_totals[method] += float(loss_sum[-1])
            ties[method] += np.count_nonzero(result["a_model"][1:] == 0)
        if verbose and ((ages[-1] + 1) % 1000000 == 0 or ages[-1] == horizon):
            print(f"N={N}, {mode}: metrics accumulated through t={ages[-1]:,}",
                  flush=True)

    if verbose:
        print(f"N={N}, {mode}: HMM direct ages={hmm_evaluation['exact_steps']:,}, "
              f"approximated ages={hmm_evaluation['approximated_steps']:,}", flush=True)
    return {"durations": durations, "methods": sampled,
            "hmm_evaluation": hmm_evaluation,
            "wrong_totals": wrong_totals, "loss_totals": loss_totals,
            "ties": ties, "true_ties": true_ties}
