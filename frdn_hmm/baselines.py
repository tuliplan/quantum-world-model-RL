"""Analytic age-counter and rational phase-cycle classical models."""

import numpy as np

from .config import BASELINE_WORLD
from .frdn import f_alpha, tick_probabilities
from .hmm import compute_predicted_quantities_from_DT


def build_phase_cycle_hmm(p, q, *, world=BASELINE_WORLD):
    """Build reset + q cycle states for Wait, Probe and Maintain actions.

    Every matrix uses [next_state, current_state]. Wait advances the cycle on
    Tick; Probe resets on either outcome; Maintain always breaks and resets.
    """
    if p < 1 or q < 1 or np.gcd(p, q) != 1:
        raise ValueError("p and q must be positive and coprime")
    beta = 2.0 * np.pi * p / q
    N = q + 1
    reset = float(tick_probabilities(0, world=world))
    cycle_probabilities = f_alpha(np.arange(q) * beta, world=world)
    probabilities = np.r_[reset, cycle_probabilities]
    current_states = 1 + np.arange(q)
    next_states = 1 + (np.arange(q) + 1) % q

    tick_wait = np.zeros((N, N))
    tick_wait[1 + 1 % q, 0] = reset
    tick_wait[next_states, current_states] = cycle_probabilities
    break_wait = np.zeros((N, N))
    break_wait[0, :] = 1.0 - probabilities
    tick_probe = np.zeros((N, N))
    tick_probe[0, :] = probabilities
    break_maintain = np.zeros((N, N))
    break_maintain[0, :] = 1.0
    return dict(world=world, p=p, q=q, N=N, beta=beta, s_r=reset,
                s_cycle=cycle_probabilities, D_T_W=tick_wait, D_B_W=break_wait,
                D_T_P=tick_probe, D_B_P=break_wait.copy(),
                D_T_M=np.zeros((N, N)), D_B_M=break_maintain)


def check_model(model, tol=1e-12):
    """
    Check:
        - non-negativity
        - each action is column-normalized
    """

    action_pairs = {
        "Wait":
            (model["D_T_W"], model["D_B_W"]),

        "Probe":
            (model["D_T_P"], model["D_B_P"]),

        "Maintain":
            (model["D_T_M"], model["D_B_M"]),
    }

    for action, (D_T, D_B) in action_pairs.items():

        if np.min(D_T) < -tol or np.min(D_B) < -tol:
            raise ValueError(
                f"{action}: negative transition probability"
            )

        column_sums = np.sum(D_T + D_B, axis=0)

        if not np.allclose(
            column_sums,
            np.ones_like(column_sums),
            atol=tol
        ):
            raise ValueError(
                f"{action}: columns are not normalized."
            )

    return True


def S_age_counter(N, T, *, world=BASELINE_WORLD):
    """Exact age-counter curve: ages >= N-1 reuse the final state's prediction."""
    if N < 1 or T < 0:
        raise ValueError("N must be positive and T nonnegative")
    return tick_probabilities(np.minimum(np.arange(T + 1), N - 1), world=world)


def S_cycle_from_hmm(model, T):
    """Predict a phase cycle through the common conditional-belief recurrence."""
    b0 = np.zeros(model["N"])
    b0[0] = 1.0
    return compute_predicted_quantities_from_DT(
        model["D_T_W"], b0, age_max=T, store_history=False,
    )["S_pred"]


def S_cycle_formula(model, T):
    """Analytic phase-cycle prediction, also used to check the HMM recurrence."""
    world = model.get("world", BASELINE_WORLD)
    prediction = f_alpha(np.arange(T + 1) * model["beta"], world=world)
    prediction[0] = model["s_r"]
    return prediction
