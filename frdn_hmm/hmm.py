"""NumPy HMM prediction; no training or file-system dependencies."""

import numpy as np


def _tick_step(transition, belief):
    continuation = transition @ belief
    return continuation, float(continuation.sum())


def compute_predicted_quantities_from_DT(
    D_T,
    b0,
    age_max=1000,
    eps=1e-300,
    validate=True,
    store_history=True,
):
    """
    Stable evaluation of a finite classical hidden model.

    Classical hidden model:
        P_C(t) = 1^T D_T^t b0

    Conditional Tick probability:
        S_C(t) = P_C(t+1) / P_C(t)
               = 1^T D_T eta_t

    Breakdown hazard:
        q_C(t) = 1 - S_C(t)

    Conditional hidden belief:
        eta_t = D_T^t b0 / P_C(t)

    Parameters
    ----------
    D_T : np.ndarray, shape (K, K)
        Non-negative substochastic Tick-transition matrix.

    b0 : np.ndarray, shape (K,)
        Initial hidden-state distribution after reset.

    age_max : int
        Compute t = 0, ..., age_max.

    eps : float
        Only used to protect log operations and detect numerical failure.
        It is not used to replace denominators in the reported S_C(t).

    validate : bool
        Check whether D_T satisfies the substochastic constraints.

    store_history : bool
        Save every conditional belief. Disable for long-horizon curve loading
        to avoid allocating an array of shape (age_max + 2, K).

    Returns
    -------
    dict with:
        S_pred      : S_C(t), shape (age_max + 1,)
        q_pred      : q_C(t), shape (age_max + 1,)
        log_P_C     : log P_C(t), shape (age_max + 2,)
        P_C         : exp(log P_C(t)); may underflow to zero at large t
        eta_history : conditional hidden beliefs eta_t
        valid_until : largest age for which continuation remains positive
    """

    D_T = np.asarray(D_T, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)

    if D_T.ndim != 2 or D_T.shape[0] != D_T.shape[1]:
        raise ValueError("D_T must be a square matrix.")

    K = D_T.shape[0]

    if b0.shape != (K,):
        raise ValueError(f"b0 must have shape ({K},).")

    if b0.sum() <= 0:
        raise ValueError("b0 must have positive total mass.")

    b0 = b0 / b0.sum()

    if validate:
        tol = 1e-12

        if np.any(D_T < -tol):
            raise ValueError("D_T contains negative entries.")

        col_sums = D_T.sum(axis=0)

        if np.any(col_sums > 1.0 + tol):
            raise ValueError(
                "D_T is not substochastic: some column sums exceed 1."
            )

    eta = b0.copy()

    S_pred = []
    q_pred = []
    eta_history = [eta.copy()] if store_history else None

    # P_C(0) = b0 1 = 1
    log_P_C = [0.0]

    valid_until = age_max

    for t in range(age_max + 1):
        # S_C(t) = 1^T D_T eta_t
        cont, S_t = _tick_step(D_T, eta)

        # Numerical noise may cause tiny excursions outside [0, 1].
        if S_t < -1e-12 or S_t > 1.0 + 1e-12:
            raise FloatingPointError(
                f"Invalid continuation probability at age {t}: {S_t}"
            )

        S_t = float(np.clip(S_t, 0.0, 1.0))

        S_pred.append(S_t)
        q_pred.append(1.0 - S_t)

        if S_t <= 0.0:
            # The learned HMM assigns zero probability to any longer Tick run.
            # This already means it fails to represent the strictly positive
            # FRDN tail beyond age t.
            valid_until = t
            log_P_C.append(-np.inf)

            # Fill remaining outputs with NaN because S_C is undefined later.
            remaining = age_max - t

            if remaining > 0:
                S_pred.extend([np.nan] * remaining)
                q_pred.extend([np.nan] * remaining)
                log_P_C.extend([-np.inf] * remaining)

            break

        log_P_C.append(log_P_C[-1] + np.log(max(S_t, eps)))

        # eta_{t+1} = D_T eta_t / S_C(t)
        eta = cont / S_t
        if store_history:
            eta_history.append(eta.copy())

    log_P_C = np.asarray(log_P_C, dtype=np.float64)

    # This is useful for plotting the survival tail.
    # At large ages exp(log_P_C) may underflow to zero, which is acceptable.
    P_C = np.exp(log_P_C)

    return {
        "S_pred": np.asarray(S_pred, dtype=np.float64),
        "q_pred": np.asarray(q_pred, dtype=np.float64),
        "log_P_C": log_P_C,
        "P_C": P_C,
        "eta_history": np.asarray(eta_history, dtype=np.float64) if store_history else None,
        "valid_until": valid_until,
    }


def iter_probability_chunks(D_T, b0, T, chunk_size=100000, *,
                         mode="exact", threshold=0.4, convergence_tol=1e-10,
                         convergence_window=1000, diagnostics=None):
    """Yield (start_age, S_C_chunk), evaluating every age 0,...,T.

    mode="exact" preserves the conditional-belief recurrence at every age.
    mode="fast" freezes the prediction after a numerically stable window:
    probability range and belief L1 distance from the window's first belief
    must stay within convergence_tol, with probabilities safely away from
    threshold. This is a tail approximation, not a convergence proof. If no
    such window is found, all ages are computed directly.

    Chunking only limits memory. An optional diagnostics dict reports the
    number of direct/approximated ages and the start/value of a frozen tail.
    """
    if T < 0 or chunk_size <= 0:
        raise ValueError("T must be non-negative and chunk_size positive")
    if mode not in ("exact", "fast"):
        raise ValueError("mode must be 'exact' or 'fast'")
    if (not np.isfinite(convergence_tol) or convergence_tol <= 0
            or isinstance(convergence_window, (bool, np.bool_))
            or not isinstance(convergence_window, (int, np.integer))
            or convergence_window < 2):
        raise ValueError("convergence_tol must be positive and convergence_window >= 2")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    stats = diagnostics if diagnostics is not None else {}
    stats.update(mode=mode, exact_steps=0, approximated_steps=0,
                 tail_start_age=None, tail_probability=None,
                 convergence_tol=convergence_tol,
                 convergence_window=convergence_window)
    transition = np.asarray(D_T, dtype=np.float64)
    belief = np.asarray(b0, dtype=np.float64)
    N = len(belief)
    if transition.shape != (N, N):
        raise ValueError("D_T must be square and match b0")
    if (not np.all(np.isfinite(transition)) or np.any(transition < -1e-12)
            or np.any(transition.sum(axis=0) > 1 + 1e-12)
            or not np.all(np.isfinite(belief)) or np.any(belief < 0)
            or belief.sum() <= 0):
        raise ValueError(f"Invalid HMM parameters for N={N}")
    belief = belief / belief.sum()
    tail_probability = None
    stable_count = 0
    stable_belief = None
    stable_min = stable_max = None
    # decision_metrics uses a 1e-12 tie tolerance. Avoid extrapolating near ties.
    decision_margin = max(10 * convergence_tol, 1e-12)
    for start in range(0, T + 1, chunk_size):
        predictions = np.empty(min(chunk_size, T + 1 - start))
        if tail_probability is not None:
            predictions.fill(tail_probability)
            stats["approximated_steps"] += len(predictions)
            yield start, predictions
            continue
        for offset in range(len(predictions)):
            continuation, probability = _tick_step(transition, belief)
            if not np.isfinite(probability) or probability <= 0 or probability > 1 + 1e-12:
                raise ValueError(f"Invalid Tick probability at age {start + offset}")
            probability = min(probability, 1.0)
            predictions[offset] = probability
            belief = continuation / probability
            if mode == "fast":
                if abs(probability - threshold) <= decision_margin:
                    stable_count = 0
                    continue
                if (stable_count == 0
                        or max(stable_max, probability) - min(stable_min, probability)
                        > convergence_tol
                        or np.sum(np.abs(belief - stable_belief)) > convergence_tol):
                    stable_count = 1
                    stable_belief = belief.copy()
                    stable_min = stable_max = probability
                else:
                    stable_count += 1
                    stable_min = min(stable_min, probability)
                    stable_max = max(stable_max, probability)
                age = start + offset
                if stable_count >= convergence_window and age < T:
                    tail_probability = probability
                    stats["tail_start_age"] = age + 1
                    stats["tail_probability"] = tail_probability
                    predictions[offset + 1:] = tail_probability
                    stats["approximated_steps"] += len(predictions) - offset - 1
                    break
        stats["exact_steps"] += offset + 1
        yield start, predictions
