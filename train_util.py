import numpy as np
import torch


from frdn_util import make_age_weights
#compute_theoretical_frdn_gap, compute_true_frdn_analytic,frdn_f


def project_simplex(v, z=1.0):
    """
    Euclidean projection onto:
        {x : x >= 0, sum(x) = z}
    """
    if v.ndim == 1:
        v = v.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - z

    ind = torch.arange(
        1, v.shape[1] + 1,
        dtype=v.dtype,
        device=v.device
    ).unsqueeze(0)

    cond = u - cssv / ind > 0
    rho = cond.sum(dim=1) - 1

    theta = cssv[
        torch.arange(v.shape[0], device=v.device),
        rho
    ] / (rho.to(v.dtype) + 1.0)

    w = torch.clamp(v - theta.unsqueeze(1), min=0.0)

    return w.squeeze(0) if squeeze else w


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

def compute_predicted_quantities_from_DT(
    D_T,
    b0,
    age_max=1000,
    eps=1e-300,
    validate=True,
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

    ones = np.ones(K, dtype=np.float64)

    eta = b0.copy()

    S_pred = []
    q_pred = []
    eta_history = [eta.copy()]

    # P_C(0) = b0 1 = 1
    log_P_C = [0.0]

    valid_until = age_max

    for t in range(age_max + 1):
        cont = D_T @ eta

        # S_C(t) = 1^T D_T eta_t
        S_t = float(ones @ cont)

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
        "eta_history": np.asarray(eta_history, dtype=np.float64),
        "valid_until": valid_until,
    }


def train_classical_hidden_model_DT(
    true,K,age_max=30,weight_mode="balanced",num_restarts=10,steps=200,
    lr=1e-3,seed=0,device="cpu",eps=1e-9,verbose=False,
):
    """
    Optimize a K-state classical hidden model:

        b0 in simplex
        D_T >= 0, D_T 1 <= 1

    using the hazard loss:

        qhat_t = 1 - P_C(t+1) / P_C(t)
        loss = CE(Ber(q_t), Ber(qhat_t))

    Parameterization:
        column-wise projection over K+1 outcomes:
            probs[0, i]     = termination probability of state i
            probs[1:K+1, i] = D_T[:, i]
    """


    Phi = true["Phi"]
    q_true_np = true["q"]
    S = true["S_cond"]


    age_max = len(q_true_np) - 1   # sync with actual range from compute_true_hazard
    weights_np = make_age_weights(Phi, age_max, mode=weight_mode)
    q_true = torch.tensor(q_true_np, dtype=torch.float64, device=device)
    weights = torch.tensor(weights_np, dtype=torch.float64, device=device)
    bayes_ce = bayes_cross_entropy(q_true_np, weights_np)

    best = None

    for restart in range(num_restarts):

        best_local_loss = float("inf")
        best_local_theta = None
        best_local_beta = None

        torch.manual_seed(seed + restart)
        np.random.seed(seed + restart)

        # theta: K+1 rows, K cols; each column i has K+1 outcomes:
        # [termination_i, D_T[0,i], ..., D_T[K-1,i]]
        '''
        theta = 0.5 * torch.randn(K + 1, K, dtype=torch.float64, device=device)
        beta = 0.5 * torch.randn(K, dtype=torch.float64, device=device)
        '''

        theta = torch.rand(K + 1, K, dtype=torch.float64, device=device)
        theta = project_simplex(theta.T).T

        beta = torch.rand(K, dtype=torch.float64, device=device)
        beta = project_simplex(beta)

        theta.requires_grad_(True)
        beta.requires_grad_(True)

        optimizer = torch.optim.Adam([theta, beta], lr=lr)

        for step in range(steps):

            probs = theta
            b0 = beta

            D_T = probs[1:, :]

            termination_prob = probs[0, :]
            eta = beta
            qhats = []

            for _ in range(age_max + 1):
                qhat_t = termination_prob @ eta
                qhats.append(qhat_t)

                cont = D_T @ eta
                cont_prob = torch.clamp(cont.sum(), min=eps)
                eta = cont / cont_prob

            qhat = torch.stack(qhats)
            qhat = torch.clamp(qhat, min=eps, max=1.0 - eps)

            loss = -torch.sum(weights * (q_true * torch.log(qhat)+ (1.0 - q_true) * torch.log(1.0 - qhat)))

            optimizer.zero_grad()
            loss.backward()

            if loss.item() < best_local_loss:
                best_local_loss = loss.item()
                best_local_theta = theta.detach().clone()
                best_local_beta = beta.detach().clone()
        
            optimizer.step()

            with torch.no_grad():
                theta.copy_(project_simplex(theta.T).T)
                beta.copy_(project_simplex(beta))

        with torch.no_grad():
            probs = best_local_theta
            D_T_np = probs[1:, :].detach().cpu().numpy()
            term_np = probs[0, :].detach().cpu().numpy()
            b0_np = best_local_beta.detach().cpu().numpy()
        

        pred = compute_predicted_quantities_from_DT(D_T_np, b0_np, age_max=age_max, eps=eps)
        qhat_np = pred["q_pred"][:age_max + 1]
        P_vals_np = pred["P_C"][:age_max + 1]


        ce = bernoulli_cross_entropy(q_true_np, qhat_np, weights_np)
        gap = ce - bayes_ce

        result = {
            "K": K,
            "restart": restart,
            "D_T": D_T_np,
            "termination_prob": term_np,
            "b0": b0_np,
            "q_true": q_true_np,
            "qhat": qhat_np,
            "P_vals": P_vals_np,
            "weights": weights_np,
            "cross_entropy": ce,
            "bayes_cross_entropy": bayes_ce,
            "kl_gap": gap,
            "weight_mode": weight_mode,
            "age_max": age_max,
        }

        if best is None or result["cross_entropy"] < best["cross_entropy"]:
            best = result

        if verbose:
            print(f"K={K}, restart={restart}, CE={ce:.6f}, Bayes CE={bayes_ce:.6f}, gap={gap:.6e}")

    return best