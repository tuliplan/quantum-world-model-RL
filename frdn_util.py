import numpy as np
from scipy.optimize import minimize_scalar

def frdn_f(x, lambda_val, alpha):
    """
    Continuous periodic phase function satisfying

        S(t) = f(t * alpha),  for t >= 1,

    where S(t) is the true FRDN conditional Tick probability.
    The reset-state value S(0) = Phi(1) is treated separately.
    """

    x = np.asarray(x)
    R = np.abs(1.0 - lambda_val * np.exp(1j * alpha))
    A = 1.0 / (2.0 * (1.0 - lambda_val))
    B = 1.0 / (2.0 * R)

    phi = np.angle(1.0 / (1.0 - lambda_val * np.exp(1j * alpha)))

    numerator = A - B * np.cos(x + alpha + phi)
    denominator = A - B * np.cos(x + phi)

    return lambda_val * numerator / denominator


def compute_theoretical_frdn_gap(lambda_val,alpha,num_grid=100_000,):
    """
    Compute:
        f_min
        f_max
        w_FRDN = f_max - f_min
        epsilon_FRDN = 0.5 * w_FRDN

    The coarse grid identifies the extrema neighborhoods.
    A local bounded optimizer then refines them.
    """

    xs = np.linspace(0.0,2.0 * np.pi,num_grid,endpoint=False,)

    fx = frdn_f(xs, lambda_val, alpha)

    idx_min = int(np.argmin(fx))
    idx_max = int(np.argmax(fx))

    dx = 2.0 * np.pi / num_grid

    # f is periodic, so optimization outside [0, 2pi] is harmless.
    def periodic_f(x):
        return float(
            frdn_f(
                np.mod(x, 2.0 * np.pi),
                lambda_val,
                alpha,
            )
        )

    x_min_seed = xs[idx_min]
    x_max_seed = xs[idx_max]

    min_result = minimize_scalar(
        periodic_f,
        bounds=(x_min_seed - 2 * dx, x_min_seed + 2 * dx),
        method="bounded",
    )

    max_result = minimize_scalar(
        lambda x: -periodic_f(x),
        bounds=(x_max_seed - 2 * dx, x_max_seed + 2 * dx),
        method="bounded",
    )

    f_min = float(min_result.fun)
    f_max = float(-max_result.fun)

    w_frdn = f_max - f_min
    epsilon_frdn = 0.5 * w_frdn

    fx = np.clip(fx, 0.0, 1.0)

    # For min_c E|X-c|, every median is an optimizer.
    # np.median returns one valid optimizer.
    c_star = float(np.median(fx))
    kappa_frdn = float(np.mean(np.abs(fx - c_star)))


    return {
        "f_min": f_min,
        "f_max": f_max,
        "x_at_min": float(np.mod(min_result.x, 2.0 * np.pi)),
        "x_at_max": float(np.mod(max_result.x, 2.0 * np.pi)),
        "w_FRDN": w_frdn,
        "epsilon_FRDN": epsilon_frdn,
        "c_star_FRDN": c_star,
        "kappa_FRDN": kappa_frdn,
    }
    
def compute_true_frdn_analytic(
    lambda_val=0.4,
    alpha=np.pi / np.sqrt(2),
    age_max=1000,
    eps=1e-15,
):
    """
    Compute analytical FRDN quantities without truncating the lifetime tail.

    Returns
    -------
    h : ndarray, shape (age_max + 1,)
        h[t] = P(L = t)

    Phi : ndarray, shape (age_max + 2,)
        Phi[t] = P(L >= t)

    log_Phi : ndarray, shape (age_max + 2,)
        Stable log survival tail.

    S_cond : ndarray, shape (age_max + 1,)
        S(t) = P(next Tick | already observed t Ticks)

    q : ndarray, shape (age_max + 1,)
        q(t) = P(next Breakdown | already observed t Ticks)
             = 1 - S(t)
    """

    if not (0.0 < lambda_val < 1.0):
        raise ValueError("lambda_val must lie in (0, 1).")

    ages_phi = np.arange(age_max + 2, dtype=np.float64)

    z = 1.0 - lambda_val * np.exp(1j * alpha)

    # For t >= 1:
    # Phi(t) = lambda^t * g(t)
    g = 0.5 * (
        1.0 / (1.0 - lambda_val)
        - np.real(np.exp(1j * ages_phi * alpha) / z)
    )

    g = np.maximum(g, eps)

    log_Phi = np.empty(age_max + 2, dtype=np.float64)
    log_Phi[0] = 0.0  # Phi(0) = 1

    t_positive = ages_phi[1:]
    log_Phi[1:] = (
        t_positive * np.log(lambda_val)
        + np.log(g[1:])
    )

    # Phi may underflow for large t. This is normal.
    Phi = np.exp(log_Phi)

    S_cond = np.empty(age_max + 1, dtype=np.float64)

    # Special case:
    # S(0) = Phi(1) / Phi(0) = Phi(1)
    S_cond[0] = Phi[1]

    # Stable formula for t >= 1:
    # S(t) = lambda * g(t+1) / g(t)
    S_cond[1:] = (
        lambda_val
        * g[2:age_max + 2]
        / g[1:age_max + 1]
    )

    S_cond = np.clip(S_cond, 0.0, 1.0)
    q = 1.0 - S_cond
    h = np.empty(age_max + 1, dtype=np.float64)

    # h(0) = 1 - Phi(1)
    h[0] = 1.0 - Phi[1]

    ages_h = np.arange(1, age_max + 1, dtype=np.float64)
    h[1:] = (lambda_val ** ages_h* np.sin(ages_h * alpha / 2.0) ** 2)

    return {
        "ages": np.arange(age_max + 1),
        "h": h,
        "Phi": Phi,
        "log_Phi": log_Phi,
        "S_cond": S_cond,
        "q": q,
    }



def make_age_weights(Phi, age_max, mode="balanced"):
    """
    balanced: uniform over ages 0,...,age_max
    natural: proportional to Phi(t), the natural context frequency
    """
    if mode == "balanced":
        w = np.ones(age_max + 1, dtype=np.float64)
    elif mode == "natural":
        w = Phi[:age_max + 1].copy()
    else:
        raise ValueError("mode must be 'balanced' or 'natural'.")

    w = w / w.sum()
    return w