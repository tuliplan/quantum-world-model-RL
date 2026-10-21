"""Torch training with explicit target preparation, restarts and projection."""

import math
import warnings

import numpy as np
import torch

from .frdn import make_age_weights
from .hmm import compute_predicted_quantities_from_DT
from .evaluation import bernoulli_cross_entropy, bayes_cross_entropy


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


def project_weighted_simplex(values, metric):
    """Minimize sum_i metric_i * (x_i-values_i)^2, x>=0, sum(x)=1.

    Each row is independent. The KKT solution is max(values-tau/metric, 0).
    Sorting metric*values finds its active set without iterative root searches.
    Equal metrics recover the original Euclidean simplex projection.
    """
    if values.shape != metric.shape or values.ndim not in (1, 2):
        raise ValueError("values and metric must have identical 1D or 2D shapes")
    if not torch.all(torch.isfinite(metric)) or torch.any(metric <= 0):
        raise ValueError("projection metric must be finite and strictly positive")
    squeeze = values.ndim == 1
    v = values.unsqueeze(0) if squeeze else values
    w = metric.unsqueeze(0) if squeeze else metric
    # Multiplying all weights in one row by a scalar does not change the solution.
    w = w / w.max(dim=1, keepdim=True).values
    order = torch.argsort(w * v, dim=1, descending=True)
    sorted_v, sorted_w = v.gather(1, order), w.gather(1, order)
    thresholds = (sorted_v.cumsum(1) - 1) / (1 / sorted_w).cumsum(1)
    active = sorted_w * sorted_v > thresholds
    count = active.sum(1).clamp_min(1) - 1
    tau = thresholds.gather(1, count[:, None])
    projected = (v - tau / w).clamp_min(0)
    return projected.squeeze(0) if squeeze else projected


def adam_projection_metric(optimizer, parameter):
    """The same bias-corrected denominator used by the current Adam update."""
    state = optimizer.state[parameter]
    group = optimizer.param_groups[0]
    bias_correction = 1 - group["betas"][1] ** float(state["step"])
    return (state["exp_avg_sq"] / bias_correction).sqrt() + group["eps"]


def sampled_hazards_DT(D_T, b0, termination_prob, ages):
    """Differentiable hazards at sorted ages, using normalized binary powers."""
    gaps = np.diff(np.r_[0, ages])
    tiny = torch.finfo(D_T.dtype).tiny
    powers = [D_T / D_T.amax().clamp_min(tiny)]
    for _ in range(1, int(gaps.max()).bit_length()):
        squared = powers[-1] @ powers[-1]
        powers.append(squared / squared.amax().clamp_min(tiny))
    eta = b0
    hazards = []
    for gap in gaps:
        gap, bit = int(gap), 0
        while gap:
            if gap & 1:
                eta = powers[bit] @ eta
                eta = eta / eta.sum().clamp_min(tiny)
            gap >>= 1
            bit += 1
        hazards.append(termination_prob @ eta)
    return torch.stack(hazards)


def train_classical_hidden_model_DT(
    true,K,age_max=30,weight_mode="balanced",num_restarts=10,steps=200,
    lr=1e-3,seed=0,device="cpu",eps=1e-9,verbose=False, *,
    optimizer_name="adam", adam_betas=(0.9, 0.999), grad_clip_value=1e4,
):
    """Original trainer over every age 0..age_max."""
    return _train_classical_hidden_model_DT(
        true, K, age_max, weight_mode, num_restarts, steps, lr, seed,
        device, eps, verbose, optimizer_name=optimizer_name, adam_betas=adam_betas,
        grad_clip_value=grad_clip_value,
    )


def train_classical_hidden_model_DT_sampled(
    true,K,age_max=30,weight_mode="balanced",num_restarts=10,steps=200,
    lr=1e-3,seed=0,device="cpu",eps=1e-9,verbose=False, *, training_ages,
    optimizer_name="adam", adam_betas=(0.9, 0.999), grad_clip_value=1e4,
):
    """Train only at explicit ages, with balanced or natural weights."""
    if training_ages is None:
        raise ValueError("training_ages is required for sampled training")
    return _train_classical_hidden_model_DT(
        true, K, age_max, weight_mode, num_restarts, steps, lr, seed,
        device, eps, verbose, training_ages=training_ages,
        optimizer_name=optimizer_name, adam_betas=adam_betas,
        grad_clip_value=grad_clip_value,
    )


def _target_arrays(true, weight_mode, training_ages):
    """Align hazards, weights and optional sparse ages before entering Torch."""
    hazards = np.asarray(true["q"], dtype=np.float64)
    if training_ages is None:
        return hazards, make_age_weights(true["Phi"], len(hazards) - 1, mode=weight_mode), None
    ages = np.asarray(training_ages)
    if (ages.ndim != 1 or len(ages) == 0 or not np.issubdtype(ages.dtype, np.integer)
            or ages[0] < 0 or np.any(np.diff(ages) <= 0) or len(ages) != len(hazards)):
        raise ValueError("training_ages must be sorted unique nonnegative integers matching true['q']")
    if weight_mode == "balanced":
        weights = np.full(len(ages), 1.0 / len(ages))
    elif weight_mode == "natural":
        log_survival = np.asarray(true["log_Phi"], dtype=np.float64)
        if log_survival.shape != ages.shape or not np.all(np.isfinite(log_survival)):
            raise ValueError("Sampled natural weights require finite log_Phi aligned with training_ages")
        weights = np.exp(log_survival - log_survival.max())
        weights /= weights.sum()
    else:
        raise ValueError("weight_mode must be balanced or natural")
    return hazards, weights, ages


def _initialize_parameters(K, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    theta = project_simplex(torch.rand(K + 1, K, dtype=torch.float64, device=device).T).T
    beta = project_simplex(torch.rand(K, dtype=torch.float64, device=device))
    return theta.requires_grad_(True), beta.requires_grad_(True)


def _predict_hazards(probs, belief, age_max, eps, ages=None, grad_clip_value=None):
    """Differentiable conditional-belief recurrence for one training objective."""
    transition, termination = probs[1:, :], probs[0, :]
    if ages is not None:
        return sampled_hazards_DT(transition, belief, termination, ages)
    hazards = []
    for _ in range(age_max + 1):
        hazards.append(termination @ belief)
        continuation = transition @ belief
        belief = continuation / continuation.sum().clamp_min(eps)
        if grad_clip_value is not None and belief.requires_grad:
            # Bound adjoints within the long recurrence, before they can
            # overflow. Register only on fresh tensors, never the b0 leaf.
            belief.register_hook(lambda gradient: gradient.clamp(-grad_clip_value, grad_clip_value))
    return torch.stack(hazards)


@torch.no_grad()
def _prepare_gradients(parameters, clip_value):
    """Reject NaN/Inf before Adam; bound values without squaring a huge norm."""
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("non-finite gradient before Adam update")
    clipped = clip_value is not None and any(
        bool((gradient.abs() > clip_value).any()) for gradient in gradients)
    if clipped:
        # A gradient norm can itself overflow for otherwise finite 1e159 entries.
        torch.nn.utils.clip_grad_value_(parameters, clip_value)
    return clipped


@torch.no_grad()
def _project_parameters(optimizer, theta, beta, optimizer_name):
    if not all(bool(torch.isfinite(optimizer.state[parameter]["exp_avg_sq"]).all())
               for parameter in (theta, beta)):
        raise FloatingPointError("non-finite Adam second moment")
    if optimizer_name == "adam-metric":
        theta_metric = adam_projection_metric(optimizer, theta)
        beta_metric = adam_projection_metric(optimizer, beta)
        if not all(bool(torch.isfinite(metric).all()) for metric in (theta_metric, beta_metric)):
            raise FloatingPointError("non-finite Adam projection metric")
        theta.copy_(project_weighted_simplex(theta.T, theta_metric.T).T)
        beta.copy_(project_weighted_simplex(beta, beta_metric))
    else:
        theta.copy_(project_simplex(theta.T).T)
        beta.copy_(project_simplex(beta))
    if not all(bool(torch.isfinite(parameter).all()) for parameter in (theta, beta)):
        raise FloatingPointError("non-finite parameters after projection")


def _fit_restart(K, targets, weights, *, age_max, ages, steps, lr, seed,
                 device, eps, optimizer_name, adam_betas, grad_clip_value=1e4,
                 diagnostics=None):
    """Return the best evaluated parameters from one deterministic restart."""
    theta, beta = _initialize_parameters(K, seed, device)
    optimizer = torch.optim.Adam([theta, beta], lr=lr, betas=adam_betas)
    best_loss, best_theta, best_beta = float("inf"), None, None
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(seed=seed, completed_steps=0, clipped_steps=0, stopped_early=False, stop_reason="")
    for step in range(1, steps + 1):
        try:
            predicted = _predict_hazards(theta, beta, age_max, eps, ages,
                                         grad_clip_value=grad_clip_value).clamp(eps, 1.0 - eps)
            loss = -torch.sum(weights * (targets * torch.log(predicted)
                              + (1.0 - targets) * torch.log(1.0 - predicted)))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            # Preserve the finite candidate before backward/update can fail.
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_theta, best_beta = theta.detach().clone(), beta.detach().clone()
            optimizer.zero_grad()
            loss.backward()
            diagnostics["clipped_steps"] += int(_prepare_gradients((theta, beta), grad_clip_value))
            optimizer.step()
            _project_parameters(optimizer, theta, beta, optimizer_name)
            diagnostics["completed_steps"] = step
        except FloatingPointError as exc:
            diagnostics.update(stopped_early=True, stop_reason=f"step {step}: {exc}")
            if best_theta is None:
                raise
            warnings.warn(f"K={K}, seed={seed}, step={step}: {exc}; "
                          "returning the best finite parameters from this restart",
                          RuntimeWarning, stacklevel=2)
            break
    return best_theta, best_beta


def _restart_result(theta, beta, targets, weights, *, K, restart, age_max,
                    ages, eps, weight_mode, optimizer_name, adam_betas):
    """Convert a fitted restart into the existing parameter/statistics contract."""
    transition = theta[1:, :].cpu().numpy()
    termination = theta[0, :].cpu().numpy()
    belief = beta.cpu().numpy()
    if ages is None:
        prediction = compute_predicted_quantities_from_DT(transition, belief, age_max, eps=eps)
        hazards, survival = prediction["q_pred"], prediction["P_C"][:age_max + 1]
    else:
        with torch.no_grad():
            hazards = sampled_hazards_DT(theta[1:, :], beta, theta[0, :], ages).cpu().numpy()
        survival = None
    cross_entropy = bernoulli_cross_entropy(targets, hazards, weights)
    bayes_ce = bayes_cross_entropy(targets, weights)
    if not np.isfinite(cross_entropy):
        raise FloatingPointError("non-finite model evaluation after training")
    result = dict(K=K, restart=restart, D_T=transition, termination_prob=termination,
                  b0=belief, q_true=targets, qhat=hazards, P_vals=survival, weights=weights,
                  cross_entropy=cross_entropy, bayes_cross_entropy=bayes_ce,
                  kl_gap=cross_entropy - bayes_ce, weight_mode=weight_mode,
                  optimizer=optimizer_name, adam_betas=np.asarray(adam_betas), age_max=age_max)
    if ages is not None:
        result["training_ages"] = ages.copy()
    return result


def _train_classical_hidden_model_DT(
    true, K, age_max=30, weight_mode="balanced", num_restarts=10, steps=200,
    lr=1e-3, seed=0, device="cpu", eps=1e-9, verbose=False, *, training_ages=None,
    optimizer_name="adam", adam_betas=(0.9, 0.999), grad_clip_value=1e4,
):
    """Prepare targets, fit independent restarts and return the lowest-CE model.

    Each column of theta is a simplex over termination and K Tick outcomes;
    beta is the reset-distribution simplex. Projection follows each Adam step.
    """
    if optimizer_name not in ("adam", "adam-metric"):
        raise ValueError("optimizer_name must be adam or adam-metric")
    if K < 1 or steps < 1 or num_restarts < 1:
        raise ValueError("K, steps and num_restarts must be positive")
    if grad_clip_value is not None and (not math.isfinite(grad_clip_value) or grad_clip_value <= 0):
        raise ValueError("grad_clip_value must be finite and positive, or None")
    targets_np, weights_np, ages = _target_arrays(true, weight_mode, training_ages)
    if ages is None:
        age_max = len(targets_np) - 1
    targets = torch.tensor(targets_np, dtype=torch.float64, device=device)
    weights = torch.tensor(weights_np, dtype=torch.float64, device=device)
    best, restart_diagnostics = None, []
    for restart in range(num_restarts):
        diagnostics = dict(seed=seed + restart, completed_steps=0, clipped_steps=0,
                           stopped_early=False, stop_reason="")
        restart_diagnostics.append(diagnostics)
        try:
            theta, beta = _fit_restart(K, targets, weights, age_max=age_max, ages=ages,
                                      steps=steps, lr=lr, seed=seed + restart, device=device,
                                      eps=eps, optimizer_name=optimizer_name, adam_betas=adam_betas,
                                      grad_clip_value=grad_clip_value, diagnostics=diagnostics)
            result = _restart_result(theta, beta, targets_np, weights_np, K=K, restart=restart,
                                     age_max=age_max, ages=ages, eps=eps, weight_mode=weight_mode,
                                     optimizer_name=optimizer_name, adam_betas=adam_betas)
        except FloatingPointError as exc:
            diagnostics.update(stopped_early=True, stop_reason=str(exc))
            warnings.warn(f"K={K}, restart={restart}: {exc}; skipping this restart",
                          RuntimeWarning, stacklevel=2)
            continue
        if best is None or result["cross_entropy"] < best["cross_entropy"]:
            best = result
        if verbose:
            print(f"K={K}, restart={restart}, CE={result['cross_entropy']:.6f}, "
                  f"Bayes CE={result['bayes_cross_entropy']:.6f}, gap={result['kl_gap']:.6e}; "
                  f"steps={diagnostics['completed_steps']}/{steps}, clipped={diagnostics['clipped_steps']}",
                  flush=True)
    if best is None:
        raise FloatingPointError("All training restarts failed without a finite model")
    best.update(grad_clip_value=grad_clip_value, restart_diagnostics=restart_diagnostics,
                training_complete=all(not item["stopped_early"] for item in restart_diagnostics))
    return best
