"""Explicit world, training and evaluation settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "main-figure"
BASELINE_FIGURE_DIR = PROJECT_ROOT / "app-figure"
PHASE_CYCLES = ((6, 17), (35, 99), (169, 478))


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class WorldConfig:
    lambda_val: float = 0.4
    alpha: float = math.pi / math.sqrt(2)
    reset_probability: float | None = None

    def __post_init__(self):
        if not 0 < self.lambda_val < 1 or not math.isfinite(self.alpha):
            raise ValueError("lambda_val must lie in (0, 1) and alpha must be finite")
        if self.reset_probability is not None and not 0 <= self.reset_probability <= 1:
            raise ValueError("reset_probability must lie in [0, 1]")


# Preserve the rounded reset value used by the original baseline script.
BASELINE_WORLD = WorldConfig(reset_probability=0.455640)


@dataclass(frozen=True)
class TrainingConfig:
    age_max: int = 1000
    weight_mode: str = "balanced"
    num_restarts: int = 1
    steps: int = 3000
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    eps: float = 1e-9
    optimizer_name: str | None = None
    adam_betas: tuple[float, float] | None = None
    grad_clip_value: float | None = 1e4

    def __post_init__(self):
        for name, minimum in (("age_max", 0), ("num_restarts", 1), ("steps", 1), ("seed", 0)):
            _integer(getattr(self, name), name, minimum)
        if self.weight_mode not in ("balanced", "natural"):
            raise ValueError("weight_mode must be balanced or natural")
        if self.optimizer not in ("adam", "adam-metric"):
            raise ValueError("optimizer_name must be adam or adam-metric")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("adam_betas must contain two values in [0, 1)")
        if not math.isfinite(self.lr) or self.lr <= 0 or not 0 < self.eps < 0.5:
            raise ValueError("lr must be finite and positive; eps must lie in (0, 0.5)")
        if self.grad_clip_value is not None and (
                not math.isfinite(self.grad_clip_value) or self.grad_clip_value <= 0):
            raise ValueError("grad_clip_value must be finite and positive, or None")

    @property
    def optimizer(self):
        return self.optimizer_name or ("adam-metric" if self.weight_mode == "balanced" else "adam")

    @property
    def betas(self):
        if self.adam_betas is not None:
            return self.adam_betas
        return (0.9, 0.5) if self.optimizer == "adam-metric" else (0.9, 0.999)

    def trainer_kwargs(self):
        return dict(age_max=self.age_max, weight_mode=self.weight_mode,
                    num_restarts=self.num_restarts, steps=self.steps, lr=self.lr,
                    seed=self.seed, device=self.device, eps=self.eps,
                    optimizer_name=self.optimizer, adam_betas=self.betas,
                    grad_clip_value=self.grad_clip_value)


@dataclass(frozen=True)
class EvaluationConfig:
    horizon: int = 100000
    threshold: float = 0.4
    mode: str = "exact"
    chunk_size: int = 100000
    convergence_tol: float = 1e-10
    convergence_window: int = 1000
    tie_tol: float = 1e-12
    reward_scale: float = 2.0
    breakdown_cost: float = 1.0

    def __post_init__(self):
        _integer(self.horizon, "horizon", 1)
        _integer(self.chunk_size, "chunk_size", 1)
        _integer(self.convergence_window, "convergence_window", 2)
        if self.mode not in ("exact", "fast"):
            raise ValueError("mode must be exact or fast")
        if not math.isfinite(self.convergence_tol) or self.convergence_tol <= 0:
            raise ValueError("convergence_tol must be finite and positive")
        if not math.isfinite(self.tie_tol) or self.tie_tol < 0:
            raise ValueError("tie_tol must be finite and nonnegative")
        if not all(math.isfinite(value) for value in
                   (self.threshold, self.reward_scale, self.breakdown_cost)):
            raise ValueError("threshold and reward parameters must be finite")
