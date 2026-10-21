# FRDN World Models

Benchmarks for classical hidden-state world models (HMMs) on the analytic FRDN process introduced by [Fanizza, Lumbreras & Winter (2024)](https://doi.org/10.1007/s00220-023-04913-4). The code trains finite-state HMMs to predict the process's conditional Tick probability, compares them with age-counter and rational phase-cycle baselines, and evaluates prediction error and downstream decision quality over long horizons.

## Installation

Requires Python ≥ 3.8.

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `torch`. Torch is used only for training; prediction, evaluation and model loading need only NumPy.

## Quick Start

Train an 18-state HMM on ages `0..20`:

```bash
python run.py --dimensions 18 --train-age 20 --eval-horizon 200 --steps 20 --restarts 1
```

Compare the trained model with the baselines:

```bash
python compare_baselines.py --dimensions 18 --model-age 20 \
  --short-horizon 150 --error-horizon 1000 --decision-horizon 1000 --mode exact
```

These short settings only check that the pipeline runs. Run either script with `--help` to see every option and the full-scale defaults.

| Output | Default location | Override |
|---|---|---|
| Trained models (`.npz`) | `data/` | `--data-dir` |
| Training figures | `main-figure/` | `--figure-dir` |
| Baseline comparison figures | `app-figure/` | `--output-dir` |

Paths are relative to the project root, not the current working directory.

## Pretrained Models

To load existing models and generate the four figures containing all three dimensions:

```bash
python run.py --evaluate-only --dimensions 18 100 479 --train-age 1000
```

Evaluation-only matches the world parameters, dimension, training age and weighting, allowing models trained with different optimizer settings. It checks that every model exists before evaluation and never starts training. Figures are written to `main-figure/` with an `_existing` filename suffix. This option cannot be combined with `--force-retrain`.

Trained HMMs are included in `data/` (balanced weighting, Adam-metric optimizer, `lambda = 0.4`). Scripts load a matching model automatically, so no training is needed to reproduce the comparison:

```bash
python compare_baselines.py --dimensions 18 100 --model-age 1000
```

Model files are named `classical_N{states}_age{training age}_alpha_{alpha}_lambda_{lambda}_{weighting}_{optimizer}_b2-{beta2}.npz`, e.g. `classical_N18_age1000_alpha_2.22_lambda_0.40_balanced_adam-metric_b2-0.5.npz`.

## Project Structure

```text
├── run.py                 # Contiguous-age HMM training and figures
├── compare_baselines.py   # Age-counter vs. phase-cycle vs. fitted HMM comparison
├── frdn_hmm/
│   ├── config.py          # World, training and evaluation configs
│   ├── frdn.py            # Analytic FRDN probabilities, survival and theoretical bounds
│   ├── hmm.py             # Belief recursion and chunked prediction (NumPy)
│   ├── baselines.py       # Age-counter and rational phase-cycle models
│   ├── training.py        # Losses, projected optimization, restarts (Torch)
│   ├── evaluation.py      # Prediction error, Q-gap and decision metrics
│   ├── model_io.py        # NPZ save/load and config matching
│   └── plotting.py        # Figures from precomputed results
└── tests/
```

Importing any module has no side effects: training, experiments and plotting run only when called explicitly.

## Python API

```python
from frdn_hmm.config import TrainingConfig, EvaluationConfig
from run import _different_K

paths = _different_K(
    K_list=(18,),
    training=TrainingConfig(age_max=100, steps=100, num_restarts=1),
    evaluation=EvaluationConfig(horizon=1000),
)
```

## Training

- **Weighting and optimizer.** `balanced` weighting uses Adam with metric projection, betas `(0.9, 0.5)`; `natural` uses standard Adam, betas `(0.9, 0.999)`. Override with `--optimizer` and `--beta2`.
- **Gradient clipping.** Gradients through the belief recursion and parameter gradients before each Adam step are clipped element-wise to `±1e4` (`--grad-clip-value`; set `grad_clip_value=None` in the API to disable). This prevents overflow without changing the forward model or loss.
- **Robust restarts.** If a restart hits a non-finite loss, gradient or optimizer state, it warns and keeps its best valid parameters, or is skipped if it has none. Per-restart diagnostics are stored in `training_diagnostics_json`. Training raises an error only if every restart fails.
- **Caching.** Models are reused when the world parameters, dimension, training age, weighting and full training config all match. Use `--force-retrain` to always train a new model.

## Conventions

- `D_T[next_state, current_state]` is column-substochastic; `b0` is the reset distribution.
- `S(t)` is the conditional Tick probability, `q(t) = 1 - S(t)`, and `Phi(t)` is the survival probability.
- `t = 0` uses the reset probability. Training uses its analytic value; baselines default to `0.455640` (pass `--analytic-reset` to use the analytic value).
- Cumulative error, wrong-action fraction and decision loss are summed over ages `1..T`; ties use `tie_tol = 1e-12`.
- `--mode exact` evaluates every age. `--mode fast` freezes the tail once probabilities and beliefs have converged away from the decision threshold, and reports diagnostics for the approximation.
- If an HMM can no longer emit a Tick, full prediction returns `NaN` beyond `valid_until`, and chunked iteration raises an error.

## Tests

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

## Reference

The FRDN process is from:

> M. Fanizza, J. Lumbreras, and A. Winter, "Quantum Theory in Finite Dimension Cannot Explain Every General Process with Finite Memory," *Communications in Mathematical Physics* (2024). [doi:10.1007/s00220-023-04913-4](https://doi.org/10.1007/s00220-023-04913-4) · [arXiv:2209.11225](https://arxiv.org/abs/2209.11225)

```bibtex
@article{fanizza2024quantum,
  title   = {Quantum Theory in Finite Dimension Cannot Explain Every General Process with Finite Memory},
  author  = {Fanizza, Marco and Lumbreras, Josep and Winter, Andreas},
  journal = {Communications in Mathematical Physics},
  year    = {2024},
  doi     = {10.1007/s00220-023-04913-4},
  eprint  = {2209.11225},
  archivePrefix = {arXiv}
}
```
