# Classical Memory Lower Bound — Code Release

This repository accompanies the paper:

> **Provably unbounded memory advantage in reinforcement learning with quantum world models**  
> Josep Lumbreras, Hailan Ma, Jayne Thompson, Mile Gu

## Overview

This code reproduces the numerical experiments on the **Fox–Rubin–Dharmadhikari–Nadkarni (FRDN)** process, a canonical non-Markovian renewal process whose conditional tick probability is provably non-approximable by any finite classical hidden Markov model (HMM).

The experiment trains K-state classical HMMs to approximate the FRDN conditional tick probability S(t) for varying K, and produces:

1. **Conditional tick probability plot** — predicted S_C(t) vs. the true S(t) for each K
2. **Cumulative action-value gap plot** — the uniform-average action-value gap over horizon T, benchmarked against the theoretical lower bound (R_T + C_B) · κ_FRDN

## File Structure

```
Release-code/
├── frdn_util.py   # FRDN analytical quantities: S(t), kappa_FRDN, h[l], etc.
├── train_util.py  # Classical HMM training (gradient descent on simplex) and evaluation
└── run.py         # Main experiment script — reproduces the paper figures
```

## Requirements

```
numpy
torch
scipy
matplotlib
```

Install via:

```bash
pip install numpy torch scipy matplotlib
```

## Running

```bash
cd Release-code
python run.py
```

This trains classical HMMs with N ∈ {5, 10, 50, 100, 500} states on the FRDN process with parameters λ = 0.4, α = π/√2, and saves outputs to:

| Path | Content |
|------|---------|
| `data/different_K_age1000_alpha_2.22_lambda_0.40_natural.pkl` | Raw numerical results |
| `figure/prob_different_K_age1000_alpha_2.22_lambda_0.40_natural.pdf` | Tick probability figure |
| `figure/value_different_K_age1000_alpha_2.22_lambda_0.40_natural.pdf` | Action-value gap figure |

## Citation

If you use this code, please cite:

```bibtex
@article{[key],
  title   = {Provably unbounded memory advantage in reinforcement learning with quantum world models},
  author  = {Lumbreras, Josep and Ma, Hailan and Thompson, Jayne and Gu, Mile},
  year    = {[Year]},
  journal = {[Venue]},
}
```
