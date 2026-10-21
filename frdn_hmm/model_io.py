"""Parameter-only NPZ storage and metadata-based model selection."""

import json
from pathlib import Path
import numpy as np

from .config import DATA_DIR, BASELINE_WORLD, TrainingConfig
from .hmm import compute_predicted_quantities_from_DT, iter_probability_chunks


def save_optimized_model(model, *, lambda_val, alpha_val, age_max,
                         weight_mode, data_dir=DATA_DIR, training_config=None):
    """Save independent parameters and experiment metadata, without curves.

    D_T[next_state, current_state] is the Tick transition matrix; b0 is
    the reset distribution. Termination probabilities equal 1 - D_T.sum(0).
    Existing files are retained with numbered suffixes on new saves.
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    N = int(model["K"])
    stem = (f"classical_N{N}_age{age_max}_alpha_{alpha_val:.2f}"
            f"_lambda_{lambda_val:.2f}_{weight_mode}")
    if "training_ages" in model:
        stem = (f"classical_N{N}_sampled{len(model['training_ages'])}"
                f"_prefix{model['prefix_count']}_maxage{model['training_max_age']}"
                f"_seed{model['sampling_seed']}_plotage{age_max}"
                f"_alpha_{alpha_val:.2f}_lambda_{lambda_val:.2f}_{weight_mode}")
    fields = dict(schema_version=4, N=N, D_T=np.asarray(model["D_T"]),
                  b0=np.asarray(model["b0"]), lambda_val=lambda_val,
                  alpha_val=alpha_val, age_max=age_max, weight_mode=weight_mode)
    if "training_ages" in model:
        fields["training_ages"] = np.asarray(model["training_ages"], dtype=np.int64)
        fields.update(training_scheme="sampled", prefix_count=model["prefix_count"],
                      tail_count=model["tail_count"], training_max_age=model["training_max_age"],
                      sampling_seed=model["sampling_seed"])
    if "optimizer" in model:
        fields["optimizer"] = model["optimizer"]
        fields["adam_betas"] = np.asarray(model["adam_betas"], dtype=np.float64)
        if model["optimizer"] == "adam-metric":
            stem += f"_adam-metric_b2-{float(model['adam_betas'][1]):g}"
    if training_config is not None:
        fields["training_config_json"] = _training_signature(training_config)
    if "restart_diagnostics" in model:
        fields["training_diagnostics_json"] = json.dumps(model["restart_diagnostics"], sort_keys=True)
        fields["training_complete"] = model["training_complete"]
    index = 0
    while True:
        path = directory / f"{stem}{'_' + str(index) if index else ''}.npz"
        try:
            stream = path.open("xb")
        except FileExistsError:
            index += 1
            continue
        with stream:
            np.savez_compressed(stream, **fields)
        return path


def find_optimized_model(N, data_dir=DATA_DIR):
    """Find the newest contiguous-training file; sampled models need an explicit path."""
    candidates = list(Path(data_dir).glob(f"classical_N{N}_age*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No parameter file for N={N} in {data_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_optimized_model(N, data_path=None):
    """Load parameters and metadata; data_path can be an NPZ file or directory."""
    if data_path is None or Path(data_path).is_dir():
        data_path = find_optimized_model(N, DATA_DIR if data_path is None else data_path)
    with np.load(data_path, allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in saved.files}
    if int(model["N"]) != N:
        raise KeyError(f"Requested N={N}, but {data_path} contains N={int(model['N'])}")
    if model["D_T"].shape != (N, N) or model["b0"].shape != (N,):
        raise ValueError(f"Invalid parameter shapes for N={N} in {data_path}")
    return model


def load_optimized_curve(N, T=None, data_path=None):
    """Generate S_C(t), t=0..T, directly from a saved NPZ model.

    data_path can be a parameter file or directory. If omitted, select the
    newest parameter file for N in data/. T defaults to the training age_max;
    larger horizons are recomputed from D_T and b0 without cached curves.
    """
    if (
        isinstance(N, (bool, np.bool_))
        or not isinstance(N, (int, np.integer))
        or N <= 0
    ):
        raise ValueError("N must be a positive integer")
    if T is not None and (
        isinstance(T, (bool, np.bool_))
        or not isinstance(T, (int, np.integer))
        or T < 0
    ):
        raise ValueError("T must be a non-negative integer or None")

    model = load_optimized_model(N, data_path)

    return compute_predicted_quantities_from_DT(
        D_T=model["D_T"],
        b0=model["b0"],
        age_max=int(model["age_max"]) if T is None else T,
        store_history=False,
    )["S_pred"]


def iter_optimized_curve(N, T, data_path=None, chunk_size=100000, **options):
    """Load once, then stream prediction through the shared HMM evaluator."""
    model = load_optimized_model(N, data_path)
    yield from iter_probability_chunks(model["D_T"], model["b0"], T, chunk_size, **options)


def _training_signature(config):
    return json.dumps(config.trainer_kwargs(), sort_keys=True, separators=(",", ":"))


def model_matches(model, *, world, age_max, weight_mode, training=None):
    """Match full-precision metadata; rounded names are only for readability."""
    if not (int(model["age_max"]) == age_max and str(model["weight_mode"]) == weight_mode
            and np.isclose(float(model["lambda_val"]), world.lambda_val, rtol=0, atol=1e-12)
            and np.isclose(float(model["alpha_val"]), world.alpha, rtol=0, atol=1e-12)):
        return False
    if training is None:
        return True
    if not bool(model.get("training_complete", True)):
        return False
    return (str(model.get("optimizer", "adam")) == training.optimizer
            and np.allclose(model.get("adam_betas", (0.9, 0.999)), training.betas, rtol=0, atol=1e-12)
            and str(model.get("training_config_json", "")) == _training_signature(training))


def find_matching_model(N, *, data_dir=DATA_DIR, world=BASELINE_WORLD,
                        age_max=1000, weight_mode="balanced", training=None):
    """Select the newest matching contiguous model, including Adam-metric files."""
    candidates = sorted(Path(data_dir).glob(f"classical_N{N}_age*.npz"),
                        key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        model = load_optimized_model(N, path)
        if model_matches(model, world=world, age_max=age_max,
                         weight_mode=weight_mode, training=training):
            return path
    raise FileNotFoundError(f"No matching model for N={N} in {data_dir}")


def selected_model_path(N, data_path=None, **filters):
    """An explicit file overrides filters; a directory uses metadata matching."""
    if data_path is not None:
        path = Path(data_path)
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        filters["data_dir"] = path
    return find_matching_model(N, **filters)
