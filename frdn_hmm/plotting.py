"""Figure builders consuming precomputed curves and benchmark statistics."""

from functools import wraps
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIG_W, FIG_H = 3.4, 2.45
FONT, TICK_FONT, LEGEND_FONT, PANEL_FONT = 18, 14, 13, 16
METHODS = ("Age counter", "Rational cycle", "Best-found HMM")
COLORS = ("tab:blue", "tab:orange", "tab:green")
LINESTYLES = ("-", "--", "-.")


def _with_style(style):
    """Scope figure typography to the call, leaving global rcParams untouched."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            fonts = (10, 9, 10) if style == "training" else (FONT, TICK_FONT, LEGEND_FONT)
            with plt.rc_context({"font.size": fonts[0], "axes.labelsize": fonts[0],
                                 "xtick.labelsize": fonts[1], "ytick.labelsize": fonts[1],
                                 "legend.fontsize": fonts[2], "pdf.fonttype": 42, "ps.fonttype": 42}):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def save_figure(fig, path, *, dpi=300):
    """Write a figure and close only that figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_path(data, prefix, output_dir):
    dimensions = "-".join(str(N) for N in data["K_list"])
    suffix = data.get("artifact_suffix") or f"_{data['weight_model']}"
    return Path(output_dir) / f"{prefix}_N{dimensions}{suffix}.pdf"


def _finish_training_panel(fig, ax):
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.legend(frameon=False, handlelength=1.4, handletextpad=0.4,
              borderpad=0.15, labelspacing=0.25)
    ax.grid(False)
    fig.tight_layout(pad=0.2)


def _duration_axis(ax, max_duration=None):
    ticks = [10, 1000, 100000]
    if max_duration is not None:
        ticks = [tick for tick in ticks if tick <= max_duration]
    ax.set_xscale("log")
    ax.set_xlabel("Benchmark duration $T$")
    ax.set_xticks(ticks)
    ax.set_xticklabels([fr"$10^{{{int(np.log10(tick))}}}$" for tick in ticks])


@_with_style("training")
def plot_different_K(data, *, output_dir="figure"):
    """Export conditional probabilities and average Q-gaps from existing data."""
    ages = np.arange(data["age_max"] + 1)[:100]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(ages, data["result_prob"][0][:100], label="Quantum",
            color="#d0d0d0", linewidth=1, zorder=1)
    for N in data["K_list"]:
        ax.plot(ages, data["result_prob"][N][:100], label=fr"$N={N}$",
                linewidth=1.5 if N == 5 else 1, zorder=3 if N == 5 else 2)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Conditional probability")
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.5, 1.0])
    _finish_training_panel(fig, ax)
    probability_path = save_figure(fig, _figure_path(data, "prob_unified", output_dir))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.axhline(data["value_bound"], color="red", linestyle="--", label="Bound")
    for N in data["K_list"]:
        ax.plot(data["T_vals"], data["result_value"][N], label=fr"$N={N}$", linewidth=1.5)
    _duration_axis(ax)
    ax.set_ylabel("Average $Q$-gap")
    ax.set_yticks([0, 0.25, 0.5])
    _finish_training_panel(fig, ax)
    value_path = save_figure(fig, _figure_path(data, "value_unified", output_dir))
    return {"probability": probability_path, "value": value_path}


@_with_style("training")
def plot_fitted_hmm_decision(data, *, output_dir="figure"):
    """Export wrong-action fractions and mean decision losses."""
    paths = {}
    for metric, prefix, ylabel in (
        ("cum_wrong", "wrong_action_fraction", "Wrong-action fraction"),
        ("cum_loss", "decision_loss", "Mean decision loss"),
    ):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for N in data["K_list"]:
            ax.plot(data["T_vals"], data["result_decision"][N][metric][1:],
                    label=fr"$N={N}$", linewidth=1.5)
        _duration_axis(ax, data["T_vals"][-1])
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(bottom=0)
        if metric == "cum_wrong":
            ax.set_yticks([0, 0.25, 0.5])
        _finish_training_panel(fig, ax)
        paths[metric] = save_figure(fig, _figure_path(data, f"{prefix}_fitted_HMM", output_dir))
    return paths


def label_panel(ax, panel):
    """Place the panel letter inside the upper-left corner of the axes."""
    ax.text(0.03, 0.97, f"({panel})", transform=ax.transAxes,
            ha="left", va="top", fontsize=PANEL_FONT, zorder=10)


@_with_style("baseline")
def plot_decision_figure(
    T,
    lam,
    model_curves,
    title=None,
    t_zoom=None,
    wrong_action_zoom=10000,
    figsize=(13, 8),
    *,
    S_true_curve,
    metrics_dict,
):
    """
    Plot the main decision figure from precomputed true curves and metrics.

    Parameters
    ----------
    T : int
        Maximum queried age
    lam : float
        Threshold lambda
    model_curves : dict
        {model_name: S_C_curve}
    title : str or None
    t_zoom : int or None
        If not None, panel (a) only shows t=0..t_zoom.
    wrong_action_zoom : int or None
        If not None, panel (b) shows wrong actions over
        t=0..wrong_action_zoom independently of panel (a).
    figsize : tuple
    """
    t = np.arange(T + 1)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # decide displayed horizon for top row
    if t_zoom is None:
        top_mask = slice(None)
        t_top = t
    else:
        top_mask = slice(0, t_zoom + 1)
        t_top = t[top_mask]
    # Panel (a): true and classical conditional probabilities
    ax1.plot(
        t_top,
        S_true_curve[top_mask],
        color="black",
        linewidth=2.5,
        label="True / qutrit",
    )

    # Store the curve colors so panel (b) uses exactly the same model colors.
    model_colors = {}
    for name, S_C_curve in model_curves.items():
        model_line, = ax1.plot(
            t_top,
            S_C_curve[top_mask],
            linestyle="--",
            linewidth=1.2,
            label=name,
        )
        model_colors[name] = model_line.get_color()

    ax1.axhline(
        lam,
        color="0.5",
        linestyle=":",
        linewidth=1,
        label=fr"Decision threshold $\lambda={lam:g}$",
    )
    ax1.set_xlabel("Clock age $t$", fontsize=FONT)
    ax1.set_ylabel(r"Conditional Tick probability $S(t)$")
    label_panel(ax1, "a")
    ax1.legend(loc="upper right", frameon=False, fontsize=LEGEND_FONT)
    # Panel (b): wrong-action locations for each classical model
    wrong_markers = ["o", "s", "^", "D", "v", "P", "X"]
    model_names = list(metrics_dict.keys())
    number_of_models = len(model_names)
    y_positions = np.arange(number_of_models - 1, -1, -1)

    if wrong_action_zoom is None:
        wrong_mask = slice(None)
        t_wrong = t
    else:
        wrong_horizon = min(wrong_action_zoom, T)
        wrong_mask = slice(0, wrong_horizon + 1)
        t_wrong = t[wrong_mask]

    for model_index, name in enumerate(model_names):
        wrong_show = metrics_dict[name]["wrong"][wrong_mask]
        y_position = y_positions[model_index]

        # A light guide line makes an empty (no-wrong-action) row visible.
        ax2.axhline(y_position, color="0.9", linewidth=0.8, zorder=0)
        ax2.scatter(
            t_wrong[wrong_show],
            np.full(np.count_nonzero(wrong_show), y_position, dtype=float),
            color=model_colors[name],
            marker=wrong_markers[model_index % len(wrong_markers)],
            s=38,
            linewidth=0.8,
            label=name,
            zorder=2,
        )

    ax2.set_xlim(t_wrong[0] - 1, t_wrong[-1] + 1)
    ax2.set_ylim(-0.5, number_of_models - 0.5)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel(r"Clock age $t$", fontsize=FONT)
    label_panel(ax2, "b")
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", alpha=0.2)
    # Panel (c): cumulative wrong-action fraction
    for name, m in metrics_dict.items():
        ax3.plot(t[1:], m["cum_wrong"][1:], linewidth=2, label=name)

    ax3.set_xscale("log")
    ax3.set_xlabel("Maximum queried age $T$", fontsize=FONT)
    ax3.set_ylabel("cumulative wrong-action fraction")
    label_panel(ax3, "c")
    # Panel (d): cumulative decision loss
    for name, m in metrics_dict.items():
        ax4.plot(t[1:], m["cum_loss"][1:], linewidth=2, label=name)

    ax4.set_xscale("log")
    ax4.set_xlabel("Maximum queried age $T$", fontsize=FONT)
    ax4.set_ylabel("Cumulative decision loss")
    label_panel(ax4, "d")

    if title is not None:
        fig.suptitle(title, fontsize=FONT)

    plt.tight_layout()
    #plt.show()

    return fig, metrics_dict


@_with_style("baseline")
def plot_metric_panels(metrics, *, metric, group_by, dimensions=None, horizons=None):
    """Shared layout for wrong-action/loss panels grouped by dimension or method."""
    if metric not in ("cum_wrong", "cum_loss") or group_by not in ("dimension", "method"):
        raise ValueError("Unknown metric or panel grouping")
    dimensions = tuple(metrics) if dimensions is None else tuple(dimensions)
    if not dimensions:
        raise ValueError("At least one model dimension is required")
    labels = dimensions if group_by == "dimension" else METHODS
    fig, axes = plt.subplots(1, len(labels), figsize=(12, 3.8), sharey=True,
                             sharex=group_by == "method", squeeze=False)
    axes = axes[0]
    for index, (ax, label) in enumerate(zip(axes, labels)):
        if group_by == "dimension":
            durations = metrics[label]["durations"]
            horizon = (horizons or {}).get(label, {18: 100000, 100: 1000000, 479: 10000000}.get(label, durations[-1]))
            horizon = min(horizon, int(durations[-1]))
            show = durations <= horizon
            for method, color, style in zip(METHODS, COLORS, LINESTYLES):
                ax.plot(durations[show], metrics[label]["methods"][method][metric][show],
                        color=color, linestyle=style, linewidth=1.6, label=method)
        else:
            horizon = max(int(metrics[N]["durations"][-1]) for N in dimensions)
            for position, N in enumerate(dimensions):
                ax.plot(metrics[N]["durations"], metrics[N]["methods"][label][metric],
                        color=COLORS[position % len(COLORS)],
                        linestyle=LINESTYLES[position % len(LINESTYLES)],
                        linewidth=1.6, label=fr"$N={N}$")
        label_panel(ax, chr(ord("a") + index))
        ax.set_xscale("log")
        ax.set_xlim(1, max(2, horizon))
        ax.set_xlabel(r"Benchmark duration $T$", fontsize=FONT)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        if metric == "cum_loss" and group_by == "method":
            ax.set_xticks([tick for tick in (10, 10000, 10000000) if tick <= horizon])
            ax.set_xticks([], minor=True)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Wrong-action fraction" if metric == "cum_wrong" else "Mean decision loss")
    axes[0].legend(loc="upper right", frameon=False, fontsize=LEGEND_FONT)
    fig.tight_layout()
    return fig


def plot_wrong_action_frequency_by_N(metrics, **options):
    return plot_metric_panels(metrics, metric="cum_wrong", group_by="dimension", **options)


def plot_wrong_action_frequency_by_method(metrics, **options):
    return plot_metric_panels(metrics, metric="cum_wrong", group_by="method", **options)


def plot_decision_loss_by_N(metrics, **options):
    return plot_metric_panels(metrics, metric="cum_loss", group_by="dimension", **options)


def plot_decision_loss_by_method(metrics, **options):
    return plot_metric_panels(metrics, metric="cum_loss", group_by="method", **options)


@_with_style("training")
def plot_short_horizon(ages, reference, model_curves):
    """Compare the true world with the three classical approximation methods."""
    fig, ax = plt.subplots(figsize=(2 * FIG_W, FIG_W))
    ax.plot(ages, reference, color="black", linewidth=2, label="True", zorder=4)
    styles = {
        "Rational cycle": dict(color="tab:orange", linestyle=(0, (6, 3)), marker="s", markevery=(20, 30)),
        "Best-found HMM": dict(color="tab:green", linestyle="-"),
        "Age counter": dict(color="tab:blue", linestyle=(0, (1.5, 2)), marker="^", markevery=(10, 30)),
    }
    for name, curve in model_curves.items():
        ax.plot(ages, curve, label=name, linewidth=1.4, **styles[name])
    ax.set_xlabel("Age $t$")
    ax.set_ylabel("Conditional Tick probability")
    ax.set_xlim(0, ages[-1])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout(pad=0.2)
    return fig


@_with_style("baseline")
def plot_cumulative_errors(durations, cumulative_errors, *, bound):
    """One prediction-error panel per classical model family."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, method, panel in zip(axes, METHODS, "abc"):
        for position, (N, methods) in enumerate(cumulative_errors.items()):
            ax.plot(durations, methods[method], color=COLORS[position % len(COLORS)],
                    linestyle=LINESTYLES[position % len(LINESTYLES)],
                    linewidth=1.6, label=fr"$N={N}$")
        label_panel(ax, panel)
        ax.axhline(bound, color="red", linestyle="--", linewidth=1.6,
                   label=r"$\kappa_{\mathrm{FRDN}}$")
        ax.set_xscale("log")
        ax.set_xlim(1, max(2, durations[-1]))
        ax.set_xlabel(r"Benchmark duration $T$")
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].set_ylim(bottom=0)
    axes[0].set_yticks([0, 0.2, 0.4])
    axes[0].set_ylabel("Mean prediction error")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig
