"""Paper 2 non-empirical depth-scaling diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_scaling(summary: dict[str, Any], output: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Plotting requires pip install -e ".[analysis]"') from error
    cells = summary["scaling_cells"]
    engines = sorted({row["engine"] for row in cells})
    figure, axes = plt.subplots(1, len(engines), figsize=(4.2 * len(engines), 3.8), sharey=True)
    if len(engines) == 1:
        axes = [axes]
    for axis, engine in zip(axes, engines, strict=True):
        for condition in ("no_engine", "single_calculator", "single_horn", "single_graph", "heterogeneous"):
            points = [
                row
                for row in cells
                if row["engine"] == engine and row["condition"] == condition and row["distractors"] == 4
            ]
            if points:
                axis.plot(
                    [row["depth"] for row in points],
                    [row["accuracy"] for row in points],
                    marker="o",
                    label=condition.replace("_", " "),
                )
        axis.set_title(engine.replace("_", " "))
        axis.set_xlabel("derivation depth")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("scripted task accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        fontsize=8,
    )
    figure.suptitle("Paper 2 protocol smoke (non-empirical)", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
