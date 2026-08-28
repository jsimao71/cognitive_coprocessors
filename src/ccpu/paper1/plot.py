"""Reproducible scaling figures generated from machine-readable summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def plot_scaling(summary: Mapping[str, Any], output: str | Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Scaling plots require: pip install -e ".[analysis]"') from error

    rows = list(summary.get("scaling", []))
    if not rows:
        raise ValueError("summary contains no scaling rows")
    models = sorted({str(row["model_id"]) for row in rows})
    digit_widths = sorted({int(row["operand_digits"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    grouped: dict[tuple[str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), int(row["operand_digits"]), str(row["condition"]))].append(
            (int(row["operator_count"]), float(row["accuracy"]))
        )

    figure, axes = plt.subplots(
        len(models),
        len(digit_widths),
        figsize=(3.4 * len(digit_widths), 2.8 * len(models)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for model_index, model_id in enumerate(models):
        for digit_index, digits in enumerate(digit_widths):
            axis = axes[model_index][digit_index]
            for condition in conditions:
                points = sorted(grouped.get((model_id, digits, condition), []))
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=condition,
                    )
            axis.set_title(f"{model_id}\n{digits}-digit operands", fontsize=9)
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.25)
            if model_index == len(models) - 1:
                axis.set_xlabel("operator count")
            if digit_index == 0:
                axis.set_ylabel("exact accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
        figure.subplots_adjust(top=0.84)
    figure.suptitle("Paper 1 arithmetic scaling", y=0.99)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
