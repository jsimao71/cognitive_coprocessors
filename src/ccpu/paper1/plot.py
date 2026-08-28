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
        figsize=(3.8 * len(digit_widths), 3.2 * len(models) + 0.7),
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
                        label=condition.replace("_", " "),
                    )
            axis.set_title(f"{model_id}\n{digits}-digit operands", fontsize=9)
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.25)
            if model_index == len(models) - 1:
                axis.set_xlabel("operator count")
            if digit_index == 0:
                axis.set_ylabel("exact accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.suptitle("Paper 1 arithmetic scaling", y=0.98)
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=min(5, len(labels)),
        )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_interface_diagnostics(summary: Mapping[str, Any], output: str | Path) -> Path:
    """Plot held-out accuracy and the decomposed assisted-interface pipeline."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Interface plots require: pip install -e ".[analysis]"') from error

    rows = {str(row["condition"]): row for row in summary.get("by_run", [])}
    order = [
        "llm_only",
        "matched_prompt",
        "explicit_tool",
        "reflex",
        "normalized_reflex",
        "calculator_block",
        "oracle",
    ]
    if any(condition not in rows for condition in order):
        raise ValueError("summary does not contain all held-out conditions")

    palette = ["#486B57", "#6D8B74", "#D18B47", "#A84A3F", "#247B7B", "#D3B04D", "#353C47"]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    accuracy = [float(rows[condition]["accuracy"]) for condition in order]
    low = [float(rows[condition]["accuracy_ci95"][0]) for condition in order]
    high = [float(rows[condition]["accuracy_ci95"][1]) for condition in order]
    labels = [condition.replace("_", "\n") for condition in order]
    axes[0].bar(
        labels,
        accuracy,
        color=palette,
        yerr=[
            [value - lower for value, lower in zip(accuracy, low, strict=True)],
            [upper - value for value, upper in zip(accuracy, high, strict=True)],
        ],
        capsize=3,
    )
    axes[0].set_title("Held-out exact-answer accuracy")
    axes[0].set_ylabel("accuracy with 95% Wilson interval")
    axes[0].set_ylim(0, 1.08)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    assisted = ["explicit_tool", "reflex", "normalized_reflex", "calculator_block", "oracle"]
    stages = [
        ("expression_exposure_rate", "exposure"),
        ("recognition_rate", "recognition"),
        ("normalization_stage_success_rate", "normalization"),
        ("execution_stage_success_rate", "execution"),
        ("reinjection_success_rate", "reinjection"),
        ("result_use_rate", "use if executed"),
    ]
    for condition in assisted:
        values = [
            float(rows[condition][key]) if rows[condition].get(key) is not None else 0.0
            for key, _ in stages
        ]
        axes[1].plot(
            [label for _, label in stages],
            values,
            marker="o",
            linewidth=2,
            label=condition.replace("_", " "),
        )
    axes[1].set_title("Assisted-interface pipeline")
    axes[1].set_ylabel("stage success rate")
    axes[1].set_ylim(0, 1.08)
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
