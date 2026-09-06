"""Publication plots for the main Paper 1 GSM8K results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


def _plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 1 result plots require matplotlib") from error
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbf7ef",
            "axes.facecolor": "#fbf7ef",
        }
    )
    return plt


def _prediction_rate(
    path: str | Path, *, parent_ids: set[str] | None = None
) -> dict[str, Any]:
    rows = read_jsonl(path)
    if parent_ids is not None:
        rows = [
            row
            for row in rows
            if str(row.get("parent_example_id", row["example_id"])) in parent_ids
        ]
    correct = sum(bool(row["metrics"]["final_answer_correct"]) for row in rows)
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "sha256": file_sha256(path),
    }


def _annotate_points(axis, x_values, y_values, *, color: str) -> None:
    for x_value, y_value in zip(x_values, y_values, strict=True):
        axis.annotate(
            f"{100 * y_value:.1f}%",
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=8,
        )


def _robustness_plot(report: dict[str, Any], output: Path) -> dict[str, Any]:
    plt = _plotting()
    metrics = report["large_number_robustness"]
    series = {
        "B1 direct": ("#c55a35", "direct_reasoning", "--"),
        "B1L direct": ("#8f351f", "direct_reasoning_long", "-"),
        "ASL seed 99173": ("#147d84", "seed11", "-"),
        "ASL seed 23": ("#3d9a83", "seed23", "-"),
        "ASL seed 37": ("#76ad78", "seed37", "-"),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.35))
    x_values = [0, 1]
    values = {}
    for label, (color, key, style) in series.items():
        y_values = [metrics[key]["original_eligible_rate"], metrics[key]["large_rate"]]
        values[label] = y_values
        width = 3.0 if label == "B1L direct" else 2.0
        axis.plot(
            x_values,
            y_values,
            marker="o",
            markersize=6,
            linewidth=width,
            linestyle=style,
            color=color,
            label=label,
        )
    asl_keys = ("seed11", "seed23", "seed37")
    asl_mean = [
        sum(metrics[key][field] for key in asl_keys) / len(asl_keys)
        for field in ("original_eligible_rate", "large_rate")
    ]
    axis.plot(
        x_values,
        asl_mean,
        color="#123f43",
        marker="D",
        markersize=7,
        linewidth=3.5,
        label="ASL mean",
        zorder=5,
    )
    _annotate_points(axis, x_values, values["B1L direct"], color="#8f351f")
    _annotate_points(axis, x_values, asl_mean, color="#123f43")
    axis.set_xticks(x_values, [r"$\times1$", r"$\times10^3$"])
    axis.set_ylim(0.25, 0.75)
    axis.set_ylabel("Final-answer accuracy")
    axis.set_title("ASL preserves accuracy as numeric magnitude increases")
    axis.grid(axis="y", color="#d8d2c4", linewidth=0.8)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {"series": values, "asl_mean": asl_mean}


def _model_size_plot(values: dict[str, list[float]], output: Path) -> None:
    plt = _plotting()
    figure, axis = plt.subplots(figsize=(7.2, 4.25))
    labels = ["Ordinary\n(250)", "Paired ordinary\n(59)", r"Paired $\times10^3$" + "\n(59)"]
    x_values = list(range(len(labels)))
    width = 0.34
    colors = {"Qwen3-0.6B": "#d39447", "Qwen3-1.7B": "#147d84"}
    for offset, (model, scores) in zip((-width / 2, width / 2), values.items(), strict=True):
        positions = [value + offset for value in x_values]
        bars = axis.bar(positions, scores, width, label=model, color=colors[model])
        axis.bar_label(bars, labels=[f"{100 * score:.1f}%" for score in scores], padding=3)
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0, 0.75)
    axis.set_ylabel("ASL-runtime final-answer accuracy")
    axis.set_title("A larger base model improves semantic compilation")
    axis.grid(axis="y", color="#d8d2c4", linewidth=0.8)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _failure_plot(summaries: list[tuple[str, dict[str, Any]]], output: Path) -> None:
    plt = _plotting()
    categories = (
        ("correct", "Correct", "#3d9a83"),
        ("wrong_semantic_structure", "Semantic structure", "#d39447"),
        ("runtime_failure", "Runtime/output", "#c55a35"),
        ("literal_copy_error", "Literal-only", "#657786"),
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    left = [0.0] * len(summaries)
    for key, label, color in categories:
        widths = [summary["counts"][key] / summary["counts"]["total"] for _, summary in summaries]
        bars = axis.barh(
            [name for name, _ in summaries], widths, left=left, color=color, label=label
        )
        for bar, width, (_, summary) in zip(bars, widths, summaries, strict=True):
            count = summary["counts"][key]
            if count:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if width > 0.07 else "#243238",
                )
        left = [start + width for start, width in zip(left, widths, strict=True)]
    axis.set_xlim(0, 1)
    axis.set_title(r"Most ASL $\times10^3$ failures are not literal-copy errors")
    axis.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    figure.subplots_adjust(bottom=0.27, left=0.16, right=0.98, top=0.84)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_gsm8k_result_plots(
    *,
    contribution_path: str | Path,
    large_eval_path: str | Path,
    small_original_predictions: str | Path,
    small_large_predictions: str | Path,
    large_original_predictions: str | Path,
    large_large_predictions: str | Path,
    failure_summaries: list[tuple[str, str | Path]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate the main result figures directly from frozen empirical artifacts."""

    contribution = read_json(contribution_path)
    large_eval = read_jsonl(large_eval_path)
    parent_ids = {str(row["parent_example_id"]) for row in large_eval}
    small = {
        "ordinary": _prediction_rate(small_original_predictions),
        "paired_ordinary": _prediction_rate(
            small_original_predictions, parent_ids=parent_ids
        ),
        "paired_large": _prediction_rate(small_large_predictions),
    }
    large = {
        "ordinary": _prediction_rate(large_original_predictions),
        "paired_ordinary": _prediction_rate(
            large_original_predictions, parent_ids=parent_ids
        ),
        "paired_large": _prediction_rate(large_large_predictions),
    }
    if small["ordinary"]["count"] != 250 or large["ordinary"]["count"] != 250:
        raise ValueError("model-size plot requires both complete 250-question predictions")
    for values in (small, large):
        if values["paired_ordinary"]["count"] != 59 or values["paired_large"]["count"] != 59:
            raise ValueError("model-size plot requires identical 59-parent predictions")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    robustness_path = output / "matched_magnitude_robustness.png"
    scaling_path = output / "asl_model_size_scaling.png"
    failure_path = output / "asl_large_number_failures.png"
    robustness = _robustness_plot(contribution, robustness_path)
    model_values = {
        "Qwen3-0.6B": [small[key]["accuracy"] for key in small],
        "Qwen3-1.7B": [large[key]["accuracy"] for key in large],
    }
    _model_size_plot(model_values, scaling_path)
    summaries = [(label, read_json(path)) for label, path in failure_summaries]
    _failure_plot(summaries, failure_path)
    report = {
        "schema_version": "ccpu.paper1.gsm8k_result_plots.v1",
        "inputs": {
            "contribution": file_sha256(contribution_path),
            "large_eval": file_sha256(large_eval_path),
            "failure_summaries": {
                label: file_sha256(path) for label, path in failure_summaries
            },
        },
        "robustness": robustness,
        "model_size": {"small": small, "large": large, "plot_values": model_values},
        "outputs": {
            "robustness": file_sha256(robustness_path),
            "model_size": file_sha256(scaling_path),
            "failures": file_sha256(failure_path),
        },
    }
    write_json(output / "summary.json", report)
    return report
