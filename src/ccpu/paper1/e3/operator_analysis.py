"""Matched analysis and publication plot for operator-complexity pilots."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in indexed:
            raise ValueError(f"duplicate {label} prediction: {example_id}")
        indexed[example_id] = row
    return indexed


def _wilson(correct: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    radius /= denominator
    return [center - radius, center + radius]


def _exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def _correct(row: dict[str, Any]) -> bool:
    return bool(row["metrics"]["final_answer_correct"])


def _condition_summary(
    rows: dict[str, dict[str, Any]], eval_rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    correct = sum(_correct(row) for row in rows.values())
    generated_tokens = [int(row["generated_tokens"]) for row in rows.values()]
    by_operator: dict[str, dict[str, Any]] = {}
    for operator in sorted({str(row["operator_family"]) for row in eval_rows.values()}):
        identities = [
            example_id
            for example_id, row in eval_rows.items()
            if row["operator_family"] == operator
        ]
        operator_correct = sum(_correct(rows[example_id]) for example_id in identities)
        by_operator[operator] = {
            "count": len(identities),
            "correct": operator_correct,
            "accuracy": operator_correct / len(identities),
        }
    metrics = [row["metrics"] for row in rows.values()]
    available_stages = ("parse_valid", "lowerable_to_ccir", "type_valid", "executable")
    stages = {
        stage: sum(bool(metric.get(stage, metric.get("answer_scorable", False))) for metric in metrics)
        for stage in available_stages
        if any(stage in metric for metric in metrics)
    }
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "answer_wilson_95": _wilson(correct, len(rows)),
        "by_operator": by_operator,
        "stages": stages,
        "generated_tokens": {
            "mean": statistics.fmean(generated_tokens),
            "median": statistics.median(generated_tokens),
            "max": max(generated_tokens),
            "sum": sum(generated_tokens),
        },
    }


def _paired(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter()
    for example_id in left:
        left_correct = _correct(left[example_id])
        right_correct = _correct(right[example_id])
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif left_correct:
            counts["left_only"] += 1
        elif right_correct:
            counts["right_only"] += 1
        else:
            counts["both_wrong"] += 1
    return {
        **{key: counts[key] for key in ("both_correct", "left_only", "right_only", "both_wrong")},
        "exact_mcnemar_p": _exact_mcnemar(counts["left_only"], counts["right_only"]),
    }


def _plot(report: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("operator-pilot plotting requires matplotlib") from error

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbf7ef",
            "axes.facecolor": "#fbf7ef",
        }
    )
    conditions = ("direct", "ap", "as")
    labels = ("Direct", "Primitive ASL", "Semantic ASL")
    colors = ("#c55a35", "#147d84", "#d39447")
    preferred_order = ("square", "sqrt", "cube")
    observed = report["operator_counts"]
    operators = ("overall",) + tuple(
        operator for operator in preferred_order if operator in observed
    ) + tuple(sorted(set(observed) - set(preferred_order)))
    figure, (accuracy_axis, token_axis) = plt.subplots(
        1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    width = 0.24
    for condition_index, (condition, label, color) in enumerate(
        zip(conditions, labels, colors, strict=True)
    ):
        values = []
        for operator in operators:
            cell = report["conditions"][condition]
            values.append(cell["accuracy"] if operator == "overall" else cell["by_operator"][operator]["accuracy"])
        positions = [index + (condition_index - 1) * width for index in range(len(operators))]
        bars = accuracy_axis.bar(positions, values, width, color=color, label=label)
        accuracy_axis.bar_label(
            bars, labels=[f"{100 * value:.0f}" for value in values], padding=2, fontsize=8
        )
    accuracy_axis.set_xticks(range(len(operators)), [value.title() for value in operators])
    accuracy_axis.set_ylim(0, 1.10)
    accuracy_axis.set_ylabel("Final-answer accuracy")
    accuracy_axis.set_title(f"Matched {report['operator_level']} accuracy (%)")
    accuracy_axis.grid(axis="y", color="#d8d2c4", linewidth=0.8)

    token_values = [report["conditions"][condition]["generated_tokens"]["mean"] for condition in conditions]
    bars = token_axis.bar(labels, token_values, color=colors)
    token_axis.bar_label(bars, labels=[f"{value:.1f}" for value in token_values], padding=3)
    token_axis.set_yscale("log")
    token_axis.set_ylabel("Mean generated tokens (log scale)")
    token_axis.set_title("Generation cost")
    token_axis.tick_params(axis="x", rotation=25)
    token_axis.grid(axis="y", color="#d8d2c4", linewidth=0.8)
    handles, legend_labels = accuracy_axis.get_legend_handles_labels()
    figure.suptitle(
        f"Qwen3-0.6B one-seed {report['operator_level']} operator pilot",
        fontweight="bold",
        y=0.99,
    )
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze_operator_pilot(
    *,
    eval_path: str | Path,
    direct_predictions_path: str | Path,
    ap_predictions_path: str | Path,
    as_predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze a fully matched Direct/AP/AS operator pilot."""

    eval_rows = _index(read_jsonl(eval_path), "evaluation")
    predictions = {
        "direct": _index(read_jsonl(direct_predictions_path), "direct"),
        "ap": _index(read_jsonl(ap_predictions_path), "AP"),
        "as": _index(read_jsonl(as_predictions_path), "AS"),
    }
    expected = set(eval_rows)
    levels = {str(row["operator_level"]) for row in eval_rows.values()}
    if len(levels) != 1:
        raise ValueError(f"operator pilot must contain exactly one level, got {sorted(levels)}")
    operator_level = levels.pop()
    for condition, rows in predictions.items():
        if set(rows) != expected:
            missing = sorted(expected - set(rows))
            extra = sorted(set(rows) - expected)
            raise ValueError(f"{condition} identities differ: missing={missing[:3]} extra={extra[:3]}")

    as_failures = [row for row in predictions["as"].values() if not _correct(row)]
    binding_failures = sum(
        any("unresolved reference" in str(error) for error in row["metrics"].get("errors", []))
        for row in as_failures
    )
    report = {
        "schema_version": "ccpu.paper1.operator_pilot_analysis.v1",
        "operator_level": operator_level,
        "identity_count": len(expected),
        "operator_counts": dict(sorted(Counter(str(row["operator_family"]) for row in eval_rows.values()).items())),
        "conditions": {
            condition: _condition_summary(rows, eval_rows)
            for condition, rows in predictions.items()
        },
        "paired": {
            "ap_vs_direct": _paired(predictions["ap"], predictions["direct"]),
            "as_vs_direct": _paired(predictions["as"], predictions["direct"]),
            "as_vs_ap": _paired(predictions["as"], predictions["ap"]),
        },
        "as_failure_audit": {
            "incorrect": len(as_failures),
            "unresolved_reference": binding_failures,
            "unresolved_reference_rate_among_failures": binding_failures / len(as_failures) if as_failures else 0.0,
        },
        "inputs": {
            "eval": file_sha256(eval_path),
            "direct_predictions": file_sha256(direct_predictions_path),
            "ap_predictions": file_sha256(ap_predictions_path),
            "as_predictions": file_sha256(as_predictions_path),
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / f"operator_{operator_level.lower()}_pilot.png"
    _plot(report, figure_path)
    report["outputs"] = {"figure": file_sha256(figure_path)}
    write_json(output / "summary.json", report)
    with (output / "condition_by_operator.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("condition", "operator", "correct", "count", "accuracy"))
        for condition, summary in report["conditions"].items():
            writer.writerow((condition, "overall", summary["correct"], summary["count"], summary["accuracy"]))
            for operator, cell in summary["by_operator"].items():
                writer.writerow((condition, operator, cell["correct"], cell["count"], cell["accuracy"]))
    return report


def analyze_operator_ladder(
    summaries: list[tuple[str, str | Path]], output_dir: str | Path
) -> dict[str, Any]:
    """Build a categorical cross-level view from completed pilot summaries."""

    preferred_levels = ("O0", "O1", "O3", "O5", "O6")
    reports: dict[str, dict[str, Any]] = {}
    input_hashes = {}
    for label, path in summaries:
        report = read_json(path)
        level = str(report["operator_level"])
        if label != level:
            raise ValueError(f"summary label {label} does not match report level {level}")
        if level in reports:
            raise ValueError(f"duplicate operator summary: {level}")
        reports[level] = report
        input_hashes[level] = file_sha256(path)
    levels = [level for level in preferred_levels if level in reports]
    levels.extend(sorted(set(reports) - set(preferred_levels)))
    if not levels:
        raise ValueError("at least one operator summary is required")

    conditions = ("direct", "ap", "as")
    cells = {
        level: {
            condition: {
                "correct": reports[level]["conditions"][condition]["correct"],
                "count": reports[level]["conditions"][condition]["count"],
                "accuracy": reports[level]["conditions"][condition]["accuracy"],
                "mean_generated_tokens": reports[level]["conditions"][condition][
                    "generated_tokens"
                ]["mean"],
            }
            for condition in conditions
        }
        for level in levels
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "operator_ladder_pilots.png"
    _plot_ladder(levels, cells, figure_path)
    report = {
        "schema_version": "ccpu.paper1.operator_ladder_analysis.v1",
        "level_order": levels,
        "categorical_level_warning": (
            "O-levels are capability families, not equally spaced scalar complexity values"
        ),
        "cells": cells,
        "inputs": input_hashes,
        "outputs": {"figure": file_sha256(figure_path)},
    }
    write_json(output / "summary.json", report)
    return report


def _plot_ladder(
    levels: list[str], cells: dict[str, dict[str, dict[str, Any]]], output_path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("operator-ladder plotting requires matplotlib") from error
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbf7ef",
            "axes.facecolor": "#fbf7ef",
        }
    )
    styles = {
        "direct": ("Direct", "#c55a35", "o"),
        "ap": ("Primitive ASL", "#147d84", "s"),
        "as": ("Semantic ASL", "#d39447", "D"),
    }
    figure, (accuracy_axis, token_axis) = plt.subplots(1, 2, figsize=(10.2, 4.1))
    positions = list(range(len(levels)))
    for condition, (label, color, marker) in styles.items():
        accuracy = [cells[level][condition]["accuracy"] for level in levels]
        tokens = [cells[level][condition]["mean_generated_tokens"] for level in levels]
        accuracy_axis.plot(positions, accuracy, color=color, marker=marker, linewidth=2.4, label=label)
        token_axis.plot(positions, tokens, color=color, marker=marker, linewidth=2.4, label=label)
    for axis in (accuracy_axis, token_axis):
        axis.set_xticks(positions, levels)
        axis.grid(axis="y", color="#d8d2c4", linewidth=0.8)
    accuracy_axis.set_ylim(0, 1.05)
    accuracy_axis.set_ylabel("Final-answer accuracy")
    accuracy_axis.set_title("Accuracy by capability family")
    accuracy_axis.legend(frameon=False)
    token_axis.set_yscale("log")
    token_axis.set_ylabel("Mean generated tokens (log scale)")
    token_axis.set_title("Generation cost")
    figure.suptitle("Qwen3-0.6B matched operator pilots", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
