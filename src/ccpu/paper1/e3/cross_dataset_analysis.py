"""Matched analysis for Paper 1 cross-dataset transfer conditions."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    margin /= denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _condition_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    endpoints = (
        "parse_valid",
        "lowerable_to_ccir",
        "type_valid",
        "executable",
        "final_answer_correct",
    )
    counts = {
        endpoint: sum(bool(row["metrics"][endpoint]) for row in rows.values())
        for endpoint in endpoints
    }
    return {
        "count": total,
        "counts": counts,
        "rates": {endpoint: counts[endpoint] / total for endpoint in endpoints},
        "answer_wilson_95": _wilson(counts["final_answer_correct"], total),
        "generated_tokens_mean": mean(
            int(row.get("generated_tokens", 0)) for row in rows.values()
        ),
        "by_difficulty": {
            stratum: {
                "count": len(members),
                "correct": sum(
                    bool(row["metrics"]["final_answer_correct"]) for row in members
                ),
            }
            for stratum in sorted(
                {str(row["difficulty_stratum"]) for row in rows.values()}
            )
            if (
                members := [
                    row
                    for row in rows.values()
                    if str(row["difficulty_stratum"]) == stratum
                ]
            )
        },
    }


def _paired(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired transfer conditions have different identities")
    outcomes = Counter()
    for identity in left:
        left_correct = bool(left[identity]["metrics"]["final_answer_correct"])
        right_correct = bool(right[identity]["metrics"]["final_answer_correct"])
        label = (
            "both_correct"
            if left_correct and right_correct
            else "left_only"
            if left_correct
            else "right_only"
            if right_correct
            else "both_wrong"
        )
        outcomes[label] += 1
    total = len(left)
    return {
        **dict(outcomes),
        "delta_correct": outcomes["right_only"] - outcomes["left_only"],
        "delta_accuracy": (outcomes["right_only"] - outcomes["left_only"]) / total,
    }


def analyze_cross_dataset_transfer(
    *,
    manifest_paths: dict[str, str | Path],
    prediction_paths: dict[tuple[str, str], str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Analyze complete, identity-matched E0/E1/E2 predictions."""

    manifests = {dataset: read_json(path) for dataset, path in manifest_paths.items()}
    indexed: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    provenance = {}
    for (dataset, condition), path_value in prediction_paths.items():
        if dataset not in manifests:
            raise ValueError(f"predictions have no registered manifest: {dataset}")
        if condition not in {"E0", "E1", "E2"}:
            raise ValueError(f"unknown transfer condition: {condition}")
        path = Path(path_value)
        rows = read_jsonl(path)
        by_id = {str(row["example_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate prediction identity: {dataset}/{condition}")
        diagnostic_path = Path(manifest_paths[dataset]).parent / "diagnostic.jsonl"
        expected = {str(row["example_id"]) for row in read_jsonl(diagnostic_path)}
        missing = expected - set(by_id)
        unexpected = set(by_id) - expected
        if missing or unexpected:
            raise ValueError(
                f"incomplete {dataset}/{condition}: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        indexed.setdefault(dataset, {})[condition] = by_id
        provenance[f"{dataset}:{condition}"] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "diagnostic_sha256": file_sha256(diagnostic_path),
        }

    results = {}
    comparisons = {}
    for dataset, conditions in sorted(indexed.items()):
        results[dataset] = {
            condition: _condition_summary(rows)
            for condition, rows in sorted(conditions.items())
        }
        if "E0" in conditions:
            for condition in ("E1", "E2"):
                if condition in conditions:
                    comparisons[f"{dataset}:E0_to_{condition}"] = _paired(
                        conditions["E0"], conditions[condition]
                    )
        if "E1" in conditions and "E2" in conditions:
            comparisons[f"{dataset}:E2_to_E1"] = _paired(
                conditions["E2"], conditions["E1"]
            )

    all_conditions = sorted(
        {condition for conditions in indexed.values() for condition in conditions}
    )
    report = {
        "schema_version": "ccpu.paper1.cross_dataset_transfer_analysis.v1",
        "datasets": results,
        "paired_comparisons": comparisons,
        "macro_answer_accuracy": {
            condition: mean(
                results[dataset][condition]["rates"]["final_answer_correct"]
                for dataset in results
                if condition in results[dataset]
            )
            for condition in all_conditions
        },
        "condition_dataset_counts": {
            condition: sum(condition in results[dataset] for dataset in results)
            for condition in all_conditions
        },
        "manifests": {
            dataset: {"path": str(path), "sha256": file_sha256(path)}
            for dataset, path in manifest_paths.items()
        },
        "predictions": provenance,
    }
    write_json(output_path, report)
    return report


def plot_cross_dataset_transfer(
    *, analysis_path: str | Path, output_path: str | Path
) -> Path:
    """Plot matched answer and execution rates for every available condition."""

    import matplotlib.pyplot as plt

    report = read_json(analysis_path)
    datasets = list(report["datasets"])
    conditions = [
        condition
        for condition in ("E0", "E1", "E2")
        if any(condition in report["datasets"][dataset] for dataset in datasets)
    ]
    if not datasets or not conditions:
        raise ValueError("transfer analysis has no plottable conditions")
    colors = {"E0": "#0B6E75", "E1": "#D97706", "E2": "#2563A6"}
    labels = {"E0": "GSM LoRA", "E1": "GSM -> target", "E2": "Fresh target"}
    width = 0.72 / len(conditions)
    positions = list(range(len(datasets)))
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), sharey=True)
    for axis, (endpoint, title) in zip(
        axes,
        (("final_answer_correct", "Correct answers"), ("executable", "Executable ASL")),
        strict=True,
    ):
        for condition_index, condition in enumerate(conditions):
            offset = (condition_index - (len(conditions) - 1) / 2) * width
            values = [
                report["datasets"][dataset].get(condition, {}).get("rates", {}).get(
                    endpoint, 0.0
                )
                for dataset in datasets
            ]
            bars = axis.bar(
                [position + offset for position in positions],
                values,
                width=width * 0.92,
                color=colors[condition],
                label=labels[condition],
            )
            axis.bar_label(
                bars,
                labels=[f"{value * 100:.0f}" for value in values],
                padding=2,
                fontsize=8,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(positions, [name.replace("_", "-").upper() for name in datasets])
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", color="#D7D9D4", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Rate")
    axes[0].legend(frameon=False, ncol=min(3, len(conditions)), loc="upper left")
    figure.suptitle("Cross-dataset semantic compiler transfer", x=0.08, ha="left")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
