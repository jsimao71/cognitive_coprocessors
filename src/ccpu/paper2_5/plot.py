"""Paper 2.5 source-count and quality/cost plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_scaling(summary: dict[str, Any], output: str | Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 2.5 plots require matplotlib") from error
    rows = summary["by_condition_source_count"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = {
        "oracle_need_source_query": "source-native oracle",
        "real_need_heuristic_source": "heuristic router",
        "explicit_source_selection": "explicit schemas",
        "universal_retriever": "universal text",
        "broadcast": "broadcast",
    }
    for condition, label in labels.items():
        cells = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: row["source_count"],
        )
        if not cells:
            continue
        axes[0].plot(
            [row["source_count"] for row in cells],
            [row["final_accuracy"] for row in cells],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [row["source_count"] for row in cells],
            [row["mean_descriptor_tokens"] for row in cells],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="available source types", ylabel="final accuracy", ylim=(0, 1.04))
    axes[1].set(xlabel="available source types", ylabel="mean source-description tokens")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def decide_gate(summary: dict[str, Any]) -> dict[str, Any]:
    cells = {
        (row["condition"], row["source_count"]): row
        for row in summary["by_condition_source_count"]
    }
    oracle = cells[("oracle_need_source_query", 4)]
    heuristic = cells[("real_need_heuristic_source", 4)]
    universal = cells[("universal_retriever", 4)]
    broadcast = cells[("broadcast", 4)]
    explicit_one = cells[("explicit_source_selection", 1)]
    explicit_four = cells[("explicit_source_selection", 4)]
    criteria = {
        "source_native_oracle_value": oracle["final_accuracy"] > universal["final_accuracy"],
        "heuristic_gap": oracle["final_accuracy"] > heuristic["final_accuracy"],
        "context_burden_growth": explicit_four["mean_descriptor_tokens"]
        > explicit_one["mean_descriptor_tokens"],
        "broadcast_not_dominant": broadcast["mean_source_calls"]
        > heuristic["mean_source_calls"]
        and broadcast["final_accuracy"] <= oracle["final_accuracy"],
    }
    return {
        "schema_version": "ccpu.paper2_5.paper3_5_gate.v1",
        "status": "go" if all(criteria.values()) else "no_go",
        "criteria": criteria,
        "decision_rule": "All four pre-registered criteria must pass before learned source routing.",
    }
