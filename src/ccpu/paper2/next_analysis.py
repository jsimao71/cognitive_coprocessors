"""Merge heterogeneous-engine runs and decide the explicit Paper 3 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, write_json


def analyze_runs(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_json(config_path)
    rows = []
    sources = []
    for run in config["runs"]:
        path = (config_path.parent / run["summary"]).resolve()
        summary = read_json(path)
        for cell in summary["by_condition_catalog"]:
            rows.append(
                {
                    "model_label": run["model_label"],
                    "placement": run["placement"],
                    **cell,
                }
            )
        sources.append({"path": str(path), "sha256": file_sha256(path)})
    multi_engine = [
        row
        for row in rows
        if row["catalog_size"] == 5 and row["placement"] == "weights"
    ]
    passing = [
        row
        for row in multi_engine
        if row["engine_selection_accuracy"] >= 0.8
        and row["false_activation_rate"] <= 0.1
        and row["runtime_exact_rate"] >= 0.8
    ]
    gate = {
        "schema_version": "ccpu.paper2.paper3_gate.v1",
        "status": "go" if len(passing) >= 2 else "no_go",
        "criterion": (
            "At least two model families must reach >=0.8 engine selection and runtime exact "
            "rates with <=0.1 false activation at five engines."
        ),
        "passing_models": [row["model_label"] for row in passing],
    }
    result = {
        "schema_version": "ccpu.paper2.next_analysis.v1",
        "rows": rows,
        "paper3_gate": gate,
        "sources": sources,
        "config_sha256": file_sha256(config_path),
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "next_analysis.json", result)
    write_json(output_dir / "paper3_gate.json", gate)
    _plot(result, output_dir / "capability_scaling.png")
    return result


def _plot(result: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 2 next-analysis plots require matplotlib") from error
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    grouped = {}
    for row in result["rows"]:
        key = (row["model_label"], row["placement"])
        grouped.setdefault(key, []).append(row)
    for (model, placement), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row["catalog_size"])
        label = f"{model} / {placement}"
        axes[0].plot(
            [row["catalog_size"] for row in ordered],
            [row["engine_selection_accuracy"] for row in ordered],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [row["catalog_size"] for row in ordered],
            [row["mean_prompt_tokens"] for row in ordered],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="enabled engines", ylabel="engine selection accuracy", ylim=(0, 1.04))
    axes[1].set(xlabel="enabled engines", ylabel="mean prompt tokens")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
