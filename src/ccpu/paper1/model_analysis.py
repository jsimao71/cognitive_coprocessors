"""Cross-model Paper 1 comparison tables and figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, write_json

BLOCK_CONDITION = "calculator_block_icl_g"
INTERFACE_ORDER = (
    "llm_only",
    "matched_prompt",
    "explicit_tool",
    "normalized_reflex",
    BLOCK_CONDITION,
    "oracle",
)


def build_model_comparison(
    config: Mapping[str, Any], *, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Merge configured summaries while keeping evidence phases explicit."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for run in config.get("runs", []):
        summary_path = Path(str(run["summary"]))
        if not summary_path.is_absolute():
            summary_path = (config_path.parent / summary_path).resolve()
        summary = read_json(summary_path)
        label = str(run["model_label"])
        evidence_phase = str(run["evidence_phase"])
        sources.append(
            {
                "model_label": label,
                "evidence_phase": evidence_phase,
                "path": str(summary_path),
                "sha256": file_sha256(summary_path),
            }
        )
        for source_row in summary.get("by_run", []):
            condition = str(source_row["condition"])
            key = (label, evidence_phase, condition)
            if key in seen:
                raise ValueError(f"duplicate model/evidence/condition row: {key}")
            seen.add(key)
            rows.append(
                {
                    "model_label": label,
                    "model_id": str(source_row["model_id"]),
                    "parameter_billions": float(run["parameter_billions"]),
                    "evidence_phase": evidence_phase,
                    "condition": condition,
                    "arithmetic_count": int(source_row["arithmetic_count"]),
                    "accuracy": float(source_row["accuracy"]),
                    "block_execution_rate": source_row.get("block_execution_rate"),
                    "block_payload_semantically_equivalent_rate": source_row.get(
                        "block_payload_semantically_equivalent_rate"
                    ),
                    "false_block_rate": source_row.get("false_block_rate"),
                    "result_use_rate": source_row.get("result_use_rate"),
                    "mean_generated_tokens": float(source_row["mean_generated_tokens"]),
                    "mean_wall_time_ms": float(source_row["mean_wall_time_ms"]),
                    "block_failure_counts": source_row.get("block_failure_counts"),
                }
            )

    if not rows:
        raise ValueError("comparison config contains no summary rows")
    result = {
        "schema_version": "ccpu.paper1.model_comparison.v1",
        "config_sha256": file_sha256(config_path),
        "sources": sources,
        "rows": rows,
    }
    write_json(output_dir / "model_comparison.json", result)
    _plot_comparison(result, output_dir)
    return result


def _plot_comparison(comparison: Mapping[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            'Model comparison plots require: pip install -e ".[analysis]"'
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(comparison["rows"])
    block_rows = sorted(
        (row for row in rows if row["condition"] == BLOCK_CONDITION),
        key=lambda row: (row["parameter_billions"], row["evidence_phase"]),
    )
    _plot_block_capability(plt, block_rows, output_dir / "block_execution_capability.png")
    _plot_accuracy_interfaces(plt, rows, output_dir / "accuracy_interfaces.png")
    _plot_token_latency(plt, rows, output_dir / "token_latency_pareto.png")
    _plot_failure_heatmap(plt, block_rows, output_dir / "failure_mode_heatmap.png")


def _display_model(row: Mapping[str, Any]) -> str:
    suffix = "" if row["evidence_phase"] == "heldout" else " (smoke)"
    return f"{row['model_label']}{suffix}"


def _plot_block_capability(plt: Any, rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for row in rows:
        marker = "o" if row["evidence_phase"] == "heldout" else "s"
        axis.scatter(row["parameter_billions"], row["block_execution_rate"], s=75, marker=marker)
        axis.annotate(
            _display_model(row),
            (row["parameter_billions"], row["block_execution_rate"]),
            xytext=(5, -13),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0.9, color="#9B4A3F", linestyle="--", linewidth=1, label="90% threshold")
    axis.set(xlabel="model parameters (billions)", ylabel="valid block execution rate")
    axis.set_ylim(0, 1.08)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_accuracy_interfaces(plt: Any, rows: list[dict[str, Any]], output: Path) -> None:
    models = []
    for row in rows:
        key = (row["model_label"], row["evidence_phase"])
        if key not in models:
            models.append(key)
    width = 0.8 / max(len(models), 1)
    x_positions = list(range(len(INTERFACE_ORDER)))
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    for index, model in enumerate(models):
        selected = {
            row["condition"]: row
            for row in rows
            if (row["model_label"], row["evidence_phase"]) == model
        }
        values = [selected.get(condition, {}).get("accuracy", 0.0) for condition in INTERFACE_ORDER]
        positions = [value - 0.4 + width / 2 + index * width for value in x_positions]
        hatch = "//" if model[1] == "smoke" else None
        axis.bar(
            positions,
            values,
            width=width,
            label=_display_model(next(iter(selected.values()))),
            hatch=hatch,
        )
    axis.set_xticks(x_positions, [value.replace("_", "\n") for value in INTERFACE_ORDER])
    axis.set(ylabel="exact-answer accuracy", ylim=(0, 1.08))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8, ncol=len(models), loc="upper center")
    axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_token_latency(plt: Any, rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in rows if row["condition"] in INTERFACE_ORDER]
    conditions = list(INTERFACE_ORDER)
    colors = {condition: f"C{index}" for index, condition in enumerate(conditions)}
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for row in selected:
        marker = "o" if row["evidence_phase"] == "heldout" else "s"
        axis.scatter(
            row["mean_generated_tokens"],
            row["mean_wall_time_ms"] / 1000,
            color=colors[row["condition"]],
            marker=marker,
            alpha=0.82,
        )
    for condition in conditions:
        axis.scatter([], [], color=colors[condition], label=condition.replace("_", " "))
    axis.set(xlabel="mean generated tokens", ylabel="mean wall time (seconds)")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_failure_heatmap(plt: Any, rows: list[dict[str, Any]], output: Path) -> None:
    labels = [
        "execution miss",
        "semantic payload error",
        "false block",
        "result not used",
        "multiple blocks",
    ]
    values = []
    for row in rows:
        failures = row.get("block_failure_counts") or {}
        denominator = max(int(row["arithmetic_count"]), 1)
        values.append(
            [
                1 - float(row["block_execution_rate"]),
                1 - float(row["block_payload_semantically_equivalent_rate"]),
                float(row["false_block_rate"]),
                1 - float(row["result_use_rate"]),
                int(failures.get("multiple_blocks", 0)) / denominator,
            ]
        )
    figure, axis = plt.subplots(figsize=(8.8, 3.5))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(rows)), [_display_model(row) for row in rows])
    for row_index, row_values in enumerate(values):
        for column_index, value in enumerate(row_values):
            axis.text(column_index, row_index, f"{value:.0%}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="failure rate")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
