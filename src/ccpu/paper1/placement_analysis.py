"""Compare context, weight, and runtime placement of interface knowledge."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, write_json

PLACEMENT_ORDER = (
    "base_minimal",
    "context",
    "weights",
    "weights_plus_context",
    "runtime",
    "explicit_tool",
    "oracle",
)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def build_placement_comparison(
    config: Mapping[str, Any], *, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in config.get("rows", []):
        summary_path = _resolve(config_path, str(entry["summary"]))
        summary = read_json(summary_path)
        condition = str(entry["condition"])
        matches = [row for row in summary["by_run"] if row["condition"] == condition]
        if len(matches) != 1:
            raise ValueError(f"expected one {condition} row in {summary_path}")
        source = matches[0]
        model_label = str(entry["model_label"])
        placement = str(entry["placement"])
        key = (model_label, placement)
        if key in seen:
            raise ValueError(f"duplicate placement row: {key}")
        seen.add(key)
        false_intervention = source.get("false_block_rate")
        if false_intervention is None:
            false_intervention = source.get("trigger", {}).get("false_intervention_rate")
        interface_success = source.get("block_execution_rate")
        if interface_success is None:
            interface_success = source.get("execution_stage_success_rate")
        rows.append(
            {
                "model_label": model_label,
                "model_id": str(source["model_id"]),
                "placement": placement,
                "condition": condition,
                "interface_success_rate": interface_success,
                "answer_accuracy": float(source["accuracy"]),
                "false_intervention_rate": false_intervention,
                "result_use_rate": source.get("result_use_rate"),
                "mean_prompt_tokens": float(source["mean_prompt_tokens"]),
                "mean_generated_tokens": float(source["mean_generated_tokens"]),
                "mean_wall_time_ms": float(source["mean_wall_time_ms"]),
                "arithmetic_count": int(source["arithmetic_count"]),
            }
        )
        sources.append(
            {
                "model_label": model_label,
                "placement": placement,
                "path": str(summary_path),
                "sha256": file_sha256(summary_path),
            }
        )

    training = []
    for entry in config.get("training_reports", []):
        report_path = _resolve(config_path, str(entry["report"]))
        report = read_json(report_path)
        training.append(
            {
                "model_label": str(entry["model_label"]),
                "adapter_id": report["adapter_id"],
                "trainable_parameters": report["trainable_parameters"],
                "trainable_fraction": report["trainable_fraction"],
                "training_target_tokens": report["training_target_tokens"],
                "wall_time_seconds": report["wall_time_seconds"],
                "peak_memory_bytes": report["peak_memory_bytes"],
                "report_path": str(report_path),
                "report_sha256": file_sha256(report_path),
            }
        )
    if not rows:
        raise ValueError("placement config contains no rows")
    result = {
        "schema_version": "ccpu.paper1.placement_comparison.v1",
        "question": "Where should interface knowledge live: in context, in weights, or in the runtime?",
        "config_sha256": file_sha256(config_path),
        "sources": sources,
        "training": training,
        "rows": rows,
        "interpretation": {
            "selection_and_serialization": "adapter_weights_with_a_short_explicit_contract",
            "exact_computation": "deterministic_runtime_with_enforced_result_use",
            "context_role": "development_and_cold_start_with_recurring_token_and_safety_cost",
            "scope": "three developmental small-model families on one 16-arithmetic/12-control set",
        },
    }
    write_json(output_dir / "placement_comparison.json", result)
    _plot(result, output_dir)
    return result


def _plot(comparison: Mapping[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Placement plots require: pip install -e ".[analysis]"') from error

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(comparison["rows"])
    models = list(dict.fromkeys(str(row["model_label"]) for row in rows))
    placements = [
        placement
        for placement in PLACEMENT_ORDER
        if any(row["placement"] == placement for row in rows)
    ]
    width = 0.8 / len(models)
    x = list(range(len(placements)))
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for model_index, model in enumerate(models):
        selected = {
            row["placement"]: row for row in rows if row["model_label"] == model
        }
        positions = [value - 0.4 + width / 2 + model_index * width for value in x]
        execution = [
            float(selected[item]["interface_success_rate"])
            if selected.get(item, {}).get("interface_success_rate") is not None
            else 0.0
            for item in placements
        ]
        axes[0].bar(positions, execution, width=width, label=f"{model}: interface")
        accuracy = [
            float(selected[item]["answer_accuracy"]) if item in selected else 0.0
            for item in placements
        ]
        axes[0].scatter(positions, accuracy, color="black", marker="_", s=90)
    axes[0].set_xticks(x, [item.replace("_", "\n") for item in placements])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("rate (bars: interface; ticks: answer)")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=7)

    for model_index, model in enumerate(models):
        selected = {
            row["placement"]: row for row in rows if row["model_label"] == model
        }
        positions = [value - 0.4 + width / 2 + model_index * width for value in x]
        false_rates = [
            float(selected[item]["false_intervention_rate"])
            if selected.get(item, {}).get("false_intervention_rate") is not None
            else 0.0
            for item in placements
        ]
        axes[1].bar(positions, false_rates, width=width, label=model)
    axes[1].set_xticks(x, [item.replace("_", "\n") for item in placements])
    axes[1].set_ylim(0, 0.38)
    axes[1].set_ylabel("false-intervention rate")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(fontsize=7)

    cost_placements = ["context", "weights", "runtime"]
    recurring = [
        row
        for row in rows
        if row["placement"] in cost_placements
    ]
    for model_index, model in enumerate(models):
        selected = [row for row in recurring if row["model_label"] == model]
        positions = [
            cost_placements.index(row["placement"])
            - 0.4
            + width / 2
            + model_index * width
            for row in selected
        ]
        prompt = [row["mean_prompt_tokens"] for row in selected]
        generated = [row["mean_generated_tokens"] for row in selected]
        axes[2].bar(positions, prompt, width=width, label=f"{model}: prompt")
        axes[2].bar(
            positions,
            generated,
            width=width,
            bottom=prompt,
            alpha=0.5,
            label=f"{model}: generated",
        )
    axes[2].set_xticks(
        range(len(cost_placements)), [item.replace("_", "\n") for item in cost_placements]
    )
    axes[2].set_ylabel("mean recurring tokens")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "placement_reliability_cost.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
