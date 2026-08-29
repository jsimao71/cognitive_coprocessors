"""Compare retrieval-required policy in context, weights, and runtime."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


def _rates(selected: list[bool], required: list[bool]) -> dict[str, float]:
    count = len(required)
    controls = sum(not value for value in required)
    positives = sum(required)
    return {
        "selection_accuracy": sum(a == b for a, b in zip(selected, required, strict=True))
        / count,
        "retrieval_recall": sum(a and b for a, b in zip(selected, required, strict=True))
        / positives,
        "false_activation_rate": sum(a and not b for a, b in zip(selected, required, strict=True))
        / controls,
    }


def build_policy_placement(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_json(config_path)
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for model in config["models"]:
        label = str(model["model_label"])
        context_path = (config_path.parent / model["context_summary"]).resolve()
        weights_path = (config_path.parent / model["weights_summary"]).resolve()
        weights_predictions_path = (
            config_path.parent / model["weights_predictions"]
        ).resolve()
        phase_a_path = (config_path.parent / model["phase_a_predictions"]).resolve()
        context = read_json(context_path)
        weights = read_json(weights_path)
        weights_predictions = read_jsonl(weights_predictions_path)
        phase_a = read_jsonl(phase_a_path)
        rows.extend(
            [
                {"model_label": label, "placement": "context", **_summary_fields(context)},
                {"model_label": label, "placement": "weights", **_summary_fields(weights)},
            ]
        )
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in phase_a:
            if row["split"] == "test":
                by_condition[str(row["condition"])].append(row)
        for condition, placement in (
            ("flare_like", "confidence"),
            ("semantic", "runtime_semantic"),
            ("confidence_or_semantic", "confidence_or_semantic"),
            ("oracle", "oracle"),
        ):
            members = by_condition[condition]
            rates = _rates(
                [bool(row["retrieved"]) for row in members],
                [bool(row["evidence_required"]) for row in members],
            )
            rows.append(
                {
                    "model_label": label,
                    "placement": placement,
                    **rates,
                    "interface_success_rate": rates["selection_accuracy"],
                    "mean_prompt_tokens": sum(row["prompt_tokens"] for row in members)
                    / len(members),
                    "mean_generated_tokens": sum(row["generated_tokens"] for row in members)
                    / len(members),
                }
            )
        base_confidence = {
            str(row["example_id"]): bool(row["confidence_low"])
            for row in by_condition["llm_only"]
        }
        combined_selected = [
            bool(row["selected"]) or base_confidence[str(row["example_id"])]
            for row in weights_predictions
        ]
        combined_required = [bool(row["evidence_required"]) for row in weights_predictions]
        combined = _rates(combined_selected, combined_required)
        rows.append(
            {
                "model_label": label,
                "placement": "weights_plus_confidence",
                **combined,
                "interface_success_rate": combined["selection_accuracy"],
                "mean_prompt_tokens": weights["mean_prompt_tokens"],
                "mean_generated_tokens": weights["mean_generated_tokens"],
            }
        )
        for path in (context_path, weights_path, weights_predictions_path, phase_a_path):
            sources.append({"path": str(path), "sha256": file_sha256(path)})
    result = {
        "schema_version": "ccpu.paper1_5.policy_placement.v1",
        "question": (
            "Does semantic epistemic policy add information beyond uncertainty, and can "
            "that stable policy be stored efficiently in weights?"
        ),
        "rows": rows,
        "sources": sources,
        "config_sha256": file_sha256(config_path),
    }
    write_json(output_dir / "policy_placement.json", result)
    _plot(result, output_dir / "policy_placement.png")
    return result


def _summary_fields(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "selection_accuracy": float(summary["selection_accuracy"]),
        "interface_success_rate": float(summary["interface_success_rate"]),
        "retrieval_recall": float(summary["retrieval_recall"]),
        "false_activation_rate": float(summary["false_activation_rate"]),
        "mean_prompt_tokens": float(summary["mean_prompt_tokens"]),
        "mean_generated_tokens": float(summary["mean_generated_tokens"]),
    }


def _plot(result: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("policy placement plots require matplotlib") from error
    rows = result["rows"]
    models = list(dict.fromkeys(row["model_label"] for row in rows))
    placements = (
        "context",
        "weights",
        "confidence",
        "runtime_semantic",
        "confidence_or_semantic",
        "weights_plus_confidence",
        "oracle",
    )
    width = 0.8 / len(models)
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    for model_index, model in enumerate(models):
        selected = {row["placement"]: row for row in rows if row["model_label"] == model}
        positions = [index - 0.4 + width / 2 + model_index * width for index in range(len(placements))]
        axes[0].bar(
            positions,
            [selected[item]["interface_success_rate"] for item in placements],
            width=width,
            label=model,
        )
        axes[1].bar(
            positions,
            [selected[item]["false_activation_rate"] for item in placements],
            width=width,
            label=model,
        )
    labels = [item.replace("_", "\n") for item in placements]
    for axis, ylabel in zip(axes, ("interface success", "false activation"), strict=True):
        axis.set_xticks(range(len(placements)), labels)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
