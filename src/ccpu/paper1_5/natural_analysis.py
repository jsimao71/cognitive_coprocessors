"""Cross-family analysis for the natural-language robustness iteration."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json

_CONDITIONS = (
    "no_retrieval",
    "upfront_retrieval",
    "confidence_only",
    "temporal_only",
    "source_only",
    "context_sufficiency_only",
    "combined_semantic",
    "confidence_or_semantic",
    "confidence_and_semantic",
    "retrospective_verification",
    "evidence_advisory",
    "support_contract",
    "runtime_enforcement",
    "oracle",
)
_CONTROL_ANSWERS = {
    "quoted_freshness": ("latest custodian", "current registry", "updated owner"),
    "historical_date": ("18th", "yes", "1848"),
    "compute_not_retrieve": ("42", "120", "2026-08-29"),
    "stable_familiar": ("Au", "Lisbon", "H2O"),
}


def build_natural_analysis(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_json(config_path)
    lexical_path = _resolve(config_path, str(config["lexical_audit"]))
    longform_path = _resolve(config_path, str(config["longform_summary"]))
    lexical = read_json(lexical_path)
    longform = read_json(longform_path)
    rows = []
    sources = [
        {"path": str(lexical_path), "sha256": file_sha256(lexical_path)},
        {"path": str(longform_path), "sha256": file_sha256(longform_path)},
    ]
    rejected_freezes = []
    for candidate in config.get("rejected_freezes", []):
        candidate_lexical_path = _resolve(config_path, str(candidate["lexical_audit"]))
        candidate_dataset_path = _resolve(config_path, str(candidate["dataset"]))
        candidate_source_path = _resolve(config_path, str(candidate["source"]))
        candidate_lexical = read_json(candidate_lexical_path)
        candidate_rows = read_jsonl(candidate_dataset_path)
        candidate_source = read_json(candidate_source_path)
        duplicate_count = sum(
            len([row for row in candidate_rows if row["split"] == split])
            - len(
                {
                    str(row["question"]).casefold()
                    for row in candidate_rows
                    if row["split"] == split
                }
            )
            for split in ("train", "dev", "test")
        )
        source_keys = [
            (str(record["entity"]), str(record["attribute"]))
            for record in candidate_source["records"]
        ]
        collision_count = len(source_keys) - len(set(source_keys))
        gold_mismatch_count = sum(
            row["category"] in _CONTROL_ANSWERS
            and row["answer"]
            != _CONTROL_ANSWERS[str(row["category"])][
                int(row.get("template_index", {"train": 0, "dev": 1, "test": 2}[row["split"]]))
            ]
            for row in candidate_rows
        )
        failures = []
        if candidate_lexical["maximum_accuracy"] >= candidate_lexical["threshold"]:
            failures.append("lexical_triviality")
        if duplicate_count:
            failures.append("duplicate_questions")
        if collision_count:
            failures.append("source_key_collisions")
        if gold_mismatch_count:
            failures.append("gold_answer_mismatch")
        rejected_freezes.append(
            {
                "version": str(candidate["version"]),
                "status": "rejected",
                "failure_reasons": failures,
                "maximum_lexical_accuracy": candidate_lexical["maximum_accuracy"],
                "within_split_duplicate_questions": duplicate_count,
                "source_key_collisions": collision_count,
                "gold_answer_mismatches": gold_mismatch_count,
            }
        )
        for path in (candidate_lexical_path, candidate_dataset_path, candidate_source_path):
            sources.append({"path": str(path), "sha256": file_sha256(path)})
    for model in config["models"]:
        summary_path = _resolve(config_path, str(model["summary"]))
        summary = read_json(summary_path)
        by_condition = {row["condition"]: row for row in summary["by_model_condition"]}
        for condition in _CONDITIONS:
            source = by_condition[condition]
            rows.append(
                {
                    "model_label": str(model["model_label"]),
                    "condition": condition,
                    **{
                        key: source[key]
                        for key in (
                            "accuracy",
                            "trigger_precision",
                            "trigger_recall",
                            "false_activation_rate",
                            "unsupported_commitment_rate",
                            "authorized_commitment_coverage",
                            "unnecessary_retrieval_rate",
                            "evidence_override_rate",
                            "abstention_rate",
                            "retrospective_detection_rate",
                            "mean_generated_tokens",
                            "mean_wall_time_ms",
                            "mean_retrieval_time_ms",
                        )
                    },
                }
            )
        sources.append({"path": str(summary_path), "sha256": file_sha256(summary_path)})
    decision = {
        "status": "no_go",
        "reopen_only_with_new_untouched_freeze": True,
        "reasons": [
            "The natural-v5 test freeze has now been evaluated and cannot be reused for adaptation.",
            f"Shallow lexical accuracy remains {lexical['maximum_accuracy']:.4f}, so the task is richer but still easy.",
            "The transparent policy has perfect recall with only modest false-activation headroom.",
            "Prior rank-8 adapters fit synthetic development data but had zero held-out recall.",
        ],
    }
    result = {
        "schema_version": "ccpu.paper1_5.natural_analysis.v1",
        "rows": rows,
        "lexical_audit": lexical,
        "longform": longform,
        "rejected_freezes": rejected_freezes,
        "learned_policy_decision": decision,
        "sources": sources,
        "config_sha256": file_sha256(config_path),
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "natural_analysis.json", result)
    write_json(output_dir / "learned_policy_decision.json", decision)
    write_json(output_dir / "rejected_freezes.json", rejected_freezes)
    _write_csv(rows, output_dir / "natural_results.csv")
    _plot_triggers(rows, output_dir / "natural_trigger_comparison.png")
    _plot_enforcement(rows, longform, output_dir / "natural_enforcement_longform.png")
    return result


def _resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_triggers(rows: list[dict[str, Any]], output: Path) -> None:
    plt = _pyplot()
    models = list(dict.fromkeys(str(row["model_label"]) for row in rows))
    conditions = ("confidence_only", "combined_semantic")
    colors = {"confidence_only": "#147d92", "combined_semantic": "#d4643c"}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    width = 0.34
    for condition_index, condition in enumerate(conditions):
        selected = [row for row in rows if row["condition"] == condition]
        positions = [index + (condition_index - 0.5) * width for index in range(len(models))]
        axes[0].bar(
            positions,
            [row["trigger_recall"] for row in selected],
            width,
            color=colors[condition],
            label=condition.replace("_", " "),
        )
        axes[1].bar(
            positions,
            [row["false_activation_rate"] for row in selected],
            width,
            color=colors[condition],
        )
    for axis, title in zip(axes, ("Retrieval recall", "False activation"), strict=True):
        axis.set_xticks(range(len(models)), models)
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_enforcement(rows: list[dict[str, Any]], longform: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    models = list(dict.fromkeys(str(row["model_label"]) for row in rows))
    conditions = ("no_retrieval", "evidence_advisory", "support_contract", "runtime_enforcement")
    labels = ("none", "advisory", "contract", "runtime")
    colors = ("#8c8c83", "#e0a12d", "#3d8f77", "#153f52")
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    width = 0.19
    for condition_index, (condition, label, color) in enumerate(
        zip(conditions, labels, colors, strict=True)
    ):
        selected = [row for row in rows if row["condition"] == condition]
        positions = [
            index + (condition_index - 1.5) * width for index in range(len(models))
        ]
        axes[0].bar(
            positions,
            [row["unsupported_commitment_rate"] for row in selected],
            width,
            color=color,
            label=label,
        )
        axes[1].bar(
            positions,
            [row["authorized_commitment_coverage"] for row in selected],
            width,
            color=color,
        )
    for axis, title in zip(axes[:2], ("Unsupported commitments", "Authorized coverage"), strict=True):
        axis.set_xticks(range(len(models)), models)
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    longform_labels = ("early catch", "late detection", "evidence reuse")
    longform_values = (
        longform["early_catch_rate"],
        longform["late_catch_rate"],
        longform["evidence_reuse_rate"],
    )
    axes[2].bar(longform_labels, longform_values, color=("#153f52", "#d4643c", "#3d8f77"))
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Long-form controller")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].grid(axis="y", alpha=0.22)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Natural robustness analysis requires matplotlib") from error
    return plt
