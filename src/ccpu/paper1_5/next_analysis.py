"""Cross-model Phase A analysis and the explicit Paper 2.5 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, write_json

_CONDITIONS = (
    "llm_only",
    "upfront_rag",
    "flare_like",
    "semantic",
    "confidence_or_semantic",
    "evidence_advisory",
    "evidence_abstention",
    "runtime_epistemic_gate",
    "oracle",
)


def build_next_analysis(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_json(config_path)
    rows = []
    sources = []
    model_gate = []
    for entry in config["models"]:
        summary_path = (config_path.parent / entry["summary"]).resolve()
        summary = read_json(summary_path)
        by_condition = {row["condition"]: row for row in summary["by_condition"]}
        label = str(entry["model_label"])
        for condition in _CONDITIONS:
            source = by_condition[condition]
            rows.append(
                {
                    "model_label": label,
                    "condition": condition,
                    "accuracy": source["accuracy"],
                    "retrieval_rate": source["retrieval_rate"],
                    "unsupported_commitment_rate": source["unsupported_commitment_rate"],
                    "authorized_commitment_coverage": source[
                        "authorized_commitment_coverage"
                    ],
                    "retrieval_precision": source["retrieval"]["precision"],
                    "retrieval_recall": source["retrieval"]["recall"],
                    "mean_generated_tokens": source["mean_generated_tokens"],
                    "mean_wall_time_ms": source["mean_wall_time_ms"],
                }
            )
        baseline = by_condition["llm_only"]
        proposed = by_condition["runtime_epistemic_gate"]
        model_gate.append(
            {
                "model_label": label,
                "ucr_improved": proposed["unsupported_commitment_rate"]
                < baseline["unsupported_commitment_rate"],
                "less_than_upfront_retrieval": proposed["retrieval_rate"]
                < by_condition["upfront_rag"]["retrieval_rate"],
                "all_four_quadrants": bool(summary["all_four_quadrants_observed"]),
            }
        )
        sources.append({"path": str(summary_path), "sha256": file_sha256(summary_path)})
    passing_models = sum(
        row["ucr_improved"] and row["less_than_upfront_retrieval"] for row in model_gate
    )
    gate = {
        "status": "go" if passing_models >= 2 else "no_go",
        "criterion": (
            "At least two model families reduce UCR with the runtime gate while retrieving "
            "less often than upfront RAG."
        ),
        "passing_models": passing_models,
        "by_model": model_gate,
    }
    result = {
        "schema_version": "ccpu.paper1_5.next_analysis.v1",
        "rows": rows,
        "paper2_5_gate": gate,
        "sources": sources,
        "config_sha256": file_sha256(config_path),
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "next_analysis.json", result)
    write_json(output_dir / "paper2_5_gate.json", gate)
    _plot(result, output_dir / "ucr_coverage_tradeoff.png")
    return result


def _plot(result: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 1.5 next-analysis plots require matplotlib") from error
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    markers = {"Qwen3-0.6B": "o", "SmolLM2-1.7B": "s", "Gemma3-1B": "^"}
    for row in result["rows"]:
        if row["condition"] not in {
            "llm_only",
            "flare_like",
            "semantic",
            "confidence_or_semantic",
            "runtime_epistemic_gate",
            "upfront_rag",
        }:
            continue
        label = f"{row['model_label']} / {row['condition'].replace('_', ' ')}"
        marker = markers.get(row["model_label"], "o")
        axes[0].scatter(
            row["retrieval_rate"],
            row["unsupported_commitment_rate"],
            marker=marker,
            label=label,
        )
        axes[1].scatter(
            row["retrieval_rate"], row["authorized_commitment_coverage"], marker=marker
        )
    axes[0].set(xlabel="retrieval rate", ylabel="unsupported commitment rate")
    axes[1].set(xlabel="retrieval rate", ylabel="authorized commitment coverage")
    for axis in axes:
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
