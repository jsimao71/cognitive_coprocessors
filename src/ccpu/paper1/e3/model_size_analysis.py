"""Matched model-size interaction for direct and ASL GSM8K conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, write_json


def _pair(report: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        return report["original_asl_vs_direct"][key]
    except KeyError as error:
        raise ValueError(f"missing original contribution pair: {key}") from error


def _robustness(report: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        return report["large_number_differential_degradation"][key]
    except KeyError as error:
        raise ValueError(f"missing robustness contribution pair: {key}") from error


def analyze_model_size_interaction(
    *,
    small_report_path: str | Path,
    large_report_path: str | Path,
    small_pair: str,
    large_pair: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare ASL-minus-direct effects across matched model sizes."""

    small_path = Path(small_report_path)
    large_path = Path(large_report_path)
    small = read_json(small_path)
    large = read_json(large_path)
    if small.get("identity_counts") != large.get("identity_counts"):
        raise ValueError("model-size reports use different identity counts")
    for source in ("original_eval", "large_eval"):
        if small["sources"][source]["sha256"] != large["sources"][source]["sha256"]:
            raise ValueError(f"model-size reports use different {source} freezes")

    small_original = _pair(small, small_pair)
    large_original = _pair(large, large_pair)
    small_robustness = _robustness(small, small_pair)
    large_robustness = _robustness(large, large_pair)
    original_interaction = large_original["delta"] - small_original["delta"]
    robustness_interaction = (
        large_robustness["estimate"] - small_robustness["estimate"]
    )
    report = {
        "schema_version": "ccpu.paper1.gsm8k_model_size_interaction.v1",
        "identity_counts": small["identity_counts"],
        "pairs": {"small": small_pair, "large": large_pair},
        "original_answer_contribution": {
            "small_asl_minus_direct": small_original["delta"],
            "large_asl_minus_direct": large_original["delta"],
            "model_size_interaction": original_interaction,
        },
        "large_number_robustness_contribution": {
            "small_differential_degradation": small_robustness["estimate"],
            "large_differential_degradation": large_robustness["estimate"],
            "model_size_interaction": robustness_interaction,
        },
        "gates": {
            "larger_model_benefits_asl_answer_contribution": original_interaction > 0,
            "larger_model_benefits_asl_magnitude_robustness": robustness_interaction > 0,
            "replicate_if": (
                "replicate when either interaction is positive and its component effect is "
                "large enough to be scientifically material; one-seed results are exploratory"
            ),
        },
        "estimands": {
            "answer": "(ASL_large - direct_large) - (ASL_small - direct_small)",
            "robustness": (
                "ASL-minus-direct differential degradation at large model minus the same "
                "quantity at small model"
            ),
        },
        "sources": {
            "small": {"path": str(small_path), "sha256": file_sha256(small_path)},
            "large": {"path": str(large_path), "sha256": file_sha256(large_path)},
        },
        "statistical_boundary": (
            "single matched initialization gate; no model-size claim until replication"
        ),
    }
    write_json(output_path, report)
    return report
