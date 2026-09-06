"""Paired analysis and greedy acceptance for GSM8K augmentation stages."""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


def _correct(row: dict[str, Any]) -> bool:
    return bool(row["metrics"]["final_answer_correct"])


def _load_predictions(path: str | Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {str(row["example_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate prediction identity in {path}")
    if set(indexed) != expected_ids:
        missing = sorted(expected_ids - set(indexed))
        unexpected = sorted(set(indexed) - expected_ids)
        raise ValueError(
            f"prediction identities differ in {path}: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    return indexed


def _mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired(
    candidate: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(candidate) != set(baseline):
        raise ValueError("paired augmentation identities differ")
    both_correct = sum(_correct(candidate[key]) and _correct(baseline[key]) for key in candidate)
    candidate_only = sum(
        _correct(candidate[key]) and not _correct(baseline[key]) for key in candidate
    )
    baseline_only = sum(
        not _correct(candidate[key]) and _correct(baseline[key]) for key in candidate
    )
    both_wrong = len(candidate) - both_correct - candidate_only - baseline_only
    candidate_correct = both_correct + candidate_only
    baseline_correct = both_correct + baseline_only
    return {
        "count": len(candidate),
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_rate": baseline_correct / len(candidate),
        "candidate_rate": candidate_correct / len(candidate),
        "delta_correct": candidate_correct - baseline_correct,
        "delta_rate": (candidate_correct - baseline_correct) / len(candidate),
        "both_correct": both_correct,
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "both_wrong": both_wrong,
        "two_sided_exact_mcnemar_p": _mcnemar(candidate_only, baseline_only),
    }


def _summary_delta(baseline_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)
    if baseline["eval_sha256"] != candidate["eval_sha256"]:
        raise ValueError("historical summaries use different evaluation freezes")
    if baseline["prediction_count"] != candidate["prediction_count"]:
        raise ValueError("historical summaries have different prediction counts")

    def delta(section: str) -> dict[str, float]:
        baseline_values = baseline.get(section, {})
        candidate_values = candidate.get(section, {})
        keys = sorted(set(baseline_values) & set(candidate_values))
        return {key: float(candidate_values[key]) - float(baseline_values[key]) for key in keys}

    return {
        "count": baseline["prediction_count"],
        "eval_sha256": baseline["eval_sha256"],
        "baseline_rates": baseline["rates"],
        "candidate_rates": candidate["rates"],
        "rate_deltas": delta("rates"),
        "baseline_component_mean_f1": baseline.get("component_mean_f1", {}),
        "candidate_component_mean_f1": candidate.get("component_mean_f1", {}),
        "component_mean_f1_deltas": delta("component_mean_f1"),
    }


def _robustness(
    original: dict[str, dict[str, Any]],
    large: dict[str, dict[str, Any]],
    large_to_parent: dict[str, str],
) -> dict[str, Any]:
    original_correct = {
        large_id: _correct(original[parent_id]) for large_id, parent_id in large_to_parent.items()
    }
    large_correct = {large_id: _correct(large[large_id]) for large_id in large_to_parent}
    retained = sum(original_correct[key] and large_correct[key] for key in large_correct)
    gained = sum(not original_correct[key] and large_correct[key] for key in large_correct)
    lost = sum(original_correct[key] and not large_correct[key] for key in large_correct)
    stable_wrong = len(large_correct) - retained - gained - lost
    original_count = retained + lost
    large_count = retained + gained
    return {
        "count": len(large_correct),
        "original_eligible_correct": original_count,
        "large_correct": large_count,
        "original_eligible_rate": original_count / len(large_correct),
        "large_rate": large_count / len(large_correct),
        "large_minus_original": (large_count - original_count) / len(large_correct),
        "retained_correct": retained,
        "gained_on_large": gained,
        "lost_on_large": lost,
        "stable_wrong": stable_wrong,
        "two_sided_exact_mcnemar_p": _mcnemar(gained, lost),
    }


def analyze_gsm8k_augmentation_stage(
    *,
    stage: str,
    official_eval_path: str | Path,
    historical_eval_path: str | Path,
    large_eval_path: str | Path,
    baseline_official_predictions: str | Path,
    candidate_official_predictions: str | Path,
    baseline_historical_predictions: str | Path,
    candidate_historical_predictions: str | Path,
    baseline_historical_summary: str | Path,
    candidate_historical_summary: str | Path,
    baseline_large_predictions: str | Path,
    candidate_large_predictions: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare one incremental checkpoint and apply the strict-positive greedy gate."""

    official_ids = {str(row["example_id"]) for row in read_jsonl(official_eval_path)}
    historical_ids = {str(row["example_id"]) for row in read_jsonl(historical_eval_path)}
    large_rows = read_jsonl(large_eval_path)
    large_ids = {str(row["example_id"]) for row in large_rows}
    if len(official_ids) != len(read_jsonl(official_eval_path)):
        raise ValueError("official evaluation contains duplicate identities")
    if len(historical_ids) != len(read_jsonl(historical_eval_path)):
        raise ValueError("historical evaluation contains duplicate identities")
    if len(large_ids) != len(large_rows):
        raise ValueError("large-number evaluation contains duplicate identities")
    large_to_parent = {str(row["example_id"]): str(row["parent_example_id"]) for row in large_rows}
    if not set(large_to_parent.values()) <= official_ids:
        raise ValueError("large-number parent is outside official evaluation")

    baseline_official = _load_predictions(baseline_official_predictions, official_ids)
    candidate_official = _load_predictions(candidate_official_predictions, official_ids)
    baseline_historical = _load_predictions(baseline_historical_predictions, historical_ids)
    candidate_historical = _load_predictions(candidate_historical_predictions, historical_ids)
    baseline_large = _load_predictions(baseline_large_predictions, large_ids)
    candidate_large = _load_predictions(candidate_large_predictions, large_ids)

    official = _paired(candidate_official, baseline_official)
    historical = _paired(candidate_historical, baseline_historical)
    historical["semantic_diagnostics"] = _summary_delta(
        baseline_historical_summary, candidate_historical_summary
    )
    baseline_robustness = _robustness(baseline_official, baseline_large, large_to_parent)
    candidate_robustness = _robustness(candidate_official, candidate_large, large_to_parent)
    accepted = official["delta_correct"] > 0

    sources = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in {
            "official_eval": official_eval_path,
            "historical_eval": historical_eval_path,
            "large_eval": large_eval_path,
            "baseline_official_predictions": baseline_official_predictions,
            "candidate_official_predictions": candidate_official_predictions,
            "baseline_historical_predictions": baseline_historical_predictions,
            "candidate_historical_predictions": candidate_historical_predictions,
            "baseline_historical_summary": baseline_historical_summary,
            "candidate_historical_summary": candidate_historical_summary,
            "baseline_large_predictions": baseline_large_predictions,
            "candidate_large_predictions": candidate_large_predictions,
        }.items()
    }
    report = {
        "schema_version": "ccpu.paper1.gsm8k_augmentation_gate.v1",
        "stage": stage,
        "decision": {
            "accepted": accepted,
            "criterion": "candidate official correct count must strictly exceed baseline",
            "delta_correct": official["delta_correct"],
            "next_parent": "candidate" if accepted else "baseline",
        },
        "official_confirmatory": official,
        "historical_diagnostic": historical,
        "large_number_robustness": {
            "baseline": baseline_robustness,
            "candidate": candidate_robustness,
            "candidate_minus_baseline_degradation": (
                candidate_robustness["large_minus_original"]
                - baseline_robustness["large_minus_original"]
            ),
        },
        "sources": sources,
        "claim_boundary": (
            "The greedy gate is directional and exploratory. Semantic diagnostics and "
            "large-number robustness are reported but do not override the frozen official "
            "answer-count decision."
        ),
    }
    write_json(output_path, report)
    return report
