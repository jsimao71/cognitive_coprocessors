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


def analyze_gsm8k_augmentation_stage(
    *,
    stage: str,
    selection_eval_path: str | Path,
    historical_eval_path: str | Path,
    baseline_selection_predictions: str | Path,
    candidate_selection_predictions: str | Path,
    baseline_historical_predictions: str | Path,
    candidate_historical_predictions: str | Path,
    baseline_historical_summary: str | Path,
    candidate_historical_summary: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare one incremental checkpoint and apply the strict-positive greedy gate."""

    selection_rows = read_jsonl(selection_eval_path)
    selection_ids = {str(row["example_id"]) for row in selection_rows}
    historical_rows = read_jsonl(historical_eval_path)
    historical_ids = {str(row["example_id"]) for row in historical_rows}
    if len(selection_ids) != len(selection_rows):
        raise ValueError("selection evaluation contains duplicate identities")
    if len(historical_ids) != len(historical_rows):
        raise ValueError("historical evaluation contains duplicate identities")

    baseline_selection = _load_predictions(baseline_selection_predictions, selection_ids)
    candidate_selection = _load_predictions(candidate_selection_predictions, selection_ids)
    baseline_historical = _load_predictions(baseline_historical_predictions, historical_ids)
    candidate_historical = _load_predictions(candidate_historical_predictions, historical_ids)

    selection = _paired(candidate_selection, baseline_selection)
    historical = _paired(candidate_historical, baseline_historical)
    historical["semantic_diagnostics"] = _summary_delta(
        baseline_historical_summary, candidate_historical_summary
    )
    accepted = selection["delta_correct"] > 0

    sources = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in {
            "selection_eval": selection_eval_path,
            "historical_eval": historical_eval_path,
            "baseline_selection_predictions": baseline_selection_predictions,
            "candidate_selection_predictions": candidate_selection_predictions,
            "baseline_historical_predictions": baseline_historical_predictions,
            "candidate_historical_predictions": candidate_historical_predictions,
            "baseline_historical_summary": baseline_historical_summary,
            "candidate_historical_summary": candidate_historical_summary,
        }.items()
    }
    report = {
        "schema_version": "ccpu.paper1.gsm8k_augmentation_gate.v1",
        "stage": stage,
        "decision": {
            "accepted": accepted,
            "criterion": "candidate selection correct count must strictly exceed baseline",
            "delta_correct": selection["delta_correct"],
            "next_parent": "candidate" if accepted else "baseline",
        },
        "augmentation_selection": selection,
        "historical_diagnostic": historical,
        "sources": sources,
        "claim_boundary": (
            "The greedy gate is directional and exploratory. It is disjoint from final "
            "confirmation. Historical semantic diagnostics are reported but do not "
            "override the frozen selection answer-count decision."
        ),
    }
    write_json(output_path, report)
    return report
