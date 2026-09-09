"""Strict paired Direct-versus-ASL analysis for matched GSM8K interventions."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in result:
            raise ValueError(f"duplicate {label} identity: {example_id}")
        result[example_id] = row
    return result


def _mcnemar(left_only: int, right_only: int) -> float:
    total = left_only + right_only
    if not total:
        return 1.0
    tail = sum(math.comb(total, value) for value in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2**total))


def _verify_summary(summary_path: Path, eval_path: Path, label: str) -> dict[str, Any]:
    if not summary_path.exists():
        raise ValueError(f"missing {label} summary: {summary_path}")
    summary = read_json(summary_path)
    if summary.get("eval_sha256") != file_sha256(eval_path):
        raise ValueError(f"{label} summary was produced from a different evaluation file")
    return summary


def _verify_predictions(
    *,
    evaluation: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    label: str,
) -> None:
    if set(predictions) != set(evaluation):
        missing = set(evaluation) - set(predictions)
        extra = set(predictions) - set(evaluation)
        raise ValueError(f"{label} identity mismatch: missing={len(missing)} extra={len(extra)}")
    for example_id, prediction in predictions.items():
        expected = evaluation[example_id].get("question_sha256")
        if prediction.get("question_sha256") != expected:
            raise ValueError(f"{label} question hash mismatch: {example_id}")


def analyze_matched_interventions(
    *,
    cells: list[tuple[str, str | Path, str | Path, str | Path]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze only complete, hash-matched Direct and ASL cells."""

    if not cells:
        raise ValueError("at least one matched intervention cell is required")
    results = []
    seen = set()
    for label, eval_value, direct_value, asl_value in cells:
        if label in seen:
            raise ValueError(f"duplicate intervention label: {label}")
        seen.add(label)
        eval_path = Path(eval_value)
        direct_path = Path(direct_value)
        asl_path = Path(asl_value)
        _verify_summary(direct_path.with_name("summary.json"), eval_path, f"{label} Direct")
        _verify_summary(asl_path.with_name("summary.json"), eval_path, f"{label} ASL")
        evaluation = _index(read_jsonl(eval_path), f"{label} evaluation")
        direct = _index(read_jsonl(direct_path), f"{label} Direct")
        asl = _index(read_jsonl(asl_path), f"{label} ASL")
        _verify_predictions(evaluation=evaluation, predictions=direct, label=f"{label} Direct")
        _verify_predictions(evaluation=evaluation, predictions=asl, label=f"{label} ASL")

        direct_correct = sum(bool(row["metrics"]["final_answer_correct"]) for row in direct.values())
        asl_correct = sum(bool(row["metrics"]["final_answer_correct"]) for row in asl.values())
        asl_only = direct_only = both_correct = both_wrong = 0
        for example_id in evaluation:
            d_ok = bool(direct[example_id]["metrics"]["final_answer_correct"])
            a_ok = bool(asl[example_id]["metrics"]["final_answer_correct"])
            if d_ok and a_ok:
                both_correct += 1
            elif d_ok:
                direct_only += 1
            elif a_ok:
                asl_only += 1
            else:
                both_wrong += 1
        direct_tokens = [int(row["generated_tokens"]) for row in direct.values()]
        asl_tokens = [int(row["generated_tokens"]) for row in asl.values()]
        count = len(evaluation)
        results.append(
            {
                "label": label,
                "count": count,
                "eval_sha256": file_sha256(eval_path),
                "direct_correct": direct_correct,
                "direct_accuracy": direct_correct / count,
                "asl_correct": asl_correct,
                "asl_accuracy": asl_correct / count,
                "asl_minus_direct": (asl_correct - direct_correct) / count,
                "paired": {
                    "both_correct": both_correct,
                    "direct_only": direct_only,
                    "asl_only": asl_only,
                    "both_wrong": both_wrong,
                    "exact_mcnemar_p": _mcnemar(direct_only, asl_only),
                },
                "generated_tokens": {
                    "direct_mean": statistics.fmean(direct_tokens),
                    "asl_mean": statistics.fmean(asl_tokens),
                    "direct_to_asl_ratio": (
                        statistics.fmean(direct_tokens) / statistics.fmean(asl_tokens)
                        if statistics.fmean(asl_tokens)
                        else None
                    ),
                },
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "direct_vs_asl.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "label",
                "count",
                "direct_correct",
                "direct_accuracy",
                "asl_correct",
                "asl_accuracy",
                "asl_minus_direct",
                "direct_only",
                "asl_only",
                "exact_mcnemar_p",
                "direct_mean_tokens",
                "asl_mean_tokens",
            ),
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames if key in row},
                    "direct_only": row["paired"]["direct_only"],
                    "asl_only": row["paired"]["asl_only"],
                    "exact_mcnemar_p": row["paired"]["exact_mcnemar_p"],
                    "direct_mean_tokens": row["generated_tokens"]["direct_mean"],
                    "asl_mean_tokens": row["generated_tokens"]["asl_mean"],
                }
            )
    report = {
        "schema_version": "ccpu.paper1.matched_intervention_analysis.v1",
        "primary_comparison": "paired Direct versus generated ASL plus deterministic execution",
        "cells": results,
        "output_sha256": {"direct_vs_asl_csv": file_sha256(csv_path)},
    }
    write_json(output / "summary.json", report)
    return report
