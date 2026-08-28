"""Paper 1.5 retrieval, epistemic-risk, and cost metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ccpu.common.metrics import binary_classification, safe_mean, wilson_interval


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def _threshold_sweep(test_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    by_example: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in test_rows:
        by_example[str(row["example_id"])][str(row["condition"])] = row
    examples = [rows for rows in by_example.values() if {"llm_only", "upfront_rag"} <= rows.keys()]
    if not examples:
        return {"flare_like": [], "confidence_or_semantic": []}
    observed = sorted(
        {float(rows["llm_only"]["minimum_token_probability"]) for rows in examples}
    )
    thresholds = [0.0, *(math.nextafter(value, 1.0) for value in observed), 1.0]
    sweeps: dict[str, list[dict[str, float]]] = {}
    for strategy in ("flare_like", "confidence_or_semantic"):
        points = []
        for threshold in thresholds:
            retrieved: list[bool] = []
            correct: list[bool] = []
            for rows in examples:
                base = rows["llm_only"]
                confidence_trigger = float(base["minimum_token_probability"]) < threshold
                trigger = confidence_trigger or (
                    strategy == "confidence_or_semantic" and bool(base["semantic_risk"])
                )
                retrieved.append(trigger)
                correct.append(bool(rows["upfront_rag"]["correct"] if trigger else base["correct"]))
            points.append(
                {
                    "threshold": threshold,
                    "retrieval_rate": sum(retrieved) / len(retrieved),
                    "accuracy": sum(correct) / len(correct),
                }
            )
        sweeps[strategy] = points
    return sweeps


def _pareto(points: list[dict[str, float]]) -> list[dict[str, float]]:
    frontier = []
    best_accuracy = -1.0
    for point in sorted(points, key=lambda item: (item["retrieval_rate"], -item["accuracy"])):
        if point["accuracy"] > best_accuracy:
            frontier.append(point)
            best_accuracy = point["accuracy"]
    return frontier


def evaluate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    test_rows = [row for row in predictions if row["split"] == "test"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        grouped[str(row["condition"])].append(row)
    by_condition = []
    for condition, rows in sorted(grouped.items()):
        correct = sum(bool(row["correct"]) for row in rows)
        confident_hallucinations = [
            row
            for row in rows
            if row["evidence_required"] and not row["confidence_low"] and not row["baseline_correct"]
        ]
        caught = [row for row in confident_hallucinations if row["retrieved"] and row["correct"]]
        low_no_deficit = [
            row for row in rows if row["confidence_low"] and not row["evidence_required"]
        ]
        quadrants: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            quadrants[str(row["quadrant"])].append(row)
        by_condition.append(
            {
                "condition": condition,
                "count": len(rows),
                "accuracy": correct / len(rows),
                "accuracy_ci95": wilson_interval(correct, len(rows)),
                "unsupported_claim_rate": (
                    1.0 - _rate([row for row in rows if row["evidence_required"]], "correct")
                    if any(row["evidence_required"] for row in rows)
                    else None
                ),
                "retrieval": binary_classification(
                    [bool(row["evidence_required"]) for row in rows],
                    [bool(row["retrieved"]) for row in rows],
                ),
                "retrieval_rate": _rate(rows, "retrieved"),
                "confident_hallucination_count": len(confident_hallucinations),
                "confident_hallucination_catch_rate": (
                    len(caught) / len(confident_hallucinations)
                    if confident_hallucinations
                    else None
                ),
                "low_confidence_no_deficit_false_retrieval": _rate(
                    low_no_deficit, "retrieved"
                ),
                "retrieval_rate_by_quadrant": {
                    quadrant: _rate(items, "retrieved") for quadrant, items in quadrants.items()
                },
                "mean_minimum_token_probability": safe_mean(
                    row["minimum_token_probability"] for row in rows
                ),
                "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in rows),
                "mean_model_calls": safe_mean(row["model_calls"] for row in rows),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in rows) / 1e6,
            }
        )
    quadrants = sorted({str(row["quadrant"]) for row in test_rows})
    sweeps = _threshold_sweep(test_rows)
    return {
        "schema_version": "ccpu.paper1_5.evaluation.v1",
        "test_count": len({row["example_id"] for row in test_rows}),
        "conditions": len(grouped),
        "confidence_threshold": test_rows[0]["confidence_threshold"] if test_rows else None,
        "observed_quadrants": quadrants,
        "all_four_quadrants_observed": len(quadrants) == 4,
        "by_condition": by_condition,
        "threshold_sweeps": sweeps,
        "pareto_frontiers": {name: _pareto(points) for name, points in sweeps.items()},
    }
