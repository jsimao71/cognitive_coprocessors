"""Metrics for the non-empirical heterogeneous protocol smoke test."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ccpu.common.metrics import binary_classification, safe_mean, wilson_interval


def evaluate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["condition"])].append(row)
    by_condition = []
    scaling = []
    for condition, rows in sorted(grouped.items()):
        tasks = [row for row in rows if row["should_trigger"]]
        controls = [row for row in rows if not row["should_trigger"]]
        successes = sum(bool(row["correct"]) for row in tasks)
        by_condition.append(
            {
                "condition": condition,
                "task_count": len(tasks),
                "control_count": len(controls),
                "task_accuracy": successes / len(tasks) if tasks else None,
                "task_accuracy_ci95": wilson_interval(successes, len(tasks)) if tasks else None,
                "trigger": binary_classification(
                    [bool(row["should_trigger"]) for row in rows],
                    [bool(row["intervened"]) for row in rows],
                ),
                "mean_state_items": safe_mean(row["state_items"] for row in rows),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in rows) / 1e6,
            }
        )
        cells: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in tasks:
            cells[(str(row["engine"]), int(row["depth"]), int(row["distractors"]))].append(row)
        for (engine, depth, distractors), items in sorted(cells.items()):
            scaling.append(
                {
                    "condition": condition,
                    "engine": engine,
                    "depth": depth,
                    "distractors": distractors,
                    "count": len(items),
                    "accuracy": sum(bool(item["correct"]) for item in items) / len(items),
                    "mean_wall_time_ms": safe_mean(item["wall_time_ns"] for item in items) / 1e6,
                    "mean_state_items": safe_mean(item["state_items"] for item in items),
                }
            )
    return {
        "schema_version": "ccpu.paper2.evaluation.v1",
        "empirical": False,
        "warning": "Scripted protocol smoke results are not language-model evidence.",
        "prediction_count": len(predictions),
        "by_condition": by_condition,
        "scaling_cells": scaling,
    }
