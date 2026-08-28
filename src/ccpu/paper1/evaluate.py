"""Component-level and end-to-end evaluation for Paper 1."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
from math import comb
from typing import Any

from ccpu.common.metrics import binary_classification, safe_mean, wilson_interval

from .dataset import ArithmeticExample

_TOOL_RESULT = re.compile(r"<tool_result>\s*(-?\d+(?:/\d+)?)\s*</tool_result>")
_CALCULATOR_RESULT = re.compile(
    r"<calculator_result>\s*(-?\d+(?:/\d+)?)\s*</calculator_result>"
)
_EQUALS_RESULT = re.compile(r"=\s*(-?\d+(?:/\d+)?)\b")
_FINAL_RESULT = re.compile(
    r"(?:answer|result|value|expression)(?:\s+is|\s*[:=])\s*\**\s*"
    r"(-?\d+(?:/\d+)?)\b",
    re.IGNORECASE,
)
_BOXED_RESULT = re.compile(r"\\boxed\s*\{\s*(-?\d+(?:/\d+)?)\s*\}")
_BARE_RESULT = re.compile(r"\s*(-?\d+(?:/\d+)?)\s*")


def extract_answer(text: str) -> str | None:
    # End-task evaluation must prefer what the model ultimately claims over a
    # correct engine value that the model may subsequently ignore or override.
    matches = (
        _FINAL_RESULT.findall(text)
        or _BOXED_RESULT.findall(text)
        or _EQUALS_RESULT.findall(text)
        or _TOOL_RESULT.findall(text)
        or _CALCULATOR_RESULT.findall(text)
    )
    if not matches:
        bare = _BARE_RESULT.fullmatch(text)
        return bare.group(1) if bare else None
    return matches[-1]


def answers_equal(predicted: str | None, gold: str | None) -> bool:
    if predicted is None or gold is None:
        return False
    try:
        return Fraction(predicted) == Fraction(gold)
    except (ValueError, ZeroDivisionError):
        return False


def _group_summary(
    rows: list[Mapping[str, Any]],
    items: Mapping[str, ArithmeticExample],
) -> dict[str, Any]:
    arithmetic = [row for row in rows if items[str(row["example_id"])].task_kind == "arithmetic"]
    answer_correct = [
        answers_equal(row.get("predicted_answer"), items[str(row["example_id"])].answer)
        for row in arithmetic
    ]
    trigger = binary_classification(
        [items[str(row["example_id"])].should_trigger for row in rows],
        [int(row.get("interventions", 0)) > 0 for row in rows],
    )
    correct_count = sum(answer_correct)
    interval = wilson_interval(correct_count, len(arithmetic)) if arithmetic else (0.0, 0.0)
    accepted = sum(int(row.get("normalization_accepts", 0)) for row in rows)
    candidates = sum(int(row.get("candidates", 0)) for row in rows)
    normalization_labels = [
        bool(row["normalization_correct"])
        for row in rows
        if row.get("normalization_correct") is not None
    ]
    executions = sum(int(row.get("executions", 0)) for row in rows)
    engine_successes = sum(int(row.get("engine_successes", 0)) for row in rows)
    engine_labels = [
        bool(row["engine_correct"]) for row in rows if row.get("engine_correct") is not None
    ]
    use_labels = [bool(row["result_used"]) for row in rows if row.get("result_used") is not None]
    override_labels = [
        bool(row["result_overridden"]) for row in rows if row.get("result_overridden") is not None
    ]
    def arithmetic_labels(field: str) -> list[bool]:
        return [bool(row[field]) for row in arithmetic if row.get(field) is not None]

    exposure_labels = arithmetic_labels("expression_exposed")
    recognition_labels = arithmetic_labels("recognized")
    selection_labels = arithmetic_labels("selection_correct")
    normalization_stage_labels = arithmetic_labels("normalization_succeeded")
    execution_stage_labels = arithmetic_labels("execution_succeeded")
    reinjection_labels = arithmetic_labels("reinjection_succeeded")
    return {
        "count": len(rows),
        "arithmetic_count": len(arithmetic),
        "accuracy": correct_count / len(arithmetic) if arithmetic else 0.0,
        "accuracy_ci95": list(interval),
        "trigger": trigger,
        "candidate_count": candidates,
        "expression_exposure_rate": safe_mean(exposure_labels) if exposure_labels else None,
        "recognition_rate": safe_mean(recognition_labels) if recognition_labels else None,
        "selection_correctness": safe_mean(selection_labels) if selection_labels else None,
        "normalization_stage_success_rate": (
            safe_mean(normalization_stage_labels) if normalization_stage_labels else None
        ),
        "normalization_acceptance_rate": accepted / candidates if candidates else None,
        "normalization_correctness": (
            safe_mean(normalization_labels) if normalization_labels else None
        ),
        "normalization_failures": sum(int(row.get("normalization_failures", 0)) for row in rows),
        "engine_success_rate": engine_successes / executions if executions else None,
        "execution_stage_success_rate": (
            safe_mean(execution_stage_labels) if execution_stage_labels else None
        ),
        "engine_correctness": safe_mean(engine_labels) if engine_labels else None,
        "engine_failures": sum(int(row.get("engine_failures", 0)) for row in rows),
        "reinjection_failures": sum(int(row.get("reinjection_failures", 0)) for row in rows),
        "reinjection_success_rate": safe_mean(reinjection_labels) if reinjection_labels else None,
        "result_use_rate": safe_mean(use_labels) if use_labels else None,
        "result_override_rate": safe_mean(override_labels) if override_labels else None,
        "mean_generated_tokens": safe_mean(float(row.get("generated_tokens", 0)) for row in rows),
        "mean_reinjected_tokens": safe_mean(float(row.get("reinjected_tokens", 0)) for row in rows),
        "mean_model_calls": safe_mean(float(row.get("model_calls", 0)) for row in rows),
        "mean_wall_time_ms": safe_mean(float(row.get("wall_time_ns", 0)) / 1e6 for row in rows),
        "mean_engine_time_ms": safe_mean(float(row.get("engine_time_ns", 0)) / 1e6 for row in rows),
        "mean_interventions": safe_mean(float(row.get("interventions", 0)) for row in rows),
        "mean_state_items": safe_mean(float(row.get("state_items", 0)) for row in rows),
        "mean_trace_bytes": safe_mean(float(row.get("trace_bytes", 0)) for row in rows),
        "mean_state_bytes": safe_mean(float(row.get("state_bytes", 0)) for row in rows),
        "mean_invocation_overhead_chars": safe_mean(
            float(row.get("invocation_overhead_chars", 0)) for row in rows
        ),
    }


def evaluate(
    examples: Iterable[ArithmeticExample], predictions: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    items = {example.example_id: example for example in examples}
    rows = list(predictions)
    unknown = sorted({str(row["example_id"]) for row in rows} - items.keys())
    if unknown:
        raise ValueError(f"predictions contain unknown example IDs: {unknown[:3]}")
    keys = [
        (str(row["example_id"]), str(row["model_id"]), str(row["condition"]), int(row["seed"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("predictions contain duplicate example/model/condition/seed rows")
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_id"]), str(row["condition"]), int(row["seed"]))].append(row)

    by_run = []
    for (model_id, condition, seed), members in sorted(groups.items()):
        by_run.append(
            {
                "model_id": model_id,
                "condition": condition,
                "seed": seed,
                **_group_summary(members, items),
            }
        )

    scaling_groups: dict[tuple[str, str, int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        example = items[str(row["example_id"])]
        if example.task_kind != "arithmetic":
            continue
        scaling_groups[
            (
                str(row["model_id"]),
                str(row["condition"]),
                int(example.difficulty["operator_count"]),
                int(example.difficulty["operand_digits"]),
                str(example.difficulty.get("structure", "legacy")),
            )
        ].append(row)
    scaling = []
    for (model_id, condition, operator_count, operand_digits, structure), members in sorted(
        scaling_groups.items()
    ):
        correct = sum(
            answers_equal(row.get("predicted_answer"), items[str(row["example_id"])].answer)
            for row in members
        )
        scaling.append(
            {
                "model_id": model_id,
                "condition": condition,
                "operator_count": operator_count,
                "operand_digits": operand_digits,
                "structure": structure,
                "count": len(members),
                "accuracy": correct / len(members),
                "mean_wall_time_ms": safe_mean(
                    float(row.get("wall_time_ns", 0)) / 1e6 for row in members
                ),
            }
        )
    return {"schema_version": "ccpu.paper1.evaluation.v1", "by_run": by_run, "scaling": scaling}


def paired_comparisons(
    examples: Iterable[ArithmeticExample],
    predictions: Iterable[Mapping[str, Any]],
    *,
    baseline: str = "llm_only",
) -> dict[str, Any]:
    """Compute paired exact McNemar tests without changing endpoint labels."""

    items = {
        example.example_id: example
        for example in examples
        if example.task_kind == "arithmetic"
    }
    grouped: dict[tuple[str, int, str], dict[str, bool]] = defaultdict(dict)
    for row in predictions:
        example_id = str(row["example_id"])
        if example_id not in items:
            continue
        key = (str(row["model_id"]), int(row["seed"]), str(row["condition"]))
        grouped[key][example_id] = answers_equal(
            row.get("predicted_answer"), items[example_id].answer
        )

    comparisons = []
    runs = sorted({(model_id, seed) for model_id, seed, _ in grouped})
    for model_id, seed in runs:
        baseline_rows = grouped.get((model_id, seed, baseline))
        if baseline_rows is None:
            raise ValueError(f"missing baseline rows for {model_id}, seed {seed}")
        conditions = sorted(
            condition
            for candidate_model, candidate_seed, condition in grouped
            if candidate_model == model_id and candidate_seed == seed
        )
        for condition in conditions:
            members = grouped[(model_id, seed, condition)]
            if members.keys() != baseline_rows.keys():
                raise ValueError("paired comparison conditions have different example IDs")
            gains = sum(not baseline_rows[key] and members[key] for key in baseline_rows)
            losses = sum(baseline_rows[key] and not members[key] for key in baseline_rows)
            discordant = gains + losses
            tail = sum(comb(discordant, index) for index in range(min(gains, losses) + 1))
            p_value = min(1.0, 2.0 * tail / (2**discordant)) if discordant else 1.0
            comparisons.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "condition": condition,
                    "baseline": baseline,
                    "paired_count": len(baseline_rows),
                    "gains": gains,
                    "losses": losses,
                    "ties": len(baseline_rows) - discordant,
                    "exact_mcnemar_p_two_sided": p_value,
                }
            )
    return {
        "schema_version": "ccpu.paper1.paired_analysis.v1",
        "baseline": baseline,
        "comparisons": comparisons,
        "multiplicity_adjusted": False,
    }
