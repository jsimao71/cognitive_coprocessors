"""Deterministic diagnostics for the paired GSM8K magnitude ladder."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

MAGNITUDE_DIAGNOSTIC_PROTOCOL_ID = "paper1_gsm8k_magnitude_diagnostic_v1"
MAGNITUDE_CURVE_PROTOCOL_ID = "paper1_gsm8k_magnitude_curve_v1"

# ASL identifiers may contain digits, so only standalone arithmetic literals are rewritten.
_ASL_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])-?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9_.])")


def _prediction_index(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in indexed:
            raise ValueError(f"duplicate {label} prediction for {example_id}")
        indexed[example_id] = row
    return indexed


def _returned(validation: dict[str, Any], scope: dict[str, Any]) -> Any:
    if not validation.get("execution_verified"):
        return None
    return validation["execution"]["workspace"][str(scope["id"])].get("returned")


def _answer_equal(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    try:
        return abs(Decimal(str(actual)) - Decimal(str(expected))) <= Decimal("0.011")
    except (InvalidOperation, TypeError, ValueError):
        return actual == expected


def _literal_mapping(row: dict[str, Any]) -> dict[str, str]:
    transformation = row["transformation"]
    pairs = list(transformation["source_value_mapping"].items())
    pairs.extend(
        (str(step["original_result"]), str(step["transformed_result"]))
        for step in transformation["hidden_execution_trace"]
    )
    mapping: dict[str, str] = {}
    for original, transformed in pairs:
        normalized = str(Decimal(original))
        previous = mapping.get(normalized)
        if previous is not None and Decimal(previous) != Decimal(transformed):
            raise ValueError(f"ambiguous literal mapping for {original}")
        mapping[normalized] = transformed
    return mapping


def _numeric_skeleton(asl: str) -> str:
    return _ASL_NUMBER.sub("<NUM>", asl)


def _literal_free_alpha_signature(validation: dict[str, Any]) -> tuple[Any, Any]:
    if not validation.get("lower_verified"):
        return None, None
    operations = validation["ccir"]["operations"]
    assignments = {
        str(item["operation"]["target"]): item["operation"]["expr"]
        for item in operations
        if item["operation"]["op"] == "SET"
    }

    def expression(node: dict[str, Any], stack: frozenset[str]) -> Any:
        operation = str(node["op"])
        if operation == "CONST":
            return ("CONST",)
        if operation == "REF":
            path = str(node["path"])
            if path in stack:
                return ("CYCLE",)
            if path in assignments:
                return expression(assignments[path], stack | {path})
            return ("EXTERNAL",)
        arguments = [expression(argument, stack) for argument in node.get("args", [])]
        if operation in {"ADD", "MUL", "SUM", "MIN", "MAX"}:
            equivalent = "ADD" if operation == "SUM" else operation
            flattened = []
            for argument in arguments:
                if isinstance(argument, tuple) and argument and argument[0] == equivalent:
                    flattened.extend(argument[1:])
                else:
                    flattened.append(argument)
            return (equivalent, *sorted(flattened, key=repr))
        return (operation, *arguments)

    states = Counter(
        expression(value, frozenset({path})) for path, value in assignments.items()
    )
    returned = next(
        (
            expression(item["operation"]["expr"], frozenset())
            for item in reversed(operations)
            if item["operation"]["op"] == "RETURN"
        ),
        None,
    )
    return states, returned


def _oracle_literal_candidate(
    original_asl: str, transformed_asl: str, row: dict[str, Any]
) -> tuple[str | None, str]:
    """Rewrite literals only after strict or literal-free alpha structure matches."""

    reason = "strict_position_aligned"
    if _numeric_skeleton(original_asl) != _numeric_skeleton(transformed_asl):
        scope = row["effective_scope"]
        original_validation = validate_asl(original_asl, effective_scope=scope)
        transformed_validation = validate_asl(transformed_asl, effective_scope=scope)
        if _literal_free_alpha_signature(original_validation) != _literal_free_alpha_signature(
            transformed_validation
        ):
            return None, "nonliteral_structure_differs"
        reason = "alpha_structure_position_aligned"
    original_literals = _ASL_NUMBER.findall(original_asl)
    transformed_literals = _ASL_NUMBER.findall(transformed_asl)
    if len(original_literals) != len(transformed_literals):
        return None, "literal_arity_differs"
    mapping = _literal_mapping(row)
    desired = [mapping.get(str(Decimal(value)), value) for value in original_literals]
    replacements = iter(desired)
    corrected = _ASL_NUMBER.sub(lambda _: next(replacements), transformed_asl)
    return corrected, reason


def _runtime_stage(validation: dict[str, Any], scope: dict[str, Any]) -> str | None:
    checks = (
        ("syntax_verified", "parse"),
        ("lower_verified", "lower"),
        ("type_verified", "type"),
        ("execution_verified", "execute"),
    )
    for key, stage in checks:
        if not validation.get(key):
            return stage
    if _returned(validation, scope) is None:
        return "missing_return"
    return None


def analyze_magnitude_failures(
    *,
    transformed_eval_path: str | Path,
    original_predictions_path: str | Path,
    transformed_predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Partition transformed ASL outcomes and apply a literal-only oracle."""

    eval_rows = read_jsonl(transformed_eval_path)
    originals = _prediction_index(
        read_jsonl(original_predictions_path), label="original"
    )
    transformed = _prediction_index(
        read_jsonl(transformed_predictions_path), label="transformed"
    )
    diagnostics = []
    for row in eval_rows:
        example_id = str(row["example_id"])
        parent_id = str(row["parent_example_id"])
        if parent_id not in originals:
            raise ValueError(f"missing original prediction for {parent_id}")
        if example_id not in transformed:
            raise ValueError(f"missing transformed prediction for {example_id}")
        original_prediction = originals[parent_id]
        transformed_prediction = transformed[example_id]
        scope = row["effective_scope"]
        original_asl = str(original_prediction.get("predicted_asl", ""))
        candidate_asl = str(transformed_prediction.get("predicted_asl", ""))
        validation = validate_asl(candidate_asl, effective_scope=scope)
        actual = _returned(validation, scope)
        correct = _answer_equal(actual, row["reference_return"])
        runtime_stage = _runtime_stage(validation, scope)
        category = "correct" if correct else "wrong_semantic_structure"
        oracle_asl = None
        oracle_correct = False
        oracle_reason = "not_attempted"
        original_correct = bool(
            original_prediction.get("metrics", {}).get("final_answer_correct", False)
        )
        if not correct and runtime_stage is not None:
            category = "runtime_failure"
        elif not correct and original_correct:
            oracle_asl, oracle_reason = _oracle_literal_candidate(
                original_asl, candidate_asl, row
            )
            if oracle_asl is not None:
                oracle_validation = validate_asl(oracle_asl, effective_scope=scope)
                oracle_actual = _returned(oracle_validation, scope)
                oracle_correct = _answer_equal(oracle_actual, row["reference_return"])
                if oracle_correct:
                    category = "literal_copy_error"
        diagnostics.append(
            {
                "schema_version": "ccpu.paper1.gsm8k_magnitude_failure.v1",
                "example_id": example_id,
                "parent_example_id": parent_id,
                "factor": int(row["transformation"]["factor"]),
                "category": category,
                "runtime_stage": runtime_stage,
                "original_answer_correct": original_correct,
                "predicted_return": actual,
                "reference_return": row["reference_return"],
                "oracle_literal_attempted": oracle_asl is not None,
                "oracle_literal_correct": oracle_correct,
                "oracle_literal_reason": oracle_reason,
                "oracle_corrected_asl": oracle_asl if oracle_correct else None,
            }
        )

    counts = Counter(str(row["category"]) for row in diagnostics)
    failures = len(diagnostics) - counts["correct"]
    oracle_rescues = counts["literal_copy_error"]
    output = Path(output_dir)
    rows_path = write_jsonl(output / "diagnostics.jsonl", diagnostics)
    report = {
        "schema_version": "ccpu.paper1.gsm8k_magnitude_diagnostic_summary.v1",
        "protocol_id": MAGNITUDE_DIAGNOSTIC_PROTOCOL_ID,
        "factor": int(eval_rows[0]["transformation"]["factor"]) if eval_rows else None,
        "counts": {
            "total": len(diagnostics),
            "correct": counts["correct"],
            "failures": failures,
            "wrong_semantic_structure": counts["wrong_semantic_structure"],
            "literal_copy_error": oracle_rescues,
            "runtime_failure": counts["runtime_failure"],
        },
        "rates": {
            "accuracy": counts["correct"] / len(diagnostics) if diagnostics else 0.0,
            "oracle_literal_rescue_among_failures": oracle_rescues / failures if failures else 0.0,
            "oracle_corrected_accuracy": (
                (counts["correct"] + oracle_rescues) / len(diagnostics) if diagnostics else 0.0
            ),
        },
        "runtime_failure_stages": dict(
            sorted(
                Counter(
                    str(row["runtime_stage"])
                    for row in diagnostics
                    if row["category"] == "runtime_failure"
                ).items()
            )
        ),
        "inputs": {
            "transformed_eval": file_sha256(transformed_eval_path),
            "original_predictions": file_sha256(original_predictions_path),
            "transformed_predictions": file_sha256(transformed_predictions_path),
        },
        "diagnostics_sha256": file_sha256(rows_path),
        "interpretation": (
            "literal_copy_error is a conservative lower bound: the original prediction must "
            "be correct, strict or literal-free alpha structure must match, and changing only "
            "position-aligned literals must execute to the transformed answer"
        ),
    }
    write_json(output / "summary.json", report)
    return report


def _write_curve_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 820, 500
    left, right, top, bottom = 80, 30, 40, 70
    factors = sorted({int(row["factor"]) for row in rows})
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    x_values = {factor: math.log10(factor) for factor in factors}
    x_min, x_max = min(x_values.values()), max(x_values.values())
    x_span = x_max - x_min or 1.0
    colors = ["#0b6e75", "#cf5c36", "#2f4858", "#d6a419", "#6a994e", "#7f4f24"]

    def x(factor: int) -> float:
        return left + (x_values[factor] - x_min) / x_span * (width - left - right)

    def y(accuracy: float) -> float:
        return top + (1.0 - accuracy) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ef"/>',
        '<g font-family="Georgia, serif" fill="#243238">',
        '<text x="410" y="25" text-anchor="middle" font-size="18">Accuracy vs numeric magnitude</text>',
    ]
    for tick in range(0, 11, 2):
        accuracy = tick / 10
        yy = y(accuracy)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#d8d2c4"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-size="12">{accuracy:.1f}</text>')
    for factor in factors:
        xx = x(factor)
        label = "x1" if factor == 1 else f"x10^{int(math.log10(factor))}"
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="12">{label}</text>')
    parts.append(f'<text x="{left-55}" y="{height/2}" transform="rotate(-90 {left-55} {height/2})" text-anchor="middle" font-size="13">Answer accuracy</text>')
    parts.append(f'<text x="{(left+width-right)/2}" y="{height-18}" text-anchor="middle" font-size="13">Source-number scale factor</text>')
    for index, condition in enumerate(conditions):
        color = colors[index % len(colors)]
        series = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: int(row["factor"]),
        )
        points = " ".join(
            f"{x(int(row['factor'])):.1f},{y(float(row['accuracy'])):.1f}" for row in series
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in series:
            parts.append(f'<circle cx="{x(int(row["factor"])):.1f}" cy="{y(float(row["accuracy"])):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 18 * index
        parts.append(f'<line x1="{width-190}" y1="{legend_y}" x2="{width-168}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-160}" y="{legend_y+4}" font-size="12">{condition}</text>')
    parts.extend(["</g>", "</svg>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze_magnitude_curve(
    *,
    eval_paths: list[tuple[int, str | Path]],
    prediction_paths: list[tuple[str, int, str | Path]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a fixed-denominator accuracy curve for any matched conditions."""

    evaluations = {factor: read_jsonl(path) for factor, path in eval_paths}
    if len(evaluations) != len(eval_paths):
        raise ValueError("duplicate magnitude eval factor")
    parent_sets = {
        factor: [str(row.get("parent_example_id", row["example_id"])) for row in rows]
        for factor, rows in evaluations.items()
    }
    if len({tuple(ids) for ids in parent_sets.values()}) != 1:
        raise ValueError("magnitude evaluations do not share one ordered parent set")
    result_rows = []
    for condition, factor, path in prediction_paths:
        if factor not in evaluations:
            raise ValueError(f"no evaluation registered for factor {factor}")
        indexed = _prediction_index(read_jsonl(path), label=f"{condition} factor {factor}")
        eval_rows = evaluations[factor]
        missing = [str(row["example_id"]) for row in eval_rows if str(row["example_id"]) not in indexed]
        if missing:
            raise ValueError(f"{condition} factor {factor} is missing {len(missing)} predictions")
        correct = sum(
            bool(indexed[str(row["example_id"])].get("metrics", {}).get("final_answer_correct"))
            for row in eval_rows
        )
        result_rows.append(
            {
                "condition": condition,
                "factor": factor,
                "factor_label": "x1" if factor == 1 else f"x10^{int(math.log10(factor))}",
                "count": len(eval_rows),
                "correct": correct,
                "accuracy": correct / len(eval_rows) if eval_rows else 0.0,
                "predictions_sha256": file_sha256(path),
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "accuracy_by_magnitude.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]) if result_rows else [])
        if result_rows:
            writer.writeheader()
            writer.writerows(result_rows)
    svg_path = output / "accuracy_by_magnitude.svg"
    _write_curve_svg(svg_path, result_rows)
    report = {
        "schema_version": "ccpu.paper1.gsm8k_magnitude_curve_summary.v1",
        "protocol_id": MAGNITUDE_CURVE_PROTOCOL_ID,
        "common_parent_count": len(next(iter(parent_sets.values()))) if parent_sets else 0,
        "factors": sorted(evaluations),
        "results": result_rows,
        "outputs": {
            "csv": {"path": str(csv_path), "sha256": file_sha256(csv_path)},
            "svg": {"path": str(svg_path), "sha256": file_sha256(svg_path)},
        },
        "role": "exploratory common-parent magnitude curve; not a replacement for the 250/59 primary comparison",
    }
    write_json(output / "summary.json", report)
    return report
