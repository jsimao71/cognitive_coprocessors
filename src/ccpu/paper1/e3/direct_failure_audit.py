"""Deterministic endpoint and failure audit for saved Direct generations."""

from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from ccpu.common.metrics import wilson_interval

DIRECT_SCORER_V2_ID = "paper1_direct_endpoint_and_trace_audit_v2"

_NUMBER = r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\$?\d[\d,]*(?:\.\d+)?)?"
_ENDPOINT_PATTERNS = (
    re.compile(
        rf"(?:final\s+)?answer\s*(?:is|:|=)\s*\**(?P<value>{_NUMBER})",
        re.IGNORECASE,
    ),
    re.compile(rf"\\boxed\s*\{{\s*(?P<value>{_NUMBER})\s*\}}", re.IGNORECASE),
    re.compile(
        rf"<(?P<tag>answer|final_answer|result|value)>\s*(?P<value>{_NUMBER})\s*</(?P=tag)>",
        re.IGNORECASE,
    ),
    re.compile(rf"####\s*(?P<value>{_NUMBER})", re.IGNORECASE),
)
_TERMINAL_NUMBER = re.compile(rf"(?P<value>{_NUMBER})\s*[.!]?\s*$")
_NUMERIC_TOKEN = re.compile(_NUMBER)
_EXPLICIT_VALUE = re.compile(
    rf"(?:=|equals?|gives?|result(?:\s+is)?|total(?:\s+is)?|therefore|thus|so)"
    rf"\s*[:=]?\s*(?P<value>{_NUMBER})",
    re.IGNORECASE,
)
_ARITHMETIC_EQUATION = re.compile(
    rf"(?P<expression>[($\d,.\s]+(?:[+*/^xX×÷-]\s*[($\d,.\s]+)+)"
    rf"\s*=\s*(?P<result>{_NUMBER})",
    re.MULTILINE,
)


def _fraction(value: Any) -> Fraction | None:
    if value is None:
        return None
    normalized = str(value).replace(",", "").replace("$", "").replace(" ", "")
    try:
        return Fraction(normalized)
    except (ValueError, ZeroDivisionError):
        return None


def _numeric_equal(left: Any, right: Any) -> bool:
    first = _fraction(left)
    second = _fraction(right)
    return first is not None and first == second


def extract_direct_endpoint_v2(text: str, *, allow_bare_terminal: bool) -> str | None:
    """Extract the latest approved endpoint, optionally accepting a bare terminal value."""

    matches = [match for pattern in _ENDPOINT_PATTERNS for match in pattern.finditer(text)]
    if allow_bare_terminal and (terminal := _TERMINAL_NUMBER.search(text)):
        matches.append(terminal)
    if not matches:
        return None
    return max(matches, key=lambda match: match.end()).group("value")


def _evaluate_expression(expression: str) -> Fraction:
    normalized = (
        expression.replace(",", "")
        .replace("$", "")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("^", "**")
    )
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    tree = ast.parse(normalized.strip(), mode="eval")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            source = ast.get_source_segment(normalized, node)
            return Fraction(str(source if source is not None else node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.denominator == 1:
                return left ** right.numerator
        raise ValueError("unsupported arithmetic expression")

    return visit(tree)


def _arithmetic_evidence(text: str) -> dict[str, Any]:
    checked = []
    for match in _ARITHMETIC_EQUATION.finditer(text):
        expression = match.group("expression").strip()
        claimed = _fraction(match.group("result"))
        try:
            computed = _evaluate_expression(expression)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
            continue
        if claimed is None:
            continue
        checked.append(
            {
                "expression": expression,
                "claimed": str(claimed),
                "computed": str(computed),
                "correct": claimed == computed,
            }
        )
    return {
        "checked_equations": checked,
        "checked_count": len(checked),
        "incorrect_count": sum(not item["correct"] for item in checked),
    }


def _expected_explicitly_mentioned(text: str, expected: Any) -> bool:
    return any(
        _numeric_equal(match.group("value"), expected)
        for match in _EXPLICIT_VALUE.finditer(text)
    )


def _score_row(row: dict[str, Any], eval_row: dict[str, Any], token_ceiling: int) -> dict[str, Any]:
    text = str(row.get("generated_text", ""))
    expected = eval_row["reference_return"]
    generated_tokens = int(row.get("generated_tokens", 0))
    ceiling_hit = generated_tokens >= token_ceiling
    unclosed_thinking = text.count("<think>") > text.count("</think>")
    generation_complete = not ceiling_hit and not unclosed_thinking
    endpoint = extract_direct_endpoint_v2(text, allow_bare_terminal=generation_complete)
    endpoint_correct = _numeric_equal(endpoint, expected)
    strict_correct = bool(row.get("metrics", {}).get("final_answer_correct", False))
    expected_mentioned = _expected_explicitly_mentioned(text, expected)
    arithmetic = _arithmetic_evidence(text)

    if endpoint_correct:
        category = "CORRECT_ENDPOINT" if strict_correct else "CORRECT_VALUE_FORMAT_MISS"
    elif expected_mentioned and ceiling_hit:
        category = "CORRECT_INTERMEDIATE_THEN_TRUNCATED"
    elif expected_mentioned:
        category = "CORRECT_INTERMEDIATE_THEN_OVERRIDE"
    elif arithmetic["incorrect_count"]:
        category = "ARITHMETIC_EXECUTION_ERROR"
    elif endpoint is None:
        category = "NO_SCORABLE_VALUE"
    else:
        category = "SEMANTIC_STRUCTURE_ERROR"

    manual_subtype_review = category == "SEMANTIC_STRUCTURE_ERROR"
    return {
        "schema_version": "ccpu.paper1.direct_failure_audit.v2",
        "example_id": row["example_id"],
        "parent_example_id": row.get("parent_example_id", row["example_id"]),
        "question_sha256": row["question_sha256"],
        "reference_return": expected,
        "strict_predicted_answer": row.get("predicted_answer"),
        "v2_predicted_answer": endpoint,
        "strict_correct": strict_correct,
        "v2_correct": endpoint_correct,
        "category": category,
        "token_ceiling": token_ceiling,
        "token_ceiling_hit": ceiling_hit,
        "unclosed_thinking": unclosed_thinking,
        "generation_complete": generation_complete,
        "expected_explicitly_mentioned": expected_mentioned,
        "arithmetic_evidence": arithmetic,
        "manual_subtype_review": manual_subtype_review,
        "manual_subtype_choices": (
            ["SEMANTIC_STRUCTURE_ERROR", "RELATION_OR_BINDING_ERROR"]
            if manual_subtype_review
            else []
        ),
    }


def audit_direct_predictions(
    *,
    eval_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    token_ceiling: int,
    common_support_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rescore saved Direct generations without model calls or outcome-dependent tuning."""

    if token_ceiling < 1:
        raise ValueError("token ceiling must be positive")
    eval_rows = read_jsonl(eval_path)
    predictions = read_jsonl(predictions_path)
    eval_by_id = {str(row["example_id"]): row for row in eval_rows}
    prediction_by_id = {str(row["example_id"]): row for row in predictions}
    if len(eval_by_id) != len(eval_rows) or len(prediction_by_id) != len(predictions):
        raise ValueError("Direct scorer inputs contain duplicate example IDs")
    if set(eval_by_id) != set(prediction_by_id):
        raise ValueError("Direct scorer requires complete identity-matched inputs")
    for example_id, row in prediction_by_id.items():
        if row.get("question_sha256") != eval_by_id[example_id].get("question_sha256"):
            raise ValueError(f"question hash mismatch for {example_id}")

    diagnostics = [
        _score_row(prediction_by_id[str(row["example_id"])], row, token_ceiling)
        for row in eval_rows
    ]
    support_ids = None
    if common_support_path is not None:
        support = read_json(common_support_path)
        support_ids = set(map(str, support["parent_example_ids"]))

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        strict = sum(row["strict_correct"] for row in rows)
        v2 = sum(row["v2_correct"] for row in rows)
        categories = Counter(row["category"] for row in rows)
        return {
            "count": count,
            "strict_correct": strict,
            "v2_correct": v2,
            "strict_accuracy": strict / count if count else 0.0,
            "v2_accuracy": v2 / count if count else 0.0,
            "v2_wilson_95": wilson_interval(v2, count),
            "format_rescues": categories["CORRECT_VALUE_FORMAT_MISS"],
            "token_ceiling_hits": sum(row["token_ceiling_hit"] for row in rows),
            "categories": dict(sorted(categories.items())),
            "manual_semantic_subtype_review": sum(
                row["manual_subtype_review"] for row in rows
            ),
        }

    output = Path(output_dir)
    diagnostics_path = write_jsonl(output / "diagnostics.jsonl", diagnostics)
    report = {
        "schema_version": "ccpu.paper1.direct_failure_audit_summary.v2",
        "scorer_id": DIRECT_SCORER_V2_ID,
        "deterministic": True,
        "taxonomy_boundary": (
            "Endpoint, truncation, and explicit arithmetic evidence are deterministic. "
            "Residual semantic failures require blind review to distinguish structure "
            "from relation/binding errors."
        ),
        "all_rows": metrics(diagnostics),
        "strict_common_support": (
            metrics(
                [
                    row
                    for row in diagnostics
                    if str(row["parent_example_id"]) in support_ids
                ]
            )
            if support_ids is not None
            else None
        ),
        "inputs": {
            "eval": {"path": str(eval_path), "sha256": file_sha256(eval_path)},
            "predictions": {
                "path": str(predictions_path),
                "sha256": file_sha256(predictions_path),
            },
            "common_support": (
                {
                    "path": str(common_support_path),
                    "sha256": file_sha256(common_support_path),
                }
                if common_support_path is not None
                else None
            ),
        },
        "outputs": {
            "diagnostics": {
                "path": str(diagnostics_path),
                "sha256": file_sha256(diagnostics_path),
            }
        },
    }
    write_json(output / "summary.json", report)
    return report
