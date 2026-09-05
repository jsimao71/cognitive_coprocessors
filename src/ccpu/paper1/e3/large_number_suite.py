"""Deterministic, execution-verified large-number GSM8K perturbations."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl_dataset.loaders import load_dataset

LARGE_NUMBER_PROTOCOL_ID = "paper1_gsm8k_large_number_v1"

_NUMBER_TEXT = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_QUESTION_NUMBER = re.compile(
    rf"(?<![\w./])(?P<currency>\$)?(?P<number>{_NUMBER_TEXT})(?![\w/])"
)
_EXPRESSION_NUMBER = re.compile(
    rf"(?<![\w.])(?P<currency>\$)?(?P<number>{_NUMBER_TEXT})(?![\w])"
)
_TRACE = re.compile(r"<<([^<>]+)>>")
_PERCENT_VALUE = re.compile(
    rf"(?P<number>{_NUMBER_TEXT})\s*(?:%|percent(?:age)?)",
    re.IGNORECASE,
)
_FRACTION_VALUE = re.compile(r"(?<!\w)(?P<num>\d+)\s*/\s*(?P<den>\d+)(?!\w)")
_UNSAFE = {
    "bounded_time_ratio": re.compile(
        r"\b(?:hours?\s+(?:a|per|every)\s+day|minutes?\s+(?:a|per|every)\s+hour|"
        r"seconds?\s+(?:a|per|every)\s+minute)\b",
        re.IGNORECASE,
    ),
    "clock_or_ratio_notation": re.compile(r"\b\d+\s*:\s*\d+\b"),
    "percentage": re.compile(r"%|\bpercent(?:age)?\b", re.IGNORECASE),
    "calendar_or_age": re.compile(
        r"\b(?:year|age|aged|old|born|calendar|date|century)\b",
        re.IGNORECASE,
    ),
    "hyphenated_numeric_modifier": re.compile(r"\b\d+(?:\.\d+)?-[A-Za-z]"),
    "numeric_ordinal": re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE),
    "lexicalized_number": re.compile(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"dozen)\b",
        re.IGNORECASE,
    ),
}

_BINARY: dict[type[ast.operator], Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}


class LargeNumberExclusion(ValueError):
    """Expected conservative exclusion from the transformed suite."""


def _fraction(text: Any) -> Fraction:
    value = str(text).strip().replace(",", "").replace("$", "")
    return Fraction(value)


def _format(value: Fraction, *, expression: bool = False) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    fraction = f"{value.numerator}/{value.denominator}"
    return f"({fraction})" if expression else fraction


def _evaluate(expression: str) -> Fraction:
    cleaned = expression.replace(",", "").replace("$", "").strip()
    node = ast.parse(cleaned, mode="eval").body

    def visit(item: ast.AST) -> Fraction:
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return Fraction(str(item.value))
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = visit(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and type(item.op) in _BINARY:
            return _BINARY[type(item.op)](visit(item.left), visit(item.right))
        raise LargeNumberExclusion(f"unsupported trace expression: {expression}")

    try:
        return visit(node)
    except (ZeroDivisionError, OverflowError) as error:
        raise LargeNumberExclusion(f"invalid trace expression: {expression}") from error


def _operator_signature(expression: str) -> list[str]:
    node = ast.parse(expression.replace(",", "").replace("$", ""), mode="eval")
    return [type(item.op).__name__ for item in ast.walk(node) if isinstance(item, ast.BinOp)]


def _rewrite_expression(expression: str, values: dict[Fraction, Fraction]) -> str:
    def replace(match: re.Match[str]) -> str:
        original = _fraction(match.group("number"))
        replacement = values.get(original, original)
        prefix = match.group("currency") or ""
        return f"{prefix}{_format(replacement, expression=True)}"

    return _EXPRESSION_NUMBER.sub(replace, expression)


def _question_mapping(question: str, factor: int) -> dict[Fraction, Fraction]:
    matches = list(_QUESTION_NUMBER.finditer(question))
    if not matches:
        raise LargeNumberExclusion("no transformable digit source quantities")
    percent_values = {
        _fraction(match.group("number")) for match in _PERCENT_VALUE.finditer(question)
    }
    fraction_values = {
        _fraction(match.group(name))
        for match in _FRACTION_VALUE.finditer(question)
        for name in ("num", "den")
    }
    mapping: dict[Fraction, Fraction] = {}
    for match in matches:
        value = _fraction(match.group("number"))
        suffix = question[match.end() :]
        if re.match(r"\s*(?:%|percent(?:age)?\b)", suffix, re.IGNORECASE):
            continue
        transformed = value * factor
        if value in mapping and mapping[value] != transformed:
            raise LargeNumberExclusion("inconsistent source-number mapping")
        mapping[value] = transformed
    if not mapping:
        raise LargeNumberExclusion("no non-ratio digit source quantities")
    if set(mapping) & (percent_values | fraction_values):
        raise LargeNumberExclusion("ambiguous source and dimensionless numeric value")
    if not any(abs(value) >= 1000 for value in mapping.values()):
        raise LargeNumberExclusion("transformation does not reach large-number threshold")
    return mapping


def _transform_question(question: str, mapping: dict[Fraction, Fraction]) -> str:
    transformed = _QUESTION_NUMBER.sub(
        lambda match: (
            f"{match.group('currency') or ''}"
            f"{_format(mapping.get(_fraction(match.group('number')), _fraction(match.group('number'))))}"
        ),
        question,
    )
    original_skeleton = _QUESTION_NUMBER.sub("<NUMBER>", question)
    transformed_skeleton = _QUESTION_NUMBER.sub("<NUMBER>", transformed)
    if original_skeleton != transformed_skeleton:
        raise LargeNumberExclusion("nonnumeric question surface changed")
    if len(_QUESTION_NUMBER.findall(question)) != len(_QUESTION_NUMBER.findall(transformed)):
        raise LargeNumberExclusion("numeric span count changed")
    return transformed


def _transform_trace(
    reasoning: str, source_mapping: dict[Fraction, Fraction], expected: Any
) -> tuple[list[dict[str, Any]], Fraction]:
    mapping = dict(source_mapping)
    trace = []
    traced_source_values: set[Fraction] = set()
    for index, match in enumerate(_TRACE.finditer(reasoning)):
        inner = match.group(1)
        if "=" not in inner:
            raise LargeNumberExclusion("trace marker has no result separator")
        expression, result_text = inner.rsplit("=", 1)
        traced_source_values.update(
            value
            for match in _EXPRESSION_NUMBER.finditer(expression)
            if (value := _fraction(match.group("number"))) in source_mapping
        )
        try:
            original_result = _fraction(result_text)
            evaluated_original = _evaluate(expression)
        except (ValueError, SyntaxError) as error:
            raise LargeNumberExclusion("trace is not deterministic arithmetic") from error
        if evaluated_original != original_result:
            raise LargeNumberExclusion("source trace equation does not verify")
        transformed_expression = _rewrite_expression(expression, mapping)
        try:
            transformed_result = _evaluate(transformed_expression)
        except (ValueError, SyntaxError) as error:
            raise LargeNumberExclusion("transformed trace does not execute") from error
        if _operator_signature(expression) != _operator_signature(transformed_expression):
            raise LargeNumberExclusion("operator graph changed during transformation")
        if original_result in mapping and mapping[original_result] != transformed_result:
            raise LargeNumberExclusion("ambiguous numeric-value dependency collision")
        mapping[original_result] = transformed_result
        trace.append(
            {
                "index": index,
                "original_expression": expression,
                "original_result": _format(original_result),
                "transformed_expression": transformed_expression,
                "transformed_result": _format(transformed_result),
                "operator_signature": _operator_signature(expression),
            }
        )
    if not trace:
        raise LargeNumberExclusion("no executable hidden arithmetic trace")
    missing_sources = set(source_mapping) - traced_source_values
    if missing_sources:
        raise LargeNumberExclusion("source quantities missing from hidden trace")
    original_answer = _fraction(expected)
    transformed_answer = mapping.get(original_answer)
    if transformed_answer is None:
        raise LargeNumberExclusion("final answer is not grounded in the hidden trace")
    if transformed_answer != _evaluate(trace[-1]["transformed_expression"]):
        raise LargeNumberExclusion("final transformed answer is not the terminal trace result")
    if transformed_answer.denominator != 1:
        raise LargeNumberExclusion("non-integer transformed answer")
    if transformed_answer == original_answer:
        raise LargeNumberExclusion("final answer is unchanged by magnitude transformation")
    return trace, transformed_answer


def _transform_row(raw: dict[str, Any], frozen: dict[str, Any], factor: int) -> dict[str, Any]:
    question = str(raw["question"])
    normalized_question = re.sub(r"\s+", " ", question).strip().casefold()
    if fingerprint(normalized_question) != frozen["question_sha256"]:
        raise ValueError(f"source question hash changed for {frozen['example_id']}")
    for reason, pattern in _UNSAFE.items():
        if pattern.search(question):
            raise LargeNumberExclusion(reason)
    mapping = _question_mapping(question, factor)
    transformed_question = _transform_question(question, mapping)
    trace, transformed_answer = _transform_trace(
        str(raw["gold_reasoning"]), mapping, raw["answer"]
    )
    if _fraction(raw["answer"]) != _fraction(frozen["reference_return"]):
        raise ValueError(f"source answer changed for {frozen['example_id']}")
    return {
        "schema_version": "ccpu.paper1.gsm8k_large_number.v1",
        "protocol_id": LARGE_NUMBER_PROTOCOL_ID,
        "dataset": "gsm8k",
        "split": "test_large_number",
        "example_id": f"{frozen['example_id']}:large-v1",
        "parent_example_id": frozen["example_id"],
        "source_row": frozen["source_row"],
        "difficulty_steps": frozen["difficulty_steps"],
        "difficulty_stratum": frozen["difficulty_stratum"],
        "question": transformed_question,
        "question_sha256": fingerprint(transformed_question),
        "original_question_sha256": frozen["question_sha256"],
        "reference_return": _format(transformed_answer),
        "original_reference_return": str(frozen["reference_return"]),
        "effective_scope": {
            **frozen["effective_scope"],
            "id": f"{frozen['effective_scope']['id']}:large-v1",
        },
        "source_fields_visible_to_model": ["question"],
        "transformation": {
            "factor": factor,
            "transformed_source_value_count": len(mapping),
            "source_value_mapping": {
                _format(key): _format(value)
                for key, value in sorted(mapping.items(), key=lambda item: item[0])
            },
            "hidden_execution_trace": trace,
            "trace_visible_to_model": False,
            "answer_visible_to_model": False,
        },
    }


def freeze_large_number_gsm8k(
    *,
    source_path: str | Path,
    official_eval_path: str | Path,
    output_dir: str | Path,
    expected_source_sha256: str,
    factor: int = 1000,
) -> dict[str, Any]:
    """Freeze conservative paired large-number descendants without model calls."""

    if factor < 100:
        raise ValueError("large-number factor must be at least 100")
    if file_sha256(source_path) != expected_source_sha256:
        raise ValueError("GSM8K source hash differs from the pinned source")
    raw_rows = load_dataset("gsm8k", source_path, "test")
    frozen_rows = read_jsonl(official_eval_path)
    accepted = []
    excluded = []
    for frozen in frozen_rows:
        source_row = int(frozen["source_row"])
        try:
            accepted.append(_transform_row(raw_rows[source_row], frozen, factor))
        except LargeNumberExclusion as error:
            excluded.append(
                {
                    "example_id": frozen["example_id"],
                    "source_row": source_row,
                    "question_sha256": frozen["question_sha256"],
                    "reason": str(error),
                }
            )
    if not accepted:
        raise ValueError("large-number eligibility produced no accepted records")
    output = Path(output_dir)
    accepted_path = write_jsonl(output / "large.jsonl", accepted)
    excluded_path = write_jsonl(output / "excluded.jsonl", excluded)
    reason_counts: dict[str, int] = {}
    for row in excluded:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
    eligible_by_difficulty: dict[str, int] = {}
    answer_magnitude_bands = {"4_to_6_digits": 0, "7_to_9_digits": 0, "10plus_digits": 0}
    source_value_counts = []
    for row in accepted:
        stratum = str(row["difficulty_stratum"])
        eligible_by_difficulty[stratum] = eligible_by_difficulty.get(stratum, 0) + 1
        answer = _fraction(row["reference_return"])
        digits = len(str(abs(answer.numerator)))
        band = "4_to_6_digits" if digits <= 6 else "7_to_9_digits" if digits <= 9 else "10plus_digits"
        answer_magnitude_bands[band] += 1
        source_value_counts.append(int(row["transformation"]["transformed_source_value_count"]))
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_large_number_manifest.v1",
        "protocol_id": LARGE_NUMBER_PROTOCOL_ID,
        "factor": factor,
        "source": {
            "path": str(source_path),
            "sha256": expected_source_sha256,
        },
        "official_eval": {
            "path": str(official_eval_path),
            "sha256": file_sha256(official_eval_path),
            "count": len(frozen_rows),
        },
        "counts": {
            "eligible": len(accepted),
            "excluded": len(excluded),
            "exclusion_reasons": dict(sorted(reason_counts.items())),
            "eligible_by_difficulty": dict(sorted(eligible_by_difficulty.items())),
            "answer_magnitude_bands": answer_magnitude_bands,
            "transformed_source_values": {
                "minimum": min(source_value_counts),
                "maximum": max(source_value_counts),
                "total": sum(source_value_counts),
            },
        },
        "output_sha256": {
            "large": file_sha256(accepted_path),
            "excluded": file_sha256(excluded_path),
        },
        "eligible_parent_ids_sha256": fingerprint(
            [row["parent_example_id"] for row in accepted]
        ),
        "prompt_fields": ["question"],
        "hidden_trace_visible_to_model": False,
        "answers_visible_to_model": False,
        "selection_role": (
            "exploratory paired large-number robustness; frozen before transformed inference"
        ),
        "transformation_policy": (
            "scale registered digit source quantities, propagate values through every "
            "verified hidden arithmetic equation, preserve operator signatures, and reject "
            "unsafe or ambiguous records"
        ),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
