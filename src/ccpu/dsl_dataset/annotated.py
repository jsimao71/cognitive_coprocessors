"""Low-grade ASL bootstrap from dataset-provided arithmetic annotations."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

_GSM_EQUATION = re.compile(r"<<(.+?)>>")
_PERCENT = re.compile(r"(?P<number>(?:\d+(?:\.\d+)?|\.\d+))\s*%")


def _clean_expression(value: str) -> str:
    expression = value.strip().replace("$", "").replace(",", "")
    expression = expression.replace("[", "(").replace("]", ")")
    return _PERCENT.sub(r"(\g<number> / 100)", expression)


def _expected_decimal(value: Any) -> Decimal:
    if isinstance(value, list):
        if len(value) != 1:
            raise InvalidOperation("answer is not scalar")
        value = value[0]
    cleaned = re.sub(r"[^0-9.+-]", "", str(value).replace(",", ""))
    return Decimal(cleaned)


def _returned(execution: dict[str, Any], scope_id: str) -> Decimal:
    value = execution["workspace"][scope_id]["returned"]
    return Decimal(str(value))


def _gsm_program(row: dict[str, Any]) -> str:
    expressions = []
    for match in _GSM_EQUATION.finditer(str(row["gold_reasoning"])):
        annotated = match.group(1)
        if "=" not in annotated:
            continue
        expression, _ = annotated.rsplit("=", 1)
        expressions.append(_clean_expression(expression))
    if not expressions:
        raise ValueError("no GSM8K annotated equations")
    statements = [f"step_{index} = {expression}" for index, expression in enumerate(expressions, 1)]
    statements.append(f"RETURN step_{len(expressions)}")
    return "\n".join(statements)


def _tatqa_programs(row: dict[str, Any]) -> list[str]:
    expression = _clean_expression(str(row["gold_reasoning"]))
    if not expression:
        raise ValueError("no TAT-QA derivation")
    programs = [f"result = {expression}\nRETURN result"]
    if str(row["metadata"].get("scale", "")) == "percent":
        programs.append(f"result = ({expression}) * 100\nRETURN result")
    return programs


def bootstrap_annotated(input_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    accepted = []
    rejected = []
    input_hashes = {}
    for input_path in input_paths:
        input_path = Path(input_path)
        input_hashes[str(input_path)] = file_sha256(input_path)
        for row in read_jsonl(input_path):
            if not row["metadata"].get("arithmetic_compatible"):
                continue
            try:
                if row["dataset"] == "gsm8k":
                    programs = [_gsm_program(row)]
                elif row["dataset"] == "tatqa":
                    programs = _tatqa_programs(row)
                else:
                    raise ValueError("no annotation adapter for dataset")
                expected = _expected_decimal(row["answer"])
                failures = []
                validation = None
                asl = ""
                for candidate in programs:
                    candidate_validation = validate_asl(
                        candidate, effective_scope=row["effective_scope"]
                    )
                    if not candidate_validation["execution_verified"]:
                        failures.extend(candidate_validation["errors"])
                        continue
                    actual = _returned(
                        candidate_validation["execution"],
                        str(row["effective_scope"]["id"]),
                    )
                    if abs(actual - expected) <= Decimal("0.011"):
                        asl = candidate
                        validation = candidate_validation
                        break
                    failures.append(f"answer mismatch: actual={actual}, expected={expected}")
                if validation is None:
                    raise ValueError("; ".join(failures))
                accepted.append(
                    {
                        "schema_version": "ccpu.dsl_dataset.annotated_mapping.v1",
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "source_id": row["source_id"],
                        "record_sha256": row["record_sha256"],
                        "effective_scope": row["effective_scope"],
                        "question": row["question"],
                        "parts": [
                            part for part in row["parts"] if part.get("teacher_input_default", True)
                        ],
                        "asl": asl,
                        "ast": validation["ast"],
                        "ccir": validation["ccir"],
                        "state_after": validation["execution"]["workspace"],
                        "validation": {
                            "syntax_verified": True,
                            "type_verified": True,
                            "scope_verified": True,
                            "execution_verified": True,
                            "final_answer_verified": True,
                            "intermediate_trace_verified": False,
                            "manually_reviewed": False,
                        },
                        "quality_grade": "Q0_DATASET_DERIVATION_EXEC_VERIFIED",
                        "claim_boundary": "annotation-derived operation plan; not semantic teacher gold",
                    }
                )
            except (InvalidOperation, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                rejected.append(
                    {
                        "dataset": row["dataset"],
                        "source_id": row["source_id"],
                        "record_sha256": row["record_sha256"],
                        "reason": str(error),
                    }
                )
    output = Path(output_dir)
    accepted_path = write_jsonl(output / "accepted.jsonl", accepted)
    rejected_path = write_jsonl(output / "rejected.jsonl", rejected)
    summary = {
        "schema_version": "ccpu.dsl_dataset.annotated_summary.v1",
        "input_sha256": input_hashes,
        "accepted_sha256": file_sha256(accepted_path),
        "rejected_sha256": file_sha256(rejected_path),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "by_dataset": {
            dataset: {
                "accepted": sum(row["dataset"] == dataset for row in accepted),
                "rejected": sum(row["dataset"] == dataset for row in rejected),
            }
            for dataset in sorted(
                {row["dataset"] for row in accepted}.union(row["dataset"] for row in rejected)
            )
        },
        "quality_grade": "Q0_DATASET_DERIVATION_EXEC_VERIFIED",
        "teacher_calls": 0,
    }
    write_json(output / "summary.json", summary)
    return summary
