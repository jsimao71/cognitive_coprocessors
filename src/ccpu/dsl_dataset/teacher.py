"""Shared request contract for remote LiteLLM and local Codex teachers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.dsl import parse_asl, validate_asl

ALLOWED_OPERATORS = [
    "=",
    "RETURN",
    "+",
    "-",
    "*",
    "/",
    "abs",
    "dec_pct",
    "inc_pct",
    "mean",
    "min",
    "max",
    "percent_of",
    "rate_times_duration",
    "sum",
]


def _teacher_parts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [part for part in row["parts"] if part.get("teacher_input_default", True)]


def teacher_request(
    row: dict[str, Any], part: dict[str, Any], *, skill_sha256: str, state_before: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "ccpu.dsl_dataset.teacher_request.v1",
        "asl_version": "asl-core-v0",
        "profile": "asl-arith-v0",
        "dataset": row["dataset"],
        "source_id": row["source_id"],
        "effective_scope": row["effective_scope"],
        "question": row["question"],
        "correct_answer": row["answer"],
        "parts": _teacher_parts(row),
        "current_part": part,
        "state_before": state_before,
        "operator_registry": ALLOWED_OPERATORS,
        "skill_sha256": skill_sha256,
    }


def prepare_teacher_requests(
    input_path: str | Path,
    skill_path: str | Path,
    output_path: str | Path,
    *,
    max_examples: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    skill_sha = file_sha256(skill_path)
    requests = [
        teacher_request(row, part, skill_sha256=skill_sha, state_before={})
        for row in rows
        for part in _teacher_parts(row)
    ]
    output = write_jsonl(output_path, requests)
    manifest = {
        "schema_version": "ccpu.dsl_dataset.teacher_request_manifest.v1",
        "input_sha256": file_sha256(input_path),
        "skill_sha256": skill_sha,
        "output_sha256": file_sha256(output),
        "example_count": len(rows),
        "request_count": len(requests),
        "remote_calls": 0,
    }
    write_json(Path(output_path).with_suffix(".manifest.json"), manifest)
    return manifest


def _response_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("teacher response contains no JSON object") from None
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("teacher response must be a JSON object")
    return value


def _completion(config: dict[str, Any], skill: str, request: dict[str, Any]) -> str:
    try:
        import litellm
    except ImportError as error:
        raise RuntimeError("install the 'teacher' extra to call a remote teacher") from error
    provider = str(config.get("provider", "")).strip()
    model = str(config["model"])
    qualified_model = (
        model if not provider or model.startswith(f"{provider}/") else f"{provider}/{model}"
    )
    response = litellm.completion(
        model=qualified_model,
        messages=[
            {"role": "system", "content": skill},
            {
                "role": "user",
                "content": "Compile only current_part. Return the required JSON response.\n"
                + json.dumps(request, ensure_ascii=False),
            },
        ],
        temperature=float(config.get("temperature", 0)),
        max_tokens=int(config.get("max_tokens", 512)),
        timeout=float(config.get("timeout_seconds", 90)),
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise TypeError("teacher returned no text content")
    return content


def _root_values(execution: dict[str, Any], scope_id: str) -> dict[str, Any]:
    return dict(execution["workspace"].get(scope_id, {}).get("values", {}))


def generate_teacher_mappings(
    input_path: str | Path,
    skill_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    max_examples: int | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    skill_path = Path(skill_path)
    skill = skill_path.read_text(encoding="utf-8")
    skill_sha = file_sha256(skill_path)
    raw_records = []
    accepted = []
    review: dict[str, list[dict[str, Any]]] = {
        "invalid_syntax": [],
        "dangling_refs": [],
        "answer_mismatch": [],
        "teacher_disagreement": [],
        "ambiguous": [],
    }
    for row in rows:
        cumulative_asl: list[str] = []
        state_before: dict[str, Any] = {}
        for part in _teacher_parts(row):
            request = teacher_request(row, part, skill_sha256=skill_sha, state_before=state_before)
            candidate = None
            raw_text = ""
            error_text = ""
            for attempt in range(retries + 1):
                try:
                    raw_text = _completion(config, skill, request)
                    candidate = _response_json(raw_text)
                    break
                except (RuntimeError, TypeError, ValueError) as error:
                    error_text = str(error)
                    if attempt < retries:
                        time.sleep(min(2**attempt, 4))
            raw_record = {
                "schema_version": "ccpu.dsl_dataset.teacher_raw.v1",
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "part_id": part["part_id"],
                "provider": config.get("provider"),
                "model": config["model"],
                "skill_sha256": skill_sha,
                "response_text": raw_text,
                "parse_error": error_text or None,
            }
            raw_records.append(raw_record)
            if candidate is None:
                review["invalid_syntax"].append(raw_record)
                continue
            status = str(candidate.get("status", "unsupported"))
            if status != "ok":
                review["ambiguous"].append({**raw_record, "teacher_response": candidate})
                continue
            asl = str(candidate.get("asl", "")).strip()
            full_asl = "\n".join([*cumulative_asl, asl])
            validation = validate_asl(full_asl, effective_scope=row["effective_scope"])
            if not validation["execution_verified"]:
                bucket = (
                    "dangling_refs"
                    if any("unresolved reference" in error for error in validation["errors"])
                    else "invalid_syntax"
                )
                review[bucket].append(
                    {**raw_record, "teacher_response": candidate, "validation": validation}
                )
                continue
            current_ast = parse_asl(asl, effective_scope=row["effective_scope"])
            state_after = _root_values(validation["execution"], str(row["effective_scope"]["id"]))
            accepted.append(
                {
                    "schema_version": "ccpu.dsl_dataset.accepted_mapping.v1",
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "source_id": row["source_id"],
                    "effective_scope": row["effective_scope"],
                    "part": part,
                    "asl": asl,
                    "ast": current_ast["records"],
                    "state_before": state_before,
                    "state_after": state_after,
                    "validation": {
                        key: validation[key]
                        for key in (
                            "syntax_verified",
                            "type_verified",
                            "scope_verified",
                            "execution_verified",
                        )
                    },
                    "quality_grade": "Q1_SINGLE_TEACHER_EXEC_VERIFIED",
                    "teacher": {
                        "provider": config.get("provider"),
                        "model": config["model"],
                        "skill_sha256": skill_sha,
                    },
                }
            )
            cumulative_asl.append(asl)
            state_before = state_after
    output = Path(output_dir)
    raw_path = write_jsonl(output / "raw.jsonl", raw_records)
    accepted_path = write_jsonl(output / "accepted.jsonl", accepted)
    for name, records in review.items():
        write_jsonl(output / "review" / f"{name}.jsonl", records)
    summary = {
        "schema_version": "ccpu.dsl_dataset.teacher_summary.v1",
        "input_sha256": file_sha256(input_path),
        "skill_sha256": skill_sha,
        "provider": config.get("provider"),
        "model": config["model"],
        "raw_sha256": file_sha256(raw_path),
        "accepted_sha256": file_sha256(accepted_path),
        "raw_count": len(raw_records),
        "accepted_count": len(accepted),
        "review_counts": {name: len(records) for name, records in review.items()},
    }
    write_json(output / "summary.json", summary)
    return summary
