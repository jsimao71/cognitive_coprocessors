"""Resumable multi-model remote teacher for answer-hidden semantic programs."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

from .teacher import ALLOWED_OPERATORS, _response_json, _skill_bundle_sha256, _skill_bundle_text


def _model_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("remote teacher config requires a non-empty models list")
    if len(entries) > 5:
        raise ValueError("remote teacher cascade is limited to five models")
    normalized = []
    for entry in entries:
        value = {"id": entry} if isinstance(entry, str) else dict(entry)
        if not value.get("id"):
            raise ValueError("every remote teacher model requires an id")
        normalized.append(value)
    return normalized


def _answer_hidden_request(row: dict[str, Any]) -> dict[str, Any]:
    request = {
        "dataset": row["dataset"],
        "source_id": row["source_id"],
        "effective_scope": row["effective_scope"],
        "question": row["question"],
        "parts": [part for part in row["parts"] if part.get("teacher_input_default", True)],
        "operator_registry": ALLOWED_OPERATORS,
    }
    if row.get("source_context"):
        request["source_context"] = row["source_context"]
    return request


def _completion(
    config: dict[str, Any],
    model: dict[str, Any],
    skill: str,
    request: dict[str, Any],
    feedback: str,
) -> str:
    try:
        import litellm
    except ImportError as error:
        raise RuntimeError("install the 'teacher' extra to call a remote teacher") from error

    api_key_env = str(config.get("api_key_env", "OPENROUTER_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"required teacher credential environment variable is unset: {api_key_env}"
        )
    provider = str(config.get("provider", "openrouter"))
    model_id = str(model["id"])
    qualified = model_id if model_id.startswith(f"{provider}/") else f"{provider}/{model_id}"
    user = (
        "Compile every supplied question part in source order. Return one JSON object with an "
        "annotations array containing exactly one annotation. Each part_mapping requires "
        "part_id, status, asl, semantic_notes, assumptions, and confidence. Set answer_hidden "
        "and rationale_hidden true; rationale_assisted and manually_reviewed false; repair_round "
        "0. Do not expose or infer a hidden benchmark answer. Return JSON only.\n"
    )
    if feedback:
        user += f"A previous model failed validation. Correct this issue: {feedback}\n"
    user += json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    kwargs: dict[str, Any] = {
        "model": qualified,
        "api_key": api_key,
        "messages": [{"role": "system", "content": skill}, {"role": "user", "content": user}],
        "temperature": float(config.get("temperature", 0)),
        "max_tokens": int(config.get("max_tokens", 1200)),
        "timeout": float(config.get("timeout_seconds", 120)),
    }
    if model.get("response_format", False):
        kwargs["response_format"] = {"type": "json_object"}
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise TypeError("teacher returned no text content")
    return content


def _safe_error(error: BaseException) -> str:
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(error))
    return f"{type(error).__name__}: {message}"[:1000]


def _is_rate_limit(error: str) -> bool:
    lowered = error.casefold()
    return any(cue in lowered for cue in ("ratelimit", "rate limit", "429", "quota", "credits"))


def _annotation(value: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    annotations = value.get("annotations")
    if (
        isinstance(annotations, list)
        and annotations
        and all(isinstance(item, dict) and "part_id" in item for item in annotations)
    ):
        annotation = {
            "dataset": row["dataset"],
            "source_id": row["source_id"],
            "part_mappings": annotations,
        }
    elif isinstance(annotations, list) and len(annotations) == 1:
        annotation = annotations[0]
    elif "part_mappings" in value:
        annotation = value
    else:
        raise ValueError("response must contain one annotation or a part-mapping array")
    if not isinstance(annotation, dict):
        raise TypeError("annotation must be a JSON object")
    if str(annotation.get("dataset")) != str(row["dataset"]):
        raise ValueError("annotation dataset does not match request")
    if str(annotation.get("source_id")) != str(row["source_id"]):
        raise ValueError("annotation source_id does not match request")
    return annotation


def _candidate_asl(annotation: dict[str, Any], row: dict[str, Any]) -> tuple[str, list[str]]:
    mappings = annotation.get("part_mappings")
    if not isinstance(mappings, list):
        raise TypeError("annotation part_mappings must be a list")
    expected_ids = [
        int(part["part_id"]) for part in row["parts"] if part.get("teacher_input_default", True)
    ]
    actual_ids = [int(mapping.get("part_id", -1)) for mapping in mappings]
    if actual_ids != expected_ids:
        raise ValueError(f"part ids must exactly match source order: expected {expected_ids}")
    statements = []
    for mapping in mappings:
        if mapping.get("status") != "ok":
            raise ValueError(f"part {mapping.get('part_id')} status is not ok")
        asl = mapping.get("asl")
        if not isinstance(asl, list) or not all(isinstance(item, str) for item in asl):
            raise TypeError(f"part {mapping.get('part_id')} asl must be a string list")
        statements.extend(item.strip() for item in asl if item.strip())
    full_asl = "\n".join(statements)
    validation = validate_asl(full_asl, effective_scope=row["effective_scope"])
    required = ("syntax_verified", "lower_verified", "type_verified", "execution_verified")
    errors = [name for name in required if not validation[name]]
    errors.extend(validation["errors"])
    if errors:
        raise ValueError("ASL validation failed: " + "; ".join(dict.fromkeys(errors)))
    return full_asl, statements


def _summary(
    *,
    input_path: Path,
    output: Path,
    rows: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    models: list[dict[str, Any]],
    skill_sha: str,
    stop_reason: str | None,
) -> dict[str, Any]:
    accepted_keys = {(str(row["dataset"]), str(row["source_id"])) for row in annotations}
    failed_keys = {(str(row["dataset"]), str(row["source_id"])) for row in failures}
    by_model_attempts = Counter(str(row["model"]) for row in attempts)
    by_model_accepted = Counter(str(row["teacher"]["model"]) for row in annotations)
    error_counts = Counter(
        str(row.get("outcome")) for row in attempts if row.get("outcome") != "accepted"
    )
    summary = {
        "schema_version": "ccpu.dsl_dataset.remote_program_teacher_summary.v1",
        "input_sha256": file_sha256(input_path),
        "skill_sha256": skill_sha,
        "provider": "openrouter",
        "models": [str(model["id"]) for model in models],
        "input_program_count": len(rows),
        "accepted_program_count": len(accepted_keys),
        "failed_program_count": len(failed_keys),
        "processed_program_count": len(accepted_keys | failed_keys),
        "remaining_program_count": len(rows) - len(accepted_keys | failed_keys),
        "conversion_rate_input": len(accepted_keys) / len(rows) if rows else 0.0,
        "conversion_rate_processed": (
            len(accepted_keys) / len(accepted_keys | failed_keys)
            if accepted_keys or failed_keys
            else 0.0
        ),
        "attempt_count": len(attempts),
        "attempts_by_model": dict(sorted(by_model_attempts.items())),
        "accepted_by_model": dict(sorted(by_model_accepted.items())),
        "attempt_error_counts": dict(sorted(error_counts.items())),
        "stop_reason": stop_reason,
        "status": "complete" if len(accepted_keys | failed_keys) == len(rows) else "partial",
    }
    for name in ("attempts", "annotations", "failures"):
        path = output / f"{name}.jsonl"
        summary[f"{name}_sha256"] = file_sha256(path)
    write_json(output / "summary.json", summary)
    return summary


def generate_remote_programs(
    input_path: str | Path,
    skill_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Generate full answer-hidden ASL programs with validation-aware model fallback."""

    input_path = Path(input_path)
    rows = read_jsonl(input_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    models = _model_entries(config)
    max_attempts = min(int(config.get("max_attempts", len(models))), len(models), 5)
    output = Path(output_dir)
    annotations_path = output / "annotations.jsonl"
    failures_path = output / "failures.jsonl"
    attempts_path = output / "attempts.jsonl"
    annotations = read_jsonl(annotations_path) if annotations_path.exists() else []
    failures = read_jsonl(failures_path) if failures_path.exists() else []
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    completed = {(str(row["dataset"]), str(row["source_id"])) for row in [*annotations, *failures]}
    skill = _skill_bundle_text(skill_path)
    skill_sha = _skill_bundle_sha256(skill_path)
    checkpoint_every = int(config.get("checkpoint_every", 5))
    delay_seconds = float(config.get("delay_seconds", 0.5))
    stop_reason = None
    processed_this_run = 0

    for row in rows:
        key = (str(row["dataset"]), str(row["source_id"]))
        if key in completed:
            continue
        request = _answer_hidden_request(row)
        feedback = ""
        row_attempts = []
        accepted = None
        for attempt_index, model in enumerate(models[:max_attempts], start=1):
            response_text = ""
            try:
                response_text = _completion(config, model, skill, request, feedback)
                value = _response_json(response_text)
                candidate = _annotation(value, row)
                full_asl, _ = _candidate_asl(candidate, row)
                outcome = "accepted"
                error = None
                accepted = {
                    **candidate,
                    "schema_version": "ccpu.dsl_dataset.remote_program_annotation.v1",
                    "dataset": row["dataset"],
                    "source_id": row["source_id"],
                    "answer_hidden": True,
                    "rationale_hidden": True,
                    "rationale_assisted": False,
                    "repair_round": 0,
                    "manually_reviewed": False,
                    "full_asl": full_asl,
                    "teacher": {
                        "provider": config.get("provider", "openrouter"),
                        "model": model["id"],
                        "attempt": attempt_index,
                        "skill_sha256": skill_sha,
                    },
                }
            except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary by backend.
                error = _safe_error(exc)
                outcome = "rate_limited" if _is_rate_limit(error) else "invalid"
                feedback = error
            attempt_record = {
                "schema_version": "ccpu.dsl_dataset.remote_program_attempt.v1",
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "attempt": attempt_index,
                "model": model["id"],
                "outcome": outcome,
                "error": error,
                "response_text": response_text,
            }
            attempts.append(attempt_record)
            row_attempts.append(attempt_record)
            if accepted is not None:
                annotations.append(accepted)
                completed.add(key)
                break
            if delay_seconds:
                time.sleep(delay_seconds)

        if accepted is None:
            if row_attempts and all(item["outcome"] == "rate_limited" for item in row_attempts):
                stop_reason = (
                    "all configured models rate limited; leave current and later rows pending"
                )
                break
            failures.append(
                {
                    "schema_version": "ccpu.dsl_dataset.remote_program_failure.v1",
                    "dataset": row["dataset"],
                    "source_id": row["source_id"],
                    "attempt_count": len(row_attempts),
                    "final_error": feedback,
                }
            )
            completed.add(key)
        processed_this_run += 1
        if processed_this_run % checkpoint_every == 0:
            write_jsonl(attempts_path, attempts)
            write_jsonl(annotations_path, annotations)
            write_jsonl(failures_path, failures)
            print(f"checkpoint remote teacher: processed_this_run={processed_this_run}")

    write_jsonl(attempts_path, attempts)
    write_jsonl(annotations_path, annotations)
    write_jsonl(failures_path, failures)
    return _summary(
        input_path=input_path,
        output=output,
        rows=rows,
        annotations=annotations,
        failures=failures,
        attempts=attempts,
        models=models,
        skill_sha=skill_sha,
        stop_reason=stop_reason,
    )
