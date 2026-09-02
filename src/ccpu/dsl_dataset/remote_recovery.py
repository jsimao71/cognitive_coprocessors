"""Conservative recovery preparation for remote-teacher program campaigns."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl

from .remote_analysis import _program_metrics
from .remote_teacher import _annotation, _candidate_asl
from .teacher import _response_json


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["source_id"])


def _trusted_annotation(
    response_text: str, row: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Restore only request-owned envelope fields; never rewrite generated ASL."""

    value = _response_json(response_text)
    annotations = value.get("annotations")
    transformations: list[str] = []
    if isinstance(annotations, list) and len(annotations) == 1 and isinstance(
        annotations[0], dict
    ):
        candidate = annotations[0]
    elif "part_mappings" in value:
        candidate = value
    else:
        # The existing parser handles a flat list of part mappings without metadata.
        return _annotation(value, row), transformations

    for field in ("dataset", "source_id"):
        trusted = row[field]
        if str(candidate.get(field)) != str(trusted):
            candidate[field] = trusted
            transformations.append(f"restore_{field}")
    return _annotation(value, row), transformations


def _feedback(metrics: dict[str, Any]) -> list[str]:
    messages = []
    if not metrics["semantic_lint_valid"]:
        messages.append("The previous program failed semantic lint.")
    if not metrics["answer_correct"]:
        messages.append("The previous program did not match the hidden deterministic verifier.")
    return messages


def prepare_remote_recovery(
    *,
    source_paths: list[str | Path],
    remote_dirs: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Salvage safe envelope failures and create a strict, deduplicated retry queue."""

    sources: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes = {}
    for path_value in source_paths:
        path = Path(path_value)
        source_hashes[str(path)] = file_sha256(path)
        for row in read_jsonl(path):
            key = _key(row)
            if key in sources:
                raise ValueError(f"duplicate source key: {key}")
            sources[key] = row

    existing_strict: dict[tuple[str, str], dict[str, Any]] = {}
    prior_context: dict[tuple[str, str], dict[str, Any]] = {}
    attempts_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    remote_hashes = {}
    for directory_value in remote_dirs:
        directory = Path(directory_value)
        for name in ("annotations.jsonl", "attempts.jsonl", "failures.jsonl"):
            path = directory / name
            if path.exists():
                remote_hashes[str(path)] = file_sha256(path)
        for annotation in read_jsonl(directory / "annotations.jsonl"):
            key = _key(annotation)
            source = sources[key]
            metrics = _program_metrics(
                source, str(annotation["full_asl"]), annotation["part_mappings"]
            )
            context = {
                "previous_asl": str(annotation["full_asl"]),
                "validator_feedback": _feedback(metrics),
            }
            prior_context[key] = context
            if metrics["converted"]:
                existing_strict[key] = annotation
        attempts_path = directory / "attempts.jsonl"
        if attempts_path.exists():
            for attempt in read_jsonl(attempts_path):
                attempts_by_key[_key(attempt)].append(attempt)

    salvaged_executable: dict[tuple[str, str], dict[str, Any]] = {}
    salvaged_strict: dict[tuple[str, str], dict[str, Any]] = {}
    for key, attempts in attempts_by_key.items():
        if key in existing_strict:
            continue
        source = sources[key]
        for attempt in attempts:
            response_text = str(attempt.get("response_text", ""))
            if not response_text.strip():
                continue
            try:
                annotation, transformations = _trusted_annotation(response_text, source)
                if not transformations:
                    continue
                full_asl, _ = _candidate_asl(annotation, source)
            except (KeyError, TypeError, ValueError):
                continue
            metrics = _program_metrics(source, full_asl, annotation["part_mappings"])
            recovered = {
                **annotation,
                "schema_version": "ccpu.dsl_dataset.remote_program_recovery.v1",
                "full_asl": full_asl,
                "recovery": {
                    "method": "trusted_envelope_normalization",
                    "source_attempt": int(attempt["attempt"]),
                    "source_model": str(attempt["model"]),
                    "transformations": transformations,
                },
                "recovery_metrics": {
                    name: metrics[name]
                    for name in (
                        "syntax_verified",
                        "lower_verified",
                        "type_verified",
                        "execution_verified",
                        "semantic_lint_valid",
                        "answer_correct",
                        "converted",
                    )
                },
            }
            salvaged_executable[key] = recovered
            prior_context[key] = {
                "previous_asl": full_asl,
                "validator_feedback": _feedback(metrics),
            }
            if metrics["converted"]:
                salvaged_strict[key] = recovered
                break

    strict_keys = set(existing_strict) | set(salvaged_strict)
    retry_rows = []
    for key in sorted(set(sources) - strict_keys):
        row = dict(sources[key])
        context = prior_context.get(key)
        if context:
            row["recovery_context"] = context
        retry_rows.append(row)

    output = Path(output_dir)
    executable_path = write_jsonl(
        output / "salvaged_executable.jsonl",
        (salvaged_executable[key] for key in sorted(salvaged_executable)),
    )
    strict_path = write_jsonl(
        output / "salvaged_strict.jsonl",
        (salvaged_strict[key] for key in sorted(salvaged_strict)),
    )
    retry_path = write_jsonl(output / "retry_input.jsonl", retry_rows)
    strict_index_path = write_jsonl(
        output / "strict_index.jsonl",
        (
            {
                "dataset": dataset,
                "source_id": source_id,
                "provenance": "salvaged" if key in salvaged_strict else "existing",
            }
            for key in sorted(strict_keys)
            for dataset, source_id in (key,)
        ),
    )
    summary = {
        "schema_version": "ccpu.dsl_dataset.remote_program_recovery_summary.v1",
        "source_count": len(sources),
        "existing_strict_count": len(existing_strict),
        "salvaged_executable_count": len(salvaged_executable),
        "salvaged_strict_count": len(salvaged_strict),
        "combined_strict_count": len(strict_keys),
        "retry_count": len(retry_rows),
        "input_sha256": {"sources": source_hashes, "remote": remote_hashes},
        "output_sha256": {
            "salvaged_executable": file_sha256(executable_path),
            "salvaged_strict": file_sha256(strict_path),
            "retry_input": file_sha256(retry_path),
            "strict_index": file_sha256(strict_index_path),
        },
        "normalization_policy": [
            "restore dataset from trusted request envelope",
            "restore source_id from trusted request envelope",
            "do not alter generated ASL or part mappings",
            "rerun syntax, lowering, type, execution, semantic lint, and answer checks",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary
