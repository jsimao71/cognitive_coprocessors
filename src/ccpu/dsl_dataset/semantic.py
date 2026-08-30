"""Validate local or remote semantic annotations against source rows and ASL execution."""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

_ANONYMOUS = re.compile(r"^(?:step_?\d+|tmp_?\d*|result|value_?\d*|[xyz])$", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"^\s*([a-z_][a-z0-9_.]*)\s*(?:=|<-)\s*", re.IGNORECASE)
_RETURN = re.compile(r"^\s*RETURN\s+([a-z_][a-z0-9_.]*)\s*$", re.IGNORECASE)


def _expected_decimal(value: Any) -> Decimal:
    if isinstance(value, list):
        if len(value) != 1:
            raise InvalidOperation("answer is not scalar")
        value = value[0]
    cleaned = re.sub(r"[^0-9.+-]", "", str(value).replace(",", ""))
    return Decimal(cleaned)


def _returned(execution: dict[str, Any], scope_id: str) -> Decimal:
    value = execution["workspace"][scope_id]["returned"]
    if value is None:
        raise InvalidOperation("program did not return a value")
    return Decimal(str(value))


def _statement_list(mapping: dict[str, Any]) -> list[str]:
    value = mapping.get("asl", [])
    if isinstance(value, str):
        value = [line for line in value.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("part mapping 'asl' must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def semantic_lint(part_mappings: list[dict[str, Any]]) -> list[str]:
    """Reject operation-ledger patterns that execution checks cannot detect."""

    errors = []
    statements = [statement for mapping in part_mappings for statement in _statement_list(mapping)]
    targets = []
    for statement in statements:
        match = _ASSIGNMENT.match(statement)
        if match:
            target = match.group(1)
            targets.append(target)
            leaf = target.rsplit(".", 1)[-1]
            if _ANONYMOUS.fullmatch(target) or _ANONYMOUS.fullmatch(leaf):
                errors.append(f"anonymous operation-ledger target: {target}")
        if statement.lstrip().upper().startswith("RETURN ") and not _RETURN.fullmatch(statement):
            errors.append("RETURN must reference a named semantic state value")
    if not targets:
        errors.append("program creates no semantic state")
    if not any(_RETURN.fullmatch(statement) for statement in statements):
        errors.append("program has no named RETURN")
    return sorted(set(errors))


def prepare_local_annotation_batches(
    seed_paths: list[str | Path],
    output_dir: str | Path,
    *,
    batch_size: int = 5,
) -> dict[str, Any]:
    """Write answer-free request batches suitable for a local Codex teacher."""

    rows = []
    input_hashes = {}
    for seed_path in map(Path, seed_paths):
        input_hashes[str(seed_path)] = file_sha256(seed_path)
        for row in read_jsonl(seed_path):
            request = {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "effective_scope": row["effective_scope"],
                "question": row["question"],
                "parts": [
                    part for part in row["parts"] if part.get("teacher_input_default", True)
                ],
                "operator_registry": [
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
                ],
            }
            if row.get("source_context"):
                request["source_context"] = row["source_context"]
            rows.append(request)

    output = Path(output_dir)
    batch_paths = []
    for index in range(0, len(rows), batch_size):
        path = write_json(
            output / "requests" / f"batch_{index // batch_size:03d}.json",
            {
                "schema_version": "ccpu.dsl_dataset.local_codex_batch.v1",
                "answer_hidden": True,
                "rationale_hidden": True,
                "items": rows[index : index + batch_size],
            },
        )
        batch_paths.append(path)
    manifest = {
        "schema_version": "ccpu.dsl_dataset.local_codex_manifest.v1",
        "input_sha256": input_hashes,
        "example_count": len(rows),
        "batch_size": batch_size,
        "batch_count": len(batch_paths),
        "batches": [
            {"file": path.name, "sha256": file_sha256(path)} for path in batch_paths
        ],
        "answer_hidden": True,
        "rationale_hidden": True,
        "teacher_transport": "codex_cli",
        "codex_cli_batches_prepared": len(batch_paths),
        "litellm_calls": 0,
    }
    write_json(output / "requests.manifest.json", manifest)
    return manifest


def prepare_repair_batches(
    seed_paths: list[str | Path],
    rejected_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 5,
    repair_round: int = 1,
) -> dict[str, Any]:
    """Prepare explicitly rationale-assisted repair requests for rejected mappings."""

    seeds = {}
    input_hashes = {}
    for seed_path in map(Path, seed_paths):
        input_hashes[str(seed_path)] = file_sha256(seed_path)
        for row in read_jsonl(seed_path):
            seeds[(str(row["dataset"]), str(row["source_id"]))] = row
    rejected_path = Path(rejected_path)
    repairs = []
    for rejected in read_jsonl(rejected_path):
        key = (str(rejected["dataset"]), str(rejected["source_id"]))
        source = seeds[key]
        repairs.append(
            {
                "dataset": source["dataset"],
                "source_id": source["source_id"],
                "effective_scope": source["effective_scope"],
                "question": source["question"],
                "parts": [
                    part for part in source["parts"] if part.get("teacher_input_default", True)
                ],
                "source_context": source.get("source_context"),
                "previous_annotation": rejected.get("annotation"),
                "validation_failure": rejected["reason"],
                "validator_context": {
                    "correct_answer": source["answer"],
                    "gold_reasoning": source["gold_reasoning"],
                },
            }
        )
    output = Path(output_dir)
    paths = []
    for index in range(0, len(repairs), batch_size):
        paths.append(
            write_json(
                output / "requests" / f"batch_{index // batch_size:03d}.json",
                {
                    "schema_version": "ccpu.dsl_dataset.local_codex_repair_batch.v1",
                    "answer_hidden": False,
                    "rationale_hidden": False,
                    "rationale_assisted": True,
                    "repair_round": repair_round,
                    "items": repairs[index : index + batch_size],
                },
            )
        )
    manifest = {
        "schema_version": "ccpu.dsl_dataset.local_codex_repair_manifest.v1",
        "input_sha256": input_hashes,
        "rejected_sha256": file_sha256(rejected_path),
        "example_count": len(repairs),
        "batch_size": batch_size,
        "batch_count": len(paths),
        "batches": [{"file": path.name, "sha256": file_sha256(path)} for path in paths],
        "answer_hidden": False,
        "rationale_hidden": False,
        "rationale_assisted": True,
        "repair_round": repair_round,
        "teacher_transport": "codex_cli",
        "codex_cli_batches_prepared": len(paths),
        "litellm_calls": 0,
    }
    write_json(output / "requests.manifest.json", manifest)
    return manifest


def validate_semantic_annotations(
    seed_paths: list[str | Path],
    annotation_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Join teacher mappings to seeds, validate semantics, and write accepted/rejected rows."""

    seeds = {}
    seed_hashes = {}
    for seed_path in map(Path, seed_paths):
        seed_hashes[str(seed_path)] = file_sha256(seed_path)
        for row in read_jsonl(seed_path):
            seeds[(str(row["dataset"]), str(row["source_id"]))] = row

    annotations = {}
    annotation_hashes = {}
    duplicate_keys = set()
    for annotation_path in map(Path, annotation_paths):
        annotation_hashes[str(annotation_path)] = file_sha256(annotation_path)
        for row in read_jsonl(annotation_path):
            key = (str(row["dataset"]), str(row["source_id"]))
            if key in annotations:
                previous_round = int(annotations[key].get("repair_round", 0))
                current_round = int(row.get("repair_round", 0))
                if current_round == previous_round:
                    duplicate_keys.add(key)
                elif current_round > previous_round:
                    duplicate_keys.discard(key)
                    annotations[key] = row
            else:
                annotations[key] = row

    accepted = []
    rejected = []
    for key, source in seeds.items():
        annotation = annotations.get(key)
        try:
            if key in duplicate_keys:
                raise ValueError("duplicate semantic annotation")
            if annotation is None:
                raise ValueError("missing semantic annotation")
            mappings = annotation.get("part_mappings")
            if not isinstance(mappings, list) or not mappings:
                raise ValueError("annotation has no part_mappings")
            expected_parts = {
                int(part["part_id"])
                for part in source["parts"]
                if part.get("teacher_input_default", True)
            }
            actual_parts = {int(mapping["part_id"]) for mapping in mappings}
            if actual_parts != expected_parts:
                raise ValueError(
                    f"part coverage mismatch: actual={sorted(actual_parts)}, "
                    f"expected={sorted(expected_parts)}"
                )
            non_ok = [
                mapping
                for mapping in mappings
                if str(mapping.get("status", "unsupported")) != "ok"
            ]
            if non_ok:
                statuses = ", ".join(
                    f"part {mapping.get('part_id')}={mapping.get('status')}" for mapping in non_ok
                )
                raise ValueError(f"non-ok semantic parts: {statuses}")
            mappings = sorted(mappings, key=lambda mapping: int(mapping["part_id"]))
            lint_errors = semantic_lint(mappings)
            if lint_errors:
                raise ValueError("; ".join(lint_errors))
            asl = "\n".join(
                statement for mapping in mappings for statement in _statement_list(mapping)
            )
            validation = validate_asl(asl, effective_scope=source["effective_scope"])
            if not validation["execution_verified"]:
                raise ValueError("; ".join(validation["errors"]))
            expected = _expected_decimal(source["answer"])
            actual = _returned(validation["execution"], str(source["effective_scope"]["id"]))
            if abs(actual - expected) > Decimal("0.011"):
                raise ValueError(f"answer mismatch: actual={actual}, expected={expected}")
            accepted.append(
                {
                    "schema_version": "ccpu.dsl_dataset.semantic_mapping.v1",
                    "dataset": source["dataset"],
                    "split": source["split"],
                    "source_id": source["source_id"],
                    "record_sha256": source["record_sha256"],
                    "effective_scope": source["effective_scope"],
                    "question": source["question"],
                    "source_context": source.get("source_context"),
                    "parts": [
                        part
                        for part in source["parts"]
                        if part.get("teacher_input_default", True)
                    ],
                    "part_mappings": mappings,
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
                        "semantic_lint_verified": True,
                        "intermediate_trace_verified": bool(
                            annotation.get("rationale_assisted", False)
                        ),
                        "manually_reviewed": bool(annotation.get("manually_reviewed", False)),
                    },
                    "quality_grade": (
                        "Q4_MANUAL_SEMANTIC_GOLD"
                        if annotation.get("manually_reviewed", False)
                        else "Q1_LOCAL_CODEX_EXEC_VERIFIED"
                    ),
                    "provenance": {
                        "annotator": annotation.get("annotator", "local_codex"),
                        "answer_hidden_during_primary_generation": bool(
                            annotation.get("answer_hidden", True)
                        ),
                        "rationale_hidden_during_primary_generation": bool(
                            annotation.get("rationale_hidden", True)
                        ),
                        "rationale_assisted_repair": bool(
                            annotation.get("rationale_assisted", False)
                        ),
                        "repair_round": int(annotation.get("repair_round", 0)),
                    },
                }
            )
        except (InvalidOperation, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            rejected.append(
                {
                    "dataset": source["dataset"],
                    "source_id": source["source_id"],
                    "record_sha256": source["record_sha256"],
                    "reason": str(error),
                    "annotation": annotation,
                }
            )

    output = Path(output_dir)
    accepted_path = write_jsonl(output / "accepted.jsonl", accepted)
    rejected_path = write_jsonl(output / "rejected.jsonl", rejected)
    summary = {
        "schema_version": "ccpu.dsl_dataset.semantic_summary.v1",
        "seed_sha256": seed_hashes,
        "annotation_sha256": annotation_hashes,
        "accepted_sha256": file_sha256(accepted_path),
        "rejected_sha256": file_sha256(rejected_path),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate": len(accepted) / len(seeds) if seeds else 0.0,
        "by_dataset": {
            dataset: {
                "accepted": sum(row["dataset"] == dataset for row in accepted),
                "rejected": sum(row["dataset"] == dataset for row in rejected),
            }
            for dataset in sorted({key[0] for key in seeds})
        },
        "provenance_counts": {
            "answer_blind_primary": sum(
                row["provenance"]["answer_hidden_during_primary_generation"]
                and not row["provenance"]["rationale_assisted_repair"]
                for row in accepted
            ),
            "rationale_assisted_repair": sum(
                row["provenance"]["rationale_assisted_repair"] for row in accepted
            ),
            "manually_reviewed": sum(
                row["validation"]["manually_reviewed"] for row in accepted
            ),
            "repair_rounds": dict(
                sorted(
                    Counter(
                        str(row["provenance"]["repair_round"]) for row in accepted
                    ).items()
                )
            ),
        },
        "quality_grade": "Q1_LOCAL_CODEX_EXEC_VERIFIED",
        "claim_boundary": (
            "execution-verified semantic ASL with deterministic lint; "
            "not manual semantic gold"
        ),
        "semantic_lint": "anonymous targets and expression/literal returns rejected",
    }
    write_json(output / "summary.json", summary)
    return summary
