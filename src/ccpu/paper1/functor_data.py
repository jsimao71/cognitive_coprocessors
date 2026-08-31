"""Answer-blind annotation and matched data materialization for functor_v1."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl

from .functor_runtime import parse_functor_program, validate_functor_program

FUNCTOR_PROMPT_VERSION = "paper1-functor-student-v1"

_CONDITIONS = {"f1", "f2"}


def _raw_context(row: dict[str, Any]) -> str:
    context = row.get("source_context")
    if not context:
        return ""
    bounded = {
        # Tables are retained in full; only prose is bounded, without target-based retrieval.
        "table": list(context.get("table", [])),
        "paragraphs": [
            {"order": paragraph.get("order"), "text": str(paragraph.get("text", ""))[:600]}
            for paragraph in list(context.get("paragraphs", []))[:3]
        ],
    }
    return json.dumps(bounded, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def functor_prompt(row: dict[str, Any], condition: str) -> str:
    """Render the fixed student prompt; only raw input varies by record."""

    if condition not in _CONDITIONS:
        raise ValueError(f"unsupported functor condition: {condition}")
    description = (
        "flat low-level F1 assignment functors with explicit target and operands"
        if condition == "f1"
        else "semantic F2 relation functors whose arithmetic and dependencies are runtime-lowered"
    )
    sections = [
        (
            f"Compile the raw quantitative document into {description}. "
            "Preserve source facts, entities, measured quantities, relations, temporal state, "
            "dependencies, and query intent. Return only one functor call per line. "
            "Do not calculate hidden intermediate values or explain."
        )
    ]
    context = _raw_context(row)
    sections.append(
        "Input:\n"
        + (f"Raw evidence: {context}\n" if context else "")
        + f"Problem: {row['question']}\nProgram:"
    )
    return "\n\n".join(sections)


def protocol_rows(
    freeze_dir: str | Path, expansion_train_path: str | Path
) -> dict[str, list[dict[str, Any]]]:
    """Load the exact existing 450/25/25 partition without resplitting."""

    freeze = Path(freeze_dir)
    original = {
        split: read_jsonl(freeze / "splits" / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    rows = {
        "train": [*original["train"], *read_jsonl(expansion_train_path)],
        "dev": original["dev"],
        "test": original["test"],
    }
    expected = {"train": 450, "dev": 25, "test": 25}
    for split, count in expected.items():
        if len(rows[split]) != count:
            raise ValueError(f"expected {count} {split} rows, got {len(rows[split])}")
    identity_sets = {
        split: {(str(row["dataset"]), str(row["source_id"])) for row in members}
        for split, members in rows.items()
    }
    if any(
        identity_sets[left] & identity_sets[right]
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise ValueError("source identity overlap in frozen protocol")
    return rows


def prepare_functor_annotation_batches(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 5,
) -> dict[str, Any]:
    """Prepare answer/rationale/ASL-free local teacher requests."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    splits = protocol_rows(freeze_dir, expansion_train_path)
    requests = []
    for split in ("train", "dev", "test"):
        for row in splits[split]:
            request = {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "question": row["question"],
            }
            if row.get("source_context"):
                request["source_context"] = row["source_context"]
            requests.append(request)
    output = Path(output_dir)
    batch_paths = []
    for index in range(0, len(requests), batch_size):
        batch_paths.append(
            write_json(
                output / "requests" / f"batch_{index // batch_size:03d}.json",
                {
                    "schema_version": "ccpu.paper1.functor_annotation_batch.v1",
                    "answer_hidden": True,
                    "rationale_hidden": True,
                    "prior_asl_hidden": True,
                    "blackboard_state_hidden": True,
                    "items": requests[index : index + batch_size],
                },
            )
        )
    manifest = {
        "schema_version": "ccpu.paper1.functor_annotation_requests.v1",
        "example_count": len(requests),
        "batch_size": batch_size,
        "batch_count": len(batch_paths),
        "answer_hidden": True,
        "rationale_hidden": True,
        "prior_asl_hidden": True,
        "blackboard_state_hidden": True,
        "fixed_prompt": True,
        "fixed_icl": True,
        "teacher_fixed_icl_shots": 3,
        "input_sha256": {
            "freeze_manifest": file_sha256(Path(freeze_dir) / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
        },
        "batches": [{"file": path.name, "sha256": file_sha256(path)} for path in batch_paths],
    }
    write_json(output / "requests.manifest.json", manifest)
    return manifest


def prepare_functor_retry_batches(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    rejected_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 5,
    retry_round: int = 1,
) -> dict[str, Any]:
    """Retry rejected identities from raw input without exposing prior outputs or failures."""

    if retry_round < 1:
        raise ValueError("retry_round must be positive")
    splits = protocol_rows(freeze_dir, expansion_train_path)
    sources = {
        (str(row["dataset"]), str(row["source_id"])): row
        for members in splits.values()
        for row in members
    }
    rejected_path = Path(rejected_path)
    identities = [(str(row["dataset"]), str(row["source_id"])) for row in read_jsonl(rejected_path)]
    requests = []
    for key in identities:
        source = sources[key]
        request = {
            "dataset": source["dataset"],
            "source_id": source["source_id"],
            "question": source["question"],
        }
        if source.get("source_context"):
            request["source_context"] = source["source_context"]
        requests.append(request)
    output = Path(output_dir)
    batch_paths = []
    for index in range(0, len(requests), batch_size):
        batch_paths.append(
            write_json(
                output / "requests" / f"batch_{index // batch_size:03d}.json",
                {
                    "schema_version": "ccpu.paper1.functor_annotation_retry_batch.v1",
                    "retry_round": retry_round,
                    "answer_hidden": True,
                    "rationale_hidden": True,
                    "prior_asl_hidden": True,
                    "blackboard_state_hidden": True,
                    "prior_annotation_hidden": True,
                    "validator_failure_hidden": True,
                    "items": requests[index : index + batch_size],
                },
            )
        )
    manifest = {
        "schema_version": "ccpu.paper1.functor_annotation_retry_requests.v1",
        "retry_round": retry_round,
        "example_count": len(requests),
        "batch_size": batch_size,
        "batch_count": len(batch_paths),
        "raw_input_only": True,
        "prior_annotation_hidden": True,
        "validator_failure_hidden": True,
        "answer_hidden": True,
        "rationale_hidden": True,
        "fixed_prompt": True,
        "fixed_icl": True,
        "teacher_fixed_icl_shots": 3,
        "input_sha256": {
            "freeze_manifest": file_sha256(Path(freeze_dir) / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
            "rejected": file_sha256(rejected_path),
        },
        "batches": [{"file": path.name, "sha256": file_sha256(path)} for path in batch_paths],
    }
    write_json(output / "requests.manifest.json", manifest)
    return manifest


def _program(annotation: dict[str, Any], condition: str) -> str:
    value = annotation.get(f"{condition}_program", [])
    if isinstance(value, str):
        return "\n".join(line.strip() for line in value.splitlines() if line.strip())
    if not isinstance(value, list) or not all(isinstance(line, str) for line in value):
        raise TypeError(f"{condition}_program must be a string list")
    return "\n".join(line.strip() for line in value if line.strip())


def _returned(validation: dict[str, Any], scope_id: str) -> Any:
    return validation["validation"]["execution"]["workspace"][scope_id]["returned"]


def _equivalent(left: Any, right: Any) -> bool:
    try:
        return abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal("0.011")
    except (InvalidOperation, TypeError):
        return left == right


def validate_functor_annotations(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    annotation_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate teacher targets only after answer-blind primary generation."""

    splits = protocol_rows(freeze_dir, expansion_train_path)
    sources = {
        (str(row["dataset"]), str(row["source_id"])): (split, row)
        for split, members in splits.items()
        for row in members
    }
    annotations: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    duplicates = set()
    annotation_hashes = {}
    for attempt, raw_path in enumerate(annotation_paths):
        path = Path(raw_path)
        annotation_hashes[str(path)] = file_sha256(path)
        seen_in_path = set()
        for annotation in read_jsonl(path):
            key = (str(annotation["dataset"]), str(annotation["source_id"]))
            if key in seen_in_path:
                duplicates.add((attempt, key))
            seen_in_path.add(key)
            annotations.setdefault(key, []).append((attempt, annotation))

    accepted = []
    rejected = []
    condition_counts: Counter[str] = Counter()
    for key, (split, source) in sources.items():
        attempts = annotations.get(key, [])
        failures = {}
        programs = {}
        validations = {}
        selected_annotations = {}
        selected_attempts = {}
        if not attempts:
            failures["record"] = ["missing primary annotation"]
        else:
            scope_id = str(source["effective_scope"]["id"])
            expected = source["state_after"][scope_id]["returned"]
            for condition in ("f1", "f2"):
                attempt_errors = []
                for attempt, annotation in attempts:
                    try:
                        if (attempt, key) in duplicates:
                            raise ValueError("duplicate annotation within attempt")
                        if annotation.get(f"{condition}_status") != "ok":
                            raise ValueError(
                                f"teacher status is {annotation.get(f'{condition}_status')!r}"
                            )
                        program = _program(annotation, condition)
                        validation = validate_functor_program(
                            program, condition, effective_scope=source["effective_scope"]
                        )
                        errors = list(validation["errors"])
                        if not validation["executable"]:
                            errors.append("program is not executable")
                        elif not _equivalent(_returned(validation, scope_id), expected):
                            errors.append(
                                "answer mismatch: "
                                f"actual={_returned(validation, scope_id)}, expected={expected}"
                            )
                        if errors:
                            attempt_errors = sorted(set(errors))
                            continue
                        programs[condition] = program
                        validations[condition] = validation
                        selected_annotations[condition] = annotation
                        selected_attempts[condition] = attempt
                        condition_counts[condition] += 1
                        break
                    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                        attempt_errors = [str(error)]
                if condition not in programs:
                    failures[condition] = attempt_errors or ["no valid annotation attempt"]
        base = {
            "dataset": source["dataset"],
            "source_id": source["source_id"],
            "split": split,
            "record_sha256": source["record_sha256"],
        }
        if failures:
            rejected.append(
                {
                    **base,
                    "failures": failures,
                    "annotation": attempts[-1][1] if attempts else None,
                    "attempt_count": len(attempts),
                }
            )
            continue
        accepted.append(
            {
                "schema_version": "ccpu.paper1.functor_mapping.v1",
                **base,
                "semantic_pattern_id": source["semantic_pattern_id"],
                "question": source["question"],
                "source_context": source.get("source_context"),
                "effective_scope": source["effective_scope"],
                "reference_asl": source["asl"],
                "reference_return": source["state_after"][str(source["effective_scope"]["id"])][
                    "returned"
                ],
                "f1_program": programs["f1"],
                "f2_program": programs["f2"],
                "f1_lowered_asl": validations["f1"]["lowered_asl"],
                "f2_lowered_asl": validations["f2"]["lowered_asl"],
                "provenance": {
                    "annotator": "local_codex",
                    "answer_hidden": all(
                        bool(selected_annotations[condition].get("answer_hidden", True))
                        for condition in ("f1", "f2")
                    ),
                    "rationale_hidden": all(
                        bool(selected_annotations[condition].get("rationale_hidden", True))
                        for condition in ("f1", "f2")
                    ),
                    "prior_asl_hidden": True,
                    "blackboard_state_hidden": True,
                    "fixed_prompt": True,
                    "fixed_icl": True,
                    "teacher_fixed_icl_shots": 3,
                    "f1_annotation_attempt": selected_attempts["f1"],
                    "f2_annotation_attempt": selected_attempts["f2"],
                    "condition_attempt_selection": "first_execution_and_answer_verified",
                },
            }
        )

    output = Path(output_dir)
    accepted_path = write_jsonl(output / "accepted.jsonl", accepted)
    rejected_path = write_jsonl(output / "rejected.jsonl", rejected)
    functor_counts = {
        condition: Counter(
            call.name
            for row in accepted
            for call in parse_functor_program(row[f"{condition}_program"], condition)
        )
        for condition in ("f1", "f2")
    }
    semantic_specific = {
        "offset",
        "absolute_difference",
        "multiple",
        "fraction_of",
        "percentage_ratio",
        "increase_percent",
        "decrease_percent",
        "rate_total",
        "per_unit_total",
        "remaining",
    }
    summary = {
        "schema_version": "ccpu.paper1.functor_annotation_validation.v1",
        "source_count": len(sources),
        "paired_accepted_count": len(accepted),
        "paired_acceptance_rate": len(accepted) / len(sources),
        "condition_execution_verified": dict(sorted(condition_counts.items())),
        "rejected_count": len(rejected),
        "rejection_counts": dict(
            sorted(Counter(condition for row in rejected for condition in row["failures"]).items())
        ),
        "representation_diagnostics": {
            condition: {
                "mean_calls": (
                    sum(
                        len(parse_functor_program(row[f"{condition}_program"], condition))
                        for row in accepted
                    )
                    / len(accepted)
                    if accepted
                    else 0.0
                ),
                "mean_characters": (
                    sum(len(row[f"{condition}_program"]) for row in accepted) / len(accepted)
                    if accepted
                    else 0.0
                ),
                "functor_counts": dict(sorted(functor_counts[condition].items())),
            }
            for condition in ("f1", "f2")
        },
        "f2_semantic_specific_program_fraction": (
            sum(
                bool(
                    semantic_specific
                    & {call.name for call in parse_functor_program(row["f2_program"], "f2")}
                )
                for row in accepted
            )
            / len(accepted)
            if accepted
            else 0.0
        ),
        "annotation_sha256": annotation_hashes,
        "accepted_sha256": file_sha256(accepted_path),
        "rejected_sha256": file_sha256(rejected_path),
        "primary_annotation_answer_blind": True,
        "primary_annotation_rationale_blind": True,
        "primary_annotation_prior_asl_blind": True,
    }
    write_json(output / "summary.json", summary)
    return summary


def build_functor_data(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    accepted_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build matched F1/F2 SFT and evaluation rows for the frozen protocol."""

    splits = protocol_rows(freeze_dir, expansion_train_path)
    accepted = {
        (str(row["dataset"]), str(row["source_id"])): row for row in read_jsonl(accepted_path)
    }
    expected_keys = {
        (str(row["dataset"]), str(row["source_id"]))
        for members in splits.values()
        for row in members
    }
    extra = sorted(set(accepted) - expected_keys)
    if extra:
        raise ValueError(
            f"functor labels contain identities outside the frozen protocol: {extra[:5]}"
        )
    missing_by_split = {
        split: [
            (str(row["dataset"]), str(row["source_id"]))
            for row in members
            if (str(row["dataset"]), str(row["source_id"])) not in accepted
        ]
        for split, members in splits.items()
    }
    if missing_by_split["test"]:
        raise ValueError("all 25 frozen test identities require paired validated labels")
    output = Path(output_dir)
    files = {}
    for condition in ("f1", "f2"):
        for split in ("train", "dev"):
            records = []
            for source in splits[split]:
                key = (str(source["dataset"]), str(source["source_id"]))
                if key not in accepted:
                    continue
                label = accepted[key]
                identity = f"{condition}:{key[0]}:{key[1]}"
                records.append(
                    {
                        "schema_version": "ccpu.paper1.functor_sft.v1",
                        "example_id": f"functor-sft-{fingerprint(identity, 16)}",
                        "parent_source_id": source["source_id"],
                        "semantic_pattern_id": source["semantic_pattern_id"],
                        "dataset": source["dataset"],
                        "condition": condition,
                        "prompt_version": FUNCTOR_PROMPT_VERSION,
                        "prompt": functor_prompt(source, condition),
                        "target": label[f"{condition}_program"],
                    }
                )
            path = write_jsonl(output / condition / "sft" / f"{split}.jsonl", records)
            files[f"{condition}_{split}"] = {"rows": len(records), "sha256": file_sha256(path)}
        eval_rows = []
        for source in splits["test"]:
            key = (str(source["dataset"]), str(source["source_id"]))
            label = accepted[key]
            identity = f"{condition}:{key[0]}:{key[1]}:test"
            eval_rows.append(
                {
                    "schema_version": "ccpu.paper1.functor_eval.v1",
                    "example_id": f"functor-eval-{fingerprint(identity, 16)}",
                    "parent_source_id": source["source_id"],
                    "semantic_pattern_id": source["semantic_pattern_id"],
                    "dataset": source["dataset"],
                    "condition": condition,
                    "prompt_version": FUNCTOR_PROMPT_VERSION,
                    "prompt": functor_prompt(source, condition),
                    "reference_program": label[f"{condition}_program"],
                    "reference_asl": source["asl"],
                    "effective_scope": source["effective_scope"],
                }
            )
        path = write_jsonl(output / condition / "eval" / "test.jsonl", eval_rows)
        files[f"{condition}_test"] = {"rows": len(eval_rows), "sha256": file_sha256(path)}

    manifest = {
        "schema_version": "ccpu.paper1.functor_data.v1",
        "prompt_version": FUNCTOR_PROMPT_VERSION,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "paired_label_counts": {
            split: len(splits[split]) - len(missing_by_split[split]) for split in splits
        },
        "missing_paired_label_counts": {split: len(missing_by_split[split]) for split in splits},
        "missing_paired_label_identities": missing_by_split,
        "source_ids_preserved": True,
        "all_frozen_test_ids_labeled": True,
        "fixed_prompt_per_condition": True,
        "fixed_icl_per_condition": True,
        "student_fixed_icl_shots": 0,
        "runtime_state_in_prompt": False,
        "intermediate_values_in_prompt": False,
        "input_sha256": {
            "freeze_manifest": file_sha256(Path(freeze_dir) / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
            "accepted": file_sha256(accepted_path),
        },
        "files": files,
    }
    write_json(output / "data_manifest.json", manifest)
    return manifest
