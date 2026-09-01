"""Answer-blind annotation and exact-split materialization for F3."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.paper1.functor_data import protocol_rows

from .normalize import semantic_signature
from .parser import parse_f3_program
from .registry import registry_manifest
from .runtime import validate_f3_program

F3_PROMPT_VERSION = "paper1-f3-grounded-student-v1"

_STUDENT_INSTRUCTION = """Compile the raw quantitative document into canonical F3 grounded state/event forms. Represent only explicit observations, events, declarative relations, evidence, and query intent. Use one allowlisted call per line and end with query(...). Use at(path,time), source(exact_span), or cell(exact_row,exact_column) for grounding. Cell row and column labels must each be exact non-empty cells from the supplied table. Event forms are remove/add/consume/produce/transfer. Event quantities may be numeric, a path/reference, event_field(...), scale(...), or fraction(...). Relation forms are same/offset/older_than/younger_than/multiple/fraction_of/percent_of/percent_more/percent_less/sum_relation/difference_relation/absolute_difference/product_relation/quotient_relation/mean_relation/minimum_relation/maximum_relation/rate_relation. Query kinds are value/remaining_count/sum/mean/difference/absolute_difference/percentage_ratio/percentage_change. A mean explicitly requested by the question belongs in query("mean", reference1, reference2, ...), not mean_relation. Do not generate solution steps, hidden intermediate values, arbitrary arithmetic, assignments, explanations, or benchmark answers."""


def _raw_context(row: dict[str, Any]) -> str:
    context = row.get("source_context")
    if not context:
        return ""
    bounded = {
        "table": list(context.get("table", [])),
        "paragraphs": [
            {"order": item.get("order"), "text": str(item.get("text", ""))[:600]}
            for item in list(context.get("paragraphs", []))[:3]
        ],
    }
    return json.dumps(bounded, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def f3_prompt(row: dict[str, Any]) -> str:
    context = _raw_context(row)
    return (
        _STUDENT_INSTRUCTION
        + "\n\nInput:\n"
        + (f"Raw evidence: {context}\n" if context else "")
        + f"Problem: {row['question']}\nProgram:"
    )


def prepare_f3_annotation_batches(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 5,
    identities: set[tuple[str, str]] | None = None,
    max_train_examples: int | None = None,
) -> dict[str, Any]:
    """Write raw-source-only F3 teacher request batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_train_examples is not None and max_train_examples < 1:
        raise ValueError("max_train_examples must be positive")
    if identities is not None and max_train_examples is not None:
        raise ValueError("identities and max_train_examples are mutually exclusive")
    splits = protocol_rows(freeze_dir, expansion_train_path)
    requests = []
    for split in ("train", "dev", "test"):
        if max_train_examples is not None and split != "train":
            continue
        for row in splits[split]:
            key = (str(row["dataset"]), str(row["source_id"]))
            if identities is not None and key not in identities:
                continue
            request = {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "question": row["question"],
            }
            if row.get("source_context"):
                request["source_context"] = row["source_context"]
            requests.append(request)
            if max_train_examples is not None and len(requests) >= max_train_examples:
                break
    output = Path(output_dir)
    paths = []
    for index in range(0, len(requests), batch_size):
        paths.append(
            write_json(
                output / "requests" / f"batch_{index // batch_size:03d}.json",
                {
                    "schema_version": "ccpu.paper1.f3.annotation_batch.v1",
                    "answer_hidden": True,
                    "rationale_hidden": True,
                    "prior_programs_hidden": True,
                    "runtime_state_hidden": True,
                    "validator_values_hidden": True,
                    "items": requests[index : index + batch_size],
                },
            )
        )
    manifest = {
        "schema_version": "ccpu.paper1.f3.annotation_requests.v1",
        "example_count": len(requests),
        "batch_size": batch_size,
        "batch_count": len(paths),
        "answer_hidden": True,
        "rationale_hidden": True,
        "prior_programs_hidden": True,
        "runtime_state_hidden": True,
        "validator_values_hidden": True,
        "fixed_prompt": True,
        "fixed_icl": True,
        "teacher_fixed_icl_shots": 3,
        "selection": (
            {"kind": "train_prefix_pilot", "max_examples": max_train_examples}
            if max_train_examples is not None
            else {"kind": "exact_frozen_protocol"}
        ),
        "selected_identities": [
            {"dataset": row["dataset"], "source_id": row["source_id"]} for row in requests
        ],
        "input_sha256": {
            "freeze_manifest": file_sha256(Path(freeze_dir) / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
        },
        "batches": [{"file": path.name, "sha256": file_sha256(path)} for path in paths],
    }
    write_json(output / "requests.manifest.json", manifest)
    return manifest


def prepare_f3_retry_batches(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    rejected_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 5,
) -> dict[str, Any]:
    """Retry rejected IDs from raw input without validator or answer feedback."""

    identities = {
        (str(row["dataset"]), str(row["source_id"])) for row in read_jsonl(rejected_path)
    }
    report = prepare_f3_annotation_batches(
        freeze_dir,
        expansion_train_path,
        output_dir,
        batch_size=batch_size,
        identities=identities,
    )
    report["schema_version"] = "ccpu.paper1.f3.annotation_retry_requests.v1"
    report["raw_input_only"] = True
    report["prior_annotation_hidden"] = True
    report["validator_failure_hidden"] = True
    report["input_sha256"]["rejected"] = file_sha256(rejected_path)
    write_json(Path(output_dir) / "requests.manifest.json", report)
    return report


def _program(annotation: dict[str, Any]) -> str:
    value = annotation.get("f3_program", [])
    if isinstance(value, str):
        return "\n".join(line.strip() for line in value.splitlines() if line.strip())
    if not isinstance(value, list) or not all(isinstance(line, str) for line in value):
        raise TypeError("f3_program must be a string list")
    return "\n".join(line.strip() for line in value if line.strip())


def _returned(validation: dict[str, Any], scope_id: str) -> Any:
    return validation["validation"]["execution"]["workspace"][scope_id]["returned"]


def _equivalent(left: Any, right: Any) -> bool:
    try:
        return abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal("0.011")
    except (InvalidOperation, TypeError):
        return left == right


def validate_f3_annotations(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    annotation_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select the first grounded, executable, answer-verified attempt per source."""

    splits = protocol_rows(freeze_dir, expansion_train_path)
    sources = {
        (str(row["dataset"]), str(row["source_id"])): (split, row)
        for split, members in splits.items()
        for row in members
    }
    annotations: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    hashes = {}
    for attempt, raw_path in enumerate(annotation_paths):
        path = Path(raw_path)
        hashes[str(path)] = file_sha256(path)
        for annotation in read_jsonl(path):
            key = (str(annotation["dataset"]), str(annotation["source_id"]))
            annotations.setdefault(key, []).append((attempt, annotation))

    accepted = []
    rejected = []
    unattempted = []
    runtime_counts: Counter[str] = Counter()
    for key, (split, source) in sources.items():
        if key not in annotations:
            unattempted.append(
                {
                    "dataset": source["dataset"],
                    "source_id": source["source_id"],
                    "split": split,
                    "record_sha256": source["record_sha256"],
                }
            )
            continue
        selected = None
        failure_codes: list[str] = []
        for attempt, annotation in annotations.get(key, []):
            try:
                if annotation.get("f3_status") != "ok":
                    failure_codes = [f"teacher_{annotation.get('f3_status', 'missing')}"]
                    continue
                program = _program(annotation)
                validations = {
                    mode: validate_f3_program(
                        program,
                        question=source["question"],
                        source_context=source.get("source_context"),
                        effective_scope=source["effective_scope"],
                        mode=mode,
                    )
                    for mode in ("r0", "r1", "r2")
                }
                r2 = validations["r2"]
                if not r2["parse_valid"]:
                    failure_codes = ["parse_invalid"]
                elif not r2["evidence_valid"]:
                    failure_codes = ["evidence_invalid"]
                elif not r2["lowerable"]:
                    failure_codes = ["not_lowerable"]
                elif not r2["type_valid"]:
                    failure_codes = ["type_invalid"]
                elif not r2["executable"]:
                    failure_codes = ["not_executable"]
                else:
                    scope_id = str(source["effective_scope"]["id"])
                    expected = source["state_after"][scope_id]["returned"]
                    if not _equivalent(_returned(r2, scope_id), expected):
                        # Never persist expected, actual, or a numerical delta in retry material.
                        failure_codes = ["answer_mismatch"]
                    else:
                        selected = (attempt, annotation, program, validations)
                        break
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                failure_codes = ["annotation_invalid"]
        base = {
            "dataset": source["dataset"],
            "source_id": source["source_id"],
            "split": split,
            "record_sha256": source["record_sha256"],
        }
        if selected is None:
            rejected.append(
                {
                    **base,
                    "failure_codes": failure_codes or ["missing_annotation"],
                    "attempt_count": len(annotations.get(key, [])),
                }
            )
            continue
        attempt, annotation, program, validations = selected
        parsed = parse_f3_program(program)
        for mode, validation in validations.items():
            runtime_counts[f"{mode}_executable"] += int(validation["executable"])
        accepted.append(
            {
                "schema_version": "ccpu.paper1.f3.mapping.v1",
                **base,
                "semantic_pattern_id": source["semantic_pattern_id"],
                "f3_semantic_pattern_id": semantic_signature(parsed),
                "question": source["question"],
                "source_context": source.get("source_context"),
                "effective_scope": source["effective_scope"],
                "reference_asl": source["asl"],
                "reference_return": source["state_after"][str(source["effective_scope"]["id"])][
                    "returned"
                ],
                "f3_program": program,
                "f3_ast": validations["r2"]["ast"],
                "runtime": {
                    mode: {
                        key: validation[key]
                        for key in (
                            "parse_valid",
                            "evidence_valid",
                            "lowerable",
                            "type_valid",
                            "executable",
                            "lowered_asl",
                        )
                    }
                    for mode, validation in validations.items()
                },
                "quality_grade": "execution_and_answer_verified",
                "provenance": {
                    "annotator": annotation.get("annotator", "local_codex"),
                    "annotation_attempt": attempt,
                    "answer_hidden": bool(annotation.get("answer_hidden", True)),
                    "rationale_hidden": bool(annotation.get("rationale_hidden", True)),
                    "prior_programs_hidden": True,
                    "runtime_state_hidden": True,
                    "fixed_prompt": True,
                    "fixed_icl": True,
                    "teacher_fixed_icl_shots": 3,
                },
            }
        )

    output = Path(output_dir)
    accepted_path = write_jsonl(output / "accepted.jsonl", accepted)
    rejected_path = write_jsonl(output / "rejected.jsonl", rejected)
    unattempted_path = write_jsonl(output / "unattempted.jsonl", unattempted)
    attempted_count = len(accepted) + len(rejected)
    form_counts = Counter(
        form.name for row in accepted for form in parse_f3_program(row["f3_program"]).forms
    )
    summary = {
        "schema_version": "ccpu.paper1.f3.annotation_validation.v1",
        "source_count": len(sources),
        "attempted_count": attempted_count,
        "source_coverage": attempted_count / len(sources),
        "accepted_count": len(accepted),
        "acceptance_rate": len(accepted) / attempted_count if attempted_count else 0.0,
        "rejected_count": len(rejected),
        "unattempted_count": len(unattempted),
        "split_accepted": dict(sorted(Counter(row["split"] for row in accepted).items())),
        "runtime_counts": dict(sorted(runtime_counts.items())),
        "form_counts": dict(sorted(form_counts.items())),
        "unique_f3_patterns": len({row["f3_semantic_pattern_id"] for row in accepted}),
        "answer_mismatch_feedback_hidden": True,
        "annotation_sha256": hashes,
        "accepted_sha256": file_sha256(accepted_path),
        "rejected_sha256": file_sha256(rejected_path),
        "unattempted_sha256": file_sha256(unattempted_path),
        "registry": registry_manifest(),
    }
    write_json(output / "summary.json", summary)
    return summary


def build_f3_data(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    accepted_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build F3 SFT and evaluation files without changing frozen identities."""

    splits = protocol_rows(freeze_dir, expansion_train_path)
    labels = {
        (str(row["dataset"]), str(row["source_id"])): row for row in read_jsonl(accepted_path)
    }
    missing = {
        split: [
            (str(row["dataset"]), str(row["source_id"]))
            for row in members
            if (str(row["dataset"]), str(row["source_id"])) not in labels
        ]
        for split, members in splits.items()
    }
    if missing["test"]:
        raise ValueError("all 25 frozen test identities require validated F3 labels")
    output = Path(output_dir)
    files = {}
    for split in ("train", "dev"):
        rows = []
        for source in splits[split]:
            key = (str(source["dataset"]), str(source["source_id"]))
            if key not in labels:
                continue
            rows.append(
                {
                    "schema_version": "ccpu.paper1.f3.sft.v1",
                    "example_id": f"f3-sft-{fingerprint(f'{key[0]}:{key[1]}:{split}', 16)}",
                    "parent_source_id": source["source_id"],
                    "semantic_pattern_id": source["semantic_pattern_id"],
                    "f3_semantic_pattern_id": labels[key]["f3_semantic_pattern_id"],
                    "dataset": source["dataset"],
                    "condition": "f3",
                    "prompt_version": F3_PROMPT_VERSION,
                    "prompt": f3_prompt(source),
                    "target": labels[key]["f3_program"],
                }
            )
        path = write_jsonl(output / "sft" / f"{split}.jsonl", rows)
        files[split] = {"rows": len(rows), "sha256": file_sha256(path)}
    eval_rows = []
    for source in splits["test"]:
        key = (str(source["dataset"]), str(source["source_id"]))
        eval_rows.append(
            {
                "schema_version": "ccpu.paper1.f3.eval.v1",
                "example_id": f"f3-eval-{fingerprint(f'{key[0]}:{key[1]}:test', 16)}",
                "parent_source_id": source["source_id"],
                "semantic_pattern_id": source["semantic_pattern_id"],
                "f3_semantic_pattern_id": labels[key]["f3_semantic_pattern_id"],
                "dataset": source["dataset"],
                "condition": "f3",
                "prompt_version": F3_PROMPT_VERSION,
                "prompt": f3_prompt(source),
                "question": source["question"],
                "source_context": source.get("source_context"),
                "reference_program": labels[key]["f3_program"],
                "reference_asl": source["asl"],
                "effective_scope": source["effective_scope"],
            }
        )
    test_path = write_jsonl(output / "eval" / "test.jsonl", eval_rows)
    files["test"] = {"rows": len(eval_rows), "sha256": file_sha256(test_path)}
    manifest = {
        "schema_version": "ccpu.paper1.f3.data.v1",
        "prompt_version": F3_PROMPT_VERSION,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "retained_counts": {name: len(rows) - len(missing[name]) for name, rows in splits.items()},
        "missing_identities": missing,
        "all_frozen_test_ids_labeled": not missing["test"],
        "source_ids_preserved": True,
        "fixed_prompt": True,
        "student_fixed_icl_shots": 0,
        "runtime_state_in_prompt": False,
        "answer_in_prompt": False,
        "input_sha256": {
            "freeze_manifest": file_sha256(Path(freeze_dir) / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
            "accepted": file_sha256(accepted_path),
        },
        "files": files,
        "registry": registry_manifest(),
    }
    write_json(output / "data_manifest.json", manifest)
    return manifest
