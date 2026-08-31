"""Answer-blind annotation and matched data materialization for functor_v1."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl

from .functor_runtime import validate_functor_program

FUNCTOR_PROMPT_VERSION = "paper1-functor-student-v1"

_FIXED_EXAMPLES = {
    "f1": (
        (
            "Mira has 12 cards. Jon has 5 more cards than Mira. How many cards does Jon have?",
            'value("mira.cards", 12)\nadd("jon.cards", "mira.cards", 5)\nquery("jon.cards")',
        ),
        (
            "A crate has 8 rows with 6 jars in each row. How many jars are there?",
            (
                'value("crate.rows", 8)\n'
                'value("crate.jars_per_row", 6)\n'
                'multiply("crate.jars_total", "crate.rows", "crate.jars_per_row")\n'
                'query("crate.jars_total")'
            ),
        ),
        (
            "A club has 60 junior members out of 240 members. What percentage are juniors?",
            (
                'value("club.junior_members", 60)\n'
                'value("club.total_members", 240)\n'
                'divide("club.junior_fraction", "club.junior_members", "club.total_members")\n'
                'multiply("club.junior_percentage", "club.junior_fraction", 100)\n'
                'query("club.junior_percentage")'
            ),
        ),
    ),
    "f2": (
        (
            "Mira has 12 cards. Jon has 5 more cards than Mira. How many cards does Jon have?",
            'given("mira.cards", 12)\noffset("jon.cards", "mira.cards", 5)\nquery("jon.cards")',
        ),
        (
            "A crate has 8 rows with 6 jars in each row. How many jars are there?",
            (
                'given("crate.rows", 8)\n'
                'given("crate.jars_per_row", 6)\n'
                'per_unit_total("crate.jars_total", "crate.rows", "crate.jars_per_row")\n'
                'query("crate.jars_total")'
            ),
        ),
        (
            "A club has 60 junior members out of 240 members. What percentage are juniors?",
            (
                'given("club.junior_members", 60)\n'
                'given("club.total_members", 240)\n'
                'percentage_ratio("club.junior_percentage", "club.junior_members", '
                '"club.total_members")\n'
                'query("club.junior_percentage")'
            ),
        ),
    ),
}


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

    if condition not in _FIXED_EXAMPLES:
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
    for index, (problem, program) in enumerate(_FIXED_EXAMPLES[condition], 1):
        sections.append(f"Fixed example {index} input:\nProblem: {problem}\nProgram:\n{program}")
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
        "input_sha256": {
            "freeze_manifest": file_sha256(Path(freeze_dir) / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
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
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = set()
    annotation_hashes = {}
    for raw_path in annotation_paths:
        path = Path(raw_path)
        annotation_hashes[str(path)] = file_sha256(path)
        for annotation in read_jsonl(path):
            key = (str(annotation["dataset"]), str(annotation["source_id"]))
            if key in annotations:
                duplicates.add(key)
            annotations[key] = annotation

    accepted = []
    rejected = []
    condition_counts: Counter[str] = Counter()
    for key, (split, source) in sources.items():
        annotation = annotations.get(key)
        failures = {}
        programs = {}
        validations = {}
        if key in duplicates:
            failures["record"] = ["duplicate primary annotation"]
        elif annotation is None:
            failures["record"] = ["missing primary annotation"]
        else:
            scope_id = str(source["effective_scope"]["id"])
            expected = source["state_after"][scope_id]["returned"]
            for condition in ("f1", "f2"):
                try:
                    if annotation.get(f"{condition}_status") != "ok":
                        raise ValueError(
                            f"teacher status is {annotation.get(f'{condition}_status')!r}"
                        )
                    program = _program(annotation, condition)
                    validation = validate_functor_program(
                        program, condition, effective_scope=source["effective_scope"]
                    )
                    programs[condition] = program
                    validations[condition] = validation
                    errors = list(validation["errors"])
                    if not validation["executable"]:
                        errors.append("program is not executable")
                    elif not _equivalent(_returned(validation, scope_id), expected):
                        errors.append(
                            f"answer mismatch: actual={_returned(validation, scope_id)}, expected={expected}"
                        )
                    if errors:
                        failures[condition] = sorted(set(errors))
                    else:
                        condition_counts[condition] += 1
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                    failures[condition] = [str(error)]
        base = {
            "dataset": source["dataset"],
            "source_id": source["source_id"],
            "split": split,
            "record_sha256": source["record_sha256"],
        }
        if failures:
            rejected.append({**base, "failures": failures, "annotation": annotation})
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
                    "annotator": annotation.get("annotator", "local_codex"),
                    "answer_hidden": bool(annotation.get("answer_hidden", True)),
                    "rationale_hidden": bool(annotation.get("rationale_hidden", True)),
                    "prior_asl_hidden": True,
                    "blackboard_state_hidden": True,
                    "fixed_prompt": True,
                    "fixed_icl": True,
                },
            }
        )

    output = Path(output_dir)
    accepted_path = write_jsonl(output / "accepted.jsonl", accepted)
    rejected_path = write_jsonl(output / "rejected.jsonl", rejected)
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
    if set(accepted) != expected_keys:
        missing = sorted(expected_keys - set(accepted))
        extra = sorted(set(accepted) - expected_keys)
        raise ValueError(
            f"functor labels must cover all frozen rows; missing={missing[:5]} extra={extra[:5]}"
        )
    output = Path(output_dir)
    files = {}
    for condition in ("f1", "f2"):
        for split in ("train", "dev"):
            records = []
            for source in splits[split]:
                key = (str(source["dataset"]), str(source["source_id"]))
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
        "source_ids_preserved": True,
        "fixed_prompt_per_condition": True,
        "fixed_icl_per_condition": True,
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
