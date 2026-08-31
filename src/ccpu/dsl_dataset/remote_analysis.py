"""Quality and coverage comparison for remote-teacher semantic programs."""

from __future__ import annotations

import hashlib
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl
from ccpu.paper1.asl_pilot_data import pattern_id

from .semantic import _expected_decimal, _returned, semantic_lint


def _program_metrics(
    row: dict[str, Any], asl: str, mappings: list[dict[str, Any]]
) -> dict[str, Any]:
    validation = validate_asl(asl, effective_scope=row["effective_scope"])
    answer_correct = bool(row.get("validation", {}).get("final_answer_verified", False))
    if validation["execution_verified"] and "answer" in row:
        try:
            answer_correct = abs(
                _returned(validation["execution"], str(row["effective_scope"]["id"]))
                - _expected_decimal(row["answer"])
            ) <= Decimal("0.011")
        except (KeyError, TypeError, ValueError, ArithmeticError):
            answer_correct = False
    lint_errors = semantic_lint(mappings)
    statements = [line for line in asl.splitlines() if line.strip()]
    program = {
        **row,
        "asl": asl,
        "part_mappings": mappings,
        "ccir": validation.get("ccir", {"operations": []}),
    }
    return {
        "syntax_verified": bool(validation["syntax_verified"]),
        "lower_verified": bool(validation["lower_verified"]),
        "type_verified": bool(validation["type_verified"]),
        "execution_verified": bool(validation["execution_verified"]),
        "semantic_lint_valid": not lint_errors,
        "answer_correct": answer_correct,
        "converted": bool(validation["execution_verified"] and not lint_errors and answer_correct),
        "statement_count": len(statements),
        "part_count": len(mappings),
        "semantic_pattern_id": pattern_id(program) if validation["lower_verified"] else None,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"program_count": 0}
    boolean_names = (
        "syntax_verified",
        "lower_verified",
        "type_verified",
        "execution_verified",
        "semantic_lint_valid",
        "answer_correct",
        "converted",
    )
    return {
        "program_count": len(rows),
        "rates": {name: sum(bool(row[name]) for row in rows) / len(rows) for name in boolean_names},
        "mean_statement_count": mean(row["statement_count"] for row in rows),
        "mean_part_count": mean(row["part_count"] for row in rows),
        "unique_semantic_patterns": len(
            {row["semantic_pattern_id"] for row in rows if row["semantic_pattern_id"]}
        ),
    }


def analyze_remote_programs(
    *,
    source_paths: list[str | Path],
    remote_dirs: list[str | Path],
    baseline_paths: list[str | Path],
    output_dir: str | Path,
    sample_count: int = 5,
) -> dict[str, Any]:
    sources = {}
    source_hashes = {}
    for path in map(Path, source_paths):
        source_hashes[str(path)] = file_sha256(path)
        for row in read_jsonl(path):
            sources[(str(row["dataset"]), str(row["source_id"]))] = row

    remote_rows = []
    remote_hashes = {}
    attempt_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    failures_by_dataset: Counter[str] = Counter()
    for directory in map(Path, remote_dirs):
        annotation_path = directory / "annotations.jsonl"
        attempt_path = directory / "attempts.jsonl"
        failure_path = directory / "failures.jsonl"
        for path in (annotation_path, attempt_path, failure_path):
            if path.exists():
                remote_hashes[str(path)] = file_sha256(path)
        for attempt in read_jsonl(attempt_path) if attempt_path.exists() else []:
            attempt_counts[str(attempt["outcome"])] += 1
        for failure in read_jsonl(failure_path) if failure_path.exists() else []:
            failure_counts[str(failure.get("final_error", "unknown")).split(":", 1)[0]] += 1
            failures_by_dataset[str(failure["dataset"])] += 1
        for annotation in read_jsonl(annotation_path) if annotation_path.exists() else []:
            key = (str(annotation["dataset"]), str(annotation["source_id"]))
            source = sources[key]
            metrics = _program_metrics(
                source, str(annotation["full_asl"]), annotation["part_mappings"]
            )
            remote_rows.append({**metrics, "source": source, "annotation": annotation})

    baseline_rows = []
    baseline_hashes = {}
    for path in map(Path, baseline_paths):
        baseline_hashes[str(path)] = file_sha256(path)
        for row in read_jsonl(path):
            metrics = _program_metrics(row, str(row["asl"]), row["part_mappings"])
            baseline_rows.append({**metrics, "source": row, "annotation": row})

    by_dataset = {}
    for dataset in sorted({str(row["dataset"]) for row in sources.values()}):
        input_count = sum(row["dataset"] == dataset for row in sources.values())
        generated = [row for row in remote_rows if row["source"]["dataset"] == dataset]
        converted = sum(row["converted"] for row in generated)
        failed = failures_by_dataset[dataset]
        processed = len(generated) + failed
        by_dataset[dataset] = {
            "input_program_count": input_count,
            "generated_program_count": len(generated),
            "terminal_failure_count": failed,
            "processed_program_count": processed,
            "remaining_program_count": input_count - processed,
            "converted_program_count": converted,
            "processed_proportion": processed / input_count if input_count else 0.0,
            "generated_proportion": len(generated) / input_count if input_count else 0.0,
            "converted_proportion_full_input": converted / input_count if input_count else 0.0,
            "converted_proportion_processed": converted / processed if processed else 0.0,
            "converted_proportion_generated": converted / len(generated) if generated else 0.0,
        }

    eligible_remote = [row for row in remote_rows if row["converted"]]
    selected = sorted(
        eligible_remote,
        key=lambda row: hashlib.sha256(
            f"remote-sample:{row['source']['dataset']}:{row['source']['source_id']}".encode()
        ).hexdigest(),
    )[:sample_count]
    samples = []
    sample_index = []
    for remote in selected:
        candidates = [
            row for row in baseline_rows if row["source"]["dataset"] == remote["source"]["dataset"]
        ]
        baseline = min(
            candidates,
            key=lambda row: (
                abs(row["statement_count"] - remote["statement_count"]),
                str(row["source"]["source_id"]),
            ),
        )
        samples.append(
            {
                "dataset": remote["source"]["dataset"],
                "remote": {
                    "source_id": remote["source"]["source_id"],
                    "question": remote["source"]["question"],
                    "asl": remote["annotation"]["full_asl"],
                    "model": remote["annotation"]["teacher"]["model"],
                    "metrics": {
                        key: value for key, value in remote.items() if isinstance(value, bool)
                    },
                },
                "baseline": {
                    "source_id": baseline["source"]["source_id"],
                    "question": baseline["source"]["question"],
                    "asl": baseline["annotation"]["asl"],
                    "quality_grade": baseline["annotation"].get("quality_grade"),
                    "metrics": {
                        key: value for key, value in baseline.items() if isinstance(value, bool)
                    },
                },
            }
        )
        sample_index.append(
            {
                "dataset": remote["source"]["dataset"],
                "remote_source_id": remote["source"]["source_id"],
                "baseline_source_id": baseline["source"]["source_id"],
            }
        )

    output = Path(output_dir)
    sample_path = write_jsonl(output / "quality_samples.jsonl", samples)
    report = {
        "schema_version": "ccpu.dsl_dataset.remote_program_analysis.v1",
        "input_program_count": len(sources),
        "remote": _aggregate(remote_rows),
        "baseline": _aggregate(baseline_rows),
        "by_dataset": by_dataset,
        "attempt_outcomes": dict(sorted(attempt_counts.items())),
        "terminal_failure_types": dict(sorted(failure_counts.items())),
        "quality_sample_index": sample_index,
        "quality_samples_sha256": file_sha256(sample_path),
        "input_sha256": {
            "sources": source_hashes,
            "remote": remote_hashes,
            "baseline": baseline_hashes,
        },
        "claim_boundary": "remote outputs are deterministic-execution checked but not manually reviewed semantic gold",
    }
    write_json(output / "analysis.json", report)
    return report
