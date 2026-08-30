"""Automatic chop and external-scope audits."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import read_jsonl, write_json


def audit_chops(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    sample_per_dataset: int = 50,
    seed: int = 23001,
) -> dict[str, Any]:
    root = Path(input_dir)
    generator = random.Random(seed)
    datasets = {}
    all_scopes: set[str] = set()
    hard_errors = []
    for path in sorted(root.glob("*.jsonl")):
        rows = read_jsonl(path)
        failures = []
        teacher_failures = []
        heuristics: Counter[str] = Counter()
        nested = 0
        for row in rows:
            scope = str(row["effective_scope"]["id"])
            if scope in all_scopes:
                hard_errors.append(
                    {
                        "dataset": path.stem,
                        "source_id": row["source_id"],
                        "error": "scope_collision",
                    }
                )
            all_scopes.add(scope)
            nested += int(row["effective_scope"].get("parent") is not None)
            seen: dict[str, set[str]] = {}
            for part in row["parts"]:
                heuristics[str(part["heuristic"])] += 1
                text = str(part["text"])
                reasons = []
                if not text.strip():
                    reasons.append("empty_part")
                kind = str(part["kind"])
                if text in seen.setdefault(kind, set()):
                    reasons.append("duplicate_part")
                seen[kind].add(text)
                if (
                    part["kind"] == "question_clause"
                    and part["start"] is not None
                    and row["question"][part["start"] : part["end"]] != text
                ):
                    reasons.append("offset_mismatch")
                if "####" in text:
                    reasons.append("answer_marker_leak")
                reasons.extend(part.get("warnings", []))
                if reasons:
                    failure = {
                        "source_id": row["source_id"],
                        "part_id": part["part_id"],
                        "reasons": sorted(set(reasons)),
                        "text": text,
                    }
                    failures.append(failure)
                    if row["metadata"].get("arithmetic_compatible") and part.get(
                        "teacher_input_default", True
                    ):
                        teacher_failures.append(failure)
        sampled = generator.sample(failures, min(sample_per_dataset, len(failures)))
        datasets[path.stem] = {
            "record_count": len(rows),
            "part_count": sum(len(row["parts"]) for row in rows),
            "failure_count": len(failures),
            "teacher_failure_count": len(teacher_failures),
            "failure_sample": sampled,
            "heuristics": dict(sorted(heuristics.items())),
            "root_scope_count": len({row["effective_scope"]["id"] for row in rows}),
            "nested_scope_count": nested,
            "scope_source_distribution": dict(
                Counter(row["effective_scope"]["source"] for row in rows)
            ),
        }
    report = {
        "schema_version": "ccpu.dsl_dataset.chop_audit.v1",
        "seed": seed,
        "sample_per_dataset": sample_per_dataset,
        "datasets": datasets,
        "hard_errors": hard_errors,
        "scale_teacher_generation": not hard_errors
        and all(item["teacher_failure_count"] == 0 for item in datasets.values()),
    }
    write_json(output_path, report)
    summary = {
        **report,
        "datasets": {
            dataset: {key: value for key, value in result.items() if key != "failure_sample"}
            for dataset, result in datasets.items()
        },
    }
    write_json(Path(output_path).with_name(f"{Path(output_path).stem}_summary.json"), summary)
    return report
