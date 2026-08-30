"""Deterministic seed selection for local-agent and remote teachers."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl

_RELATION_PATTERNS = {
    "aggregation": r"\b(total|altogether|combined|sum|average|mean|minimum|maximum)\b",
    "comparison": r"\b(more|less|fewer|greater|smaller|largest|smallest|most|least)\b",
    "equal_allocation": r"\b(each|equally|shared|split|among|per person|apiece)\b",
    "multiple_entities": r"\b(and|between|together|combined|respectively)\b",
    "percentage": r"(?:%|\bpercent(?:age)?\b|\bcommission\b|\bdiscount\b)",
    "rate_or_per_unit": r"\b(per|each|every|rate|speed|mph|hourly)\b",
    "ratio_or_fraction": r"\b(ratio|fraction|half|third|quarter|twice|double|triple|out of)\b",
    "relative_quantity": r"\b(more than|less than|fewer than|as many|times as)\b",
    "remaining_or_difference": r"\b(remain|remaining|rest|left|difference|gave|lost|spent)\b",
    "temporal_or_duration": r"\b(year|month|week|day|hour|minute|before|after|later|ago|now)\b",
}


def surface_relation_classes(row: dict[str, Any]) -> set[str]:
    """Return answer-free relation cues for diversity selection before ASL exists."""

    question = str(row["question"]).casefold()
    classes = {name for name, pattern in _RELATION_PATTERNS.items() if re.search(pattern, question)}
    clauses = sum(part.get("teacher_input_default", True) for part in row.get("parts", []))
    numbers = len(re.findall(r"\d+(?:\.\d+)?", question))
    if clauses >= 3 or numbers >= 4:
        classes.add("chained_relations")
    if clauses >= 4 or (clauses >= 3 and numbers >= 3):
        classes.add("nested_dependencies")
    return classes or {"single_relation"}


def _difficulty(row: dict[str, Any]) -> str:
    question = str(row["question"]).casefold()
    cues = len(
        re.findall(r"\b(?:each|times|percent|more|less|total|average|remaining)\b", question)
    )
    numbers = len(re.findall(r"\d+(?:\.\d+)?", question))
    score = cues + max(0, numbers - 2)
    return "low" if score <= 1 else "medium" if score <= 3 else "high"


def select_seed(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_examples: int,
    seed: int = 23003,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(input_path) if row["metadata"].get("arithmetic_compatible")]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row = {**row, "selection_difficulty": _difficulty(row)}
        groups[row["selection_difficulty"]].append(row)
    for members in groups.values():
        members.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['record_sha256']}".encode("ascii")
            ).hexdigest()
        )
    selected = []
    while len(selected) < min(max_examples, len(rows)):
        progressed = False
        for label in ("low", "medium", "high"):
            if groups[label] and len(selected) < max_examples:
                selected.append(groups[label].pop(0))
                progressed = True
        if not progressed:
            break
    output = write_jsonl(output_path, selected)
    ledger = write_jsonl(
        Path(output_path).with_name(f"{Path(output_path).stem}_ledger.jsonl"),
        (
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "source_id": row["source_id"],
                "record_sha256": row["record_sha256"],
                "effective_scope": row["effective_scope"],
                "selection_difficulty": row["selection_difficulty"],
            }
            for row in selected
        ),
    )
    manifest = {
        "schema_version": "ccpu.dsl_dataset.seed_manifest.v1",
        "seed": seed,
        "input_sha256": file_sha256(input_path),
        "output_sha256": file_sha256(output),
        "ledger_sha256": file_sha256(ledger),
        "record_count": len(selected),
        "difficulty_counts": dict(Counter(row["selection_difficulty"] for row in selected)),
    }
    write_json(Path(output_path).with_suffix(".manifest.json"), manifest)
    return manifest


def select_diverse_seed(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    dataset_targets: dict[str, int],
    exclude_paths: list[str | Path] | None = None,
    seed: int = 53011,
) -> dict[str, Any]:
    """Select a deterministic relation-diverse expansion with explicit source exclusions."""

    excluded = set()
    exclusion_hashes = {}
    for path in map(Path, exclude_paths or []):
        exclusion_hashes[str(path)] = file_sha256(path)
        excluded.update((str(row["dataset"]), str(row["source_id"])) for row in read_jsonl(path))
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_hashes = {}
    for path in map(Path, input_paths):
        input_hashes[str(path)] = file_sha256(path)
        for row in read_jsonl(path):
            dataset = str(row["dataset"])
            key = (dataset, str(row["source_id"]))
            if (
                dataset not in dataset_targets
                or key in excluded
                or not row["metadata"].get("arithmetic_compatible")
            ):
                continue
            candidates[dataset].append(
                {
                    **row,
                    "selection_difficulty": _difficulty(row),
                    "selection_relation_classes": sorted(surface_relation_classes(row)),
                }
            )

    selected = []
    relation_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    for dataset, target in dataset_targets.items():
        available = candidates[dataset]
        if len(available) < target:
            raise ValueError(f"only {len(available)} eligible {dataset} rows for target {target}")
        remaining = list(available)
        dataset_selected = []
        while len(dataset_selected) < target:
            difficulty_floor = min(
                difficulty_counts.get(label, 0) for label in ("low", "medium", "high")
            )

            def rank(
                row: dict[str, Any], difficulty_floor: int = difficulty_floor
            ) -> tuple[float, str]:
                diversity = sum(
                    1.0 / (1 + relation_counts[relation])
                    for relation in row["selection_relation_classes"]
                )
                difficulty_bonus = (
                    0.25
                    if difficulty_counts[row["selection_difficulty"]] == difficulty_floor
                    else 0.0
                )
                tie = hashlib.sha256(f"{seed}:{row['record_sha256']}".encode("ascii")).hexdigest()
                return (diversity + difficulty_bonus, tie)

            choice = max(remaining, key=rank)
            remaining.remove(choice)
            dataset_selected.append(choice)
            relation_counts.update(choice["selection_relation_classes"])
            difficulty_counts.update([choice["selection_difficulty"]])
        selected.extend(dataset_selected)

    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:output:{row['record_sha256']}".encode("ascii")
        ).hexdigest()
    )
    output = write_jsonl(output_path, selected)
    ledger = write_jsonl(
        Path(output_path).with_name(f"{Path(output_path).stem}_ledger.jsonl"),
        (
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "source_id": row["source_id"],
                "record_sha256": row["record_sha256"],
                "selection_difficulty": row["selection_difficulty"],
                "selection_relation_classes": row["selection_relation_classes"],
            }
            for row in selected
        ),
    )
    manifest = {
        "schema_version": "ccpu.dsl_dataset.diverse_seed_manifest.v1",
        "seed": seed,
        "input_sha256": input_hashes,
        "exclusion_sha256": exclusion_hashes,
        "excluded_source_count": len(excluded),
        "dataset_targets": dataset_targets,
        "record_count": len(selected),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "relation_class_counts": dict(sorted(relation_counts.items())),
        "output_sha256": file_sha256(output),
        "ledger_sha256": file_sha256(ledger),
        "post_annotation_pattern_filter_required": True,
    }
    write_json(Path(output_path).with_suffix(".manifest.json"), manifest)
    return manifest
