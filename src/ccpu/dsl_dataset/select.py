"""Deterministic seed selection for local-agent and remote teachers."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl


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
