"""Finalize relation-diverse ASL expansion rows without frozen-evaluation leakage."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.paper1.asl_pilot_data import pattern_id

from .select import surface_relation_classes


def finalize_asl_expansion(
    candidate_accepted_path: str | Path,
    existing_accepted_path: str | Path,
    frozen_ledger_path: str | Path,
    output_dir: str | Path,
    *,
    target: int = 350,
    seed: int = 53011,
) -> dict[str, Any]:
    """Choose exact-size new training data after semantic-family leakage checks."""

    candidates = read_jsonl(candidate_accepted_path)
    existing = read_jsonl(existing_accepted_path)
    ledger = read_jsonl(frozen_ledger_path)
    existing_ids = {(str(row["dataset"]), str(row["source_id"])) for row in existing}
    frozen_eval_patterns = {
        str(row["semantic_pattern_id"]) for row in ledger if row["split"] in {"dev", "test"}
    }
    frozen_train_patterns = {
        str(row["semantic_pattern_id"]) for row in ledger if row["split"] == "train"
    }
    eligible = []
    quarantined = []
    for row in candidates:
        key = (str(row["dataset"]), str(row["source_id"]))
        semantic_pattern_id = pattern_id(row)
        enriched = {
            **row,
            "semantic_pattern_id": semantic_pattern_id,
            "relation_classes": sorted(surface_relation_classes(row)),
        }
        reasons = []
        if key in existing_ids:
            reasons.append("existing_source_id")
        if semantic_pattern_id in frozen_eval_patterns:
            reasons.append("frozen_eval_semantic_pattern")
        if reasons:
            quarantined.append({**enriched, "quarantine_reasons": reasons})
        else:
            eligible.append(enriched)
    if len(eligible) < target:
        raise ValueError(
            f"only {len(eligible)} eligible accepted rows remain after leakage filters; target={target}"
        )

    selected = []
    remaining = list(eligible)
    pattern_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    target_datasets = Counter(row["dataset"] for row in eligible)
    while len(selected) < target:

        def rank(row: dict[str, Any]) -> tuple[float, str]:
            pattern_bonus = 4.0 if not pattern_counts[row["semantic_pattern_id"]] else 0.0
            relation_bonus = sum(
                1.0 / (1 + relation_counts[label]) for label in row["relation_classes"]
            )
            dataset_bonus = target_datasets[row["dataset"]] / (1 + dataset_counts[row["dataset"]])
            tie = hashlib.sha256(f"{seed}:{row['record_sha256']}".encode("ascii")).hexdigest()
            return pattern_bonus + relation_bonus + dataset_bonus / 1000, tie

        choice = max(remaining, key=rank)
        remaining.remove(choice)
        selected.append(choice)
        pattern_counts.update([choice["semantic_pattern_id"]])
        relation_counts.update(choice["relation_classes"])
        dataset_counts.update([choice["dataset"]])
    quarantined.extend({**row, "quarantine_reasons": ["reserve_not_selected"]} for row in remaining)

    output = Path(output_dir)
    selected_path = write_jsonl(output / "expansion_train.jsonl", selected)
    combined_path = write_jsonl(output / "accepted_500.jsonl", [*existing, *selected])
    quarantine_path = write_jsonl(output / "quarantine.jsonl", quarantined)
    selected_ledger_path = write_jsonl(
        output / "expansion_ledger.jsonl",
        (
            {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "record_sha256": row["record_sha256"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "relation_classes": row["relation_classes"],
                "repair_round": row.get("provenance", {}).get("repair_round", 0),
            }
            for row in selected
        ),
    )
    quarantine_ledger_path = write_jsonl(
        output / "quarantine_ledger.jsonl",
        (
            {
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "record_sha256": row["record_sha256"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "quarantine_reasons": row["quarantine_reasons"],
            }
            for row in quarantined
        ),
    )
    selected_patterns = set(pattern_counts)
    existing_train_overlap = selected_patterns & frozen_train_patterns
    manifest = {
        "schema_version": "ccpu.dsl_dataset.asl_expansion_freeze.v1",
        "seed": seed,
        "target": target,
        "candidate_accepted_count": len(candidates),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "combined_count": len(existing) + len(selected),
        "unique_selected_patterns": len(pattern_counts),
        "existing_train_pattern_count": len(frozen_train_patterns),
        "existing_train_pattern_overlap": len(existing_train_overlap),
        "combined_train_row_count": sum(row["split"] == "train" for row in ledger) + len(selected),
        "combined_train_pattern_count": len(frozen_train_patterns | selected_patterns),
        "pattern_multiplicities": dict(sorted(Counter(pattern_counts.values()).items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "relation_class_counts": dict(sorted(relation_counts.items())),
        "quarantine_reason_counts": dict(
            sorted(
                Counter(
                    reason for row in quarantined for reason in row["quarantine_reasons"]
                ).items()
            )
        ),
        "frozen_eval_pattern_count": len(frozen_eval_patterns),
        "frozen_eval_pattern_overlap": [],
        "input_sha256": {
            "candidate_accepted": file_sha256(candidate_accepted_path),
            "existing_accepted": file_sha256(existing_accepted_path),
            "frozen_ledger": file_sha256(frozen_ledger_path),
        },
        "output_sha256": {
            "expansion_train": file_sha256(selected_path),
            "accepted_500": file_sha256(combined_path),
            "quarantine": file_sha256(quarantine_path),
            "expansion_ledger": file_sha256(selected_ledger_path),
            "quarantine_ledger": file_sha256(quarantine_ledger_path),
        },
        "policy": "exclude existing source IDs and frozen dev/test semantic patterns; maximize new signatures",
    }
    write_json(output / "freeze_manifest.json", manifest)
    return manifest
