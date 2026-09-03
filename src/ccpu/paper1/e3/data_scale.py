"""Leakage-audited D1 data-scale preparation for direct ASL training."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl_dataset.remote_analysis import _program_metrics
from ccpu.paper1.asl_matrix.qwen import autonomous_asl_prompt


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["source_id"])


def _diversity_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["semantic_pattern_id"])].append(row)
    for pattern, members in buckets.items():
        members.sort(
            key=lambda row: fingerprint(
                f"{seed}:{pattern}:{row['dataset']}:{row['source_id']}"
            )
        )
    pattern_order = sorted(buckets, key=lambda value: fingerprint(f"{seed}:{value}"))
    ordered = []
    for offset in range(max(map(len, buckets.values()), default=0)):
        ordered.extend(
            buckets[pattern][offset]
            for pattern in pattern_order
            if offset < len(buckets[pattern])
        )
    return ordered


def _dataset_quotas(
    rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], target: int
) -> dict[str, int]:
    available = Counter(str(row["dataset"]) for row in rows)
    reference = Counter(str(row["dataset"]) for row in reference_rows)
    if set(available) != set(reference):
        raise ValueError("D1 and frozen test must cover the same datasets")
    total_reference = sum(reference.values())
    raw = {
        dataset: target * reference[dataset] / total_reference for dataset in available
    }
    quotas = {
        dataset: min(available[dataset], int(raw[dataset])) for dataset in available
    }
    remainder = target - sum(quotas.values())
    for dataset in sorted(
        available,
        key=lambda name: (raw[name] - int(raw[name]), available[name], name),
        reverse=True,
    ):
        if remainder <= 0:
            break
        addition = min(remainder, available[dataset] - quotas[dataset])
        quotas[dataset] += addition
        remainder -= addition
    if remainder:
        raise ValueError(f"D1 dataset quotas cannot satisfy target; missing {remainder}")
    return quotas


def build_d1_f0_data(
    *,
    strict_path: str | Path,
    source_paths: list[str | Path],
    frozen_data_dir: str | Path,
    output_dir: str | Path,
    target: int = 4500,
    epochs: int = 10,
    seed: int = 11,
) -> dict[str, Any]:
    """Freeze unique D1 programs while excluding D0 evaluation leakage."""

    if target < 1 or epochs < 1 or target % epochs:
        raise ValueError("D1 target must be positive and divisible by logical epochs")
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes = {}
    for source_path_value in source_paths:
        source_path = Path(source_path_value)
        source_hashes[str(source_path)] = file_sha256(source_path)
        for source in read_jsonl(source_path):
            key = _key(source)
            if key in sources:
                raise ValueError(f"duplicate D1 source identity: {key}")
            sources[key] = source

    frozen_dir = Path(frozen_data_dir)
    frozen_rows = {
        split: read_jsonl(frozen_dir / f"{split}.jsonl") for split in ("dev", "test")
    }
    frozen_identities = {
        (str(row["dataset"]), str(row["parent_source_id"]))
        for rows in frozen_rows.values()
        for row in rows
    }
    frozen_test_patterns = {
        str(row["semantic_pattern_id"]) for row in frozen_rows["test"]
    }

    strict_rows = read_jsonl(strict_path)
    if len({_key(row) for row in strict_rows}) != len(strict_rows):
        raise ValueError("combined strict corpus contains duplicate source identities")
    eligible = []
    excluded = []
    pre_patterns: set[str] = set()
    invalid_strict = []
    for annotation in strict_rows:
        key = _key(annotation)
        if key not in sources:
            raise ValueError(f"strict annotation has no trusted source: {key}")
        source = sources[key]
        metrics = _program_metrics(
            source, str(annotation["full_asl"]), list(annotation["part_mappings"])
        )
        if not metrics["converted"]:
            invalid_strict.append(
                {"dataset": key[0], "source_id": key[1], "metrics": metrics}
            )
            continue
        semantic_pattern_id = str(metrics["semantic_pattern_id"])
        pre_patterns.add(semantic_pattern_id)
        record = {
            "dataset": key[0],
            "source_id": key[1],
            "question": str(source["question"]),
            "target": str(annotation["full_asl"]),
            "effective_scope": dict(source["effective_scope"]),
            "semantic_pattern_id": semantic_pattern_id,
            "source_record_sha256": source.get("record_sha256"),
            "annotation_sha256": fingerprint(annotation),
            "teacher": annotation.get("teacher"),
            "recovery": annotation.get("recovery"),
        }
        reasons = []
        if key in frozen_identities:
            reasons.append("frozen_dev_or_test_source")
        if semantic_pattern_id in frozen_test_patterns:
            reasons.append("frozen_test_semantic_pattern")
        if reasons:
            excluded.append({**record, "exclusion_reasons": reasons})
        else:
            eligible.append(record)
    if invalid_strict:
        raise ValueError(f"combined strict corpus contains {len(invalid_strict)} invalid rows")
    if len(eligible) < target:
        raise ValueError(f"D1 has only {len(eligible)} eligible rows; target is {target}")

    dataset_quotas = _dataset_quotas(eligible, frozen_rows["test"], target)
    selected = []
    for dataset, quota in sorted(dataset_quotas.items()):
        dataset_rows = [row for row in eligible if row["dataset"] == dataset]
        selected.extend(_diversity_order(dataset_rows, seed)[:quota])
    selected.sort(key=lambda row: fingerprint(f"{seed}:epoch:{row['annotation_sha256']}"))
    selected_identities = {(row["dataset"], row["source_id"]) for row in selected}
    selected_patterns = {str(row["semantic_pattern_id"]) for row in selected}
    if selected_identities & frozen_identities or selected_patterns & frozen_test_patterns:
        raise AssertionError("D1 selection overlaps frozen D0 evaluation")
    rows_per_epoch = target // epochs
    train = [
        {
            "schema_version": "ccpu.paper1.e3_d1_f0_sft.v1",
            "example_id": f"d1-f0-{index:05d}-{fingerprint(row['annotation_sha256'], 10)}",
            "parent_example_id": f"{row['dataset']}:{row['source_id']}",
            "parent_source_id": row["source_id"],
            "semantic_pattern_id": row["semantic_pattern_id"],
            "dataset": row["dataset"],
            "dataset_id": "D1",
            "representation_id": "F0",
            "objective_id": "L0",
            "epoch_view": index // rows_per_epoch,
            "regime": "autonomous",
            "prompt": autonomous_asl_prompt(row["question"]),
            "target": row["target"],
            "has_external_asl": False,
            "external_asl_fraction": 0.0,
            "external_asl_corruption": None,
            "source_fields_visible_to_model": ["question"],
            "teacher_provenance": {
                "teacher": row["teacher"],
                "recovery": row["recovery"],
                "annotation_sha256": row["annotation_sha256"],
                "source_record_sha256": row["source_record_sha256"],
            },
        }
        for index, row in enumerate(selected)
    ]

    output = Path(output_dir)
    eligible_path = write_jsonl(output / "eligible.jsonl", eligible)
    excluded_path = write_jsonl(output / "excluded.jsonl", excluded)
    train_path = write_jsonl(output / "train.jsonl", train)
    reason_counts = Counter(
        reason for row in excluded for reason in row["exclusion_reasons"]
    )
    manifest = {
        "schema_version": "ccpu.paper1.e3_d1_f0_manifest.v1",
        "dataset_id": "D1",
        "representation_id": "F0",
        "objective_id": "L0",
        "seed": seed,
        "selection_policy": (
            "dataset-stratified semantic-pattern round-robin with deterministic hashed order"
        ),
        "dataset_quota_policy": (
            "match frozen-test proportions, cap at available strict rows, then redistribute"
        ),
        "exposure_control": {
            "logical_epochs": epochs,
            "rows_per_epoch": rows_per_epoch,
            "unique_train_rows": target,
            "comparison": "D0 uses 450 unique rows repeated for the same 4500 exposures",
        },
        "counts": {
            "source": len(sources),
            "strict_pre_exclusion": len(strict_rows),
            "strict_pre_exclusion_patterns": len(pre_patterns),
            "excluded": len(excluded),
            "eligible_post_exclusion": len(eligible),
            "eligible_post_exclusion_patterns": len(
                {row["semantic_pattern_id"] for row in eligible}
            ),
            "selected": len(selected),
            "selected_patterns": len(selected_patterns),
            "selected_by_dataset": dict(Counter(row["dataset"] for row in selected)),
            "selected_dataset_quotas": dataset_quotas,
            "exclusion_reasons": dict(reason_counts),
        },
        "leakage_audit": {
            "frozen_dev_test_source_count": len(frozen_identities),
            "frozen_test_pattern_count": len(frozen_test_patterns),
            "selected_source_overlap": [],
            "selected_test_pattern_overlap": [],
            "passed": True,
        },
        "input_sha256": {
            "strict": file_sha256(strict_path),
            "sources": source_hashes,
            "frozen_dev": file_sha256(frozen_dir / "dev.jsonl"),
            "frozen_test": file_sha256(frozen_dir / "test.jsonl"),
        },
        "output_sha256": {
            "eligible": file_sha256(eligible_path),
            "excluded": file_sha256(excluded_path),
            "train": file_sha256(train_path),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
