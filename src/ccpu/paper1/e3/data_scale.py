"""Leakage-audited D1 data-scale preparation for direct ASL training."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    file_sha256,
    fingerprint,
    read_json,
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
    datasets: tuple[str, ...] | None = None,
    dataset_id: str = "D1",
) -> dict[str, Any]:
    """Freeze unique F0 programs while excluding frozen evaluation leakage."""

    if target < 1 or epochs < 1 or target % epochs:
        raise ValueError("D1 target must be positive and divisible by logical epochs")
    allowed_datasets = set(datasets) if datasets is not None else None
    if allowed_datasets is not None and (not allowed_datasets or "" in allowed_datasets):
        raise ValueError("dataset scope must contain non-empty dataset names")
    if not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    source_hashes = {}
    source_input_count = 0
    source_scope_rejected = 0
    for source_path_value in source_paths:
        source_path = Path(source_path_value)
        source_hashes[str(source_path)] = file_sha256(source_path)
        for source in read_jsonl(source_path):
            source_input_count += 1
            if (
                allowed_datasets is not None
                and str(source["dataset"]) not in allowed_datasets
            ):
                source_scope_rejected += 1
                continue
            key = _key(source)
            if key in sources:
                raise ValueError(f"duplicate D1 source identity: {key}")
            sources[key] = source

    frozen_dir = Path(frozen_data_dir)
    frozen_rows = {}
    for split in ("dev", "test"):
        rows = read_jsonl(frozen_dir / f"{split}.jsonl")
        frozen_rows[split] = [
            row
            for row in rows
            if allowed_datasets is None
            or str(row["dataset"]) in allowed_datasets
        ]
    frozen_identities = {
        (str(row["dataset"]), str(row["parent_source_id"]))
        for rows in frozen_rows.values()
        for row in rows
    }
    frozen_test_patterns = {
        str(row["semantic_pattern_id"]) for row in frozen_rows["test"]
    }

    strict_input_rows = read_jsonl(strict_path)
    strict_rows = [
        row
        for row in strict_input_rows
        if allowed_datasets is None
        or str(row["dataset"]) in allowed_datasets
    ]
    strict_scope_rejected = len(strict_input_rows) - len(strict_rows)
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
    selected_dataset_names = {str(row["dataset"]) for row in selected}
    if (
        allowed_datasets is not None
        and not selected_dataset_names <= allowed_datasets
    ):
        raise AssertionError("F0 selection escaped its registered dataset scope")
    rows_per_epoch = target // epochs
    legacy_d1 = dataset_id == "D1" and allowed_datasets is None
    id_prefix = (
        "d1-f0"
        if legacy_d1
        else f"{dataset_id.lower().replace('_', '-')}-f0"
    )
    train = [
        {
            "schema_version": (
                "ccpu.paper1.e3_d1_f0_sft.v1"
                if legacy_d1
                else "ccpu.paper1.e3_f0_scale_sft.v1"
            ),
            "example_id": (
                f"{id_prefix}-{index:05d}-{fingerprint(row['annotation_sha256'], 10)}"
            ),
            "parent_example_id": f"{row['dataset']}:{row['source_id']}",
            "parent_source_id": row["source_id"],
            "semantic_pattern_id": row["semantic_pattern_id"],
            "dataset": row["dataset"],
            "dataset_id": dataset_id,
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
        "schema_version": (
            "ccpu.paper1.e3_d1_f0_manifest.v1"
            if legacy_d1
            else "ccpu.paper1.e3_f0_scale_manifest.v1"
        ),
        "dataset_id": dataset_id,
        "dataset_scope": sorted(allowed_datasets or selected_dataset_names),
        "representation_id": "F0",
        "objective_id": "L0",
        "seed": seed,
        "selection_policy": (
            "semantic-pattern round-robin with deterministic hashed order"
            if len(selected_dataset_names) == 1
            else "dataset-stratified semantic-pattern round-robin with deterministic hashed order"
        ),
        "dataset_quota_policy": (
            "single registered dataset"
            if len(selected_dataset_names) == 1
            else "match frozen-test proportions, cap at available strict rows, then redistribute"
        ),
        "exposure_control": {
            "logical_epochs": epochs,
            "rows_per_epoch": rows_per_epoch,
            "unique_train_rows": target,
            "comparison": (
                "D0 uses 450 unique rows repeated for the same 4500 exposures"
                if target == 4500
                else "data-scale preparation; exposure policy is registered by the run config"
            ),
        },
        "counts": {
            "source": len(sources),
            "source_input": source_input_count,
            "source_scope_rejected": source_scope_rejected,
            "strict_pre_exclusion": len(strict_rows),
            "strict_input": len(strict_input_rows),
            "strict_scope_rejected": strict_scope_rejected,
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
            "dataset_scope_enforced": (
                allowed_datasets is None
                or selected_dataset_names <= allowed_datasets
            ),
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


def build_gsm8k_f0_data(
    *,
    strict_path: str | Path,
    source_path: str | Path,
    frozen_data_dir: str | Path,
    output_dir: str | Path,
    target: int = 4500,
    epochs: int = 10,
    seed: int = 11,
) -> dict[str, Any]:
    """Freeze a GSM8K-only F0 corpus for post-D1 Paper 1 experiments."""

    return build_d1_f0_data(
        strict_path=strict_path,
        source_paths=[source_path],
        frozen_data_dir=frozen_data_dir,
        output_dir=output_dir,
        target=target,
        epochs=epochs,
        seed=seed,
        datasets=("gsm8k",),
        dataset_id="G1_GSM8K",
    )


def freeze_gsm8k_eval_views(
    *,
    train_path: str | Path,
    dev_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze GSM8K-only dev/test views and audit them against G1 training."""

    inputs = {
        "train": Path(train_path),
        "dev": Path(dev_path),
        "test": Path(test_path),
    }
    raw_rows = {name: read_jsonl(path) for name, path in inputs.items()}
    rows = {
        name: [row for row in split_rows if str(row["dataset"]) == "gsm8k"]
        for name, split_rows in raw_rows.items()
    }
    for name, split_rows in rows.items():
        identities = [str(row["parent_source_id"]) for row in split_rows]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate GSM8K source identity in {name}")
        if name != "train" and not split_rows:
            raise ValueError(f"no GSM8K rows in {name}")
    train_ids = {str(row["parent_source_id"]) for row in rows["train"]}
    dev_ids = {str(row["parent_source_id"]) for row in rows["dev"]}
    test_ids = {str(row["parent_source_id"]) for row in rows["test"]}
    train_patterns = {str(row["semantic_pattern_id"]) for row in rows["train"]}
    test_patterns = {str(row["semantic_pattern_id"]) for row in rows["test"]}
    overlaps = {
        "train_dev_source": sorted(train_ids & dev_ids),
        "train_test_source": sorted(train_ids & test_ids),
        "dev_test_source": sorted(dev_ids & test_ids),
        "train_test_pattern": sorted(train_patterns & test_patterns),
    }
    if any(overlaps.values()):
        raise ValueError(f"GSM8K split leakage detected: {overlaps}")

    output = Path(output_dir)
    output_paths = {
        "dev": write_jsonl(output / "dev.jsonl", rows["dev"]),
        "test": write_jsonl(output / "test.jsonl", rows["test"]),
    }
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_eval_freeze.v1",
        "dataset_scope": ["gsm8k"],
        "roles": {
            "dev": "checkpoint selection only",
            "test": "historical paired evaluation only",
        },
        "counts": {
            name: {
                "input": len(raw_rows[name]),
                "selected_gsm8k": len(rows[name]),
                "scope_rejected": len(raw_rows[name]) - len(rows[name]),
            }
            for name in ("train", "dev", "test")
        },
        "leakage_audit": {**overlaps, "passed": True},
        "input_sha256": {name: file_sha256(path) for name, path in inputs.items()},
        "output_sha256": {
            name: file_sha256(path) for name, path in output_paths.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def build_gsm8k_exposure_scale(
    *,
    parent_dir: str | Path,
    output_dir: str | Path,
    unique_rows: int,
    exposures: int = 4500,
    epochs: int = 10,
    seed: int = 11,
) -> dict[str, Any]:
    """Build a balanced GSM8K subset with a fixed number of row exposures."""

    if unique_rows < 1 or exposures < unique_rows:
        raise ValueError("exposures must be at least the positive unique-row count")
    if epochs < 1 or exposures % epochs:
        raise ValueError("exposures must divide evenly across logical epochs")
    parent = Path(parent_dir)
    parent_manifest = read_json(parent / "manifest.json")
    if parent_manifest.get("dataset_scope") != ["gsm8k"]:
        raise ValueError("scale parent must be a GSM8K-only freeze")
    if not parent_manifest.get("leakage_audit", {}).get("passed"):
        raise ValueError("scale parent did not pass its leakage audit")
    eligible_path = parent / "eligible.jsonl"
    eligible = read_jsonl(eligible_path)
    if any(str(row["dataset"]) != "gsm8k" for row in eligible):
        raise ValueError("scale parent contains a non-GSM8K row")
    if len(eligible) < unique_rows:
        raise ValueError(
            f"scale parent has only {len(eligible)} rows; requested {unique_rows}"
        )
    selected = _diversity_order(eligible, seed)[:unique_rows]
    selected_ids = {str(row["source_id"]) for row in selected}
    if len(selected_ids) != unique_rows:
        raise ValueError("scale selection contains duplicate source identities")

    stream = []
    cycle = 0
    while len(stream) < exposures:
        cycle_rows = sorted(
            selected,
            key=lambda row: fingerprint(
                f"{seed}:scale-cycle:{cycle}:{row['annotation_sha256']}"
            ),
        )
        stream.extend(cycle_rows[: exposures - len(stream)])
        cycle += 1
    rows_per_epoch = exposures // epochs
    dataset_id = f"G1_GSM8K_U{unique_rows}_E{exposures}"
    train = [
        {
            "schema_version": "ccpu.paper1.e3_f0_scale_sft.v1",
            "example_id": (
                f"g1-gsm8k-u{unique_rows}-e{exposures}-f0-{index:05d}-"
                f"{fingerprint(row['annotation_sha256'], 10)}"
            ),
            "parent_example_id": f"gsm8k:{row['source_id']}",
            "parent_source_id": row["source_id"],
            "semantic_pattern_id": row["semantic_pattern_id"],
            "dataset": "gsm8k",
            "dataset_id": dataset_id,
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
        for index, row in enumerate(stream)
    ]
    reuse = Counter(str(row["source_id"]) for row in stream)
    output = Path(output_dir)
    train_path = write_jsonl(output / "train.jsonl", train)
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_exposure_scale.v1",
        "dataset_id": dataset_id,
        "dataset_scope": ["gsm8k"],
        "representation_id": "F0",
        "objective_id": "L0",
        "seed": seed,
        "selection_policy": "semantic-pattern round-robin from frozen G1 eligible pool",
        "exposure_policy": "deterministically shuffled balanced cycles",
        "counts": {
            "eligible_parent_rows": len(eligible),
            "unique_train_rows": unique_rows,
            "unique_semantic_patterns": len(
                {str(row["semantic_pattern_id"]) for row in selected}
            ),
            "exposures": exposures,
            "logical_epochs": epochs,
            "rows_per_epoch": rows_per_epoch,
            "minimum_source_reuse": min(reuse.values()),
            "maximum_source_reuse": max(reuse.values()),
        },
        "leakage_audit": {
            "inherited_parent_manifest": str(parent / "manifest.json"),
            "parent_passed": True,
            "dataset_scope_enforced": True,
            "unique_source_count_verified": len(selected_ids) == unique_rows,
            "passed": True,
        },
        "input_sha256": {
            "parent_manifest": file_sha256(parent / "manifest.json"),
            "eligible": file_sha256(eligible_path),
        },
        "output_sha256": {"train": file_sha256(train_path)},
    }
    write_json(output / "manifest.json", manifest)
    return manifest
