"""Deterministic public-dataset normalization and mining."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import environment_manifest, file_sha256, write_json, write_jsonl

from .chop import chop_example
from .loaders import load_dataset

SUPPORTED_DATASETS = {
    "gsm8k",
    "gsm_symbolic",
    "gsm_plus",
    "gsm_ranges",
    "mawps",
    "tatqa",
    "clutrr",
    "ruletaker",
    "proofwriter",
}


def mine_datasets(
    sources: dict[str, str | Path],
    output_dir: str | Path,
    *,
    split: str = "train",
    source_splits: dict[str, str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifests = {}
    all_ids: set[str] = set()
    for dataset, source in sorted(sources.items()):
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        source_path = Path(source)
        dataset_split = (source_splits or {}).get(dataset, split)
        rows = load_dataset(dataset, source_path, dataset_split)
        for row in rows:
            row["parts"] = chop_example(row)
            row["source_sha256"] = file_sha256(source_path)
            row["record_sha256"] = hashlib.sha256(
                f"{dataset}\0{row['split']}\0{row['source_id']}\0{row['question']}".encode()
            ).hexdigest()
            global_id = f"{dataset}:{row['split']}:{row['source_id']}"
            if global_id in all_ids:
                raise ValueError(f"source ID collision: {global_id}")
            all_ids.add(global_id)
        path = write_jsonl(output / f"{dataset}.jsonl", rows)
        manifests[dataset] = {
            "source_file": source_path.name,
            "source_location": "verified_local_cache",
            "source_sha256": file_sha256(source_path),
            "declared_split": dataset_split,
            "output_sha256": file_sha256(path),
            "record_count": len(rows),
            "arithmetic_compatible_count": sum(
                bool(row["metadata"].get("arithmetic_compatible")) for row in rows
            ),
            "part_count": sum(len(row["parts"]) for row in rows),
        }
    manifest = {
        "schema_version": "ccpu.dsl_dataset.raw_manifest.v1",
        "asl_version": "asl-core-v0",
        "profile": "asl-arith-v0",
        "datasets": manifests,
        "scope_policy": "one dataset record per root benchmark_case scope",
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
