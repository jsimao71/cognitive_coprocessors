"""Pinned public-benchmark loading and redistribution-safe selection manifests."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import canonical_json, file_sha256, fingerprint, read_json, write_json, write_jsonl


@dataclass(frozen=True)
class PublicSource:
    benchmark: str
    engine: str
    repo_id: str
    revision: str
    subset: str
    split: str
    local_dir: str
    file: str
    file_sha256: str
    expected_rows: int
    max_rows: int
    license: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PublicSource:
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})

    def source_path(self, cache_root: str | Path) -> Path:
        return Path(cache_root) / self.local_dir / Path(self.file)

    def provenance(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "engine": self.engine,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "subset": self.subset,
            "split": self.split,
            "file": self.file,
            "file_sha256": self.file_sha256,
            "expected_rows": self.expected_rows,
            "license": self.license,
        }


def load_config(path: str | Path) -> tuple[int, list[PublicSource], dict[str, Any]]:
    raw = read_json(path)
    if raw.get("schema_version") != "ccpu.public_benchmarks.config.v1":
        raise ValueError("unsupported public benchmark config schema")
    sources = [PublicSource.from_dict(item) for item in raw["sources"]]
    benchmarks = [source.benchmark for source in sources]
    if len(benchmarks) != len(set(benchmarks)):
        raise ValueError("public benchmark names must be unique")
    return int(raw["selection_seed"]), sources, raw


def read_verified_parquet(source: PublicSource, cache_root: str | Path) -> list[dict[str, Any]]:
    path = source.source_path(cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; download {source.repo_id}@{source.revision} file {source.file}"
        )
    observed_sha = file_sha256(path)
    if observed_sha != source.file_sha256:
        raise ValueError(
            f"checksum mismatch for {source.benchmark}: {observed_sha} != {source.file_sha256}"
        )
    try:
        from pyarrow import parquet
    except ImportError as error:
        raise RuntimeError("install the public-benchmarks extra to read Parquet") from error
    rows = parquet.read_table(path).to_pylist()
    if len(rows) != source.expected_rows:
        raise ValueError(
            f"row-count mismatch for {source.benchmark}: {len(rows)} != {source.expected_rows}"
        )
    return rows


def stable_row_key(seed: int, benchmark: str, source_row: int, row: dict[str, Any]) -> str:
    identity = canonical_json(
        {"seed": seed, "benchmark": benchmark, "source_row": source_row, "row": row}
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def stratified_select(
    records: list[dict[str, Any]], max_rows: int, seed: int
) -> list[dict[str, Any]]:
    """Select stable, near-equal strata without exposing benchmark text."""

    if len(records) <= max_rows:
        return sorted(records, key=lambda row: row["selection_key"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["difficulty_stratum"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["selection_key"])

    selected: list[dict[str, Any]] = []
    strata = sorted(groups)
    cursor = {stratum: 0 for stratum in strata}
    while len(selected) < max_rows:
        progressed = False
        order = sorted(
            strata,
            key=lambda stratum: hashlib.sha256(
                f"{seed}:{len(selected)}:{stratum}".encode("ascii")
            ).hexdigest(),
        )
        for stratum in order:
            index = cursor[stratum]
            if index < len(groups[stratum]) and len(selected) < max_rows:
                selected.append(groups[stratum][index])
                cursor[stratum] += 1
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: (row["benchmark"], row["source_row"]))


def write_selection_artifacts(
    output_dir: str | Path,
    config: dict[str, Any],
    sources: list[PublicSource],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    output = Path(output_dir)
    safe_rows = [
        {
            key: row[key]
            for key in (
                "benchmark",
                "content_sha256",
                "difficulty",
                "difficulty_stratum",
                "engine",
                "example_id",
                "selection_key",
                "source_row",
                "target_label",
            )
        }
        for row in selected
    ]
    selection_path = write_jsonl(output / "selection.jsonl", safe_rows)
    counts: dict[str, int] = defaultdict(int)
    strata: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in safe_rows:
        counts[row["benchmark"]] += 1
        strata[row["benchmark"]][str(row["difficulty_stratum"])] += 1
    manifest = {
        "schema_version": "ccpu.paper2.public_selection_manifest.v1",
        "config_fingerprint": fingerprint(config),
        "selection_sha256": file_sha256(selection_path),
        "record_count": len(safe_rows),
        "counts": dict(sorted(counts.items())),
        "difficulty_strata": {
            benchmark: dict(sorted(values.items())) for benchmark, values in sorted(strata.items())
        },
        "sources": [source.provenance() for source in sources],
        "redistribution": "IDs, labels, difficulty metadata, and content hashes only",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
