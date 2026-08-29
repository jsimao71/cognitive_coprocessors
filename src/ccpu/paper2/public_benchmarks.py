"""Public heterogeneous-compute benchmark selection and stratification."""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import canonical_json
from ccpu.common.public_benchmarks import (
    PublicSource,
    load_config,
    read_verified_parquet,
    stable_row_key,
    stratified_select,
    write_selection_artifacts,
)

Normalizer = Callable[[PublicSource, int, dict[str, Any], int], dict[str, Any]]


def _content_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def _record(
    source: PublicSource,
    source_row: int,
    row: dict[str, Any],
    seed: int,
    *,
    example_id: str,
    target_label: str,
    difficulty: int,
    difficulty_stratum: str,
) -> dict[str, Any]:
    return {
        "benchmark": source.benchmark,
        "engine": source.engine,
        "example_id": example_id,
        "source_row": source_row,
        "target_label": target_label,
        "difficulty": difficulty,
        "difficulty_stratum": difficulty_stratum,
        "content_sha256": _content_sha(row),
        "selection_key": stable_row_key(seed, source.benchmark, source_row, row),
    }


def _gsm8k(source: PublicSource, index: int, row: dict[str, Any], seed: int) -> dict[str, Any]:
    steps = max(1, str(row["answer"]).count("<<"))
    answer_match = re.search(r"####\s*([^\r\n]+)", str(row["answer"]))
    if not answer_match:
        raise ValueError(f"GSM8K row {index} has no final answer marker")
    stratum = "2_steps" if steps <= 2 else "3_4_steps" if steps <= 4 else "5plus_steps"
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"gsm8k:test:{index}",
        target_label=answer_match.group(1).strip(),
        difficulty=steps,
        difficulty_stratum=stratum,
    )


def _unit_conversion(
    source: PublicSource, index: int, row: dict[str, Any], seed: int
) -> dict[str, Any]:
    prompt = str(row["inputs"]).lower()
    power_count = prompt.count("^")
    rate_count = prompt.count(" per ") + prompt.count("/")
    difficulty = 1 + rate_count + power_count
    stratum = "powered" if power_count else "compound_rate" if rate_count else "linear"
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"bigbench:unit_conversion:{row['idx']}",
        target_label=str(row["targets"][0]),
        difficulty=difficulty,
        difficulty_stratum=stratum,
    )


_DATE_TOKEN = re.compile(
    r"\b(?:\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_TEMPORAL_CUE = re.compile(
    r"\b(?:before|after|later|earlier|tomorrow|yesterday|ago|from now|next|previous)\b",
    re.IGNORECASE,
)


def _date_understanding(
    source: PublicSource, index: int, row: dict[str, Any], seed: int
) -> dict[str, Any]:
    prompt = str(row["inputs"])
    difficulty = max(1, len(_DATE_TOKEN.findall(prompt)) + len(_TEMPORAL_CUE.findall(prompt)))
    stratum = "single_step" if difficulty <= 2 else "two_step" if difficulty <= 4 else "multi_step"
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"bigbench:date_understanding:{row['idx']}",
        target_label=str(row["targets"][0]),
        difficulty=difficulty,
        difficulty_stratum=stratum,
    )


def _proofwriter(
    source: PublicSource, index: int, row: dict[str, Any], seed: int
) -> dict[str, Any]:
    depth = int(row["QDep"])
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"proofwriter:{row['id']}",
        target_label=str(row["answer"]).upper(),
        difficulty=depth,
        difficulty_stratum=f"depth_{depth}",
    )


def _clutrr(source: PublicSource, index: int, row: dict[str, Any], seed: int) -> dict[str, Any]:
    edges = ast.literal_eval(str(row["edge_types"]))
    if not isinstance(edges, list):
        raise TypeError(f"CLUTRR row {index} has invalid edge list")
    depth = len(edges)
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"clutrr:{row['id']}",
        target_label=str(row["target_text"]),
        difficulty=depth,
        difficulty_stratum=f"depth_{depth}",
    )


_NORMALIZERS: dict[str, Normalizer] = {
    "gsm8k": _gsm8k,
    "bigbench_unit_conversion": _unit_conversion,
    "bigbench_date_understanding": _date_understanding,
    "proofwriter_balanced": _proofwriter,
    "clutrr": _clutrr,
}


def freeze_public_suite(
    config_path: str | Path, cache_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    seed, sources, config = load_config(config_path)
    selected: list[dict[str, Any]] = []
    for source in sources:
        try:
            normalizer = _NORMALIZERS[source.benchmark]
        except KeyError as error:
            raise ValueError(f"no normalizer for {source.benchmark}") from error
        source_rows = read_verified_parquet(source, cache_root)
        records = [
            normalizer(source, index, row, seed) for index, row in enumerate(source_rows)
        ]
        selected.extend(stratified_select(records, source.max_rows, seed))
    return write_selection_artifacts(output_dir, config, sources, selected)
