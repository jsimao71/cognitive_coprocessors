"""Verified GSM8K materialization shared by public compute experiments."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .artifacts import canonical_json, read_jsonl
from .public_benchmarks import load_config, read_verified_parquet

_FINAL_ANSWER = re.compile(r"####\s*([^\r\n]+)")
_ARITHMETIC_TRACE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def materialize_gsm8k(
    config_path: str | Path,
    cache_root: str | Path,
    selection_path: str | Path,
) -> list[dict[str, Any]]:
    """Load selected rows only after source, ID, label, and content verification."""
    _, sources, _ = load_config(config_path)
    matches = [source for source in sources if source.benchmark == "gsm8k"]
    if len(matches) != 1:
        raise ValueError("expected exactly one GSM8K source")
    source = matches[0]
    selected = [
        row for row in read_jsonl(selection_path) if row.get("benchmark") == "gsm8k"
    ]
    by_row = {int(row["source_row"]): row for row in selected}
    if len(by_row) != len(selected):
        raise ValueError("GSM8K selection contains duplicate source rows")

    materialized = []
    for index, raw in enumerate(read_verified_parquet(source, cache_root)):
        frozen = by_row.get(index)
        if frozen is None:
            continue
        content_sha = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
        answer_match = _FINAL_ANSWER.search(str(raw["answer"]))
        if answer_match is None:
            raise ValueError(f"GSM8K row {index} has no final answer")
        expected = {
            "example_id": f"gsm8k:test:{index}",
            "content_sha256": content_sha,
            "target_label": answer_match.group(1).strip(),
        }
        for key, value in expected.items():
            if frozen[key] != value:
                raise ValueError(f"selected GSM8K row {index} changed at {key}")
        opportunities = [
            {"expression": expression.strip(), "result": result.strip()}
            for expression, result in _ARITHMETIC_TRACE.findall(str(raw["answer"]))
        ]
        materialized.append(
            {
                **frozen,
                "question": str(raw["question"]),
                "opportunities": opportunities,
            }
        )
    if len(materialized) != len(selected):
        raise ValueError("not every selected GSM8K row was materialized")
    return materialized
