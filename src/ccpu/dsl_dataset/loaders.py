"""Normalized adapters for public arithmetic and future Paper 2 datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[Any]:
    if path.suffix.casefold() == ".parquet":
        try:
            from pyarrow import parquet
        except ImportError as error:
            raise RuntimeError("install the 'public-benchmarks' extra to read parquet") from error
        return parquet.read_table(path).to_pylist()
    if path.suffix.casefold() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else [value]


def _scope(dataset: str, split: str, source_id: str) -> dict[str, Any]:
    return {
        "id": f"{dataset}:{split}:{source_id}",
        "parent": None,
        "kind": "benchmark_case",
        "source": "dataset",
    }


def _normalized(
    dataset: str,
    split: str,
    source_id: str,
    question: Any,
    answer: Any,
    reasoning: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "source_id": str(source_id),
        "question": str(question),
        "answer": answer,
        "gold_reasoning": str(reasoning or ""),
        "metadata": metadata,
        "effective_scope": _scope(dataset, split, str(source_id)),
        "parts": [],
    }


def load_gsm8k(path: Path, split: str) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(_read_rows(path)):
        raw_answer = str(row.get("answer", ""))
        if "####" in raw_answer:
            reasoning, answer = raw_answer.rsplit("####", 1)
        else:
            reasoning, answer = row.get("reasoning", ""), raw_answer
        yield _normalized(
            "gsm8k",
            str(row.get("split", row.get("task_split", split))),
            row.get("id", row.get("source_id", index)),
            row.get("question", row.get("problem", "")),
            str(answer).strip(),
            reasoning,
            {"arithmetic_compatible": True},
        )


def load_tatqa(path: Path, split: str) -> Iterable[dict[str, Any]]:
    for document_index, document in enumerate(_read_rows(path)):
        for question_index, question in enumerate(document.get("questions", [])):
            arithmetic = question.get("answer_type") == "arithmetic"
            metadata = {
                "arithmetic_compatible": arithmetic,
                "answer_type": question.get("answer_type"),
                "answer_from": question.get("answer_from"),
                "scale": question.get("scale"),
                "document_index": document_index,
                "question_index": question_index,
            }
            yield _normalized(
                "tatqa",
                split,
                question.get("uid", f"{document_index}:{question_index}"),
                question.get("question", ""),
                question.get("answer"),
                question.get("derivation", ""),
                metadata,
            )


def load_generic(dataset: str, path: Path, split: str) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(_read_rows(path)):
        if not isinstance(row, dict):
            continue
        if dataset == "clutrr":
            story = row.get("clean_story", row.get("story", ""))
            question = f"{story}\nQuery: {row.get('query', '')}"
        elif dataset in {"ruletaker", "proofwriter"}:
            context = row.get("theory", row.get("context", ""))
            question = f"{context}\nQuery: {row.get('question', '')}"
        else:
            question = next(
                (
                    row[key]
                    for key in ("question", "problem", "query", "story", "text")
                    if key in row
                ),
                "",
            )
        answer = next(
            (row[key] for key in ("answer", "target_text", "target", "label") if key in row),
            "",
        )
        reasoning = next(
            (row[key] for key in ("reasoning", "rationale", "equation", "solution") if key in row),
            "",
        )
        source_id = next((row[key] for key in ("id", "uid", "source_id") if key in row), index)
        arithmetic = dataset not in {"clutrr", "ruletaker", "proofwriter"}
        yield _normalized(
            dataset,
            str(row.get("split", split)),
            source_id,
            question,
            answer,
            reasoning,
            {
                "arithmetic_compatible": arithmetic,
                "ingestion_only": not arithmetic,
                "source_fields": sorted(row),
            },
        )


def load_dataset(dataset: str, path: str | Path, split: str) -> list[dict[str, Any]]:
    source = Path(path)
    if dataset == "gsm8k":
        return list(load_gsm8k(source, split))
    if dataset == "tatqa":
        return list(load_tatqa(source, split))
    return list(load_generic(dataset, source, split))
