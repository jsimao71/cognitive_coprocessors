"""Frozen training data for the budget-matched Direct-answer LoRA control."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl

from .direct_answer_eval import DIRECT_PROTOCOL_ID, direct_prompt


_CALCULATION_ANNOTATION = re.compile(r"<<[^<>]*>>")


def _direct_target(raw: dict[str, Any]) -> str:
    reasoning = str(raw.get("gold_reasoning", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    if not reasoning or not answer:
        raise ValueError(f"raw GSM8K row {raw.get('source_id')} lacks supervision")
    # GSM8K's <<expression=result>> spans are dataset metadata, not model syntax.
    reasoning = _CALCULATION_ANNOTATION.sub("", reasoning)
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning).strip()
    return f"{reasoning}\nAnswer: {answer}"


def build_direct_lora_data(
    *,
    train_path: str | Path,
    dev_path: str | Path,
    raw_gsm8k_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convert frozen ASL identities into a matched question-to-rationale control."""

    train_path = Path(train_path)
    dev_path = Path(dev_path)
    raw_gsm8k_path = Path(raw_gsm8k_path)
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    raw_rows = read_jsonl(raw_gsm8k_path)
    if not train or not dev or not raw_rows:
        raise ValueError("Direct LoRA preparation requires nonempty train, dev, and raw data")

    raw_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if row.get("dataset") != "gsm8k" or row.get("split") != "train":
            continue
        source_id = str(row["source_id"])
        if source_id in raw_by_id:
            raise ValueError(f"duplicate raw GSM8K source_id: {source_id}")
        raw_by_id[source_id] = row

    def convert(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            source_id = str(row.get("parent_source_id", ""))
            if source_id not in raw_by_id:
                raise ValueError(f"{split} row lacks raw GSM8K source: {source_id}")
            raw = raw_by_id[source_id]
            question = str(raw["question"]).strip()
            if f"Problem: {question}" not in str(row.get("prompt", "")):
                raise ValueError(f"question mismatch for {split} source {source_id}")
            target = _direct_target(raw)
            converted.append(
                {
                    "schema_version": "ccpu.paper1.gsm8k_direct_lora_sft.v1",
                    "dataset": "gsm8k",
                    "dataset_id": "GSM8K_DIRECT_LORA_MATCHED_U2000_E4500",
                    "example_id": f"direct-lora-{fingerprint([split, row['example_id']], 20)}",
                    "paired_asl_example_id": row["example_id"],
                    "parent_example_id": row.get("parent_example_id", f"gsm8k:{source_id}"),
                    "parent_source_id": source_id,
                    "split": split,
                    "epoch_view": row.get("epoch_view"),
                    "prompt": direct_prompt(question, "direct_reasoning"),
                    "target": target,
                    "source_fields_visible_to_model": ["question"],
                    "supervision": "gsm8k_gold_reasoning_and_answer",
                    "source_row": index,
                }
            )
        return converted

    direct_train = convert(train, "train")
    direct_dev = convert(dev, "dev")
    train_sources = {row["parent_source_id"] for row in direct_train}
    dev_sources = {row["parent_source_id"] for row in direct_dev}
    overlap = sorted(train_sources & dev_sources)
    if overlap:
        raise ValueError(f"Direct LoRA train/dev source leakage: {overlap[:5]}")

    output = Path(output_dir)
    train_output = write_jsonl(output / "train.jsonl", direct_train)
    dev_output = write_jsonl(output / "dev.jsonl", direct_dev)
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_direct_lora_freeze.v1",
        "protocol_id": DIRECT_PROTOCOL_ID,
        "condition": "direct_reasoning",
        "comparison_role": "compute/exposure-matched Direct-answer LoRA control",
        "source_fields_visible_to_model": ["question"],
        "target_supervision": "GSM8K gold reasoning with calculation annotations removed; exact Answer line appended",
        "counts": {
            "train_rows": len(direct_train),
            "dev_rows": len(direct_dev),
            "unique_train_sources": len(train_sources),
            "unique_dev_sources": len(dev_sources),
            "train_target_characters": sum(len(row["target"]) for row in direct_train),
            "dev_target_characters": sum(len(row["target"]) for row in direct_dev),
        },
        "identity_audit": {
            "train_row_order_preserved": [row["paired_asl_example_id"] for row in direct_train]
            == [row["example_id"] for row in train],
            "dev_row_order_preserved": [row["paired_asl_example_id"] for row in direct_dev]
            == [row["example_id"] for row in dev],
            "train_dev_source_overlap": overlap,
        },
        "input_sha256": {
            "asl_train": file_sha256(train_path),
            "asl_dev": file_sha256(dev_path),
            "raw_gsm8k": file_sha256(raw_gsm8k_path),
        },
        "output_sha256": {
            "train": file_sha256(train_output),
            "dev": file_sha256(dev_output),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
