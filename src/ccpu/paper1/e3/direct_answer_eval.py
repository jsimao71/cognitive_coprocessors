"""Matched direct-answer controls for the official GSM8K evaluation."""

from __future__ import annotations

import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.metrics import wilson_interval
from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig
from ccpu.paper1.public_gsm8k import extract_gsm8k_answer

DIRECT_CONDITIONS = ("direct_concise", "direct_reasoning")
DIRECT_PROTOCOL_ID = "paper1_gsm8k_matched_direct_v1"

_INSTRUCTIONS = {
    "direct_concise": (
        "Solve the arithmetic word problem. Return only one final line in the exact form "
        "`Answer: <number>`. Do not write a program, ASL, or tool call."
    ),
    "direct_reasoning": (
        "Solve the arithmetic word problem carefully. You may reason step by step, but do not "
        "write ASL or use tools. End with one final line in the exact form `Answer: <number>`."
    ),
}


def freeze_direct_gsm8k_protocol(
    *,
    eval_path: str | Path,
    config_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze matched direct controls before any model-facing inference."""

    rows = read_jsonl(eval_path)
    if not rows or any(row.get("source_fields_visible_to_model") != ["question"] for row in rows):
        raise ValueError("direct protocol requires a nonempty question-only evaluation")
    configs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for value in config_paths:
        path = Path(value)
        config = read_json(path)
        condition = str(config.get("condition"))
        if condition not in DIRECT_CONDITIONS:
            raise ValueError(f"invalid or missing direct condition in {path}")
        if condition in configs:
            raise ValueError(f"duplicate direct condition: {condition}")
        configs[condition] = (path, config)
    if set(configs) != set(DIRECT_CONDITIONS):
        raise ValueError(f"direct protocol requires exactly {DIRECT_CONDITIONS}")

    matched_fields = ("model_id", "revision", "device", "dtype", "use_chat_template")
    reference = configs[DIRECT_CONDITIONS[0]][1]["model"]
    for condition in DIRECT_CONDITIONS[1:]:
        candidate = configs[condition][1]["model"]
        if any(candidate.get(field) != reference.get(field) for field in matched_fields):
            raise ValueError(f"direct model provenance differs for {condition}")

    output = Path(output_dir)
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_direct_protocol.v1",
        "protocol_id": DIRECT_PROTOCOL_ID,
        "eval_path": str(eval_path),
        "eval_sha256": file_sha256(eval_path),
        "identity_count": len(rows),
        "example_ids_sha256": fingerprint([row["example_id"] for row in rows]),
        "question_hashes_sha256": fingerprint([row["question_sha256"] for row in rows]),
        "prompt_fields": ["question"],
        "rationales_visible_to_model": False,
        "answers_visible_to_model": False,
        "matched_model_fields": {
            field: reference.get(field) for field in matched_fields
        },
        "declared_condition_differences": ["enable_thinking", "max_new_tokens"],
        "conditions": {
            condition: {
                "instruction": _INSTRUCTIONS[condition],
                "instruction_sha256": fingerprint(_INSTRUCTIONS[condition]),
                "config_path": str(path),
                "config_sha256": file_sha256(path),
                "enable_thinking": bool(config["model"].get("enable_thinking", False)),
                "max_new_tokens": int(config["model"]["max_new_tokens"]),
            }
            for condition, (path, config) in sorted(configs.items())
        },
        "statistical_role": (
            "primary paired controls for the autonomous ASL-runtime condition; "
            "never tune on official outcomes"
        ),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def direct_prompt(question: str, condition: str) -> str:
    """Render a fixed question-only direct-answer prompt."""

    if condition not in DIRECT_CONDITIONS:
        raise ValueError(f"unsupported direct condition: {condition}")
    return f"{_INSTRUCTIONS[condition]}\n\nProblem: {question}\n\nResponse:"


def _numeric_equal(predicted: str | None, expected: Any) -> bool:
    if predicted is None:
        return False
    try:
        return Fraction(predicted.replace(",", "").replace("$", "")) == Fraction(
            str(expected).replace(",", "").replace("$", "")
        )
    except (ValueError, ZeroDivisionError):
        return False


def _summary(
    *, eval_path: str | Path, predictions: list[dict[str, Any]], shard_index: int | None
) -> dict[str, Any]:
    count = len(predictions)
    correct = sum(bool(row["metrics"]["final_answer_correct"]) for row in predictions)
    scorable = sum(bool(row["metrics"]["answer_scorable"]) for row in predictions)
    return {
        "schema_version": "ccpu.paper1.gsm8k_direct_summary.v1",
        "prediction_count": count,
        "shard_index": shard_index,
        "eval_sha256": file_sha256(eval_path),
        "counts": {"answer_scorable": scorable, "final_answer_correct": correct},
        "rates": {
            "answer_scorable": scorable / count if count else 0.0,
            "final_answer_correct": correct / count if count else 0.0,
        },
        "answer_wilson_95": wilson_interval(correct, count),
        "by_difficulty": {
            stratum: {
                "count": len(members),
                "final_answer_correct": sum(
                    bool(row["metrics"]["final_answer_correct"]) for row in members
                ),
            }
            for stratum in sorted(
                {str(row["difficulty_stratum"]) for row in predictions}
            )
            if (
                members := [
                    row
                    for row in predictions
                    if str(row["difficulty_stratum"]) == stratum
                ]
            )
        },
    }


def _run_direct_shard_unlocked(
    *,
    eval_path: str | Path,
    model_config: dict[str, Any],
    condition: str,
    output_dir: str | Path,
    shard_index: int,
    shard_count: int,
    seed: int,
    checkpoint_every: int,
    backend_override: Any | None,
) -> dict[str, Any]:
    if condition not in DIRECT_CONDITIONS:
        raise ValueError(f"unsupported direct condition: {condition}")
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard index/count")
    model = dict(model_config["model"])
    backend = backend_override or HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_new_tokens=int(model["max_new_tokens"]),
            device=str(model.get("device", "xpu")),
            dtype=str(model.get("dtype", "float16")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            cached_generation=True,
        )
    )
    all_rows = read_jsonl(eval_path)
    rows = [row for index, row in enumerate(all_rows) if index % shard_count == shard_index]
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    expected_ids = {str(row["example_id"]) for row in rows}
    if any(
        str(row.get("example_id")) not in expected_ids
        or row.get("condition") != condition
        or row.get("protocol_id") != DIRECT_PROTOCOL_ID
        or int(row.get("shard_index", -1)) != shard_index
        or int(row.get("shard_count", -1)) != shard_count
        for row in predictions
    ):
        raise ValueError("resume output does not match the requested direct GSM8K shard")

    completed = {str(row["example_id"]) for row in predictions}
    remaining = [row for row in rows if str(row["example_id"]) not in completed]
    for index, row in enumerate(remaining, 1):
        prompt = direct_prompt(str(row["question"]), condition)
        generation = backend.generate(prompt, seed=seed)
        predicted = extract_gsm8k_answer(generation.generated_text)
        prediction = {
            "schema_version": "ccpu.paper1.gsm8k_direct_prediction.v1",
            "protocol_id": DIRECT_PROTOCOL_ID,
            "condition": condition,
            "example_id": row["example_id"],
            "parent_example_id": row.get("parent_example_id", row["example_id"]),
            "source_row": row["source_row"],
            "difficulty_stratum": row["difficulty_stratum"],
            "question_sha256": row["question_sha256"],
            "instruction_sha256": fingerprint(_INSTRUCTIONS[condition]),
            "model_id": backend.model_id,
            "seed": seed,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "generated_text": generation.generated_text,
            "predicted_answer": predicted,
            "prompt_tokens": generation.prompt_tokens,
            "generated_tokens": generation.generated_tokens,
            "wall_time_ns": generation.wall_time_ns,
            "backend_metadata": generation.metadata,
            "metrics": {
                "answer_scorable": predicted is not None,
                "final_answer_correct": _numeric_equal(
                    predicted, row["reference_return"]
                ),
                "reference_return": row["reference_return"],
            },
        }
        predictions.append(prediction)
        if index % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint {condition} shard {shard_index}: {len(predictions)}/{len(rows)}")

    predictions.sort(key=lambda row: int(row["source_row"]))
    predictions_path = write_jsonl(predictions_path, predictions)
    summary = _summary(
        eval_path=eval_path, predictions=predictions, shard_index=shard_index
    )
    summary["predictions_sha256"] = file_sha256(predictions_path)
    summary["run"] = {
        "protocol_id": DIRECT_PROTOCOL_ID,
        "condition": condition,
        "instruction": _INSTRUCTIONS[condition],
        "instruction_sha256": fingerprint(_INSTRUCTIONS[condition]),
        "model": model,
        "model_id": backend.model_id,
        "seed": seed,
        "shard_count": shard_count,
        "prompt_fields": ["question"],
        "rationales_visible_to_model": False,
        "answers_visible_to_model": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def run_direct_gsm8k_shard(
    *,
    eval_path: str | Path,
    model_config: dict[str, Any],
    condition: str,
    output_dir: str | Path,
    shard_index: int,
    shard_count: int,
    seed: int = 44017,
    checkpoint_every: int = 5,
    backend_override: Any | None = None,
) -> dict[str, Any]:
    """Run one direct-answer shard with an exclusive resumable output lock."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".run.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(f"direct GSM8K shard output is already locked: {owner}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                canonical_json(
                    {
                        "pid": os.getpid(),
                        "condition": condition,
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                    }
                )
            )
            stream.write("\n")
        return _run_direct_shard_unlocked(
            eval_path=eval_path,
            model_config=model_config,
            condition=condition,
            output_dir=output,
            shard_index=shard_index,
            shard_count=shard_count,
            seed=seed,
            checkpoint_every=checkpoint_every,
            backend_override=backend_override,
        )
    finally:
        lock_path.unlink(missing_ok=True)


def merge_direct_gsm8k_shards(
    *, eval_path: str | Path, shard_dirs: list[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Merge a complete matched direct run while enforcing shard provenance."""

    if not shard_dirs:
        raise ValueError("at least one direct shard directory is required")
    eval_rows = read_jsonl(eval_path)
    eval_positions = {
        str(row["example_id"]): position for position, row in enumerate(eval_rows)
    }
    if len(eval_positions) != len(eval_rows):
        raise ValueError("direct evaluation contains duplicate identities")
    source_paths = [Path(directory) / "predictions.jsonl" for directory in shard_dirs]
    predictions = [row for path in source_paths for row in read_jsonl(path)]
    observed_ids = [str(row["example_id"]) for row in predictions]
    duplicates = sorted(key for key, count in Counter(observed_ids).items() if count > 1)
    missing = sorted(set(eval_positions) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(eval_positions))
    if duplicates or missing or unexpected:
        raise ValueError(
            f"incomplete direct shard merge: duplicates={duplicates[:3]} "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )

    invariant_fields = (
        "protocol_id",
        "condition",
        "instruction_sha256",
        "model_id",
        "seed",
        "shard_count",
    )
    invariants = {
        field: {str(row.get(field)) for row in predictions} for field in invariant_fields
    }
    mixed = {field: values for field, values in invariants.items() if len(values) != 1}
    if mixed:
        raise ValueError(f"direct shards have mixed provenance: {mixed}")
    if next(iter(invariants["protocol_id"])) != DIRECT_PROTOCOL_ID:
        raise ValueError("direct shards use an unsupported protocol")
    shard_count = int(next(iter(invariants["shard_count"])))
    shard_indices = {int(row["shard_index"]) for row in predictions}
    if shard_indices != set(range(shard_count)):
        raise ValueError(
            f"direct shard indices are incomplete: observed={sorted(shard_indices)} "
            f"expected={list(range(shard_count))}"
        )
    misplaced = [
        row["example_id"]
        for row in predictions
        if eval_positions[str(row["example_id"])] % shard_count != int(row["shard_index"])
    ]
    if misplaced:
        raise ValueError(f"direct predictions are assigned to the wrong shard: {misplaced[:3]}")

    predictions.sort(key=lambda row: eval_positions[str(row["example_id"])])
    output = Path(output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", predictions)
    summary = _summary(eval_path=eval_path, predictions=predictions, shard_index=None)
    summary["predictions_sha256"] = file_sha256(predictions_path)
    summary["run"] = {
        "protocol_id": DIRECT_PROTOCOL_ID,
        "condition": next(iter(invariants["condition"])),
        "instruction_sha256": next(iter(invariants["instruction_sha256"])),
        "model_id": next(iter(invariants["model_id"])),
        "seed": int(next(iter(invariants["seed"]))),
        "shard_count": shard_count,
        "source_prediction_sha256": [file_sha256(path) for path in source_paths],
        "prompt_fields": ["question"],
        "rationales_visible_to_model": False,
        "answers_visible_to_model": False,
    }
    write_json(output / "summary.json", summary)
    return summary
