"""Leakage-safe answer-only evaluation on the official GSM8K test split."""

from __future__ import annotations

import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl import validate_asl
from ccpu.dsl_dataset.loaders import load_dataset
from ccpu.paper1.asl_matrix.qwen import autonomous_asl_prompt
from ccpu.paper1.asl_pilot_eval import extract_asl
from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig


def _answer(raw: Any) -> str:
    text = str(raw)
    return text.rsplit("####", 1)[-1].strip() if "####" in text else text.strip()


def _decimal(raw: Any) -> Decimal:
    text = str(raw).strip().replace(",", "").replace("$", "")
    return Decimal(text)


def _difficulty(reasoning: str) -> tuple[int, str]:
    steps = max(1, reasoning.count("<<"))
    if steps <= 2:
        return steps, "low"
    if steps <= 4:
        return steps, "medium"
    return steps, "high"


def _normalized_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _training_question(row: dict[str, Any]) -> str:
    if "question" in row:
        return str(row["question"])
    prompt = str(row.get("prompt", ""))
    marker = "\nProblem: "
    if marker not in prompt or not prompt.endswith("\nASL:"):
        raise ValueError("training row has neither question nor registered autonomous prompt")
    return prompt.split(marker, 1)[1][: -len("\nASL:")]


def _balanced_select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError("confirmatory size must be between one and the full split size")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for stratum in sorted({str(row["difficulty_stratum"]) for row in rows}):
        members = [row for row in rows if row["difficulty_stratum"] == stratum]
        buckets[stratum] = sorted(
            members, key=lambda row: fingerprint(f"{seed}:{row['example_id']}")
        )
    selected: list[dict[str, Any]] = []
    offsets = Counter()
    while len(selected) < count:
        progressed = False
        for stratum in sorted(
            buckets,
            key=lambda value: fingerprint(f"{seed}:{len(selected)}:{value}"),
        ):
            offset = offsets[stratum]
            if offset < len(buckets[stratum]) and len(selected) < count:
                selected.append(buckets[stratum][offset])
                offsets[stratum] += 1
                progressed = True
        if not progressed:
            raise AssertionError("balanced selection exhausted before reaching target")
    return sorted(selected, key=lambda row: int(row["source_row"]))


def freeze_official_gsm8k(
    *,
    source_path: str | Path,
    train_paths: list[str | Path],
    output_dir: str | Path,
    expected_sha256: str,
    expected_rows: int = 1319,
    confirmatory_size: int = 250,
    seed: int = 22901,
) -> dict[str, Any]:
    """Freeze full and stratified test views without exposing GSM8K rationales."""

    source = Path(source_path)
    observed_sha256 = file_sha256(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"GSM8K source checksum mismatch: {observed_sha256}")
    loaded = load_dataset("gsm8k", source, "test")
    if len(loaded) != expected_rows:
        raise ValueError(f"GSM8K row-count mismatch: {len(loaded)} != {expected_rows}")

    train_question_hashes = {
        fingerprint(_normalized_question(_training_question(row)))
        for path in train_paths
        for row in read_jsonl(path)
    }
    rows = []
    overlaps = []
    for index, row in enumerate(loaded):
        question = str(row["question"])
        question_hash = fingerprint(_normalized_question(question))
        if question_hash in train_question_hashes:
            overlaps.append(index)
        steps, stratum = _difficulty(str(row["gold_reasoning"]))
        rows.append(
            {
                "schema_version": "ccpu.paper1.gsm8k_confirmatory.v1",
                "example_id": f"gsm8k-official-test-{index:04d}",
                "dataset": "gsm8k",
                "split": "test",
                "source_row": index,
                "question": question,
                "question_sha256": question_hash,
                "reference_return": _answer(row["answer"]),
                "difficulty_steps": steps,
                "difficulty_stratum": stratum,
                "effective_scope": row["effective_scope"],
                "source_fields_visible_to_model": ["question"],
            }
        )
    if overlaps:
        raise ValueError(f"official test has {len(overlaps)} exact train-question overlaps")

    confirmatory = _balanced_select(rows, confirmatory_size, seed)
    output = Path(output_dir)
    full_path = write_jsonl(output / "full.jsonl", rows)
    confirmatory_path = write_jsonl(output / "confirmatory.jsonl", confirmatory)
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_confirmatory_manifest.v1",
        "dataset": "openai/gsm8k",
        "revision": "a05f38c23a0e9ab0b71de8a2b4947e20f74f68f7",
        "split": "test",
        "selection_seed": seed,
        "roles": {
            "full": "untouched answer/execution confirmation; never tune",
            "confirmatory": "stratified frozen diagnostic view; never tune",
        },
        "counts": {
            "full": len(rows),
            "confirmatory": len(confirmatory),
            "full_by_stratum": dict(Counter(row["difficulty_stratum"] for row in rows)),
            "confirmatory_by_stratum": dict(
                Counter(row["difficulty_stratum"] for row in confirmatory)
            ),
        },
        "leakage_audit": {
            "train_files": {str(path): file_sha256(path) for path in train_paths},
            "exact_normalized_question_overlap": overlaps,
            "passed": not overlaps,
        },
        "source": {
            "path": str(source),
            "sha256": observed_sha256,
            "expected_rows": expected_rows,
        },
        "output_sha256": {
            "full": file_sha256(full_path),
            "confirmatory": file_sha256(confirmatory_path),
        },
        "rationales_visible_to_model": False,
        "answers_visible_to_model": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _score_prediction(predicted_asl: str, expected: Any, scope: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "parse_valid": False,
        "lowerable_to_ccir": False,
        "type_valid": False,
        "executable": False,
        "final_answer_correct": False,
        "errors": [],
    }
    if not predicted_asl.strip():
        metrics["errors"].append("no ASL statements extracted")
        return metrics
    validation = validate_asl(predicted_asl, effective_scope=scope)
    metrics["parse_valid"] = bool(validation["syntax_verified"])
    metrics["lowerable_to_ccir"] = bool(validation["lower_verified"])
    metrics["type_valid"] = bool(validation["type_verified"])
    metrics["executable"] = bool(validation["execution_verified"])
    metrics["errors"] = list(validation["errors"])
    if not metrics["executable"]:
        return metrics
    actual = validation["execution"]["workspace"][str(scope["id"])]["returned"]
    try:
        metrics["final_answer_correct"] = abs(_decimal(actual) - _decimal(expected)) <= Decimal(
            "0.011"
        )
    except (InvalidOperation, TypeError, ValueError):
        metrics["final_answer_correct"] = str(actual).strip() == str(expected).strip()
    metrics["predicted_return"] = actual
    metrics["reference_return"] = expected
    return metrics


def _summary(
    *, eval_path: str | Path, predictions: list[dict[str, Any]], shard_index: int | None = None
) -> dict[str, Any]:
    metric_names = (
        "parse_valid",
        "lowerable_to_ccir",
        "type_valid",
        "executable",
        "final_answer_correct",
    )
    count = len(predictions)
    return {
        "schema_version": "ccpu.paper1.gsm8k_answer_evaluation.v1",
        "prediction_count": count,
        "shard_index": shard_index,
        "eval_sha256": file_sha256(eval_path),
        "rates": {
            name: sum(bool(row["metrics"][name]) for row in predictions) / count
            if count
            else 0.0
            for name in metric_names
        },
        "counts": {
            name: sum(bool(row["metrics"][name]) for row in predictions)
            for name in metric_names
        },
        "by_difficulty": {
            stratum: {
                "count": len(members),
                "final_answer_correct": sum(
                    bool(row["metrics"]["final_answer_correct"]) for row in members
                ),
            }
            for stratum in sorted({str(row["difficulty_stratum"]) for row in predictions})
            if (members := [row for row in predictions if row["difficulty_stratum"] == stratum])
        },
    }


def _run_official_gsm8k_shard_unlocked(
    *,
    eval_path: str | Path,
    model_config: dict[str, Any],
    adapter_path: str | Path,
    adapter_id: str,
    output_dir: str | Path,
    shard_index: int,
    shard_count: int,
    seed: int = 44017,
    checkpoint_every: int = 5,
    backend_override: Any | None = None,
) -> dict[str, Any]:
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard index/count")
    model = dict(model_config["model"])
    model["adapter_path"] = str(adapter_path)
    model["adapter_id"] = adapter_id
    backend = backend_override or HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_new_tokens=int(model.get("max_new_tokens", 384)),
            device=str(model.get("device", "xpu")),
            dtype=str(model.get("dtype", "float16")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            adapter_path=str(adapter_path),
            adapter_id=adapter_id,
            cached_generation=True,
        )
    )
    all_rows = read_jsonl(eval_path)
    rows = [row for index, row in enumerate(all_rows) if index % shard_count == shard_index]
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    expected_ids = {row["example_id"] for row in rows}
    if any(
        row["example_id"] not in expected_ids
        or row.get("adapter_id") != adapter_id
        or int(row.get("shard_index", -1)) != shard_index
        or int(row.get("shard_count", -1)) != shard_count
        for row in predictions
    ):
        raise ValueError("resume output does not match the requested GSM8K shard")
    completed = {row["example_id"] for row in predictions}
    for index, row in enumerate([row for row in rows if row["example_id"] not in completed], 1):
        prompt = autonomous_asl_prompt(str(row["question"]))
        generation = backend.generate(prompt, seed=seed)
        predicted_asl = extract_asl(generation.generated_text)
        prediction = {
            "schema_version": "ccpu.paper1.gsm8k_answer_prediction.v1",
            "example_id": row["example_id"],
            "source_row": row["source_row"],
            "difficulty_stratum": row["difficulty_stratum"],
            "adapter_id": adapter_id,
            "model_id": backend.model_id,
            "seed": seed,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "generated_text": generation.generated_text,
            "predicted_asl": predicted_asl,
            "prompt_tokens": generation.prompt_tokens,
            "generated_tokens": generation.generated_tokens,
            "wall_time_ns": generation.wall_time_ns,
            "backend_metadata": generation.metadata,
            "metrics": _score_prediction(
                predicted_asl, row["reference_return"], row["effective_scope"]
            ),
        }
        predictions.append(prediction)
        if index % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint shard {shard_index}: {len(predictions)}/{len(rows)}")
    predictions.sort(key=lambda row: int(row["source_row"]))
    predictions_path = write_jsonl(predictions_path, predictions)
    summary = _summary(eval_path=eval_path, predictions=predictions, shard_index=shard_index)
    summary["predictions_sha256"] = file_sha256(predictions_path)
    summary["run"] = {
        "adapter_id": adapter_id,
        "model": model,
        "seed": seed,
        "shard_count": shard_count,
        "prompt_fields": ["question"],
    }
    write_json(output / "summary.json", summary)
    return summary


def run_official_gsm8k_shard(
    *,
    eval_path: str | Path,
    model_config: dict[str, Any],
    adapter_path: str | Path,
    adapter_id: str,
    output_dir: str | Path,
    shard_index: int,
    shard_count: int,
    seed: int = 44017,
    checkpoint_every: int = 5,
    backend_override: Any | None = None,
) -> dict[str, Any]:
    """Run one shard while preventing concurrent writers for its output path."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".run.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(f"GSM8K shard output is already locked: {owner}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                canonical_json(
                    {
                        "pid": os.getpid(),
                        "adapter_id": adapter_id,
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                    }
                )
            )
            stream.write("\n")
        return _run_official_gsm8k_shard_unlocked(
            eval_path=eval_path,
            model_config=model_config,
            adapter_path=adapter_path,
            adapter_id=adapter_id,
            output_dir=output,
            shard_index=shard_index,
            shard_count=shard_count,
            seed=seed,
            checkpoint_every=checkpoint_every,
            backend_override=backend_override,
        )
    finally:
        lock_path.unlink(missing_ok=True)


def merge_official_gsm8k_shards(
    *, eval_path: str | Path, shard_dirs: list[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    eval_rows = read_jsonl(eval_path)
    expected_ids = {row["example_id"] for row in eval_rows}
    predictions = [
        row
        for directory in shard_dirs
        for row in read_jsonl(Path(directory) / "predictions.jsonl")
    ]
    observed_ids = [row["example_id"] for row in predictions]
    duplicates = sorted(key for key, count in Counter(observed_ids).items() if count > 1)
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    if duplicates or missing or unexpected:
        raise ValueError(
            f"incomplete shard merge: duplicates={duplicates[:3]} missing={missing[:3]} "
            f"unexpected={unexpected[:3]}"
        )
    adapter_ids = {row["adapter_id"] for row in predictions}
    if len(adapter_ids) != 1:
        raise ValueError(f"shards use different adapters: {sorted(adapter_ids)}")
    predictions.sort(key=lambda row: int(row["source_row"]))
    output = Path(output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", predictions)
    summary = _summary(eval_path=eval_path, predictions=predictions)
    summary["adapter_id"] = next(iter(adapter_ids))
    summary["shard_count"] = len(shard_dirs)
    summary["predictions_sha256"] = file_sha256(predictions_path)
    write_json(output / "summary.json", summary)
    return summary


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    margin /= denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def analyze_official_gsm8k_replications(
    *, candidate_paths: list[tuple[str, str | Path]], output_path: str | Path
) -> dict[str, Any]:
    """Aggregate matched-seed confirmation without pooling seed observations."""

    if not candidate_paths:
        raise ValueError("at least one official GSM8K candidate is required")
    indexed = []
    for label, path_value in candidate_paths:
        path = Path(path_value)
        rows = read_jsonl(path)
        by_id = {str(row["example_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate official GSM8K identity in {label}")
        indexed.append((label, path, by_id))
    identities = set(indexed[0][2])
    for label, _, rows in indexed[1:]:
        if set(rows) != identities:
            raise ValueError(f"official GSM8K identities differ for {label}")

    endpoints = (
        "parse_valid",
        "lowerable_to_ccir",
        "type_valid",
        "executable",
        "final_answer_correct",
    )
    per_seed = {}
    for label, path, rows in indexed:
        counts = {
            endpoint: sum(bool(row["metrics"][endpoint]) for row in rows.values())
            for endpoint in endpoints
        }
        per_seed[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "count": len(rows),
            "counts": counts,
            "rates": {endpoint: counts[endpoint] / len(rows) for endpoint in endpoints},
            "answer_wilson_95": _wilson(counts["final_answer_correct"], len(rows)),
            "by_difficulty": {
                stratum: {
                    "count": len(members),
                    "final_answer_correct": sum(
                        bool(row["metrics"]["final_answer_correct"]) for row in members
                    ),
                }
                for stratum in sorted(
                    {str(row["difficulty_stratum"]) for row in rows.values()}
                )
                if (
                    members := [
                        row
                        for row in rows.values()
                        if str(row["difficulty_stratum"]) == stratum
                    ]
                )
            },
        }

    aggregate = {}
    for endpoint in endpoints:
        values = [per_seed[label]["rates"][endpoint] for label, _, _ in indexed]
        counts = [per_seed[label]["counts"][endpoint] for label, _, _ in indexed]
        aggregate[endpoint] = {
            "counts": counts,
            "rates": values,
            "mean_rate": mean(values),
            "min_rate": min(values),
            "max_rate": max(values),
        }
    difficulty_strata = sorted(
        {str(row["difficulty_stratum"]) for row in indexed[0][2].values()}
    )
    aggregate_by_difficulty = {}
    for stratum in difficulty_strata:
        stratum_counts = [
            per_seed[label]["by_difficulty"][stratum]["count"]
            for label, _, _ in indexed
        ]
        if len(set(stratum_counts)) != 1:
            raise ValueError(f"difficulty count differs across seeds for {stratum}")
        correct_counts = [
            per_seed[label]["by_difficulty"][stratum]["final_answer_correct"]
            for label, _, _ in indexed
        ]
        rates = [correct / stratum_counts[0] for correct in correct_counts]
        aggregate_by_difficulty[stratum] = {
            "count_per_seed": stratum_counts[0],
            "final_answer_correct_counts": correct_counts,
            "rates": rates,
            "mean_rate": mean(rates),
            "min_rate": min(rates),
            "max_rate": max(rates),
        }
    answer_vectors = {
        label: {
            identity: bool(rows[identity]["metrics"]["final_answer_correct"])
            for identity in identities
        }
        for label, _, rows in indexed
    }
    pairwise = {}
    labels = [label for label, _, _ in indexed]
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "both_correct": sum(
                    answer_vectors[left][key] and answer_vectors[right][key]
                    for key in identities
                ),
                "left_only": sum(
                    answer_vectors[left][key] and not answer_vectors[right][key]
                    for key in identities
                ),
                "right_only": sum(
                    not answer_vectors[left][key] and answer_vectors[right][key]
                    for key in identities
                ),
                "both_wrong": sum(
                    not answer_vectors[left][key] and not answer_vectors[right][key]
                    for key in identities
                ),
            }
    correct_seed_histogram = Counter(
        sum(answer_vectors[label][identity] for label in labels)
        for identity in identities
    )
    report = {
        "schema_version": "ccpu.paper1.gsm8k_answer_replications.v1",
        "identity_count": len(identities),
        "seed_count": len(indexed),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "aggregate_by_difficulty": aggregate_by_difficulty,
        "answer_agreement": {
            "correct_seed_count_histogram": {
                str(correct_count): correct_seed_histogram.get(correct_count, 0)
                for correct_count in range(len(labels) + 1)
            },
            "unanimous_correct": correct_seed_histogram.get(len(labels), 0),
            "unanimous_wrong": correct_seed_histogram.get(0, 0),
        },
        "pairwise_answer_outcomes": pairwise,
        "statistical_boundary": (
            "Seeds reuse identical frozen questions; report seed means and ranges without "
            "pooling seed-question predictions as independent observations."
        ),
    }
    write_json(output_path, report)
    return report
