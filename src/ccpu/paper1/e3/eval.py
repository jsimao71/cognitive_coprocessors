"""Autonomous generation and deterministic evaluation for the F4 semantic IR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    file_sha256,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.paper1.asl_pilot_eval import score_asl
from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig

from .bottleneck import lower_bottleneck_to_asl, parse_bottleneck


def extract_bottleneck(text: str) -> str:
    """Extract the first structurally valid JSON object from model output."""

    decoder = json.JSONDecoder()
    stripped = text.strip()
    for offset, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[offset:])
            return canonical_json(parse_bottleneck(canonical_json(value)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return stripped


def score_bottleneck(
    *,
    reference_program: str,
    reference_asl: str,
    predicted_program: str,
    effective_scope: dict[str, Any],
) -> dict[str, Any]:
    """Score F4 structure, deterministic lowering, and ordinary ASL semantics."""

    result: dict[str, Any] = {
        "exact_program": False,
        "parse_valid": False,
        "lowerable_to_asl": False,
        "lowerable_to_ccir": False,
        "type_valid": False,
        "semantic_lint_valid": False,
        "executable": False,
        "dependency_correct": False,
        "semantic_return_equivalent": False,
        "semantic_state_equivalent": False,
        "final_answer_correct": False,
        "lowered_asl": "",
        "errors": [],
    }
    for name in ("paths", "source_facts", "operators", "edges", "semantic_state"):
        result[f"{name}_metrics"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        parsed = parse_bottleneck(predicted_program)
        canonical_prediction = canonical_json(parsed)
        result["parse_valid"] = True
        result["exact_program"] = canonical_prediction == canonical_json(
            parse_bottleneck(reference_program)
        )
        lowered = lower_bottleneck_to_asl(parsed)
        result["lowerable_to_asl"] = True
        result["lowered_asl"] = lowered
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        result["errors"] = [str(error)]
        return result

    semantic = score_asl(reference_asl, lowered, effective_scope)
    for name in (
        "lowerable_to_ccir",
        "type_valid",
        "semantic_lint_valid",
        "executable",
        "dependency_correct",
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
    ):
        result[name] = semantic[name]
    for name in ("paths", "source_facts", "operators", "edges", "semantic_state"):
        result[f"{name}_metrics"] = semantic.get(
            f"{name}_metrics", {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        )
    result["errors"] = semantic["errors"]
    result["predicted_return"] = semantic.get("predicted_return")
    result["reference_return"] = semantic.get("reference_return")
    return result


def analyze_bottleneck_predictions(
    eval_path: str | Path, predictions_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Score saved F4 predictions without any model dependency."""

    references = {row["example_id"]: row for row in read_jsonl(eval_path)}
    predictions = read_jsonl(predictions_path)
    prediction_ids = [row["example_id"] for row in predictions]
    if (
        len(prediction_ids) != len(references)
        or len(set(prediction_ids)) != len(prediction_ids)
        or set(references) != set(prediction_ids)
    ):
        raise ValueError("F4 predictions must cover the exact evaluation identities")
    scored = []
    for prediction in predictions:
        reference = references[prediction["example_id"]]
        metrics = score_bottleneck(
            reference_program=str(reference["target"]),
            reference_asl=str(reference["target_asl"]),
            predicted_program=str(prediction["predicted_program"]),
            effective_scope=dict(reference["effective_scope"]),
        )
        scored.append({**prediction, "metrics": metrics})
    rate_names = (
        "exact_program",
        "parse_valid",
        "lowerable_to_asl",
        "lowerable_to_ccir",
        "type_valid",
        "semantic_lint_valid",
        "executable",
        "dependency_correct",
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
    )
    components = ("paths", "source_facts", "operators", "edges", "semantic_state")
    summary = {
        "schema_version": "ccpu.paper1.e3_bottleneck_evaluation.v1",
        "representation_id": "F4",
        "prediction_count": len(scored),
        "rates": {
            name: sum(bool(row["metrics"].get(name)) for row in scored) / len(scored)
            if scored
            else 0.0
            for name in rate_names
        },
        "component_mean_f1": {
            name: sum(
                float(row["metrics"].get(f"{name}_metrics", {}).get("f1", 0.0))
                for row in scored
            )
            / len(scored)
            if scored
            else 0.0
            for name in components
        },
        "by_dataset": {},
        "eval_sha256": file_sha256(eval_path),
        "predictions_sha256": file_sha256(predictions_path),
    }
    for dataset in sorted({row["dataset"] for row in scored}):
        members = [row for row in scored if row["dataset"] == dataset]
        summary["by_dataset"][dataset] = {
            "count": len(members),
            "rates": {
                name: sum(bool(row["metrics"].get(name)) for row in members) / len(members)
                for name in rate_names
            },
        }
    output = Path(output_dir)
    scored_path = write_jsonl(output / "scored_predictions.jsonl", scored)
    summary["scored_predictions_sha256"] = file_sha256(scored_path)
    write_json(output / "summary.json", summary)
    return summary


def run_bottleneck_condition(
    *,
    eval_path: str | Path,
    model_config: dict[str, Any],
    adapter_path: str | Path,
    adapter_id: str,
    output_dir: str | Path,
    objective_id: str = "L0",
    seed: int = 44017,
    checkpoint_every: int = 5,
    backend_override: Any | None = None,
) -> dict[str, Any]:
    """Generate F4 from NL only and evaluate through deterministic ASL lowering."""

    model = dict(model_config["model"])
    model["adapter_path"] = str(adapter_path)
    model["adapter_id"] = adapter_id
    backend = backend_override
    if backend is None:
        backend = HuggingFaceBackend(
            HuggingFaceGenerationConfig(
                model_id=str(model["model_id"]),
                revision=str(model["revision"]),
                max_new_tokens=int(model.get("max_new_tokens", 1200)),
                device=str(model.get("device", "xpu")),
                dtype=str(model.get("dtype", "float16")),
                use_chat_template=bool(model.get("use_chat_template", True)),
                enable_thinking=bool(model.get("enable_thinking", False)),
                adapter_path=model["adapter_path"],
                adapter_id=model["adapter_id"],
                cached_generation=bool(model.get("cached_generation", True)),
                attn_implementation=model.get("attn_implementation"),
            )
        )
    eval_rows = read_jsonl(eval_path)
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    expected_ids = {row["example_id"] for row in eval_rows}
    if any(
        row.get("representation_id") != "F4"
        or row.get("objective_id") != objective_id
        or row.get("model_id") != backend.model_id
        or row.get("example_id") not in expected_ids
        for row in predictions
    ):
        raise ValueError("resume output does not match the requested F4 run")
    completed = {row["example_id"] for row in predictions}
    for index, row in enumerate(
        [row for row in eval_rows if row["example_id"] not in completed], 1
    ):
        generation = backend.generate(str(row["prompt"]), seed=seed)
        predictions.append(
            {
                "schema_version": "ccpu.paper1.e3_bottleneck_prediction.v1",
                "example_id": row["example_id"],
                "parent_source_id": row["parent_source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "representation_id": "F4",
                "objective_id": objective_id,
                "historical_alias": "M1" if objective_id == "L0" else None,
                "model_id": backend.model_id,
                "adapter_id": adapter_id,
                "seed": seed,
                "generated_text": generation.generated_text,
                "predicted_program": extract_bottleneck(generation.generated_text),
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "wall_time_ns": generation.wall_time_ns,
                "backend_metadata": generation.metadata,
            }
        )
        if index % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint F4/{objective_id}: {len(predictions)}/{len(eval_rows)}")
    write_jsonl(predictions_path, predictions)
    summary = analyze_bottleneck_predictions(eval_path, predictions_path, output)
    summary["run"] = {
        "representation_id": "F4",
        "objective_id": objective_id,
        "seed": seed,
        "model": model,
        "eval_sha256": file_sha256(eval_path),
        "fixed_prompt": True,
        "icl_shots": 0,
        "autonomous_nl_only": True,
    }
    write_json(output / "summary.json", summary)
    return summary
