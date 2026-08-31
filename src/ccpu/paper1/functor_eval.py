"""Matched model execution and scoring for Paper 1 functor conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl

from .asl_pilot_eval import score_asl
from .functor_runtime import extract_functor_program, validate_functor_program
from .generation import HuggingFaceBackend, HuggingFaceGenerationConfig


def score_functor(
    *,
    reference_program: str,
    reference_asl: str,
    predicted_program: str,
    condition: str,
    effective_scope: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_functor_program(
        predicted_program, condition, effective_scope=effective_scope
    )
    result = {
        "exact_program": "\n".join(reference_program.split())
        == "\n".join(predicted_program.split()),
        "parse_valid": validation["parse_valid"],
        "lowerable_to_ccir": validation["lowerable"],
        "type_valid": validation["type_valid"],
        "executable": validation["executable"],
        "dependency_correct": False,
        "semantic_return_equivalent": False,
        "semantic_state_equivalent": False,
        "final_answer_correct": False,
        "errors": validation["errors"],
        "lowered_asl": validation["lowered_asl"],
    }
    if not validation["lowerable"]:
        return result
    semantic = score_asl(reference_asl, validation["lowered_asl"], effective_scope)
    for name in (
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
    result["semantic_errors"] = semantic["errors"]
    result["predicted_return"] = semantic.get("predicted_return")
    result["reference_return"] = semantic.get("reference_return")
    return result


def analyze_functor_predictions(
    eval_path: str | Path, predictions_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    references = {row["example_id"]: row for row in read_jsonl(eval_path)}
    predictions = read_jsonl(predictions_path)
    scored = []
    for prediction in predictions:
        reference = references[prediction["example_id"]]
        metrics = score_functor(
            reference_program=reference["reference_program"],
            reference_asl=reference["reference_asl"],
            predicted_program=prediction["predicted_program"],
            condition=prediction["condition"],
            effective_scope=reference["effective_scope"],
        )
        scored.append({**prediction, "metrics": metrics})
    rate_names = (
        "exact_program",
        "parse_valid",
        "lowerable_to_ccir",
        "type_valid",
        "executable",
        "dependency_correct",
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
    )
    components = ("paths", "source_facts", "operators", "edges", "semantic_state")
    summary = {
        "schema_version": "ccpu.paper1.functor_evaluation.v1",
        "prediction_count": len(scored),
        "condition": scored[0]["condition"] if scored else None,
        "rates": {
            name: sum(bool(row["metrics"].get(name)) for row in scored) / len(scored)
            if scored
            else 0.0
            for name in rate_names
        },
        "component_mean_f1": {
            name: sum(
                float(row["metrics"].get(f"{name}_metrics", {}).get("f1", 0.0)) for row in scored
            )
            / len(scored)
            if scored
            else 0.0
            for name in components
        },
        "eval_sha256": file_sha256(eval_path),
        "predictions_sha256": file_sha256(predictions_path),
    }
    for dataset in sorted({row["dataset"] for row in scored}):
        members = [row for row in scored if row["dataset"] == dataset]
        summary.setdefault("by_dataset", {})[dataset] = {
            "count": len(members),
            "rates": {
                name: sum(bool(row["metrics"].get(name)) for row in members) / len(members)
                for name in rate_names
            },
        }
    scored_path = write_jsonl(Path(output_dir) / "scored_predictions.jsonl", scored)
    summary["scored_predictions_sha256"] = file_sha256(scored_path)
    write_json(Path(output_dir) / "summary.json", summary)
    return summary


def run_functor_condition(
    *,
    eval_path: str | Path,
    model_config: dict[str, Any],
    condition: str,
    output_dir: str | Path,
    seed: int = 44017,
    checkpoint_every: int = 5,
) -> dict[str, Any]:
    if condition not in {"f1", "f2"}:
        raise ValueError(f"unsupported functor condition: {condition}")
    model = dict(model_config["model"])
    model["adapter_path"] = model_config["adapter_path"]
    model["adapter_id"] = model_config["adapter_id"]
    backend = HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_new_tokens=int(model.get("max_new_tokens", 384)),
            device=str(model.get("device", "xpu")),
            dtype=str(model.get("dtype", "float16")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            adapter_path=model["adapter_path"],
            adapter_id=model["adapter_id"],
            cached_generation=True,
        )
    )
    eval_rows = read_jsonl(eval_path)
    if any(row["condition"] != condition for row in eval_rows):
        raise ValueError("evaluation rows do not match requested condition")
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed = {row["example_id"] for row in predictions}
    for index, row in enumerate(
        [row for row in eval_rows if row["example_id"] not in completed], 1
    ):
        generation = backend.generate(row["prompt"], seed=seed)
        predictions.append(
            {
                "schema_version": "ccpu.paper1.functor_prediction.v1",
                "example_id": row["example_id"],
                "parent_source_id": row["parent_source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "condition": condition,
                "model_id": backend.model_id,
                "adapter_id": model["adapter_id"],
                "seed": seed,
                "generated_text": generation.generated_text,
                "predicted_program": extract_functor_program(generation.generated_text, condition),
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "wall_time_ns": generation.wall_time_ns,
                "backend_metadata": generation.metadata,
            }
        )
        if index % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint {condition}: {len(predictions)}/{len(eval_rows)}")
    write_jsonl(predictions_path, predictions)
    summary = analyze_functor_predictions(eval_path, predictions_path, output)
    summary["run"] = {
        "condition": condition,
        "seed": seed,
        "model": model,
        "eval_sha256": file_sha256(eval_path),
        "fixed_prompt": True,
        "fixed_icl": True,
    }
    write_json(output / "summary.json", summary)
    return summary


def compare_functor_conditions(
    *, f0_summary: str | Path, f1_summary: str | Path, f2_summary: str | Path, output: str | Path
) -> dict[str, Any]:
    import json

    summaries = {}
    scored = {}
    for name, path in (("F0", f0_summary), ("F1", f1_summary), ("F2", f2_summary)):
        summary_path = Path(path)
        summaries[name] = json.loads(summary_path.read_text(encoding="utf-8"))
        scored_path = summary_path.parent / "scored_predictions.jsonl"
        scored[name] = {
            (str(row["dataset"]), str(row["parent_source_id"])): row
            for row in read_jsonl(scored_path)
        }
    identity_sets = {name: set(rows) for name, rows in scored.items()}
    if len(identity_sets["F0"]) != 25 or not (
        identity_sets["F0"] == identity_sets["F1"] == identity_sets["F2"]
    ):
        raise ValueError("F0/F1/F2 must contain the identical frozen 25 source identities")
    metrics = (
        "parse_valid",
        "executable",
        "dependency_correct",
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
    )
    paired = {}
    for candidate, baseline in (("F1", "F0"), ("F2", "F0"), ("F2", "F1")):
        outcomes = []
        for identity in sorted(identity_sets["F0"]):
            candidate_correct = bool(scored[candidate][identity]["metrics"]["final_answer_correct"])
            baseline_correct = bool(scored[baseline][identity]["metrics"]["final_answer_correct"])
            outcomes.append((candidate_correct, baseline_correct))
        paired[f"{candidate}_minus_{baseline}"] = {
            "answer_rate_delta": (
                summaries[candidate]["rates"]["final_answer_correct"]
                - summaries[baseline]["rates"]["final_answer_correct"]
            ),
            "candidate_only_correct": sum(left and not right for left, right in outcomes),
            "baseline_only_correct": sum(right and not left for left, right in outcomes),
            "both_correct": sum(left and right for left, right in outcomes),
            "both_incorrect": sum(not left and not right for left, right in outcomes),
        }
    comparison = {
        "schema_version": "ccpu.paper1.functor_comparison.v1",
        "conditions": {
            name: {metric: summary["rates"].get(metric) for metric in metrics}
            for name, summary in summaries.items()
        },
        "frozen_identity_count": len(identity_sets["F0"]),
        "identical_frozen_source_ids": True,
        "paired_answer_comparisons": paired,
        "interpretation_gate": {
            "semantic_decomposition_supported": (
                summaries["F2"]["rates"]["final_answer_correct"]
                > summaries["F1"]["rates"]["final_answer_correct"]
                and summaries["F2"]["rates"]["final_answer_correct"]
                > summaries["F0"]["rates"]["final_answer_correct"]
            ),
            "rule": "attribute a gain to semantic decomposition only when F2 exceeds both F0 and F1",
        },
        "input_sha256": {
            "F0": file_sha256(f0_summary),
            "F1": file_sha256(f1_summary),
            "F2": file_sha256(f2_summary),
        },
    }
    write_json(output, comparison)
    return comparison
