"""F3 model generation, runtime ablations, and matched ASL scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.paper1.asl_pilot_eval import score_asl
from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig

from .parser import extract_f3_program
from .runtime import validate_f3_program


def score_f3(
    *,
    reference_program: str,
    reference_asl: str,
    predicted_program: str,
    question: str,
    source_context: dict[str, Any] | None,
    effective_scope: dict[str, Any],
    primary_mode: str = "r1",
    reference_status: str = "accepted",
) -> dict[str, Any]:
    validations = {
        mode: validate_f3_program(
            predicted_program,
            question=question,
            source_context=source_context,
            effective_scope=effective_scope,
            mode=mode,
        )
        for mode in ("r0", "r1", "r2")
    }
    primary = validations[primary_mode]
    result: dict[str, Any] = {
        "reference_status": reference_status,
        "exact_program": reference_status == "accepted"
        and "\n".join(reference_program.split()) == "\n".join(predicted_program.split()),
        "parse_valid": primary["parse_valid"],
        "evidence_valid": primary["evidence_valid"],
        "lowerable_to_ccir": primary["lowerable"],
        "type_valid": primary["type_valid"],
        "executable": primary["executable"],
        "dependency_correct": False,
        "semantic_return_equivalent": False,
        "semantic_state_equivalent": False,
        "final_answer_correct": False,
        "errors": primary["errors"],
        "lowered_asl": primary["lowered_asl"],
        "primary_mode": primary_mode,
        "runtime_modes": {
            mode: {
                key: validation[key]
                for key in (
                    "parse_valid",
                    "evidence_valid",
                    "lowerable",
                    "type_valid",
                    "executable",
                )
            }
            for mode, validation in validations.items()
        },
    }
    for mode, validation in validations.items():
        result[f"{mode}_executable"] = validation["executable"]
        result[f"{mode}_final_answer_correct"] = False
        if not validation["lowerable"]:
            continue
        semantic = score_asl(reference_asl, validation["lowered_asl"], effective_scope)
        result[f"{mode}_final_answer_correct"] = semantic["final_answer_correct"]
        if mode != primary_mode:
            continue
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
        result["predicted_return"] = semantic.get("predicted_return")
        result["reference_return"] = semantic.get("reference_return")
    return result


def analyze_f3_predictions(
    eval_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    primary_mode: str = "r1",
) -> dict[str, Any]:
    references = {row["example_id"]: row for row in read_jsonl(eval_path)}
    scored = []
    for prediction in read_jsonl(predictions_path):
        reference = references[prediction["example_id"]]
        metrics = score_f3(
            reference_program=reference["reference_program"],
            reference_asl=reference["reference_asl"],
            predicted_program=prediction["predicted_program"],
            question=reference["question"],
            source_context=reference.get("source_context"),
            effective_scope=reference["effective_scope"],
            primary_mode=primary_mode,
            reference_status=reference.get("reference_status", "accepted"),
        )
        scored.append({**prediction, "metrics": metrics})
    metric_names = (
        "exact_program",
        "parse_valid",
        "evidence_valid",
        "lowerable_to_ccir",
        "type_valid",
        "executable",
        "dependency_correct",
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
        "r0_final_answer_correct",
        "r1_final_answer_correct",
        "r2_final_answer_correct",
    )
    components = ("paths", "source_facts", "operators", "edges", "semantic_state")
    summary = {
        "schema_version": "ccpu.paper1.f3.evaluation.v1",
        "prediction_count": len(scored),
        "reference_status_counts": {
            status: sum(row["metrics"]["reference_status"] == status for row in scored)
            for status in sorted({row["metrics"]["reference_status"] for row in scored})
        },
        "primary_mode": primary_mode,
        "rates": {
            name: sum(bool(row["metrics"].get(name)) for row in scored) / len(scored)
            if scored
            else 0.0
            for name in metric_names
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
                for name in metric_names
            },
        }
    output = Path(output_dir)
    scored_path = write_jsonl(output / "scored_predictions.jsonl", scored)
    summary["scored_predictions_sha256"] = file_sha256(scored_path)
    write_json(output / "summary.json", summary)
    return summary


def run_f3_condition(
    *,
    eval_path: str | Path,
    model_config: dict[str, Any],
    output_dir: str | Path,
    primary_mode: str = "r1",
    seed: int = 44017,
    checkpoint_every: int = 5,
) -> dict[str, Any]:
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
                "schema_version": "ccpu.paper1.f3.prediction.v1",
                "example_id": row["example_id"],
                "parent_source_id": row["parent_source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "condition": "f3",
                "model_id": backend.model_id,
                "adapter_id": model["adapter_id"],
                "seed": seed,
                "generated_text": generation.generated_text,
                "predicted_program": extract_f3_program(generation.generated_text),
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "wall_time_ns": generation.wall_time_ns,
                "backend_metadata": generation.metadata,
            }
        )
        if index % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint f3: {len(predictions)}/{len(eval_rows)}")
    write_jsonl(predictions_path, predictions)
    summary = analyze_f3_predictions(
        eval_path, predictions_path, output, primary_mode=primary_mode
    )
    summary["run"] = {
        "condition": "f3",
        "primary_mode": primary_mode,
        "seed": seed,
        "model": model,
        "eval_sha256": file_sha256(eval_path),
        "fixed_prompt": True,
        "fixed_icl": True,
    }
    write_json(output / "summary.json", summary)
    return summary
