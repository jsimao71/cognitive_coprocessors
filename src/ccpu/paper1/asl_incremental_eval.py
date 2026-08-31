"""Closed-loop evaluation for clause-local ASL compilation with executed state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

from .asl_pilot_data import incremental_prompt
from .asl_pilot_eval import analyze_asl_predictions, extract_asl
from .generation import HuggingFaceBackend, HuggingFaceGenerationConfig


def run_incremental_program(
    row: dict[str, Any], backend: Any, *, seed: int
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Generate and execute deltas in source order, halting on an invalid prefix."""

    parts = {int(part["part_id"]): part for part in row["parts"]}
    state: dict[str, Any] = {"values": {}, "unresolved": []}
    statements: list[str] = []
    traces = []
    stopped_reason = None
    for mapping in sorted(row["part_mappings"], key=lambda item: int(item["part_id"])):
        part_id = int(mapping["part_id"])
        prompt = incremental_prompt(row, parts[part_id], state)
        generation = backend.generate(prompt, seed=seed + part_id)
        delta = extract_asl(generation.generated_text)
        candidate = "\n".join([*statements, delta]) if delta else "\n".join(statements)
        validation = validate_asl(candidate, effective_scope=row["effective_scope"])
        accepted = bool(delta) and all(
            validation[key] for key in ("syntax_verified", "lower_verified", "type_verified")
        )
        traces.append(
            {
                "part_id": part_id,
                "generated_text": generation.generated_text,
                "predicted_delta": delta,
                "accepted": accepted,
                "validation_errors": validation["errors"],
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "wall_time_ns": generation.wall_time_ns,
            }
        )
        if not accepted:
            stopped_reason = (
                "; ".join(validation["errors"])
                if validation["errors"]
                else "no ASL delta extracted"
            )
            break
        statements.extend(delta.splitlines())
        workspace = validation["execution"]["workspace"][str(row["effective_scope"]["id"])]
        state = {
            "values": workspace["values"],
            "returned": workspace["returned"],
            "unresolved": validation["execution"]["unresolved"],
        }
    return "\n".join(statements), traces, stopped_reason


def run_asl_incremental(
    *,
    programs_path: str | Path,
    model_config: dict[str, Any],
    output_dir: str | Path,
    seed: int = 44017,
    checkpoint_every: int = 5,
) -> dict[str, Any]:
    """Evaluate an incremental adapter with predicted-state feedback."""

    model = dict(model_config["model"])
    model["adapter_path"] = model_config["adapter_path"]
    model["adapter_id"] = model_config["adapter_id"]
    backend = HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_new_tokens=int(model.get("max_new_tokens", 192)),
            device=str(model.get("device", "xpu")),
            dtype=str(model.get("dtype", "float16")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            adapter_path=model["adapter_path"],
            adapter_id=model["adapter_id"],
            cached_generation=True,
        )
    )
    programs = read_jsonl(programs_path)
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed = {row["example_id"] for row in predictions}
    references = []
    for row in programs:
        example_id = f"asl-inc-eval-{fingerprint(row['record_sha256'], 12)}"
        references.append(
            {
                "schema_version": "ccpu.paper1.asl_eval.v1",
                "example_id": example_id,
                "parent_source_id": row["source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "suite": "incremental_original",
                "reference_asl": row["asl"],
                "effective_scope": row["effective_scope"],
            }
        )
        if example_id in completed:
            continue
        predicted_asl, traces, stopped_reason = run_incremental_program(row, backend, seed=seed)
        predictions.append(
            {
                "schema_version": "ccpu.paper1.asl_incremental_prediction.v1",
                "example_id": example_id,
                "parent_source_id": row["source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "suite": "incremental_original",
                "condition": "lora_incremental",
                "shots": 0,
                "model_id": backend.model_id,
                "seed": seed,
                "predicted_asl": predicted_asl,
                "part_traces": traces,
                "completed_parts": sum(trace["accepted"] for trace in traces),
                "part_count": len(row["part_mappings"]),
                "stopped_reason": stopped_reason,
            }
        )
        if len(predictions) % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint incremental: {len(predictions)}/{len(programs)}")
    references_path = write_jsonl(output / "references.jsonl", references)
    write_jsonl(predictions_path, predictions)
    report = analyze_asl_predictions(references_path, predictions_path, output)
    traces = [trace for row in predictions for trace in row["part_traces"]]
    report["incremental"] = {
        "program_count": len(predictions),
        "completed_program_count": sum(row["stopped_reason"] is None for row in predictions),
        "accepted_transition_count": sum(trace["accepted"] for trace in traces),
        "attempted_transition_count": len(traces),
        "total_reference_transition_count": sum(row["part_count"] for row in predictions),
        "fail_closed_program_count": sum(row["stopped_reason"] is not None for row in predictions),
    }
    report["run"] = {
        "condition": "lora_incremental",
        "seed": seed,
        "model": model,
        "programs_sha256": file_sha256(programs_path),
        "references_sha256": file_sha256(references_path),
        "predicted_state_feedback": True,
        "fail_closed_invalid_delta": True,
    }
    write_json(output / "summary.json", report)
    return report
