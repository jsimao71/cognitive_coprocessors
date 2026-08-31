"""Closed-loop evaluation for clause-local ASL compilation with executed state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

from .asl_pilot_data import incremental_prompt
from .asl_pilot_eval import analyze_asl_predictions, extract_asl, score_asl_delta
from .generation import HuggingFaceBackend, HuggingFaceGenerationConfig


def _transition_rates(traces: list[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        return {}
    metric_names = {
        "parse_valid_delta": "parse_valid",
        "lowerable_delta": "lowerable_to_ccir",
        "type_valid_delta": "type_valid",
        "executable_delta": "fully_resolved",
    }
    rates = {
        output_name: sum(bool(trace["metrics"][metric]) for trace in traces) / len(traces)
        for output_name, metric in metric_names.items()
    }
    rates.update(
        {
            "operator_exact": sum(
                trace["metrics"]["operators_metrics"]["f1"] == 1 for trace in traces
            )
            / len(traces),
            "path_exact": sum(trace["metrics"]["paths_metrics"]["f1"] == 1 for trace in traces)
            / len(traces),
            "source_fact_exact": sum(
                trace["metrics"]["source_facts_metrics"]["f1"] == 1 for trace in traces
            )
            / len(traces),
            "dependency_exact": sum(
                trace["metrics"]["edges_metrics"]["f1"] == 1 for trace in traces
            )
            / len(traces),
            "state_delta_exact": sum(
                trace["metrics"]["semantic_state_metrics"]["f1"] == 1 for trace in traces
            )
            / len(traces),
        }
    )
    return rates


def _rescore_transition_traces(
    prediction: dict[str, Any],
    row: dict[str, Any],
    *,
    state_mode: str,
    context_mode: str,
) -> None:
    mappings = {int(item["part_id"]): item for item in row["part_mappings"]}
    predicted_statements: list[str] = []
    gold_statements: list[str] = []
    for trace in prediction["part_traces"]:
        part_id = int(trace["part_id"])
        context = predicted_statements if state_mode == "predicted" else gold_statements
        trace["metrics"] = score_asl_delta(
            "\n".join(mappings[part_id]["asl"]),
            trace["predicted_delta"],
            row["effective_scope"],
            context_asl="\n".join(context),
        )
        if trace["accepted"] and trace["predicted_delta"]:
            predicted_statements.extend(trace["predicted_delta"].splitlines())
        gold_statements.extend(mappings[part_id]["asl"])
    prediction["suite"] = f"incremental_{state_mode}_state_{context_mode}"
    prediction["condition"] = f"lora_incremental_{state_mode}_state_{context_mode}"


def run_incremental_program(
    row: dict[str, Any],
    backend: Any,
    *,
    seed: int,
    state_mode: str = "predicted",
    context_mode: str = "causal",
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Generate and execute deltas in source order, halting on an invalid prefix."""

    parts = {int(part["part_id"]): part for part in row["parts"]}
    if state_mode not in {"predicted", "oracle"}:
        raise ValueError(f"unsupported incremental state mode: {state_mode}")
    empty_state: dict[str, Any] = {"values": {}, "unresolved": []}
    state = empty_state
    statements: list[str] = []
    gold_statements: list[str] = []
    traces = []
    stopped_reason = None
    for mapping in sorted(row["part_mappings"], key=lambda item: int(item["part_id"])):
        part_id = int(mapping["part_id"])
        prompt = incremental_prompt(row, parts[part_id], state, context_mode=context_mode)
        generation = backend.generate(prompt, seed=seed + part_id)
        delta = extract_asl(generation.generated_text)
        context_statements = statements if state_mode == "predicted" else gold_statements
        context_asl = "\n".join(context_statements)
        candidate = "\n".join([*context_statements, delta]) if delta else context_asl
        validation = validate_asl(candidate, effective_scope=row["effective_scope"])
        accepted = bool(delta) and all(
            validation[key] for key in ("syntax_verified", "lower_verified", "type_verified")
        )
        delta_metrics = score_asl_delta(
            "\n".join(mapping["asl"]),
            delta,
            row["effective_scope"],
            context_asl=context_asl,
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
                "metrics": delta_metrics,
            }
        )
        if not accepted and state_mode == "predicted":
            stopped_reason = (
                "; ".join(validation["errors"])
                if validation["errors"]
                else "no ASL delta extracted"
            )
            break
        if delta:
            statements.extend(delta.splitlines())
        gold_statements.extend(mapping["asl"])
        state_source = statements if state_mode == "predicted" else gold_statements
        state_validation = validate_asl(
            "\n".join(state_source), effective_scope=row["effective_scope"]
        )
        if state_validation["type_verified"]:
            workspace = state_validation["execution"]["workspace"][
                str(row["effective_scope"]["id"])
            ]
            state = {
                "values": workspace["values"],
                "returned": workspace["returned"],
                "unresolved": state_validation["execution"]["unresolved"],
            }
        else:
            state = empty_state
    return "\n".join(statements), traces, stopped_reason


def run_asl_incremental(
    *,
    programs_path: str | Path,
    model_config: dict[str, Any],
    output_dir: str | Path,
    seed: int = 44017,
    checkpoint_every: int = 5,
    state_mode: str = "predicted",
    context_mode: str = "causal",
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
                "suite": f"incremental_{state_mode}_state_{context_mode}",
                "reference_asl": row["asl"],
                "effective_scope": row["effective_scope"],
            }
        )
        if example_id in completed:
            continue
        predicted_asl, traces, stopped_reason = run_incremental_program(
            row,
            backend,
            seed=seed,
            state_mode=state_mode,
            context_mode=context_mode,
        )
        predictions.append(
            {
                "schema_version": "ccpu.paper1.asl_incremental_prediction.v1",
                "example_id": example_id,
                "parent_source_id": row["source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "suite": f"incremental_{state_mode}_state_{context_mode}",
                "condition": f"lora_incremental_{state_mode}_state_{context_mode}",
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
            print(f"checkpoint incremental {state_mode}: {len(predictions)}/{len(programs)}")
    programs_by_example = {
        f"asl-inc-eval-{fingerprint(row['record_sha256'], 12)}": row for row in programs
    }
    for prediction in predictions:
        _rescore_transition_traces(
            prediction,
            programs_by_example[prediction["example_id"]],
            state_mode=state_mode,
            context_mode=context_mode,
        )
    references_path = write_jsonl(output / "references.jsonl", references)
    write_jsonl(predictions_path, predictions)
    report = analyze_asl_predictions(references_path, predictions_path, output)
    traces = [trace for row in predictions for trace in row["part_traces"]]
    completed_fractions = [row["completed_parts"] / row["part_count"] for row in predictions]
    traces_by_dataset = {
        dataset: [
            trace
            for row in predictions
            if row["dataset"] == dataset
            for trace in row["part_traces"]
        ]
        for dataset in sorted({str(row["dataset"]) for row in predictions})
    }
    report["incremental"] = {
        "state_mode": state_mode,
        "context_mode": context_mode,
        "program_count": len(predictions),
        "completed_program_count": sum(
            row["completed_parts"] == row["part_count"] for row in predictions
        ),
        "accepted_transition_count": sum(trace["accepted"] for trace in traces),
        "attempted_transition_count": len(traces),
        "total_reference_transition_count": sum(row["part_count"] for row in predictions),
        "fail_closed_program_count": sum(row["stopped_reason"] is not None for row in predictions),
        "accepted_per_attempted": sum(trace["accepted"] for trace in traces) / len(traces),
        "mean_completed_fraction": sum(completed_fractions) / len(completed_fractions),
        "transition_rates": _transition_rates(traces),
        "transition_by_dataset": {
            dataset: {"count": len(dataset_traces), "rates": _transition_rates(dataset_traces)}
            for dataset, dataset_traces in traces_by_dataset.items()
        },
    }
    report["run"] = {
        "condition": f"lora_incremental_{state_mode}_state_{context_mode}",
        "seed": seed,
        "model": model,
        "programs_sha256": file_sha256(programs_path),
        "references_sha256": file_sha256(references_path),
        "predicted_state_feedback": state_mode == "predicted",
        "fail_closed_invalid_delta": True,
        "state_mode": state_mode,
        "context_mode": context_mode,
    }
    write_json(output / "summary.json", report)
    return report
