"""Short-context result reinjection experiment for Paper 1."""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import canonical_json, file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.common.metrics import wilson_interval
from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig

from .direct_failure_audit import extract_direct_endpoint_v2

REINJECTION_CONDITIONS = (
    "value_text",
    "typed_result_text",
    "asl_plus_result_text",
    "oracle_value",
    "corrupted_value",
    "continuation_no_value",
)
REINJECTION_PROTOCOL_ID = "paper1_short_context_reinjection_v1"


def _fraction(value: Any) -> Fraction | None:
    if value is None:
        return None
    try:
        return Fraction(str(value).replace(",", "").replace("$", "").replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _equal(left: Any, right: Any) -> bool:
    a, b = _fraction(left), _fraction(right)
    return a is not None and a == b


def _display(value: Any) -> str:
    parsed = _fraction(value)
    if parsed is None:
        return "UNAVAILABLE"
    return str(parsed.numerator) if parsed.denominator == 1 else f"{parsed.numerator}/{parsed.denominator}"


def _corrupt(value: Any) -> str:
    parsed = _fraction(value)
    if parsed is None:
        raise ValueError("oracle reference is not numeric")
    return _display(parsed + 1)


def reinjection_prompt(
    *, question: str, condition: str, runtime_value: Any, predicted_asl: str
) -> tuple[str, str | None, str]:
    """Render one condition-fixed continuation prompt and injected value."""

    if condition not in REINJECTION_CONDITIONS:
        raise ValueError(f"unsupported reinjection condition: {condition}")
    value = None
    grounding = "NOT_PROVIDED"
    if condition in {"value_text", "typed_result_text", "asl_plus_result_text"}:
        value = _display(runtime_value)
        grounding = "UNVERIFIED"
    elif condition == "oracle_value":
        value = _display(runtime_value)
        grounding = "ORACLE_CONTROL"
    elif condition == "corrupted_value":
        value = _display(runtime_value)
        grounding = "CORRUPTED_CONTROL"

    if condition == "continuation_no_value":
        annotation = "COGNITIVE_PROCESSOR: UNAVAILABLE"
    elif condition == "value_text":
        annotation = f"VALUE: {value}"
    else:
        source = (
            "ORACLE_CONTROL"
            if condition == "oracle_value"
            else "DETERMINISTIC_CORRUPTION_CONTROL"
            if condition == "corrupted_value"
            else "MODEL_GENERATED_ASL"
        )
        annotation = (
            f"VALUE: {value}\nEXECUTION_STATUS: VERIFIED\n"
            f"SEMANTIC_GROUNDING: {grounding}\nSOURCE: {source}"
        )
        if condition == "asl_plus_result_text":
            annotation = f"ASL:\n{predicted_asl}\nEND_ASL\n{annotation}"
    instruction = (
        "Answer the original arithmetic problem using the cognitive-processor annotation. "
        "Execution status means only that the supplied program ran; semantic grounding may "
        "still be wrong. End with one final line in the exact form `Answer: <number>`."
    )
    return f"{instruction}\n\nProblem: {question}\n\nAnnotation:\n{annotation}\n\nResponse:", value, grounding


def _summary(eval_path: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    correct = sum(row["metrics"]["final_answer_correct"] for row in rows)
    metric_names = (
        "runtime_value_correct",
        "final_answer_correct",
        "value_uptake",
        "wrong_value_uptake",
        "wrong_value_corrected",
        "correct_value_override",
        "contradiction",
    )
    counts = {
        name: sum(bool(row["metrics"][name]) for row in rows) for name in metric_names
    }
    return {
        "schema_version": "ccpu.paper1.reinjection_summary.v1",
        "protocol_id": REINJECTION_PROTOCOL_ID,
        "prediction_count": count,
        "eval_sha256": file_sha256(eval_path),
        "counts": counts,
        "rates": {name: value / count if count else 0.0 for name, value in counts.items()},
        "final_answer_wilson_95": wilson_interval(correct, count),
    }


def _run_unlocked(
    *,
    eval_path: str | Path,
    asl_predictions_path: str | Path,
    model_config: dict[str, Any],
    condition: str,
    output_dir: str | Path,
    seed: int,
    checkpoint_every: int,
    backend_override: Any | None,
) -> dict[str, Any]:
    if condition not in REINJECTION_CONDITIONS:
        raise ValueError(f"unsupported reinjection condition: {condition}")
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
    evaluation = read_jsonl(eval_path)
    source_rows = read_jsonl(asl_predictions_path)
    source = {str(row["example_id"]): row for row in source_rows}
    if len(source) != len(source_rows):
        raise ValueError("reinjection source contains duplicate identities")
    if set(source) != {str(row["example_id"]) for row in evaluation}:
        raise ValueError("reinjection requires complete identity-matched ASL predictions")
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    prediction_ids = [str(row["example_id"]) for row in predictions]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("reinjection resume contains duplicate identities")
    eval_by_id = {str(row["example_id"]): row for row in evaluation}
    if any(
        example_id not in eval_by_id
        or row.get("condition") != condition
        or row.get("question_sha256") != eval_by_id[example_id].get("question_sha256")
        for example_id, row in zip(prediction_ids, predictions, strict=True)
    ):
        raise ValueError("reinjection resume does not match the requested condition")
    completed = {str(row["example_id"]) for row in predictions}
    for index, row in enumerate(evaluation, 1):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        asl = source[example_id]
        if asl.get("question_sha256") != row.get("question_sha256"):
            raise ValueError(f"reinjection question hash mismatch: {example_id}")
        runtime_value = asl.get("metrics", {}).get("predicted_return")
        if condition == "oracle_value":
            injected_source = row["reference_return"]
        elif condition == "corrupted_value":
            injected_source = _corrupt(row["reference_return"])
        else:
            injected_source = runtime_value
        prompt, injected_value, grounding = reinjection_prompt(
            question=str(row["question"]),
            condition=condition,
            runtime_value=injected_source,
            predicted_asl=str(asl.get("predicted_asl", "")),
        )
        generation = backend.generate(prompt, seed=seed)
        complete = generation.generated_tokens < int(model["max_new_tokens"])
        answer = extract_direct_endpoint_v2(
            generation.generated_text, allow_bare_terminal=complete
        )
        runtime_correct = _equal(runtime_value, row["reference_return"])
        final_correct = _equal(answer, row["reference_return"])
        uptake = injected_value is not None and _equal(answer, injected_value)
        injected_correct = injected_value is not None and _equal(
            injected_value, row["reference_return"]
        )
        predictions.append(
            {
                "schema_version": "ccpu.paper1.reinjection_prediction.v1",
                "protocol_id": REINJECTION_PROTOCOL_ID,
                "condition": condition,
                "example_id": example_id,
                "parent_example_id": row.get("parent_example_id", example_id),
                "question_sha256": row["question_sha256"],
                "reference_return": row["reference_return"],
                "runtime_value": runtime_value,
                "injected_value": injected_value,
                "semantic_grounding": grounding,
                "predicted_answer": answer,
                "generated_text": generation.generated_text,
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "wall_time_ns": generation.wall_time_ns,
                "backend_metadata": generation.metadata,
                "metrics": {
                    "runtime_value_correct": runtime_correct,
                    "final_answer_correct": final_correct,
                    "value_uptake": uptake,
                    "wrong_value_uptake": uptake and not injected_correct,
                    "wrong_value_corrected": final_correct and injected_value is not None and not injected_correct,
                    "correct_value_override": injected_correct and not final_correct,
                    "contradiction": answer is not None and not final_correct and not uptake,
                },
            }
        )
        if index % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
    predictions_path = write_jsonl(predictions_path, predictions)
    summary = _summary(eval_path, predictions)
    summary["condition"] = condition
    summary["inputs"] = {
        "eval_sha256": file_sha256(eval_path),
        "asl_predictions_sha256": file_sha256(asl_predictions_path),
    }
    summary["predictions_sha256"] = file_sha256(predictions_path)
    write_json(output / "summary.json", summary)
    return summary


def run_reinjection(
    *,
    eval_path: str | Path,
    asl_predictions_path: str | Path,
    model_config: dict[str, Any],
    condition: str,
    output_dir: str | Path,
    seed: int = 55017,
    checkpoint_every: int = 5,
    backend_override: Any | None = None,
) -> dict[str, Any]:
    """Run one resumable reinjection condition under an exclusive output lock."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"reinjection output is already locked: {lock}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json({"pid": os.getpid(), "condition": condition}))
        return _run_unlocked(
            eval_path=eval_path,
            asl_predictions_path=asl_predictions_path,
            model_config=model_config,
            condition=condition,
            output_dir=output,
            seed=seed,
            checkpoint_every=checkpoint_every,
            backend_override=backend_override,
        )
    finally:
        lock.unlink(missing_ok=True)
