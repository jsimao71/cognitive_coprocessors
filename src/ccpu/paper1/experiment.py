"""Run Paper 1 conditions and emit prediction plus component-trace records."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ccpu.common.artifacts import canonical_json
from ccpu.common.schema import DetectionCandidate, GenerationResult, TraceStage, TraceStatus

from .arithmetic import ArithmeticNormalizationError, ArithmeticNormalizer
from .block_protocol import analyze_block_protocol
from .dataset import ArithmeticExample, reference_answer
from .evaluate import answers_equal, extract_answer
from .generation import HuggingFaceBackend, ScriptedProtocolBackend
from .prompts import (
    BLOCK_CONDITIONS,
    CONDITIONS,
    CORE_CONDITIONS,
    SEEDED_OPEN_BLOCK_CONDITIONS,
    SEEDED_WRAPPER_BLOCK_CONDITIONS,
    condition_prompt,
)
from .recognizer import (
    CalculatorBlockRecognizer,
    ExplicitCalculatorToolRecognizer,
    NormalizedArithmeticRecognizer,
)
from .reflex import (
    build_calculator_block_runtime,
    build_explicit_tool_runtime,
    build_normalized_reflex_runtime,
    build_oracle_runtime,
    build_reflex_runtime,
)


def _controller(example: ArithmeticExample, condition: str, run_id: str):
    if condition == "explicit_tool":
        return build_explicit_tool_runtime(run_id=run_id)
    if condition == "reflex":
        return build_reflex_runtime(run_id=run_id)
    if condition == "normalized_reflex":
        return build_normalized_reflex_runtime(run_id=run_id)
    if condition in BLOCK_CONDITIONS:
        return build_calculator_block_runtime(run_id=run_id)
    if condition == "oracle" and example.expression is not None:
        return build_oracle_runtime(example.expression, run_id=run_id)
    return None


def _gold_instructions(example: ArithmeticExample):
    if example.expression is None:
        return None
    candidate = DetectionCandidate(
        candidate_id=f"gold:{example.example_id}",
        family="compute",
        raw_text=example.expression,
        start_offset=0,
        end_offset=len(example.expression),
        detector="gold",
    )
    return ArithmeticNormalizer().normalize(candidate).payload.get("instructions")


def _expression_exposed(text: str, example: ArithmeticExample, gold_instructions) -> bool | None:
    if example.expression is None or gold_instructions is None:
        return None
    compact = re.sub(r"\s+", "", text)
    if re.sub(r"\s+", "", example.expression) in compact:
        return True
    detectors = (
        NormalizedArithmeticRecognizer(),
        ExplicitCalculatorToolRecognizer(),
        CalculatorBlockRecognizer(),
    )
    candidates = [candidate for detector in detectors for candidate in detector.feed(text)]
    from .surface import ArithmeticSurfaceNormalizer

    normalizer = ArithmeticSurfaceNormalizer()
    for candidate in candidates:
        try:
            if normalizer.normalize(candidate).payload.get("instructions") == gold_instructions:
                return True
        except ArithmeticNormalizationError:
            continue
    return False


def _invocation_overhead_chars(condition: str, recognized: bool) -> int:
    if not recognized:
        return 0
    if condition == "explicit_tool":
        return len("<tool:calculator></tool>")
    if condition in BLOCK_CONDITIONS:
        return len("```calculator\n\n```")
    if condition in {"reflex", "normalized_reflex", "oracle"}:
        return 1
    return 0


def _prediction(
    example: ArithmeticExample,
    *,
    condition: str,
    model_id: str,
    seed: int,
    generation: GenerationResult,
    controller,
    protocol_text: str | None = None,
) -> dict[str, Any]:
    trace = controller.trace if controller else []
    state = controller.state if controller else []
    candidates = sum(event.stage == TraceStage.DETECTION for event in trace)
    normalization_failures = sum(
        event.stage == TraceStage.NORMALIZATION and event.status == TraceStatus.REJECTED
        for event in trace
    )
    engine_failures = sum(
        event.stage == TraceStage.EXECUTION and event.status == TraceStatus.FAILED
        for event in trace
    )
    engine_time_ns = sum(
        event.duration_ns or 0 for event in trace if event.stage == TraceStage.EXECUTION
    )
    trace_bytes = sum(len(canonical_json(event.to_dict()).encode()) + 1 for event in trace)
    state_bytes = sum(len(canonical_json(item.to_dict()).encode()) + 1 for item in state)
    normalization_accepts = sum(
        event.stage == TraceStage.NORMALIZATION and event.status == TraceStatus.ACCEPTED
        for event in trace
    )
    executions = sum(event.stage == TraceStage.EXECUTION for event in trace)
    engine_successes = sum(
        event.stage == TraceStage.EXECUTION and event.status == TraceStatus.SUCCEEDED
        for event in trace
    )
    reinjection_failures = sum(
        event.stage == TraceStage.REINJECTION and event.status == TraceStatus.FAILED
        for event in trace
    )
    reinjection_successes = sum(
        event.stage == TraceStage.REINJECTION and event.status == TraceStatus.SUCCEEDED
        for event in trace
    )
    predicted_answer = extract_answer(generation.rendered_text)
    engine_answer = state[-1].result.display if state else None
    normalized_expression = (
        str(state[-1].request.payload.get("canonical_expression")) if state else None
    )
    normalized_instructions = state[-1].request.payload.get("instructions") if state else None
    gold_instructions = _gold_instructions(example)
    recognized = candidates > 0
    selection_correct = (
        normalized_instructions == gold_instructions
        if normalized_instructions is not None and gold_instructions is not None
        else None
    )
    block_metrics: dict[str, Any] = {}
    if condition in BLOCK_CONDITIONS:
        protocol = protocol_text if protocol_text is not None else generation.generated_text
        block_metrics = {
            "block_protocol_evaluated": True,
            "block_wrapper_seeded": condition
            in (*SEEDED_OPEN_BLOCK_CONDITIONS, *SEEDED_WRAPPER_BLOCK_CONDITIONS),
            "block_open_model_emitted": analyze_block_protocol(
                generation.generated_text, example.expression
            )["block_open"],
            **analyze_block_protocol(protocol, example.expression),
            "block_execution": engine_successes > 0,
        }
    return {
        "schema_version": "ccpu.paper1.prediction.v2",
        "example_id": example.example_id,
        "task_kind": example.task_kind,
        "model_id": model_id,
        "condition": condition,
        "seed": seed,
        "generated_text": generation.generated_text,
        "rendered_text": generation.rendered_text,
        "predicted_answer": predicted_answer,
        "prompt_tokens": generation.prompt_tokens,
        "generated_tokens": generation.generated_tokens,
        "reinjected_tokens": generation.reinjected_tokens,
        "model_calls": generation.model_calls,
        "wall_time_ns": generation.wall_time_ns,
        "engine_time_ns": engine_time_ns,
        "trace_bytes": trace_bytes,
        "state_bytes": state_bytes,
        "candidates": candidates,
        "expression_exposed": _expression_exposed(
            generation.generated_text, example, gold_instructions
        ),
        "recognized": recognized,
        "selection_correct": selection_correct,
        "normalization_accepts": normalization_accepts,
        "normalization_failures": normalization_failures,
        "normalization_correct": (
            selection_correct
        ),
        "normalization_succeeded": normalization_accepts > 0,
        "normalized_expression": normalized_expression,
        "executions": executions,
        "engine_successes": engine_successes,
        "engine_failures": engine_failures,
        "execution_succeeded": engine_successes > 0,
        "engine_correct": (
            answers_equal(engine_answer, reference_answer(normalized_expression))
            if engine_answer is not None and normalized_expression is not None
            else None
        ),
        "reinjection_failures": reinjection_failures,
        "reinjection_succeeded": reinjection_successes > 0,
        "interventions": len(state),
        "state_items": len(state),
        "engine_answer": engine_answer,
        "result_used": (
            answers_equal(predicted_answer, engine_answer)
            if engine_answer is not None and condition not in SEEDED_WRAPPER_BLOCK_CONDITIONS
            else None
        ),
        "result_overridden": (
            not answers_equal(predicted_answer, engine_answer)
            if engine_answer is not None and condition not in SEEDED_WRAPPER_BLOCK_CONDITIONS
            else None
        ),
        "invocation_overhead_chars": _invocation_overhead_chars(condition, recognized),
        "run_id": controller.run_id if controller else None,
        "backend_metadata": dict(generation.metadata),
        **block_metrics,
    }


def _wrap_seeded_payload(
    generation: GenerationResult,
    controller,
    *,
    tokenizer=None,
) -> tuple[GenerationResult, str]:
    payload = generation.generated_text.strip()
    protocol_text = f"```calculator\n{payload}\n```"
    step = controller.feed(protocol_text)
    inserted = "".join(item.text for item in step.reinjections)
    if tokenizer is not None:
        reinjected_tokens = len(tokenizer.encode(inserted, add_special_tokens=False))
    else:
        reinjected_tokens = len(inserted.split())
    return (
        replace(
            generation,
            rendered_text=step.rendered_text,
            reinjected_tokens=reinjected_tokens,
            metadata={**dict(generation.metadata), "harness_seeded_wrapper": True},
        ),
        protocol_text,
    )


def _trace_rows(example: ArithmeticExample, condition: str, model_id: str, seed: int, controller):
    if controller is None:
        return
    for event in controller.trace:
        yield {
            "schema_version": "ccpu.trace.v1",
            "example_id": example.example_id,
            "condition": condition,
            "model_id": model_id,
            "seed": seed,
            **event.to_dict(),
        }


def run_scripted(
    examples: Iterable[ArithmeticExample],
    *,
    conditions: Iterable[str] = CORE_CONDITIONS,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    backend = ScriptedProtocolBackend()
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for example in examples:
        for condition in conditions:
            run_id = f"scripted:{example.example_id}:{condition}:{seed}"
            controller = _controller(example, condition, run_id)
            completion = backend.completion(example, condition)
            protocol_text = None
            if condition in SEEDED_OPEN_BLOCK_CONDITIONS and example.task_kind == "arithmetic":
                opening = "```calculator\n"
                controller.feed(opening)
                protocol_text = f"{opening}{completion}"
                generation = backend.generate(
                    condition_prompt(example, condition),
                    controller=controller,
                    seed=seed,
                    completion=completion,
                )
            elif (
                condition in SEEDED_WRAPPER_BLOCK_CONDITIONS
                and example.task_kind == "arithmetic"
            ):
                generation = backend.generate(
                    condition_prompt(example, condition),
                    controller=None,
                    seed=seed,
                    completion=completion,
                )
                generation, protocol_text = _wrap_seeded_payload(generation, controller)
            else:
                generation = backend.generate(
                    condition_prompt(example, condition),
                    controller=controller,
                    seed=seed,
                    completion=completion,
                )
            predictions.append(
                _prediction(
                    example,
                    condition=condition,
                    model_id=backend.model_id,
                    seed=seed,
                    generation=generation,
                    controller=controller,
                    protocol_text=protocol_text,
                )
            )
            traces.extend(_trace_rows(example, condition, backend.model_id, seed, controller) or ())
    return predictions, traces


def run_replay(
    examples: Iterable[ArithmeticExample],
    completions: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = {example.example_id: example for example in examples}
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for row in completions:
        example_id = str(row["example_id"])
        if example_id not in items:
            raise ValueError(f"unknown replay example ID: {example_id}")
        example = items[example_id]
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"unknown replay condition: {condition}")
        model_id = str(row["model_id"])
        seed = int(row.get("seed", 0))
        generated_text = str(row["generated_text"])
        run_id = f"replay:{model_id}:{example_id}:{condition}:{seed}"
        controller = _controller(example, condition, run_id)
        started = time.perf_counter_ns()
        protocol_text = None
        if condition == "oracle" and controller is not None and example.expression is not None:
            controller.feed(f"{example.expression} =")
        if condition in SEEDED_OPEN_BLOCK_CONDITIONS and example.task_kind == "arithmetic":
            opening = "```calculator\n"
            controller.feed(opening)
            protocol_text = f"{opening}{generated_text}"
            rendered = controller.feed(generated_text).rendered_text
        elif (
            condition in SEEDED_WRAPPER_BLOCK_CONDITIONS and example.task_kind == "arithmetic"
        ):
            rendered = generated_text
        else:
            rendered = controller.feed(generated_text).rendered_text if controller else generated_text
        controller_time = time.perf_counter_ns() - started
        generation = GenerationResult(
            generated_text=generated_text,
            rendered_text=rendered,
            prompt_tokens=int(row.get("prompt_tokens", 0)),
            generated_tokens=int(row.get("generated_tokens", len(generated_text.split()))),
            reinjected_tokens=int(
                row.get(
                    "reinjected_tokens",
                    max(0, len(rendered.split()) - len(generated_text.split())),
                )
            ),
            model_calls=int(row.get("model_calls", 1)),
            wall_time_ns=int(row.get("wall_time_ns", controller_time)),
            metadata={
                **dict(row.get("backend_metadata", {})),
                "empirical": bool(row.get("empirical", True)),
                "rescored_with": "replay",
            },
        )
        if condition in SEEDED_WRAPPER_BLOCK_CONDITIONS and example.task_kind == "arithmetic":
            generation, protocol_text = _wrap_seeded_payload(generation, controller)
        predictions.append(
            _prediction(
                example,
                condition=condition,
                model_id=model_id,
                seed=seed,
                generation=generation,
                controller=controller,
                protocol_text=protocol_text,
            )
        )
        traces.extend(_trace_rows(example, condition, model_id, seed, controller) or ())
    return predictions, traces


def run_huggingface(
    examples: Iterable[ArithmeticExample],
    backend: HuggingFaceBackend,
    *,
    conditions: Iterable[str],
    seeds: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for seed in seeds:
        for example in examples:
            for condition in conditions:
                run_id = f"hf:{backend.model_id}:{example.example_id}:{condition}:{seed}"
                controller = _controller(example, condition, run_id)
                protocol_text = None
                seeded_open = (
                    condition in SEEDED_OPEN_BLOCK_CONDITIONS
                    and example.task_kind == "arithmetic"
                )
                seeded_wrapper = (
                    condition in SEEDED_WRAPPER_BLOCK_CONDITIONS
                    and example.task_kind == "arithmetic"
                )
                generation = backend.generate(
                    condition_prompt(example, condition),
                    controller=None if seeded_wrapper else controller,
                    seed=seed,
                    controller_seed_text=(
                        f"{example.expression} ="
                        if condition == "oracle"
                        else "```calculator\n" if seeded_open else None
                    ),
                )
                if seeded_open:
                    protocol_text = f"```calculator\n{generation.generated_text}"
                elif seeded_wrapper:
                    generation, protocol_text = _wrap_seeded_payload(
                        generation, controller, tokenizer=backend.tokenizer
                    )
                predictions.append(
                    _prediction(
                        example,
                        condition=condition,
                        model_id=backend.model_id,
                        seed=seed,
                        generation=generation,
                        controller=controller,
                        protocol_text=protocol_text,
                    )
                )
                traces.extend(
                    _trace_rows(example, condition, backend.model_id, seed, controller) or ()
                )
    return predictions, traces
