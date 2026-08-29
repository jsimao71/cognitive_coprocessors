"""Ten-condition Paper 1.5 experiment over a single controlled source."""

from __future__ import annotations

import re
import time
from typing import Any

from .dataset import RetrievalExample
from .generation import ConfidenceBackend, ConfidenceSpan
from .source import ControlledFactStore
from .triggers import decide, fit_confidence_threshold

CONDITIONS = (
    "llm_only",
    "anti_hallucination",
    "upfront_rag",
    "explicit_retrieval",
    "flare_like",
    "semantic",
    "confidence_or_semantic",
    "confidence_and_semantic",
    "retrospective",
    "evidence_advisory",
    "evidence_abstention",
    "runtime_epistemic_gate",
    "oracle",
)

_FINAL = re.compile(r"(?:FINAL|ANSWER)\s*:\s*([^\n]+)", re.IGNORECASE)
_TOOL = re.compile(r"<retrieve>\s*([^|<>]+)\|([^<>]+)\s*</retrieve>", re.IGNORECASE)


def extract_answer(text: str) -> str:
    match = _FINAL.search(text)
    answer = match.group(1) if match else text.splitlines()[0] if text.strip() else ""
    return answer.strip().strip("`*_\"'. ")


def answers_equal(predicted: str, gold: str) -> bool:
    return " ".join(predicted.casefold().split()) == " ".join(gold.casefold().split())


def base_prompt(example: RetrievalExample) -> str:
    return f"Answer with only the value and no explanation.\nQuestion: {example.question}\nAnswer:"


def _anti_prompt(example: RetrievalExample) -> str:
    return (
        "Avoid unsupported claims. If the question itself does not determine the answer and you "
        f"are unsure, answer ABSTAIN. Return only the value.\nQuestion: {example.question}\nAnswer:"
    )


def _tool_prompt(example: RetrievalExample) -> str:
    return (
        "If a controlled source is needed, output exactly "
        f"<retrieve>{example.entity}|{example.attribute}</retrieve>. Otherwise answer directly.\n"
        f"Question: {example.question}\nOutput:"
    )


def _evidence_prompt(
    example: RetrievalExample, evidence: dict[str, Any], draft: str | None = None
) -> str:
    draft_text = f"\nDraft answer to verify: {draft}" if draft is not None else ""
    status = str(evidence["status"])
    values = tuple(str(value) for value in evidence["values"])
    if len(values) == 1:
        selection = f"Selected final value from the source: {values[0]}"
    elif status == "conflict":
        selection = "Selected final value from the conflict policy: ABSTAIN"
    else:
        selection = (
            "No source value is available. Copy a value stated explicitly in the question; "
            "otherwise use ABSTAIN."
        )
    return (
        "Return only the selected final value after FINAL:. Never copy an evidence status."
        f"\nSource status: {status}; source values: {', '.join(values) or 'none'}; "
        f"source: {evidence['source_id']}@{evidence['source_version']}."
        f"{draft_text}\nQuestion: {example.question}\n{selection}\nFINAL:"
    )


def _advisory_prompt(example: RetrievalExample, evidence: dict[str, Any]) -> str:
    return (
        "The following controlled-source evidence is advisory. Answer with only a value; "
        "you may abstain if evidence is insufficient or conflicting."
        f"\nEvidence status: {evidence['status']}; values: "
        f"{', '.join(str(value) for value in evidence['values']) or 'none'}."
        f"\nQuestion: {example.question}\nAnswer:"
    )


def _authorized_answer(evidence: dict[str, Any]) -> str:
    values = tuple(str(value) for value in evidence["values"])
    return values[0] if len(values) == 1 else "ABSTAIN"


def _cost(*spans: ConfidenceSpan) -> dict[str, int]:
    return {
        "prompt_tokens": sum(span.prompt_tokens for span in spans),
        "generated_tokens": sum(span.generated_tokens for span in spans),
        "model_calls": sum(span.model_calls for span in spans),
        "wall_time_ns": sum(span.wall_time_ns for span in spans),
    }


def run_huggingface(
    examples: list[RetrievalExample],
    store: ControlledFactStore,
    backend: ConfidenceBackend,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    base = {example.example_id: backend.complete(base_prompt(example), seed=seed) for example in examples}
    development = [
        (base[example.example_id].token_probabilities, example.evidence_required)
        for example in examples
        if example.split == "dev"
    ]
    threshold = fit_confidence_threshold(development)
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for example in examples:
        forecast = base[example.example_id]
        forecast_answer = extract_answer(forecast.text)
        decision = decide(
            example.question,
            forecast.text,
            forecast.token_probabilities,
            threshold,
        )
        anti = backend.complete(_anti_prompt(example), seed=seed)
        tool = backend.complete(_tool_prompt(example), seed=seed)
        tool_match = _TOOL.search(tool.text)
        tool_valid = bool(
            tool_match
            and tool_match.group(1).strip().casefold() == example.entity.casefold()
            and tool_match.group(2).strip().casefold() == example.attribute.casefold()
        )

        request = store.request(
            example_id=example.example_id,
            entity=example.entity,
            attribute=example.attribute,
            as_of=example.as_of,
            forecast=forecast.text,
            candidate_answer=forecast_answer,
        )
        retrieval_started = time.perf_counter_ns()
        result = store.execute(request)
        retrieval_time_ns = time.perf_counter_ns() - retrieval_started
        grounded = backend.complete(_evidence_prompt(example, result.value), seed=seed)
        advisory = backend.complete(_advisory_prompt(example, result.value), seed=seed)
        retrospective = backend.complete(
            _evidence_prompt(example, result.value, draft=forecast_answer), seed=seed
        )

        triggers = {
            "llm_only": False,
            "anti_hallucination": False,
            "upfront_rag": True,
            "explicit_retrieval": tool_valid,
            "flare_like": decision.for_condition("flare_like"),
            "semantic": decision.for_condition("semantic"),
            "confidence_or_semantic": decision.for_condition("confidence_or_semantic"),
            "confidence_and_semantic": decision.for_condition("confidence_and_semantic"),
            "retrospective": decision.for_condition("retrospective"),
            "evidence_advisory": decision.for_condition("confidence_or_semantic"),
            "evidence_abstention": decision.for_condition("confidence_or_semantic"),
            "runtime_epistemic_gate": decision.for_condition("confidence_or_semantic"),
            "oracle": example.evidence_required,
        }

        for condition in CONDITIONS:
            retrieved = triggers[condition]
            if condition == "llm_only":
                final_span, spans = forecast, (forecast,)
            elif condition == "anti_hallucination":
                final_span, spans = anti, (anti,)
            elif condition == "upfront_rag":
                final_span, spans = grounded, (grounded,)
            elif condition == "explicit_retrieval":
                final_span = grounded if retrieved else tool
                spans = (tool, grounded) if retrieved else (tool,)
            elif condition == "retrospective" and retrieved:
                final_span, spans = retrospective, (forecast, retrospective)
            elif condition == "evidence_advisory" and retrieved:
                final_span, spans = advisory, (forecast, advisory)
            elif condition == "evidence_abstention" and retrieved:
                final_span, spans = grounded, (forecast, grounded)
            elif condition == "runtime_epistemic_gate" and retrieved:
                final_span, spans = forecast, (forecast,)
            elif retrieved:
                final_span, spans = grounded, (forecast, grounded)
            else:
                final_span, spans = forecast, (forecast,)

            final_answer = extract_answer(final_span.text)
            runtime_enforced = condition == "runtime_epistemic_gate" and retrieved
            if runtime_enforced:
                final_answer = _authorized_answer(result.value)
            low_confidence = decision.confidence
            quadrant = f"{'low' if low_confidence else 'high'}_confidence__{'high' if example.evidence_required else 'low'}_risk"
            prediction = {
                "schema_version": "ccpu.paper1_5.prediction.v2",
                "example_id": example.example_id,
                "split": example.split,
                "condition": condition,
                "model_id": backend.model_id,
                "revision": backend.revision,
                "device": backend.device,
                "seed": seed,
                "category": example.category,
                "evidence_required": example.evidence_required,
                "confidence_threshold": threshold,
                "minimum_token_probability": min(forecast.token_probabilities, default=1.0),
                "token_probabilities": forecast.token_probabilities,
                "confidence_low": low_confidence,
                "semantic_risk": decision.semantic,
                "semantic_reasons": decision.reasons,
                "quadrant": quadrant,
                "forecast_text": forecast.text,
                "forecast_answer": forecast_answer,
                "final_text": final_span.text,
                "predicted_answer": final_answer,
                "gold_answer": example.answer,
                "correct": answers_equal(final_answer, example.answer),
                "baseline_correct": answers_equal(forecast_answer, example.answer),
                "retrieved": retrieved,
                "retrieval_calls": int(retrieved),
                "request": request.to_dict() if retrieved else None,
                "evidence": result.value if retrieved else None,
                "evidence_status": result.value["status"] if retrieved else None,
                "evidence_sufficient": bool(len(result.value["values"]) == 1),
                "runtime_enforced": runtime_enforced,
                "unsupported_commitment": bool(
                    example.evidence_required
                    and final_answer.casefold() != "abstain"
                    and not answers_equal(final_answer, example.answer)
                ),
                "authorized_commitment": bool(
                    example.evidence_required
                    and example.answer.casefold() != "abstain"
                    and answers_equal(final_answer, example.answer)
                ),
                "retrieval_time_ns": retrieval_time_ns if retrieved else 0,
                **_cost(*spans),
            }
            predictions.append(prediction)
            traces.append(
                {
                    "schema_version": "ccpu.paper1_5.trace.v2",
                    "example_id": example.example_id,
                    "condition": condition,
                    "forecast": forecast.to_dict(),
                    "confidence_threshold": threshold,
                    "confidence_trigger": decision.confidence,
                    "semantic_trigger": decision.semantic,
                    "semantic_reasons": decision.reasons,
                    "triggered": retrieved,
                    "explicit_query_parsed": bool(tool_match),
                    "explicit_query_valid": tool_valid,
                    "explicit_forecast": tool.to_dict(),
                    "request": request.to_dict() if retrieved else None,
                    "result": result.to_dict() if retrieved else None,
                    "retrospective": condition == "retrospective",
                    "runtime_enforced": runtime_enforced,
                }
            )
    return predictions, traces, threshold
