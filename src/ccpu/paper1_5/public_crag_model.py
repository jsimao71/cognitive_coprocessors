"""Model-facing CRAG retrieval and Automatic Rescue experiment."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.generic_gateway import GenericCognitiveGateway, GenericToolCall
from ccpu.common.metrics import safe_mean, wilson_interval

from .generation import ConfidenceBackend, ConfidenceSpan
from .natural_robustness import semantic_features
from .public_benchmarks import _materialize
from .triggers import fit_confidence_threshold

CRAG_MODEL_CONDITIONS = (
    "llm_only",
    "upfront_retrieval",
    "voluntary_retrieve_tool",
    "semantic_runtime",
    "confidence_flare",
    "generic_retrieve_block",
    "oracle_retrieval",
    "runtime_copy",
)

_TOOL = re.compile(
    r"__retrieve\s*\(\s*(\{.*?\})\s*\)", re.IGNORECASE | re.DOTALL
)
_BLOCK = re.compile(r"\[\[COGCOP:RETRIEVE\]\]", re.IGNORECASE)


def _balanced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["static_or_dynamic"])].append(row)
    for members in groups.values():
        members.sort(key=lambda row: str(row["selection_key"]))
    selected = []
    while len(selected) < count:
        progressed = False
        for label in sorted(groups):
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError("not enough CRAG rows for a balanced model-facing slice")
    return selected


def freeze_crag_model_slice(
    source_selection: str | Path,
    output_dir: str | Path,
    *,
    calibration_count: int = 20,
    evaluation_count: int = 40,
) -> dict[str, Any]:
    source = read_jsonl(source_selection)
    calibration = _balanced(source, calibration_count)
    calibration_ids = {str(row["example_id"]) for row in calibration}
    evaluation = _balanced(
        [row for row in source if str(row["example_id"]) not in calibration_ids],
        evaluation_count,
    )
    rows = [
        {**row, "model_split": split}
        for split, members in (("calibration", calibration), ("evaluation", evaluation))
        for row in members
    ]
    rows.sort(key=lambda row: (str(row["model_split"]), str(row["selection_key"])))
    output = Path(output_dir)
    selection = write_jsonl(output / "selection.jsonl", rows)
    manifest = {
        "schema_version": "ccpu.paper1_5.crag_model_slice.v1",
        "record_count": len(rows),
        "calibration_count": calibration_count,
        "evaluation_count": evaluation_count,
        "matched_evaluation_rows": 2 * evaluation_count,
        "calibration_evaluation_id_overlap": 0,
        "source_selection_sha256": file_sha256(source_selection),
        "selection_sha256": file_sha256(selection),
        "counts": {
            split: {
                value: sum(
                    row["model_split"] == split and row["static_or_dynamic"] == value
                    for row in rows
                )
                for value in sorted({str(row["static_or_dynamic"]) for row in rows})
            }
            for split in ("calibration", "evaluation")
        },
        "redistribution": "registered IDs, strata, hashes, and model split only",
        "developmental": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _question_prompt(row: dict[str, Any], availability: str) -> str:
    context = ""
    if availability == "context":
        context = f"Authoritative supplied context: {row['answer']}\n"
    return (
        f"{context}Answer the question concisely. If it is invalid, answer `invalid question`; "
        "if the answer is unavailable, answer `I don't know`.\n"
        f"Question: {row['query']}\nAnswer:"
    )


def _grounded_prompt(row: dict[str, Any]) -> str:
    return (
        "A bounded retrieval backend returned this authoritative CRAG evidence. "
        "Use it exactly and do not replace it with parametric knowledge.\n"
        f"Evidence answer: {row['answer']}\nQuestion: {row['query']}\nAnswer:"
    )


def _tool_prompt(row: dict[str, Any], availability: str) -> str:
    return (
        "You may retrieve when external evidence is required. Emit exactly "
        '`__retrieve({"payload":{"request":"search"}})` and nothing else to retrieve; '
        "otherwise answer directly.\n" + _question_prompt(row, availability)
    )


def _block_prompt(row: dict[str, Any], availability: str) -> str:
    return (
        "Emit exactly `[[COGCOP:RETRIEVE]]` and nothing else when external evidence is "
        "required; otherwise answer directly.\n" + _question_prompt(row, availability)
    )


def _tool_attempt(text: str) -> tuple[bool, bool]:
    match = _TOOL.search(text)
    if match is None:
        return False, "__retrieve" in text.casefold()
    try:
        value = json.loads(match[1])
    except json.JSONDecodeError:
        return False, True
    valid = value == {"payload": {"request": "search"}}
    return valid, not valid


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _answer_equal(predicted: str, expected: str) -> bool:
    actual = _normalize(predicted)
    target = _normalize(expected)
    if not actual or not target:
        return False
    return actual == target or (len(target) >= 4 and target in actual)


def _abstained(text: str) -> bool:
    normalized = _normalize(text)
    return any(value in normalized for value in ("i don t know", "cannot answer", "unavailable"))


def _span_cost(span: ConfidenceSpan) -> dict[str, int]:
    return {
        "prompt_tokens": span.prompt_tokens,
        "generated_tokens": span.generated_tokens,
        "model_calls": span.model_calls,
        "wall_time_ns": span.wall_time_ns,
    }


def run_crag_model_matrix(
    config_path: str | Path,
    cache_root: str | Path,
    selection_path: str | Path,
    backend: ConfidenceBackend,
    *,
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], float]:
    materialized = _materialize(config_path, cache_root, selection_path)
    calibration = [row for row in materialized if row["model_split"] == "calibration"]
    evaluation = [row for row in materialized if row["model_split"] == "evaluation"]
    calibration_spans = []
    for row in calibration:
        for availability in ("external", "context"):
            span = backend.complete(_question_prompt(row, availability), seed=seed)
            calibration_spans.append((span.token_probabilities, availability == "external"))
    threshold = fit_confidence_threshold(calibration_spans)

    retrieved_calls = 0

    def resolve(intent: str, payload: Any, active_task: str) -> str:
        nonlocal retrieved_calls
        if intent != "retrieve" or payload != {"request": "search"}:
            raise ValueError("invalid CRAG retrieval call")
        retrieved_calls += 1
        return active_task

    gateway = GenericCognitiveGateway(resolve)
    predictions = []
    for row_index, row in enumerate(evaluation, 1):
        grounded = backend.complete(_grounded_prompt(row), seed=seed + 1)
        for availability in ("external", "context"):
            required = availability == "external"
            prompt = _question_prompt(row, availability)
            baseline = backend.complete(prompt, seed=seed)
            tool = backend.complete(_tool_prompt(row, availability), seed=seed)
            block = backend.complete(_block_prompt(row, availability), seed=seed)
            tool_trigger, tool_malformed = _tool_attempt(tool.text)
            block_trigger = bool(_BLOCK.search(block.text))
            block_malformed = "COGCOP" in block.text.upper() and not block_trigger
            confidence_trigger = bool(baseline.token_probabilities) and min(
                baseline.token_probabilities
            ) < threshold
            semantic_trigger = bool(semantic_features(prompt)["combined"])
            triggers = {
                "llm_only": False,
                "upfront_retrieval": True,
                "voluntary_retrieve_tool": tool_trigger,
                "semantic_runtime": semantic_trigger,
                "confidence_flare": confidence_trigger,
                "generic_retrieve_block": block_trigger,
                "oracle_retrieval": required,
                "runtime_copy": required,
            }
            initial = {
                "voluntary_retrieve_tool": tool,
                "generic_retrieve_block": block,
            }
            malformed = {
                "voluntary_retrieve_tool": tool_malformed,
                "generic_retrieve_block": block_malformed,
            }
            for condition in CRAG_MODEL_CONDITIONS:
                retrieved = triggers[condition]
                first = initial.get(condition, baseline)
                if condition == "runtime_copy":
                    final_text = str(row["answer"])
                    costs = {key: 0 for key in _span_cost(baseline)}
                elif retrieved:
                    if condition == "voluntary_retrieve_tool":
                        gateway.invoke(
                            GenericToolCall("__retrieve", {"request": "search"}),
                            active_task=str(row["answer"]),
                        )
                    final_text = grounded.text
                    first_cost = _span_cost(first)
                    grounded_cost = _span_cost(grounded)
                    costs = {key: first_cost[key] + grounded_cost[key] for key in first_cost}
                else:
                    final_text = first.text
                    costs = _span_cost(first)
                correct = _answer_equal(final_text, str(row["answer"]))
                abstained = _abstained(final_text)
                predictions.append(
                    {
                        "schema_version": "ccpu.paper1_5.crag_model_prediction.v1",
                        "example_id": row["example_id"],
                        "content_sha256": row["content_sha256"],
                        "availability": availability,
                        "evidence_required": required,
                        "condition": condition,
                        "model_id": backend.model_id,
                        "revision": backend.revision,
                        "answer_type": row["answer_type"],
                        "domain": row["domain"],
                        "question_type": row["question_type"],
                        "static_or_dynamic": row["static_or_dynamic"],
                        "retrieved": retrieved,
                        "malformed_retrieval": malformed.get(condition, False),
                        "confidence_threshold": threshold,
                        "minimum_token_probability": min(
                            first.token_probabilities, default=1.0
                        ),
                        "first_text": first.text,
                        "grounded_text": grounded.text if retrieved else "",
                        "predicted_answer": final_text,
                        "correct": correct,
                        "abstained": abstained,
                        "unsupported_commitment": required and not retrieved and not abstained,
                        "authorized_commitment": correct and (retrieved or not required),
                        "evidence_override": retrieved and not correct,
                        **costs,
                    }
                )
        if checkpoint_path is not None and row_index % 5 == 0:
            write_jsonl(checkpoint_path, predictions)
    if checkpoint_path is not None:
        write_jsonl(checkpoint_path, predictions)
    if retrieved_calls != sum(
        row["retrieved"] and row["condition"] == "voluntary_retrieve_tool"
        for row in predictions
    ):
        raise ValueError("CRAG generic retrieval gateway accounting mismatch")
    return predictions, threshold


def summarize_crag_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    results = []
    for condition, members in sorted(grouped.items()):
        required = [row for row in members if row["evidence_required"]]
        controls = [row for row in members if not row["evidence_required"]]
        correct = sum(bool(row["correct"]) for row in members)
        results.append(
            {
                "condition": condition,
                "count": len(members),
                "answer_accuracy": correct / len(members),
                "answer_accuracy_ci95": wilson_interval(correct, len(members)),
                "external_answer_accuracy": safe_mean(row["correct"] for row in required),
                "context_answer_accuracy": safe_mean(row["correct"] for row in controls),
                "retrieval_needed_recall": safe_mean(row["retrieved"] for row in required),
                "false_retrieval_rate": safe_mean(row["retrieved"] for row in controls),
                "ucr": safe_mean(row["unsupported_commitment"] for row in required),
                "authorized_commitment_coverage": safe_mean(
                    row["authorized_commitment"] for row in required
                ),
                "evidence_override_rate": safe_mean(
                    row["evidence_override"] for row in members if row["retrieved"]
                ),
                "malformed_rate": safe_mean(row["malformed_retrieval"] for row in members),
                "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in members),
                "mean_model_calls": safe_mean(row["model_calls"] for row in members),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in members) / 1e6,
            }
        )
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(str(row["example_id"]), str(row["availability"]))][
            str(row["condition"])
        ] = row
    voluntary_misses = [
        values
        for (_, availability), values in paired.items()
        if availability == "external" and not values["voluntary_retrieve_tool"]["retrieved"]
    ]
    rescue = []
    for condition in ("semantic_runtime", "confidence_flare", "oracle_retrieval"):
        rescued = sum(
            values[condition]["retrieved"] and values[condition]["correct"]
            for values in voluntary_misses
        )
        rescue.append(
            {
                "condition": condition,
                "eligible_voluntary_misses": len(voluntary_misses),
                "rescued_correctly": rescued,
                "automatic_rescue_rate": rescued / len(voluntary_misses)
                if voluntary_misses
                else None,
            }
        )
    return {
        "schema_version": "ccpu.paper1_5.crag_model_summary.v1",
        "prediction_count": len(rows),
        "base_question_count": len({str(row["example_id"]) for row in rows}),
        "matched_evaluation_rows": len(paired),
        "confidence_threshold": rows[0]["confidence_threshold"] if rows else None,
        "results": results,
        "automatic_rescue": rescue,
        "claim_boundary": {
            "evidence_backend": "frozen CRAG reference answer",
            "source_routing": "registered single source",
            "status": "developmental model-facing slice",
        },
    }


def write_crag_model_run(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    model_config: str | Path,
    selection_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = summarize_crag_model(rows)
    summary_path = write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "ccpu.paper1_5.crag_model_manifest.v1",
            "model_config_sha256": file_sha256(model_config),
            "selection_sha256": file_sha256(selection_path),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary_path),
            "environment": environment_manifest(Path(__file__).resolve().parents[3]),
        },
    )
    return summary


def plot_crag_model(summary: dict[str, Any], output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("CRAG model plotting requires the analysis extra") from error
    labels = [row["condition"].replace("_", "\n") for row in summary["results"]]
    x = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.4, 4.5))
    axis.bar(
        [value - width / 2 for value in x],
        [row["answer_accuracy"] for row in summary["results"]],
        width,
        label="Answer accuracy",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [row["retrieval_needed_recall"] for row in summary["results"]],
        width,
        label="Retrieval-needed recall",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
