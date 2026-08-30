"""Model-facing GSM8K calculator, generic-tool, and automatic-runtime runs."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from fractions import Fraction
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
from ccpu.common.schema import DetectionCandidate

from .arithmetic import ArithmeticNormalizationError, BoundedCalculator
from .generation import HuggingFaceBackend
from .reflex import build_calculator_block_runtime, build_normalized_reflex_runtime
from .surface import ArithmeticSurfaceNormalizer

PUBLIC_GSM8K_CONDITIONS = (
    "llm_only",
    "matched_icl",
    "generic_compute",
    "calculator_block",
    "runtime_trigger",
    "oracle_calculator",
    "lora_calculator_block",
)

_GENERIC_CALL = re.compile(r"__compute\s*\(\s*(\{.*?\})\s*\)", re.DOTALL)
_ANSWER_PATTERNS = (
    re.compile(
        r"(?:final\s+)?answer\s*(?:is|:|=)\s*\**\$?\s*(-?[\d,]+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"\\boxed\s*\{\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*\}"),
    re.compile(r"(-?[\d,]+(?:\.\d+)?)\s*[.!]?\s*$"),
)


def freeze_gsm8k_slice(
    source_selection: str | Path,
    output_dir: str | Path,
    *,
    per_stratum: int = 40,
) -> dict[str, Any]:
    rows = [
        row for row in read_jsonl(source_selection) if row.get("benchmark") == "gsm8k"
    ]
    selected = []
    counts = {}
    for stratum in ("2_steps", "3_4_steps", "5plus_steps"):
        members = sorted(
            [row for row in rows if row["difficulty_stratum"] == stratum],
            key=lambda row: row["selection_key"],
        )
        if len(members) < per_stratum:
            raise ValueError(f"not enough frozen GSM8K rows in {stratum}")
        selected.extend(members[:per_stratum])
        counts[stratum] = per_stratum
    selected = sorted(selected, key=lambda row: row["selection_key"])
    output = Path(output_dir)
    selection = write_jsonl(output / "selection.jsonl", selected)
    manifest = {
        "schema_version": "ccpu.paper1.public_gsm8k_slice.v1",
        "record_count": len(selected),
        "per_stratum": per_stratum,
        "counts": counts,
        "source_selection_sha256": file_sha256(source_selection),
        "selection_sha256": file_sha256(selection),
        "selection_rule": "lowest frozen selection_key within each difficulty stratum",
        "redistribution": "IDs, labels, difficulty metadata, and content hashes only",
        "developmental": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _fraction(value: str | None) -> Fraction | None:
    if value is None:
        return None
    try:
        return Fraction(value.replace(",", "").replace("$", "").strip())
    except (ValueError, ZeroDivisionError):
        return None


def extract_gsm8k_answer(text: str) -> str | None:
    matches = [match for pattern in _ANSWER_PATTERNS for match in pattern.finditer(text)]
    return max(matches, key=lambda item: item.end()).group(1) if matches else None


def _gold_operations(example: dict[str, Any]) -> list[str]:
    normalizer = ArithmeticSurfaceNormalizer()
    operations = []
    for index, item in enumerate(example["opportunities"]):
        expression = str(item["expression"])
        candidate = DetectionCandidate(
            candidate_id=f"{example['example_id']}:gold:{index}",
            family="compute",
            raw_text=expression,
            start_offset=0,
            end_offset=len(expression),
            detector="gsm8k_gold",
        )
        try:
            request = normalizer.normalize(candidate)
        except ArithmeticNormalizationError:
            continue
        operations.append(str(request.payload["canonical_expression"]))
    return operations


def _oracle_calls(example: dict[str, Any]) -> list[dict[str, str]]:
    calculator = BoundedCalculator()
    normalizer = ArithmeticSurfaceNormalizer()
    calls = []
    for index, item in enumerate(example["opportunities"]):
        expression = str(item["expression"])
        candidate = DetectionCandidate(
            candidate_id=f"{example['example_id']}:oracle:{index}",
            family="compute",
            raw_text=expression,
            start_offset=0,
            end_offset=len(expression),
            detector="gsm8k_oracle",
        )
        try:
            request = normalizer.normalize(candidate)
        except ArithmeticNormalizationError:
            continue
        result = calculator.execute(request)
        if not result.ok:
            continue
        calls.append(
            {
                "expression": expression,
                "canonical_expression": str(request.payload["canonical_expression"]),
                "result": str(result.display),
            }
        )
    return calls


def _prompt(
    example: dict[str, Any],
    condition: str,
    *,
    oracle_calls: list[dict[str, str]] | None = None,
) -> str:
    question = str(example["question"])
    endpoint = (
        "Use compact equations rather than prose and at most six calculation lines. "
        "On the final line write `Answer:` followed by the numeric result; never write a "
        "placeholder word."
    )
    if condition == "llm_only":
        instruction = f"Solve the word problem carefully. {endpoint}"
    elif condition == "matched_icl":
        instruction = (
            "Example: Ava has 3 bags with 4 marbles each and gives away 2. "
            "Answer: 10\nNow solve the next word problem carefully. " + endpoint
        )
    elif condition == "generic_compute":
        instruction = (
            "Solve the problem. When exact arithmetic is needed, emit only one call such as "
            '`__compute({"expression":"7 * 8"})` and wait for its Tool result. '
            "Never repeat a completed call. Use the current problem's numbers and one operation "
            "at a time. If no call is needed, solve directly. " + endpoint
        )
    elif condition in {"calculator_block", "lora_calculator_block"}:
        instruction = (
            "Solve the problem step by step. For each exact arithmetic operation, write one "
            "fenced calculator block containing only the expression, continue from the inserted "
            "result, and then continue. Inside a block write only numbers and arithmetic "
            "operators from the current problem; never write placeholder words."
        )
    elif condition == "runtime_trigger":
        instruction = (
            "Solve step by step. Write every exact arithmetic operation with numbers and "
            "operators followed by `=` (for example, `7 * 8 =`) and continue from any "
            "automatically inserted result. Never write placeholder words. " + endpoint
        )
    elif condition == "oracle_calculator":
        ledger = "; ".join(
            f"{item['canonical_expression']} = {item['result']}"
            for item in (oracle_calls or [])
        )
        instruction = (
            "Solve the problem using this bounded-calculator ledger. Do not change its "
            f"arithmetic results: {ledger}. {endpoint}"
        )
    else:
        raise ValueError(f"unsupported public GSM8K condition: {condition}")
    return f"{instruction}\n\nProblem: {question}\n\nResponse:"


def _operation_overlap(gold: list[str], attempted: list[str]) -> int:
    gold_counts = Counter(gold)
    attempted_counts = Counter(attempted)
    return sum((gold_counts & attempted_counts).values())


def _generic_run(
    example: dict[str, Any], backend: HuggingFaceBackend, seed: int, max_episodes: int
) -> dict[str, Any]:
    calculator = BoundedCalculator()
    normalizer = ArithmeticSurfaceNormalizer()
    calls = []

    def resolve(intent: str, payload: Any, active_task: str) -> dict[str, Any]:
        if intent != "compute" or active_task != "gsm8k":
            raise ValueError("unexpected GSM8K generic-tool route")
        if not isinstance(payload, dict) or set(payload) != {"expression"}:
            raise ValueError("__compute requires exactly one expression field")
        expression = str(payload["expression"])
        candidate = DetectionCandidate(
            candidate_id=f"{example['example_id']}:generic:{len(calls)}",
            family="compute",
            raw_text=expression,
            start_offset=0,
            end_offset=len(expression),
            detector="generic_compute",
        )
        request = normalizer.normalize(candidate)
        result = calculator.execute(request)
        if not result.ok:
            raise ValueError(result.error_message or "calculator rejected expression")
        record = {
            "expression": expression,
            "canonical_expression": request.payload["canonical_expression"],
            "result": result.display,
        }
        calls.append(record)
        return record

    gateway = GenericCognitiveGateway(resolve)
    prompt = _prompt(example, "generic_compute")
    transcript = []
    malformed = 0
    duplicate = 0
    prompt_tokens = generated_tokens = model_calls = wall_time_ns = 0
    first_assistance_char = None
    for episode in range(max_episodes + 1):
        generation = backend.generate(prompt, seed=seed + episode)
        output = generation.generated_text
        transcript.append(output)
        prompt_tokens += generation.prompt_tokens
        generated_tokens += generation.generated_tokens
        model_calls += generation.model_calls
        wall_time_ns += generation.wall_time_ns
        match = _GENERIC_CALL.search(output)
        if match is None:
            if "__compute" in output:
                malformed += 1
            break
        if first_assistance_char is None:
            first_assistance_char = sum(len(item) for item in transcript[:-1]) + match.start()
        try:
            payload = json.loads(match.group(1))
            result = gateway.invoke(
                GenericToolCall("__compute", payload), active_task="gsm8k"
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            malformed += 1
            break
        if len(calls) >= 2 and calls[-1]["canonical_expression"] == calls[-2][
            "canonical_expression"
        ]:
            duplicate += 1
            break
        prompt = (
            f"{prompt}\nAssistant: {output}\n"
            f"Tool: {json.dumps({'value': result['result']})}\nAssistant:"
        )
    rendered = "\n".join(transcript)
    return {
        "generated_text": rendered,
        "rendered_text": rendered,
        "calls": calls,
        "runtime_trace": [],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "reinjected_tokens": 0,
        "model_calls": model_calls,
        "wall_time_ns": wall_time_ns,
        "malformed_assistance": malformed,
        "duplicate_assistance": duplicate,
        "first_assistance_char": first_assistance_char,
        "backend_metadata": generation.metadata,
    }


def run_gsm8k_example(
    example: dict[str, Any],
    backend: HuggingFaceBackend,
    *,
    condition: str,
    seed: int,
    max_assistance_episodes: int = 4,
) -> dict[str, Any]:
    if condition not in PUBLIC_GSM8K_CONDITIONS:
        raise ValueError(f"unknown GSM8K condition: {condition}")
    started = time.perf_counter_ns()
    if condition == "generic_compute":
        run = _generic_run(example, backend, seed, max_assistance_episodes)
    else:
        controller = None
        oracle_calls = _oracle_calls(example) if condition == "oracle_calculator" else []
        if condition in {"calculator_block", "lora_calculator_block"}:
            controller = build_calculator_block_runtime(
                run_id=f"gsm8k:{backend.model_id}:{example['example_id']}:{condition}:{seed}"
            )
        elif condition == "runtime_trigger":
            controller = build_normalized_reflex_runtime(
                run_id=f"gsm8k:{backend.model_id}:{example['example_id']}:{condition}:{seed}"
            )
        generation = backend.generate(
            _prompt(example, condition, oracle_calls=oracle_calls),
            controller=controller,
            seed=seed,
        )
        calls = oracle_calls
        runtime_trace = []
        first_assistance_char = None
        malformed = 0
        if controller is not None:
            runtime_trace = [event.to_dict() for event in controller.trace]
            calls = [
                {
                    "expression": item.request.payload["canonical_expression"],
                    "canonical_expression": item.request.payload["canonical_expression"],
                    "result": item.result.display,
                }
                for item in controller.state
            ]
            detections = [
                event for event in runtime_trace if event["stage"] == "detection"
            ]
            if detections:
                raw = str(detections[0]["details"]["raw_text"])
                first_assistance_char = generation.generated_text.find(raw)
            malformed = sum(
                event["stage"] == "normalization" and event["status"] == "rejected"
                for event in runtime_trace
            )
        run = {
            **generation.to_dict(),
            "calls": calls,
            "runtime_trace": runtime_trace,
            "malformed_assistance": malformed,
            "duplicate_assistance": 0,
            "first_assistance_char": first_assistance_char,
            "backend_metadata": dict(generation.metadata),
        }

    predicted = extract_gsm8k_answer(str(run["rendered_text"]))
    gold_operations = _gold_operations(example)
    attempted = [str(call["canonical_expression"]) for call in run["calls"]]
    matched = _operation_overlap(gold_operations, attempted)
    target = str(example["target_label"])
    return {
        "schema_version": "ccpu.paper1.public_gsm8k_prediction.v1",
        "benchmark": "gsm8k",
        "example_id": example["example_id"],
        "content_sha256": example["content_sha256"],
        "difficulty": example["difficulty"],
        "difficulty_stratum": example["difficulty_stratum"],
        "target_label": target,
        "model_id": backend.model_id,
        "condition": condition,
        "seed": seed,
        "generated_text": run["generated_text"],
        "rendered_text": run["rendered_text"],
        "predicted_answer": predicted,
        "correct": _fraction(predicted) == _fraction(target),
        "gold_operation_count": len(example["opportunities"]),
        "supported_gold_operation_count": len(gold_operations),
        "assistance_attempted": bool(run["calls"]) or bool(run["malformed_assistance"]),
        "assistance_valid": bool(run["calls"]),
        "assistance_calls": len(run["calls"]),
        "matched_gold_operations": matched,
        "malformed_assistance": run["malformed_assistance"],
        "duplicate_assistance": run["duplicate_assistance"],
        "first_assistance_char": run["first_assistance_char"],
        "assistance_lead_chars": (
            len(str(run["generated_text"])) - int(run["first_assistance_char"])
            if run["first_assistance_char"] is not None
            and int(run["first_assistance_char"]) >= 0
            else None
        ),
        "calls": run["calls"],
        "runtime_trace": run["runtime_trace"],
        "prompt_tokens": run["prompt_tokens"],
        "generated_tokens": run["generated_tokens"],
        "reinjected_tokens": run["reinjected_tokens"],
        "model_calls": run["model_calls"],
        "wall_time_ns": run["wall_time_ns"],
        "total_runner_time_ns": time.perf_counter_ns() - started,
        "backend_metadata": run["backend_metadata"],
    }


def summarize_gsm8k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    by_condition = []
    for condition, members in sorted(grouped.items()):
        correct = sum(bool(row["correct"]) for row in members)
        gold_operations = sum(int(row["gold_operation_count"]) for row in members)
        supported_gold_operations = sum(
            int(row["supported_gold_operation_count"]) for row in members
        )
        by_condition.append(
            {
                "condition": condition,
                "count": len(members),
                "accuracy": correct / len(members),
                "accuracy_ci95": wilson_interval(correct, len(members)),
                "scorable_answer_rate": safe_mean(
                    row["predicted_answer"] is not None for row in members
                ),
                "assistance_rate": safe_mean(row["assistance_valid"] for row in members),
                "malformed_assistance_rate": safe_mean(
                    row["malformed_assistance"] > 0 for row in members
                ),
                "duplicate_assistance_rate": safe_mean(
                    row["duplicate_assistance"] > 0 for row in members
                ),
                "registered_operation_recall": (
                    sum(int(row["matched_gold_operations"]) for row in members)
                    / supported_gold_operations
                    if supported_gold_operations
                    else None
                ),
                "registered_operation_backend_coverage": (
                    supported_gold_operations / gold_operations if gold_operations else None
                ),
                "mean_assistance_calls": safe_mean(row["assistance_calls"] for row in members),
                "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in members),
                "mean_model_calls": safe_mean(row["model_calls"] for row in members),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in members) / 1e6,
                "by_difficulty": {
                    stratum: {
                        "count": len(items),
                        "accuracy": safe_mean(item["correct"] for item in items),
                        "assistance_rate": safe_mean(
                            item["assistance_valid"] for item in items
                        ),
                    }
                    for stratum, items in sorted(
                        {
                            stratum: [
                                item
                                for item in members
                                if item["difficulty_stratum"] == stratum
                            ]
                            for stratum in {
                                str(item["difficulty_stratum"]) for item in members
                            }
                        }.items()
                    )
                },
            }
        )

    paired = defaultdict(dict)
    for row in rows:
        paired[str(row["example_id"])][str(row["condition"])] = row
    eligible = [
        values
        for values in paired.values()
        if {"generic_compute", "runtime_trigger"} <= values.keys()
        and not values["generic_compute"]["assistance_valid"]
    ]
    rescued = [
        values
        for values in eligible
        if values["runtime_trigger"]["assistance_valid"]
        and values["runtime_trigger"]["correct"]
    ]
    paired_vs_llm_only = {}
    for condition in sorted(grouped):
        if condition == "llm_only":
            continue
        comparisons = [
            (values["llm_only"], values[condition])
            for values in paired.values()
            if {"llm_only", condition} <= values.keys()
        ]
        if not comparisons:
            continue
        paired_vs_llm_only[condition] = {
            "count": len(comparisons),
            "gains": sum(not base["correct"] and candidate["correct"] for base, candidate in comparisons),
            "losses": sum(base["correct"] and not candidate["correct"] for base, candidate in comparisons),
            "both_correct": sum(base["correct"] and candidate["correct"] for base, candidate in comparisons),
            "both_wrong": sum(not base["correct"] and not candidate["correct"] for base, candidate in comparisons),
        }
    return {
        "schema_version": "ccpu.paper1.public_gsm8k_summary.v1",
        "record_count": len(rows),
        "base_question_count": len(paired),
        "by_condition": by_condition,
        "automatic_rescue": {
            "eligible_voluntary_misses": len(eligible),
            "rescued_correctly": len(rescued),
            "rate": len(rescued) / len(eligible) if eligible else None,
            "definition": (
                "runtime valid intervention plus correct answer among paired rows where "
                "voluntary __compute made no valid call"
            ),
        },
        "paired_vs_llm_only": paired_vs_llm_only,
        "claim_boundary": {
            "false_activation_rate": None,
            "first_wrong_token_prevention": None,
            "reason": "GSM8K selection has no matched no-compute controls or token alignment",
        },
    }


def write_gsm8k_run(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    config_path: str | Path,
    selection_path: str | Path,
    condition: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = summarize_gsm8k(rows)
    summary_path = write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "ccpu.paper1.public_gsm8k_run_manifest.v1",
            "condition": condition,
            "record_count": len(rows),
            "config_sha256": file_sha256(config_path),
            "selection_sha256": file_sha256(selection_path),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary_path),
            "environment": environment_manifest(Path(__file__).resolve().parents[3]),
        },
    )
    return summary


def analyze_gsm8k_runs(
    prediction_paths: list[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    rows = [row for path in prediction_paths for row in read_jsonl(path)]
    keys = [(row["example_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate GSM8K example/condition predictions")
    conditions = {str(row["condition"]) for row in rows}
    missing = set(PUBLIC_GSM8K_CONDITIONS) - conditions
    if missing:
        raise ValueError(f"missing GSM8K conditions: {sorted(missing)}")
    id_sets = {
        condition: {
            str(row["example_id"]) for row in rows if row["condition"] == condition
        }
        for condition in conditions
    }
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise ValueError("GSM8K conditions do not contain identical selected IDs")

    summary = summarize_gsm8k(rows)
    summary["prediction_sources"] = [
        {"path": str(path), "sha256": file_sha256(path)} for path in prediction_paths
    ]
    output = Path(output_dir)
    write_json(output / "summary.json", summary)
    _plot_gsm8k(summary, output / "gsm8k_accuracy.png")
    return summary


def _plot_gsm8k(summary: dict[str, Any], output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("GSM8K plotting requires the analysis extra") from error

    labels = {
        "llm_only": "LLM only",
        "matched_icl": "Matched ICL",
        "generic_compute": "Generic compute",
        "calculator_block": "Calculator block",
        "runtime_trigger": "Runtime trigger",
        "oracle_calculator": "Oracle ledger",
        "lora_calculator_block": "LoRA block",
    }
    by_condition = {row["condition"]: row for row in summary["by_condition"]}
    order = [condition for condition in PUBLIC_GSM8K_CONDITIONS if condition in by_condition]
    strata = ("2_steps", "3_4_steps", "5plus_steps")
    stratum_labels = {
        "2_steps": "2 steps",
        "3_4_steps": "3-4 steps",
        "5plus_steps": "5+ steps",
    }
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    x = list(range(len(order)))
    width = 0.24
    for index, stratum in enumerate(strata):
        values = [by_condition[item]["by_difficulty"][stratum]["accuracy"] for item in order]
        axis.bar(
            [position + (index - 1) * width for position in x],
            values,
            width=width,
            label=stratum_labels[stratum],
        )
    axis.set_xticks(x, [labels[item] for item in order], rotation=22, ha="right")
    axis.set_ylabel("Exact final-answer accuracy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
