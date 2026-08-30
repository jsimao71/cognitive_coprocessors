"""Matched model-facing public compute runs over the fixed four-tool gateway."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
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
from ccpu.common.generic_gateway import (
    GenericCognitiveGateway,
    GenericToolCall,
    generic_tool_schemas,
)
from ccpu.common.lexical_routing import current_word_tokens
from ccpu.common.metrics import safe_mean, wilson_interval
from ccpu.paper1.generation import HuggingFaceBackend

from .diagnostic import _t1
from .public_adapters import registered_assistance
from .public_benchmarks import _materialize_selected

PUBLIC_COMPUTE_CONDITIONS = (
    "llm_only",
    "four_tools",
    "generic_cogcop",
    "cpu_t1",
    "oracle_route",
)
PUBLIC_COMPUTE_BENCHMARKS = (
    "gsm8k",
    "bigbench_unit_conversion",
    "bigbench_date_understanding",
    "proofwriter_balanced",
    "clutrr",
)

_TOOL_CALL = re.compile(r"__(compute|retrieve|verify|help)\s*\(\s*(\{.*\})\s*\)", re.DOTALL)
_COGCOP = re.compile(r"\[\[COGCOP:(COMPUTE|RETRIEVE|VERIFY|HELP)\]\]", re.IGNORECASE)
_ANSWER = re.compile(r"(?:final\s+)?answer\s*(?:is|:|=)\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", re.IGNORECASE)
_T1_LABEL = {
    "calculator": "CALCULATOR",
    "units": "UNITS",
    "date_time": "DATE",
    "datalog": "DATALOG",
    "graph": "GRAPH",
}


def _balanced_take(
    rows: list[dict[str, Any]], key: str, count: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: str(row["selection_key"]))
    selected = []
    while len(selected) < count:
        progressed = False
        for group in sorted(groups):
            if groups[group] and len(selected) < count:
                selected.append(groups[group].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) < count:
        raise ValueError(f"not enough executable public rows balanced by {key}")
    return selected


def freeze_executable_public_slice(
    config_path: str | Path,
    cache_root: str | Path,
    source_selection: str | Path,
    output_dir: str | Path,
    *,
    per_benchmark: int = 12,
) -> dict[str, Any]:
    materialized, _ = _materialize_selected(config_path, cache_root, source_selection)
    supported: dict[str, list[dict[str, Any]]] = defaultdict(list)
    adapter_counts: dict[str, int] = defaultdict(int)
    for row in materialized:
        try:
            adapter = registered_assistance(row)
        except ValueError:
            continue
        enriched = {
            **row,
            "registered_intent": adapter["intent"],
            "registered_engine": adapter["engine"],
            "formalization_source": adapter["formalization_source"],
        }
        supported[str(row["benchmark"])].append(enriched)
        adapter_counts[str(row["benchmark"])] += 1

    selected = []
    for benchmark in PUBLIC_COMPUTE_BENCHMARKS:
        members = supported[benchmark]
        if benchmark == "proofwriter_balanced":
            chosen = _balanced_take(members, "target_label", per_benchmark)
        elif benchmark in {"gsm8k", "bigbench_date_understanding", "clutrr"}:
            chosen = _balanced_take(members, "difficulty_stratum", per_benchmark)
        else:
            chosen = sorted(members, key=lambda row: str(row["selection_key"]))[
                :per_benchmark
            ]
        if len(chosen) < per_benchmark:
            raise ValueError(f"not enough executable rows for {benchmark}")
        selected.extend(chosen)

    redistributed_keys = {
        "benchmark",
        "engine",
        "example_id",
        "source_row",
        "target_label",
        "difficulty",
        "difficulty_stratum",
        "content_sha256",
        "selection_key",
        "registered_intent",
        "registered_engine",
        "formalization_source",
    }
    records = [
        {key: row[key] for key in redistributed_keys}
        for row in sorted(selected, key=lambda row: str(row["selection_key"]))
    ]
    output = Path(output_dir)
    selection = write_jsonl(output / "selection.jsonl", records)
    manifest = {
        "schema_version": "ccpu.paper2.public_executable_slice.v1",
        "record_count": len(records),
        "per_benchmark": per_benchmark,
        "selected_counts": {
            benchmark: sum(row["benchmark"] == benchmark for row in records)
            for benchmark in PUBLIC_COMPUTE_BENCHMARKS
        },
        "adapter_coverage_counts": dict(sorted(adapter_counts.items())),
        "formalization_sources": {
            source: sum(row["formalization_source"] == source for row in records)
            for source in sorted({str(row["formalization_source"]) for row in records})
        },
        "selection_rule": "balanced registered-adapter rows, then lowest frozen selection_key",
        "redistribution": "IDs, labels, difficulty, hashes, and adapter provenance only",
        "source_selection_sha256": file_sha256(source_selection),
        "selection_sha256": file_sha256(selection),
        "developmental": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def materialize_executable_public_slice(
    config_path: str | Path, cache_root: str | Path, selection_path: str | Path
) -> list[dict[str, Any]]:
    rows, _ = _materialize_selected(config_path, cache_root, selection_path)
    selected = {str(row["example_id"]): row for row in read_jsonl(selection_path)}
    for row in rows:
        registered = registered_assistance(row)
        frozen = selected[str(row["example_id"])]
        for key, expected in (
            ("registered_intent", registered["intent"]),
            ("registered_engine", registered["engine"]),
            ("formalization_source", registered["formalization_source"]),
        ):
            if frozen[key] != expected:
                raise ValueError(f"public executable adapter changed at {row['example_id']}:{key}")
        row["registered"] = registered
    return sorted(rows, key=lambda row: str(row["selection_key"]))


def _endpoint(benchmark: str) -> str:
    if benchmark in {"gsm8k", "bigbench_unit_conversion"}:
        return "On the final line write `Answer:` followed by the numeric result."
    if benchmark == "bigbench_date_understanding":
        return "On the final line write `Answer:` followed by the date in MM/DD/YYYY."
    if benchmark == "proofwriter_balanced":
        return "On the final line write exactly `Answer: TRUE`, `Answer: FALSE`, or `Answer: UNCERTAIN`."
    return "On the final line write `Answer:` followed by the kinship relation only."


def _base_prompt(row: dict[str, Any]) -> str:
    return f"Solve this task carefully. {_endpoint(str(row['benchmark']))}\n\nTask:\n{row['prompt']}"


def _condition_prompt(row: dict[str, Any], condition: str) -> str:
    base = _base_prompt(row)
    if condition == "llm_only":
        return base
    if condition == "four_tools":
        schemas = json.dumps(generic_tool_schemas(), sort_keys=True, separators=(",", ":"))
        return (
            "You have exactly four stable cognitive tools. COMPUTE handles arithmetic, units, "
            "and dates; VERIFY checks logical claims; RETRIEVE resolves registered relations; "
            "HELP reports unsupported tasks. If assistance is needed, output exactly one call "
            "with payload request=solve and nothing else. Otherwise answer directly.\n"
            f"Schemas: {schemas}\n{base}"
        )
    if condition == "generic_cogcop":
        return (
            "A local runtime accepts one generic control token: [[COGCOP:COMPUTE]], "
            "[[COGCOP:VERIFY]], [[COGCOP:RETRIEVE]], or [[COGCOP:HELP]]. Emit only one "
            "token if exact assistance is needed; otherwise answer directly.\n" + base
        )
    raise ValueError(f"condition {condition} requires a registered result decision")


def _result_prompt(row: dict[str, Any], result: str, source: str) -> str:
    return (
        f"An authoritative bounded runtime result is available ({source}): {result}. "
        "Use it without recomputing. "
        f"{_endpoint(str(row['benchmark']))}\n\nTask:\n{row['prompt']}"
    )


def _tool_attempt(text: str) -> tuple[str | None, Any, bool]:
    match = _TOOL_CALL.search(text)
    if match is None:
        return None, None, "__" in text
    try:
        payload = json.loads(match[2])
    except json.JSONDecodeError:
        return match[1], None, True
    if not isinstance(payload, dict) or set(payload) != {"payload"}:
        return match[1], None, True
    request = payload["payload"]
    if not isinstance(request, dict) or request != {"request": "solve"}:
        return match[1], None, True
    return match[1], request, False


def _cogcop_attempt(text: str) -> tuple[str | None, bool]:
    match = _COGCOP.search(text)
    return (match[1].casefold(), False) if match else (None, "COGCOP" in text.upper())


def _extract_answer(text: str) -> str | None:
    matches = list(_ANSWER.finditer(text))
    if not matches:
        return None
    value = matches[-1].group(1).strip().splitlines()[0].strip(" `*$.")
    return value or None


def _score_answer(benchmark: str, predicted: str | None, target: str) -> bool:
    if predicted is None:
        return False
    if benchmark == "gsm8k":
        try:
            predicted_match = _NUMBER.search(predicted.replace(",", ""))
            return bool(predicted_match) and Fraction(predicted_match.group(0)) == Fraction(
                target.replace(",", "")
            )
        except (ValueError, ZeroDivisionError):
            return False
    if benchmark == "bigbench_unit_conversion":
        predicted_match = _NUMBER.search(predicted.replace(",", ""))
        target_match = _NUMBER.search(target.replace(",", ""))
        if predicted_match is None or target_match is None:
            return False
        try:
            actual = Decimal(predicted_match.group(0))
            expected = Decimal(target_match.group(0))
        except InvalidOperation:
            return False
        return abs(actual - expected) <= max(abs(expected) * Decimal("0.015"), Decimal("1e-12"))
    if benchmark == "bigbench_date_understanding":
        def parse_date(value: str) -> date:
            match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", value.strip())
            if match is None:
                raise ValueError("date answer is not MM/DD/YYYY")
            return date(int(match[3]), int(match[1]), int(match[2]))

        try:
            return parse_date(predicted) == parse_date(target)
        except ValueError:
            return False
    return predicted.casefold().replace(" ", "-") == target.casefold().replace(" ", "-")


def run_public_compute_example(
    row: dict[str, Any], backend: HuggingFaceBackend, *, condition: str, seed: int
) -> dict[str, Any]:
    if condition not in PUBLIC_COMPUTE_CONDITIONS:
        raise ValueError(f"unsupported public compute condition: {condition}")
    registered = dict(row["registered"])
    first_text = ""
    second_text = ""
    selected_intent = None
    malformed = False
    assistance_valid = False
    prompt_tokens = generated_tokens = model_calls = wall_time_ns = 0

    def generate(prompt: str, generation_seed: int) -> str:
        nonlocal prompt_tokens, generated_tokens, model_calls, wall_time_ns
        result = backend.generate(prompt, seed=generation_seed)
        prompt_tokens += result.prompt_tokens
        generated_tokens += result.generated_tokens
        model_calls += result.model_calls
        wall_time_ns += result.wall_time_ns
        return result.generated_text

    if condition in {"llm_only", "four_tools", "generic_cogcop"}:
        first_text = generate(_condition_prompt(row, condition), seed)
        if condition == "four_tools":
            selected_intent, payload, malformed = _tool_attempt(first_text)

            def resolve(intent: str, request: Any, active_task: str) -> str:
                del active_task
                if request != {"request": "solve"}:
                    raise ValueError("generic tool request is invalid")
                return str(registered["result"])

            if (
                selected_intent
                and not malformed
                and selected_intent == registered["intent"]
            ):
                value = GenericCognitiveGateway(resolve).invoke(
                    GenericToolCall(f"__{selected_intent}", payload),
                    active_task=str(row["benchmark"]),
                )
                assistance_valid = True
                second_text = generate(
                    _result_prompt(row, value, "four-tool R2 result"), seed + 1
                )
        elif condition == "generic_cogcop":
            selected_intent, malformed = _cogcop_attempt(first_text)
            if selected_intent == registered["intent"] and not malformed:
                assistance_valid = True
                second_text = generate(
                    _result_prompt(row, str(registered["result"]), "generic CogCop R2 result"),
                    seed + 1,
                )
    else:
        if condition == "cpu_t1":
            route = _t1(str(row["prompt"]))
            if route == _T1_LABEL[str(row["engine"])]:
                selected_intent = registered["intent"]
        else:
            selected_intent = registered["intent"]
        assistance_valid = selected_intent == registered["intent"]
        prompt = (
            _result_prompt(row, str(registered["result"]), f"{condition} R2 result")
            if assistance_valid
            else _base_prompt(row)
        )
        second_text = generate(prompt, seed)

    rendered = "\n".join(value for value in (first_text, second_text) if value)
    predicted = _extract_answer(rendered)
    return {
        "schema_version": "ccpu.paper2.public_compute_prediction.v1",
        "benchmark": row["benchmark"],
        "example_id": row["example_id"],
        "content_sha256": row["content_sha256"],
        "difficulty": row["difficulty"],
        "difficulty_stratum": row["difficulty_stratum"],
        "target_label": row["target"],
        "condition": condition,
        "model_id": backend.model_id,
        "seed": seed,
        "registered_intent": registered["intent"],
        "selected_intent": selected_intent,
        "registered_engine": registered["engine"],
        "formalization_source": registered["formalization_source"],
        "backend_exact": assistance_valid,
        "assistance_valid": assistance_valid,
        "malformed_assistance": malformed,
        "first_text": first_text,
        "second_text": second_text,
        "predicted_answer": predicted,
        "correct": _score_answer(str(row["benchmark"]), predicted, str(row["target"])),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "model_calls": model_calls,
        "wall_time_ns": wall_time_ns,
    }


def summarize_public_compute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    def summarize(members: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(row["correct"]) for row in members)
        assisted = [row for row in members if row["assistance_valid"]]
        return {
            "count": len(members),
            "accuracy": correct / len(members),
            "accuracy_ci95": wilson_interval(correct, len(members)),
            "scorable_answer_rate": safe_mean(row["predicted_answer"] is not None for row in members),
            "intent_selection_rate": safe_mean(
                row["selected_intent"] == row["registered_intent"] for row in members
            ),
            "assistance_rate": safe_mean(row["assistance_valid"] for row in members),
            "malformed_rate": safe_mean(row["malformed_assistance"] for row in members),
            "backend_exact_rate_on_assisted": safe_mean(row["backend_exact"] for row in assisted),
            "final_accuracy_on_assisted": safe_mean(row["correct"] for row in assisted),
            "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in members),
            "mean_model_calls": safe_mean(row["model_calls"] for row in members),
            "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in members) / 1e6,
        }

    by_condition = [
        {"condition": condition, **summarize(members)}
        for condition, members in sorted(grouped.items())
    ]
    by_benchmark = []
    for condition, members in sorted(grouped.items()):
        for benchmark in PUBLIC_COMPUTE_BENCHMARKS:
            selected = [row for row in members if row["benchmark"] == benchmark]
            if selected:
                by_benchmark.append(
                    {"condition": condition, "benchmark": benchmark, **summarize(selected)}
                )

    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[str(row["example_id"])][str(row["condition"])] = row
    eligible = [
        values
        for values in paired.values()
        if {"four_tools", "cpu_t1"} <= values.keys()
        and not values["four_tools"]["assistance_valid"]
    ]
    rescued = [
        values
        for values in eligible
        if values["cpu_t1"]["assistance_valid"] and values["cpu_t1"]["correct"]
    ]
    return {
        "schema_version": "ccpu.paper2.public_compute_summary.v1",
        "record_count": len(rows),
        "base_question_count": len(paired),
        "by_condition": by_condition,
        "by_benchmark": by_benchmark,
        "automatic_rescue": {
            "eligible_voluntary_misses": len(eligible),
            "rescued_correctly": len(rescued),
            "rate": len(rescued) / len(eligible) if eligible else None,
            "definition": "correct T1 CPU assistance among rows where four tools made no valid call",
        },
        "four_tool_schema": {
            "tool_count": len(generic_tool_schemas()),
            "names": [schema["name"] for schema in generic_tool_schemas()],
            "word_tokens": sum(
                len(current_word_tokens(json.dumps(schema, sort_keys=True)))
                for schema in generic_tool_schemas()
            ),
        },
        "claim_boundary": {
            "clutrr_formalization": "annotated proof replay",
            "gsm8k_formalization": "annotated arithmetic trace",
            "status": "developmental executable slice, not full-suite end-to-end coverage",
        },
    }


def write_public_compute_run(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    model_config: str | Path,
    selection_path: str | Path,
    condition: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = summarize_public_compute(rows)
    summary_path = write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "ccpu.paper2.public_compute_run_manifest.v1",
            "condition": condition,
            "record_count": len(rows),
            "model_config_sha256": file_sha256(model_config),
            "selection_sha256": file_sha256(selection_path),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary_path),
            "environment": environment_manifest(Path(__file__).resolve().parents[3]),
        },
    )
    return summary


def analyze_public_compute_runs(
    prediction_paths: list[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    rows = [row for path in prediction_paths for row in read_jsonl(path)]
    keys = [(row["example_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate public compute example/condition rows")
    conditions = {str(row["condition"]) for row in rows}
    missing = set(PUBLIC_COMPUTE_CONDITIONS) - conditions
    if missing:
        raise ValueError(f"missing public compute conditions: {sorted(missing)}")
    id_sets = {
        condition: {str(row["example_id"]) for row in rows if row["condition"] == condition}
        for condition in conditions
    }
    if len({frozenset(values) for values in id_sets.values()}) != 1:
        raise ValueError("public compute conditions do not contain identical IDs")
    summary = summarize_public_compute(rows)
    summary["prediction_sources"] = [
        {"path": str(path), "sha256": file_sha256(path)} for path in prediction_paths
    ]
    output = Path(output_dir)
    write_json(output / "summary.json", summary)
    _plot_public_compute(summary, output / "public_compute_accuracy.png")
    return summary


def _plot_public_compute(summary: dict[str, Any], output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("public compute plotting requires the analysis extra") from error
    labels = {
        "gsm8k": "GSM8K",
        "bigbench_unit_conversion": "Units",
        "bigbench_date_understanding": "Date",
        "proofwriter_balanced": "ProofWriter",
        "clutrr": "CLUTRR",
    }
    conditions = list(PUBLIC_COMPUTE_CONDITIONS)
    lookup = {
        (row["condition"], row["benchmark"]): row for row in summary["by_benchmark"]
    }
    figure, axis = plt.subplots(figsize=(10, 4.8))
    x = list(range(len(PUBLIC_COMPUTE_BENCHMARKS)))
    width = 0.16
    for index, condition in enumerate(conditions):
        values = [lookup[(condition, benchmark)]["accuracy"] for benchmark in PUBLIC_COMPUTE_BENCHMARKS]
        axis.bar(
            [value + (index - 2) * width for value in x],
            values,
            width,
            label=condition.replace("_", " "),
        )
    axis.set_xticks(x, [labels[benchmark] for benchmark in PUBLIC_COMPUTE_BENCHMARKS])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Exact final-answer accuracy")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
