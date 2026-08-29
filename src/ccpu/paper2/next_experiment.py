"""Model-facing five-engine placement and capability-count experiment."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import canonical_json, read_jsonl
from ccpu.common.metrics import safe_mean
from ccpu.paper1.generation import HuggingFaceBackend

from .benchmark_next import ENGINE_CATALOGS
from .runtime import HeterogeneousRuntime, StrictEventRouter

_BLOCK = re.compile(
    r"```(?:calculator|datalog|graph|date|units)\r?\n.*?\r?\n```",
    re.DOTALL,
)
_CALCULATOR_REFLEX = re.compile(r"^Compute the exact product of (\d+) and (\d+)\.$")
_DATALOG_REFLEX = re.compile(
    r"^Given directed links ([a-z0-9_]+) to ([a-z0-9_]+) and "
    r"([a-z0-9_]+) to ([a-z0-9_]+), determine exactly whether "
    r"([a-z0-9_]+) is reachable from ([a-z0-9_]+)\.$"
)
_GRAPH_REFLEX = re.compile(
    r"^In an ISA hierarchy, ([a-z0-9_]+) is a ([a-z0-9_]+) and "
    r"([a-z0-9_]+) is a ([a-z0-9_]+)\. Is ([a-z0-9_]+) a ([a-z0-9_]+)\?$"
)
_DATE_REFLEX = re.compile(
    r"^What ISO calendar date is (\d+) days after (\d{4}-\d{2}-\d{2})\?$"
)
_UNITS_REFLEX = re.compile(r"^Convert exactly (\d+(?:\.\d+)?) miles to kilometers\.$")
_CONTROL_REFLEX = re.compile(
    r"^Repeat the already supplied label [A-Z0-9-]+ without computation\.$"
)
_ENGINE_NAMES = {
    "calculator": "calculator",
    "datalog": "horn",
    "graph": "frame_graph",
    "date": "date_time",
    "units": "units",
}

_DEMOS = {
    "calculator": "Question: Exact product of 17 and 19.\nOutput:\n```calculator\n17 * 19\n```",
    "datalog": (
        "Question: Links demo_a to demo_b and demo_b to demo_c. Is demo_c reachable "
        "from demo_a?\nOutput:\n```datalog\nfact link(demo_a,demo_b)\n"
        "fact link(demo_b,demo_c)\nquery reachable(demo_a,demo_c)\n```"
    ),
    "graph": (
        "Question: A tern is a bird and a bird is an animal. Is a tern an animal?\n"
        "Output:\n```graph\nisa tern bird\nisa bird animal\nquery isa tern animal\n```"
    ),
    "date": "Question: Date 10 days after 2031-01-01?\nOutput:\n```date\nadd 2031-01-01 P10D\n```",
    "units": (
        "Question: Convert 2 miles to kilometers.\nOutput:\n"
        "```units\nconvert 2 mile -> kilometer\n```"
    ),
}


def interface_prompt(row: dict[str, Any], condition: str, catalog_size: int) -> str:
    contract = (
        "Select a deterministic engine only when needed. Emit exactly one complete typed fenced "
        "block, or NO_EXECUTION for a supplied-answer control. Do not compute the result yourself."
    )
    if condition == "weights":
        prefix = contract
    elif condition == "context":
        demos = "\n\n".join(_DEMOS[name] for name in ENGINE_CATALOGS[catalog_size])
        prefix = f"{contract}\n\n{demos}"
    elif condition == "explicit_tools":
        schemas = "\n".join(
            f"Tool {name}: accepts the exact typed {name} fenced schema shown by its name."
            for name in ENGINE_CATALOGS[catalog_size]
        )
        prefix = f"{contract}\nAvailable textual tool schemas:\n{schemas}"
    else:
        raise ValueError(f"unsupported model condition: {condition}")
    return f"{prefix}\n\nQuestion: {row['prompt']}\nOutput:"


def interface_lexical_tokens(condition: str, catalog_size: int) -> int:
    """Count the static contract, demonstrations, or schemas apart from the task."""
    if condition not in {"weights", "context", "explicit_tools"}:
        return 0
    prompt = interface_prompt({"prompt": ""}, condition, catalog_size)
    prefix = prompt.rsplit("\n\nQuestion:", maxsplit=1)[0]
    return len(prefix.split())


def extract_block(text: str) -> str | None:
    matches = _BLOCK.findall(text)
    return matches[0] if len(matches) == 1 else None


def deterministic_reflex(prompt: str) -> str | None:
    """Translate only frozen, anchored benchmark forms; unknown text fails closed."""
    if match := _CALCULATOR_REFLEX.fullmatch(prompt):
        return f"```calculator\n{match[1]} * {match[2]}\n```"
    if match := _DATALOG_REFLEX.fullmatch(prompt):
        return (
            f"```datalog\nfact link({match[1]},{match[2]})\n"
            f"fact link({match[3]},{match[4]})\n"
            f"query reachable({match[6]},{match[5]})\n```"
        )
    if match := _GRAPH_REFLEX.fullmatch(prompt):
        return (
            f"```graph\nisa {match[1]} {match[2]}\nisa {match[3]} {match[4]}\n"
            f"query isa {match[5]} {match[6]}\n```"
        )
    if match := _DATE_REFLEX.fullmatch(prompt):
        return f"```date\nadd {match[2]} P{match[1]}D\n```"
    if match := _UNITS_REFLEX.fullmatch(prompt):
        return f"```units\nconvert {match[1]} mile -> kilometer\n```"
    if _CONTROL_REFLEX.fullmatch(prompt):
        return "NO_EXECUTION"
    return None


def _runtime_for_catalog(catalog_size: int) -> HeterogeneousRuntime:
    return HeterogeneousRuntime(
        enabled_engines={_ENGINE_NAMES[name] for name in ENGINE_CATALOGS[catalog_size]},
        max_state_items=4096,
    )


def _score_generated(
    row: dict[str, Any], generated: str, runtime: HeterogeneousRuntime
) -> tuple[dict[str, Any], Any]:
    block = extract_block(generated)
    selected = block is not None
    expected = bool(row["should_trigger"])
    request = None
    result = None
    started = time.perf_counter_ns()
    if block:
        try:
            request = StrictEventRouter().parse(block, event_id=str(row["example_id"]))
        except Exception:  # noqa: BLE001 - malformed model output is a measured outcome
            request = None
        if request is not None:
            result = runtime.execute_event(block, event_id=str(row["example_id"]))
    engine_duration = time.perf_counter_ns() - started
    expected_engine = _ENGINE_NAMES.get(str(row["engine"]))
    engine_selected = request.engine == expected_engine if request is not None else False
    executed = bool(result and result.ok)
    exact = bool(executed and result.display == row["answer"])
    return (
        {
            "detected": selected,
            "detect_correct": selected == expected,
            "engine_selected": engine_selected if expected else not selected,
            "payload_normalized": request is not None if expected else not selected,
            "executed": executed if expected else not selected,
            "runtime_exact": exact if expected else not selected,
            "false_activation": selected and not expected,
            "wrong_engine": selected and expected and not engine_selected,
            "engine_time_ns": engine_duration,
            "state_items": len(runtime.state.items),
            "state_bytes": len(
                canonical_json([item.to_dict() for item in runtime.state.items]).encode("utf-8")
            ),
            "request": request.to_dict() if request else None,
            "result": result.to_dict() if result else None,
        },
        result,
    )


def run_model_condition(
    *,
    dataset_path: str | Path,
    backend: HuggingFaceBackend,
    condition: str,
    catalog_size: int,
    seed: int,
    assess_use: bool = True,
) -> list[dict[str, Any]]:
    allowed = set(ENGINE_CATALOGS[catalog_size])
    examples = [
        row
        for row in read_jsonl(dataset_path)
        if row["engine"] == "control" or row["engine"] in allowed
    ]
    runtime = _runtime_for_catalog(catalog_size)
    rows = []
    for row in examples:
        if condition == "no_engine":
            prompt = f"Answer with only the exact answer and no explanation.\nQuestion: {row['prompt']}"
            generation = backend.generate(prompt, seed=seed)
            answer = generation.generated_text.strip()
            rows.append(
                {
                    "schema_version": "ccpu.paper2.next_prediction.v1",
                    "example_id": row["example_id"],
                    "model_id": backend.model_id,
                    "condition": condition,
                    "catalog_size": catalog_size,
                    "engine": row["engine"],
                    "should_trigger": row["should_trigger"],
                    "generated_text": generation.generated_text,
                    "gold_target": row["target"],
                    "gold_answer": row["answer"],
                    "detected": False,
                    "detect_correct": not row["should_trigger"],
                    "engine_selected": False,
                    "payload_normalized": False,
                    "executed": False,
                    "runtime_exact": False,
                    "false_activation": False,
                    "wrong_engine": False,
                    "engine_time_ns": 0,
                    "state_items": 0,
                    "state_bytes": 2,
                    "request": None,
                    "result": None,
                    "reinjected": False,
                    "use_text": None,
                    "use_correct": None,
                    "use_assessed": False,
                    "ignored_result": False,
                    "overridden_result": False,
                    "wrong_reinterpretation": False,
                    "final_correct": answer == str(row["answer"]),
                    "prompt_tokens": generation.prompt_tokens,
                    "generated_tokens": generation.generated_tokens,
                    "reinjected_tokens": 0,
                    "model_calls": 1,
                    "wall_time_ns": generation.wall_time_ns,
                    "backend_metadata": generation.metadata,
                }
            )
            continue
        prompt = interface_prompt(row, condition, catalog_size)
        generation = backend.generate(prompt, seed=seed)
        score, result = _score_generated(row, generation.generated_text, runtime)
        use_text = None
        use_correct = None
        use_generation = None
        if assess_use and result is not None and result.ok:
            use_prompt = (
                f"Question: {row['prompt']}\nThe deterministic {result.engine} runtime returned "
                f"the exact result: {result.display}\nAnswer using exactly that result and no explanation."
            )
            use_generation = backend.generate(use_prompt, seed=seed)
            use_text = use_generation.generated_text.strip()
            use_correct = use_text == str(row["answer"])
        final_correct = (
            bool(use_correct)
            if assess_use and row["should_trigger"]
            else bool(score["runtime_exact"])
        )
        rows.append(
            {
                "schema_version": "ccpu.paper2.next_prediction.v1",
                "example_id": row["example_id"],
                "model_id": backend.model_id,
                "condition": condition,
                "catalog_size": catalog_size,
                "engine": row["engine"],
                "should_trigger": row["should_trigger"],
                "generated_text": generation.generated_text,
                "gold_target": row["target"],
                "gold_answer": row["answer"],
                **score,
                "reinjected": result is not None and result.ok,
                "use_text": use_text,
                "use_correct": use_correct,
                "use_assessed": use_correct is not None,
                "ignored_result": bool(
                    assess_use and result and result.ok and not use_text
                ),
                "overridden_result": bool(
                    assess_use
                    and result
                    and result.ok
                    and use_text
                    and use_text != str(row["answer"])
                ),
                "wrong_reinterpretation": bool(
                    assess_use
                    and result
                    and result.ok
                    and use_text
                    and str(row["answer"]) not in use_text
                ),
                "final_correct": final_correct,
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens
                + (use_generation.generated_tokens if use_generation else 0),
                "reinjected_tokens": len(str(result.display).split()) if result else 0,
                "model_calls": 1 + int(use_generation is not None),
                "wall_time_ns": generation.wall_time_ns
                + (use_generation.wall_time_ns if use_generation else 0),
                "backend_metadata": generation.metadata,
            }
        )
    return rows


def run_oracle_condition(
    dataset_path: str | Path, *, condition: str, catalog_size: int
) -> list[dict[str, Any]]:
    if condition not in {"runtime", "oracle"}:
        raise ValueError("scripted conditions are runtime or oracle")
    allowed = set(ENGINE_CATALOGS[catalog_size])
    examples = [
        row
        for row in read_jsonl(dataset_path)
        if row["engine"] == "control" or row["engine"] in allowed
    ]
    runtime = _runtime_for_catalog(catalog_size)
    rows = []
    for row in examples:
        if condition == "runtime":
            generated = deterministic_reflex(str(row["prompt"])) or "NO_EXECUTION"
        else:
            generated = row["target"] if row["should_trigger"] else "NO_EXECUTION"
        score, result = _score_generated(row, generated, runtime)
        rows.append(
            {
                "schema_version": "ccpu.paper2.next_prediction.v1",
                "example_id": row["example_id"],
                "model_id": "deterministic_runtime",
                "condition": condition,
                "catalog_size": catalog_size,
                "engine": row["engine"],
                "should_trigger": row["should_trigger"],
                "generated_text": generated,
                "gold_target": row["target"],
                "gold_answer": row["answer"],
                **score,
                "reinjected": result is not None and result.ok,
                "use_text": result.display if result else None,
                "use_correct": bool(result and result.display == row["answer"]),
                "use_assessed": result is not None and result.ok,
                "ignored_result": False,
                "overridden_result": False,
                "wrong_reinterpretation": False,
                "final_correct": bool(score["runtime_exact"]),
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "reinjected_tokens": len(str(result.display).split()) if result else 0,
                "model_calls": 0,
                "wall_time_ns": score["engine_time_ns"],
                "backend_metadata": {
                    "empirical": False,
                    "oracle_target": condition == "oracle",
                },
            }
        )
    return rows


def summarize_next(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Paper 2 evaluation requires rows")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), int(row["catalog_size"]))].append(row)
    summaries = []
    by_engine = []
    for (condition, catalog_size), group in sorted(grouped.items()):
        positives = [row for row in group if row["should_trigger"]]
        controls = [row for row in group if not row["should_trigger"]]
        use_assessed = [row for row in positives if row["use_correct"] is not None]
        summaries.append(
            {
                "condition": condition,
                "catalog_size": catalog_size,
                "count": len(group),
                "detect_accuracy": safe_mean(row["detect_correct"] for row in group),
                "engine_selection_accuracy": safe_mean(
                    row["engine_selected"] for row in positives
                ),
                "payload_normalization_rate": safe_mean(
                    row["payload_normalized"] for row in positives
                ),
                "execution_rate": safe_mean(row["executed"] for row in positives),
                "runtime_exact_rate": safe_mean(row["runtime_exact"] for row in positives),
                "use_rate": safe_mean(
                    row["use_correct"] for row in use_assessed
                ),
                "use_assessment_coverage": len(use_assessed) / len(positives),
                "final_accuracy": safe_mean(row["final_correct"] for row in group),
                "false_activation_rate": safe_mean(row["false_activation"] for row in controls),
                "wrong_engine_rate": safe_mean(row["wrong_engine"] for row in positives),
                "ignored_result_rate": safe_mean(
                    row["ignored_result"] for row in use_assessed
                ),
                "overridden_result_rate": safe_mean(
                    row["overridden_result"] for row in use_assessed
                ),
                "wrong_reinterpretation_rate": safe_mean(
                    row["wrong_reinterpretation"] for row in use_assessed
                ),
                "mean_prompt_tokens": safe_mean(row["prompt_tokens"] for row in group),
                "interface_lexical_tokens": interface_lexical_tokens(
                    condition, catalog_size
                ),
                "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in group),
                "mean_reinjected_tokens": safe_mean(row["reinjected_tokens"] for row in group),
                "mean_model_calls": safe_mean(row["model_calls"] for row in group),
                "mean_engine_time_ms": safe_mean(row["engine_time_ns"] for row in group) / 1e6,
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in group) / 1e6,
                "mean_state_items": safe_mean(row["state_items"] for row in group),
                "mean_state_bytes": safe_mean(row["state_bytes"] for row in group),
            }
        )
        for engine in sorted({str(row["engine"]) for row in group}):
            members = [row for row in group if row["engine"] == engine]
            by_engine.append(
                {
                    "condition": condition,
                    "catalog_size": catalog_size,
                    "engine": engine,
                    "count": len(members),
                    "detect_accuracy": safe_mean(row["detect_correct"] for row in members),
                    "engine_selection_accuracy": safe_mean(
                        row["engine_selected"] for row in members
                    ),
                    "runtime_exact_rate": safe_mean(row["runtime_exact"] for row in members),
                    "final_accuracy": safe_mean(row["final_correct"] for row in members),
                    "mean_generated_tokens": safe_mean(
                        row["generated_tokens"] for row in members
                    ),
                    "mean_prompt_tokens": safe_mean(row["prompt_tokens"] for row in members),
                    "mean_reinjected_tokens": safe_mean(
                        row["reinjected_tokens"] for row in members
                    ),
                    "mean_model_calls": safe_mean(row["model_calls"] for row in members),
                    "mean_engine_time_ms": safe_mean(
                        row["engine_time_ns"] for row in members
                    )
                    / 1e6,
                    "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in members)
                    / 1e6,
                    "mean_state_items": safe_mean(row["state_items"] for row in members),
                    "mean_state_bytes": safe_mean(row["state_bytes"] for row in members),
                }
            )
    return {
        "schema_version": "ccpu.paper2.next_evaluation.v1",
        "prediction_count": len(rows),
        "by_condition_catalog": summaries,
        "by_engine": by_engine,
    }
