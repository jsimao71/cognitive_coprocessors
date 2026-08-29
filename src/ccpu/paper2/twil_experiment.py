"""Empirical TwIL/SmolLM3 comparison and deterministic reuse measurements."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import canonical_json, read_jsonl
from ccpu.common.metrics import safe_mean

from .next_experiment import extract_block
from .runtime import HeterogeneousRuntime, StrictEventRouter

_ANSWER_WORD = re.compile(r"\b(true|false|unknown)\b", re.IGNORECASE)

_INTERFACE_DEMOS = """Use an exact engine only for an unambiguous supported computation.
Return exactly one typed fenced block and no explanation. For ambiguous or unsupported semantic
questions return NO_EXECUTION followed by your answer: true, false, or unknown.

Examples:
Question: Links a to b and b to c. Is c reachable from a?
Output:
```datalog
fact link(a,b)
fact link(b,c)
query reachable(a,c)
```
Question: A robin is a bird and a bird is an animal. Is a robin an animal?
Output:
```graph
isa robin bird
isa bird animal
query isa robin animal
```
Question: What is 17 times 19?
Output:
```calculator
17 * 19
```
Question: What date is 10 days after 2031-01-01?
Output:
```date
add 2031-01-01 P10D
```
Question: Convert 2 miles to kilometers.
Output:
```units
convert 2 mile -> kilometer
```"""


def neural_prompt(row: dict[str, Any]) -> str:
    return (
        "Answer the question with only its exact answer. Use true, false, or unknown for "
        f"logical questions. Do not explain.\nQuestion: {row['prompt']}\nAnswer:"
    )


def hybrid_prompt(row: dict[str, Any]) -> str:
    return f"{_INTERFACE_DEMOS}\nQuestion: {row['prompt']}\nOutput:"


def _answer_correct(text: str, answer: str) -> bool:
    stripped = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    expected = answer.strip().casefold()
    if expected in {"true", "false", "unknown"}:
        found = _ANSWER_WORD.findall(stripped)
        return bool(found) and found[-1].casefold() == expected
    return bool(re.search(rf"(?<![\w.]){re.escape(answer)}(?![\w.])", stripped))


def _components(request: Any) -> dict[str, Any]:
    if request is None:
        return {}
    payload = dict(request.payload)
    if request.engine == "horn":
        facts = payload.get("facts", [])
        return {
            "facts": facts,
            "relations": sorted({str(row.get("predicate")) for row in facts}),
            "rules": payload.get("rules", []),
            "query": payload.get("query"),
        }
    if request.engine == "frame_graph":
        return {
            "facts": {"isa": payload.get("isa", []), "frames": payload.get("frames", [])},
            "relations": [request.operation],
            "rules": [],
            "query": payload.get("query"),
        }
    return {
        "facts": payload,
        "relations": [request.operation],
        "rules": [],
        "query": payload,
    }


def _score_formalization(row: dict[str, Any], generated: str) -> tuple[dict[str, Any], str | None]:
    block = extract_block(generated)
    router = StrictEventRouter()
    gold = router.parse(str(row["target"]), event_id=f"{row['example_id']}:gold") if row["should_trigger"] else None
    try:
        predicted = router.parse(block, event_id=str(row["example_id"])) if block else None
    except Exception:  # noqa: BLE001 - malformed formalizations are measured outcomes
        predicted = None
    expected_trigger = bool(row["should_trigger"])
    detected = block is not None
    gold_parts, predicted_parts = _components(gold), _components(predicted)
    engine_correct = bool(predicted and gold and predicted.engine == gold.engine)
    return (
        {
            "semantic_intent_correct": detected == expected_trigger,
            "engine_family_correct": engine_correct if expected_trigger else not detected,
            "entities_facts_correct": bool(
                engine_correct and predicted_parts.get("facts") == gold_parts.get("facts")
            ) if expected_trigger else not detected,
            "predicates_relations_correct": bool(
                engine_correct and predicted_parts.get("relations") == gold_parts.get("relations")
            ) if expected_trigger else not detected,
            "rules_constraints_correct": bool(
                engine_correct and predicted_parts.get("rules") == gold_parts.get("rules")
            ) if expected_trigger else not detected,
            "query_correct": bool(
                engine_correct and predicted_parts.get("query") == gold_parts.get("query")
            ) if expected_trigger else not detected,
            "formalization_correct": bool(
                engine_correct and predicted_parts == gold_parts
            ) if expected_trigger else not detected,
            "false_activation": detected and not expected_trigger,
            "malformed": block is not None and predicted is None,
        },
        block,
    )


def run_twil_condition(
    rows: list[dict[str, Any]], *, backend: Any | None, condition: str, seed: int
) -> list[dict[str, Any]]:
    if condition not in {"neural", "hybrid", "oracle"}:
        raise ValueError(f"unsupported TwIL comparison condition: {condition}")
    if condition != "oracle" and backend is None:
        raise ValueError("model conditions require a generation backend")
    runtime = HeterogeneousRuntime(max_state_items=4096)
    predictions = []
    for row in rows:
        generation = None
        if condition == "oracle":
            generated = (
                str(row["target"])
                if row["should_trigger"]
                else f"NO_EXECUTION {row['answer']}"
            )
        else:
            prompt = neural_prompt(row) if condition == "neural" else hybrid_prompt(row)
            generation = backend.generate(prompt, seed=seed)
            generated = generation.generated_text
        score, block = _score_formalization(row, generated)
        result = None
        engine_time_ns = 0
        if condition in {"hybrid", "oracle"} and block:
            started = time.perf_counter_ns()
            result = runtime.execute_event(block, event_id=str(row["example_id"]))
            engine_time_ns = time.perf_counter_ns() - started
        if row["should_trigger"]:
            neural_correct = _answer_correct(generated, str(row["answer"]))
            execution_correct = bool(
                score["formalization_correct"]
                and result
                and result.ok
                and result.display == row["answer"]
            )
            final_correct = neural_correct if condition == "neural" else execution_correct
        else:
            neural_correct = _answer_correct(generated, str(row["answer"]))
            execution_correct = not score["false_activation"]
            final_correct = neural_correct and not score["false_activation"]
        predictions.append(
            {
                "schema_version": "ccpu.paper2.twil_prediction.v1",
                **row,
                "model_id": backend.model_id if backend is not None else "oracle",
                "condition": condition,
                "generated_text": generated,
                **score,
                "neural_answer_correct": neural_correct,
                "engine_executed": bool(result and result.ok),
                "execution_correct": execution_correct,
                "execution_correct_given_formalization": bool(
                    score["formalization_correct"] and execution_correct
                ),
                "result_integration_correct": final_correct,
                "final_correct": final_correct,
                "engine_time_ns": engine_time_ns,
                "prompt_tokens": generation.prompt_tokens if generation else 0,
                "generated_tokens": generation.generated_tokens if generation else 0,
                "accelerator_time_ns": generation.wall_time_ns if generation else 0,
                "backend_metadata": generation.metadata if generation else {"empirical": False},
                "truncated_at_budget": bool(
                    generation
                    and generation.generated_tokens >= backend.config.max_new_tokens
                ),
                "state_items": len(runtime.state.items),
                "state_bytes": len(canonical_json([item.to_dict() for item in runtime.state.items])),
            }
        )
    return predictions


def _failure_type(row: dict[str, Any]) -> str:
    if row["final_correct"]:
        return "none"
    if row.get("truncated_at_budget"):
        return "truncated"
    if row["condition"] == "neural":
        return "neural_answer"
    if row["should_trigger"]:
        if not row["semantic_intent_correct"]:
            return "missed_delegation"
        if row["malformed"]:
            return "malformed_ir"
        if not row["engine_family_correct"]:
            return "wrong_engine"
        if not row["entities_facts_correct"]:
            return "wrong_facts"
        if not row["query_correct"]:
            return "wrong_query"
        if not row["engine_executed"]:
            return "runtime_rejection"
        return "wrong_execution"
    if row["false_activation"]:
        return "false_activation"
    return "semantic_answer"


def rescore_twil_predictions(
    rows: list[dict[str, Any]], *, max_new_tokens: int = 160
) -> list[dict[str, Any]]:
    """Recompute labels from preserved generations under strict semantic scoring."""
    rescored = []
    for original in rows:
        row = dict(original)
        score, block = _score_formalization(row, str(row["generated_text"]))
        result = None
        engine_time_ns = 0
        if row["condition"] in {"hybrid", "oracle"} and block:
            runtime = HeterogeneousRuntime(max_state_items=4096)
            started = time.perf_counter_ns()
            result = runtime.execute_event(block, event_id=str(row["example_id"]))
            engine_time_ns = time.perf_counter_ns() - started
        neural_correct = _answer_correct(str(row["generated_text"]), str(row["answer"]))
        if row["should_trigger"]:
            execution_correct = bool(
                score["formalization_correct"]
                and result
                and result.ok
                and result.display == row["answer"]
            )
            final_correct = (
                neural_correct if row["condition"] == "neural" else execution_correct
            )
        else:
            execution_correct = not score["false_activation"]
            final_correct = neural_correct and not score["false_activation"]
        row.update(
            {
                "schema_version": "ccpu.paper2.twil_prediction.v2",
                **score,
                "neural_answer_correct": neural_correct,
                "engine_executed": bool(result and result.ok),
                "execution_correct": execution_correct,
                "execution_correct_given_formalization": bool(
                    score["formalization_correct"] and execution_correct
                ),
                "result_integration_correct": final_correct,
                "final_correct": final_correct,
                "engine_time_ns": engine_time_ns,
                "truncated_at_budget": bool(
                    row["condition"] != "oracle"
                    and int(row["generated_tokens"]) >= max_new_tokens
                ),
            }
        )
        row["failure_type"] = _failure_type(row)
        rescored.append(row)
    return rescored


def summarize_twil(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_id"]), str(row["condition"]), str(row["family"]))].append(row)
    cells = []
    for (model_id, condition, family), members in sorted(groups.items()):
        exact = [row for row in members if row["should_trigger"]]
        valid_ir = [row for row in exact if row["formalization_correct"]]
        cells.append(
            {
                "model_id": model_id,
                "condition": condition,
                "family": family,
                "count": len(members),
                "semantic_intent_accuracy": safe_mean(
                    row["semantic_intent_correct"] for row in members
                ),
                "engine_selection_accuracy": safe_mean(
                    row["engine_family_correct"] for row in exact
                ),
                "formalization_accuracy": safe_mean(
                    row["formalization_correct"] for row in exact
                ),
                "execution_given_valid_ir": safe_mean(
                    row["execution_correct"] for row in valid_ir
                ),
                "final_accuracy": safe_mean(row["final_correct"] for row in members),
                "closure_recall": safe_mean(row["final_correct"] for row in exact),
                "false_activation_rate": safe_mean(
                    row["false_activation"] for row in members if not row["should_trigger"]
                ),
                "mean_prompt_tokens": safe_mean(row["prompt_tokens"] for row in members),
                "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in members),
                "mean_accelerator_time_ms": safe_mean(
                    row["accelerator_time_ns"] for row in members
                ) / 1e6,
                "mean_engine_time_ms": safe_mean(row["engine_time_ns"] for row in members) / 1e6,
            }
        )
    depth_cells = []
    for model_id, condition, family in sorted(groups):
        if family not in {"datalog", "graph"}:
            continue
        members = groups[(model_id, condition, family)]
        for depth in sorted({int(row["depth"]) for row in members}):
            cell = [row for row in members if row["depth"] == depth]
            depth_cells.append(
                {
                    "model_id": model_id,
                    "condition": condition,
                    "family": family,
                    "depth": depth,
                    "count": len(cell),
                    "accuracy": safe_mean(row["final_correct"] for row in cell),
                    "formalization_accuracy": safe_mean(
                        row["formalization_correct"] for row in cell
                    ),
                    "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in cell),
                    "mean_accelerator_time_ms": safe_mean(
                        row["accelerator_time_ns"] for row in cell
                    ) / 1e6,
                    "mean_engine_time_ms": safe_mean(row["engine_time_ns"] for row in cell) / 1e6,
                }
            )
    return {
        "schema_version": "ccpu.paper2.twil_summary.v1",
        "prediction_count": len(rows),
        "by_family": cells,
        "by_depth": depth_cells,
    }


def run_reuse_workload(query_counts: tuple[int, ...] = (1, 5, 20, 100)) -> list[dict[str, Any]]:
    rows = []
    for family in ("datalog", "graph"):
        if family == "datalog":
            facts = "\n".join(f"fact link(r{index},r{index + 1})" for index in range(8))
            build = f"```datalog\n{facts}\nquery reachable(r0,r8)\n```"
            query = "```datalog\nquery reachable(r0,r8)\n```"
        else:
            facts = "\n".join(f"isa g{index} g{index + 1}" for index in range(8))
            build = f"```graph\n{facts}\nquery isa g0 g8\n```"
            query = "```graph\nquery isa g0 g8\n```"
        for count in query_counts:
            persistent = HeterogeneousRuntime(max_state_items=4096)
            started = time.perf_counter_ns()
            initial = persistent.execute_event(build, event_id=f"{family}:build")
            build_time = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            reused = [
                persistent.execute_event(query, event_id=f"{family}:reuse:{index}")
                for index in range(count)
            ]
            reuse_time = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            fresh = []
            for index in range(count):
                runtime = HeterogeneousRuntime(max_state_items=4096)
                fresh.append(runtime.execute_event(build, event_id=f"{family}:fresh:{index}"))
            fresh_time = time.perf_counter_ns() - started
            rows.append(
                {
                    "schema_version": "ccpu.paper2.twil_reuse.v1",
                    "family": family,
                    "query_count": count,
                    "build_correct": bool(initial and initial.ok and initial.display == "true"),
                    "reuse_accuracy": safe_mean(
                        bool(result and result.ok and result.display == "true") for result in reused
                    ),
                    "fresh_accuracy": safe_mean(
                        bool(result and result.ok and result.display == "true") for result in fresh
                    ),
                    "build_time_ms": build_time / 1e6,
                    "reuse_total_ms": reuse_time / 1e6,
                    "fresh_total_ms": fresh_time / 1e6,
                    "persistent_amortized_ms": (build_time + reuse_time) / count / 1e6,
                    "fresh_amortized_ms": fresh_time / count / 1e6,
                    "speedup": fresh_time / (build_time + reuse_time),
                    "state_items": len(persistent.state.items),
                }
            )
    return rows


def load_twil_dataset(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
