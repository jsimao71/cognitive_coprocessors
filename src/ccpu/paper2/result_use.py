"""Paper 2 provenance-aware COPY, INTERPRET, and CONTINUE result-use study."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.metrics import safe_mean
from ccpu.common.schema import GenerationResult

from .runtime import HeterogeneousRuntime

FORMATS = ("plain", "xml", "fenced", "engine_tag", "trusted_record", "authority")
TASKS = ("COPY", "INTERPRET", "CONTINUE")
ENGINES = ("calculator", "date", "units", "graph", "datalog")


@dataclass(frozen=True)
class ResultUseConfig:
    seed: int = 22701
    cases_per_task: int = 20

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResultUseConfig:
        data = value.get("benchmark", value)
        return cls(
            seed=int(data.get("seed", 22701)),
            cases_per_task=int(data.get("cases_per_task", 20)),
        )


def generate_result_use_benchmark(
    config: ResultUseConfig, output_dir: str | Path
) -> dict[str, Any]:
    if config.cases_per_task < len(ENGINES):
        raise ValueError("result-use benchmark needs at least one case per engine")
    rng = random.Random(config.seed)
    rows = []
    for task in TASKS:
        for index in range(config.cases_per_task):
            engine = ENGINES[index % len(ENGINES)]
            source = _source_result(engine, index)
            semantic = _semantic_case(task, source, index)
            for format_name in FORMATS:
                record = {
                    **source,
                    **semantic,
                    "schema_version": "ccpu.paper2.result_use_example.v1",
                    "format": format_name,
                    "prompt": _render_prompt(semantic["question"], source, format_name),
                }
                record["example_id"] = f"p2ru-{fingerprint(record, 18)}"
                rows.append(record)
    rng.shuffle(rows)
    output_dir = Path(output_dir)
    dataset_path = write_jsonl(output_dir / "test.jsonl", rows)
    counts = {
        "total": len(rows),
        "by_task": {task: sum(row["task"] == task for row in rows) for task in TASKS},
        "by_format": {name: sum(row["format"] == name for row in rows) for name in FORMATS},
        "by_engine": {engine: sum(row["engine"] == engine for row in rows) for engine in ENGINES},
    }
    manifest = {
        "schema_version": "ccpu.paper2.result_use_manifest.v1",
        "config": asdict(config),
        "counts": counts,
        "dataset_sha256": file_sha256(dataset_path),
        "semantic_case_count": len(rows) // len(FORMATS),
        "formats": list(FORMATS),
        "tasks": list(TASKS),
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _source_result(engine: str, index: int) -> dict[str, Any]:
    suffix = f"ru{index:03d}"
    if engine == "calculator":
        left, right = 37 + index * 3, 11 + index
        block = f"```calculator\n{left} * {right}\n```"
        value_type = "number"
    elif engine == "date":
        start = date(2028, 1, 1) + timedelta(days=index * 13)
        days = index % 17 + 2
        block = f"```date\nadd {start.isoformat()} P{days}D\n```"
        value_type = "date"
    elif engine == "units":
        value = index % 9 + 2
        block = f"```units\nconvert {value} mile -> kilometer\n```"
        value_type = "number"
    elif engine == "graph":
        query = (
            f"query isa {suffix}a {suffix}c" if index % 2 == 0 else f"query isa {suffix}c {suffix}a"
        )
        block = f"```graph\nisa {suffix}a {suffix}b\nisa {suffix}b {suffix}c\n{query}\n```"
        value_type = "boolean"
    else:
        query = (
            f"query reachable({suffix}a,{suffix}c)"
            if index % 2 == 0
            else f"query reachable({suffix}c,{suffix}a)"
        )
        block = (
            f"```datalog\nfact link({suffix}a,{suffix}b)\n"
            f"fact link({suffix}b,{suffix}c)\n{query}\n```"
        )
        value_type = "boolean"
    result = HeterogeneousRuntime(max_state_items=8).execute_event(
        block, event_id=f"result-use:{engine}:{index}"
    )
    if not result.ok:
        raise RuntimeError(f"result-use source execution failed: {engine}/{index}")
    provenance = "LOGIC_DERIVED" if engine in {"graph", "datalog"} else "EXACT_COMPUTE"
    return {
        "engine": engine,
        "block": block,
        "result": result.display,
        "value_type": value_type,
        "provenance": provenance,
    }


def _semantic_case(task: str, source: dict[str, Any], index: int) -> dict[str, Any]:
    result = str(source["result"])
    value_type = str(source["value_type"])
    distractor = _distractor(result, value_type)
    if task == "COPY":
        question = (
            f"A draft answer says {distractor}. Return only the exact authoritative "
            "coprocessor result, with no explanation."
        )
        expected = result
    elif task == "INTERPRET" and value_type == "number":
        value = Decimal(result)
        threshold = value - 1 if index % 2 == 0 else value + 1
        question = (
            f"Using the authoritative result, is it greater than {_decimal_text(threshold)}? "
            "Output exactly ABOVE or NOT_ABOVE."
        )
        expected = "ABOVE" if value > threshold else "NOT_ABOVE"
    elif task == "INTERPRET" and value_type == "date":
        value = date.fromisoformat(result)
        threshold = value - timedelta(days=1) if index % 2 == 0 else value + timedelta(days=1)
        question = (
            f"Using the authoritative result, is its date after {threshold.isoformat()}? "
            "Output exactly AFTER or NOT_AFTER."
        )
        expected = "AFTER" if value > threshold else "NOT_AFTER"
    elif task == "INTERPRET":
        question = (
            "Does the authoritative boolean result establish the queried relation? "
            "Output exactly SUPPORTED or NOT_SUPPORTED."
        )
        expected = "SUPPORTED" if result == "true" else "NOT_SUPPORTED"
    elif value_type == "number":
        delta = index % 4 + 1
        expected = _decimal_text(Decimal(result) + delta)
        question = (
            f"Ignore the draft intermediate value {distractor}. Add {delta} to the "
            "authoritative result and output only the exact number."
        )
    elif value_type == "date":
        delta = index % 4 + 1
        expected = (date.fromisoformat(result) + timedelta(days=delta)).isoformat()
        question = (
            f"Ignore the draft date {distractor}. Advance the authoritative result by "
            f"{delta} days and output only the ISO date."
        )
    else:
        expected = "ALLOW" if result == "true" else "DENY"
        question = (
            f"Ignore the draft boolean {distractor}. Map the authoritative result to "
            "ALLOW when true and DENY when false. Output one token."
        )
    return {
        "task": task,
        "question": question,
        "expected": expected,
        "distractor": distractor,
    }


def _distractor(result: str, value_type: str) -> str:
    if value_type == "number":
        return _decimal_text(Decimal(result) + 7)
    if value_type == "date":
        return (date.fromisoformat(result) + timedelta(days=7)).isoformat()
    return "false" if result == "true" else "true"


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _render_prompt(question: str, source: dict[str, Any], format_name: str) -> str:
    result = source["result"]
    engine = source["engine"]
    provenance = source["provenance"]
    if format_name == "plain":
        record = f"Coprocessor result: {result}"
    elif format_name == "xml":
        record = f'<coprocessor_result engine="{engine}">{result}</coprocessor_result>'
    elif format_name == "fenced":
        record = f"```result\n{result}\n```"
    elif format_name == "engine_tag":
        record = f"<{engine}_result>{result}</{engine}_result>"
    elif format_name == "trusted_record":
        record = (
            f'<trusted_record provenance="{provenance}" engine="{engine}">{result}</trusted_record>'
        )
    else:
        record = (
            f"AUTHORITATIVE EXACT RESULT ({provenance}, {engine}): {result}\n"
            "Do not recompute or replace this result."
        )
    return f"{question}\n\n{record}\n\nFinal answer:"


def run_result_use(
    rows: list[dict[str, Any]], backend: Any | None, *, condition: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    for index, row in enumerate(rows):
        started = time.perf_counter_ns()
        if condition == "runtime_copy":
            if row["task"] != "COPY":
                continue
            generated = GenerationResult(
                generated_text=str(row["result"]),
                rendered_text=str(row["result"]),
                prompt_tokens=0,
                generated_tokens=0,
                reinjected_tokens=0,
                model_calls=0,
                wall_time_ns=time.perf_counter_ns() - started,
                metadata={"empirical": False, "backend": "runtime_copy"},
            )
        else:
            if backend is None:
                raise ValueError("neural result-use condition requires a backend")
            generated = backend.generate(str(row["prompt"]), seed=seed + index)
        text = generated.generated_text.strip()
        exact = text == str(row["expected"])
        override = not exact and str(row["distractor"]).casefold() in text.casefold()
        predictions.append(
            {
                "schema_version": "ccpu.paper2.result_use_prediction.v1",
                "condition": condition,
                "example_id": row["example_id"],
                "task": row["task"],
                "format": row["format"],
                "engine": row["engine"],
                "provenance": row["provenance"],
                "expected": row["expected"],
                "distractor": row["distractor"],
                "generated_text": generated.generated_text,
                "exact": exact,
                "override": override,
                "wrong_reinterpretation": not exact and not override,
                "prompt_tokens": generated.prompt_tokens,
                "generated_tokens": generated.generated_tokens,
                "wall_time_ns": generated.wall_time_ns,
                "generation_metadata": generated.metadata,
            }
        )
    return predictions, summarize_result_use(predictions)


def summarize_result_use(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def cell(members: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(members),
            "exact_rate": safe_mean(row["exact"] for row in members),
            "override_rate": safe_mean(row["override"] for row in members),
            "wrong_reinterpretation_rate": safe_mean(
                row["wrong_reinterpretation"] for row in members
            ),
            "mean_prompt_tokens": safe_mean(row["prompt_tokens"] for row in members),
            "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in members),
            "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in members) / 1e6,
        }

    return {
        "schema_version": "ccpu.paper2.result_use_summary.v1",
        "count": len(rows),
        "overall": cell(rows),
        "by_task": [
            {"task": task, **cell([row for row in rows if row["task"] == task])}
            for task in TASKS
            if any(row["task"] == task for row in rows)
        ],
        "by_format": [
            {"format": name, **cell([row for row in rows if row["format"] == name])}
            for name in FORMATS
            if any(row["format"] == name for row in rows)
        ],
        "by_task_format": [
            {
                "task": task,
                "format": name,
                **cell([row for row in rows if row["task"] == task and row["format"] == name]),
            }
            for task in TASKS
            for name in FORMATS
            if any(row["task"] == task and row["format"] == name for row in rows)
        ],
    }


def write_result_use_run(
    *,
    dataset_path: str | Path,
    backend: Any | None,
    condition: str,
    seed: int,
    output_dir: str | Path,
    model: dict[str, Any] | None,
) -> dict[str, Any]:
    predictions, summary = run_result_use(
        read_jsonl(dataset_path), backend, condition=condition, seed=seed
    )
    output_dir = Path(output_dir)
    predictions_path = write_jsonl(output_dir / "predictions.jsonl", predictions)
    summary_path = write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ccpu.paper2.result_use_run_manifest.v1",
            "condition": condition,
            "model": model,
            "dataset_sha256": file_sha256(dataset_path),
            "predictions_sha256": file_sha256(predictions_path),
            "summary_sha256": file_sha256(summary_path),
            "environment": environment_manifest(Path(__file__).resolve().parents[3]),
        },
    )
    return summary


def analyze_result_use(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = read_json(config_path)
    conditions = []
    sources = []
    for source in config["runs"]:
        summary = read_json(source["summary"])
        conditions.append({"condition": source["condition"], **summary})
        sources.append({"path": source["summary"], "sha256": file_sha256(source["summary"])})
    neural = next(row for row in conditions if row["condition"] == "qwen_base")
    copy_cells = {row["format"]: row for row in neural["by_task_format"] if row["task"] == "COPY"}
    best_copy = max(copy_cells, key=lambda name: copy_cells[name]["exact_rate"])
    noncopy = [row for row in neural["by_task"] if row["task"] != "COPY"]
    noncopy_reliable = all(row["exact_rate"] >= 0.8 for row in noncopy)
    result = {
        "schema_version": "ccpu.paper2.result_use_analysis.v1",
        "conditions": conditions,
        "decision": {
            "runtime_copy_default": True,
            "best_neural_copy_format": best_copy,
            "best_neural_copy_rate": copy_cells[best_copy]["exact_rate"],
            "attention_diagnostics": (
                "not_needed" if noncopy_reliable else "run_attention_and_causal_masking"
            ),
            "provenance_bias_gate": "pending_causal_evidence",
            "criterion": (
                "run diagnostics when INTERPRET or CONTINUE is below 0.8; "
                "bias requires causal support"
            ),
        },
        "sources": sources,
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "result_use_analysis.json", result)
    _plot_result_use(result, output_dir / "result_use_formats.png")
    return result


def _plot_result_use(result: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 2 result-use plots require matplotlib") from error
    neural = next(row for row in result["conditions"] if row["condition"] == "qwen_base")
    lookup = {(row["task"], row["format"]): row["exact_rate"] for row in neural["by_task_format"]}
    values = [[lookup[(task, name)] for name in FORMATS] for task in TASKS]
    figure, axis = plt.subplots(figsize=(9.0, 3.8))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(range(len(FORMATS)), [name.replace("_", " ") for name in FORMATS], rotation=20)
    axis.set_yticks(range(len(TASKS)), TASKS)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="strict exact rate")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
