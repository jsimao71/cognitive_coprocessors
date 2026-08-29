"""Factorized six-way neural routing for the Paper 2 diagnostic benchmark."""

from __future__ import annotations

import re
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
from ccpu.common.metrics import binary_classification, safe_mean
from ccpu.common.schema import GenerationResult

from .diagnostic import _score_parser_execution

ROUTER_LABELS = ("NONE", "CALCULATOR", "DATE", "UNITS", "GRAPH", "DATALOG")
_LABEL_PATTERN = re.compile(rf"\s*({'|'.join(ROUTER_LABELS)})\s*[.!]?\s*", re.IGNORECASE)
_ROUTER_INSTRUCTION = (
    "Route the request to exactly one execution engine. "
    "Output exactly one label and no other text: "
    "NONE, CALCULATOR, DATE, UNITS, GRAPH, or DATALOG. "
    "Use NONE when the text discusses, quotes, or copies engine-like content "
    "without requesting execution.\n\nRequest:\n{prompt}\n\nLabel:"
)


def router_prompt(prompt: str) -> str:
    return _ROUTER_INSTRUCTION.format(prompt=prompt)


def prepare_router_data(source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        source_path = source_dir / f"{split}.jsonl"
        rows = []
        for row in read_jsonl(source_path):
            label = str(row["classification_label"])
            if label not in ROUTER_LABELS:
                raise ValueError(f"unsupported router label: {label}")
            rows.append(
                {
                    **row,
                    "schema_version": "ccpu.paper2.router_example.v1",
                    "source_prompt": row["prompt"],
                    "prompt": router_prompt(str(row["prompt"])),
                    "runtime_target": row["target"],
                    "target": label,
                }
            )
        paths[split] = write_jsonl(output_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": "ccpu.paper2.router_dataset_manifest.v1",
        "source_dir": str(source_dir),
        "source_sha256": {
            split: file_sha256(source_dir / f"{split}.jsonl") for split in ("train", "dev", "test")
        },
        "paths": {split: str(path) for split, path in paths.items()},
        "dataset_sha256": {split: file_sha256(path) for split, path in paths.items()},
        "counts": counts,
        "prompt_fingerprint": fingerprint(_ROUTER_INSTRUCTION),
        "labels": list(ROUTER_LABELS),
        "environment": environment_manifest(root),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_router_label(text: str) -> str | None:
    match = _LABEL_PATTERN.fullmatch(text)
    return match[1].upper() if match else None


def _router_metrics(
    condition: str,
    rows: list[dict[str, Any]],
    labels: list[str],
    elapsed_ns: int,
) -> dict[str, Any]:
    gold = [str(row["classification_label"]) for row in rows]
    trigger = binary_classification(
        [label != "NONE" for label in gold],
        [label != "NONE" for label in labels],
    )
    f1_scores = []
    for target in ROUTER_LABELS:
        true_positive = sum(
            expected == target and actual == target
            for expected, actual in zip(gold, labels, strict=True)
        )
        false_positive = sum(
            expected != target and actual == target
            for expected, actual in zip(gold, labels, strict=True)
        )
        false_negative = sum(
            expected == target and actual != target
            for expected, actual in zip(gold, labels, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    positives = [index for index, label in enumerate(gold) if label != "NONE"]
    return {
        "condition": condition,
        "accuracy": safe_mean(
            actual == expected for actual, expected in zip(labels, gold, strict=True)
        ),
        "macro_f1": safe_mean(f1_scores),
        "trigger_recall": trigger["recall"],
        "false_activation_rate": trigger["false_intervention_rate"],
        "engine_selection_accuracy": safe_mean(labels[index] == gold[index] for index in positives),
        "mean_route_latency_us": elapsed_ns / max(len(rows), 1) / 1000,
    }


def run_router_condition(
    rows: list[dict[str, Any]],
    backend: Any,
    *,
    condition: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    labels = []
    for index, row in enumerate(rows):
        generated: GenerationResult = backend.generate(str(row["prompt"]), seed=seed + index)
        parsed = parse_router_label(generated.generated_text)
        predicted = parsed or "NONE"
        labels.append(predicted)
        predictions.append(
            {
                "schema_version": "ccpu.paper2.router_prediction.v1",
                "condition": condition,
                "example_id": row["example_id"],
                "gold_engine": row["classification_label"],
                "predicted_engine": predicted,
                "parse_valid": parsed is not None,
                "raw_exact": generated.generated_text.strip() == str(row["classification_label"]),
                "generated_text": generated.generated_text,
                "prompt_tokens": generated.prompt_tokens,
                "generated_tokens": generated.generated_tokens,
                "wall_time_ns": generated.wall_time_ns,
                "generation_metadata": generated.metadata,
            }
        )
    elapsed = sum(row["wall_time_ns"] for row in predictions)
    metrics = _router_metrics(condition, rows, labels, elapsed)
    execution_rows = [
        {**row, "prompt": row["source_prompt"], "target": row["runtime_target"]} for row in rows
    ]
    execution = _score_parser_execution(execution_rows, labels, condition=condition)
    metrics.update(execution["summary"])
    metrics.update(
        {
            "raw_exact_rate": sum(row["raw_exact"] for row in predictions)
            / max(len(predictions), 1),
            "parse_valid_rate": sum(row["parse_valid"] for row in predictions)
            / max(len(predictions), 1),
            "mean_prompt_tokens": sum(row["prompt_tokens"] for row in predictions)
            / max(len(predictions), 1),
            "mean_generated_tokens": sum(row["generated_tokens"] for row in predictions)
            / max(len(predictions), 1),
            "total_wall_time_seconds": elapsed / 1_000_000_000,
        }
    )
    return predictions, metrics


def run_and_write_router(
    *,
    dataset_path: str | Path,
    backend: Any,
    condition: str,
    seed: int,
    output_dir: str | Path,
    model: dict[str, Any],
    adapter_path: str | None,
) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    predictions, summary = run_router_condition(rows, backend, condition=condition, seed=seed)
    output_dir = Path(output_dir)
    predictions_path = write_jsonl(output_dir / "predictions.jsonl", predictions)
    summary_path = write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": "ccpu.paper2.router_run_manifest.v1",
        "condition": condition,
        "model": model,
        "adapter_path": adapter_path,
        "dataset_sha256": file_sha256(dataset_path),
        "predictions_sha256": file_sha256(predictions_path),
        "summary_sha256": file_sha256(summary_path),
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(output_dir / "manifest.json", manifest)
    return summary


def analyze_router_runs(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = read_json(config_path)
    cpu = read_json(config["cpu_trigger_ladder"])
    cpu_by_name = {row["condition"]: row for row in cpu["trigger_ladder"]}
    rows = []
    for condition in config.get("cpu_conditions", []):
        source = cpu_by_name[condition]
        rows.append(
            {
                **source,
                "source": "cpu",
                "training_minutes": 0.0,
                "trainable_parameters": 0,
                "peak_memory_bytes": None,
            }
        )
    source_hashes = [
        {"path": config["cpu_trigger_ladder"], "sha256": file_sha256(config["cpu_trigger_ladder"])}
    ]
    test_hashes = set()
    for run in config["neural_runs"]:
        summary = read_json(run["summary"])
        manifest = read_json(run["manifest"])
        training = read_json(run["training_report"]) if run.get("training_report") else None
        test_hashes.add(manifest["dataset_sha256"])
        rows.append(
            {
                **summary,
                "source": "xpu",
                "training_minutes": training["wall_time_seconds"] / 60 if training else 0.0,
                "trainable_parameters": training["trainable_parameters"] if training else 0,
                "peak_memory_bytes": training["peak_memory_bytes"] if training else None,
            }
        )
        for key in ("summary", "manifest", "training_report"):
            if run.get(key):
                source_hashes.append({"path": run[key], "sha256": file_sha256(run[key])})
    if len(test_hashes) != 1:
        raise ValueError("neural router runs must share one test dataset hash")
    neural = [row for row in rows if row["source"] == "xpu" and row["condition"] != "T5_qwen_base"]
    passing = [
        row["condition"]
        for row in neural
        if row["engine_selection_accuracy"] >= 0.9
        and row["runtime_exact_rate"] >= 0.9
        and row["false_activation_rate"] <= 0.1
    ]
    result = {
        "schema_version": "ccpu.paper2.router_analysis.v1",
        "test_dataset_sha256": next(iter(test_hashes)),
        "rows": rows,
        "decision": {
            "criterion": ">=0.9 engine selection/runtime exact and <=0.1 FAR",
            "passing_neural_conditions": passing,
            "status": "prefer_neural_router" if passing else "retain_cpu_baseline",
            "preferred_deployment": "T1_lexical_regex",
            "next_experiment": "result_use_factorization",
        },
        "sources": source_hashes,
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "router_comparison.json", result)
    _plot_router_comparison(result, output_dir / "router_comparison.png")
    return result


def _plot_router_comparison(result: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 2 router plots require matplotlib") from error
    rows = result["rows"]
    labels = [
        row["condition"]
        .replace("T5_qwen_router_", "Qwen ")
        .replace("T5_qwen_base", "Qwen base")
        .replace("T1_lexical_regex", "T1 regex")
        .replace("T2_semantic_rules", "T2 rules")
        for row in rows
    ]
    x = list(range(len(rows)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar(
        [value - width for value in x],
        [row["engine_selection_accuracy"] for row in rows],
        width,
        label="engine selection",
        color="#176b87",
    )
    axis.bar(
        x,
        [row["runtime_exact_rate"] for row in rows],
        width,
        label="runtime exact",
        color="#d99b2b",
    )
    axis.bar(
        [value + width for value in x],
        [row["false_activation_rate"] for row in rows],
        width,
        label="false activation",
        color="#b33f40",
    )
    axis.axhline(0.9, color="#333333", linestyle="--", linewidth=1, alpha=0.6)
    axis.set(xticks=x, xticklabels=labels, ylabel="rate", ylim=(0, 1.05))
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
