"""Aggregate the frozen Paper 1 ASL pilot without hand-transcribed results."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import read_json, read_jsonl, write_json

from .asl_pilot_eval import analyze_asl_predictions

_ORIGINAL_RUNS = {
    "base": "base_original",
    "icl_5": "icl5_original",
    "icl_10": "icl10_original",
    "lora_25": "lora25_original",
    "lora_50": "lora50_original",
    "lora_100": "lora100_original",
    "lora_100_icl_3": "lora100_icl3_original",
    "lora_augmented": "lora_aug_original",
}
_ROBUSTNESS_RUNS = {
    "lora_100": {
        "original": "lora100_original",
        "numeric": "lora100_numeric",
        "large": "lora100_large",
        "paraphrase": "lora100_paraphrase",
    },
    "lora_augmented": {
        "original": "lora_aug_original",
        "numeric": "lora_aug_numeric",
        "large": "lora_aug_large",
        "paraphrase": "lora_aug_paraphrase",
    },
}


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count == 0:
        return [0.0, 0.0]
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
    radius /= denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _walk_expression(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [node] + [
        descendant for argument in node.get("args", []) for descendant in _walk_expression(argument)
    ]


def _relation_classes(row: dict[str, Any]) -> set[str]:
    operations = row["ccir"]["operations"]
    assignments = [item["operation"] for item in operations if item["operation"]["op"] == "SET"]
    expressions = [
        node for operation in assignments for node in _walk_expression(operation["expr"])
    ]
    operators = Counter(str(node["op"]) for node in expressions)
    references = [str(node["path"]) for node in expressions if node["op"] == "REF"]
    targets = [str(operation["target"]) for operation in assignments]
    question = str(row["question"]).casefold()
    classes = set()
    if operators["DIV"] or re.search(r"\b(ratio|fraction|half|third|quarter|out of)\b", question):
        classes.add("ratio_or_fraction")
    if re.search(r"\b(more than|less than|fewer than|as many|times (?:as|more))\b", question):
        classes.add("relative_quantity")
    if operators["RATE_TIMES_DURATION"] or re.search(r"\b(per|each|every|rate|speed)\b", question):
        classes.add("rate_or_per_unit")
    if any(
        operators[name] for name in ("PERCENT_OF", "INCREASE_BY_PERCENT", "DECREASE_BY_PERCENT")
    ):
        classes.add("percentage")
    if operators["SUB"] or re.search(r"\b(remain|remaining|rest|left|difference)\b", question):
        classes.add("remaining_or_difference")
    if re.search(r"\b(year|month|week|day|hour|minute|before|after|later|ago|now)\b", question):
        classes.add("temporal_or_duration")
    if operators["DIV"] and re.search(r"\b(each|equally|shared|split|among|per)\b", question):
        classes.add("equal_allocation")
    if any(operators[name] for name in ("SUM", "MEAN", "MIN", "MAX")) or operators["ADD"] >= 2:
        classes.add("aggregation")
    if any(operators[name] for name in ("EQ", "NE", "LT", "LE", "GT", "GE")) or re.search(
        r"\b(compare|greater|smaller|largest|smallest|most|least|more|fewer)\b", question
    ):
        classes.add("comparison")
    roots = {path.split(".", 1)[0] for path in targets + references}
    if len(roots) >= 2:
        classes.add("multiple_entities")
    assignment_targets = {operation["target"] for operation in assignments}
    edges = sum(reference in assignment_targets for reference in references)
    if edges >= 2:
        classes.add("nested_dependencies")
    if len(assignments) >= 4 or edges >= 3:
        classes.add("chained_relations")
    return classes or {"single_relation"}


def _rescore(eval_path: Path, run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    previous = read_json(summary_path) if summary_path.exists() else {}
    summary = analyze_asl_predictions(eval_path, run_dir / "predictions.jsonl", run_dir)
    if "run" in previous:
        summary["run"] = previous["run"]
    write_json(summary_path, summary)
    return summary


def _condition_row(name: str, run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    count = int(summary["prediction_count"])
    correct = round(float(summary["rates"]["final_answer_correct"]) * count)
    return {
        "condition": name,
        "count": count,
        "correct": correct,
        "final_answer_wilson_95": _wilson(correct, count),
        "rates": summary["rates"],
        "component_mean_f1": summary["component_mean_f1"],
        "mean_prompt_tokens": sum(int(row["prompt_tokens"]) for row in predictions) / count,
        "mean_generated_tokens": sum(int(row["generated_tokens"]) for row in predictions) / count,
        "mean_wall_seconds": sum(int(row["wall_time_ns"]) for row in predictions) / count / 1e9,
    }


def _metric_by_parent(scored_path: Path) -> dict[str, bool]:
    return {
        str(row["parent_source_id"]): bool(row["metrics"]["final_answer_correct"])
        for row in read_jsonl(scored_path)
    }


def _transition(reference: dict[str, bool], changed: dict[str, bool]) -> dict[str, int]:
    parents = sorted(reference.keys() & changed.keys())
    return {
        "count": len(parents),
        "both_correct": sum(reference[parent] and changed[parent] for parent in parents),
        "lost": sum(reference[parent] and not changed[parent] for parent in parents),
        "gained": sum(not reference[parent] and changed[parent] for parent in parents),
        "both_incorrect": sum(not reference[parent] and not changed[parent] for parent in parents),
    }


def build_asl_checkpoint_report(
    *, freeze_dir: str | Path, data_dir: str | Path, runs_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    freeze = Path(freeze_dir)
    data = Path(data_dir)
    runs = Path(runs_dir)
    eval_paths = {
        name: data / "eval" / f"{name}.jsonl"
        for name in ("original", "numeric", "large", "paraphrase")
    }
    summaries = {}
    for name, run_name in _ORIGINAL_RUNS.items():
        summaries[name] = _rescore(eval_paths["original"], runs / run_name)
    for model_runs in _ROBUSTNESS_RUNS.values():
        for suite, run_name in model_runs.items():
            if suite == "original":
                continue
            _rescore(eval_paths[suite], runs / run_name)

    conditions = [
        _condition_row(name, runs / _ORIGINAL_RUNS[name], summaries[name])
        for name in _ORIGINAL_RUNS
    ]
    train_rows = read_jsonl(freeze / "splits" / "train.jsonl")
    class_counts = Counter(
        relation_class for row in train_rows for relation_class in _relation_classes(row)
    )
    pattern_counts = Counter(str(row["semantic_pattern_id"]) for row in train_rows)

    robustness = {}
    for model, model_runs in _ROBUSTNESS_RUNS.items():
        original = _metric_by_parent(runs / model_runs["original"] / "scored_predictions.jsonl")
        robustness[model] = {}
        for suite in ("numeric", "large", "paraphrase"):
            changed = _metric_by_parent(runs / model_runs[suite] / "scored_predictions.jsonl")
            robustness[model][suite] = _transition(original, changed)
    cross_model = {}
    for suite in ("original", "numeric", "large", "paraphrase"):
        original_model = _metric_by_parent(
            runs / _ROBUSTNESS_RUNS["lora_100"][suite] / "scored_predictions.jsonl"
        )
        augmented_model = _metric_by_parent(
            runs / _ROBUSTNESS_RUNS["lora_augmented"][suite] / "scored_predictions.jsonl"
        )
        cross_model[suite] = _transition(original_model, augmented_model)

    training = {}
    for name, directory in {
        "lora_25": "qwen_original_25",
        "lora_50": "qwen_original_50",
        "lora_100": "qwen_original_100",
        "lora_augmented": "qwen_augmented_1000",
    }.items():
        report = read_json(runs.parent / "adapters" / directory / "training_report.json")
        training[name] = {
            key: report[key]
            for key in (
                "adapter_id",
                "train_rows",
                "training_target_tokens",
                "optimizer_steps",
                "trainable_parameters",
                "wall_time_seconds",
                "peak_memory_bytes",
            )
        }
        training[name]["final_dev_loss"] = report["history"][-1]["mean_dev_loss"]

    report = {
        "schema_version": "ccpu.paper1.asl_checkpoint.v1",
        "claim_boundary": "Q1 diagnostic learning signal; not a robust general semantic compiler",
        "training_structure_coverage": {
            "rows": len(train_rows),
            "unique_normalized_semantic_patterns": len(pattern_counts),
            "pattern_multiplicity": dict(sorted(Counter(pattern_counts.values()).items())),
            "relation_class_counts": dict(sorted(class_counts.items())),
        },
        "conditions": conditions,
        "training": training,
        "paired_robustness": robustness,
        "paired_lora_a_vs_b": cross_model,
    }
    write_json(output, report)
    return report
