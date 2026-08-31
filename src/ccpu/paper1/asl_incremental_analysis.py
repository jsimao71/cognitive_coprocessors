"""Paired, transition, and structural diagnostics for incremental ASL compilation."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return [center - half / denominator, center + half / denominator]


def _paired_binomial_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def _correct_by_parent(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        str(row["parent_source_id"]): bool(row["metrics"]["final_answer_correct"]) for row in rows
    }


def _paired(
    reference: dict[str, bool],
    candidate: dict[str, bool],
    *,
    reference_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    keys = sorted(reference.keys() & candidate.keys())
    gain_label = f"{reference_name}_wrong_{candidate_name}_correct"
    loss_label = f"{reference_name}_correct_{candidate_name}_wrong"
    groups = {
        gain_label: [key for key in keys if not reference[key] and candidate[key]],
        loss_label: [key for key in keys if reference[key] and not candidate[key]],
        "both_correct": [key for key in keys if reference[key] and candidate[key]],
        "both_wrong": [key for key in keys if not reference[key] and not candidate[key]],
    }
    gains = len(groups[gain_label])
    losses = len(groups[loss_label])
    return {
        "counts": {name: len(ids) for name, ids in groups.items()},
        "source_ids": groups,
        "exploratory_exact_binomial_p": _paired_binomial_p(gains, losses),
    }


def _transition_taxonomy(
    predicted: list[dict[str, Any]], oracle: list[dict[str, Any]]
) -> dict[str, Any]:
    oracle_traces = {
        (str(row["parent_source_id"]), int(trace["part_id"])): trace
        for row in oracle
        for trace in row["part_traces"]
    }
    counts: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predicted:
        for trace in row["part_traces"]:
            metrics = trace["metrics"]
            categories = []
            if not trace["predicted_delta"]:
                categories.append("surface_no_asl")
            elif not metrics["parse_valid"]:
                categories.append("surface_malformed_syntax")
            elif not metrics["lowerable_to_ccir"]:
                categories.append("lowering_unsupported_form")
            elif not metrics["type_valid"]:
                categories.append("type_invalid")
            else:
                if metrics["paths_metrics"]["f1"] < 1:
                    categories.append("semantic_wrong_path_entity")
                if metrics["operators_metrics"]["f1"] < 1:
                    categories.append("semantic_wrong_operator")
                if metrics["source_facts_metrics"]["f1"] < 1:
                    categories.append("semantic_wrong_source_fact")
                if metrics["edges_metrics"]["f1"] < 1:
                    categories.append("reference_wrong_dependency")
                if metrics["semantic_state_metrics"]["f1"] < 1:
                    categories.append("semantic_wrong_state_delta")
                if (
                    "RETURN" in trace["predicted_delta"].upper()
                    and not metrics["semantic_return_equivalent"]
                ):
                    categories.append("query_wrong_return")
            key = (str(row["parent_source_id"]), int(trace["part_id"]))
            oracle_trace = oracle_traces.get(key)
            if (
                oracle_trace
                and oracle_trace["metrics"]["semantic_state_metrics"]["f1"] == 1
                and metrics["semantic_state_metrics"]["f1"] < 1
            ):
                categories.append("propagation_oracle_recovers_state_delta")
            for category in set(categories):
                counts[category] += 1
                by_dataset[str(row["dataset"])][category] += 1
                if len(examples[category]) < 3:
                    examples[category].append(
                        {
                            "dataset": row["dataset"],
                            "source_id": row["parent_source_id"],
                            "part_id": trace["part_id"],
                        }
                    )
    return {
        "counts": dict(sorted(counts.items())),
        "by_dataset": {
            dataset: dict(sorted(values.items())) for dataset, values in sorted(by_dataset.items())
        },
        "representative_ids": dict(sorted(examples.items())),
        "categories_are_multi_label": True,
    }


def _refs(node: dict[str, Any]) -> list[str]:
    if node["op"] == "REF":
        return [str(node["path"])]
    return [path for argument in node.get("args", []) for path in _refs(argument)]


def _dependency_depth(row: dict[str, Any]) -> int:
    dependencies = {
        str(item["operation"]["target"]): _refs(item["operation"]["expr"])
        for item in row["ccir"]["operations"]
        if item["operation"]["op"] == "SET"
    }

    def depth(path: str, stack: frozenset[str]) -> int:
        if path in stack:
            return 0
        internal = [ref for ref in dependencies.get(path, []) if ref in dependencies]
        return 1 + max((depth(ref, stack | {path}) for ref in internal), default=0)

    return max((depth(path, frozenset()) for path in dependencies), default=0)


def _structural_stats(programs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for dataset in sorted({str(row["dataset"]) for row in programs}):
        rows = [row for row in programs if row["dataset"] == dataset]
        per_row = []
        for row in rows:
            operations = row["ccir"]["operations"]
            paths = {
                str(item["operation"]["target"])
                for item in operations
                if item["operation"]["op"] == "SET"
            }
            paths.update(
                ref
                for item in operations
                for ref in _refs(item["operation"].get("expr", {"op": "CONST"}))
            )
            per_row.append(
                {
                    "parts": len(row["part_mappings"]),
                    "statements": sum(len(mapping["asl"]) for mapping in row["part_mappings"]),
                    "dependency_depth": _dependency_depth(row),
                    "entities": len({path.split(".", 1)[0] for path in paths}),
                    "paths": len(paths),
                    "asl_lexical_tokens": len(
                        re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|[-+*/=()]|\d+(?:\.\d+)?", row["asl"])
                    ),
                }
            )
        result[dataset] = {
            "programs": len(rows),
            **{f"mean_{key}": mean(item[key] for item in per_row) for key in per_row[0]},
        }
    return result


def analyze_incremental_capacity(
    *,
    programs_path: str | Path,
    whole_scored_path: str | Path,
    predicted_scored_path: str | Path,
    oracle_scored_path: str | Path,
    output_path: str | Path,
    full_context_scored_path: str | Path | None = None,
) -> dict[str, Any]:
    programs = read_jsonl(programs_path)
    whole = read_jsonl(whole_scored_path)
    predicted = read_jsonl(predicted_scored_path)
    oracle = read_jsonl(oracle_scored_path)
    conditions = {"whole": whole, "predicted_state": predicted, "oracle_state": oracle}
    full_context = read_jsonl(full_context_scored_path) if full_context_scored_path else []
    if full_context:
        conditions["predicted_state_full_question"] = full_context
    condition_results = {}
    for name, rows in conditions.items():
        successes = sum(bool(row["metrics"]["final_answer_correct"]) for row in rows)
        condition_results[name] = {
            "successes": successes,
            "total": len(rows),
            "accuracy": successes / len(rows),
            "wilson_95": _wilson(successes, len(rows)),
        }
    whole_map = _correct_by_parent(whole)
    predicted_map = _correct_by_parent(predicted)
    oracle_map = _correct_by_parent(oracle)
    report = {
        "schema_version": "ccpu.paper1.asl_incremental_capacity_analysis.v1",
        "conditions": condition_results,
        "state_error_propagation_gap": (
            condition_results["oracle_state"]["accuracy"]
            - condition_results["predicted_state"]["accuracy"]
        ),
        "paired_whole_vs_predicted": _paired(
            whole_map,
            predicted_map,
            reference_name="whole",
            candidate_name="predicted",
        ),
        "paired_predicted_vs_oracle": _paired(
            predicted_map,
            oracle_map,
            reference_name="predicted",
            candidate_name="oracle",
        ),
        "transition_error_taxonomy": _transition_taxonomy(predicted, oracle),
        "test_structure_by_dataset": _structural_stats(programs),
        "input_sha256": {
            "programs": file_sha256(programs_path),
            "whole_scored": file_sha256(whole_scored_path),
            "predicted_scored": file_sha256(predicted_scored_path),
            "oracle_scored": file_sha256(oracle_scored_path),
        },
        "claim_boundary": "single model, seed, and 25-program diagnostic; paired p-values exploratory",
    }
    if full_context:
        full_context_map = _correct_by_parent(full_context)
        report["full_question_context_gain"] = (
            condition_results["predicted_state_full_question"]["accuracy"]
            - condition_results["predicted_state"]["accuracy"]
        )
        report["paired_predicted_vs_full_question"] = _paired(
            predicted_map,
            full_context_map,
            reference_name="predicted",
            candidate_name="full_question",
        )
        report["paired_whole_vs_full_question"] = _paired(
            whole_map,
            full_context_map,
            reference_name="whole",
            candidate_name="full_question",
        )
        report["input_sha256"]["full_context_scored"] = file_sha256(full_context_scored_path)
    write_json(output_path, report)
    return report
