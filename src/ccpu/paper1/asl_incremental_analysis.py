"""Paired, transition, and structural diagnostics for incremental ASL compilation."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


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


def _run_snapshot(path: str | Path) -> dict[str, Any]:
    report = read_json(path)
    rates = report["rates"]
    transitions = report["incremental"]["transition_rates"]
    return {
        "answer": rates["final_answer_correct"],
        "parse": rates["parse_valid"],
        "executable": rates["executable"],
        "semantic_state": rates["semantic_state_equivalent"],
        "dependency": rates["dependency_correct"],
        "transition_operator": transitions["operator_exact"],
        "transition_path": transitions["path_exact"],
        "transition_state_delta": transitions["state_delta_exact"],
        "completed_programs": report["incremental"]["completed_program_count"],
        "program_count": report["prediction_count"],
        "programs_sha256": report["run"]["programs_sha256"],
        "model_id": report["run"]["model"]["model_id"],
        "model_revision": report["run"]["model"]["revision"],
        "seed": report["run"]["seed"],
        "scored_predictions_sha256": report["scored_predictions_sha256"],
    }


def _training_snapshot(path: str | Path) -> dict[str, Any]:
    report = read_json(path)
    final = report["history"][-1]
    train_loss = final["mean_train_loss"]
    dev_loss = final["mean_dev_loss"]
    return {
        "adapter_id": report["adapter_id"],
        "rank": report["training"]["rank"],
        "target_modules": report["training"]["target_modules"],
        "trainable_parameters": report["trainable_parameters"],
        "train_rows": report["train_rows"],
        "dev_rows": report["dev_rows"],
        "epochs": report["training"]["epochs"],
        "final_train_loss": train_loss,
        "final_dev_loss": dev_loss,
        "generalization_gap": dev_loss - train_loss,
        "train_sha256": report["train_sha256"],
        "dev_sha256": report["dev_sha256"],
    }


def _pilot_condition(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in report["conditions"] if row["condition"] == name)


def analyze_adapter_capacity_interventions(
    *,
    baseline_predicted_summary: str | Path,
    baseline_oracle_summary: str | Path,
    baseline_full_summary: str | Path,
    candidate_predicted_summary: str | Path,
    candidate_oracle_summary: str | Path,
    candidate_full_summary: str | Path,
    baseline_training_report: str | Path,
    candidate_training_report: str | Path,
    pilot_checkpoint: str | Path,
    semantic_summary: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare matched adapter ranks and rank non-data interventions."""

    summary_paths = {
        "r8": {
            "predicted": Path(baseline_predicted_summary),
            "oracle": Path(baseline_oracle_summary),
            "full_question": Path(baseline_full_summary),
        },
        "r16": {
            "predicted": Path(candidate_predicted_summary),
            "oracle": Path(candidate_oracle_summary),
            "full_question": Path(candidate_full_summary),
        },
    }
    runs = {
        rank: {mode: _run_snapshot(path) for mode, path in modes.items()}
        for rank, modes in summary_paths.items()
    }
    program_hashes = {
        runs[rank][mode]["programs_sha256"] for rank in runs for mode in runs[rank]
    }
    model_revisions = {
        (runs[rank][mode]["model_id"], runs[rank][mode]["model_revision"])
        for rank in runs
        for mode in runs[rank]
    }
    seeds = {runs[rank][mode]["seed"] for rank in runs for mode in runs[rank]}
    if len(program_hashes) != 1 or len(model_revisions) != 1 or len(seeds) != 1:
        raise ValueError("capacity comparison requires matched programs, model revision, and seed")

    training = {
        "r8": _training_snapshot(baseline_training_report),
        "r16": _training_snapshot(candidate_training_report),
    }
    if (
        training["r8"]["train_sha256"] != training["r16"]["train_sha256"]
        or training["r8"]["dev_sha256"] != training["r16"]["dev_sha256"]
    ):
        raise ValueError("capacity comparison requires identical train and development rows")

    paired = {}
    for mode in ("predicted", "oracle", "full_question"):
        baseline_rows = read_jsonl(summary_paths["r8"][mode].with_name("scored_predictions.jsonl"))
        candidate_rows = read_jsonl(summary_paths["r16"][mode].with_name("scored_predictions.jsonl"))
        paired[mode] = _paired(
            _correct_by_parent(baseline_rows),
            _correct_by_parent(candidate_rows),
            reference_name="r8",
            candidate_name="r16",
        )

    metric_names = (
        "answer",
        "parse",
        "executable",
        "semantic_state",
        "dependency",
        "transition_operator",
        "transition_path",
        "transition_state_delta",
    )
    rank_deltas = {
        mode: {
            metric: runs["r16"][mode][metric] - runs["r8"][mode][metric]
            for metric in metric_names
        }
        for mode in runs["r8"]
    }
    context_gains = {
        rank: runs[rank]["full_question"]["answer"] - runs[rank]["predicted"]["answer"]
        for rank in runs
    }
    oracle_gaps = {
        rank: runs[rank]["oracle"]["answer"] - runs[rank]["predicted"]["answer"]
        for rank in runs
    }

    pilot = read_json(pilot_checkpoint)
    lora100 = _pilot_condition(pilot, "lora_100")
    lora100_icl3 = _pilot_condition(pilot, "lora_100_icl_3")
    hybrid_gain = (
        lora100_icl3["rates"]["final_answer_correct"]
        - lora100["rates"]["final_answer_correct"]
    )
    semantics = read_json(semantic_summary)
    whole_semantics = semantics["conditions"]["whole_lora500"]["semantic"]
    teacher = semantics["teacher_consistency"]

    interventions = [
        {
            "priority": 1,
            "intervention": "raw_context_or_self_generated_intent_state",
            "status": "empirically_supported",
            "evidence": {
                "r8_full_question_answer_gain": context_gains["r8"],
                "r16_full_question_answer_gain": context_gains["r16"],
                "r8_oracle_state_answer_gap": oracle_gaps["r8"],
                "r16_oracle_state_answer_gap": oracle_gaps["r16"],
            },
            "next_test": (
                "provide the raw observed question or retain only a ledger generated by the "
                "model from raw text; never inject a gold semantic frame"
            ),
        },
        {
            "priority": 2,
            "intervention": "factorized_grounding_over_self_generated_symbols",
            "status": "strongly_diagnostic",
            "evidence": {
                "whole_attribute_f1": whole_semantics["attribute"]["f1"],
                "whole_dependency_f1": whole_semantics["dependency"]["f1"],
                "r8_predicted_transition_path": runs["r8"]["predicted"]["transition_path"],
                "r8_predicted_transition_operator": runs["r8"]["predicted"][
                    "transition_operator"
                ],
                "teacher_attribute_shapes": teacher["canonical_attribute_shapes"],
            },
            "next_test": (
                "factorize choices over symbols already created by accepted model output; the "
                "model must still propose unseen targets from raw text"
            ),
        },
        {
            "priority": 3,
            "intervention": "early_stopping_and_regularization",
            "status": "supported_by_loss_divergence",
            "evidence": {
                "r8_generalization_gap": training["r8"]["generalization_gap"],
                "r16_generalization_gap": training["r16"]["generalization_gap"],
                "r16_minus_r8_dev_loss": (
                    training["r16"]["final_dev_loss"] - training["r8"]["final_dev_loss"]
                ),
            },
            "next_test": "evaluate every epoch and select by dev semantic execution, not final loss",
        },
        {
            "priority": 4,
            "intervention": "fixed_demonstration_hybrid",
            "status": "pilot_supported",
            "evidence": {"lora100_plus_icl3_answer_gain": hybrid_gain},
            "next_test": (
                "freeze one demonstration set from training before evaluation and use the same "
                "set for every test item"
            ),
        },
        {
            "priority": 5,
            "intervention": "qkvo_plus_mlp_adapter_placement",
            "status": "clean_unrun_ablation",
            "evidence": {
                "r16_minus_r8_predicted_answer": rank_deltas["predicted"]["answer"],
                "r16_minus_r8_full_question_answer": rank_deltas["full_question"]["answer"],
            },
            "next_test": "run the pinned QKVO+MLP rank-8 condition on identical transitions",
        },
        {
            "priority": 6,
            "intervention": "larger_base_model",
            "status": "plausible_but_unisolated",
            "evidence": {"current_base_model": runs["r8"]["predicted"]["model_id"]},
            "next_test": "repeat one frozen condition on a viable 1B-1.7B base model",
        },
        {
            "priority": 7,
            "intervention": "grammar_constrained_decoding",
            "status": "bounded_surface_benefit",
            "evidence": {
                "r8_predicted_parse_error_ceiling": 1 - runs["r8"]["predicted"]["parse"],
                "r16_parse_gain_without_answer_gain": rank_deltas["predicted"]["parse"],
            },
            "next_test": "use constraints for fail-closed robustness, not as the semantic remedy",
        },
    ]
    report = {
        "schema_version": "ccpu.paper1.asl_adapter_capacity_interventions.v1",
        "matched_protocol": {
            "programs_sha256": next(iter(program_hashes)),
            "model_id": next(iter(model_revisions))[0],
            "model_revision": next(iter(model_revisions))[1],
            "seed": next(iter(seeds)),
            "program_count": runs["r8"]["predicted"]["program_count"],
        },
        "runs": runs,
        "training": training,
        "rank_deltas_r16_minus_r8": rank_deltas,
        "paired_r8_vs_r16": paired,
        "context_answer_gains": context_gains,
        "oracle_state_answer_gaps": oracle_gaps,
        "interventions_other_than_more_data": interventions,
        "capacity_conclusion": (
            "doubling QKVO rank improves surface validity but does not improve final-answer "
            "accuracy or held-out semantic reconstruction"
        ),
        "claim_boundary": (
            "single base model, seed, and 25-program frozen diagnostic; intervention ranking "
            "combines measured evidence with prespecified engineering hypotheses; no proposed "
            "test input may depend on gold test semantics or per-item demonstration selection"
        ),
        "input_sha256": {
            str(path): file_sha256(path)
            for path in [
                *(path for modes in summary_paths.values() for path in modes.values()),
                baseline_training_report,
                candidate_training_report,
                pilot_checkpoint,
                semantic_summary,
            ]
        },
    }
    write_json(output_path, report)
    return report
