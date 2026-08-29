"""Pinned CRAG selection and matched context-sufficiency trigger diagnostics."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    environment_manifest,
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.generic_gateway import GenericCognitiveGateway, GenericToolCall
from ccpu.common.public_benchmarks import read_verified_bz2_jsonl, stratified_select

from .natural_robustness import semantic_features
from .triggers import semantic_risk


def _source_path(config: dict[str, Any], cache_root: str | Path) -> Path:
    return Path(cache_root) / "quivr_crag" / str(config["source"]["file"])


def _load_source(config: dict[str, Any], cache_root: str | Path) -> list[dict[str, Any]]:
    source = config["source"]
    return read_verified_bz2_jsonl(
        _source_path(config, cache_root),
        expected_sha256=str(source["file_sha256"]),
        expected_rows=int(source["expected_rows"]),
    )


def _content_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def freeze_crag_subset(
    config_path: str | Path, cache_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != "ccpu.paper1_5.public_crag_config.v1":
        raise ValueError("unsupported Paper 1.5 CRAG config schema")
    seed = int(config["selection_seed"])
    records = []
    for index, row in enumerate(_load_source(config, cache_root)):
        example_id = str(row["interaction_id"])
        stratum = "|".join(
            str(row[key])
            for key in ("domain", "question_type", "static_or_dynamic", "answer_type")
        )
        selection_key = hashlib.sha256(
            f"{seed}:{example_id}:{_content_sha(row)}".encode("ascii")
        ).hexdigest()
        records.append(
            {
                "benchmark": "crag",
                "example_id": example_id,
                "source_row": index,
                "content_sha256": _content_sha(row),
                "selection_key": selection_key,
                "difficulty": 1,
                "difficulty_stratum": stratum,
                "domain": str(row["domain"]),
                "question_type": str(row["question_type"]),
                "static_or_dynamic": str(row["static_or_dynamic"]),
                "answer_type": str(row["answer_type"]),
                "split": int(row["split"]),
            }
        )
    selected = stratified_select(records, int(config["max_rows"]), seed)
    output = Path(output_dir)
    selection_path = write_jsonl(output / "selection.jsonl", selected)
    counts: dict[str, dict[str, int]] = {}
    for key in ("domain", "question_type", "static_or_dynamic", "answer_type", "split"):
        values: dict[str, int] = defaultdict(int)
        for row in selected:
            values[str(row[key])] += 1
        counts[key] = dict(sorted(values.items()))
    manifest = {
        "schema_version": "ccpu.paper1_5.public_crag_manifest.v1",
        "config_fingerprint": fingerprint(config),
        "record_count": len(selected),
        "selection_sha256": file_sha256(selection_path),
        "counts": counts,
        "source": config["source"],
        "matched_evaluation_rows": 2 * len(selected),
        "control_construction": "same question with gold answer supplied in active context",
        "redistribution": "IDs, strata, split, and content hashes only",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _materialize(
    config_path: str | Path, cache_root: str | Path, selection_path: str | Path
) -> list[dict[str, Any]]:
    config = read_json(config_path)
    source = _load_source(config, cache_root)
    selected = read_jsonl(selection_path)
    materialized = []
    for item in selected:
        row = source[int(item["source_row"])]
        if str(row["interaction_id"]) != item["example_id"] or _content_sha(row) != item[
            "content_sha256"
        ]:
            raise ValueError(f"CRAG source changed for {item['example_id']}")
        materialized.append({**item, "query": str(row["query"]), "answer": str(row["answer"])})
    return materialized


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [row for row in rows if row["evidence_required"]]
    controls = [row for row in rows if not row["evidence_required"]]
    recall = sum(row["triggered"] for row in required) / len(required) if required else None
    false_rate = sum(row["triggered"] for row in controls) / len(controls) if controls else None
    accuracy = sum(row["triggered"] == row["evidence_required"] for row in rows) / len(rows)
    return {
        "count": len(rows),
        "retrieval_needed_recall": recall,
        "false_retrieval_rate": false_rate,
        "accuracy": accuracy,
        "ucr": 1.0 - recall if recall is not None else None,
        "retrieval_rate": sum(row["triggered"] for row in rows) / len(rows),
    }


def analyze_crag_triggers(
    config_path: str | Path,
    cache_root: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_rows = _materialize(config_path, cache_root, selection_path)
    conditions = (
        "llm_only_no_retrieval",
        "upfront_retrieval",
        "generic_retrieve_oracle_transport",
        "legacy_semantic",
        "natural_semantic",
        "dynamic_metadata_proxy",
        "registered_oracle",
    )
    transported_calls = 0

    def resolve(intent: str, payload: Any, active_task: str) -> dict[str, Any]:
        nonlocal transported_calls
        if intent != "retrieve" or active_task != "crag":
            raise ValueError("unexpected CRAG generic-tool route")
        transported_calls += 1
        return {"accepted": True, "example_id": payload["example_id"]}

    gateway = GenericCognitiveGateway(resolve)
    predictions = []
    latencies: dict[str, int] = defaultdict(int)
    for source in source_rows:
        texts = {
            "external": str(source["query"]),
            "context": (
                f"Supplied context: the answer for this question is {source['answer']}. "
                f"Question: {source['query']}"
            ),
        }
        for availability, text in texts.items():
            required = availability == "external"
            features = semantic_features(text)
            for condition in conditions:
                started = time.perf_counter_ns()
                if condition == "llm_only_no_retrieval":
                    triggered = False
                elif condition == "upfront_retrieval":
                    triggered = True
                elif condition == "generic_retrieve_oracle_transport":
                    triggered = required
                    if triggered:
                        gateway.invoke(
                            GenericToolCall(
                                "__retrieve", {"example_id": source["example_id"]}
                            ),
                            active_task="crag",
                        )
                elif condition == "legacy_semantic":
                    triggered = semantic_risk(text)[0]
                elif condition == "natural_semantic":
                    triggered = bool(features["combined"])
                elif condition == "dynamic_metadata_proxy":
                    triggered = (
                        source["static_or_dynamic"] != "static" and not features["active_context"]
                    )
                else:
                    triggered = required
                latencies[condition] += time.perf_counter_ns() - started
                predictions.append(
                    {
                        "schema_version": "ccpu.paper1_5.public_crag_prediction.v2",
                        "condition": condition,
                        "example_id": source["example_id"],
                        "availability": availability,
                        "evidence_required": required,
                        "triggered": triggered,
                        "domain": source["domain"],
                        "question_type": source["question_type"],
                        "static_or_dynamic": source["static_or_dynamic"],
                        "answer_type": source["answer_type"],
                    }
                )

    results = []
    for condition in conditions:
        members = [row for row in predictions if row["condition"] == condition]
        paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            paired[str(row["example_id"])].append(row)
        suppression = sum(
            any(row["availability"] == "external" and row["triggered"] for row in pair)
            and any(row["availability"] == "context" and not row["triggered"] for row in pair)
            for pair in paired.values()
        ) / len(paired)
        by_dynamic = {}
        for value in sorted({str(row["static_or_dynamic"]) for row in members}):
            by_dynamic[value] = _rates(
                [row for row in members if row["static_or_dynamic"] == value]
            )
        results.append(
            {
                "condition": condition,
                **_rates(members),
                "matched_context_suppression": suppression,
                "mean_cpu_latency_us": latencies[condition] / len(members) / 1000,
                "by_static_or_dynamic": by_dynamic,
                "deployment_status": (
                    "oracle_timed_transport_only"
                    if condition == "generic_retrieve_oracle_transport"
                    else "oracle_only"
                    if condition == "registered_oracle"
                    else "diagnostic"
                ),
            }
        )
    output = Path(output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", predictions)
    summary = {
        "schema_version": "ccpu.paper1_5.public_crag_analysis.v2",
        "base_question_count": len(source_rows),
        "matched_row_count": len(source_rows) * 2,
        "selection_sha256": file_sha256(selection_path),
        "predictions_sha256": file_sha256(predictions_path),
        "results": results,
        "generic_retrieve_transport": {
            "tool_name": "__retrieve",
            "accepted_calls": transported_calls,
            "expected_calls": len(source_rows),
            "backend_agreement_with_registered_oracle": all(
                tool["triggered"] == oracle["triggered"]
                for tool, oracle in zip(
                    (
                        row
                        for row in predictions
                        if row["condition"] == "generic_retrieve_oracle_transport"
                    ),
                    (
                        row
                        for row in predictions
                        if row["condition"] == "registered_oracle"
                    ),
                )
            ),
            "timing_policy": "registered evidence-required label; not model initiated",
        },
        "confidence_condition": "not run; requires matched model token traces",
        "generation_metrics": {
            "answer_accuracy": None,
            "abstention_quality": None,
            "authorized_commitment_coverage": None,
            "evidence_override": None,
            "automatic_rescue_rate": None,
            "reason": "no model generation or frozen evidence backend in this trigger audit",
        },
        "pending_conditions": [
            "confidence_flare",
            "voluntary_generic_retrieve",
            "generic_retrieve_intent_block",
            "frozen_evidence_rag",
        ],
        "interpretation": {
            "status": "semantic_policy_does_not_transfer",
            "paper3_5_gate": "closed_pending_model_and_evidence_runs",
        },
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(output / "summary.json", summary)
    _plot_crag(summary, output / "crag_trigger_comparison.png")
    return summary


def _plot_crag(summary: dict[str, Any], output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("CRAG plots require the analysis extra") from error
    rows = summary["results"]
    labels = [str(row["condition"]).replace("_", "\n") for row in rows]
    x = list(range(len(rows)))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.6, 3.8))
    axis.bar(
        [value - width / 2 for value in x],
        [row["retrieval_needed_recall"] for row in rows],
        width,
        label="Needed recall",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [row["false_retrieval_rate"] for row in rows],
        width,
        label="False retrieval",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title("CRAG matched evidence-availability trigger audit")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
