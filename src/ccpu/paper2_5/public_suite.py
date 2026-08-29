"""Generic-tool transport and readiness audits for public Paper 2.5 benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.generic_gateway import GenericCognitiveGateway, GenericToolCall


def audit_tatqa_generic_transport(
    predictions_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Replay registered TAT-QA assistance types through the stable gateway."""
    predictions = read_jsonl(predictions_path)
    accepted = 0

    def resolve(intent: str, payload: Any, active_task: str) -> dict[str, Any]:
        nonlocal accepted
        if active_task != "tatqa" or intent not in {"retrieve", "compute"}:
            raise ValueError("unexpected TAT-QA generic-tool route")
        accepted += 1
        return {"accepted": True, "example_id": payload["example_id"]}

    gateway = GenericCognitiveGateway(resolve)
    records = []
    for row in predictions:
        intents = ["retrieve"]
        if bool(row["compute_required"]):
            intents.append("compute")
        for intent in intents:
            result = gateway.invoke(
                GenericToolCall(f"__{intent}", {"example_id": row["example_id"]}),
                active_task="tatqa",
            )
            if not result["accepted"]:
                raise RuntimeError("generic gateway rejected a registered TAT-QA call")
        records.append(
            {
                "schema_version": "ccpu.paper2_5.public_tatqa_tool_record.v1",
                "example_id": row["example_id"],
                "registered_intents": [intent.upper() for intent in intents],
                "assistance_episodes": len(intents),
                "transport_accepted": True,
            }
        )

    output = Path(output_dir)
    records_path = write_jsonl(output / "records.jsonl", records)
    multi_episode = sum(row["assistance_episodes"] > 1 for row in records)
    total_episodes = sum(row["assistance_episodes"] for row in records)
    summary = {
        "schema_version": "ccpu.paper2_5.public_tatqa_tool_summary.v1",
        "record_count": len(records),
        "records_sha256": file_sha256(records_path),
        "composition_predictions_sha256": file_sha256(predictions_path),
        "accepted_calls": accepted,
        "total_assistance_episodes": total_episodes,
        "mean_assistance_episodes": total_episodes / len(records) if records else None,
        "single_episode_questions": len(records) - multi_episode,
        "multiple_episode_questions": multi_episode,
        "tool_names": ["__retrieve", "__compute"],
        "timing_policy": "registered gold assistance type; not model initiated",
        "claim_boundary": {
            "transport_only": True,
            "source_execution": False,
            "operation_selection": False,
            "final_answer_evaluated": False,
            "automatic_rescue_rate": None,
        },
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(output / "summary.json", summary)
    return summary


def freeze_public_suite_readiness(
    *,
    tatqa_manifest_path: str | Path,
    tatqa_composition_path: str | Path,
    tatqa_retrieval_path: str | Path,
    tatqa_tools_path: str | Path,
    crag_manifest_path: str | Path,
    crag_analysis_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Bind completed diagnostics and fail closed on missing public conditions."""
    inputs = {
        "tatqa_manifest": Path(tatqa_manifest_path),
        "tatqa_composition": Path(tatqa_composition_path),
        "tatqa_retrieval": Path(tatqa_retrieval_path),
        "tatqa_tools": Path(tatqa_tools_path),
        "crag_manifest": Path(crag_manifest_path),
        "crag_analysis": Path(crag_analysis_path),
    }
    loaded = {name: read_json(path) for name, path in inputs.items()}
    tatqa_selection = loaded["tatqa_manifest"]["selection_sha256"]
    if any(
        loaded[name]["selection_sha256"] != tatqa_selection
        for name in ("tatqa_composition", "tatqa_retrieval")
    ):
        raise ValueError("TAT-QA diagnostics do not share one frozen selection")
    if (
        loaded["tatqa_tools"]["composition_predictions_sha256"]
        != loaded["tatqa_composition"]["predictions_sha256"]
    ):
        raise ValueError("TAT-QA tool audit does not match composition predictions")
    if (
        loaded["crag_manifest"]["selection_sha256"]
        != loaded["crag_analysis"]["selection_sha256"]
    ):
        raise ValueError("CRAG diagnostics do not share one frozen selection")

    tatqa_retrieval = loaded["tatqa_retrieval"]["by_condition"]["structured_hybrid"]
    crag_transport = loaded["crag_analysis"]["generic_retrieve_transport"]
    benchmarks = {
        "tatqa": {
            "freeze_status": "complete",
            "record_count": loaded["tatqa_manifest"]["record_count"],
            "universal_text_retrieval_status": "retrieval_only_complete",
            "structured_hybrid_recall_at_5": tatqa_retrieval["mean_evidence_recall_at_k"],
            "generic_tool_status": "oracle_timed_transport_only",
            "generic_tool_accepted_calls": loaded["tatqa_tools"]["accepted_calls"],
            "source_native_status": "blocked_missing_table_text_operation_adapter",
            "final_answer_status": "pending",
        },
        "crag": {
            "freeze_status": "complete_shared_from_paper1_5",
            "record_count": loaded["crag_manifest"]["record_count"],
            "generic_tool_status": "oracle_timed_transport_only",
            "generic_tool_oracle_agreement": crag_transport[
                "backend_agreement_with_registered_oracle"
            ],
            "source_native_status": "blocked_missing_frozen_evidence_backend",
            "final_answer_status": "pending",
        },
        "spider2": {"freeze_status": "blocked_no_local_subset"},
        "frames": {"freeze_status": "blocked_no_bounded_subset"},
        "bird_interact": {"freeze_status": "deferred_until_postgres_live_integration"},
    }
    manifest = {
        "schema_version": "ccpu.paper2_5.public_suite_readiness.v1",
        "input_sha256": {name: file_sha256(path) for name, path in inputs.items()},
        "benchmarks": benchmarks,
        "headline_ready": False,
        "completed_empirical_benchmarks": ["tatqa_retrieval_only"],
        "blocking_requirements": [
            "tatqa_source_native_table_text_operation_adapter",
            "spider2_local_subset",
            "crag_frozen_evidence_backend",
            "frames_bounded_subset",
            "matched_model_facing_conditions",
        ],
        "claim_boundary": (
            "readiness and transport ledger; not a cross-benchmark end-to-end comparison"
        ),
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(Path(output_dir) / "manifest.json", manifest)
    return manifest
