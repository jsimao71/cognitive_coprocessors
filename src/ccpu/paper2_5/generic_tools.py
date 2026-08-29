"""Generic-tool result transport audit over the governed enterprise runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, write_json, write_jsonl
from ccpu.common.generic_gateway import (
    GenericCognitiveGateway,
    GenericToolCall,
    generic_tool_schemas,
)
from ccpu.common.lexical_routing import current_word_tokens

from .enterprise import run_enterprise_evaluation


def compare_enterprise_result_transports(
    fixture_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    predictions, source_summary = run_enterprise_evaluation(fixture_root)
    native = [row for row in predictions if row["condition"] == "native_governed"]

    def resolve(intent: str, payload: Any, active_task: str) -> Any:
        del active_task
        if intent not in {"retrieve", "help"} or not isinstance(payload, dict):
            return None
        return payload.get("typed_record")

    gateway = GenericCognitiveGateway(resolve)
    rows = []
    for row in native:
        typed_record = {
            "answer": row["answer"],
            "provenance": row["provenance"],
            "source_calls": row["source_calls"],
        }
        tool_record = gateway.invoke(
            GenericToolCall("__help", {"typed_record": typed_record}),
            active_task=row["question"],
        )
        rows.append(
            {
                "example_id": row["example_id"],
                "question_class": row["question_class"],
                "expected": row["expected"],
                "cogcop_record": typed_record,
                "tool_record": tool_record,
                "record_equal": tool_record == typed_record,
                "correct": tool_record["answer"] == row["expected"],
            }
        )
    schemas = generic_tool_schemas()
    schema_tokens = sum(len(current_word_tokens(str(schema))) for schema in schemas)
    registry_scaling = [
        {"stage": "duckdb_fts5_faiss", "capability_count": 3},
        {"stage": "iceberg_metrics_oxigraph", "capability_count": 6},
        {"stage": "service_adapters", "capability_count": 9},
    ]
    for item in registry_scaling:
        item.update(tool_count=4, schema_tokens=schema_tokens)
    output = Path(output_dir)
    prediction_path = write_jsonl(output / "predictions.jsonl", rows)
    summary = {
        "schema_version": "ccpu.paper2_5.generic_tool_transport.v1",
        "row_count": len(rows),
        "record_agreement": sum(row["record_equal"] for row in rows) / len(rows),
        "tool_result_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "registry_scaling": registry_scaling,
        "source_summary": source_summary,
        "predictions_sha256": file_sha256(prediction_path),
        "claim_boundary": {
            "timing": "oracle",
            "r2_execution": "precomputed shared native_governed run",
            "voluntary_model_call": False,
            "automatic_rescue_rate": None,
            "result": "result transport and schema invariance only",
        },
    }
    write_json(output / "summary.json", summary)
    return summary
