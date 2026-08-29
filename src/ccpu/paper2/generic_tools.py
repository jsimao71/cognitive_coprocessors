"""Matched oracle transport audit for Paper 2 generic cognitive tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.common.generic_gateway import (
    GenericCognitiveGateway,
    GenericToolCall,
    generic_tool_schemas,
)
from ccpu.common.lexical_routing import current_word_tokens

from .benchmark_next import ENGINE_CATALOGS
from .runtime import HeterogeneousRuntime


def compare_oracle_transports(benchmark: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = []
    for item in read_jsonl(benchmark):
        direct_runtime = HeterogeneousRuntime()
        tool_runtime = HeterogeneousRuntime()

        def resolve(
            intent: str,
            payload: Any,
            active_task: str,
            runtime: HeterogeneousRuntime = tool_runtime,
            example_id: str = str(item["example_id"]),
        ) -> Any:
            del active_task
            if intent != "compute" or not isinstance(payload, dict):
                return None
            return runtime.execute_event(
                str(payload.get("event", "")), event_id=f"tool:{example_id}"
            )

        direct = None
        tool = None
        if item["should_trigger"]:
            direct = direct_runtime.execute_event(
                str(item["target"]), event_id=f"block:{item['example_id']}"
            )
            tool = GenericCognitiveGateway(resolve).invoke(
                GenericToolCall("__compute", {"event": item["target"]}),
                active_task=str(item["prompt"]),
            )
        direct_display = direct.display if direct and direct.ok else None
        tool_display = tool.display if tool and tool.ok else None
        rows.append(
            {
                "example_id": item["example_id"],
                "should_assist": item["should_trigger"],
                "block_result": direct_display,
                "tool_result": tool_display,
                "backend_equal": direct_display == tool_display,
                "block_correct": direct_display == item["answer"] if direct_display else not item["should_trigger"],
                "tool_correct": tool_display == item["answer"] if tool_display else not item["should_trigger"],
            }
        )
    schemas = generic_tool_schemas()
    schema_tokens = sum(len(current_word_tokens(str(schema))) for schema in schemas)
    scaling = [
        {
            "engine_count": count,
            "engines": list(engines),
            "tool_count": len(schemas),
            "schema_tokens": schema_tokens,
        }
        for count, engines in ENGINE_CATALOGS.items()
    ]
    output = Path(output_dir)
    prediction_path = write_jsonl(output / "predictions.jsonl", rows)
    summary = {
        "schema_version": "ccpu.paper2.generic_tool_transport.v1",
        "row_count": len(rows),
        "assistance_required": sum(row["should_assist"] for row in rows),
        "backend_result_agreement": sum(row["backend_equal"] for row in rows) / len(rows),
        "block_accuracy": sum(row["block_correct"] for row in rows) / len(rows),
        "tool_accuracy": sum(row["tool_correct"] for row in rows) / len(rows),
        "registry_scaling": scaling,
        "benchmark_sha256": file_sha256(benchmark),
        "predictions_sha256": file_sha256(prediction_path),
        "claim_boundary": {
            "timing": "oracle",
            "voluntary_model_call": False,
            "automatic_rescue_rate": None,
            "result": "transport and schema invariance only",
        },
    }
    write_json(output / "summary.json", summary)
    return summary
