"""Cross-paper public benchmark registry for matched control experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json, write_jsonl


def freeze_public_control_registry(
    compute_selection: str | Path,
    crag_selection: str | Path,
    tatqa_selection: str | Path,
    output_dir: str | Path,
    *,
    per_benchmark: int = 40,
    crag_diagnostics: str | Path | None = None,
    paper2_5_readiness: str | Path | None = None,
) -> dict[str, Any]:
    sources = {
        "paper2_compute": Path(compute_selection),
        "paper1_5_crag": Path(crag_selection),
        "paper2_5_tatqa": Path(tatqa_selection),
    }
    candidates = []
    for source_name, path in sources.items():
        for row in read_jsonl(path):
            benchmark = str(row["benchmark"])
            assistance = "COMPUTE" if source_name == "paper2_compute" else "RETRIEVE"
            secondary = (
                "COMPUTE"
                if benchmark == "tatqa" and row.get("answer_type") in {"arithmetic", "count"}
                else None
            )
            candidates.append(
                {
                    "schema_version": "ccpu.paper3.public_control_selection.v1",
                    "source": source_name,
                    "benchmark": benchmark,
                    "example_id": row["example_id"],
                    "content_sha256": row["content_sha256"],
                    "selection_key": row["selection_key"],
                    "assistance_type": assistance,
                    "secondary_assistance_type": secondary,
                    "span_annotation_status": "pending_audit",
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["benchmark"]].append(row)
    selected = []
    for benchmark in sorted(grouped):
        selected.extend(
            sorted(grouped[benchmark], key=lambda row: row["selection_key"])[
                :per_benchmark
            ]
        )
    output = Path(output_dir)
    selection_path = write_jsonl(output / "selection.jsonl", selected)
    counts = {benchmark: len(rows[:per_benchmark]) for benchmark, rows in sorted(grouped.items())}
    upstream_diagnostics = {}
    if crag_diagnostics is not None:
        crag = read_json(crag_diagnostics)
        upstream_diagnostics["paper1_5_crag"] = {
            "sha256": file_sha256(crag_diagnostics),
            "paper3_5_gate": crag["interpretation"]["paper3_5_gate"],
            "generic_tool_status": "oracle_timed_transport_only",
        }
    if paper2_5_readiness is not None:
        readiness = read_json(paper2_5_readiness)
        if readiness["headline_ready"] is not False:
            raise ValueError("Paper 2.5 public suite must remain fail closed")
        upstream_diagnostics["paper2_5_public_suite"] = {
            "sha256": file_sha256(paper2_5_readiness),
            "headline_ready": readiness["headline_ready"],
            "claim_boundary": readiness["claim_boundary"],
        }
    manifest = {
        "schema_version": "ccpu.paper3.public_control_manifest.v2",
        "record_count": len(selected),
        "per_benchmark": per_benchmark,
        "counts": counts,
        "source_sha256": {name: file_sha256(path) for name, path in sources.items()},
        "upstream_diagnostics": upstream_diagnostics,
        "selection_sha256": file_sha256(selection_path),
        "required_conditions": [
            "four_generic_tools",
            "paired_tags",
            "fenced_blocks",
            "label_only",
            "cpu_trigger",
            "final_layer_latent",
            "multi_layer_latent",
            "tools_plus_watchdog",
            "oracle_timing_type",
        ],
        "headline_ready": False,
        "blocking_requirements": [
            "frames_subset",
            "audited_causal_span_annotations",
            "matched_model_facing_public_condition_results",
        ],
        "claim_boundary": "registry freeze only; no control condition has run",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
