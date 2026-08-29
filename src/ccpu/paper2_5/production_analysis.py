"""Backend-substitution and operational analysis for local production sources."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ccpu.common.metrics import safe_mean

REQUIRED_PROVENANCE = (
    "backend",
    "backend_version",
    "resource",
    "normalized_query",
    "record_ids",
    "snapshot",
)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def analyze_substitution(
    controlled: list[dict[str, Any]],
    production: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare matched predictions and summarize production-native evidence."""

    key = lambda row: (row["example_id"], row["condition"], int(row["source_count"]))
    controlled_by_key = {key(row): row for row in controlled}
    production_by_key = {key(row): row for row in production}
    if controlled_by_key.keys() != production_by_key.keys():
        raise ValueError("controlled and production prediction keys do not match")

    comparisons = [
        (
            controlled_by_key[item_key],
            production_by_key[item_key],
        )
        for item_key in sorted(controlled_by_key)
    ]
    oracle_rows = [
        row
        for row in production
        if row["condition"] == "oracle_need_source_query" and row["gold_need"]
    ]
    trace_by_example = {
        trace["example_id"]: trace
        for trace in traces
        if trace["condition"] == "oracle_need_source_query"
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        grouped[str(row["gold_source"])].append(row)

    by_source = []
    for source_type, rows in sorted(grouped.items()):
        evidence = [
            item
            for row in rows
            for item in trace_by_example[row["example_id"]]["evidence"]
        ]
        provenance = [dict(item["provenance"]) for item in evidence]
        latencies = [float(row["source_latency_ns"]) / 1e6 for row in rows]
        backends = sorted({str(item.get("backend", "controlled")) for item in provenance})
        versions = sorted({str(item.get("backend_version", "n/a")) for item in provenance})
        by_source.append(
            {
                "source_type": source_type,
                "backend": backends,
                "backend_version": versions,
                "count": len(rows),
                "final_accuracy": safe_mean(row["final_correct"] for row in rows),
                "evidence_support_rate": safe_mean(
                    row["evidence_supported"] for row in rows
                ),
                "mean_source_latency_ms": safe_mean(latencies),
                "p95_source_latency_ms": _p95(latencies),
                "provenance_complete_rate": safe_mean(
                    all(field in item and item[field] not in (None, "", []) for field in REQUIRED_PROVENANCE)
                    for item in provenance
                ),
            }
        )

    from .production_sources import build_production_sources

    sources = build_production_sources()
    operations = []
    for source_type in ("db", "lexical", "vector"):
        source = sources[source_type]
        operations.append(
            {
                "source_type": source_type,
                "backend": source.descriptor.backend,
                "backend_version": source.descriptor.backend_version,
                "startup_ms": source.startup_ns / 1e6,
                "capability_count": len(source.descriptor.capabilities),
                "resource_count": len(source.descriptor.resources),
                "external_service_count": 0,
                "network_calls": 0,
            }
        )

    return {
        "schema_version": "ccpu.paper2_5.production_substitution.v1",
        "matched_prediction_count": len(comparisons),
        "final_decision_agreement": safe_mean(
            controlled_row["final_correct"] == production_row["final_correct"]
            for controlled_row, production_row in comparisons
        ),
        "support_decision_agreement": safe_mean(
            controlled_row["evidence_supported"] == production_row["evidence_supported"]
            for controlled_row, production_row in comparisons
        ),
        "production_final_accuracy": safe_mean(row["final_correct"] for row in production),
        "by_source_oracle": by_source,
        "local_operations": operations,
        "claim_boundary": {
            "validated": ["duckdb", "sqlite_fts5", "faiss"],
            "not_executed": [
                "postgres",
                "pgvector",
                "iceberg_rest",
                "qdrant",
                "weaviate",
            ],
            "docker_used": False,
        },
    }
