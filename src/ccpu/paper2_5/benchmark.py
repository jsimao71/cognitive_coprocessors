"""Deterministic source-optimal benchmark for heterogeneous retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccpu.common.artifacts import fingerprint, write_json, write_jsonl

from .sources import _DOCUMENTS, _REPORTS


def benchmark_rows() -> list[dict[str, Any]]:
    templates = [
        ("db", "db.lookup", {"product": "Aster", "year": 2026}, "125", "What were Aster sales in the 2026 table?"),
        ("db", "db.max_sales", {"year": 2026}, "Birch", "Which product has maximum sales in 2026?"),
        ("db", "db.sum_sales", {"year": 2026}, "500", "What is the aggregate sum of all 2026 sales?"),
        ("db", "db.count_products", {"category": "hardware"}, "2", "Count hardware products in the product table."),
        ("db", "db.avg_sales", {"year": 2025}, "95.0", "What is the average sales amount for 2025?"),
        ("db", "db.owner_join", {"product": "Cedar"}, "Tala", "Join products to owners: who owns Cedar?"),
        ("lexical", "lexical.search", {"document": "policy_17", "query": "termination notice"}, _DOCUMENTS["policy_17"][1], "In document policy_17, retrieve the exact termination notice clause."),
        ("lexical", "lexical.search", {"document": "policy_23", "query": "travel approval"}, _DOCUMENTS["policy_23"][1], "Find the exact travel approval clause in policy_23."),
        ("lexical", "lexical.search", {"document": "handbook_9", "query": "data retention"}, _DOCUMENTS["handbook_9"][1], "What does handbook_9 say exactly about data retention?"),
        ("vector", "vector.search", {"collection": "reports", "query": "reasons profitability declined due to shipping and discounts"}, _REPORTS["report_margin_q2"], "Semantically retrieve the report explaining why Q2 profitability declined."),
        ("vector", "vector.search", {"collection": "reports", "query": "why customer churn improved"}, _REPORTS["report_churn_q3"], "Which report explains the reasons customer churn improved?"),
        ("vector", "vector.search", {"collection": "reports", "query": "theme behind lower energy use"}, _REPORTS["report_energy_q1"], "Find the semantic report about why energy use fell."),
        ("web", "web.lookup", {"entity": "Northstar mission", "relation": "launch date", "time_window": "current"}, "2032-04-18", "What is the latest public launch date for Northstar mission?"),
        ("web", "web.lookup", {"entity": "Lumen index", "relation": "current level", "time_window": "current"}, "417", "What is the current public level of the Lumen index?"),
        ("web", "web.lookup", {"entity": "Port Azure", "relation": "current status", "time_window": "current"}, "restricted", "What is Port Azure's latest public status?"),
        ("web", "web.lookup", {"entity": "Orion bulletin", "relation": "current code", "time_window": "current"}, "ABSTAIN", "What is the current public code in the conflicting Orion bulletin feeds?"),
    ]
    rows = []
    for index, (source, operation, payload, answer, prompt) in enumerate(templates):
        need = {
            "entity": payload.get("product") or payload.get("document") or payload.get("entity", ""),
            "relation": operation,
            "source_hint": prompt,
            "time": payload.get("year") or payload.get("time_window", ""),
        }
        rows.append(
            {
                "example_id": f"p25-{source}-{index:03d}",
                "source": source,
                "operation": operation,
                "payload": payload,
                "prompt": prompt,
                "answer": answer,
                "need": need,
                "should_retrieve": True,
            }
        )
    for index in range(6):
        value = f"SUPPLIED-{index:03d}"
        rows.append(
            {
                "example_id": f"p25-control-{index:03d}",
                "source": "control",
                "operation": "none",
                "payload": {},
                "prompt": f"The active context supplies {value}; repeat it without retrieval.",
                "answer": value,
                "need": {"source_hint": "supplied active context"},
                "should_retrieve": False,
            }
        )
    return rows


def freeze_benchmark(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows = benchmark_rows()
    path = write_jsonl(output_dir / "benchmark.jsonl", rows)
    manifest = {
        "schema_version": "ccpu.paper2_5.benchmark.v1",
        "count": len(rows),
        "source_counts": {
            source: sum(row["source"] == source for row in rows)
            for source in ("db", "lexical", "vector", "web", "control")
        },
        "fingerprint": fingerprint(rows),
        "path": str(path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
