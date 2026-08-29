"""Bounded deterministic preprocessing compositions for heterogeneous retrieval."""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

from ccpu.common.metrics import safe_mean
from ccpu.common.retrieval import SourceRequest

from .sources import ControlledWebSource, StructuredDBSource


def run_compositions(count_per_family: int = 12) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count_per_family < 1:
        raise ValueError("composition count must be positive")
    rows = []
    aliases = {"aster labs": "Aster", "birch suite": "Birch", "cedar works": "Cedar"}
    answers = {"Aster": "Mina", "Birch": "Rui", "Cedar": "Tala"}
    db = StructuredDBSource()
    for index in range(count_per_family):
        alias = tuple(aliases)[index % len(aliases)]
        started = time.perf_counter_ns()
        canonical = aliases[alias]
        resolver_id = f"entity-resolver:{alias}:{canonical}"
        request = SourceRequest(
            request_id=f"p25-compose-entity-{index:03d}:db",
            source_type="db",
            operation="db.owner_join",
            payload={"product": canonical},
        )
        evidence = db.retrieve(request)
        rows.append(
            {
                "schema_version": "ccpu.paper2_5.composition_prediction.v1",
                "composition_id": f"p25-compose-entity-{index:03d}",
                "family": "entity_resolver_to_db",
                "dependency_dag": [
                    {"id": resolver_id, "kind": "entity_resolution", "depends_on": []},
                    {"id": request.request_id, "kind": "db_request", "depends_on": [resolver_id]},
                ],
                "stage1_correct": canonical == aliases[alias],
                "stage2_correct": bool(evidence and evidence[0].value == answers[canonical]),
                "correct": bool(evidence and evidence[0].value == answers[canonical]),
                "source_calls": 1,
                "bytes_retrieved": sum(item.bytes_retrieved for item in evidence),
                "wall_time_ns": time.perf_counter_ns() - started,
            }
        )

    web = ControlledWebSource()
    web_cases = (
        ("Northstar mission", "launch date", "2032-04-18"),
        ("Lumen index", "current level", "417"),
        ("Port Azure", "current status", "restricted"),
    )
    for index in range(count_per_family):
        entity, relation, answer = web_cases[index % len(web_cases)]
        started = time.perf_counter_ns()
        resolved = date(2026, 8, 29) + timedelta(days=index % 3)
        resolver_id = f"date-resolver:today-plus-{index % 3}:{resolved.isoformat()}"
        request = SourceRequest(
            request_id=f"p25-compose-date-{index:03d}:web",
            source_type="web",
            operation="web.lookup",
            payload={
                "entity": entity,
                "relation": relation,
                "time_window": resolved.isoformat(),
            },
        )
        evidence = web.retrieve(request)
        rows.append(
            {
                "schema_version": "ccpu.paper2_5.composition_prediction.v1",
                "composition_id": f"p25-compose-date-{index:03d}",
                "family": "date_resolver_to_web",
                "dependency_dag": [
                    {"id": resolver_id, "kind": "date_resolution", "depends_on": []},
                    {"id": request.request_id, "kind": "web_request", "depends_on": [resolver_id]},
                ],
                "stage1_correct": request.payload["time_window"] == resolved.isoformat(),
                "stage2_correct": bool(evidence and evidence[0].value == answer),
                "correct": bool(evidence and evidence[0].value == answer),
                "source_calls": 1,
                "bytes_retrieved": sum(item.bytes_retrieved for item in evidence),
                "wall_time_ns": time.perf_counter_ns() - started,
            }
        )
    by_family = []
    for family in ("entity_resolver_to_db", "date_resolver_to_web"):
        group = [row for row in rows if row["family"] == family]
        by_family.append(
            {
                "family": family,
                "count": len(group),
                "stage1_accuracy": safe_mean(row["stage1_correct"] for row in group),
                "stage2_accuracy": safe_mean(row["stage2_correct"] for row in group),
                "final_accuracy": safe_mean(row["correct"] for row in group),
                "mean_source_calls": safe_mean(row["source_calls"] for row in group),
                "mean_bytes_retrieved": safe_mean(row["bytes_retrieved"] for row in group),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in group) / 1e6,
            }
        )
    return rows, {
        "schema_version": "ccpu.paper2_5.composition_evaluation.v1",
        "count": len(rows),
        "by_family": by_family,
    }
