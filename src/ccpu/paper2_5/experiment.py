"""Oracle matrix, source-count scaling, universal, and broadcast baselines."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import read_jsonl
from ccpu.common.metrics import safe_mean
from ccpu.common.retrieval import SourceRequest

from .runtime import (
    SOURCE_CATALOGS,
    HeuristicSourceRouter,
    RetrievalRegistry,
    UniversalTextSource,
    source_request,
)
from .sources import build_sources

CONDITIONS = (
    "oracle_need_source_query",
    "real_need_oracle_source",
    "real_need_heuristic_source",
    "heuristic_need_heuristic_source",
    "explicit_source_selection",
    "universal_retriever",
    "broadcast",
)


def _credentials(sources: dict[str, Any]) -> set[str]:
    return {source.policy.credential_scope for source in sources.values()}


def _heuristic_need(row: dict[str, Any]) -> bool:
    text = str(row["prompt"]).casefold()
    if "active context supplies" in text:
        return False
    cues = (
        "sales",
        "table",
        "join",
        "count",
        "document",
        "clause",
        "exact",
        "report",
        "why",
        "semantic",
        "latest",
        "current",
        "public",
    )
    return any(cue in text for cue in cues)


def _answer_supported(answer: str, evidence: tuple[Any, ...]) -> bool:
    expected = answer.casefold().strip()
    for item in evidence:
        if str(item.value).casefold().strip() == expected:
            return True
        content = item.content.casefold().strip()
        if expected == content:
            return True
        if len(expected) <= 16 and re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", content):
            return True
    return False


def _schema_tokens(source_count: int) -> int:
    descriptors = {
        "db": "structured database lookup aggregate join typed relational request",
        "lexical": "exact document phrase lexical request document identifier query",
        "vector": "semantic report vector collection natural language query",
        "web": "current public web entity relation time window request",
    }
    return sum(len(descriptors[source].split()) for source in SOURCE_CATALOGS[source_count])


def run_matrix(
    benchmark_path: str | Path, *, source_count: int, backend_suite: str = "controlled"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = SOURCE_CATALOGS[source_count]
    examples = [
        row
        for row in read_jsonl(benchmark_path)
        if row["source"] == "control" or row["source"] in available
    ]
    if backend_suite == "controlled":
        source_suite = build_sources()
    elif backend_suite == "local_production":
        from .production_sources import build_production_sources

        source_suite = build_production_sources()
    else:
        raise ValueError(f"unknown backend suite: {backend_suite}")
    sources = {name: source for name, source in source_suite.items() if name in available}
    registry = RetrievalRegistry(sources, _credentials(sources))
    router = HeuristicSourceRouter()
    universal = UniversalTextSource(available)
    rows = []
    traces = []
    for row in examples:
        for condition in CONDITIONS:
            started = time.perf_counter_ns()
            gold_need = bool(row["should_retrieve"])
            need = gold_need
            selected_sources: list[str] = []
            evidence = ()
            requests: list[SourceRequest] = []
            descriptor_tokens = 0
            if condition == "heuristic_need_heuristic_source":
                need = _heuristic_need(row)
            if need:
                if condition in {"oracle_need_source_query", "real_need_oracle_source"}:
                    selected_sources = [str(row["source"])]
                elif condition in {
                    "real_need_heuristic_source",
                    "heuristic_need_heuristic_source",
                    "explicit_source_selection",
                }:
                    selected = router.select(dict(row["need"]), available)
                    selected_sources = [selected] if selected else []
                    if condition == "explicit_source_selection":
                        descriptor_tokens = _schema_tokens(source_count)
                elif condition == "universal_retriever":
                    selected_sources = ["universal"]
                    descriptor_tokens = 5
                elif condition == "broadcast":
                    selected_sources = list(available)

            if selected_sources == ["universal"]:
                request = SourceRequest(
                    request_id=f"{row['example_id']}:universal",
                    source_type="universal",
                    operation="universal.search",
                    payload={"query": row["prompt"]},
                    budget={"max_records": 1},
                )
                requests.append(request)
                evidence = universal.retrieve(request)
            else:
                gathered = []
                for source_type in selected_sources:
                    request = source_request(row, source_type)
                    requests.append(request)
                    gathered.extend(registry.retrieve(request))
                evidence = tuple(gathered)

            source_correct = (
                not gold_need and not selected_sources
                or gold_need and str(row["source"]) in selected_sources
            )
            query_correct = bool(
                not gold_need
                or any(
                    request.source_type == row["source"]
                    and request.operation == row["operation"]
                    and dict(request.payload) == row["payload"]
                    for request in requests
                )
            )
            conflict = bool(evidence) and (
                any(item.status == "CONFLICT" for item in evidence)
                or len({str(item.value) for item in evidence}) > 1
            )
            supported = _answer_supported(str(row["answer"]), evidence) or (
                row["answer"] == "ABSTAIN" and conflict
            )
            final_correct = supported if gold_need else not evidence
            status = (
                "CONFLICT"
                if conflict
                else "SUPPORTED"
                if supported
                else "UNVERIFIED"
            )
            elapsed = time.perf_counter_ns() - started
            prediction = {
                "schema_version": "ccpu.paper2_5.prediction.v1",
                "backend_suite": backend_suite,
                "example_id": row["example_id"],
                "condition": condition,
                "source_count": source_count,
                "gold_need": gold_need,
                "need_detected": need,
                "gold_source": row["source"],
                "selected_sources": selected_sources,
                "source_correct": source_correct,
                "query_correct": query_correct,
                "retrieved": bool(evidence),
                "evidence_status": status,
                "evidence_supported": supported,
                "final_correct": final_correct,
                "unsupported_commitment": gold_need and not supported,
                "fanout": len(selected_sources),
                "source_calls": len(requests),
                "bytes_retrieved": sum(item.bytes_retrieved for item in evidence),
                "evidence_tokens": sum(len(item.content.split()) for item in evidence),
                "descriptor_tokens": descriptor_tokens,
                "query_tokens": sum(
                    len(str(request.payload).split()) for request in requests
                ),
                "source_latency_ns": sum(item.latency_ns for item in evidence),
                "wall_time_ns": elapsed,
            }
            rows.append(prediction)
            traces.append(
                {
                    "schema_version": "ccpu.paper2_5.trace.v1",
                    "backend_suite": backend_suite,
                    "example_id": row["example_id"],
                    "condition": condition,
                    "source_count": source_count,
                    "need": {"gold": gold_need, "detected": need},
                    "routing": {
                        "available": list(available),
                        "selected": selected_sources,
                        "gold": row["source"],
                    },
                    "requests": [request.to_dict() for request in requests],
                    "evidence": [item.to_dict() for item in evidence],
                    "decision": {"status": status, "accepted": supported},
                }
            )
    return rows, traces


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), int(row["source_count"]))].append(row)
    cells = []
    for (condition, source_count), group in sorted(grouped.items()):
        tasks = [row for row in group if row["gold_need"]]
        cells.append(
            {
                "condition": condition,
                "source_count": source_count,
                "count": len(group),
                "need_accuracy": safe_mean(
                    row["need_detected"] == row["gold_need"] for row in group
                ),
                "source_accuracy": safe_mean(row["source_correct"] for row in tasks),
                "query_accuracy": safe_mean(row["query_correct"] for row in tasks),
                "retrieval_success": safe_mean(row["retrieved"] for row in tasks),
                "evidence_support_rate": safe_mean(
                    row["evidence_supported"] for row in tasks
                ),
                "final_accuracy": safe_mean(row["final_correct"] for row in group),
                "unsupported_commitment_rate": safe_mean(
                    row["unsupported_commitment"] for row in tasks
                ),
                "mean_fanout": safe_mean(row["fanout"] for row in group),
                "mean_source_calls": safe_mean(row["source_calls"] for row in group),
                "mean_bytes_retrieved": safe_mean(row["bytes_retrieved"] for row in group),
                "mean_evidence_tokens": safe_mean(row["evidence_tokens"] for row in group),
                "mean_descriptor_tokens": safe_mean(
                    row["descriptor_tokens"] for row in group
                ),
                "mean_query_tokens": safe_mean(row["query_tokens"] for row in group),
                "mean_source_latency_ms": safe_mean(
                    row["source_latency_ns"] for row in group
                )
                / 1e6,
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in group) / 1e6,
            }
        )
    return {
        "schema_version": "ccpu.paper2_5.evaluation.v1",
        "prediction_count": len(rows),
        "by_condition_source_count": cells,
    }
