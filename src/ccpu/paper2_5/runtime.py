"""Read-only source registry, heuristic router, and universal baseline."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from ccpu.common.retrieval import EvidenceRecord, EvidenceSource, SourceRequest

from .sources import _record, _tokens, controlled_corpora

SOURCE_CATALOGS = {
    1: ("db",),
    2: ("db", "lexical"),
    3: ("db", "lexical", "vector"),
    4: ("db", "lexical", "vector", "web"),
}


class RetrievalRegistry:
    def __init__(self, sources: dict[str, EvidenceSource], credentials: set[str]) -> None:
        self.sources = dict(sources)
        self.credentials = set(credentials)

    def retrieve(self, request: SourceRequest) -> tuple[EvidenceRecord, ...]:
        source = self.sources.get(request.source_type)
        if source is None:
            return ()
        if source.policy.side_effect_class != "read_only":
            raise PermissionError("Paper 2.5 registry accepts read-only sources only")
        if source.policy.credential_scope not in self.credentials:
            raise PermissionError(f"credential scope denied for source: {request.source_type}")
        return source.retrieve(request)

    def public_catalog(self) -> list[dict[str, str]]:
        rows = []
        for source in self.sources.values():
            policy = source.policy.to_dict()
            policy.pop("credential_scope")
            rows.append(policy)
        return sorted(rows, key=lambda row: row["source_type"])


class HeuristicSourceRouter:
    """Transparent source selection over a typed information-need event."""

    def select(self, need: dict[str, Any], available: tuple[str, ...]) -> str | None:
        text = " ".join(str(value) for value in need.values()).casefold()
        preferences = []
        if any(term in text for term in ("aggregate", "sales", "table", "join", "count")):
            preferences.append("db")
        if any(term in text for term in ("document", "clause", "exact", "policy", "handbook")):
            preferences.append("lexical")
        if any(term in text for term in ("semantic", "reason", "why", "report", "theme")):
            preferences.append("vector")
        if any(term in text for term in ("current", "latest", "recent", "public", "fresh")):
            preferences.append("web")
        return next((source for source in preferences if source in available), None)


class UniversalTextSource:
    """Strong single-index baseline over textualized records from available sources."""

    def __init__(self, source_types: tuple[str, ...]) -> None:
        from ccpu.common.retrieval import SourcePolicy

        self.policy = SourcePolicy(
            source_type="universal",
            source_id=f"universal-text-{'-'.join(source_types)}",
            locality="local",
            credential_scope="runtime:universal:read",
            latency_class="local_medium",
            cost_class="cpu",
            privacy_class="mixed",
            freshness="mixed_snapshots",
        )
        corpora = controlled_corpora()
        self.documents = {
            identifier: content
            for source_type in source_types
            for identifier, content in corpora[source_type].items()
        }

    def retrieve(self, request: SourceRequest) -> tuple[EvidenceRecord, ...]:
        started = time.perf_counter_ns()
        query = Counter(_tokens(str(request.payload.get("query", ""))))
        scores = []
        for identifier, content in self.documents.items():
            document = Counter(_tokens(content))
            score = sum(min(count, document[token]) for token, count in query.items())
            scores.append((score, identifier, content))
        score, identifier, content = max(scores)
        if score <= 0:
            return ()
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=identifier,
                value=content,
                content=content,
                started=started,
                relevance=score / max(1, sum(query.values())),
                provenance={"baseline": "universal_text_top1"},
            ),
        )


def source_request(row: dict[str, Any], source_type: str) -> SourceRequest:
    if source_type == row["source"]:
        operation = str(row["operation"])
        payload = dict(row["payload"])
    else:
        operation = f"{source_type}.search"
        payload = {"query": row["prompt"]}
    return SourceRequest(
        request_id=f"{row['example_id']}:{source_type}",
        source_type=source_type,
        operation=operation,
        payload=payload,
        budget={"max_records": 4, "max_bytes": 4096},
    )
