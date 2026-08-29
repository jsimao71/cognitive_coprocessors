"""Four semantically distinct controlled evidence adapters."""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter
from typing import Any

from ccpu.common.artifacts import canonical_json
from ccpu.common.retrieval import EvidenceRecord, SourcePolicy, SourceRequest


def _record(
    *,
    request: SourceRequest,
    policy: SourcePolicy,
    record_id: str,
    value: Any,
    content: str,
    started: int,
    relevance: float = 1.0,
    status: str = "SUPPORTED",
    provenance: dict[str, Any] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        request_id=request.request_id,
        source_type=policy.source_type,
        source_id=policy.source_id,
        record_id=record_id,
        status=status,
        value=value,
        content=content,
        observed_at="2026-08-29",
        relevance=relevance,
        provenance={"operation": request.operation, **(provenance or {})},
        latency_ns=time.perf_counter_ns() - started,
        bytes_retrieved=len(content.encode("utf-8")),
    )


class StructuredDBSource:
    policy = SourcePolicy(
        source_type="db",
        source_id="sqlite-sales-2026.08",
        locality="local",
        credential_scope="runtime:db:read",
        latency_class="local_low",
        cost_class="cpu",
        privacy_class="internal",
        freshness="snapshot:2026-08-28",
    )

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript(
            """
            CREATE TABLE products(product TEXT PRIMARY KEY, category TEXT, owner_id INTEGER);
            CREATE TABLE sales(product TEXT, year INTEGER, amount INTEGER);
            CREATE TABLE owners(owner_id INTEGER PRIMARY KEY, owner TEXT);
            INSERT INTO products VALUES
              ('Aster','hardware',1),('Birch','software',2),('Cedar','hardware',3);
            INSERT INTO owners VALUES (1,'Mina'),(2,'Rui'),(3,'Tala');
            INSERT INTO sales VALUES
              ('Aster',2025,80),('Aster',2026,125),
              ('Birch',2025,110),('Birch',2026,205),
              ('Cedar',2025,95),('Cedar',2026,170);
            """
        )

    def retrieve(self, request: SourceRequest) -> tuple[EvidenceRecord, ...]:
        started = time.perf_counter_ns()
        payload = request.payload
        operation = request.operation
        if operation == "db.lookup":
            sql = "SELECT amount FROM sales WHERE product=? AND year=?"
            parameters = (payload["product"], int(payload["year"]))
        elif operation == "db.max_sales":
            sql = "SELECT product, amount FROM sales WHERE year=? ORDER BY amount DESC, product LIMIT 1"
            parameters = (int(payload["year"]),)
        elif operation == "db.sum_sales":
            sql = "SELECT SUM(amount) FROM sales WHERE year=?"
            parameters = (int(payload["year"]),)
        elif operation == "db.count_products":
            sql = "SELECT COUNT(*) FROM products WHERE category=?"
            parameters = (payload["category"],)
        elif operation == "db.avg_sales":
            sql = "SELECT AVG(amount) FROM sales WHERE year=?"
            parameters = (int(payload["year"]),)
        elif operation == "db.owner_join":
            sql = "SELECT owners.owner FROM products JOIN owners USING(owner_id) WHERE product=?"
            parameters = (payload["product"],)
        else:
            return ()
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            return ()
        if operation == "db.max_sales":
            value: Any = str(row[0])
        elif operation == "db.avg_sales":
            value = format(float(row[0]), ".1f")
        else:
            value = str(row[0])
        content = f"{operation}({canonical_json(dict(payload))}) = {value}"
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=f"db:{operation}:{canonical_json(dict(payload))}",
                value=value,
                content=content,
                started=started,
                provenance={"sql_template": operation, "parameters": list(parameters)},
            ),
        )


_DOCUMENTS = {
    "policy_17": (
        "Termination notice",
        "A supplier termination requires ninety calendar days of written notice.",
    ),
    "policy_23": (
        "Travel approval",
        "International travel requires director approval before ticket purchase.",
    ),
    "handbook_9": (
        "Data retention",
        "Audit exports must be retained for seven complete years.",
    ),
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


class LexicalSource:
    policy = SourcePolicy(
        source_type="lexical",
        source_id="policy-bm25-2026.08",
        locality="local",
        credential_scope="runtime:docs:read",
        latency_class="local_low",
        cost_class="cpu",
        privacy_class="internal",
        freshness="snapshot:2026-08-20",
    )

    def retrieve(self, request: SourceRequest) -> tuple[EvidenceRecord, ...]:
        started = time.perf_counter_ns()
        document = str(request.payload.get("document", ""))
        query = str(request.payload.get("query", ""))
        candidates = [document] if document in _DOCUMENTS else list(_DOCUMENTS)
        query_tokens = Counter(_tokens(query))
        scored = []
        for identifier in candidates:
            title, content = _DOCUMENTS[identifier]
            words = Counter(_tokens(f"{title} {content}"))
            score = sum(min(count, words[token]) for token, count in query_tokens.items())
            scored.append((score, identifier, content))
        score, identifier, content = max(scored)
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
                relevance=score / max(1, sum(query_tokens.values())),
                provenance={"matching": "token_overlap", "document": identifier},
            ),
        )


_REPORTS = {
    "report_margin_q2": "Second-quarter margin fell because freight expenses rose and discounting increased.",
    "report_churn_q3": "Customer attrition improved after onboarding was simplified and response time fell.",
    "report_energy_q1": "Electricity consumption declined after cooling controls and server consolidation.",
}
_CONCEPTS = {
    "margin": "margin",
    "profitability": "margin",
    "freight": "shipping",
    "shipping": "shipping",
    "transport": "shipping",
    "discounting": "discount",
    "discounts": "discount",
    "churn": "attrition",
    "attrition": "attrition",
    "onboarding": "onboarding",
    "electricity": "energy",
    "energy": "energy",
    "cooling": "cooling",
}


def _semantic_vector(text: str) -> Counter[str]:
    return Counter(_CONCEPTS.get(token, token) for token in _tokens(text))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right[key] for key, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class VectorSource:
    policy = SourcePolicy(
        source_type="vector",
        source_id="report-semantic-v1",
        locality="local",
        credential_scope="runtime:vectors:read",
        latency_class="local_medium",
        cost_class="cpu",
        privacy_class="internal",
        freshness="snapshot:2026-Q3",
    )

    def retrieve(self, request: SourceRequest) -> tuple[EvidenceRecord, ...]:
        started = time.perf_counter_ns()
        query = _semantic_vector(str(request.payload.get("query", "")))
        scored = [
            (_cosine(query, _semantic_vector(content)), identifier, content)
            for identifier, content in _REPORTS.items()
        ]
        score, identifier, content = max(scored)
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
                relevance=score,
                provenance={"embedding": "controlled-semantic-v1", "collection": "reports"},
            ),
        )


_WEB = {
    ("Northstar mission", "launch date"): ("2032-04-18", "2026-08-29"),
    ("Lumen index", "current level"): ("417", "2026-08-29"),
    ("Port Azure", "current status"): ("restricted", "2026-08-29"),
}
_WEB_CONFLICTS = {
    ("Orion bulletin", "current code"): (("OR-7", "2026-08-29"), ("OR-9", "2026-08-29"))
}


class ControlledWebSource:
    policy = SourcePolicy(
        source_type="web",
        source_id="controlled-public-feed-2026-08-29",
        locality="remote_analogue",
        credential_scope="runtime:web:read",
        latency_class="network",
        cost_class="external_call",
        privacy_class="public",
        freshness="daily",
    )

    def retrieve(self, request: SourceRequest) -> tuple[EvidenceRecord, ...]:
        started = time.perf_counter_ns()
        key = (str(request.payload.get("entity", "")), str(request.payload.get("relation", "")))
        conflicts = _WEB_CONFLICTS.get(key)
        if conflicts:
            return tuple(
                _record(
                    request=request,
                    policy=self.policy,
                    record_id=f"web:{key[0]}:{key[1]}:{index}",
                    value=value,
                    content=f"As observed {observed}, {key[0]} {key[1]} is {value}.",
                    started=started,
                    status="CONFLICT",
                    provenance={
                        "feed_observed_at": observed,
                        "controlled_analogue": True,
                        "conflict_set": key,
                    },
                )
                for index, (value, observed) in enumerate(conflicts)
            )
        item = _WEB.get(key)
        if item is None:
            return ()
        value, observed = item
        content = f"As observed {observed}, {key[0]} {key[1]} is {value}."
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=f"web:{key[0]}:{key[1]}",
                value=value,
                content=content,
                started=started,
                provenance={"feed_observed_at": observed, "controlled_analogue": True},
            ),
        )


def controlled_corpora() -> dict[str, dict[str, str]]:
    """Textual snapshots used by the matched universal-retriever baseline."""

    db_rows = {
        f"db:sales:{product}:{year}": f"{product} sales in {year} were {amount}."
        for product, year, amount in (
            ("Aster", 2025, 80),
            ("Aster", 2026, 125),
            ("Birch", 2025, 110),
            ("Birch", 2026, 205),
            ("Cedar", 2025, 95),
            ("Cedar", 2026, 170),
        )
    }
    lexical = {identifier: content for identifier, (_, content) in _DOCUMENTS.items()}
    web = {
        f"web:{entity}:{relation}": f"Current {relation} for {entity} is {value}."
        for (entity, relation), (value, _) in _WEB.items()
    }
    for (entity, relation), records in _WEB_CONFLICTS.items():
        for index, (value, _) in enumerate(records):
            web[f"web:{entity}:{relation}:{index}"] = (
                f"Current {relation} for {entity} is {value}."
            )
    return {"db": db_rows, "lexical": lexical, "vector": dict(_REPORTS), "web": web}


def build_sources() -> dict[str, Any]:
    return {
        "db": StructuredDBSource(),
        "lexical": LexicalSource(),
        "vector": VectorSource(),
        "web": ControlledWebSource(),
    }
