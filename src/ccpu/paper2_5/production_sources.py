"""Local production-shaped DuckDB, SQLite FTS5, and FAISS adapters."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import numpy as np

from ccpu.common.artifacts import canonical_json
from ccpu.common.data_coprocessor import DataCoprocessorDescriptor, production_provenance
from ccpu.common.retrieval import SourcePolicy, SourceRequest

from .sources import (
    _DOCUMENTS,
    _REPORTS,
    ControlledWebSource,
    _record,
    _semantic_vector,
    _tokens,
)


class MissingDataBackendError(RuntimeError):
    """Raised when a requested production backend is not installed."""


def _duckdb_module() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise MissingDataBackendError("install the 'data' extra to use DuckDB") from exc
    return duckdb


def _faiss_module() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise MissingDataBackendError("install the 'data' extra to use FAISS") from exc
    return faiss


class DuckDBSource:
    policy = SourcePolicy(
        source_type="db",
        source_id="duckdb-sales-2026.08",
        locality="local",
        credential_scope="runtime:db:read",
        latency_class="local_low",
        cost_class="cpu",
        privacy_class="internal",
        freshness="snapshot:2026-08-28",
    )

    def __init__(self) -> None:
        duckdb = _duckdb_module()
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="duckdb",
            backend_version=duckdb.__version__,
            capabilities=("lookup", "aggregate", "argmax", "join"),
            request_fields={
                "db.lookup": ("product", "year"),
                "db.max_sales": ("year",),
                "db.sum_sales": ("year",),
                "db.count_products": ("category",),
                "db.avg_sales": ("year",),
                "db.owner_join": ("product",),
            },
            resources=("products", "sales", "owners"),
            snapshot="sales-snapshot-2026-08-28",
        )
        started = time.perf_counter_ns()
        self.connection = duckdb.connect(":memory:")
        self.connection.execute(
            "CREATE TABLE products(product VARCHAR PRIMARY KEY, category VARCHAR, owner_id INTEGER)"
        )
        self.connection.execute("CREATE TABLE sales(product VARCHAR, year INTEGER, amount INTEGER)")
        self.connection.execute("CREATE TABLE owners(owner_id INTEGER PRIMARY KEY, owner VARCHAR)")
        self.connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?)",
            (("Aster", "hardware", 1), ("Birch", "software", 2), ("Cedar", "hardware", 3)),
        )
        self.connection.executemany(
            "INSERT INTO owners VALUES (?, ?)", ((1, "Mina"), (2, "Rui"), (3, "Tala"))
        )
        self.connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?)",
            (
                ("Aster", 2025, 80),
                ("Aster", 2026, 125),
                ("Birch", 2025, 110),
                ("Birch", 2026, 205),
                ("Cedar", 2025, 95),
                ("Cedar", 2026, 170),
            ),
        )
        self.startup_ns = time.perf_counter_ns() - started

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        payload = dict(request.payload)
        operation = request.operation
        queries = {
            "db.lookup": (
                "SELECT amount FROM sales WHERE product=? AND year=?",
                (payload.get("product"), int(payload.get("year", 0))),
                "sales",
            ),
            "db.max_sales": (
                (
                    "SELECT product, amount FROM sales WHERE year=? "
                    "ORDER BY amount DESC, product LIMIT 1"
                ),
                (int(payload.get("year", 0)),),
                "sales",
            ),
            "db.sum_sales": (
                "SELECT SUM(amount) FROM sales WHERE year=?",
                (int(payload.get("year", 0)),),
                "sales",
            ),
            "db.count_products": (
                "SELECT COUNT(*) FROM products WHERE category=?",
                (payload.get("category"),),
                "products",
            ),
            "db.avg_sales": (
                "SELECT AVG(amount) FROM sales WHERE year=?",
                (int(payload.get("year", 0)),),
                "sales",
            ),
            "db.owner_join": (
                "SELECT owners.owner FROM products JOIN owners USING(owner_id) WHERE product=?",
                (payload.get("product"),),
                "products,owners",
            ),
        }
        sql, parameters, resource = queries[operation]
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            return ()
        if operation == "db.max_sales":
            value: Any = str(row[0])
        elif operation == "db.avg_sales":
            value = format(float(row[0]), ".1f")
        else:
            value = str(row[0])
        record_id = f"duckdb:{operation}:{canonical_json(payload)}"
        content = f"{operation}({canonical_json(payload)}) = {value}"
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=record_id,
                value=value,
                content=content,
                started=started,
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query=sql,
                    resource=resource,
                    record_ids=[record_id],
                    parameters=payload,
                    query_language="sql",
                    parameterized=True,
                ),
            ),
        )


class SQLiteFTS5Source:
    policy = SourcePolicy(
        source_type="lexical",
        source_id="sqlite-fts5-policies-2026.08",
        locality="local",
        credential_scope="runtime:docs:read",
        latency_class="local_low",
        cost_class="cpu",
        privacy_class="internal",
        freshness="snapshot:2026-08-20",
    )

    def __init__(self) -> None:
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="sqlite_fts5",
            backend_version=sqlite3.sqlite_version,
            capabilities=("lexical_search", "document_filter", "rank"),
            request_fields={"lexical.search": ("query",)},
            resources=("documents_fts",),
            snapshot="policy-snapshot-2026-08-20",
        )
        started = time.perf_counter_ns()
        self.connection = sqlite3.connect(":memory:")
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE documents_fts USING fts5(identifier UNINDEXED, title, content)"
            )
        except sqlite3.OperationalError as exc:
            raise MissingDataBackendError("this Python SQLite build does not include FTS5") from exc
        self.connection.executemany(
            "INSERT INTO documents_fts(identifier, title, content) VALUES (?, ?, ?)",
            ((identifier, title, content) for identifier, (title, content) in _DOCUMENTS.items()),
        )
        self.startup_ns = time.perf_counter_ns() - started

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        document = str(request.payload.get("document", ""))
        tokens = _tokens(str(request.payload["query"]))
        if not tokens:
            return ()
        match_query = " OR ".join(f'"{token}"' for token in tokens)
        if document:
            sql = (
                "SELECT identifier, content, bm25(documents_fts) FROM documents_fts "
                "WHERE documents_fts MATCH ? AND identifier=? ORDER BY bm25(documents_fts) LIMIT 1"
            )
            parameters = (match_query, document)
        else:
            sql = (
                "SELECT identifier, content, bm25(documents_fts) FROM documents_fts "
                "WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT 1"
            )
            parameters = (match_query,)
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            return ()
        identifier, content, rank = str(row[0]), str(row[1]), float(row[2])
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=identifier,
                value=content,
                content=content,
                started=started,
                relevance=1.0 / (1.0 + abs(rank)),
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query=match_query,
                    resource="documents_fts",
                    record_ids=[identifier],
                    parameters={"document": document} if document else {},
                    query_language="fts5",
                    rank=rank,
                ),
            ),
        )


class FAISSVectorSource:
    policy = SourcePolicy(
        source_type="vector",
        source_id="faiss-flat-reports-2026q3",
        locality="local",
        credential_scope="runtime:vectors:read",
        latency_class="local_medium",
        cost_class="cpu",
        privacy_class="internal",
        freshness="snapshot:2026-Q3",
    )

    def __init__(self) -> None:
        faiss = _faiss_module()
        self.faiss = faiss
        self.identifiers = sorted(_REPORTS)
        report_vectors = [_semantic_vector(_REPORTS[identifier]) for identifier in self.identifiers]
        self.vocabulary = sorted({term for vector in report_vectors for term in vector})
        self.term_index = {term: index for index, term in enumerate(self.vocabulary)}
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="faiss",
            backend_version=getattr(faiss, "__version__", "unknown"),
            capabilities=("exact_vector_search", "cosine_similarity"),
            request_fields={"vector.search": ("query",)},
            resources=("reports_flat_ip",),
            snapshot="reports-snapshot-2026-Q3",
        )
        started = time.perf_counter_ns()
        matrix = np.vstack([self._embed(vector) for vector in report_vectors]).astype("float32")
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(len(self.vocabulary))
        self.index.add(matrix)
        self.startup_ns = time.perf_counter_ns() - started

    def _embed(self, vector: Any) -> np.ndarray:
        values = np.zeros(len(self.vocabulary), dtype="float32")
        for term, count in vector.items():
            index = self.term_index.get(term)
            if index is not None:
                values[index] = float(count)
        return values

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        embedding_started = time.perf_counter_ns()
        query = self._embed(_semantic_vector(str(request.payload["query"]))).reshape(1, -1)
        if not np.any(query):
            return ()
        self.faiss.normalize_L2(query)
        embedding_ns = time.perf_counter_ns() - embedding_started
        retrieval_started = time.perf_counter_ns()
        scores, indices = self.index.search(query, 1)
        retrieval_ns = time.perf_counter_ns() - retrieval_started
        index = int(indices[0, 0])
        score = float(scores[0, 0])
        if index < 0 or score <= 0:
            return ()
        identifier = self.identifiers[index]
        content = _REPORTS[identifier]
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=identifier,
                value=content,
                content=content,
                started=started,
                relevance=score,
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query="L2_NORMALIZE(concept_bow(query)); IndexFlatIP(k=1)",
                    resource="reports_flat_ip",
                    record_ids=[identifier],
                    parameters={"collection": request.payload.get("collection", "reports")},
                    index_type="IndexFlatIP",
                    embedding="controlled-concept-bow-v1",
                    embedding_latency_ns=embedding_ns,
                    retrieval_latency_ns=retrieval_ns,
                    vector_id=index,
                ),
            ),
        )


def build_production_sources() -> dict[str, Any]:
    """Build local production backends while retaining the controlled web analogue."""

    return {
        "db": DuckDBSource(),
        "lexical": SQLiteFTS5Source(),
        "vector": FAISSVectorSource(),
        "web": ControlledWebSource(),
    }
