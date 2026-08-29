"""Explicit service-backed Postgres and pgvector data coprocessors."""

from __future__ import annotations

import time
from importlib.metadata import version as package_version
from typing import Any, ClassVar

import numpy as np

from ccpu.common.artifacts import canonical_json
from ccpu.common.data_coprocessor import DataCoprocessorDescriptor, production_provenance
from ccpu.common.retrieval import SourcePolicy, SourceRequest

from .production_sources import MissingDataBackendError
from .sources import _record, _semantic_vector


def _service_modules() -> tuple[Any, Any]:
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise MissingDataBackendError(
            "install the 'data-services' extra to use Postgres and pgvector"
        ) from exc
    return psycopg, register_vector


def postgres_query(operation: str, payload: dict[str, Any]) -> tuple[str, tuple[Any, ...], str]:
    """Compile typed relational IR to one fixed, parameterized Postgres query."""

    queries = {
        "db.lookup": (
            "SELECT amount FROM sales WHERE product=%s AND year=%s",
            (payload.get("product"), int(payload.get("year", 0))),
            "sales",
        ),
        "db.max_sales": (
            (
                "SELECT product, amount FROM sales WHERE year=%s "
                "ORDER BY amount DESC, product LIMIT 1"
            ),
            (int(payload.get("year", 0)),),
            "sales",
        ),
        "db.sum_sales": (
            "SELECT SUM(amount) FROM sales WHERE year=%s",
            (int(payload.get("year", 0)),),
            "sales",
        ),
        "db.count_products": (
            "SELECT COUNT(*) FROM products WHERE category=%s",
            (payload.get("category"),),
            "products",
        ),
        "db.avg_sales": (
            "SELECT AVG(amount) FROM sales WHERE year=%s",
            (int(payload.get("year", 0)),),
            "sales",
        ),
        "db.owner_join": (
            "SELECT owners.owner FROM products JOIN owners USING(owner_id) WHERE product=%s",
            (payload.get("product"),),
            "products,owners",
        ),
        "db.schema": (
            (
                "SELECT string_agg(table_name, ',' ORDER BY table_name) "
                "FROM information_schema.tables WHERE table_schema='public' "
                "AND table_name IN ('owners','products','reports','sales')"
            ),
            (),
            "information_schema.tables",
        ),
    }
    try:
        return queries[operation]
    except KeyError as exc:
        raise ValueError(f"unsupported Postgres operation: {operation}") from exc


class PostgresSource:
    policy = SourcePolicy(
        source_type="db",
        source_id="postgres-sales-sidekick-v1",
        locality="wsl_service",
        credential_scope="runtime:postgres:read",
        latency_class="local_service",
        cost_class="cpu_network",
        privacy_class="internal",
        freshness="snapshot:sidekick-seed-v1",
    )

    def __init__(self, dsn: str) -> None:
        if not dsn.strip():
            raise ValueError("PostgresSource requires an explicit DSN")
        self._dsn = dsn
        self.psycopg, _ = _service_modules()
        started = time.perf_counter_ns()
        with self.psycopg.connect(self._dsn, autocommit=True) as connection:
            server_version = str(connection.execute("SHOW server_version").fetchone()[0])
        self.startup_ns = time.perf_counter_ns() - started
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="postgresql",
            backend_version=f"server:{server_version};client:{self.psycopg.__version__}",
            capabilities=("lookup", "aggregate", "argmax", "join", "schema_introspection"),
            request_fields={
                "db.lookup": ("product", "year"),
                "db.max_sales": ("year",),
                "db.sum_sales": ("year",),
                "db.count_products": ("category",),
                "db.avg_sales": ("year",),
                "db.owner_join": ("product",),
                "db.schema": (),
            },
            resources=("products", "sales", "owners"),
            snapshot="sidekick-seed-v1",
        )

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        payload = dict(request.payload)
        sql, parameters, resource = postgres_query(request.operation, payload)
        with self.psycopg.connect(self._dsn) as connection, connection.transaction():
            connection.execute("SET TRANSACTION READ ONLY")
            row = connection.execute(sql, parameters).fetchone()
        if row is None:
            return ()
        if request.operation == "db.max_sales":
            value: Any = str(row[0])
        elif request.operation == "db.avg_sales":
            value = format(float(row[0]), ".1f")
        else:
            value = str(row[0])
        record_id = f"postgres:{request.operation}:{canonical_json(payload)}"
        content = f"{request.operation}({canonical_json(payload)}) = {value}"
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
                    transaction_mode="read_only",
                ),
            ),
        )


def controlled_query_embedding(query: str) -> np.ndarray:
    """Map the frozen semantic concepts to the sidekick's three-vector fixture."""

    concepts = _semantic_vector(query)
    return np.asarray(
        [
            sum(concepts[term] for term in ("margin", "shipping", "discount")),
            sum(concepts[term] for term in ("attrition", "onboarding")),
            sum(concepts[term] for term in ("energy", "cooling")),
        ],
        dtype="float32",
    )


class PgvectorSource:
    policy = SourcePolicy(
        source_type="vector",
        source_id="pgvector-reports-sidekick-v1",
        locality="wsl_service",
        credential_scope="runtime:pgvector:read",
        latency_class="local_service",
        cost_class="cpu_network",
        privacy_class="internal",
        freshness="snapshot:sidekick-seed-v1",
    )

    def __init__(self, dsn: str) -> None:
        if not dsn.strip():
            raise ValueError("PgvectorSource requires an explicit DSN")
        self._dsn = dsn
        self.psycopg, self.register_vector = _service_modules()
        started = time.perf_counter_ns()
        with self.psycopg.connect(self._dsn, autocommit=True) as connection:
            server_version = str(connection.execute("SHOW server_version").fetchone()[0])
            extension_version = str(
                connection.execute(
                    "SELECT extversion FROM pg_extension WHERE extname='vector'"
                ).fetchone()[0]
            )
        self.startup_ns = time.perf_counter_ns() - started
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="pgvector",
            backend_version=f"postgres:{server_version};vector:{extension_version}",
            capabilities=("exact_vector_search", "metadata_filter", "persistent_index"),
            request_fields={"vector.search": ("query",)},
            resources=("reports",),
            snapshot="sidekick-seed-v1",
        )

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        embedding_started = time.perf_counter_ns()
        embedding = controlled_query_embedding(str(request.payload["query"]))
        embedding_ns = time.perf_counter_ns() - embedding_started
        if not np.any(embedding):
            return ()
        sql = (
            "SELECT identifier, content, 1 - (embedding <=> %s) AS score "
            "FROM reports WHERE collection=%s ORDER BY embedding <=> %s LIMIT 1"
        )
        retrieval_started = time.perf_counter_ns()
        with self.psycopg.connect(self._dsn) as connection:
            self.register_vector(connection)
            with connection.transaction():
                connection.execute("SET TRANSACTION READ ONLY")
                row = connection.execute(sql, (embedding, "reports", embedding)).fetchone()
        retrieval_ns = time.perf_counter_ns() - retrieval_started
        if row is None:
            return ()
        identifier, content, score = str(row[0]), str(row[1]), float(row[2])
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
                    normalized_query=sql,
                    resource="reports",
                    record_ids=[identifier],
                    parameters={"collection": "reports", "limit": 1},
                    query_language="sql+vector_cosine",
                    index_type="exact",
                    embedding="controlled-concept-3d-v1",
                    embedding_latency_ns=embedding_ns,
                    retrieval_latency_ns=retrieval_ns,
                    transaction_mode="read_only",
                ),
            ),
        )


class QdrantVectorSource:
    policy = SourcePolicy(
        source_type="vector",
        source_id="qdrant-reports-service-v1",
        locality="wsl_service",
        credential_scope="runtime:qdrant:read",
        latency_class="local_service",
        cost_class="cpu_network",
        privacy_class="internal",
        freshness="collection_snapshot",
    )

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        server_version: str | None = None,
    ) -> None:
        if not url.strip():
            raise ValueError("QdrantVectorSource requires an explicit URL")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise MissingDataBackendError("install the 'data-services' extra") from exc
        self.client = client or QdrantClient(url=url, api_key=api_key)
        version = server_version or str(self.client.info().version)
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="qdrant",
            backend_version=f"server:{version};client:{package_version('qdrant-client')}",
            capabilities=("vector_search", "metadata_filter", "persistent_collection"),
            request_fields={"vector.search": ("query",)},
            resources=("reports",),
            snapshot="qdrant-reports-v1",
        )

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        embedding_started = time.perf_counter_ns()
        embedding = controlled_query_embedding(str(request.payload["query"]))
        embedding_ns = time.perf_counter_ns() - embedding_started
        if not np.any(embedding):
            return ()
        response = self.client.query_points(
            collection_name="reports",
            query=embedding.tolist(),
            limit=1,
            with_payload=True,
        )
        if not response.points:
            return ()
        point = response.points[0]
        payload = dict(point.payload or {})
        identifier = str(payload.get("identifier", point.id))
        content = str(payload.get("content", ""))
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=identifier,
                value=content,
                content=content,
                started=started,
                relevance=float(point.score),
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query="query_points(collection=reports,limit=1)",
                    resource="reports",
                    record_ids=[identifier],
                    parameters={"collection": "reports", "limit": 1},
                    query_language="qdrant_query_points",
                    embedding="controlled-concept-3d-v1",
                    embedding_latency_ns=embedding_ns,
                    vector_id=str(point.id),
                ),
            ),
        )


class RestIcebergSource:
    policy = SourcePolicy(
        source_type="iceberg",
        source_id="pyiceberg-rest-enterprise-v1",
        locality="remote_catalog",
        credential_scope="runtime:iceberg-rest:read",
        latency_class="network_storage",
        cost_class="network_storage",
        privacy_class="internal",
        freshness="catalog_snapshot",
    )

    _tables: ClassVar[dict[str, str]] = {"sales": "enterprise.sales"}

    def __init__(
        self,
        catalog_uri: str,
        warehouse: str,
        *,
        token: str | None = None,
        catalog: Any | None = None,
    ) -> None:
        if not catalog_uri.strip() or not warehouse.strip():
            raise ValueError("RestIcebergSource requires explicit catalog URI and warehouse")
        try:
            import pyiceberg
            from pyiceberg.catalog import load_catalog
        except ImportError as exc:  # pragma: no cover
            raise MissingDataBackendError("install the 'data-governance' extra") from exc
        self.catalog = catalog or load_catalog(
            "ccpu_rest",
            type="rest",
            uri=catalog_uri,
            warehouse=warehouse,
            token=token,
        )
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="pyiceberg_rest",
            backend_version=pyiceberg.__version__,
            capabilities=("catalog_load", "snapshot_read", "predicate_pushdown"),
            request_fields={"iceberg_rest.sum_revenue": ("table", "year")},
            resources=tuple(self._tables.values()),
            snapshot="catalog_selected",
        )

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        table_name = self._tables.get(str(request.payload["table"]))
        if table_name is None:
            return ()
        from pyiceberg.expressions import EqualTo

        table = self.catalog.load_table(table_name)
        arrow = table.scan(
            row_filter=EqualTo("year", int(request.payload["year"])),
            selected_fields=("revenue",),
        ).to_arrow()
        value = format(sum(arrow.column("revenue").to_pylist()), ".1f")
        snapshot_id = int(table.current_snapshot().snapshot_id)
        record_id = f"iceberg-rest:{table_name}:{snapshot_id}"
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=record_id,
                value=value,
                content=f"iceberg_rest.sum_revenue = {value}",
                started=started,
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query="scan enterprise.sales where year=? select revenue; sum",
                    resource=table_name,
                    record_ids=[record_id],
                    parameters={"year": int(request.payload["year"])},
                    query_language="pyiceberg_scan",
                    iceberg_snapshot_id=snapshot_id,
                ),
            ),
        )
