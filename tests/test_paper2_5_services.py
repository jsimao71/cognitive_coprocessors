import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccpu.common.retrieval import SourceRequest
from ccpu.paper2_5.service_sources import (
    PgvectorSource,
    PostgresSource,
    QdrantVectorSource,
    RestIcebergSource,
    controlled_query_embedding,
    postgres_query,
)

POSTGRES_DSN = os.environ.get("CCPU_POSTGRES_DSN", "")
QDRANT_URL = os.environ.get("CCPU_QDRANT_URL", "")


def test_postgres_compiler_uses_fixed_parameterized_queries():
    sql, parameters, resource = postgres_query(
        "db.lookup", {"product": "Aster'; DROP TABLE sales; --", "year": 2026}
    )
    assert "%s" in sql
    assert "DROP TABLE" not in sql
    assert "DROP TABLE" in parameters[0]
    assert resource == "sales"
    schema_sql, schema_parameters, schema_resource = postgres_query("db.schema", {})
    assert "information_schema.tables" in schema_sql
    assert schema_parameters == ()
    assert schema_resource == "information_schema.tables"
    with pytest.raises(ValueError, match="unsupported"):
        postgres_query("db.raw_sql", {"sql": "SELECT 1"})


def test_pgvector_fixture_embedding_keeps_embedding_separate_from_search():
    assert controlled_query_embedding("profitability shipping discounts").tolist() == [3, 0, 0]
    assert controlled_query_embedding("why customer churn improved").tolist() == [0, 1, 0]
    assert controlled_query_embedding("theme behind lower energy use").tolist() == [0, 0, 1]


def test_wsl_sidekick_is_explicit_and_contains_no_real_environment_file():
    root = Path(__file__).resolve().parents[1] / "sidekick" / "data_stack"
    required = (
        "compose.yaml",
        ".env.example",
        "README.md",
        "healthcheck.sh",
        "init/001_schema.sql",
        "init/002_read_only.sh",
    )
    assert all((root / relative).is_file() for relative in required)
    assert not (root / ".env").exists()
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    assert "pgvector/pgvector:0.8.6-pg17-bookworm" in compose
    assert "healthcheck" in compose
    assert "qdrant/qdrant:v1.19.0" in compose


def test_qdrant_adapter_is_read_only_and_keeps_credentials_runtime_owned():
    class FakeQdrant:
        def query_points(self, **kwargs):
            assert kwargs["collection_name"] == "reports"
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id=7,
                        score=0.99,
                        payload={"identifier": "report_margin_q2", "content": "freight expenses"},
                    )
                ]
            )

    source = QdrantVectorSource(
        "http://localhost:6333",
        api_key="not-recorded",
        client=FakeQdrant(),
        server_version="test",
    )
    evidence = source.retrieve(
        SourceRequest(
            request_id="qdrant",
            source_type="vector",
            operation="vector.search",
            payload={"query": "profitability shipping discounts"},
        )
    )
    assert evidence[0].record_id == "report_margin_q2"
    assert evidence[0].provenance["backend"] == "qdrant"
    assert "not-recorded" not in str(source.descriptor.public_dict())


def test_rest_iceberg_adapter_allowlists_tables_and_retains_snapshot():
    class FakeScan:
        def to_arrow(self):
            import pyarrow as pa

            return pa.table({"revenue": [125.0, 205.0, 170.0]})

    class FakeTable:
        def scan(self, **kwargs):
            assert kwargs["selected_fields"] == ("revenue",)
            return FakeScan()

        def current_snapshot(self):
            return SimpleNamespace(snapshot_id=42)

    class FakeCatalog:
        def load_table(self, name):
            assert name == "enterprise.sales"
            return FakeTable()

    source = RestIcebergSource(
        "https://catalog.invalid",
        "warehouse",
        token="not-recorded",
        catalog=FakeCatalog(),
    )
    evidence = source.retrieve(
        SourceRequest(
            request_id="rest",
            source_type="iceberg",
            operation="iceberg_rest.sum_revenue",
            payload={"table": "sales", "year": 2026},
        )
    )
    assert evidence[0].value == "500.0"
    assert evidence[0].provenance["iceberg_snapshot_id"] == 42
    assert "not-recorded" not in str(source.descriptor.public_dict())


@pytest.mark.integration
@pytest.mark.skipif(not POSTGRES_DSN, reason="CCPU_POSTGRES_DSN is not configured")
def test_postgres_and_pgvector_sidekick_integration():
    db = PostgresSource(POSTGRES_DSN)
    db_evidence = db.retrieve(
        SourceRequest(
            request_id="integration-db",
            source_type="db",
            operation="db.sum_sales",
            payload={"year": 2026},
        )
    )
    assert db_evidence[0].value == "500"
    assert db_evidence[0].provenance["transaction_mode"] == "read_only"

    vector = PgvectorSource(POSTGRES_DSN)
    vector_evidence = vector.retrieve(
        SourceRequest(
            request_id="integration-vector",
            source_type="vector",
            operation="vector.search",
            payload={"query": "profitability shipping discounts"},
        )
    )
    assert vector_evidence[0].record_id == "report_margin_q2"
    assert vector_evidence[0].provenance["backend"] == "pgvector"


@pytest.mark.integration
@pytest.mark.skipif(not QDRANT_URL, reason="CCPU_QDRANT_URL is not configured")
def test_qdrant_service_is_explicitly_reachable():
    from qdrant_client import QdrantClient

    assert QdrantClient(url=QDRANT_URL).get_collections() is not None
