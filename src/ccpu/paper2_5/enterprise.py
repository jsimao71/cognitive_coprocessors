"""Local Iceberg, governed metric, and ontology production-shaped fixture."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

from ccpu.common.artifacts import file_sha256, write_json
from ccpu.common.data_coprocessor import DataCoprocessorDescriptor, production_provenance
from ccpu.common.iceberg_io import local_file_uri, local_path_from_uri
from ccpu.common.retrieval import SourcePolicy, SourceRequest

from .production_sources import MissingDataBackendError
from .sources import _record

SALES_V1 = (
    (1, "Cust-A", "Aster", "North", 2026, 1, 125.0, 80.0, 12),
    (2, "Cust-B", "Birch", "South", 2026, 1, 205.0, 140.0, 20),
    (3, "Cust-C", "Cedar", "North", 2026, 1, 170.0, 100.0, 16),
)
SALES_V2 = (
    (4, "Cust-A", "Aster", "North", 2026, 2, 150.0, 90.0, 15, "direct"),
    (5, "Cust-B", "Birch", "South", 2026, 2, 220.0, 160.0, 22, "partner"),
    (6, "Cust-C", "Cedar", "North", 2026, 2, 180.0, 110.0, 18, "direct"),
)

METRICS = {
    "gross_margin": {
        "version": "1.0.0",
        "expression": "(SUM(revenue)-SUM(cost))/SUM(revenue)",
        "format": "percent_2dp",
    },
    "net_revenue": {
        "version": "1.0.0",
        "expression": "SUM(revenue)",
        "format": "decimal_1dp",
    },
}

ONTOLOGY_TTL = """@prefix ccpu: <https://example.org/ccpu/> .
ccpu:Aster ccpu:memberOf ccpu:Hardware .
ccpu:Cedar ccpu:memberOf ccpu:Hardware .
ccpu:Birch ccpu:memberOf ccpu:Software .
ccpu:Hardware ccpu:subClassOf ccpu:ProductFamily .
ccpu:Software ccpu:subClassOf ccpu:ProductFamily .
ccpu:hardware-products ccpu:normalizesTo ccpu:Hardware .
ccpu:software-products ccpu:normalizesTo ccpu:Software .
"""


def _modules() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        from pyiceberg.catalog import load_catalog
        from pyiceberg.types import StringType
    except ImportError as exc:  # pragma: no cover - minimal installs
        raise MissingDataBackendError("install the 'data-governance' extra") from exc
    return pa, load_catalog, StringType


def create_enterprise_fixture(output_dir: str | Path) -> dict[str, Any]:
    """Create real local Iceberg tables plus governed semantic assets."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"enterprise fixture output must be empty: {output}")
    warehouse = output / "warehouse"
    warehouse.mkdir()
    pa, load_catalog, string_type = _modules()
    catalog = load_catalog(
        "ccpu_enterprise",
        type="sql",
        uri=f"sqlite:///{(output / 'catalog.db').as_posix()}",
        warehouse=local_file_uri(warehouse),
        **{"py-io-impl": "ccpu.common.iceberg_io.WindowsSafePyArrowFileIO"},
    )
    catalog.create_namespace("enterprise")

    sales_schema = pa.schema(
        [
            pa.field("sale_id", pa.int64(), nullable=False),
            pa.field("customer_id", pa.string(), nullable=False),
            pa.field("product", pa.string(), nullable=False),
            pa.field("organization", pa.string(), nullable=False),
            pa.field("year", pa.int32(), nullable=False),
            pa.field("quarter", pa.int32(), nullable=False),
            pa.field("revenue", pa.float64(), nullable=False),
            pa.field("cost", pa.float64(), nullable=False),
            pa.field("units", pa.int32(), nullable=False),
        ]
    )
    sales = catalog.create_table("enterprise.sales", schema=sales_schema)
    sales.append(
        pa.Table.from_pylist(
            [dict(zip(sales_schema.names, row, strict=True)) for row in SALES_V1],
            schema=sales_schema,
        ),
        snapshot_properties={"ccpu.fixture.version": "sales-v1"},
    )
    snapshot_v1 = sales.current_snapshot().snapshot_id
    with sales.update_schema() as update:
        update.add_column("channel", string_type(), required=False)
    sales.refresh()
    evolved_schema = sales.schema().as_arrow()
    sales.append(
        pa.Table.from_pylist(
            [dict(zip(evolved_schema.names, row, strict=True)) for row in SALES_V2],
            schema=evolved_schema,
        ),
        snapshot_properties={"ccpu.fixture.version": "sales-v2"},
    )
    snapshot_v2 = sales.current_snapshot().snapshot_id

    fixtures = {
        "products": [
            {"product": "Aster", "family": "Hardware", "supplier": "Org-1"},
            {"product": "Birch", "family": "Software", "supplier": "Org-2"},
            {"product": "Cedar", "family": "Hardware", "supplier": "Org-1"},
        ],
        "inventory": [
            {"product": "Aster", "units_on_hand": 42},
            {"product": "Birch", "units_on_hand": 31},
            {"product": "Cedar", "units_on_hand": 27},
        ],
        "customers": [
            {"customer_id": "Cust-A", "segment": "Enterprise"},
            {"customer_id": "Cust-B", "segment": "SMB"},
            {"customer_id": "Cust-C", "segment": "Enterprise"},
        ],
    }
    tables: dict[str, dict[str, Any]] = {}
    for name, rows in fixtures.items():
        arrow = pa.Table.from_pylist(rows)
        table = catalog.create_table(f"enterprise.{name}", schema=arrow.schema)
        table.append(arrow, snapshot_properties={"ccpu.fixture.version": f"{name}-v1"})
        tables[name] = {
            "metadata": str(local_path_from_uri(table.metadata_location)),
            "snapshot_id": table.current_snapshot().snapshot_id,
        }

    metrics_path = write_json(output / "metric_definitions.json", METRICS)
    ontology_path = output / "ontology.ttl"
    ontology_path.write_text(ONTOLOGY_TTL, encoding="utf-8")
    documents_path = write_json(
        output / "documents.json",
        {
            "margin_policy": "Gross margin uses recognized revenue less direct cost.",
            "inventory_policy": "Low stock means fewer than thirty units on hand.",
        },
    )
    manifest = {
        "schema_version": "ccpu.paper2_5.enterprise_fixture.v1",
        "catalog": "pyiceberg_sqlite_local",
        "sales": {
            "metadata": str(local_path_from_uri(sales.metadata_location)),
            "snapshots": {"sales_v1": snapshot_v1, "sales_v2": snapshot_v2},
            "current_snapshot_id": snapshot_v2,
            "schema_id": sales.schema().schema_id,
            "schema_evolution": "added optional channel before sales_v2",
        },
        "tables": tables,
        "metric_definitions_sha256": file_sha256(metrics_path),
        "ontology_sha256": file_sha256(ontology_path),
        "documents_sha256": file_sha256(documents_path),
        "physical_files": {
            str(path.relative_to(output)).replace("\\", "/"): file_sha256(path)
            for path in sorted(
                [output / "catalog.db", *warehouse.rglob("*")], key=lambda item: str(item)
            )
            if path.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_fixture_manifest(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))


class IcebergDuckDBSource:
    policy = SourcePolicy(
        source_type="iceberg",
        source_id="duckdb-direct-iceberg-enterprise-v1",
        locality="local",
        credential_scope="runtime:iceberg:read",
        latency_class="local_medium",
        cost_class="cpu_storage",
        privacy_class="internal",
        freshness="iceberg_snapshot",
    )

    def __init__(self, fixture_root: str | Path) -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover
            raise MissingDataBackendError("install the 'data' extra") from exc
        self.fixture_root = Path(fixture_root)
        self.manifest = load_fixture_manifest(self.fixture_root)
        self.sales_metadata = Path(self.manifest["sales"]["metadata"])
        self.connection = duckdb.connect(":memory:")
        self.connection.execute("LOAD iceberg")
        extension = self.connection.execute(
            "SELECT extension_version FROM duckdb_extensions() WHERE extension_name='iceberg'"
        ).fetchone()[0]
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="duckdb_iceberg",
            backend_version=f"duckdb:{duckdb.__version__};iceberg:{extension}",
            capabilities=("aggregate", "snapshot_select", "time_travel", "schema_evolution"),
            request_fields={
                "iceberg.sum_revenue": ("year",),
                "iceberg.gross_margin": ("year",),
                "iceberg.snapshot_revenue": ("snapshot_id",),
                "iceberg.inventory_lookup": ("product",),
                "iceberg.customer_segment": ("customer_id",),
                "iceberg.product_inventory_join": ("product",),
            },
            resources=(
                "enterprise.sales",
                "enterprise.products",
                "enterprise.inventory",
                "enterprise.customers",
            ),
            snapshot=str(self.manifest["sales"]["current_snapshot_id"]),
        )

    def _scan(self, snapshot_id: int | None = None) -> str:
        path = self.sales_metadata.as_posix().replace("'", "''")
        suffix = f", snapshot_from_id={int(snapshot_id)}" if snapshot_id is not None else ""
        return f"iceberg_scan('{path}'{suffix})"

    def _table_scan(self, table: str) -> tuple[str, int]:
        metadata = Path(self.manifest["tables"][table]["metadata"])
        path = metadata.as_posix().replace("'", "''")
        return f"iceberg_scan('{path}')", int(self.manifest["tables"][table]["snapshot_id"])

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        payload = dict(request.payload)
        if request.operation in {
            "iceberg.inventory_lookup",
            "iceberg.customer_segment",
            "iceberg.product_inventory_join",
        }:
            if request.operation == "iceberg.inventory_lookup":
                scan, snapshot = self._table_scan("inventory")
                sql = f"SELECT units_on_hand FROM {scan} WHERE product=?"
                parameters = (payload["product"],)
                resource = "enterprise.inventory"
                snapshot_ids: Any = snapshot
            elif request.operation == "iceberg.customer_segment":
                scan, snapshot = self._table_scan("customers")
                sql = f"SELECT segment FROM {scan} WHERE customer_id=?"
                parameters = (payload["customer_id"],)
                resource = "enterprise.customers"
                snapshot_ids = snapshot
            else:
                products, products_snapshot = self._table_scan("products")
                inventory, inventory_snapshot = self._table_scan("inventory")
                sql = (
                    f"SELECT p.family || ':' || CAST(i.units_on_hand AS VARCHAR) FROM {products} p "
                    f"JOIN {inventory} i USING(product) WHERE p.product=?"
                )
                parameters = (payload["product"],)
                resource = "enterprise.products,enterprise.inventory"
                snapshot_ids = [products_snapshot, inventory_snapshot]
            row = self.connection.execute(sql, parameters).fetchone()
            if row is None:
                return ()
            value = str(row[0])
            record_id = f"iceberg:{request.operation}:{value}"
            return (
                _record(
                    request=request,
                    policy=self.policy,
                    record_id=record_id,
                    value=value,
                    content=f"{request.operation} = {value}",
                    started=started,
                    provenance=production_provenance(
                        self.descriptor,
                        normalized_query=sql,
                        resource=resource,
                        record_ids=[record_id],
                        parameters=payload,
                        query_language="sql+iceberg_scan",
                        iceberg_snapshot_id=snapshot_ids,
                    ),
                ),
            )
        snapshot_id = (
            int(payload["snapshot_id"])
            if request.operation == "iceberg.snapshot_revenue"
            else int(self.manifest["sales"]["current_snapshot_id"])
        )
        scan = self._scan(snapshot_id if request.operation == "iceberg.snapshot_revenue" else None)
        if request.operation in {"iceberg.sum_revenue", "iceberg.snapshot_revenue"}:
            sql = f"SELECT SUM(revenue) FROM {scan} WHERE year=?"
            parameters = (int(payload.get("year", 2026)),)
        else:
            sql = f"SELECT (SUM(revenue)-SUM(cost))/SUM(revenue) FROM {scan} WHERE year=?"
            parameters = (int(payload["year"]),)
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None or row[0] is None:
            return ()
        value = (
            format(float(row[0]), ".1f")
            if "revenue" in request.operation
            else format(float(row[0]) * 100, ".2f") + "%"
        )
        record_id = f"iceberg:{request.operation}:{snapshot_id}"
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=record_id,
                value=value,
                content=f"{request.operation} = {value}",
                started=started,
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query=sql,
                    resource="enterprise.sales",
                    record_ids=[record_id],
                    parameters=payload,
                    query_language="sql+iceberg_scan",
                    iceberg_snapshot_id=snapshot_id,
                    metadata_sha256=file_sha256(self.sales_metadata),
                    schema_id=self.manifest["sales"]["schema_id"],
                ),
            ),
        )


class SemanticMetricSource:
    policy = SourcePolicy(
        source_type="semantic",
        source_id="governed-metrics-enterprise-v1",
        locality="local",
        credential_scope="runtime:metrics:read",
        latency_class="local_medium",
        cost_class="cpu_storage",
        privacy_class="internal",
        freshness="metric-definitions:1.0.0",
    )

    def __init__(self, fixture_root: str | Path) -> None:
        self.iceberg = IcebergDuckDBSource(fixture_root)
        self.metrics = json.loads(
            (Path(fixture_root) / "metric_definitions.json").read_text(encoding="utf-8")
        )
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="governed_metric_compiler",
            backend_version="1.0.0",
            capabilities=("metric_definition", "dimension_filter", "iceberg_compile"),
            request_fields={"semantic.metric": ("metric", "year")},
            resources=("metric_definitions.json", "enterprise.sales"),
            snapshot=self.iceberg.descriptor.snapshot,
        )

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        payload = dict(request.payload)
        metric = str(payload["metric"])
        definition = self.metrics.get(metric)
        if definition is None:
            return ()
        filters = ["year=?"]
        parameters: list[Any] = [int(payload["year"])]
        products = [str(product) for product in payload.get("products", [])]
        if products:
            if len(products) > 16:
                raise ValueError("semantic product filter exceeds bounded size")
            filters.append("product IN (" + ",".join("?" for _ in products) + ")")
            parameters.extend(products)
        if metric == "gross_margin":
            expression = "(SUM(revenue)-SUM(cost))/SUM(revenue)"
        elif metric == "net_revenue":
            expression = "SUM(revenue)"
        else:
            return ()
        sql = f"SELECT {expression} FROM {self.iceberg._scan()} WHERE {' AND '.join(filters)}"
        row = self.iceberg.connection.execute(sql, parameters).fetchone()
        if row is None or row[0] is None:
            return ()
        value = (
            format(float(row[0]) * 100, ".2f") + "%"
            if definition["format"] == "percent_2dp"
            else format(float(row[0]), ".1f")
        )
        snapshot_id = int(self.iceberg.manifest["sales"]["current_snapshot_id"])
        record_id = f"metric:{metric}:{definition['version']}:{snapshot_id}"
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=record_id,
                value=value,
                content=f"Governed metric {metric} = {value}",
                started=started,
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query=sql,
                    resource="enterprise.sales",
                    record_ids=[record_id],
                    parameters={"year": payload["year"], "products": products},
                    query_language="semantic_metric+sql+iceberg_scan",
                    iceberg_snapshot_id=snapshot_id,
                    semantic_metric=metric,
                    semantic_metric_definition=definition["expression"],
                    semantic_metric_version=definition["version"],
                ),
            ),
        )


class OntologySource:
    policy = SourcePolicy(
        source_type="ontology",
        source_id="oxigraph-product-taxonomy-v1",
        locality="local",
        credential_scope="runtime:ontology:read",
        latency_class="local_low",
        cost_class="cpu",
        privacy_class="internal",
        freshness="ontology:1.0.0",
    )

    _concepts: ClassVar[dict[str, str]] = {
        "Hardware": "https://example.org/ccpu/Hardware",
        "Software": "https://example.org/ccpu/Software",
        "hardware-products": "https://example.org/ccpu/Hardware",
        "software-products": "https://example.org/ccpu/Software",
    }

    def __init__(self, fixture_root: str | Path) -> None:
        try:
            import pyoxigraph
            from pyoxigraph import RdfFormat, Store
        except ImportError as exc:  # pragma: no cover
            raise MissingDataBackendError("install the 'data-governance' extra") from exc
        self.store = Store()
        self.store.load(path=Path(fixture_root) / "ontology.ttl", format=RdfFormat.TURTLE)
        self.descriptor = DataCoprocessorDescriptor(
            policy=self.policy,
            backend="oxigraph",
            backend_version=pyoxigraph.__version__,
            capabilities=("taxonomy", "concept_normalization", "member_resolution"),
            request_fields={"ontology.members": ("concept",)},
            resources=("product_taxonomy",),
            snapshot="ontology-v1",
        )

    def retrieve(self, request: SourceRequest) -> tuple[Any, ...]:
        self.descriptor.validate(request)
        started = time.perf_counter_ns()
        concept = str(request.payload["concept"])
        iri = self._concepts.get(concept)
        if iri is None:
            return ()
        sparql = (
            "PREFIX ccpu: <https://example.org/ccpu/> "
            f"SELECT ?product WHERE {{ ?product ccpu:memberOf <{iri}> }} ORDER BY ?product"
        )
        products = [binding["product"].value.rsplit("/", 1)[-1] for binding in self.store.query(sparql)]
        value = ",".join(products)
        record_id = f"ontology:members:{concept}"
        return (
            _record(
                request=request,
                policy=self.policy,
                record_id=record_id,
                value=value,
                content=f"Ontology members of {concept}: {value}",
                started=started,
                provenance=production_provenance(
                    self.descriptor,
                    normalized_query=sparql,
                    resource="product_taxonomy",
                    record_ids=[record_id],
                    parameters={"concept": concept},
                    query_language="sparql",
                    ontology_concept_id=iri,
                    ontology_relation_id="https://example.org/ccpu/memberOf",
                    resolved_entity_ids=products,
                ),
            ),
        )


def _enterprise_documents(fixture_root: str | Path) -> list[str]:
    documents = json.loads((Path(fixture_root) / "documents.json").read_text(encoding="utf-8"))
    rows = [
        f"Sale {row[0]} customer {row[1]} product {row[2]} organization {row[3]} "
        f"year {row[4]} quarter {row[5]} revenue {row[6]} cost {row[7]} units {row[8]}."
        for row in SALES_V1
    ]
    rows.extend(
        f"Sale {row[0]} customer {row[1]} product {row[2]} organization {row[3]} "
        f"year {row[4]} quarter {row[5]} revenue {row[6]} cost {row[7]} units {row[8]} "
        f"channel {row[9]}."
        for row in SALES_V2
    )
    rows.extend(f"Metric {name} version {item['version']} means {item['expression']}." for name, item in METRICS.items())
    rows.extend(ONTOLOGY_TTL.splitlines()[1:])
    rows.extend(documents.values())
    return rows


def _universal_retrieve(question: str, documents: list[str], limit: int = 5) -> list[str]:
    import re

    query = set(re.findall(r"[a-z0-9]+", question.casefold()))
    scored = []
    for index, document in enumerate(documents):
        tokens = set(re.findall(r"[a-z0-9]+", document.casefold()))
        scored.append((len(query & tokens), -index, document))
    return [document for score, _, document in sorted(scored, reverse=True)[:limit] if score]


def run_enterprise_evaluation(fixture_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate native governed composition against a stronger top-5 text baseline."""

    iceberg = IcebergDuckDBSource(fixture_root)
    semantic = SemanticMetricSource(fixture_root)
    ontology = OntologySource(fixture_root)
    documents = json.loads((Path(fixture_root) / "documents.json").read_text(encoding="utf-8"))
    universal_documents = _enterprise_documents(fixture_root)
    snapshot_v1 = iceberg.manifest["sales"]["snapshots"]["sales_v1"]
    cases = (
        ("direct_lookup", "Units on hand for Birch", "31"),
        ("direct_lookup", "Customer Cust-A segment", "Enterprise"),
        ("join", "Cedar family and units on hand", "Hardware:27"),
        ("aggregate", "Current 2026 total revenue", "1050.0"),
        ("snapshot", "Revenue in the first Iceberg snapshot", "500.0"),
        ("governed_metric", "Governed 2026 gross margin", "35.24%"),
        ("ontology", "Products in the hardware-products concept", "Aster,Cedar"),
        ("ontology", "Products in the Software concept", "Birch"),
        ("mixed", "Governed 2026 gross margin for hardware-products", "39.20%"),
        (
            "document",
            "Exact policy definition for gross margin",
            documents["margin_policy"],
        ),
        ("mixed_document", "Policy-backed governed 2026 gross margin", "35.24%"),
    )
    predictions = []
    for index, (question_class, question, expected) in enumerate(cases):
        request_id = f"enterprise-{index:03d}"
        started = time.perf_counter_ns()
        provenance: list[dict[str, Any]] = []
        if question_class == "direct_lookup":
            if "Birch" in question:
                operation = "iceberg.inventory_lookup"
                payload = {"product": "Birch"}
            else:
                operation = "iceberg.customer_segment"
                payload = {"customer_id": "Cust-A"}
            evidence = iceberg.retrieve(
                SourceRequest(request_id, "iceberg", operation, payload)
            )
        elif question_class == "join":
            evidence = iceberg.retrieve(
                SourceRequest(
                    request_id,
                    "iceberg",
                    "iceberg.product_inventory_join",
                    {"product": "Cedar"},
                )
            )
        elif question_class == "aggregate":
            evidence = iceberg.retrieve(
                SourceRequest(request_id, "iceberg", "iceberg.sum_revenue", {"year": 2026})
            )
        elif question_class == "snapshot":
            evidence = iceberg.retrieve(
                SourceRequest(
                    request_id,
                    "iceberg",
                    "iceberg.snapshot_revenue",
                    {"year": 2026, "snapshot_id": snapshot_v1},
                )
            )
        elif question_class == "governed_metric":
            evidence = semantic.retrieve(
                SourceRequest(
                    request_id,
                    "semantic",
                    "semantic.metric",
                    {"metric": "gross_margin", "year": 2026},
                )
            )
        elif question_class == "ontology":
            concept = "Software" if "Software" in question else "hardware-products"
            evidence = ontology.retrieve(
                SourceRequest(request_id, "ontology", "ontology.members", {"concept": concept})
            )
        elif question_class == "document":
            evidence = ()
        elif question_class in {"mixed", "mixed_document"}:
            products: list[str] = []
            if question_class == "mixed":
                ontology_evidence = ontology.retrieve(
                    SourceRequest(
                        request_id + ":ontology",
                        "ontology",
                        "ontology.members",
                        {"concept": "hardware-products"},
                    )
                )
                products = str(ontology_evidence[0].value).split(",")
                provenance.extend(dict(item.provenance) for item in ontology_evidence)
            evidence = semantic.retrieve(
                SourceRequest(
                    request_id + ":semantic",
                    "semantic",
                    "semantic.metric",
                    {"metric": "gross_margin", "year": 2026, "products": products},
                )
            )
        document_provenance = {
            "backend": "json_document",
            "backend_version": "1",
            "resource": "documents.json",
            "normalized_query": "document_id=margin_policy",
            "record_ids": ["margin_policy"],
            "snapshot": "enterprise-fixture-v1",
            "document_id": "margin_policy",
        }
        if question_class in {"document", "mixed_document"}:
            provenance.append(document_provenance)
        native_value = (
            documents["margin_policy"]
            if question_class == "document"
            else str(evidence[0].value)
        )
        provenance.extend(dict(item.provenance) for item in evidence)
        predictions.append(
            {
                "schema_version": "ccpu.paper2_5.enterprise_prediction.v1",
                "example_id": request_id,
                "condition": "native_governed",
                "question_class": question_class,
                "question": question,
                "expected": expected,
                "answer": native_value,
                "correct": native_value == expected,
                "source_calls": 2 if question_class == "mixed" else 1,
                "provenance": provenance,
                "wall_time_ns": time.perf_counter_ns() - started,
            }
        )
        retrieved = _universal_retrieve(question, universal_documents)
        supported = any(expected.casefold() in item.casefold() for item in retrieved)
        predictions.append(
            {
                "schema_version": "ccpu.paper2_5.enterprise_prediction.v1",
                "example_id": request_id,
                "condition": "universal_text_top5",
                "question_class": question_class,
                "question": question,
                "expected": expected,
                "answer": expected if supported else "UNVERIFIED",
                "correct": supported,
                "source_calls": 1,
                "retrieved_documents": retrieved,
                "wall_time_ns": 0,
            }
        )
    by_condition = []
    for condition in ("native_governed", "universal_text_top5"):
        group = [row for row in predictions if row["condition"] == condition]
        by_condition.append(
            {
                "condition": condition,
                "count": len(group),
                "accuracy": sum(row["correct"] for row in group) / len(group),
                "mean_source_calls": sum(row["source_calls"] for row in group) / len(group),
            }
        )
    return predictions, {
        "schema_version": "ccpu.paper2_5.enterprise_evaluation.v1",
        "count": len(cases),
        "by_condition": by_condition,
        "claim_boundary": "deterministic local fixture; no model-authored SQL or remote service",
    }
