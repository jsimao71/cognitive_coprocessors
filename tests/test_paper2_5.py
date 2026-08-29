import pytest

from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl
from ccpu.common.retrieval import SourceRequest
from ccpu.paper2_5.composition import run_compositions
from ccpu.paper2_5.production_analysis import analyze_substitution
from ccpu.paper2_5.runtime import RetrievalRegistry
from ccpu.paper2_5.sources import build_sources


def test_source_registry_hides_credentials_and_denies_missing_scope():
    sources = build_sources()
    registry = RetrievalRegistry(sources, set())
    request = SourceRequest(
        request_id="one",
        source_type="db",
        operation="db.max_sales",
        payload={"year": 2026},
    )
    try:
        registry.retrieve(request)
    except PermissionError:
        pass
    else:
        raise AssertionError("missing source credentials must fail closed")
    assert all("credential_scope" not in row for row in registry.public_catalog())


def test_distinct_sources_execute_native_semantics():
    sources = build_sources()
    cases = (
        ("db", "db.sum_sales", {"year": 2026}, "500"),
        (
            "lexical",
            "lexical.search",
            {"document": "policy_17", "query": "termination notice"},
            "ninety calendar days",
        ),
        (
            "vector",
            "vector.search",
            {"query": "profitability shipping discounts"},
            "freight expenses",
        ),
        (
            "web",
            "web.lookup",
            {"entity": "Lumen index", "relation": "current level"},
            "417",
        ),
    )
    for source_type, operation, payload, expected in cases:
        evidence = sources[source_type].retrieve(
            SourceRequest(
                request_id=source_type,
                source_type=source_type,
                operation=operation,
                payload=payload,
            )
        )
        assert evidence and expected in evidence[0].content


def test_local_production_sources_are_substitutable_and_preserve_provenance():
    pytest.importorskip("duckdb")
    pytest.importorskip("faiss")
    from ccpu.paper2_5.production_sources import build_production_sources

    sources = build_production_sources()
    cases = (
        ("db", "db.sum_sales", {"year": 2026}, "500", "duckdb"),
        (
            "lexical",
            "lexical.search",
            {"document": "policy_17", "query": "termination notice"},
            "ninety calendar days",
            "sqlite_fts5",
        ),
        (
            "vector",
            "vector.search",
            {"query": "profitability shipping discounts"},
            "freight expenses",
            "faiss",
        ),
    )
    for source_type, operation, payload, expected, backend in cases:
        evidence = sources[source_type].retrieve(
            SourceRequest(
                request_id=f"production-{source_type}",
                source_type=source_type,
                operation=operation,
                payload=payload,
            )
        )
        assert evidence and expected in evidence[0].content
        provenance = evidence[0].provenance
        assert provenance["backend"] == backend
        assert provenance["backend_version"]
        assert provenance["normalized_query"]
        assert provenance["record_ids"] == [evidence[0].record_id]
        assert provenance["snapshot"]

    registry = RetrievalRegistry(sources, {source.policy.credential_scope for source in sources.values()})
    catalog = registry.public_catalog()
    assert all("credential_scope" not in row for row in catalog)
    assert all(row.get("capabilities") for row in catalog[:3])


def test_local_production_suite_preserves_frozen_native_results(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("faiss")
    from ccpu.paper2_5.benchmark import freeze_benchmark
    from ccpu.paper2_5.experiment import run_matrix

    data = tmp_path / "data"
    freeze_benchmark(data)
    rows, traces = run_matrix(
        data / "benchmark.jsonl", source_count=4, backend_suite="local_production"
    )
    native = [row for row in rows if row["condition"] == "oracle_need_source_query"]
    heuristic = [row for row in rows if row["condition"] == "real_need_heuristic_source"]
    assert len(rows) == 154
    assert all(row["final_correct"] for row in native)
    assert all(row["final_correct"] for row in heuristic)
    production_evidence = [
        item
        for trace in traces
        for item in trace["evidence"]
        if item["source_type"] in {"db", "lexical", "vector"}
    ]
    assert {item["provenance"]["backend"] for item in production_evidence} == {
        "duckdb",
        "sqlite_fts5",
        "faiss",
    }


def test_production_substitution_analysis_is_matched_and_fail_closed(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("faiss")
    from ccpu.paper2_5.benchmark import freeze_benchmark
    from ccpu.paper2_5.experiment import run_matrix

    data = tmp_path / "data"
    freeze_benchmark(data)
    controlled, _ = run_matrix(data / "benchmark.jsonl", source_count=4)
    production, traces = run_matrix(
        data / "benchmark.jsonl", source_count=4, backend_suite="local_production"
    )
    summary = analyze_substitution(controlled, production, traces)
    assert summary["matched_prediction_count"] == 154
    assert summary["final_decision_agreement"] == 1.0
    assert summary["support_decision_agreement"] == 1.0
    assert all(row["provenance_complete_rate"] == 1.0 for row in summary["by_source_oracle"][:3])
    assert summary["claim_boundary"]["docker_used"] is False


def test_enterprise_iceberg_semantic_ontology_composition(tmp_path):
    pytest.importorskip("pyiceberg")
    pytest.importorskip("pyoxigraph")
    from ccpu.paper2_5.enterprise import (
        IcebergDuckDBSource,
        OntologySource,
        SemanticMetricSource,
        create_enterprise_fixture,
        run_enterprise_evaluation,
    )

    fixture = tmp_path / "enterprise"
    manifest = create_enterprise_fixture(fixture)
    assert manifest["sales"]["snapshots"]["sales_v1"] != manifest["sales"]["snapshots"][
        "sales_v2"
    ]
    assert manifest["sales"]["schema_id"] == 1

    iceberg = IcebergDuckDBSource(fixture)
    current = iceberg.retrieve(
        SourceRequest(
            request_id="current",
            source_type="iceberg",
            operation="iceberg.sum_revenue",
            payload={"year": 2026},
        )
    )
    historical = iceberg.retrieve(
        SourceRequest(
            request_id="historical",
            source_type="iceberg",
            operation="iceberg.snapshot_revenue",
            payload={
                "year": 2026,
                "snapshot_id": manifest["sales"]["snapshots"]["sales_v1"],
            },
        )
    )
    assert current[0].value == "1050.0"
    assert historical[0].value == "500.0"
    assert current[0].provenance["iceberg_snapshot_id"] == manifest["sales"][
        "current_snapshot_id"
    ]

    ontology = OntologySource(fixture)
    members = ontology.retrieve(
        SourceRequest(
            request_id="members",
            source_type="ontology",
            operation="ontology.members",
            payload={"concept": "hardware-products"},
        )
    )
    assert members[0].value == "Aster,Cedar"
    metric = SemanticMetricSource(fixture).retrieve(
        SourceRequest(
            request_id="metric",
            source_type="semantic",
            operation="semantic.metric",
            payload={
                "metric": "gross_margin",
                "year": 2026,
                "products": members[0].value.split(","),
            },
        )
    )
    assert metric[0].value == "39.20%"
    assert metric[0].provenance["semantic_metric_version"] == "1.0.0"

    rows, summary = run_enterprise_evaluation(fixture)
    conditions = {row["condition"]: row for row in summary["by_condition"]}
    assert conditions["native_governed"]["accuracy"] == 1.0
    assert conditions["universal_text_top5"]["accuracy"] < 0.5
    native = [row for row in rows if row["condition"] == "native_governed"]
    assert all(row["provenance"] for row in native)


def test_bounded_source_compositions_log_dependency_dags():
    rows, summary = run_compositions(2)
    assert len(rows) == 4
    assert all(row["correct"] and row["dependency_dag"][1]["depends_on"] for row in rows)
    assert all(cell["final_accuracy"] == 1.0 for cell in summary["by_family"])


def test_cli_freezes_runs_scales_and_decides_gate(tmp_path):
    data = tmp_path / "data"
    assert main(["paper2.5", "freeze", "--output-dir", str(data)]) == 0
    assert len(read_jsonl(data / "benchmark.jsonl")) == 22
    predictions = []
    for source_count in (1, 2, 3, 4):
        run = tmp_path / f"run_{source_count}"
        assert main(
            [
                "paper2.5",
                "run",
                "--benchmark",
                str(data / "benchmark.jsonl"),
                "--source-count",
                str(source_count),
                "--output-dir",
                str(run),
            ]
        ) == 0
        predictions.append(str(run / "predictions.jsonl"))
    analysis = tmp_path / "analysis"
    assert main(
        [
            "paper2.5",
            "analyze",
            "--predictions",
            *predictions,
            "--output-dir",
            str(analysis),
        ]
    ) == 0
    summary = read_json(analysis / "summary.json")
    assert summary["prediction_count"] == 469
    gate = read_json(analysis / "paper3_5_gate.json")
    assert gate["criteria"]["source_native_oracle_value"] is True
    assert gate["criteria"]["context_burden_growth"] is True
