from decimal import Decimal

import pytest

from ccpu.cli import main
from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json
from ccpu.common.retrieval import SourceRequest
from ccpu.paper2_5.composition import run_compositions
from ccpu.paper2_5.production_analysis import analyze_substitution
from ccpu.paper2_5.public_benchmarks import _decimal_expression, _score_derivation
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
    from ccpu.paper2_5.generic_tools import compare_enterprise_result_transports

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
    transport = compare_enterprise_result_transports(fixture, tmp_path / "transport")
    assert transport["record_agreement"] == 1.0
    assert transport["tool_result_accuracy"] == 1.0
    assert {row["schema_tokens"] for row in transport["registry_scaling"]} == {72}
    assert transport["claim_boundary"]["automatic_rescue_rate"] is None


def test_bounded_source_compositions_log_dependency_dags():
    rows, summary = run_compositions(2)
    assert len(rows) == 4
    assert all(row["correct"] and row["dependency_dag"][1]["depends_on"] for row in rows)
    assert all(cell["final_accuracy"] == 1.0 for cell in summary["by_family"])


def test_tatqa_arithmetic_uses_exact_decimal_and_percent_scaling():
    assert _decimal_expression("(14,740 + 1,910) / 2") == 8325
    assert _decimal_expression("53% * $23,406") == Decimal("12405.18")
    assert abs(_decimal_expression("1,027 / 11%") - Decimal("9336.363636")) < Decimal(
        "0.000001"
    )
    scored = _score_derivation(
        {
            "answer_type": "arithmetic",
            "derivation": "(104 - 89) / 89",
            "answer": 16.85,
            "scale": "percent",
        }
    )
    assert scored["oracle_compute_available"] is True
    assert scored["oracle_compute_exact"] is True
    unsupported = _score_derivation(
        {
            "answer_type": "arithmetic",
            "derivation": "104 minus 89",
            "answer": 15,
            "scale": "",
        }
    )
    assert unsupported["oracle_compute_available"] is False
    assert unsupported["oracle_compute_exact"] is None


def test_tatqa_public_cli_freezes_and_analyzes_verified_source(tmp_path):
    cache_root = tmp_path / "cache"
    source_path = cache_root / "tatqa" / "tatqa_dataset_dev.json"
    questions = [
        {
            "uid": "arithmetic-average",
            "question": "What is the average?",
            "answer": 8325,
            "derivation": "(14,740 + 1,910) / 2",
            "answer_type": "arithmetic",
            "answer_from": "table",
            "scale": "",
            "req_comparison": False,
            "rel_paragraphs": [],
        },
        {
            "uid": "arithmetic-percent",
            "question": "What is the percentage increase?",
            "answer": 16.85,
            "derivation": "(104 - 89) / 89",
            "answer_type": "arithmetic",
            "answer_from": "table-text",
            "scale": "percent",
            "req_comparison": True,
            "rel_paragraphs": ["1"],
        },
        {
            "uid": "count",
            "question": "How many entries?",
            "answer": 2,
            "derivation": "",
            "answer_type": "count",
            "answer_from": "text",
            "scale": "",
            "req_comparison": False,
            "rel_paragraphs": ["1"],
        },
        {
            "uid": "span",
            "question": "Which entry?",
            "answer": ["Aster"],
            "derivation": "",
            "answer_type": "span",
            "answer_from": "text",
            "scale": "",
            "req_comparison": False,
            "rel_paragraphs": ["1"],
        },
    ]
    write_json(
        source_path,
        [
            {
                "table": {
                    "uid": "table",
                    "table": [
                        ["Company", "Value A", "Value B", "Before", "After"],
                        ["Aster", "14,740", "1,910", "89", "104"],
                    ],
                },
                "paragraphs": [{"uid": "paragraph", "order": 1, "text": "Aster"}],
                "questions": questions,
            }
        ],
    )
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "schema_version": "ccpu.paper2_5.public_tatqa_config.v1",
            "selection_seed": 7,
            "max_rows": 4,
            "source": {
                "dataset": "next-tat/TAT-QA",
                "revision": "test-revision",
                "file": source_path.name,
                "file_sha256": file_sha256(source_path),
                "expected_documents": 1,
                "expected_questions": 4,
            },
        },
    )
    frozen = tmp_path / "frozen"
    assert main(
        [
            "paper2.5",
            "freeze-public-tatqa",
            "--config",
            str(config),
            "--cache-root",
            str(cache_root),
            "--output-dir",
            str(frozen),
        ]
    ) == 0
    assert read_json(frozen / "manifest.json")["record_count"] == 4

    analysis = tmp_path / "analysis"
    assert main(
        [
            "paper2.5",
            "analyze-public-tatqa",
            "--config",
            str(config),
            "--cache-root",
            str(cache_root),
            "--selection",
            str(frozen / "selection.jsonl"),
            "--output-dir",
            str(analysis),
        ]
    ) == 0
    summary = read_json(analysis / "summary.json")
    assert summary["record_count"] == 4
    assert summary["compute_required_rate"] == 0.75
    assert summary["oracle_compute_coverage"] == 0.5
    assert summary["oracle_compute_exact_rate"] == 1.0
    assert summary["source_native_adapter_coverage"] == 0.0

    retrieval = tmp_path / "retrieval"
    assert main(
        [
            "paper2.5",
            "analyze-public-tatqa-retrieval",
            "--config",
            str(config),
            "--cache-root",
            str(cache_root),
            "--selection",
            str(frozen / "selection.jsonl"),
            "--output-dir",
            str(retrieval),
        ]
    ) == 0
    retrieval_summary = read_json(retrieval / "retrieval_summary.json")
    assert retrieval_summary["evaluable_count"] == 4
    assert retrieval_summary["top_k"] == 5
    assert retrieval_summary["claim_boundary"]["ranking_uses_gold"] is False


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
