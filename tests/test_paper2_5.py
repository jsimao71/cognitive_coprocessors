from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl
from ccpu.common.retrieval import SourceRequest
from ccpu.paper2_5.composition import run_compositions
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
