import json

from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl
from ccpu.common.schema import CoprocessorRequest
from ccpu.paper2.graph import FrameGraphEngine
from ccpu.paper2.logic import HornEngine
from ccpu.paper2.runtime import HeterogeneousRuntime, StrictEventRouter
from ccpu.paper2.state import TypedMicroState


def _request(engine: str, operation: str, payload: dict) -> CoprocessorRequest:
    return CoprocessorRequest(
        request_id="request",
        candidate_id="candidate",
        family="reasoning",
        operation=operation,
        engine=engine,
        payload=payload,
    )


def test_horn_engine_derives_transitive_fact_and_persists_it():
    state = TypedMicroState()
    engine = HornEngine(state)
    payload = {
        "facts": [
            {"predicate": "link", "arguments": ["a", "b"]},
            {"predicate": "link", "arguments": ["b", "c"]},
        ],
        "rules": [
            {
                "head": {"predicate": "reach", "arguments": ["?x", "?y"]},
                "body": [{"predicate": "link", "arguments": ["?x", "?y"]}],
            },
            {
                "head": {"predicate": "reach", "arguments": ["?x", "?z"]},
                "body": [
                    {"predicate": "reach", "arguments": ["?x", "?y"]},
                    {"predicate": "link", "arguments": ["?y", "?z"]},
                ],
            },
        ],
        "query": {"predicate": "reach", "arguments": ["a", "c"]},
    }
    first = engine.execute(_request("horn", "horn.query", payload))
    second = engine.execute(
        _request(
            "horn",
            "horn.query",
            {"facts": [], "rules": [], "query": payload["query"]},
        )
    )

    assert first.ok and first.display == "true"
    assert first.metadata["derived_facts"] >= 3
    assert second.ok and second.display == "true"
    assert any(item.payload["predicate"] == "reach" for item in state.by_kind("horn_fact"))
    assert any(item.dependencies for item in state.by_kind("horn_fact"))


def test_frame_graph_supports_isa_closure_and_inherited_slots():
    state = TypedMicroState()
    engine = FrameGraphEngine(state)
    isa = [["robin", "bird"], ["bird", "animal"]]
    assert engine.execute(
        _request(
            "frame_graph",
            "graph.isa",
            {"isa": isa, "frames": [], "query": ["robin", "animal"]},
        )
    ).display == "true"
    result = engine.execute(
        _request(
            "frame_graph",
            "graph.frame",
            {"isa": [], "frames": [["bird", "covering", "feathers"]], "query": ["robin", "covering"]},
        )
    )
    assert result.ok and result.display == "feathers"


def test_strict_router_rejects_embedded_or_unknown_tags():
    router = StrictEventRouter()
    assert router.parse("prose [calculator] 2 + 2", event_id="one") is None
    assert router.parse("[unknown] 2 + 2", event_id="two") is None
    assert router.parse("[calculator] 2 + 2", event_id="three").engine == "calculator"


def test_runtime_honors_single_engine_availability():
    runtime = HeterogeneousRuntime(enabled_engines={"horn"})
    assert runtime.execute_event("[calculator] 2 + 2", event_id="calc") is None
    assert runtime.trace[-1]["status"] == "unavailable"


def test_paper2_cli_generates_and_simulates_non_empirical_protocol(tmp_path):
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset.jsonl"
    run = tmp_path / "run"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "depths": [1, 2],
                    "distractors": [0],
                    "examples_per_cell": 1,
                    "control_examples": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    assert main(["paper2", "generate", "--config", str(config), "--output", str(dataset)]) == 0
    assert main(["paper2", "validate", "--dataset", str(dataset)]) == 0
    assert main(["paper2", "simulate", "--dataset", str(dataset), "--output-dir", str(run)]) == 0

    assert len(read_jsonl(dataset)) == 8
    assert len(read_jsonl(run / "predictions.jsonl")) == 56
    summary = read_json(run / "summary.json")
    assert summary["empirical"] is False
    assert summary["prediction_count"] == 56
    by_condition = {row["condition"]: row for row in summary["by_condition"]}
    assert by_condition["heterogeneous"]["task_accuracy"] == 1.0
    assert by_condition["no_engine"]["task_accuracy"] == 0.0
