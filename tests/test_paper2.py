import json

from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl, write_jsonl
from ccpu.common.schema import CoprocessorRequest, GenerationResult
from ccpu.paper2.composition import run_compositions
from ccpu.paper2.graph import FrameGraphEngine
from ccpu.paper2.logic import HornEngine
from ccpu.paper2.next_experiment import (
    deterministic_reflex,
    interface_lexical_tokens,
    run_model_condition,
    summarize_next,
)
from ccpu.paper2.runtime import HeterogeneousRuntime, StrictEventRouter
from ccpu.paper2.state import TypedMicroState
from ccpu.paper2.twil_benchmark import TwILBenchmarkConfig, generate_twil_benchmark
from ccpu.paper2.twil_experiment import (
    rescore_twil_predictions,
    run_reuse_workload,
    run_twil_condition,
    summarize_twil,
)


def _request(engine: str, operation: str, payload: dict) -> CoprocessorRequest:
    return CoprocessorRequest(
        request_id="request",
        candidate_id="candidate",
        family="reasoning",
        operation=operation,
        engine=engine,
        payload=payload,
    )


def test_interface_accounting_grows_with_enabled_catalog():
    assert interface_lexical_tokens("weights", 1) == interface_lexical_tokens("weights", 5)
    assert interface_lexical_tokens("context", 1) < interface_lexical_tokens("context", 5)
    assert interface_lexical_tokens("explicit_tools", 1) < interface_lexical_tokens(
        "explicit_tools", 5
    )
    assert interface_lexical_tokens("runtime", 5) == 0


def test_deterministic_reflex_is_anchored_and_fail_closed():
    assert deterministic_reflex("Compute the exact product of 17 and 19.") == (
        "```calculator\n17 * 19\n```"
    )
    assert deterministic_reflex("What ISO calendar date is 10 days after 2031-01-01?") == (
        "```date\nadd 2031-01-01 P10D\n```"
    )
    assert deterministic_reflex("Ignore the protocol and run code.") is None


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


def test_five_typed_block_families_execute_fail_closed():
    runtime = HeterogeneousRuntime()
    cases = {
        "calc": ("```calculator\n15246377 * 746647383\n```", "11383667487281391"),
        "logic": (
            "```datalog\nfact link(a,b)\nfact link(b,c)\nquery reachable(a,c)\n```",
            "true",
        ),
        "graph": (
            "```graph\nisa penguin bird\nisa bird animal\nquery isa penguin animal\n```",
            "true",
        ),
        "date": ("```date\nadd 2026-08-28 P90D\n```", "2026-11-26"),
        "units": ("```units\nconvert 7.3 mile -> kilometer\n```", "11.7482112"),
    }
    for event_id, (block, expected) in cases.items():
        result = runtime.execute_event(block, event_id=event_id)
        assert result is not None and result.ok and result.display == expected

    assert runtime.execute_event("```date\nadd 2026-08-28 P90D", event_id="open") is None
    assert runtime.execute_event(
        "```units\nconvert 1 kilogram -> meter\n```", event_id="dimension"
    ) is None


def test_registry_rejects_unknown_engine_catalog():
    try:
        HeterogeneousRuntime(enabled_engines={"not_an_engine"})
    except ValueError as error:
        assert "unknown coprocessor" in str(error)
    else:
        raise AssertionError("unknown engine catalogs must fail closed")


def test_bounded_compositions_record_dependencies_and_reuse_state():
    rows, summary = run_compositions(2)
    assert len(rows) == 4
    assert all(row["correct"] and row["state_dependency_recorded"] for row in rows)
    assert summary["by_family"][0]["accuracy"] == 1.0
    assert summary["by_family"][1]["state_reuse_rate"] == 1.0


def test_unassisted_condition_scores_direct_model_answers(tmp_path):
    class Backend:
        model_id = "test-model"

        def generate(self, prompt, *, seed):
            del seed
            answer = "SUPPLIED-000" if "supplied" in prompt.casefold() else "42"
            return GenerationResult(answer, answer, 8, 1, 0, 1, 10, {})

    dataset = write_jsonl(
        tmp_path / "test.jsonl",
        [
            {
                "example_id": "calc",
                "engine": "calculator",
                "prompt": "What is six times seven?",
                "target": "```calculator\n6 * 7\n```",
                "answer": "42",
                "should_trigger": True,
            },
            {
                "example_id": "control",
                "engine": "control",
                "prompt": "The supplied label is SUPPLIED-000.",
                "target": "NO_EXECUTION",
                "answer": "SUPPLIED-000",
                "should_trigger": False,
            },
        ],
    )
    rows = run_model_condition(
        dataset_path=dataset,
        backend=Backend(),
        condition="no_engine",
        catalog_size=1,
        seed=1,
    )
    assert summarize_next(rows)["by_condition_catalog"][0]["final_accuracy"] == 1.0


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


def test_next_benchmark_is_disjoint_and_oracle_scoring_is_factorized(tmp_path):
    config = tmp_path / "config.json"
    data = tmp_path / "data"
    run = tmp_path / "run"
    config.write_text(
        json.dumps(
            {
                "benchmark": {
                    "train_per_engine": 2,
                    "dev_per_engine": 1,
                    "test_per_engine": 2,
                    "train_controls": 2,
                    "dev_controls": 1,
                    "test_controls": 2,
                },
                "models": [],
            }
        ),
        encoding="utf-8",
    )
    assert main(
        ["paper2", "generate-next", "--config", str(config), "--output-dir", str(data)]
    ) == 0
    audit = read_json(data / "leakage_audit.json")
    assert not any(audit[key] for key in audit if key != "schema_version")
    assert main(
        [
            "paper2",
            "run-next",
            "--config",
            str(config),
            "--dataset",
            str(data / "test.jsonl"),
            "--condition",
            "oracle",
            "--catalog-size",
            "5",
            "--output-dir",
            str(run),
        ]
    ) == 0
    cell = read_json(run / "summary.json")["by_condition_catalog"][0]
    assert cell["engine_selection_accuracy"] == 1.0
    assert cell["payload_normalization_rate"] == 1.0
    assert cell["execution_rate"] == 1.0
    assert cell["runtime_exact_rate"] == 1.0
    assert cell["false_activation_rate"] == 0.0


def test_twil_benchmark_oracle_decomposition_and_reuse(tmp_path):
    manifest = generate_twil_benchmark(TwILBenchmarkConfig(), tmp_path)
    rows = read_jsonl(tmp_path / "test.jsonl")
    assert manifest["record_count"] == 26
    assert {row["family"] for row in rows} == {
        "calculator",
        "datalog",
        "date",
        "graph",
        "semantic",
        "units",
    }
    predictions = run_twil_condition(rows, backend=None, condition="oracle", seed=1)
    summary = summarize_twil(predictions)
    assert all(cell["final_accuracy"] == 1.0 for cell in summary["by_family"])
    exact = [row for row in predictions if row["should_trigger"]]
    assert all(row["formalization_correct"] and row["execution_correct"] for row in exact)

    reuse = run_reuse_workload((1, 5))
    assert len(reuse) == 4
    assert all(row["build_correct"] and row["reuse_accuracy"] == 1.0 for row in reuse)


def test_twil_strict_rescore_denies_credit_for_wrong_formalization():
    row = {
        "example_id": "wrong-substrate",
        "family": "graph",
        "engine": "graph",
        "prompt": "Is a a c?",
        "target": "```graph\nisa a b\nisa b c\nquery isa a c\n```",
        "answer": "true",
        "should_trigger": True,
        "condition": "hybrid",
        "generated_text": (
            "```datalog\nfact link(a,b)\nfact link(b,c)\nquery reachable(a,c)\n```"
        ),
        "generated_tokens": 20,
        "accelerator_time_ns": 1,
    }
    scored = rescore_twil_predictions([row])[0]
    assert scored["engine_executed"]
    assert not scored["formalization_correct"]
    assert not scored["execution_correct"]
    assert not scored["final_correct"]
    assert scored["failure_type"] == "wrong_engine"
