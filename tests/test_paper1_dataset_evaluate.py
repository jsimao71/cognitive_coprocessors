from ccpu.common.metrics import binary_classification, wilson_interval
from ccpu.paper1.dataset import (
    ArithmeticDatasetConfig,
    HardArithmeticDatasetConfig,
    iter_dataset,
    iter_hard_dataset,
    reference_answer,
)
from ccpu.paper1.evaluate import answers_equal, evaluate, extract_answer
from ccpu.paper1.experiment import run_replay, run_scripted


def smoke_config() -> ArithmeticDatasetConfig:
    return ArithmeticDatasetConfig(
        seed=5,
        examples_per_cell=2,
        operator_counts=(1, 2),
        operand_digits=(1, 2),
        control_examples=4,
    )


def test_dataset_is_deterministic_unique_and_has_expected_factorial_size():
    first = list(iter_dataset(smoke_config()))
    second = list(iter_dataset(smoke_config()))

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == smoke_config().record_count == 12
    assert len({item.example_id for item in first}) == len(first)
    assert sum(item.should_trigger for item in first) == 8


def test_scripted_run_is_complete_but_explicitly_not_empirical():
    examples = list(iter_dataset(smoke_config()))
    predictions, traces = run_scripted(examples)
    summary = evaluate(examples, predictions)

    assert len(predictions) == len(examples) * 7
    assert traces
    assert all(not row["backend_metadata"]["empirical"] for row in predictions)
    reflex = next(row for row in summary["by_run"] if row["condition"] == "reflex")
    assert reflex["accuracy"] == 1.0
    assert reflex["trigger"]["precision"] == reflex["trigger"]["recall"] == 1.0
    assert reflex["normalization_correctness"] == 1.0
    assert reflex["engine_correctness"] == 1.0
    assert reflex["result_use_rate"] == 1.0
    assert reflex["mean_trace_bytes"] > 0
    assert reflex["mean_state_bytes"] > 0
    llm_only = next(row for row in summary["by_run"] if row["condition"] == "llm_only")
    assert llm_only["normalization_correctness"] is None
    assert llm_only["engine_correctness"] is None


def test_oracle_replay_primes_the_gold_expression():
    example = next(item for item in iter_dataset(smoke_config()) if item.task_kind == "arithmetic")
    predictions, traces = run_replay(
        [example],
        [
            {
                "example_id": example.example_id,
                "condition": "oracle",
                "model_id": "saved-model",
                "seed": 7,
                "generated_text": f"Exact answer: {example.answer}",
                "backend_metadata": {"device": "xpu"},
            }
        ],
    )

    assert traces
    assert predictions[0]["interventions"] == 1
    assert predictions[0]["engine_correct"] is True
    assert predictions[0]["backend_metadata"]["device"] == "xpu"


def test_answer_extraction_scores_model_override_not_only_engine_output():
    text = "<tool_result>42</tool_result> I ignored it. Final answer: 41"
    assert extract_answer(text) == "41"
    assert not answers_equal(extract_answer(text), "42")
    assert answers_equal("2/4", "1/2")


def test_answer_extraction_supports_boxed_and_bare_exact_answers():
    assert extract_answer(r"The answer is \boxed{6}.") == "6"
    assert extract_answer("Exact value of the integer arithmetic expression: 6.") == "6"
    assert extract_answer("The exact value of the expression is **39**.") == "39"
    assert extract_answer("-3/4") == "-3/4"


def test_binary_metrics_and_wilson_interval_boundaries():
    metrics = binary_classification([True, True, False, False], [True, False, True, False])
    assert metrics == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "false_intervention_rate": 0.5,
    }
    low, high = wilson_interval(8, 8)
    assert 0.67 < low < 0.68
    assert high == 1.0


def test_hard_dataset_covers_factorial_axes_and_is_bounded():
    config = HardArithmeticDatasetConfig(
        seed=9,
        examples_per_cell=1,
        operator_counts=(1, 2, 3, 4),
        operand_digits=(1, 2, 3, 4),
        control_examples=2,
    )
    examples = list(iter_hard_dataset(config))
    arithmetic = [item for item in examples if item.task_kind == "arithmetic"]

    assert len(arithmetic) == 4 * 4 * 3
    assert {item.difficulty["structure"] for item in arithmetic} == {
        "left_chain",
        "nested",
        "multiplication_tree",
    }
    assert all(item.answer == reference_answer(item.expression) for item in arithmetic)


def test_hard_dataset_selected_cells_and_surface_variants_are_frozen():
    config = HardArithmeticDatasetConfig(
        examples_per_cell=4,
        selected_cells=((2, 4, "nested"),),
        control_examples=0,
    )
    examples = list(iter_hard_dataset(config))

    assert len(examples) == 4
    assert [item.difficulty["surface_variant"] for item in examples] == [
        "ascii",
        "latex",
        "unicode",
        "brackets",
    ]
