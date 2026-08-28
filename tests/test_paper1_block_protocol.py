from ccpu.paper1.block_protocol import analyze_block_protocol
from ccpu.paper1.dataset import ArithmeticDatasetConfig, iter_dataset
from ccpu.paper1.evaluate import evaluate
from ccpu.paper1.experiment import run_scripted
from ccpu.paper1.prompts import ICL_BLOCK_CONDITIONS, condition_prompt


def _examples():
    return list(
        iter_dataset(
            ArithmeticDatasetConfig(
                seed=31,
                examples_per_cell=1,
                operator_counts=(1,),
                operand_digits=(2,),
                control_examples=1,
            )
        )
    )


def test_block_protocol_decomposes_wrapper_payload_and_semantics():
    exact = analyze_block_protocol("```calculator\n(12 + 3) * 4\n```", "(12 + 3) * 4")
    equivalent = analyze_block_protocol("```calculator\n(12+3)*4\n```", "(12 + 3) * 4")
    unclosed = analyze_block_protocol("```calculator\n(12 + 3) * 4", "(12 + 3) * 4")
    redundant_close = analyze_block_protocol(
        "```calculator\n(12 + 3) * 4\n``````", "(12 + 3) * 4"
    )

    assert exact["block_open"] and exact["block_close"]
    assert exact["block_payload_exact"] and exact["block_payload_semantically_equivalent"]
    assert not equivalent["block_payload_exact"]
    assert equivalent["block_payload_semantically_equivalent"]
    assert unclosed["block_open"] and not unclosed["block_close"]
    assert unclosed["block_payload_present"]
    assert redundant_close["block_close"]
    assert redundant_close["block_payload_semantically_equivalent"]


def test_icl_prompts_use_concrete_demonstrations_without_placeholder_target():
    arithmetic = next(item for item in _examples() if item.task_kind == "arithmetic")
    control = next(item for item in _examples() if item.task_kind == "control")

    for condition in ICL_BLOCK_CONDITIONS:
        arithmetic_prompt = condition_prompt(arithmetic, condition)
        control_prompt = condition_prompt(control, condition)
        assert "EXPRESSION" not in arithmetic_prompt
        assert str(arithmetic.expression) in arithmetic_prompt
        assert control.reference_completion in control_prompt

    assert "Negative example" in condition_prompt(arithmetic, "calculator_block_icl_c")
    assert "verbatim" in condition_prompt(arithmetic, "calculator_block_icl_d")


def test_scripted_icl_conditions_report_complete_block_protocol_metrics():
    examples = _examples()
    predictions, traces = run_scripted(examples, conditions=ICL_BLOCK_CONDITIONS)
    summary = evaluate(examples, predictions)

    assert traces
    assert len(predictions) == len(examples) * len(ICL_BLOCK_CONDITIONS)
    for row in summary["by_run"]:
        assert row["block_payload_semantically_equivalent_rate"] == 1.0
        assert row["block_execution_rate"] == 1.0
        assert row["false_block_rate"] == 0.0
    seeded_open = next(
        row for row in summary["by_run"] if row["condition"] == "calculator_block_icl_e"
    )
    seeded_wrapper = next(
        row for row in summary["by_run"] if row["condition"] == "calculator_block_icl_f"
    )
    assert seeded_open["block_open_rate"] == 1.0
    assert seeded_open["block_open_model_rate"] == 0.0
    assert seeded_wrapper["result_use_rate"] is None
