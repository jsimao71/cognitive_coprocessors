import pytest

from ccpu.common.schema import TraceStage, TraceStatus
from ccpu.paper1.reflex import (
    build_calculator_block_runtime,
    build_explicit_tool_runtime,
    build_normalized_reflex_runtime,
    build_reflex_runtime,
)


def test_reflex_detects_ordinary_completed_arithmetic_and_injects_immediately():
    runtime = build_reflex_runtime(run_id="arithmetic")
    step = runtime.feed("The total is 37 * 48 = and checked.")

    assert step.rendered_text == "The total is 37 * 48 = 1776 and checked."
    assert runtime.state[0].request.payload["canonical_expression"] == "37*48"
    assert runtime.state[0].result.display == "1776"


def test_reflex_is_incremental_across_arbitrary_fragment_boundaries():
    runtime = build_reflex_runtime(run_id="fragmented")
    fragments = ("Result: (2", " + 3)", " * 4", " =")
    rendered = "".join(runtime.feed(fragment).rendered_text for fragment in fragments)

    assert rendered == "Result: (2 + 3) * 4 = 20"
    assert runtime.intervention_count == 1


@pytest.mark.parametrize(
    "text",
    [
        'The quote "2 + 2 =" is not an asserted calculation.',
        "Version 2.5 = stable",
        "Use a == b in pseudocode.",
        "There are 12 and 7 rooms.",
    ],
)
def test_non_trigger_controls_do_not_intervene(text):
    runtime = build_reflex_runtime()
    assert runtime.feed(text).rendered_text == text
    assert runtime.intervention_count == 0


def test_lexical_candidate_can_fail_normalization_without_execution():
    runtime = build_reflex_runtime()
    runtime.feed("The equation is x + 2 = 5.")

    assert runtime.intervention_count == 0
    assert any(
        event.stage == TraceStage.NORMALIZATION and event.status == TraceStatus.REJECTED
        for event in runtime.trace
    )
    assert not any(event.stage == TraceStage.EXECUTION for event in runtime.trace)


def test_explicit_tool_baseline_uses_same_normalizer_and_engine():
    runtime = build_explicit_tool_runtime(run_id="tool")
    step = runtime.feed("<tool:calculator>(9 + 1) * 5</tool>")

    assert step.rendered_text.endswith("<tool_result>50</tool_result>")
    assert runtime.state[0].request.operation == "arithmetic.evaluate"
    assert runtime.state[0].result.display == "50"


def test_failed_engine_result_is_not_reinjected():
    runtime = build_reflex_runtime()
    step = runtime.feed("4 / 0 =")

    assert step.rendered_text == "4 / 0 ="
    assert runtime.intervention_count == 0
    assert runtime.trace[-1].stage == TraceStage.EXECUTION
    assert runtime.trace[-1].status == TraceStatus.FAILED


@pytest.mark.parametrize(
    "surface",
    [
        r"[3 \times 2]",
        r"{3 \cdot 2}",
        r"6 \div 2",
        "3 × 2",
        "6 ÷ 2",
        "8 − 3",
    ],
)
def test_normalized_reflex_accepts_allowlisted_surface_aliases(surface):
    runtime = build_normalized_reflex_runtime()
    step = runtime.feed(f"Calculation: {surface} =")

    assert runtime.intervention_count == 1
    assert runtime.state[0].request.payload["surface_normalizer"] == "surface_normalizer_v1"
    assert step.reinjections


@pytest.mark.parametrize(
    "text",
    [
        'The quoted code "3 × 2 =" is documentation.',
        r"The LaTeX command \times controls spacing, not arithmetic = prose.",
        r"The equation x \times 2 = 6 contains a variable.",
        "The phrase three times two = six is prose.",
    ],
)
def test_normalized_reflex_rejects_adversarial_non_arithmetic(text):
    runtime = build_normalized_reflex_runtime()
    assert runtime.feed(text).rendered_text == text
    assert runtime.intervention_count == 0


def test_calculator_block_waits_for_closing_fence_and_executes_full_expression():
    runtime = build_calculator_block_runtime()
    opening = "```calculator\n(12 + 3) * 4\n"

    assert runtime.feed(opening).reinjections == ()
    assert runtime.intervention_count == 0
    step = runtime.feed("```")

    assert runtime.intervention_count == 1
    assert runtime.state[0].request.payload["canonical_expression"] == "(12+3)*4"
    assert step.rendered_text.endswith(
        "```\n<calculator_result>60</calculator_result>\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        "> ```calculator\n2 + 2\n```",
        "Inline example: ```calculator\n2 + 2\n```",
        "```python\n2 + 2\n```",
        "```calculator\nx + 2\n```",
    ],
)
def test_calculator_block_rejects_quoted_examples_and_wrong_grammar(text):
    runtime = build_calculator_block_runtime()
    runtime.feed(text)
    assert runtime.intervention_count == 0
