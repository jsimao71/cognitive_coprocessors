import pytest

from ccpu.common.schema import TraceStage, TraceStatus
from ccpu.paper1.reflex import build_explicit_tool_runtime, build_reflex_runtime


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
