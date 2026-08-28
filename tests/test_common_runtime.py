from dataclasses import dataclass

from ccpu.common.runtime import ReflexRuntime
from ccpu.common.schema import (
    CoprocessorRequest,
    CoprocessorResult,
    DetectionCandidate,
    MicroStateItem,
    TraceStage,
)


class BangDetector:
    name = "bang"

    def __init__(self):
        self.offset = 0

    def reset(self):
        self.offset = 0

    def feed(self, text):
        candidates = []
        for character in text:
            if character == "!":
                candidates.append(
                    DetectionCandidate(
                        candidate_id=f"bang-{self.offset}",
                        family="test",
                        raw_text="ping",
                        start_offset=self.offset,
                        end_offset=self.offset + 1,
                        detector=self.name,
                    )
                )
            self.offset += 1
        return candidates


class PingNormalizer:
    def normalize(self, candidate):
        return CoprocessorRequest(
            request_id=f"{candidate.candidate_id}:request",
            candidate_id=candidate.candidate_id,
            family="test",
            operation="ping",
            engine="echo",
            payload={"value": candidate.raw_text},
        )


class EchoEngine:
    name = "echo"

    def execute(self, request):
        return CoprocessorResult(
            request_id=request.request_id,
            engine=self.name,
            ok=True,
            value=request.payload["value"],
            display="pong",
        )


@dataclass
class BracketMaterializer:
    def materialize(self, request, result, state: MicroStateItem):
        return f"[{result.display}]"


def test_runtime_injects_before_remainder_of_input_fragment():
    runtime = ReflexRuntime(
        detector=BangDetector(),
        normalizer=PingNormalizer(),
        engines={"echo": EchoEngine()},
        materializer=BracketMaterializer(),
        run_id="test-run",
    )
    step = runtime.feed("before!after")

    assert step.rendered_text == "before![pong]after"
    assert runtime.rendered_text == step.rendered_text
    assert runtime.intervention_count == 1
    assert [event.stage for event in runtime.trace] == [
        TraceStage.DETECTION,
        TraceStage.NORMALIZATION,
        TraceStage.ROUTING,
        TraceStage.EXECUTION,
        TraceStage.STATE_UPDATE,
        TraceStage.REINJECTION,
    ]
    assert runtime.trace[1].details["request"]["operation"] == "ping"
    assert runtime.trace[3].details["result"]["display"] == "pong"
    assert runtime.trace[4].details["state"]["result"]["display"] == "pong"


def test_missing_engine_is_traced_without_state_or_reinjection():
    runtime = ReflexRuntime(
        detector=BangDetector(),
        normalizer=PingNormalizer(),
        engines={},
        materializer=BracketMaterializer(),
        run_id="missing-engine",
    )
    step = runtime.feed("!")

    assert step.rendered_text == "!"
    assert not runtime.state
    assert runtime.trace[-1].stage == TraceStage.ROUTING
    assert runtime.trace[-1].details["error"] == "engine_not_registered"


def test_reset_clears_detector_state_trace_and_micro_state():
    runtime = ReflexRuntime(
        detector=BangDetector(),
        normalizer=PingNormalizer(),
        engines={"echo": EchoEngine()},
        materializer=BracketMaterializer(),
    )
    runtime.feed("!")
    runtime.reset(run_id="fresh")

    assert runtime.run_id == "fresh"
    assert runtime.rendered_text == ""
    assert runtime.trace == []
    assert runtime.state == []
