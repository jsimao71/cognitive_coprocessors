"""Generic observable runtime for incremental semantic interrupts."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from .interfaces import Coprocessor, IncrementalDetector, Materializer, Normalizer
from .schema import (
    CoprocessorRequest,
    DetectionCandidate,
    MicroStateItem,
    Reinjection,
    RuntimeStep,
    TraceEvent,
    TraceStage,
    TraceStatus,
)


class ReflexRuntime:
    """Compose detection, normalization, routing, execution, state, and reinjection.

    The detector receives one character at a time even when callers provide a
    larger token fragment. This guarantees that a result is inserted immediately
    after the character completing a strict event, before the rest of the fragment.
    """

    def __init__(
        self,
        *,
        detector: IncrementalDetector,
        normalizer: Normalizer,
        engines: Mapping[str, Coprocessor],
        materializer: Materializer,
        run_id: str | None = None,
        clock_ns=time.time_ns,
        timer_ns=time.perf_counter_ns,
    ) -> None:
        self.detector = detector
        self.normalizer = normalizer
        self.engines = dict(engines)
        self.materializer = materializer
        self._clock_ns = clock_ns
        self._timer_ns = timer_ns
        self.run_id = run_id or uuid.uuid4().hex
        self.trace: list[TraceEvent] = []
        self.state: list[MicroStateItem] = []
        self.rendered_text = ""
        self._sequence = 0

    @property
    def intervention_count(self) -> int:
        return len(self.state)

    def reset(self, *, run_id: str | None = None) -> None:
        self.detector.reset()
        self.run_id = run_id or uuid.uuid4().hex
        self.trace.clear()
        self.state.clear()
        self.rendered_text = ""
        self._sequence = 0

    def _record(
        self,
        stage: TraceStage,
        status: TraceStatus,
        *,
        candidate: DetectionCandidate | None = None,
        request: CoprocessorRequest | None = None,
        engine: str | None = None,
        duration_ns: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            sequence=self._sequence,
            timestamp_ns=self._clock_ns(),
            stage=stage,
            status=status,
            candidate_id=candidate.candidate_id if candidate else None,
            request_id=request.request_id if request else None,
            engine=engine,
            duration_ns=duration_ns,
            details=dict(details or {}),
        )
        self._sequence += 1
        self.trace.append(event)
        return event

    def _handle(self, candidate: DetectionCandidate) -> Reinjection | None:
        self._record(
            TraceStage.DETECTION,
            TraceStatus.CANDIDATE,
            candidate=candidate,
            details={"raw_text": candidate.raw_text, "detector": candidate.detector},
        )
        started = self._timer_ns()
        try:
            request = self.normalizer.normalize(candidate)
        except Exception as error:  # noqa: BLE001 - extension boundary
            self._record(
                TraceStage.NORMALIZATION,
                TraceStatus.REJECTED,
                candidate=candidate,
                duration_ns=self._timer_ns() - started,
                details={"error_type": type(error).__name__, "error": str(error)},
            )
            return None

        self._record(
            TraceStage.NORMALIZATION,
            TraceStatus.ACCEPTED,
            candidate=candidate,
            request=request,
            duration_ns=self._timer_ns() - started,
            details={"operation": request.operation, "request": request.to_dict()},
        )
        engine = self.engines.get(request.engine)
        if engine is None:
            self._record(
                TraceStage.ROUTING,
                TraceStatus.FAILED,
                candidate=candidate,
                request=request,
                engine=request.engine,
                details={"error": "engine_not_registered"},
            )
            return None

        self._record(
            TraceStage.ROUTING,
            TraceStatus.SELECTED,
            candidate=candidate,
            request=request,
            engine=engine.name,
        )
        started = self._timer_ns()
        try:
            result = engine.execute(request)
        except Exception as error:  # noqa: BLE001 - extension boundary
            self._record(
                TraceStage.EXECUTION,
                TraceStatus.FAILED,
                candidate=candidate,
                request=request,
                engine=engine.name,
                duration_ns=self._timer_ns() - started,
                details={"error_type": type(error).__name__, "error": str(error)},
            )
            return None

        duration = self._timer_ns() - started
        self._record(
            TraceStage.EXECUTION,
            TraceStatus.SUCCEEDED if result.ok else TraceStatus.FAILED,
            candidate=candidate,
            request=request,
            engine=engine.name,
            duration_ns=duration,
            details={"result": result.to_dict()},
        )
        if not result.ok:
            return None

        state = MicroStateItem(
            state_id=f"{self.run_id}:state:{len(self.state)}",
            request=request,
            result=result,
            created_at_ns=self._clock_ns(),
            provenance={
                "candidate_id": candidate.candidate_id,
                "detector": candidate.detector,
                "engine": engine.name,
            },
        )
        self.state.append(state)
        self._record(
            TraceStage.STATE_UPDATE,
            TraceStatus.SUCCEEDED,
            candidate=candidate,
            request=request,
            engine=engine.name,
            details={"state_id": state.state_id, "state": state.to_dict()},
        )

        try:
            text = self.materializer.materialize(request, result, state)
        except Exception as error:  # noqa: BLE001 - extension boundary
            self._record(
                TraceStage.REINJECTION,
                TraceStatus.FAILED,
                candidate=candidate,
                request=request,
                engine=engine.name,
                details={"error_type": type(error).__name__, "error": str(error)},
            )
            return None
        self._record(
            TraceStage.REINJECTION,
            TraceStatus.SUCCEEDED,
            candidate=candidate,
            request=request,
            engine=engine.name,
            details={"state_id": state.state_id, "text": text},
        )
        return Reinjection(text=text, request_id=request.request_id, state_id=state.state_id)

    def feed(self, text: str) -> RuntimeStep:
        trace_start = len(self.trace)
        rendered_parts: list[str] = []
        reinjections: list[Reinjection] = []
        for character in text:
            rendered_parts.append(character)
            for candidate in self.detector.feed(character):
                reinjection = self._handle(candidate)
                if reinjection is not None:
                    reinjections.append(reinjection)
                    rendered_parts.append(reinjection.text)
        rendered = "".join(rendered_parts)
        self.rendered_text += rendered
        return RuntimeStep(
            source_text=text,
            rendered_text=rendered,
            reinjections=tuple(reinjections),
            new_trace_events=tuple(self.trace[trace_start:]),
        )
