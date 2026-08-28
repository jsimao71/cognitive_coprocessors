"""Strict heterogeneous event routing and observable execution."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ccpu.common.schema import CoprocessorRequest, CoprocessorResult, DetectionCandidate
from ccpu.paper1.arithmetic import ArithmeticNormalizer, BoundedCalculator

from .graph import FrameGraphEngine
from .logic import HornEngine
from .state import StateLimitError, TypedMicroState

_EVENT = re.compile(r"^\[(calculator|horn|graph)\]\s+(.+)$", re.DOTALL)


class StrictEventRouter:
    def __init__(self, max_event_chars: int = 4096) -> None:
        self.max_event_chars = max_event_chars

    def parse(self, text: str, *, event_id: str) -> CoprocessorRequest | None:
        if len(text) > self.max_event_chars:
            raise ValueError("heterogeneous event exceeds character budget")
        match = _EVENT.fullmatch(text.strip())
        if match is None:
            return None
        family, body = match.groups()
        if family == "calculator":
            candidate = DetectionCandidate(
                candidate_id=event_id,
                family="compute",
                raw_text=body,
                start_offset=0,
                end_offset=len(text),
                detector="paper2.strict_event",
            )
            return ArithmeticNormalizer().normalize(candidate)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid strict event JSON: {error.msg}") from error
        if not isinstance(payload, dict):
            raise TypeError("strict event payload must be an object")
        operation = "horn.query" if family == "horn" else str(payload.pop("operation", "graph.isa"))
        return CoprocessorRequest(
            request_id=f"{event_id}:{family}",
            candidate_id=event_id,
            family="reasoning",
            operation=operation,
            engine="horn" if family == "horn" else "frame_graph",
            payload=payload,
            budget={"max_event_chars": self.max_event_chars},
            metadata={"detector": "paper2.strict_event"},
        )


class HeterogeneousRuntime:
    def __init__(self, *, enabled_engines: set[str] | None = None, max_state_items: int = 512) -> None:
        self.state = TypedMicroState(max_state_items)
        engines = (
            BoundedCalculator(),
            HornEngine(self.state),
            FrameGraphEngine(self.state),
        )
        enabled = enabled_engines or {engine.name for engine in engines}
        self.engines = {engine.name: engine for engine in engines if engine.name in enabled}
        self.router = StrictEventRouter()
        self.trace: list[dict[str, Any]] = []

    def execute_event(self, text: str, *, event_id: str) -> CoprocessorResult | None:
        started = time.perf_counter_ns()
        state_start = len(self.state.items)
        try:
            request = self.router.parse(text, event_id=event_id)
        except Exception as error:  # noqa: BLE001 - strict extension boundary
            self.trace.append(
                {
                    "event_id": event_id,
                    "stage": "normalization",
                    "status": "rejected",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            return None
        if request is None:
            self.trace.append({"event_id": event_id, "stage": "detection", "status": "no_event"})
            return None
        self.trace.append(
            {
                "event_id": event_id,
                "stage": "routing",
                "status": "selected" if request.engine in self.engines else "unavailable",
                "request": request.to_dict(),
            }
        )
        engine = self.engines.get(request.engine)
        if engine is None:
            return None
        result = engine.execute(request)
        duration = time.perf_counter_ns() - started
        self.trace.append(
            {
                "event_id": event_id,
                "stage": "execution",
                "status": "succeeded" if result.ok else "failed",
                "duration_ns": duration,
                "result": result.to_dict(),
            }
        )
        if result.ok:
            try:
                self.state.add(
                    "engine_result",
                    {"engine": result.engine, "display": result.display, "request_id": request.request_id},
                    provenance={"event_id": event_id},
                )
            except StateLimitError:
                return CoprocessorResult(
                    request_id=request.request_id,
                    engine=result.engine,
                    ok=False,
                    error_code="state_limit",
                    error_message="result could not be persisted",
                )
            for item in self.state.items[state_start:]:
                self.trace.append(
                    {
                        "event_id": event_id,
                        "stage": "state_update",
                        "status": "succeeded",
                        "state": item.to_dict(),
                    }
                )
        return result
