"""Strict heterogeneous event routing and observable execution."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ccpu.common.registry import CoprocessorRegistry
from ccpu.common.schema import CoprocessorRequest, CoprocessorResult, DetectionCandidate
from ccpu.paper1.arithmetic import ArithmeticNormalizer, BoundedCalculator

from .date_time import DateTimeEngine, normalize_date_payload
from .graph import FrameGraphEngine
from .logic import HornEngine
from .state import StateLimitError, TypedMicroState
from .units import UnitsEngine, normalize_units_payload

_LEGACY_EVENT = re.compile(r"^\[(calculator|horn|graph)\]\s+(.+)$", re.DOTALL)
_BLOCK = re.compile(
    r"^```(calculator|datalog|graph|date|units)\r?\n(.+?)\r?\n```$",
    re.DOTALL,
)
_ATOM = re.compile(r"^([A-Za-z_]\w*)\(([^()]*)\)$")


def _atom(text: str) -> dict[str, Any]:
    match = _ATOM.fullmatch(text.strip())
    if not match:
        raise ValueError(f"invalid Datalog atom: {text}")
    predicate, arguments = match.groups()
    values = [value.strip() for value in arguments.split(",")]
    if not all(values):
        raise ValueError("Datalog atoms require non-empty arguments")
    return {"predicate": predicate, "arguments": values}


def _datalog_payload(text: str) -> dict[str, Any]:
    facts = []
    query = None
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        if line.startswith("fact "):
            facts.append(_atom(line.removeprefix("fact ")))
        elif line.startswith("query ") and query is None:
            query = _atom(line.removeprefix("query "))
        else:
            raise ValueError(f"unsupported Datalog statement: {line}")
    if query is None:
        raise ValueError("Datalog block requires exactly one query")
    predicates = {str(fact["predicate"]) for fact in facts}
    rules = []
    if query["predicate"] == "reachable" and "link" in predicates:
        rules = [
            {
                "head": {"predicate": "reachable", "arguments": ["?x", "?y"]},
                "body": [{"predicate": "link", "arguments": ["?x", "?y"]}],
            },
            {
                "head": {"predicate": "reachable", "arguments": ["?x", "?z"]},
                "body": [
                    {"predicate": "reachable", "arguments": ["?x", "?y"]},
                    {"predicate": "link", "arguments": ["?y", "?z"]},
                ],
            },
        ]
    return {"facts": facts, "rules": rules, "query": query}


def _graph_payload(text: str) -> tuple[str, dict[str, Any]]:
    edges = []
    frames = []
    operation = None
    query = None
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        parts = line.split()
        if len(parts) == 3 and parts[0] == "isa":
            edges.append(parts[1:])
        elif len(parts) == 4 and parts[0] == "frame":
            frames.append(parts[1:])
        elif len(parts) == 4 and parts[:2] == ["query", "isa"]:
            operation, query = "graph.isa", parts[2:]
        elif len(parts) == 4 and parts[:2] == ["query", "frame"]:
            operation, query = "graph.frame", parts[2:]
        else:
            raise ValueError(f"unsupported graph statement: {line}")
    if operation is None or query is None:
        raise ValueError("graph block requires one typed query")
    return operation, {"isa": edges, "frames": frames, "query": query}


class StrictEventRouter:
    def __init__(self, max_event_chars: int = 4096) -> None:
        self.max_event_chars = max_event_chars

    def parse(self, text: str, *, event_id: str) -> CoprocessorRequest | None:
        if len(text) > self.max_event_chars:
            raise ValueError("heterogeneous event exceeds character budget")
        stripped = text.strip()
        block = _BLOCK.fullmatch(stripped)
        legacy = _LEGACY_EVENT.fullmatch(stripped)
        if block is None and legacy is None:
            return None
        family, body = (block or legacy).groups()
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
        if block is not None:
            if family == "datalog":
                operation, engine, payload = "horn.query", "horn", _datalog_payload(body)
            elif family == "graph":
                operation, payload = _graph_payload(body)
                engine = "frame_graph"
            elif family == "date":
                operation, payload = normalize_date_payload(body)
                engine = "date_time"
            elif family == "units":
                operation, engine = "units.convert", "units"
                payload = normalize_units_payload(body)
            else:
                raise ValueError(f"unsupported block family: {family}")
            return CoprocessorRequest(
                request_id=f"{event_id}:{engine}",
                candidate_id=event_id,
                family="compute" if family in {"date", "units"} else "reasoning",
                operation=operation,
                engine=engine,
                payload=payload,
                budget={"max_event_chars": self.max_event_chars},
                metadata={"detector": "paper2.strict_block", "block_family": family},
            )
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
        registry = CoprocessorRegistry(
            (
                BoundedCalculator(),
                HornEngine(self.state),
                FrameGraphEngine(self.state),
                DateTimeEngine(),
                UnitsEngine(),
            )
        )
        self.engines = registry.select(enabled_engines)
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
