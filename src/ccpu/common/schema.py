"""Serializable contracts shared by all cognitive-coprocessor papers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class TraceStage(StrEnum):
    DETECTION = "detection"
    NORMALIZATION = "normalization"
    ROUTING = "routing"
    EXECUTION = "execution"
    STATE_UPDATE = "state_update"
    REINJECTION = "reinjection"


class TraceStatus(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    SELECTED = "selected"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class DetectionCandidate:
    """A detector-owned source span that may normalize into typed micro-IR."""

    candidate_id: str
    family: str
    raw_text: str
    start_offset: int
    end_offset: int
    detector: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoprocessorRequest:
    """Canonical request envelope; each engine owns its typed payload schema."""

    request_id: str
    candidate_id: str
    family: str
    operation: str
    engine: str
    payload: Mapping[str, Any]
    confidence: float = 1.0
    budget: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoprocessorResult:
    """Typed engine outcome. Failures are values so they remain traceable."""

    request_id: str
    engine: str
    ok: bool
    value: Any = None
    display: str = ""
    error_code: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MicroStateItem:
    """Append-only state item used before later transactional-state papers."""

    state_id: str
    request: CoprocessorRequest
    result: CoprocessorResult
    created_at_ns: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    """One observable transition through the interrupt pipeline."""

    run_id: str
    sequence: int
    timestamp_ns: int
    stage: TraceStage
    status: TraceStatus
    candidate_id: str | None = None
    request_id: str | None = None
    engine: str | None = None
    duration_ns: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["stage"] = self.stage.value
        row["status"] = self.status.value
        return row


@dataclass(frozen=True)
class Reinjection:
    text: str
    request_id: str
    state_id: str


@dataclass(frozen=True)
class RuntimeStep:
    """Rendered output and interventions produced while consuming one fragment."""

    source_text: str
    rendered_text: str
    reinjections: tuple[Reinjection, ...]
    new_trace_events: tuple[TraceEvent, ...]


@dataclass(frozen=True)
class GenerationResult:
    """Backend-neutral generation output and cost accounting."""

    generated_text: str
    rendered_text: str
    prompt_tokens: int
    generated_tokens: int
    reinjected_tokens: int
    model_calls: int
    wall_time_ns: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
