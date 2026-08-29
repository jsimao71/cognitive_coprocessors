"""Typed read-only retrieval contracts shared by epistemic coprocessors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SourcePolicy:
    source_type: str
    source_id: str
    locality: str
    credential_scope: str
    latency_class: str
    cost_class: str
    privacy_class: str
    freshness: str
    side_effect_class: str = "read_only"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SourceRequest:
    request_id: str
    source_type: str
    operation: str
    payload: Mapping[str, Any]
    budget: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    request_id: str
    source_type: str
    source_id: str
    record_id: str
    status: str
    value: Any
    content: str
    observed_at: str
    relevance: float
    provenance: Mapping[str, Any]
    latency_ns: int
    bytes_retrieved: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceSource(Protocol):
    policy: SourcePolicy

    def retrieve(self, request: SourceRequest) -> tuple[EvidenceRecord, ...]: ...
