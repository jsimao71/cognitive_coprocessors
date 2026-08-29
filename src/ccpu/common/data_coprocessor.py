"""Reusable contracts for swappable read-only data coprocessors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .retrieval import SourcePolicy, SourceRequest


@dataclass(frozen=True)
class DataCoprocessorDescriptor:
    """Public capabilities plus runtime-owned backend and request metadata."""

    policy: SourcePolicy
    backend: str
    backend_version: str
    capabilities: tuple[str, ...]
    request_fields: Mapping[str, tuple[str, ...]]
    resources: tuple[str, ...]
    snapshot: str

    def public_dict(self) -> dict[str, Any]:
        policy = self.policy.to_dict()
        policy.pop("credential_scope")
        return {
            **policy,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "capabilities": list(self.capabilities),
            "request_fields": {
                operation: list(fields) for operation, fields in self.request_fields.items()
            },
            "resources": list(self.resources),
            "snapshot": self.snapshot,
        }

    def validate(self, request: SourceRequest) -> None:
        if request.source_type != self.policy.source_type:
            raise ValueError(
                f"request source {request.source_type!r} does not match "
                f"{self.policy.source_type!r}"
            )
        required = self.request_fields.get(request.operation)
        if required is None:
            raise ValueError(f"unsupported {self.policy.source_type} operation: {request.operation}")
        missing = [field for field in required if field not in request.payload]
        if missing:
            raise ValueError(f"missing required request fields: {', '.join(missing)}")
        max_records = request.budget.get("max_records")
        if max_records is not None and int(max_records) < 1:
            raise ValueError("max_records must be positive")


def production_provenance(
    descriptor: DataCoprocessorDescriptor,
    *,
    normalized_query: str,
    resource: str,
    record_ids: list[str],
    parameters: Mapping[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Build the minimum provenance envelope shared by production adapters."""

    return {
        "backend": descriptor.backend,
        "backend_version": descriptor.backend_version,
        "resource": resource,
        "normalized_query": normalized_query,
        "parameters": dict(parameters or {}),
        "record_ids": list(record_ids),
        "snapshot": descriptor.snapshot,
        **details,
    }
