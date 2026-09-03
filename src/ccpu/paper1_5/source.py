"""Versioned single-source retrieval with typed evidence outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ccpu.common.artifacts import fingerprint
from ccpu.common.compat import StrEnum
from ccpu.common.schema import CoprocessorRequest, CoprocessorResult


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    STALE = "stale"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class FactRecord:
    record_id: str
    entity: str
    attribute: str
    value: str
    valid_from: str
    valid_to: str | None = None
    available: bool = True

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> FactRecord:
        return cls(**row)


@dataclass(frozen=True)
class Evidence:
    status: EvidenceStatus
    values: tuple[str, ...]
    record_ids: tuple[str, ...]
    source_id: str
    source_version: str
    as_of: str

    @property
    def answer(self) -> str | None:
        return self.values[0] if len(self.values) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    def render(self) -> str:
        values = ", ".join(self.values) if self.values else "none"
        records = ", ".join(self.record_ids) if self.record_ids else "none"
        return (
            f"STATUS={self.status.value.upper()}; VALUE={values}; "
            f"SOURCE={self.source_id}@{self.source_version}; RECORDS={records}; AS_OF={self.as_of}"
        )


class ControlledFactStore:
    name = "controlled_fact_store"

    def __init__(
        self,
        *,
        source_id: str,
        version: str,
        records: tuple[FactRecord, ...],
    ) -> None:
        self.source_id = source_id
        self.version = version
        self.records = records

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ControlledFactStore:
        return cls(
            source_id=str(raw["source_id"]),
            version=str(raw["version"]),
            records=tuple(FactRecord.from_dict(row) for row in raw["records"]),
        )

    def request(
        self,
        *,
        example_id: str,
        entity: str,
        attribute: str,
        as_of: str,
        forecast: str,
        candidate_answer: str | None = None,
    ) -> CoprocessorRequest:
        payload = {
            "entity": entity,
            "attribute": attribute,
            "as_of": as_of,
            "forecast": forecast,
            "candidate_answer": candidate_answer,
            "source_version": self.version,
        }
        return CoprocessorRequest(
            request_id=f"retrieval:{fingerprint({'example': example_id, **payload}, 16)}",
            candidate_id=f"forecast:{example_id}",
            family="retrieval",
            operation="lookup_versioned_fact",
            engine=self.name,
            payload=payload,
            budget={"max_records": 8, "max_evidence_chars": 512},
        )

    def execute(self, request: CoprocessorRequest) -> CoprocessorResult:
        payload = request.payload
        entity = str(payload["entity"]).casefold()
        attribute = str(payload["attribute"]).casefold()
        as_of = str(payload["as_of"])
        candidate = str(payload.get("candidate_answer") or "").strip().casefold()
        matched = [
            record
            for record in self.records
            if record.available
            and record.entity.casefold() == entity
            and record.attribute.casefold() == attribute
        ]
        active = [
            record
            for record in matched
            if record.valid_from <= as_of
            and (record.valid_to is None or as_of <= record.valid_to)
        ]
        values = tuple(dict.fromkeys(record.value for record in active))
        if len(values) > 1:
            status = EvidenceStatus.CONFLICT
        elif not values:
            status = EvidenceStatus.UNVERIFIED
        elif not candidate or candidate == values[0].casefold():
            status = EvidenceStatus.SUPPORTED
        elif any(record.value.casefold() == candidate for record in matched if record not in active):
            status = EvidenceStatus.STALE
        else:
            status = EvidenceStatus.CONTRADICTED
        evidence = Evidence(
            status=status,
            values=values,
            record_ids=tuple(record.record_id for record in active),
            source_id=self.source_id,
            source_version=self.version,
            as_of=as_of,
        )
        return CoprocessorResult(
            request_id=request.request_id,
            engine=self.name,
            ok=True,
            value=evidence.to_dict(),
            display=evidence.render(),
            metadata={"record_count": len(active)},
        )
