"""Controlled-source benchmark loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ccpu.common.artifacts import fingerprint

from .source import ControlledFactStore


@dataclass(frozen=True)
class RetrievalExample:
    example_id: str
    split: str
    question: str
    answer: str
    entity: str
    attribute: str
    as_of: str
    evidence_required: bool
    category: str
    retrieval_subclass: str | None = None
    design_group: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> RetrievalExample:
        payload = dict(row)
        payload.setdefault(
            "example_id",
            fingerprint({key: value for key, value in payload.items() if key != "example_id"}, 16),
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_benchmark(raw: dict[str, Any]) -> tuple[ControlledFactStore, list[RetrievalExample]]:
    store = ControlledFactStore.from_dict(raw["source"])
    examples = [RetrievalExample.from_dict(row) for row in raw["examples"]]
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Paper 1.5 example IDs must be unique")
    if not {"dev", "test"}.issubset({example.split for example in examples}):
        raise ValueError("Paper 1.5 benchmark requires dev and test splits")
    return store, examples
