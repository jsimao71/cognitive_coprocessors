"""Bounded append-only typed micro-state shared by Paper 2 engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ccpu.common.artifacts import fingerprint


@dataclass(frozen=True)
class KnowledgeItem:
    state_id: str
    kind: str
    payload: dict[str, Any]
    dependencies: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateLimitError(RuntimeError):
    pass


class TypedMicroState:
    def __init__(self, max_items: int = 512) -> None:
        self.max_items = max_items
        self.items: list[KnowledgeItem] = []
        self._identities: dict[str, KnowledgeItem] = {}

    def add(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        dependencies: tuple[str, ...] = (),
        provenance: dict[str, Any] | None = None,
    ) -> KnowledgeItem:
        identity = fingerprint({"kind": kind, "payload": payload})
        if identity in self._identities:
            return self._identities[identity]
        if len(self.items) >= self.max_items:
            raise StateLimitError("typed micro-state item budget exceeded")
        item = KnowledgeItem(
            state_id=f"state:{len(self.items)}:{identity[:12]}",
            kind=kind,
            payload=dict(payload),
            dependencies=dependencies,
            provenance=dict(provenance or {}),
        )
        self.items.append(item)
        self._identities[identity] = item
        return item

    def by_kind(self, kind: str) -> tuple[KnowledgeItem, ...]:
        return tuple(item for item in self.items if item.kind == kind)
