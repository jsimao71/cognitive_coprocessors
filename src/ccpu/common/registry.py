"""Small fail-closed registry for typed coprocessor engines."""

from __future__ import annotations

from collections.abc import Iterable

from .interfaces import Coprocessor


class CoprocessorRegistry:
    """Register engines by stable name without benchmark-specific routing."""

    def __init__(self, engines: Iterable[Coprocessor] = ()) -> None:
        self._engines: dict[str, Coprocessor] = {}
        for engine in engines:
            self.register(engine)

    def register(self, engine: Coprocessor) -> None:
        if not engine.name or engine.name in self._engines:
            raise ValueError(f"duplicate or empty coprocessor name: {engine.name!r}")
        self._engines[engine.name] = engine

    def get(self, name: str) -> Coprocessor | None:
        return self._engines.get(name)

    def select(self, names: set[str] | None = None) -> dict[str, Coprocessor]:
        selected = set(self._engines) if names is None else names
        unknown = selected.difference(self._engines)
        if unknown:
            raise ValueError(f"unknown coprocessor engines: {sorted(unknown)}")
        return {name: self._engines[name] for name in sorted(selected)}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._engines))
