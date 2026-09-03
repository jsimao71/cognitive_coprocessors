"""Compatibility helpers for supported Python versions."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised on Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return self.value


__all__ = ["StrEnum"]
