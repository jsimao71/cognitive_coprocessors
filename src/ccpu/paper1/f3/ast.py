"""Immutable syntax records for the F3 language."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias

Scalar: TypeAlias = str | int | Decimal


@dataclass(frozen=True)
class Form:
    """One allowlisted F3 form, including nested reference forms."""

    name: str
    args: tuple[Value, ...]


Value: TypeAlias = Scalar | Form


@dataclass(frozen=True)
class Program:
    """A canonical sequence of F3 source assertions and one terminal query."""

    forms: tuple[Form, ...]

    @property
    def query(self) -> Form:
        return self.forms[-1]
