"""Canonical serialization and signatures for F3 artifacts."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from ccpu.common.artifacts import fingerprint

from .ast import Form, Program, Value

_SPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return _SPACE.sub(" ", value.strip()).casefold()


def value_record(value: Value) -> Any:
    if isinstance(value, Form):
        return {"form": value.name, "args": [value_record(item) for item in value.args]}
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


def program_record(program: Program) -> dict[str, Any]:
    return {
        "schema_version": "ccpu.paper1.f3.ast.v1",
        "forms": [value_record(form) for form in program.forms],
    }


def semantic_signature(program: Program) -> str:
    """Hash structure while abstracting labels, paths, evidence text, and constants."""

    def shape(value: Value) -> Any:
        if isinstance(value, Form):
            if value.name in {"source", "cell"}:
                return (value.name, "EVIDENCE")
            if value.name == "at":
                return ("at", "PATH", str(value.args[1]))
            return (value.name, *(shape(item) for item in value.args))
        if isinstance(value, (int, Decimal)):
            return "CONST"
        return "SYMBOL"

    return f"f3-pattern-{fingerprint(repr(tuple(shape(form) for form in program.forms)), 16)}"
