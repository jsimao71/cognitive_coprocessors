"""Exact bounded unit conversion with dimensional validation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext

from ccpu.common.schema import CoprocessorRequest, CoprocessorResult

_CONVERT = re.compile(
    r"^convert\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+([A-Za-z]+)\s*->\s*([A-Za-z]+)$",
    re.IGNORECASE,
)
_UNITS: dict[str, tuple[str, Decimal]] = {
    "meter": ("length", Decimal(1)),
    "kilometer": ("length", Decimal(1000)),
    "centimeter": ("length", Decimal("0.01")),
    "mile": ("length", Decimal("1609.344")),
    "foot": ("length", Decimal("0.3048")),
    "kilogram": ("mass", Decimal(1)),
    "gram": ("mass", Decimal("0.001")),
    "pound": ("mass", Decimal("0.45359237")),
    "second": ("time", Decimal(1)),
    "minute": ("time", Decimal(60)),
    "hour": ("time", Decimal(3600)),
}
_ALIASES = {
    "m": "meter",
    "km": "kilometer",
    "cm": "centimeter",
    "mi": "mile",
    "ft": "foot",
    "kg": "kilogram",
    "g": "gram",
    "lb": "pound",
    "s": "second",
    "min": "minute",
    "h": "hour",
}


def _unit(value: str) -> str:
    normalized = value.lower()
    if normalized.endswith("s") and normalized[:-1] in _UNITS:
        normalized = normalized[:-1]
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _UNITS:
        raise ValueError(f"unsupported unit: {value}")
    return normalized


def normalize_units_payload(text: str) -> dict[str, str]:
    match = _CONVERT.fullmatch(text.strip())
    if not match:
        raise ValueError("units block must be 'convert NUMBER UNIT -> UNIT'")
    value, source, target = match.groups()
    if len(value.replace(".", "").lstrip("+-")) > 32:
        raise ValueError("numeric literal exceeds digit budget")
    source = _unit(source)
    target = _unit(target)
    if _UNITS[source][0] != _UNITS[target][0]:
        raise ValueError("source and target dimensions differ")
    try:
        Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal value") from error
    return {"value": value, "source_unit": source, "target_unit": target}


class UnitsEngine:
    name = "units"

    def execute(self, request: CoprocessorRequest) -> CoprocessorResult:
        try:
            if request.operation != "units.convert":
                raise ValueError(f"unsupported operation: {request.operation}")
            value = Decimal(str(request.payload["value"]))
            source = _unit(str(request.payload["source_unit"]))
            target = _unit(str(request.payload["target_unit"]))
            if _UNITS[source][0] != _UNITS[target][0]:
                raise ValueError("source and target dimensions differ")
            with localcontext() as context:
                context.prec = 40
                converted = value * _UNITS[source][1] / _UNITS[target][1]
            display = format(converted.normalize(), "f")
            if "." in display:
                display = display.rstrip("0").rstrip(".")
            return CoprocessorResult(
                request_id=request.request_id,
                engine=self.name,
                ok=True,
                value=display,
                display=display,
                metadata={"dimension": _UNITS[source][0], "exact_decimal": True},
            )
        except (KeyError, InvalidOperation, TypeError, ValueError) as error:
            return CoprocessorResult(
                request_id=request.request_id,
                engine=self.name,
                ok=False,
                error_code="invalid_or_bounded_ir",
                error_message=str(error),
            )
