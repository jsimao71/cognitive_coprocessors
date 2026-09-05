"""ASL-Arith semantic operator registry."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

ArithmeticFunction = Callable[[list[Decimal]], Decimal]


def _require(arguments: list[Decimal], count: int, name: str) -> None:
    if len(arguments) != count:
        raise ValueError(f"{name} expects {count} arguments, got {len(arguments)}")


def _percent_of(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "percent_of")
    return arguments[0] * arguments[1] / Decimal(100)


def _absolute(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 1, "abs")
    return abs(arguments[0])


def _rate_times_duration(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "rate_times_duration")
    return arguments[0] * arguments[1]


def _inc_pct(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "inc_pct")
    return arguments[0] * (Decimal(1) + arguments[1] / Decimal(100))


def _dec_pct(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "dec_pct")
    return arguments[0] * (Decimal(1) - arguments[1] / Decimal(100))


def _mean(arguments: list[Decimal]) -> Decimal:
    if not arguments:
        raise ValueError("mean expects at least one argument")
    return sum(arguments) / len(arguments)


ARITHMETIC_FUNCTIONS: dict[str, ArithmeticFunction] = {
    "abs": _absolute,
    "dec_pct": _dec_pct,
    "decrease_by_percent": _dec_pct,
    "inc_pct": _inc_pct,
    "increase_by_percent": _inc_pct,
    "max": max,
    "mean": _mean,
    "min": min,
    "percent_of": _percent_of,
    "rate_times_duration": _rate_times_duration,
    "sum": sum,
}

CCIR_CALL_OPERATORS = {
    "abs": "ABS",
    "dec_pct": "DECREASE_BY_PERCENT",
    "decrease_by_percent": "DECREASE_BY_PERCENT",
    "inc_pct": "INCREASE_BY_PERCENT",
    "increase_by_percent": "INCREASE_BY_PERCENT",
    "max": "MAX",
    "mean": "MEAN",
    "min": "MIN",
    "percent_of": "PERCENT_OF",
    "rate_times_duration": "RATE_TIMES_DURATION",
    "sum": "SUM",
}
