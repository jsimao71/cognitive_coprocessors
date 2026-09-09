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


def _square(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 1, "square")
    return arguments[0] * arguments[0]


def _cube(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 1, "cube")
    return arguments[0] * arguments[0] * arguments[0]


def _exact_sqrt(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 1, "sqrt")
    value = arguments[0]
    if value < 0:
        raise ValueError("sqrt requires a non-negative argument")
    root = value.sqrt()
    if root * root != value:
        raise ValueError("sqrt is restricted to exact roots in operator registry v1")
    return root


def _subtract(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "difference")
    return arguments[0] - arguments[1]


def _add(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "total_of")
    return arguments[0] + arguments[1]


def _multiply(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "product")
    return arguments[0] * arguments[1]


def _divide(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 2, "quotient")
    if arguments[1] == 0:
        raise ValueError("quotient requires a non-zero divisor")
    return arguments[0] / arguments[1]


_SIN_DEGREES = {
    Decimal(0): Decimal(0),
    Decimal(30): Decimal("0.5"),
    Decimal(45): Decimal("0.7071067811865475244008443621"),
    Decimal(60): Decimal("0.8660254037844386467637231708"),
    Decimal(90): Decimal(1),
}
_COS_DEGREES = {
    Decimal(0): Decimal(1),
    Decimal(30): _SIN_DEGREES[Decimal(60)],
    Decimal(45): _SIN_DEGREES[Decimal(45)],
    Decimal(60): Decimal("0.5"),
    Decimal(90): Decimal(0),
}


def _common_angle(arguments: list[Decimal], values: dict[Decimal, Decimal], name: str) -> Decimal:
    _require(arguments, 1, name)
    try:
        return values[arguments[0]]
    except KeyError as error:
        raise ValueError(f"{name} is restricted to frozen common angles") from error


def _sin_degrees(arguments: list[Decimal]) -> Decimal:
    return _common_angle(arguments, _SIN_DEGREES, "sin_degrees")


def _cos_degrees(arguments: list[Decimal]) -> Decimal:
    return _common_angle(arguments, _COS_DEGREES, "cos_degrees")


def _exact_log10(arguments: list[Decimal]) -> Decimal:
    _require(arguments, 1, "log10")
    value = arguments[0]
    if value <= 0:
        raise ValueError("log10 requires a positive argument")
    exponent = 0
    while value > 1 and value % 10 == 0:
        value /= 10
        exponent += 1
    if value != 1:
        raise ValueError("log10 is restricted to exact positive powers of ten")
    return Decimal(exponent)


ARITHMETIC_FUNCTIONS: dict[str, ArithmeticFunction] = {
    "abs": _absolute,
    "dec_pct": _dec_pct,
    "decrease_by_percent": _dec_pct,
    "inc_pct": _inc_pct,
    "increase_by_percent": _inc_pct,
    "max": max,
    "mean": _mean,
    "min": min,
    "cube": _cube,
    "sqrt": _exact_sqrt,
    "square": _square,
    "area_of_square": _square,
    "side_of_square": _exact_sqrt,
    "volume_of_cube": _cube,
    "difference": _subtract,
    "total_of": _add,
    "product": _multiply,
    "quotient": _divide,
    "sin_degrees": _sin_degrees,
    "cos_degrees": _cos_degrees,
    "log10": _exact_log10,
    "vertical_component_of_unit": _sin_degrees,
    "horizontal_component_of_unit": _cos_degrees,
    "decimal_order": _exact_log10,
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
    "cube": "CUBE",
    "sqrt": "SQRT",
    "square": "SQUARE",
    "area_of_square": "SQUARE",
    "side_of_square": "SQRT",
    "volume_of_cube": "CUBE",
    "difference": "SUB",
    "total_of": "ADD",
    "product": "MUL",
    "quotient": "DIV",
    "sin_degrees": "SIN_DEGREES",
    "cos_degrees": "COS_DEGREES",
    "log10": "LOG10",
    "vertical_component_of_unit": "SIN_DEGREES",
    "horizontal_component_of_unit": "COS_DEGREES",
    "decimal_order": "LOG10",
    "percent_of": "PERCENT_OF",
    "rate_times_duration": "RATE_TIMES_DURATION",
    "sum": "SUM",
}
