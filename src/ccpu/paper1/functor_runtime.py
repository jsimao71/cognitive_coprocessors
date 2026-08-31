"""Strict functor grammars and deterministic lowering for Paper 1."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ccpu.dsl import validate_asl

_PATH = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*$")
_FENCE = re.compile(r"```(?:functor|text|python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Any, ...]


_F1_ARITY: dict[str, tuple[int, int | None]] = {
    "set": (2, 2),
    "query": (1, 1),
    "const": (1, 1),
    "ref": (1, 1),
    "add": (2, None),
    "sub": (2, 2),
    "mul": (2, None),
    "div": (2, 2),
    "abs": (1, 1),
    "sum": (1, None),
    "mean": (1, None),
    "min": (1, None),
    "max": (1, None),
    "percent_of": (2, 2),
    "increase_percent": (2, 2),
    "decrease_percent": (2, 2),
    "rate_times_duration": (2, 2),
}

_F2_ARITY: dict[str, tuple[int, int | None]] = {
    "given": (2, 2),
    "same": (2, 2),
    "offset": (3, 3),
    "difference": (3, 3),
    "absolute_difference": (3, 3),
    "sum_of": (3, None),
    "product_of": (3, None),
    "quotient": (3, 3),
    "multiple": (3, 3),
    "fraction_of": (4, 4),
    "percent_of": (3, 3),
    "percentage_ratio": (3, 3),
    "increase_percent": (3, 3),
    "decrease_percent": (3, 3),
    "rate_total": (3, 3),
    "per_unit_total": (3, 3),
    "remaining": (3, None),
    "mean_of": (3, None),
    "minimum_of": (3, None),
    "maximum_of": (3, None),
    "query": (1, 1),
}


def _value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("functors require a simple name and positional arguments")
        return Call(node.func.id, tuple(_value(argument) for argument in node.args))
    raise ValueError(f"unsupported functor value: {ast.dump(node, include_attributes=False)}")


def _check_arity(call: Call, registry: dict[str, tuple[int, int | None]]) -> None:
    if call.name not in registry:
        raise ValueError(f"unsupported functor: {call.name}")
    minimum, maximum = registry[call.name]
    count = len(call.args)
    if count < minimum or (maximum is not None and count > maximum):
        expected = str(minimum) if minimum == maximum else f"{minimum}..{maximum or '*'}"
        raise ValueError(f"{call.name} expects {expected} arguments, got {count}")


def parse_functor_program(program: str, condition: str) -> list[Call]:
    """Parse one-call-per-line F1 or F2 without evaluating Python."""

    registry = _F1_ARITY if condition == "f1" else _F2_ARITY if condition == "f2" else None
    if registry is None:
        raise ValueError(f"unsupported functor condition: {condition}")
    calls = []
    for line_number, raw_line in enumerate(program.splitlines(), 1):
        line = raw_line.strip().rstrip(";")
        if not line or line.startswith("#"):
            continue
        try:
            expression = ast.parse(line, mode="eval").body
            call = _value(expression)
        except (SyntaxError, TypeError, ValueError) as error:
            raise ValueError(f"line {line_number}: {error}") from error
        if not isinstance(call, Call):
            raise TypeError(f"line {line_number}: expected one functor call")
        _check_arity(call, registry)
        calls.append(call)
    if not calls:
        raise ValueError("program contains no functor calls")
    if calls[-1].name != "query" or any(call.name == "query" for call in calls[:-1]):
        raise ValueError("program must end with exactly one query(path)")
    return calls


def _path(value: Any) -> str:
    if not isinstance(value, str) or not _PATH.fullmatch(value):
        raise ValueError(f"invalid semantic path: {value!r}")
    return value


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric literal, got {value!r}")
    decimal = Decimal(str(value))
    return (
        format(decimal, "f").rstrip("0").rstrip(".")
        if decimal != decimal.to_integral()
        else str(int(decimal))
    )


def _operand(value: Any) -> str:
    if isinstance(value, str):
        return _path(value)
    return _number(value)


def _f1_expression(value: Any) -> str:
    if not isinstance(value, Call):
        raise TypeError("F1 set values must use const/ref/operator functors")
    _check_arity(value, _F1_ARITY)
    name, args = value.name, value.args
    if name == "const":
        return _number(args[0])
    if name == "ref":
        return _path(args[0])
    if name in {"add", "mul"}:
        operator = "+" if name == "add" else "*"
        return "(" + f" {operator} ".join(_f1_expression(arg) for arg in args) + ")"
    if name in {"sub", "div"}:
        operator = "-" if name == "sub" else "/"
        return f"({_f1_expression(args[0])} {operator} {_f1_expression(args[1])})"
    function = {
        "abs": "abs",
        "sum": "sum",
        "mean": "mean",
        "min": "min",
        "max": "max",
        "percent_of": "percent_of",
        "increase_percent": "inc_pct",
        "decrease_percent": "dec_pct",
        "rate_times_duration": "rate_times_duration",
    }.get(name)
    if function:
        return f"{function}({', '.join(_f1_expression(arg) for arg in args)})"
    raise ValueError(f"{name} is not valid inside set")


def lower_f1(calls: list[Call]) -> str:
    lines = []
    for call in calls:
        if call.name == "set":
            lines.append(f"{_path(call.args[0])} = {_f1_expression(call.args[1])}")
        elif call.name == "query":
            lines.append(f"RETURN {_path(call.args[0])}")
        else:
            raise ValueError(f"top-level F1 functor must be set/query, got {call.name}")
    return "\n".join(lines)


def _join(operator: str, values: tuple[Any, ...]) -> str:
    return "(" + f" {operator} ".join(_operand(value) for value in values) + ")"


def lower_f2(calls: list[Call]) -> str:
    """Lower semantic relation functors into the existing typed ASL runtime."""

    lines = []
    for call in calls:
        name, args = call.name, call.args
        if name == "query":
            lines.append(f"RETURN {_path(args[0])}")
            continue
        target = _path(args[0])
        if name == "given":
            expression = _number(args[1])
        elif name == "same":
            expression = _operand(args[1])
        elif name == "offset":
            delta = args[2]
            if not isinstance(delta, (int, float)) or isinstance(delta, bool):
                raise ValueError("offset delta must be a signed numeric literal")
            operator = "+" if delta >= 0 else "-"
            expression = f"({_operand(args[1])} {operator} {_number(abs(delta))})"
        elif name == "difference":
            expression = f"({_operand(args[1])} - {_operand(args[2])})"
        elif name == "absolute_difference":
            expression = f"abs({_operand(args[1])} - {_operand(args[2])})"
        elif name == "sum_of":
            expression = _join("+", args[1:])
        elif name == "product_of":
            expression = _join("*", args[1:])
        elif name == "quotient":
            expression = f"({_operand(args[1])} / {_operand(args[2])})"
        elif name in {"multiple", "per_unit_total", "rate_total"}:
            expression = f"({_operand(args[1])} * {_operand(args[2])})"
        elif name == "fraction_of":
            expression = f"({_operand(args[1])} * {_operand(args[2])} / {_operand(args[3])})"
        elif name == "percent_of":
            expression = f"percent_of({_operand(args[1])}, {_operand(args[2])})"
        elif name == "percentage_ratio":
            expression = f"({_operand(args[1])} / {_operand(args[2])} * 100)"
        elif name == "increase_percent":
            expression = f"inc_pct({_operand(args[1])}, {_operand(args[2])})"
        elif name == "decrease_percent":
            expression = f"dec_pct({_operand(args[1])}, {_operand(args[2])})"
        elif name == "remaining":
            expression = f"({_operand(args[1])} - {_join('+', args[2:])})"
        elif name in {"mean_of", "minimum_of", "maximum_of"}:
            function = {"mean_of": "mean", "minimum_of": "min", "maximum_of": "max"}[name]
            expression = f"{function}({', '.join(_operand(value) for value in args[1:])})"
        else:
            raise ValueError(f"cannot lower F2 functor: {name}")
        lines.append(f"{target} = {expression}")
    return "\n".join(lines)


def lower_functor_program(program: str, condition: str) -> str:
    calls = parse_functor_program(program, condition)
    return lower_f1(calls) if condition == "f1" else lower_f2(calls)


def validate_functor_program(
    program: str, condition: str, *, effective_scope: dict[str, Any]
) -> dict[str, Any]:
    """Parse, lower, type-check, and execute a functor program deterministically."""

    result = {
        "parse_valid": False,
        "lowerable": False,
        "type_valid": False,
        "executable": False,
        "lowered_asl": "",
        "errors": [],
    }
    try:
        calls = parse_functor_program(program, condition)
        result["parse_valid"] = True
        lowered = lower_f1(calls) if condition == "f1" else lower_f2(calls)
        result["lowered_asl"] = lowered
        result["lowerable"] = True
        validation = validate_asl(lowered, effective_scope=effective_scope)
        result["type_valid"] = bool(validation["type_verified"])
        result["executable"] = bool(validation["execution_verified"])
        result["errors"].extend(validation["errors"])
        result["validation"] = validation
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        result["errors"].append(str(error))
    return result


def extract_functor_program(text: str, condition: str) -> str:
    """Extract the first contiguous valid-looking call block from model output."""

    candidates = [match.strip() for match in _FENCE.findall(text) if match.strip()]
    candidates.append(text.strip())
    for candidate in candidates:
        lines = []
        for raw_line in candidate.splitlines():
            line = raw_line.strip().rstrip(";")
            if re.fullmatch(r"[a-z_][a-z0-9_]*\(.*\)", line):
                lines.append(line)
            elif lines and line:
                break
        program = "\n".join(lines)
        if program:
            try:
                parse_functor_program(program, condition)
                return program
            except ValueError:
                continue
    return ""
