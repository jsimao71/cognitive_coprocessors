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


def _canonical_path(value: str) -> str:
    parts = value.split(".")
    canonical = [f"y{part}" if part and part[0].isdigit() else part for part in parts]
    path = ".".join(canonical)
    if not _PATH.fullmatch(path):
        raise ValueError(f"invalid semantic path: {value!r}")
    return path


_F1_ARITY: dict[str, tuple[int, int | None]] = {
    "value": (2, 2),
    "copy": (2, 2),
    "add": (3, None),
    "subtract": (3, None),
    "multiply": (3, None),
    "divide": (3, 3),
    "absolute": (2, 2),
    "sum_values": (2, None),
    "mean_values": (2, None),
    "minimum": (2, None),
    "maximum": (2, None),
    "percent_of": (3, 3),
    "increase_percent": (3, 3),
    "decrease_percent": (3, 3),
    "rate_times_duration": (3, 3),
    "query": (1, 1),
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


def functor_registry(condition: str) -> dict[str, tuple[int, int | None]]:
    """Return a copy of the allowlisted functor vocabulary and arities."""

    if condition == "f1":
        return dict(_F1_ARITY)
    if condition == "f2":
        return dict(_F2_ARITY)
    raise ValueError(f"unsupported functor condition: {condition}")


def _value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _canonical_path(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.left.value, (int, float))
        and isinstance(node.right.value, (int, float))
    ):
        denominator = Decimal(str(node.right.value))
        if denominator == 0:
            raise ZeroDivisionError("rational literal denominator is zero")
        return Decimal(str(node.left.value)) / denominator
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
    if not isinstance(value, str):
        raise TypeError(f"invalid semantic path: {value!r}")
    return _canonical_path(value)


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
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


def lower_f1(calls: list[Call]) -> str:
    lines = []
    for call in calls:
        name, args = call.name, call.args
        if name == "query":
            lines.append(f"RETURN {_path(call.args[0])}")
            continue
        target = _path(args[0])
        if name == "value":
            expression = _number(args[1])
        elif name == "copy":
            expression = _path(args[1])
        elif name in {"add", "multiply"}:
            expression = _join("+" if name == "add" else "*", args[1:])
        elif name == "subtract":
            expression = _join("-", args[1:])
        elif name == "divide":
            expression = f"({_operand(args[1])} / {_operand(args[2])})"
        elif name == "absolute":
            expression = f"abs({_operand(args[1])})"
        elif name in {"sum_values", "mean_values", "minimum", "maximum"}:
            function = {
                "sum_values": "sum",
                "mean_values": "mean",
                "minimum": "min",
                "maximum": "max",
            }[name]
            expression = f"{function}({', '.join(_operand(value) for value in args[1:])})"
        elif name in {"percent_of", "increase_percent", "decrease_percent"}:
            function = {
                "percent_of": "percent_of",
                "increase_percent": "inc_pct",
                "decrease_percent": "dec_pct",
            }[name]
            expression = f"{function}({_operand(args[1])}, {_operand(args[2])})"
        elif name == "rate_times_duration":
            expression = f"rate_times_duration({_operand(args[1])}, {_operand(args[2])})"
        else:
            raise ValueError(f"cannot lower F1 functor: {name}")
        lines.append(f"{target} = {expression}")
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


def _unknown_paths(args: tuple[Any, ...], known: set[str]) -> list[str]:
    return list(
        dict.fromkeys(value for value in args if isinstance(value, str) and value not in known)
    )


def _inverse_f2_expression(call: Call, unknown: str) -> str | None:
    """Return an ASL expression for the sole unknown in one semantic equation."""

    name, args = call.name, call.args
    target = _path(args[0])
    operands = args[1:]
    if unknown == target:
        return lower_f2([call, Call("query", (target,))]).splitlines()[0].split(" = ", 1)[1]
    if name == "same":
        return target
    if name == "offset" and unknown == args[1]:
        delta = args[2]
        operator = "-" if delta >= 0 else "+"
        return f"({target} {operator} {_number(abs(delta))})"
    if name == "difference":
        if unknown == args[1]:
            return f"({target} + {_operand(args[2])})"
        if unknown == args[2]:
            return f"({_operand(args[1])} - {target})"
    if name == "sum_of":
        others = tuple(value for value in operands if value != unknown)
        return f"({target} - {_join('+', others)})"
    if name in {"product_of", "multiple", "per_unit_total", "rate_total"}:
        others = tuple(value for value in operands if value != unknown)
        return f"({target} / {_join('*', others)})"
    if name == "quotient":
        if unknown == args[1]:
            return f"({target} * {_operand(args[2])})"
        if unknown == args[2]:
            return f"({_operand(args[1])} / {target})"
    if name == "fraction_of":
        base, numerator, denominator = args[1:]
        if unknown == base:
            return f"({target} * {_operand(denominator)} / {_operand(numerator)})"
        if unknown == numerator:
            return f"({target} * {_operand(denominator)} / {_operand(base)})"
        if unknown == denominator:
            return f"({_operand(base)} * {_operand(numerator)} / {target})"
    if name == "percent_of":
        if unknown == args[1]:
            return f"({target} * 100 / {_operand(args[2])})"
        if unknown == args[2]:
            return f"({target} * 100 / {_operand(args[1])})"
    if name == "percentage_ratio":
        if unknown == args[1]:
            return f"({target} * {_operand(args[2])} / 100)"
        if unknown == args[2]:
            return f"({_operand(args[1])} * 100 / {target})"
    if name in {"increase_percent", "decrease_percent"}:
        base, percentage = args[1:]
        sign = "+" if name == "increase_percent" else "-"
        if unknown == base:
            return f"({target} / (1 {sign} {_operand(percentage)} / 100))"
        if unknown == percentage:
            if name == "increase_percent":
                return f"(({target} / {_operand(base)}) - 1) * 100"
            return f"(1 - ({target} / {_operand(base)})) * 100"
    if name == "remaining":
        whole, *used = operands
        if unknown == whole:
            return f"({target} + {_join('+', tuple(used))})"
        other_used = tuple(value for value in used if value != unknown)
        suffix = f" - {_join('+', other_used)}" if other_used else ""
        return f"({_operand(whole)} - {target}{suffix})"
    if name == "mean_of":
        other_values = tuple(value for value in operands if value != unknown)
        return f"({target} * {len(operands)} - {_join('+', other_values)})"
    return None


def solve_f2_to_asl(calls: list[Call]) -> str:
    """Resolve single-unknown semantic constraints and emit an executable ASL plan."""

    givens = [call for call in calls if call.name == "given"]
    relations = [call for call in calls if call.name not in {"given", "query"}]
    query = calls[-1]
    known = {_path(call.args[0]) for call in givens}
    lines = [f"{_path(call.args[0])} = {_number(call.args[1])}" for call in givens]
    pending = list(relations)
    while pending:
        progress = False
        next_pending = []
        for call in pending:
            unknowns = _unknown_paths(call.args, known)
            if not unknowns:
                continue
            if len(unknowns) != 1:
                next_pending.append(call)
                continue
            unknown = unknowns[0]
            expression = _inverse_f2_expression(call, unknown)
            if expression is None:
                next_pending.append(call)
                continue
            lines.append(f"{unknown} = {expression}")
            known.add(unknown)
            progress = True
        if not progress:
            unresolved = sorted(
                {path for call in next_pending for path in _unknown_paths(call.args, known)}
            )
            raise ValueError(f"semantic constraints remain underdetermined: {unresolved}")
        pending = next_pending
    query_path = _path(query.args[0])
    if query_path not in known:
        raise ValueError(f"semantic query remains unresolved: {query_path}")
    lines.append(f"RETURN {query_path}")
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
        lowered = lower_f1(calls) if condition == "f1" else solve_f2_to_asl(calls)
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
