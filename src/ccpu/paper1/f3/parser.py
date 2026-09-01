"""Safe parser and extractor for canonical F3-v1 call programs."""

from __future__ import annotations

import ast
import re
from decimal import Decimal

from .ast import Form, Program, Value
from .registry import NESTED, TOP_LEVEL, Signature

_FENCE = re.compile(r"```(?:f3|asl|text)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_CALL_LINE = re.compile(r"[a-z_][a-z0-9_]*\(.*\)")


def _check_arity(name: str, count: int, signature: Signature) -> None:
    if count < signature.minimum or (
        signature.maximum is not None and count > signature.maximum
    ):
        maximum = "*" if signature.maximum is None else str(signature.maximum)
        expected = (
            str(signature.minimum)
            if signature.minimum == signature.maximum
            else f"{signature.minimum}..{maximum}"
        )
        raise ValueError(f"{name} expects {expected} arguments, got {count}")


def _value(node: ast.AST, *, nested: bool) -> Value:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value, bool
    ):
        return node.value
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return Decimal(str(node.value))
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        value = Decimal(str(node.operand.value))
        return -value if isinstance(node.op, ast.USub) else value
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
        raise ValueError("F3 allows only literals and allowlisted positional calls")
    registry = NESTED if nested else TOP_LEVEL
    name = node.func.id
    if name not in registry:
        location = "nested" if nested else "top-level"
        raise ValueError(f"unsupported {location} F3 form: {name}")
    args = tuple(_value(argument, nested=True) for argument in node.args)
    _check_arity(name, len(args), registry[name])
    return Form(name, args)


def parse_f3_program(text: str) -> Program:
    """Parse one allowlisted form per line without evaluating Python."""

    forms: list[Form] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().rstrip(";")
        if not line or line.startswith("#"):
            continue
        try:
            expression = ast.parse(line, mode="eval").body
            value = _value(expression, nested=False)
        except (SyntaxError, TypeError, ValueError) as error:
            raise ValueError(f"line {line_number}: {error}") from error
        if not isinstance(value, Form):
            raise TypeError(f"line {line_number}: expected an F3 call")
        forms.append(value)
    if not forms:
        raise ValueError("program contains no F3 forms")
    if forms[-1].name != "query" or any(form.name == "query" for form in forms[:-1]):
        raise ValueError("program must end with exactly one query(...)")
    return Program(tuple(forms))


def extract_f3_program(text: str) -> str:
    """Extract the first contiguous parseable F3 call block from generated text."""

    candidates = [match.strip() for match in _FENCE.findall(text) if match.strip()]
    candidates.append(text.strip())
    for candidate in candidates:
        lines: list[str] = []
        for raw in candidate.splitlines():
            line = raw.strip().rstrip(";")
            if _CALL_LINE.fullmatch(line):
                lines.append(line)
            elif lines and line:
                break
        program = "\n".join(lines)
        if not program:
            continue
        try:
            parse_f3_program(program)
            return program
        except ValueError:
            continue
    return ""
