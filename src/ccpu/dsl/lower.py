"""Mechanical lowering from generic ASL syntax trees to CCIR-Arith v0."""

from __future__ import annotations

from typing import Any

from .registry import CCIR_CALL_OPERATORS

_BINARY = {
    "+": "ADD",
    "-": "SUB",
    "*": "MUL",
    "/": "DIV",
    "==": "EQ",
    "!=": "NE",
    "<": "LT",
    "<=": "LE",
    ">": "GT",
    ">=": "GE",
}


def _expression(node: dict[str, Any]) -> dict[str, Any]:
    node_type = node["type"]
    if node_type == "number":
        return {"op": "CONST", "value": node["value"]}
    if node_type in {"identifier", "path"}:
        path = node["name"] if node_type == "identifier" else ".".join(node["parts"])
        return {"op": "REF", "path": path}
    if node_type == "unary" and node["operator"] == "-":
        return {"op": "NEG", "value": _expression(node["operand"])}
    if node_type == "binary" and node["operator"] in _BINARY:
        return {
            "op": _BINARY[node["operator"]],
            "args": [_expression(node["left"]), _expression(node["right"])],
        }
    if node_type == "call":
        name = str(node["function"]).casefold()
        if name not in CCIR_CALL_OPERATORS:
            raise ValueError(f"cannot lower unregistered ASL-Arith call: {name}")
        return {
            "op": CCIR_CALL_OPERATORS[name],
            "args": [_expression(argument) for argument in node["arguments"]],
        }
    raise ValueError(f"cannot lower ASL expression node: {node_type}")


def lower_program(program: dict[str, Any]) -> dict[str, Any]:
    operations = []
    for record in program["records"]:
        statement = record["statement"]
        if statement["type"] == "statement" and statement["operator"] in {"=", "<-"}:
            left = statement["left"]
            target = left["name"] if left["type"] == "identifier" else ".".join(left["parts"])
            operation = {"op": "SET", "target": target, "expr": _expression(statement["right"])}
        elif statement["type"] == "return":
            operation = {"op": "RETURN", "expr": _expression(statement["expression"])}
        elif statement["type"] in {"scope_start", "scope_end"}:
            operation = {
                "op": statement["type"].upper(),
                **{k: v for k, v in statement.items() if k != "type"},
            }
        else:
            continue
        operations.append(
            {"scope": record["scope"], "source_line": record["source_line"], "operation": operation}
        )
    return {"ccir_version": "ccir-arith-v0", "operations": operations}
