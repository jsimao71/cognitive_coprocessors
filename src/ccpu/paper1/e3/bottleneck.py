"""Canonical, ASL-isomorphic semantic bottleneck with exact runtime lowering."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from ccpu.common.artifacts import canonical_json
from ccpu.dsl import parse_asl

SCHEMA_VERSION = "ccpu.paper1.semantic_bottleneck.v1"

_BINARY_TO_SEMANTIC = {
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
    "AND": "AND",
    "OR": "OR",
    ",": "AND",
    ";": "OR",
}
_SEMANTIC_TO_BINARY = {
    value: key for key, value in _BINARY_TO_SEMANTIC.items() if key not in {",", ";"}
}


def _path(node: dict[str, Any]) -> str:
    if node["type"] == "identifier":
        return str(node["name"])
    if node["type"] == "path":
        return ".".join(str(part) for part in node["parts"])
    raise ValueError("semantic bottleneck assignments require an identifier or path")


def _bind(path: str, slots: dict[str, str]) -> str:
    if path not in slots:
        slots[path] = f"s{len(slots)}"
    return slots[path]


def _encode_expression(node: dict[str, Any], slots: dict[str, str]) -> dict[str, Any]:
    kind = str(node["type"])
    if kind in {"number", "string", "boolean", "null"}:
        return {"kind": "literal", "literal_type": kind, "value": node["value"]}
    if kind in {"identifier", "path"}:
        return {"kind": "ref", "slot": _bind(_path(node), slots)}
    if kind == "unary":
        operator = str(node["operator"])
        semantic = "NEG" if operator == "-" else "NOT"
        return {
            "kind": "apply",
            "operator": semantic,
            "arguments": [_encode_expression(node["operand"], slots)],
        }
    if kind == "binary":
        operator = str(node["operator"])
        if operator not in _BINARY_TO_SEMANTIC:
            raise ValueError(f"unsupported bottleneck binary operator: {operator}")
        return {
            "kind": "apply",
            "operator": _BINARY_TO_SEMANTIC[operator],
            "arguments": [
                _encode_expression(node["left"], slots),
                _encode_expression(node["right"], slots),
            ],
        }
    if kind == "call":
        arguments = []
        for argument in node["arguments"]:
            if argument["type"] == "named_argument":
                arguments.append(
                    {
                        "kind": "named",
                        "name": str(argument["name"]),
                        "value": _encode_expression(argument["value"], slots),
                    }
                )
            else:
                arguments.append(_encode_expression(argument, slots))
        return {
            "kind": "call",
            "operator": str(node["function"]).casefold(),
            "arguments": arguments,
        }
    if kind == "list":
        return {
            "kind": "list",
            "items": [_encode_expression(item, slots) for item in node["items"]],
        }
    if kind == "record":
        return {
            "kind": "record",
            "fields": [
                {"name": field["name"], "value": _encode_expression(field["value"], slots)}
                for field in node["fields"]
            ],
        }
    raise ValueError(f"unsupported bottleneck expression node: {kind}")


def asl_to_bottleneck(asl: str, *, effective_scope: dict[str, Any] | None = None) -> dict[str, Any]:
    """Factor an ASL program into a symbol table and canonical expression graph."""

    parsed = parse_asl(asl, effective_scope=effective_scope)
    slots: dict[str, str] = {}
    steps: list[dict[str, Any]] = []
    for record in parsed["records"]:
        statement = record["statement"]
        kind = statement["type"]
        if kind == "statement" and statement["operator"] in {"=", "<-"}:
            target = _bind(_path(statement["left"]), slots)
            steps.append(
                {
                    "kind": "set",
                    "target": target,
                    "expression": _encode_expression(statement["right"], slots),
                }
            )
        elif kind == "return":
            steps.append(
                {
                    "kind": "return",
                    "expression": _encode_expression(statement["expression"], slots),
                }
            )
        elif kind in {"scope_start", "scope_end"}:
            raise ValueError("v1 semantic bottleneck excludes explicit nested scopes")
        else:
            raise ValueError(f"v1 semantic bottleneck cannot encode statement type: {kind}")
    return {
        "schema_version": SCHEMA_VERSION,
        "bindings": [
            {"slot": slot, "path": path}
            for path, slot in sorted(slots.items(), key=lambda item: int(item[1][1:]))
        ],
        "steps": steps,
    }


def render_bottleneck(program: dict[str, Any]) -> str:
    """Render the one-line canonical model target."""

    _validate_structure(program)
    return canonical_json(program)


def parse_bottleneck(text: str) -> dict[str, Any]:
    """Parse and structurally validate a generated bottleneck target."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid semantic bottleneck JSON: {error}") from error
    if not isinstance(value, dict):
        raise TypeError("semantic bottleneck root must be an object")
    _validate_structure(value)
    return value


def _validate_expression(node: Any, slots: set[str]) -> None:
    if not isinstance(node, dict):
        raise TypeError("bottleneck expression must be an object")
    kind = node.get("kind")
    if kind == "literal":
        if node.get("literal_type") not in {"number", "string", "boolean", "null"}:
            raise ValueError("invalid bottleneck literal type")
        return
    if kind == "ref":
        if node.get("slot") not in slots:
            raise ValueError(f"reference uses unknown slot: {node.get('slot')}")
        return
    if kind in {"apply", "call"}:
        if not isinstance(node.get("operator"), str) or not isinstance(node.get("arguments"), list):
            raise ValueError(f"invalid {kind} expression")
        for argument in node["arguments"]:
            if isinstance(argument, dict) and argument.get("kind") == "named":
                _validate_expression(argument.get("value"), slots)
            else:
                _validate_expression(argument, slots)
        return
    if kind == "list":
        for item in node.get("items", []):
            _validate_expression(item, slots)
        return
    if kind == "record":
        for field in node.get("fields", []):
            if not isinstance(field, dict) or not isinstance(field.get("name"), str):
                raise TypeError("invalid bottleneck record field")
            _validate_expression(field.get("value"), slots)
        return
    raise ValueError(f"invalid bottleneck expression kind: {kind}")


def _validate_structure(program: dict[str, Any]) -> None:
    if program.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected semantic bottleneck schema {SCHEMA_VERSION}")
    bindings = program.get("bindings")
    steps = program.get("steps")
    if not isinstance(bindings, list) or not isinstance(steps, list):
        raise TypeError("semantic bottleneck requires binding and step lists")
    expected = [f"s{index}" for index in range(len(bindings))]
    observed = [binding.get("slot") for binding in bindings if isinstance(binding, dict)]
    paths = [binding.get("path") for binding in bindings if isinstance(binding, dict)]
    if observed != expected or len(paths) != len(bindings):
        raise ValueError("bottleneck slots must be unique, contiguous, and ordered")
    if any(not isinstance(path, str) or not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("bottleneck paths must be non-empty and unique")
    slots = set(expected)
    returns = 0
    for step in steps:
        if not isinstance(step, dict) or step.get("kind") not in {"set", "return"}:
            raise ValueError("invalid bottleneck step")
        if step["kind"] == "set" and step.get("target") not in slots:
            raise ValueError(f"assignment uses unknown slot: {step.get('target')}")
        if step["kind"] == "return":
            returns += 1
        _validate_expression(step.get("expression"), slots)
    if returns != 1 or not steps or steps[-1].get("kind") != "return":
        raise ValueError("bottleneck requires exactly one final return")


def _literal(value: Any, literal_type: str) -> str:
    if literal_type == "string":
        return json.dumps(value, ensure_ascii=True)
    if literal_type == "boolean":
        return "TRUE" if value else "FALSE"
    if literal_type == "null":
        return "NULL"
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    return str(value)


def _decode_expression(node: dict[str, Any], paths: dict[str, str]) -> str:
    kind = node["kind"]
    if kind == "literal":
        return _literal(node.get("value"), str(node["literal_type"]))
    if kind == "ref":
        return paths[str(node["slot"])]
    if kind == "apply":
        operator = str(node["operator"])
        arguments = [_decode_expression(argument, paths) for argument in node["arguments"]]
        if operator == "NEG":
            return f"(-{arguments[0]})"
        if operator == "NOT":
            return f"(NOT {arguments[0]})"
        if operator not in _SEMANTIC_TO_BINARY or len(arguments) != 2:
            raise ValueError(f"invalid semantic application: {operator}")
        return f"({arguments[0]} {_SEMANTIC_TO_BINARY[operator]} {arguments[1]})"
    if kind == "call":
        arguments = []
        for argument in node["arguments"]:
            if argument.get("kind") == "named":
                arguments.append(f"{argument['name']}={_decode_expression(argument['value'], paths)}")
            else:
                arguments.append(_decode_expression(argument, paths))
        return f"{node['operator']}({', '.join(arguments)})"
    if kind == "list":
        return "[" + ", ".join(_decode_expression(item, paths) for item in node["items"]) + "]"
    if kind == "record":
        fields = [
            f"{field['name']}: {_decode_expression(field['value'], paths)}"
            for field in node["fields"]
        ]
        return "{" + ", ".join(fields) + "}"
    raise ValueError(f"cannot lower bottleneck expression kind: {kind}")


def lower_bottleneck_to_asl(program: dict[str, Any] | str) -> str:
    """Lower a validated bottleneck object to ordinary executable ASL."""

    value = parse_bottleneck(program) if isinstance(program, str) else program
    _validate_structure(value)
    paths = {str(item["slot"]): str(item["path"]) for item in value["bindings"]}
    lines = []
    for step in value["steps"]:
        expression = _decode_expression(step["expression"], paths)
        if step["kind"] == "set":
            lines.append(f"{paths[str(step['target'])]} = {expression}")
        else:
            lines.append(f"RETURN {expression}")
    return "\n".join(lines)
