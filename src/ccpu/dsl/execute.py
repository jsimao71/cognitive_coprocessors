"""Deterministic ASL-Arith execution over a hierarchical workspace."""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Any

from .registry import ARITHMETIC_FUNCTIONS
from .state import ScopeError, Workspace


def _path(node: dict[str, Any]) -> str:
    if node["type"] == "path":
        return ".".join(node["parts"])
    if node["type"] == "identifier":
        return str(node["name"])
    raise ValueError("assignment target must be an identifier or path")


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or value is None:
        raise TypeError("boolean/null is not arithmetic")
    return Decimal(str(value))


def _evaluate(node: dict[str, Any], workspace: Workspace, scope: str) -> Any:
    node_type = node["type"]
    if node_type in {"number", "string", "boolean", "null"}:
        value = node["value"]
        return _decimal(value) if node_type == "number" else value
    if node_type == "identifier":
        return workspace.get(scope, str(node["name"]))
    if node_type == "path":
        return workspace.get(scope, ".".join(node["parts"]))
    if node_type == "unary":
        value = _evaluate(node["operand"], workspace, scope)
        if node["operator"] == "-":
            return -_decimal(value)
        if node["operator"] in {"!", "NOT"}:
            return not bool(value)
        raise ValueError(f"unsupported unary operator: {node['operator']}")
    if node_type == "binary":
        left = _evaluate(node["left"], workspace, scope)
        right = _evaluate(node["right"], workspace, scope)
        operator = node["operator"]
        if operator == "+":
            return _decimal(left) + _decimal(right)
        if operator == "-":
            return _decimal(left) - _decimal(right)
        if operator == "*":
            return _decimal(left) * _decimal(right)
        if operator == "/":
            return _decimal(left) / _decimal(right)
        if operator in {"==", "=", "!=", "<", "<=", ">", ">="}:
            functions = {
                "==": lambda: left == right,
                "=": lambda: left == right,
                "!=": lambda: left != right,
                "<": lambda: left < right,
                "<=": lambda: left <= right,
                ">": lambda: left > right,
                ">=": lambda: left >= right,
            }
            return functions[operator]()
        if operator in {"AND", ","}:
            return bool(left) and bool(right)
        if operator in {"OR", ";"}:
            return bool(left) or bool(right)
        raise ValueError(f"unsupported binary operator: {operator}")
    if node_type == "call":
        name = str(node["function"]).casefold()
        if name not in ARITHMETIC_FUNCTIONS:
            raise ValueError(f"operator is not registered in ASL-Arith: {name}")
        if any(argument["type"] == "named_argument" for argument in node["arguments"]):
            raise ValueError(f"named arguments are not supported by arithmetic operator {name}")
        arguments = [_decimal(_evaluate(arg, workspace, scope)) for arg in node["arguments"]]
        return ARITHMETIC_FUNCTIONS[name](arguments)
    if node_type == "list":
        return [_evaluate(item, workspace, scope) for item in node["items"]]
    if node_type == "record":
        return {
            field["name"]: _evaluate(field["value"], workspace, scope) for field in node["fields"]
        }
    raise ValueError(f"unsupported expression node: {node_type}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral() else format(value.normalize(), "f")
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def execute_program(program: dict[str, Any]) -> dict[str, Any]:
    workspace = Workspace(program["root_scope"])
    scope_lookup = {str(program["root_scope"]["id"]): workspace.root_id}
    for scope in program["scopes"][1:]:
        scope_lookup[str(scope["id"])] = workspace.add_scope(scope)
    trace = []
    pending: list[dict[str, Any]] = []

    def run_statement(record: dict[str, Any], scope: str) -> Any:
        statement = record["statement"]
        statement_type = statement["type"]
        if statement_type in {"scope_start", "scope_end"}:
            return None
        if statement_type == "statement":
            if statement["operator"] not in {"=", "<-"}:
                raise ValueError(
                    f"statement operator is not executable in ASL-Arith: {statement['operator']}"
                )
            value = _evaluate(statement["right"], workspace, scope)
            workspace.set(scope, _path(statement["left"]), value)
            return value
        if statement_type == "return":
            value = _evaluate(statement["expression"], workspace, scope)
            workspace.return_value(scope, value)
            return value
        if statement_type in {"query", "expression_statement"}:
            return _evaluate(statement["expression"], workspace, scope)
        raise ValueError(f"unsupported statement node: {statement_type}")

    def resolve_pending() -> None:
        progressed = True
        while pending and progressed:
            progressed = False
            for item in list(pending):
                try:
                    value = run_statement(item["record"], item["scope"])
                except ScopeError as error:
                    item["error"] = str(error)
                    continue
                pending.remove(item)
                progressed = True
                trace.append(
                    {
                        "source_line": item["record"]["source_line"],
                        "scope": item["scope"],
                        "statement_type": item["record"]["statement"]["type"],
                        "status": "resolved",
                        "value": _json_value(value),
                    }
                )

    with localcontext() as context:
        context.prec = 40
        for record in program["records"]:
            statement = record["statement"]
            statement_type = statement["type"]
            scope = scope_lookup[str(record["scope"]["id"])]
            status = "executed"
            try:
                value = run_statement(record, scope)
            except ScopeError as error:
                value = None
                status = "deferred"
                pending.append({"record": record, "scope": scope, "error": str(error)})
            trace.append(
                {
                    "source_line": record["source_line"],
                    "scope": scope,
                    "statement_type": statement_type,
                    "status": status,
                    "value": _json_value(value),
                }
            )
            if status == "executed":
                resolve_pending()
    return {
        "workspace": _json_value(workspace.snapshot()),
        "trace": trace,
        "unresolved": [
            {
                "source_line": item["record"]["source_line"],
                "scope": item["scope"],
                "statement_type": item["record"]["statement"]["type"],
                "reason": item["error"],
            }
            for item in pending
        ],
    }
