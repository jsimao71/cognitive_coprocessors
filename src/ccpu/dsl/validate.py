"""Parse, lower, scope-check, and execute ASL candidates."""

from __future__ import annotations

from typing import Any

from .execute import execute_program
from .lower import lower_program
from .parser import parse_asl


def validate_asl(text: str, *, effective_scope: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "syntax_verified": False,
        "type_verified": False,
        "scope_verified": False,
        "execution_verified": False,
        "deferred_verified": False,
        "errors": [],
    }
    try:
        ast = parse_asl(text, effective_scope=effective_scope)
        result["syntax_verified"] = True
        ccir = lower_program(ast)
        result["type_verified"] = True
        execution = execute_program(ast)
        result["scope_verified"] = True
        result["deferred_verified"] = bool(execution["unresolved"])
        result["execution_verified"] = not execution["unresolved"]
        if execution["unresolved"]:
            result["errors"].extend(item["reason"] for item in execution["unresolved"])
        result.update({"ast": ast, "ccir": ccir, "execution": execution})
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        result["errors"].append(str(error))
    return result
