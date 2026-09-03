"""Deterministic semantic-decision labels for bottleneck supervision."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _walk(node: dict[str, Any], *, operators: list[str], references: list[str]) -> None:
    kind = node["kind"]
    if kind == "ref":
        references.append(str(node["slot"]))
    elif kind in {"apply", "call"}:
        operators.append(str(node["operator"]))
        for argument in node["arguments"]:
            child = argument["value"] if argument.get("kind") == "named" else argument
            _walk(child, operators=operators, references=references)
    elif kind == "list":
        for item in node["items"]:
            _walk(item, operators=operators, references=references)
    elif kind == "record":
        for field in node["fields"]:
            _walk(field["value"], operators=operators, references=references)


def component_labels(program: dict[str, Any]) -> dict[str, Any]:
    """Extract auditable targets for grounding, operations, dependencies, and query."""

    paths = {str(item["slot"]): str(item["path"]) for item in program["bindings"]}
    operators: list[str] = []
    references: list[str] = []
    source_facts = []
    dependencies = []
    query = None
    for step in program["steps"]:
        local_refs: list[str] = []
        _walk(step["expression"], operators=operators, references=local_refs)
        references.extend(local_refs)
        if step["kind"] == "set":
            target = str(step["target"])
            dependencies.extend({"source": source, "target": target} for source in local_refs)
            if step["expression"]["kind"] == "literal":
                source_facts.append(
                    {
                        "target": target,
                        "literal_type": step["expression"]["literal_type"],
                        "value": step["expression"].get("value"),
                    }
                )
        else:
            query = step["expression"]
    return {
        "bindings": [{"slot": slot, "path": paths[slot]} for slot in paths],
        "source_facts": source_facts,
        "operator_counts": dict(sorted(Counter(operators).items())),
        "reference_counts": dict(sorted(Counter(references).items())),
        "dependencies": dependencies,
        "query": query,
    }
