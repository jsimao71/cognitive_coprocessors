"""Deterministic, executable hard negatives for semantic decisions."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any

from ccpu.common.artifacts import canonical_json
from ccpu.dsl import validate_asl
from ccpu.paper1.asl_pilot_eval import score_asl

from .bottleneck import lower_bottleneck_to_asl

_OPERATOR_SWAPS = {"ADD": "SUB", "SUB": "ADD", "MUL": "DIV", "DIV": "MUL"}


def _expressions(program: dict[str, Any]) -> Iterator[dict[str, Any]]:
    def walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield node
        if node.get("kind") in {"apply", "call"}:
            for argument in node["arguments"]:
                child = argument["value"] if argument.get("kind") == "named" else argument
                yield from walk(child)
        elif node.get("kind") == "list":
            for item in node["items"]:
                yield from walk(item)
        elif node.get("kind") == "record":
            for field in node["fields"]:
                yield from walk(field["value"])

    for step in program["steps"]:
        yield from walk(step["expression"])


def _operator(candidate: dict[str, Any]) -> bool:
    for node in _expressions(candidate):
        if node.get("kind") == "apply" and node.get("operator") in _OPERATOR_SWAPS:
            node["operator"] = _OPERATOR_SWAPS[str(node["operator"])]
            return True
    return False


def _dependency(candidate: dict[str, Any]) -> bool:
    assigned: list[str] = []
    for step in candidate["steps"]:
        if step["kind"] == "set":
            for node in _walk_one(step["expression"]):
                if node.get("kind") == "ref" and len(assigned) > 1:
                    alternatives = [slot for slot in assigned if slot != node["slot"]]
                    if alternatives:
                        node["slot"] = alternatives[0]
                        return True
            assigned.append(str(step["target"]))
    return False


def _walk_one(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    if node.get("kind") in {"apply", "call"}:
        for argument in node["arguments"]:
            child = argument["value"] if argument.get("kind") == "named" else argument
            yield from _walk_one(child)
    elif node.get("kind") == "list":
        for item in node["items"]:
            yield from _walk_one(item)
    elif node.get("kind") == "record":
        for field in node["fields"]:
            yield from _walk_one(field["value"])


def _query(candidate: dict[str, Any]) -> bool:
    returned = candidate["steps"][-1]["expression"]
    if returned.get("kind") != "ref":
        return False
    targets = [step["target"] for step in candidate["steps"] if step["kind"] == "set"]
    alternatives = [slot for slot in reversed(targets) if slot != returned["slot"]]
    if not alternatives:
        return False
    returned["slot"] = alternatives[0]
    return True


def _binding(candidate: dict[str, Any]) -> bool:
    bindings = candidate["bindings"]
    if len(bindings) < 2:
        return False
    bindings[0]["path"], bindings[1]["path"] = bindings[1]["path"], bindings[0]["path"]
    return True


def _source_fact(candidate: dict[str, Any]) -> bool:
    literals = [
        step["expression"]
        for step in candidate["steps"]
        if step["kind"] == "set"
        and step["expression"].get("kind") == "literal"
        and step["expression"].get("literal_type") == "number"
    ]
    for index, left in enumerate(literals):
        for right in literals[index + 1 :]:
            if left.get("value") != right.get("value"):
                left["value"], right["value"] = right["value"], left["value"]
                return True
    return False


_MUTATORS: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
    ("operator_swap", _operator),
    ("dependency_rebind", _dependency),
    ("query_target_swap", _query),
    ("path_binding_swap", _binding),
    ("source_fact_swap", _source_fact),
)


def generate_hard_negatives(
    program: dict[str, Any],
    *,
    reference_asl: str,
    effective_scope: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate one valid semantic counterexample per available decision class."""

    original = canonical_json(program)
    negatives = []
    seen = {original}
    for negative_type, mutate in _MUTATORS:
        candidate = deepcopy(program)
        if not mutate(candidate):
            continue
        serialized = canonical_json(candidate)
        if serialized in seen:
            continue
        try:
            asl = lower_bottleneck_to_asl(candidate)
        except (KeyError, TypeError, ValueError):
            continue
        validation = validate_asl(asl, effective_scope=effective_scope)
        if not validation["execution_verified"]:
            continue
        score = score_asl(reference_asl, asl, effective_scope)
        scorer_blind_binding = negative_type == "path_binding_swap"
        if (
            score["semantic_state_equivalent"]
            and score["semantic_return_equivalent"]
            and not scorer_blind_binding
        ):
            continue
        seen.add(serialized)
        negatives.append(
            {
                "negative_type": negative_type,
                "bottleneck": candidate,
                "lowered_asl": asl,
                "final_answer_accidentally_correct": score["final_answer_correct"],
                "semantic_state_equivalent": score["semantic_state_equivalent"],
                "semantic_return_equivalent": score["semantic_return_equivalent"],
                "binding_changed": scorer_blind_binding,
                "legacy_semantic_scorer_detected": not (
                    score["semantic_state_equivalent"] and score["semantic_return_equivalent"]
                ),
            }
        )
    return negatives
