"""Deterministic fine-grained semantic diagnostics for saved Paper 1 ASL runs."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json, write_jsonl
from ccpu.dsl import parse_asl, validate_asl

_TOKEN_ALIAS = {
    "current": "now",
    "present": "now",
    "presently": "now",
    "qty": "count",
    "quantity": "count",
    "number": "count",
    "amount": "count",
    "overall": "total",
    "all": "total",
    "left": "remaining",
    "remainder": "remaining",
    "avg": "average",
    "mean": "average",
    "pct": "percent",
    "percentage": "percent",
}
_TEMPORAL = {
    "now",
    "today",
    "yesterday",
    "tomorrow",
    "future",
    "past",
    "before",
    "after",
    "previous",
    "next",
    "april",
    "may",
    "january",
    "february",
    "march",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "day",
    "week",
    "month",
    "year",
}
_CARDINALITY = {
    "each",
    "per",
    "total",
    "remaining",
    "count",
    "average",
    "sum",
    "combined",
    "group",
    "groups",
    "member",
    "members",
}
_STATUS = {
    "accepted",
    "rejected",
    "returned",
    "damaged",
    "sold",
    "used",
    "unused",
    "ending",
    "beginning",
    "completed",
}
_UNIT = {
    "dollar",
    "dollars",
    "euro",
    "euros",
    "cent",
    "cents",
    "hour",
    "hours",
    "minute",
    "minutes",
    "day",
    "days",
    "week",
    "weeks",
    "year",
    "years",
    "meter",
    "meters",
    "mile",
    "miles",
    "foot",
    "feet",
    "kilogram",
    "kilograms",
    "gram",
    "grams",
    "liter",
    "liters",
    "percent",
    "thousand",
    "thousands",
    "million",
    "millions",
}
_COMMUTATIVE = {"ADD", "MUL", "SUM", "MIN", "MAX", "MEAN"}
_AGGREGATION = {"SUM", "MEAN", "MIN", "MAX"}
_OPERATOR_FAMILIES = {
    "ADD": "add",
    "SUB": "sub",
    "MUL": "mul",
    "DIV": "div",
    "PERCENT_OF": "percentage",
    "PERCENT_CHANGE": "percentage",
    "SUM": "aggregation",
    "MEAN": "aggregation",
    "MIN": "min_max",
    "MAX": "min_max",
}


def _tokens(value: str) -> tuple[str, ...]:
    raw = re.findall(r"[a-z]+|y\d{4}|\d+", value.casefold().replace("-", "_"))
    return tuple(_TOKEN_ALIAS.get(token, token) for token in raw)


def _qualifier_category(token: str) -> str | None:
    if token in _TEMPORAL or re.fullmatch(r"y\d{4}|(?:month|week|day)\d+", token):
        return "temporal"
    if token in _CARDINALITY:
        return "cardinality"
    if token in _STATUS:
        return "status"
    if re.fullmatch(r"group\d+", token):
        return "grouping"
    return None


def _path_features(path: str) -> dict[str, Any]:
    parts = path.split(".")
    entity = "_".join(_tokens(parts[0]))
    semantic_tokens = [token for part in parts[1:] for token in _tokens(part)]
    qualifiers = [
        (category, token)
        for token in semantic_tokens
        if (category := _qualifier_category(token)) is not None
    ]
    attributes = tuple(token for token in semantic_tokens if _qualifier_category(token) is None)
    units = tuple(token for token in semantic_tokens if token in _UNIT)
    return {
        "entity": entity,
        "attributes": attributes,
        "qualifiers": tuple(qualifiers),
        "units": units,
        "semantic_tokens": tuple(semantic_tokens),
    }


def _counter_metric(reference: Counter[Any], predicted: Counter[Any]) -> dict[str, Any]:
    overlap = sum((reference & predicted).values())
    reference_count = sum(reference.values())
    predicted_count = sum(predicted.values())
    precision = overlap / predicted_count if predicted_count else float(reference_count == 0)
    recall = overlap / reference_count if reference_count else float(predicted_count == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap": overlap,
        "reference_count": reference_count,
        "predicted_count": predicted_count,
        "support": reference_count,
    }


def _accuracy(correct: int, support: int) -> dict[str, Any]:
    return {"correct": correct, "support": support, "accuracy": correct / support if support else None}


def _expr_refs(node: dict[str, Any]) -> list[str]:
    if node.get("op") == "REF":
        return [str(node["path"])]
    return [path for argument in node.get("args", []) for path in _expr_refs(argument)]


def _expr_constants(node: dict[str, Any]) -> list[str]:
    if node.get("op") == "CONST":
        return [str(node.get("value"))]
    return [value for argument in node.get("args", []) for value in _expr_constants(argument)]


def _expr_leaves(node: dict[str, Any]) -> list[tuple[str, str]]:
    if node.get("op") == "REF":
        return [("REF", str(node["path"]))]
    if node.get("op") == "CONST":
        return [("CONST", str(node.get("value")))]
    return [leaf for argument in node.get("args", []) for leaf in _expr_leaves(argument)]


def _expr_operators(node: dict[str, Any]) -> list[str]:
    operation = str(node.get("op", ""))
    values = [] if operation in {"CONST", "REF", ""} else [operation]
    return values + [
        nested for argument in node.get("args", []) for nested in _expr_operators(argument)
    ]


def _program_view(asl: str, scope: dict[str, Any]) -> dict[str, Any]:
    validation = validate_asl(asl, effective_scope=scope)
    view: dict[str, Any] = {
        "validation": validation,
        "paths": set(),
        "relations": [],
        "return": None,
        "return_kind": "missing",
        "source_facts": [],
        "operators": [],
        "scopes": [],
    }
    if validation["syntax_verified"]:
        parsed = parse_asl(asl, effective_scope=scope)
        view["scopes"] = [str(item["id"]) for item in parsed["scopes"]]
    if not validation["lower_verified"]:
        return view
    for item in validation["ccir"]["operations"]:
        operation = item["operation"]
        if operation["op"] == "SET":
            target = str(operation["target"])
            expression = operation["expr"]
            refs = _expr_refs(expression)
            constants = _expr_constants(expression)
            operators = _expr_operators(expression)
            relation = {
                "target": target,
                "operator": str(expression["op"]),
                "refs": refs,
                "constants": constants,
                "leaves": _expr_leaves(expression),
                "operators": operators,
            }
            view["relations"].append(relation)
            view["paths"].add(target)
            view["paths"].update(refs)
            view["operators"].extend(operators)
            if expression["op"] == "CONST":
                view["source_facts"].append((target, str(expression.get("value"))))
        elif operation["op"] == "RETURN":
            expression = operation["expr"]
            view["operators"].extend(_expr_operators(expression))
            refs = _expr_refs(expression)
            view["paths"].update(refs)
            if expression["op"] == "REF":
                view["return"] = str(expression["path"])
                view["return_kind"] = "path"
            elif expression["op"] == "CONST":
                view["return"] = str(expression.get("value"))
                view["return_kind"] = "literal"
            else:
                view["return"] = expression
                view["return_kind"] = "expression"
    return view


def _role_signatures(view: dict[str, Any]) -> dict[str, set[tuple[Any, ...]]]:
    roles: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for relation in view["relations"]:
        roles[relation["target"]].add(("target", relation["operator"], len(relation["refs"])))
        for index, reference in enumerate(relation["refs"]):
            roles[reference].add(("source", relation["operator"], index))
    if view["return_kind"] == "path":
        roles[str(view["return"])].add(("return",))
    return roles


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def _align_paths(reference: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    reference_roles = _role_signatures(reference)
    predicted_roles = _role_signatures(predicted)
    candidates = []
    ambiguous = []
    for reference_path in sorted(reference["paths"]):
        rf = _path_features(reference_path)
        row = []
        for predicted_path in sorted(predicted["paths"]):
            pf = _path_features(predicted_path)
            if rf["entity"] != pf["entity"]:
                continue
            token_score = _jaccard(rf["semantic_tokens"], pf["semantic_tokens"])
            role_score = _jaccard(reference_roles[reference_path], predicted_roles[predicted_path])
            exact = reference_path == predicted_path
            if not exact and token_score < 0.45:
                continue
            score = 10.0 * exact + 3.0 * token_score + 2.0 * role_score
            row.append((score, predicted_path))
            candidates.append((score, reference_path, predicted_path))
        row.sort(reverse=True)
        if len(row) > 1 and math.isclose(row[0][0], row[1][0]):
            ambiguous.append(reference_path)
    mapping: dict[str, str] = {}
    used = set()
    for _, reference_path, predicted_path in sorted(candidates, reverse=True):
        if reference_path not in mapping and predicted_path not in used:
            mapping[reference_path] = predicted_path
            used.add(predicted_path)
    unresolved = []
    for reference_path in sorted(reference["paths"] - mapping.keys()):
        rf = _path_features(reference_path)
        if any(
            rf["entity"] == _path_features(predicted_path)["entity"]
            and _jaccard(reference_roles[reference_path], predicted_roles[predicted_path]) >= 0.5
            for predicted_path in predicted["paths"] - used
        ):
            unresolved.append(reference_path)
    return {
        "mapping": mapping,
        "ambiguous_reference_paths": ambiguous,
        "unresolved_reference_paths": unresolved,
    }


def _translated(path: str, reverse_mapping: dict[str, str]) -> str:
    return reverse_mapping.get(path, f"UNALIGNED:{path}")


def _relation_signatures(
    view: dict[str, Any], reverse_mapping: dict[str, str] | None = None
) -> Counter[Any]:
    reverse = reverse_mapping or {}
    signatures: Counter[Any] = Counter()
    for relation in view["relations"]:
        target = _translated(relation["target"], reverse) if reverse_mapping is not None else relation["target"]
        refs = [
            _translated(path, reverse) if reverse_mapping is not None else path
            for path in relation["refs"]
        ]
        if relation["operator"] in _COMMUTATIVE:
            refs = sorted(refs)
        signatures[(target, relation["operator"], tuple(refs), tuple(relation["constants"]))] += 1
    return signatures


def _path_counters(view: dict[str, Any]) -> dict[str, Counter[Any]]:
    entity_values = set()
    attributes: Counter[Any] = Counter()
    qualifiers: Counter[Any] = Counter()
    temporal: Counter[Any] = Counter()
    cardinality: Counter[Any] = Counter()
    status: Counter[Any] = Counter()
    units: Counter[Any] = Counter()
    for path in view["paths"]:
        features = _path_features(path)
        entity_values.add(features["entity"])
        attributes[features["attributes"]] += 1
        units.update(features["units"])
        for category, token in features["qualifiers"]:
            qualifiers[(category, token)] += 1
            {"temporal": temporal, "cardinality": cardinality, "status": status}.get(
                category, Counter()
            )[token] += 1
    return {
        "entity": Counter(entity_values),
        "attribute": attributes,
        "qualifier": qualifiers,
        "temporal_qualifier": temporal,
        "cardinality_qualifier": cardinality,
        "status_qualifier": status,
        "unit": units,
    }


def _relation_by_target(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {relation["target"]: relation for relation in view["relations"]}


def _dependency_depth(view: dict[str, Any]) -> int:
    relations = _relation_by_target(view)

    def depth(path: str, stack: frozenset[str]) -> int:
        if path in stack:
            return 0
        internal = [ref for ref in relations.get(path, {}).get("refs", []) if ref in relations]
        return 1 + max((depth(ref, stack | {path}) for ref in internal), default=0)

    return max((depth(path, frozenset()) for path in relations), default=0)


def _ancestors(view: dict[str, Any]) -> dict[str, set[str]]:
    relations = _relation_by_target(view)

    def visit(path: str, stack: frozenset[str]) -> set[str]:
        if path in stack:
            return set()
        direct = set(relations.get(path, {}).get("refs", []))
        return direct | set().union(*(visit(item, stack | {path}) for item in direct), set())

    return {path: visit(path, frozenset()) for path in relations}


def _execution_values(view: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    execution = view["validation"].get("execution") or {}
    workspace = execution.get("workspace", {}).get(str(scope["id"]), {})
    return dict(workspace.get("values", {})) if isinstance(workspace.get("values"), dict) else {}


def analyze_program(
    reference_asl: str,
    predicted_asl: str,
    scope: dict[str, Any],
    *,
    workspace_paths: Iterable[str] = (),
    condition: str = "whole_program",
) -> dict[str, Any]:
    """Decompose one ASL prediction without model-based adjudication."""

    reference = _program_view(reference_asl, scope)
    predicted = _program_view(predicted_asl, scope)
    rv, pv = reference["validation"], predicted["validation"]
    errors: list[dict[str, Any]] = []

    def add(error_type: str, confidence: float = 1.0, **details: Any) -> None:
        record = {"type": error_type, "source": "deterministic", "confidence": confidence}
        record.update(details)
        if not any(item["type"] == error_type for item in errors):
            errors.append(record)

    result: dict[str, Any] = {
        "parse_valid": bool(pv["syntax_verified"]),
        "lowerable_to_ccir": bool(pv["lower_verified"]),
        "type_valid": bool(pv["type_verified"]),
        "executable": bool(pv["execution_verified"]),
        "condition": condition,
        "errors": errors,
        "measurement_limits": {
            "coreference": "not_measurable_without_mention_annotations",
            "dimension_compatibility": "not_measurable_with_current_type_system",
        },
    }
    if not predicted_asl.strip():
        add("surface_no_asl")
    elif not result["parse_valid"]:
        add("surface_malformed_syntax")
    elif not result["lowerable_to_ccir"]:
        add("lowering_unsupported_form")
    elif not result["type_valid"]:
        add("type_invalid")

    path_counters = _path_counters(reference), _path_counters(predicted)
    for name in (
        "entity",
        "attribute",
        "qualifier",
        "temporal_qualifier",
        "cardinality_qualifier",
        "status_qualifier",
        "unit",
    ):
        result[f"{name}_metrics"] = _counter_metric(path_counters[0][name], path_counters[1][name])
    if result["entity_metrics"]["f1"] < 1:
        add("entity_grounding")
    if result["attribute_metrics"]["f1"] < 1:
        add("attribute_grounding")
    if result["qualifier_metrics"]["f1"] < 1:
        add("qualifier_grounding")
    if result["temporal_qualifier_metrics"]["f1"] < 1:
        add("temporal_qualifier")
    if result["cardinality_qualifier_metrics"]["f1"] < 1:
        add("cardinality_qualifier")

    alignment = _align_paths(reference, predicted)
    mapping = alignment["mapping"]
    reverse_mapping = {predicted_path: reference_path for reference_path, predicted_path in mapping.items()}
    result["symbol_alignment"] = alignment
    result["path_exact_metrics"] = _counter_metric(
        Counter(reference["paths"]), Counter(predicted["paths"])
    )
    aligned_structure = _relation_signatures(reference) == _relation_signatures(
        predicted, reverse_mapping
    )
    aligned_return = (
        reference["return_kind"] == predicted["return_kind"]
        and (
            reference["return"]
            == (
                _translated(str(predicted["return"]), reverse_mapping)
                if predicted["return_kind"] == "path"
                else predicted["return"]
            )
        )
    )
    result["semantic_name_equivalent"] = (
        len(mapping) == len(reference["paths"]) == len(predicted["paths"])
        and not alignment["ambiguous_reference_paths"]
        and aligned_structure
        and aligned_return
    )
    if alignment["ambiguous_reference_paths"]:
        add("ambiguous_symbol_alignment", 0.5)
    if alignment["unresolved_reference_paths"]:
        add("unresolved_symbol_equivalence", 0.5)

    reference_relations = _relation_by_target(reference)
    predicted_relations = _relation_by_target(predicted)
    operator_reference = Counter(reference["operators"])
    operator_predicted = Counter(predicted["operators"])
    result["operator_metrics"] = _counter_metric(operator_reference, operator_predicted)
    if result["operator_metrics"]["f1"] < 1:
        add("operator_mapping")

    reference_edges: Counter[Any] = Counter()
    predicted_edges: Counter[Any] = Counter()
    for relation in reference["relations"]:
        reference_edges.update((source, relation["target"], relation["operator"]) for source in relation["refs"])
    for relation in predicted["relations"]:
        predicted_edges.update(
            (
                _translated(source, reverse_mapping),
                _translated(relation["target"], reverse_mapping),
                relation["operator"],
            )
            for source in relation["refs"]
        )
    result["dependency_metrics"] = _counter_metric(reference_edges, predicted_edges)
    if result["dependency_metrics"]["f1"] < 1:
        add("dependency_incomplete")

    participant_correct = 0
    participant_support = 0
    direction_correct = 0
    direction_support = 0
    argument_correct = 0
    argument_support = 0
    constant_correct = 0
    constant_support = 0
    inter_correct = inter_support = intra_correct = intra_support = 0
    confusion: Counter[tuple[str, str]] = Counter()
    reference_ancestors = _ancestors(reference)
    collapse_count = collapse_value_correct = shortcut_count = 0
    reference_values = _execution_values(reference, scope)
    predicted_values = _execution_values(predicted, scope)
    for target, relation in reference_relations.items():
        predicted_target = mapping.get(target)
        candidate = predicted_relations.get(predicted_target or "")
        if candidate is None:
            for source in relation["refs"]:
                inverse = predicted_relations.get(mapping.get(source, ""))
                if inverse is None or mapping.get(target) not in inverse["refs"]:
                    continue
                direction_support += 1
                add("relation_direction")
                if any(
                    category == "temporal"
                    for path in (target, source)
                    for category, _ in _path_features(path)["qualifiers"]
                ):
                    add("temporal_semantics")
            continue
        confusion[(relation["operator"], candidate["operator"])] += 1
        translated_refs = [_translated(path, reverse_mapping) for path in candidate["refs"]]
        if relation["refs"]:
            participant_support += 1
            if Counter(relation["refs"]) == Counter(translated_refs):
                participant_correct += 1
            for source in relation["refs"]:
                direction_support += 1
                if source in translated_refs:
                    direction_correct += 1
                elif any(
                    item.get("target") == mapping.get(source)
                    and mapping.get(target) in item.get("refs", [])
                    for item in predicted["relations"]
                ):
                    add("relation_direction")
            relation_kind = (
                "intra" if all(_path_features(source)["entity"] == _path_features(target)["entity"] for source in relation["refs"]) else "inter"
            )
            if relation_kind == "inter":
                inter_support += 1
                inter_correct += int(Counter(relation["refs"]) == Counter(translated_refs))
            else:
                intra_support += 1
                intra_correct += int(Counter(relation["refs"]) == Counter(translated_refs))
        if relation["operator"] in {"SUB", "DIV"}:
            argument_support += 1
            reference_leaves = relation["leaves"]
            predicted_leaves = [
                (kind, _translated(value, reverse_mapping) if kind == "REF" else value)
                for kind, value in candidate["leaves"]
            ]
            if reference_leaves == predicted_leaves:
                argument_correct += 1
            elif Counter(reference_leaves) == Counter(predicted_leaves):
                add("argument_order")
        if relation["constants"]:
            constant_support += 1
            if Counter(relation["constants"]) == Counter(candidate["constants"]):
                constant_correct += 1
            else:
                add("relation_constant")
        if relation["operator"] != "CONST" and candidate["operator"] == "CONST":
            collapse_count += 1
            add("premature_literal_collapse")
            if (
                target in reference_values
                and predicted_target in predicted_values
                and str(reference_values[target]) == str(predicted_values[predicted_target])
            ):
                collapse_value_correct += 1
                add("correct_value_lost_derivation")
        direct = set(relation["refs"])
        if any(source in reference_ancestors.get(target, set()) - direct for source in translated_refs):
            shortcut_count += 1
            add("transitive_shortcut")
    result["relation_participant_accuracy"] = _accuracy(participant_correct, participant_support)
    result["relation_direction_accuracy"] = _accuracy(direction_correct, direction_support)
    result["argument_order_accuracy"] = _accuracy(argument_correct, argument_support)
    result["relation_constant_accuracy"] = _accuracy(constant_correct, constant_support)
    result["inter_object_accuracy"] = _accuracy(inter_correct, inter_support)
    result["intra_object_accuracy"] = _accuracy(intra_correct, intra_support)
    result["operator_confusion"] = {
        f"{reference_op}->{predicted_op}": count
        for (reference_op, predicted_op), count in sorted(confusion.items())
    }
    result["dependency_collapse"] = {
        "count": collapse_count,
        "support": sum(item["operator"] != "CONST" for item in reference["relations"]),
        "correct_value_lost_derivation_count": collapse_value_correct,
    }
    result["transitive_shortcut"] = {
        "count": shortcut_count,
        "support": sum(bool(item["refs"]) for item in reference["relations"]),
    }

    reference_literals = Counter(value for _, value in reference["source_facts"])
    predicted_literals = Counter(value for _, value in predicted["source_facts"])
    result["source_literal_metrics"] = _counter_metric(reference_literals, predicted_literals)
    reference_attachments = Counter(reference["source_facts"])
    predicted_attachments = Counter(
        (_translated(target, reverse_mapping), value) for target, value in predicted["source_facts"]
    )
    result["source_fact_attachment_metrics"] = _counter_metric(
        reference_attachments, predicted_attachments
    )
    if result["source_fact_attachment_metrics"]["f1"] < 1:
        add("source_fact_grounding")
    if reference_literals - predicted_literals:
        add("omitted_number")
    if predicted_literals - reference_literals:
        add("hallucinated_or_mutated_number")
    if (reference_literals & predicted_literals) and result["source_fact_attachment_metrics"]["f1"] < result["source_literal_metrics"]["f1"]:
        add("number_attached_to_wrong_path")
    for predicted_target, _ in predicted["source_facts"]:
        reference_target = reverse_mapping.get(predicted_target)
        if reference_target and reference_relations.get(reference_target, {}).get("operator") != "CONST":
            add("derived_literal_presented_as_source_fact")

    reference_return = reference["return"] if reference["return_kind"] == "path" else None
    predicted_return = predicted["return"] if predicted["return_kind"] == "path" else None
    return_correct = bool(
        reference_return
        and predicted_return
        and _translated(str(predicted_return), reverse_mapping) == reference_return
    )
    result["return_target_accuracy"] = _accuracy(int(return_correct), int(reference_return is not None))
    if reference_return and not return_correct:
        if predicted["return_kind"] == "missing":
            add("return_missing")
        elif predicted["return_kind"] == "literal":
            add("return_literal_instead_of_slot")
        else:
            add("return_wrong_target")

    workspace = set(workspace_paths)
    expected_existing = {
        source
        for relation in reference["relations"]
        for source in relation["refs"]
        if source in workspace
    }
    exact_reused = sum(
        source in {item for relation in predicted["relations"] for item in relation["refs"]}
        for source in expected_existing
    )
    renamed = sum(
        mapping.get(source) is not None and mapping[source] != source for source in expected_existing
    )
    result["workspace_path_reuse_rate"] = _accuracy(exact_reused, len(expected_existing))
    result["workspace_symbol_stable"] = renamed == 0
    result["accidental_rename_count"] = renamed
    if renamed:
        add("forbidden_stateful_rename")
        add("unresolved_due_to_rename")
    duplicate_slots = sum(
        1
        for target in predicted_relations
        if reverse_mapping.get(target) in workspace and target != reverse_mapping.get(target)
    )
    result["duplicate_semantic_slot_count"] = duplicate_slots
    if duplicate_slots:
        add("duplicate_semantic_slot")

    aggregation_relations = [
        item
        for item in reference["relations"]
        if item["operator"] != "CONST"
        and (
            item["operator"] in _AGGREGATION
            or any(
                token in _CARDINALITY
                for path in [item["target"], *item["refs"]]
                for token in _path_features(path)["semantic_tokens"]
            )
        )
    ]
    aggregation_correct = 0
    for relation in aggregation_relations:
        candidate = predicted_relations.get(mapping.get(relation["target"], ""))
        if candidate and candidate["operator"] == relation["operator"] and Counter(
            relation["refs"]
        ) == Counter(_translated(path, reverse_mapping) for path in candidate["refs"]):
            aggregation_correct += 1
    result["aggregation_cardinality_accuracy"] = _accuracy(
        aggregation_correct, len(aggregation_relations)
    )
    if aggregation_relations and aggregation_correct != len(aggregation_relations):
        add("aggregation_cardinality")

    temporal_reference_paths = {
        path for path in reference["paths"] if any(cat == "temporal" for cat, _ in _path_features(path)["qualifiers"])
    }
    temporal_correct = sum(
        path in mapping
        and _path_features(path)["qualifiers"] == _path_features(mapping[path])["qualifiers"]
        for path in temporal_reference_paths
    )
    result["temporal_accuracy"] = _accuracy(temporal_correct, len(temporal_reference_paths))
    if temporal_reference_paths and temporal_correct != len(temporal_reference_paths):
        add("temporal_semantics")

    result["unit_support"] = result["unit_metrics"]["support"]
    result["dimension_compatibility"] = "not_measurable_with_current_type_system"
    result["scope_metrics"] = {
        "root_scope_correct": reference["scopes"][:1] == predicted["scopes"][:1],
        "cross_record_reference_count": 0,
        "note": "cross-record references are not representable in this frozen per-case replay",
    }
    result["complexity"] = {
        "statement_count": len(reference["relations"]) + int(reference["return_kind"] != "missing"),
        "dependency_depth": _dependency_depth(reference),
        "entity_count": len({_path_features(path)["entity"] for path in reference["paths"]}),
        "semantic_path_count": len(reference["paths"]),
        "operator_count": len(reference["operators"]),
        "unresolved_dependency_count": len(rv.get("errors", [])),
    }
    return result


def _pattern_families(reference_asl: str, scope: dict[str, Any]) -> list[str]:
    view = _program_view(reference_asl, scope)
    operators = set(view["operators"])
    tokens = {
        token for path in view["paths"] for token in _path_features(path)["semantic_tokens"]
    }
    families = set()
    if operators & {"ADD", "SUB"} and any(item["refs"] for item in view["relations"]):
        families.add("relative_quantity")
    if "DIV" in operators:
        families.add("ratio_fraction")
    if operators & {"PERCENT_OF", "PERCENT_CHANGE"} or "percent" in tokens:
        families.add("percentage")
    if operators & _AGGREGATION or "total" in tokens:
        families.add("aggregation")
    if "per" in tokens:
        families.add("rate")
    if "each" in tokens or ("DIV" in operators and "count" in tokens):
        families.add("equal_allocation")
    if any(_qualifier_category(token) == "temporal" for token in tokens):
        families.add("temporal_shift")
    if tokens & {"remaining", "difference", "change"}:
        families.add("remaining_difference")
    entities = {_path_features(path)["entity"] for path in view["paths"]}
    if len(entities) > 1:
        families.add("multi_entity")
    depth = _dependency_depth(view)
    if depth > 1:
        families.add("chain")
    if depth > 2:
        families.add("nested_dependency")
    return sorted(families or {"other"})


_F1_METRICS = (
    "entity_metrics",
    "attribute_metrics",
    "qualifier_metrics",
    "temporal_qualifier_metrics",
    "cardinality_qualifier_metrics",
    "status_qualifier_metrics",
    "unit_metrics",
    "path_exact_metrics",
    "operator_metrics",
    "dependency_metrics",
    "source_literal_metrics",
    "source_fact_attachment_metrics",
)
_ACCURACY_METRICS = (
    "relation_participant_accuracy",
    "relation_direction_accuracy",
    "argument_order_accuracy",
    "relation_constant_accuracy",
    "inter_object_accuracy",
    "intra_object_accuracy",
    "return_target_accuracy",
    "workspace_path_reuse_rate",
    "aggregation_cardinality_accuracy",
    "temporal_accuracy",
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    output: dict[str, Any] = {"count": len(rows), "surface_and_compute": {}}
    for name in ("parse_valid", "lowerable_to_ccir", "type_valid", "executable", "final_answer_correct", "semantic_state_equivalent"):
        support = sum(name in item for item in metrics)
        output["surface_and_compute"][name] = _accuracy(
            sum(bool(item.get(name)) for item in metrics), support
        )
    output["semantic"] = {}
    for name in _F1_METRICS:
        reference_count = sum(item[name]["reference_count"] for item in metrics)
        predicted_count = sum(item[name]["predicted_count"] for item in metrics)
        overlap = sum(item[name]["overlap"] for item in metrics)
        output["semantic"][name.removesuffix("_metrics")] = _counter_metric(
            Counter({"x": reference_count}), Counter({"x": overlap})
        )
        metric = output["semantic"][name.removesuffix("_metrics")]
        metric["precision"] = overlap / predicted_count if predicted_count else float(reference_count == 0)
        metric["recall"] = overlap / reference_count if reference_count else float(predicted_count == 0)
        metric["f1"] = (
            2 * metric["precision"] * metric["recall"] / (metric["precision"] + metric["recall"])
            if metric["precision"] + metric["recall"]
            else 0.0
        )
        metric["predicted_count"] = predicted_count
    for name in _ACCURACY_METRICS:
        correct = sum(item[name]["correct"] for item in metrics)
        support = sum(item[name]["support"] for item in metrics)
        output["semantic"][name.removesuffix("_accuracy")] = _accuracy(correct, support)
    output["semantic"]["semantic_name_equivalent"] = _accuracy(
        sum(bool(item["semantic_name_equivalent"]) for item in metrics), len(metrics)
    )
    stateful_metrics = [
        item for item in metrics if item["workspace_path_reuse_rate"]["support"] > 0
    ]
    output["semantic"]["workspace_symbol_stable"] = _accuracy(
        sum(bool(item["workspace_symbol_stable"]) for item in stateful_metrics),
        len(stateful_metrics),
    )
    output["error_counts"] = dict(
        sorted(Counter(error["type"] for item in metrics for error in item["errors"]).items())
    )
    confusion = Counter()
    for item in metrics:
        confusion.update(item["operator_confusion"])
    output["operator_confusion"] = dict(sorted(confusion.items()))
    output["conditional"] = _conditional(metrics)
    transition_metrics = [
        transition["metrics"] for row in rows for transition in row.get("transitions", [])
    ]
    if transition_metrics:
        transition_summary = _aggregate(
            [{"metrics": item, "transitions": []} for item in transition_metrics]
        )
        output["transition_count"] = len(transition_metrics)
        output["transition_analysis"] = transition_summary
        output["semantic"]["workspace_path_reuse_rate"] = transition_summary["semantic"][
            "workspace_path_reuse_rate"
        ]
        output["semantic"]["workspace_symbol_stable"] = transition_summary["semantic"][
            "workspace_symbol_stable"
        ]
    return output


def _conditional(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    def probability(predicate, condition) -> dict[str, Any]:
        selected = [item for item in metrics if condition(item)]
        return _accuracy(sum(bool(predicate(item)) for item in selected), len(selected))

    valid = lambda item: item["parse_valid"] and item["type_valid"]
    operator_correct = lambda item: item["operator_metrics"]["f1"] == 1.0
    path_correct = lambda item: item["semantic_name_equivalent"]
    semantic_correct = lambda item: bool(item.get("semantic_state_equivalent"))
    answer_correct = lambda item: bool(item.get("final_answer_correct"))
    return {
        "operator_correct_given_parse_type": probability(operator_correct, valid),
        "path_correct_given_parse_type": probability(path_correct, valid),
        "direction_correct_given_operator_correct": probability(
            lambda item: item["relation_direction_accuracy"]["accuracy"] == 1.0,
            lambda item: operator_correct(item) and item["relation_direction_accuracy"]["support"] > 0,
        ),
        "answer_correct_given_semantic_state_correct": probability(answer_correct, semantic_correct),
        "answer_correct_given_semantic_state_incorrect": probability(
            answer_correct, lambda item: not semantic_correct(item)
        ),
        "answer_correct_given_operator_correct_path_wrong": probability(
            answer_correct, lambda item: operator_correct(item) and not path_correct(item)
        ),
    }


def _attach_existing_metrics(metrics: dict[str, Any], scored: dict[str, Any]) -> None:
    old = scored.get("metrics", {})
    for name in (
        "parse_valid",
        "lowerable_to_ccir",
        "type_valid",
        "executable",
        "final_answer_correct",
        "semantic_state_equivalent",
        "semantic_return_equivalent",
    ):
        metrics[name] = bool(old.get(name))


def _program_record(
    row: dict[str, Any],
    reference: dict[str, Any],
    *,
    condition: str,
    program: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = analyze_program(
        reference["reference_asl"],
        str(row.get("predicted_asl", "")),
        reference["effective_scope"],
        condition=condition,
    )
    _attach_existing_metrics(metrics, row)
    alignments = [
        {
            "example_id": row["example_id"],
            "condition": condition,
            **metrics["symbol_alignment"],
        }
    ]
    judges = []
    unresolved = [
        *metrics["symbol_alignment"]["ambiguous_reference_paths"],
        *metrics["symbol_alignment"]["unresolved_reference_paths"],
    ]
    if unresolved:
        judges.append(
            {
                "example_id": row["example_id"],
                "dataset": row["dataset"],
                "condition": condition,
                "reason": "ambiguous_deterministic_symbol_alignment",
                "reference_paths": sorted(set(unresolved)),
                "adjudication_status": "pending_optional_strong_judge",
            }
        )
    transitions = []
    if program is not None:
        traces = {int(trace["part_id"]): trace for trace in row.get("part_traces", [])}
        workspace_paths: set[str] = set()
        for mapping in program["part_mappings"]:
            part_id = int(mapping["part_id"])
            reference_delta = "\n".join(mapping["asl"])
            trace = traces.get(part_id, {})
            transition_metrics = analyze_program(
                reference_delta,
                str(trace.get("predicted_delta", "")),
                reference["effective_scope"],
                workspace_paths=workspace_paths,
                condition=condition,
            )
            transitions.append({"part_id": part_id, "metrics": transition_metrics})
            workspace_paths.update(_program_view(reference_delta, reference["effective_scope"])["paths"])
    record = {
        "schema_version": "ccpu.paper1.semantic_failure_record.v1",
        "example_id": row["example_id"],
        "source_id": row["parent_source_id"],
        "dataset": row["dataset"],
        "condition": condition,
        "semantic_pattern_id": row["semantic_pattern_id"],
        "pattern_families": _pattern_families(reference["reference_asl"], reference["effective_scope"]),
        "metrics": metrics,
        "transitions": transitions,
    }
    return record, alignments, judges


def _by_groups(rows: list[dict[str, Any]], key) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for value in key(row):
            groups[str(value)].append(row)
    return {name: _aggregate(members) for name, members in sorted(groups.items())}


def _complexity_groups(row: dict[str, Any]) -> list[str]:
    complexity = row["metrics"]["complexity"]
    statements = complexity["statement_count"]
    depth = complexity["dependency_depth"]
    entities = complexity["entity_count"]
    return [
        f"statements:{'1-4' if statements <= 4 else '5-8' if statements <= 8 else '9+'}",
        f"depth:{'0-1' if depth <= 1 else '2' if depth == 2 else '3+'}",
        f"entities:{'1' if entities <= 1 else '2+'}",
    ]


def _teacher_consistency(paths: list[str | Path]) -> dict[str, Any]:
    programs = []
    for path in paths:
        for row in read_jsonl(path):
            asl = row.get("asl")
            if not asl and row.get("part_mappings"):
                asl = "\n".join(line for item in row["part_mappings"] for line in item["asl"])
            if asl:
                programs.append(str(asl))
    shapes: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for asl in programs:
        for match in re.findall(r"\b[a-z_][a-z0-9_.]*\b", asl.casefold()):
            if "." not in match:
                continue
            features = _path_features(match)
            shapes[features["attributes"]][match] += 1
    clusters = [
        {"semantic_tokens": list(tokens), "variants": len(values), "examples": sorted(values)[:5]}
        for tokens, values in shapes.items()
        if tokens and len(values) > 1
    ]
    total = sum(sum(values.values()) for values in shapes.values())
    entropy = 0.0
    if total:
        for values in shapes.values():
            count = sum(values.values())
            probability = count / total
            entropy -= probability * math.log2(probability)
    return {
        "program_count": len(programs),
        "canonical_attribute_shapes": len(shapes),
        "likely_synonym_or_shape_clusters": clusters[:25],
        "path_shape_entropy_bits": entropy,
        "automatic_rewrite_performed": False,
    }


def analyze_saved_semantic_failures(
    *,
    eval_path: str | Path,
    programs_path: str | Path,
    whole_scored_path: str | Path,
    predicted_scored_path: str | Path,
    oracle_scored_path: str | Path,
    output_dir: str | Path,
    teacher_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    references = read_jsonl(eval_path)
    by_example = {row["example_id"]: row for row in references}
    by_source = {str(row["parent_source_id"]): row for row in references}
    programs = {str(row["source_id"]): row for row in read_jsonl(programs_path)}
    conditions = {
        "whole_lora500": read_jsonl(whole_scored_path),
        "incremental_predicted": read_jsonl(predicted_scored_path),
        "incremental_oracle": read_jsonl(oracle_scored_path),
    }
    output = Path(output_dir)
    all_rows = []
    alignment_rows = []
    judge_rows = []
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    filenames = {
        "whole_lora500": "whole_lora500.jsonl",
        "incremental_predicted": "incremental_predicted.jsonl",
        "incremental_oracle": "incremental_oracle.jsonl",
    }
    for condition, scored_rows in conditions.items():
        records = []
        for row in scored_rows:
            reference = by_example.get(row["example_id"]) or by_source[str(row["parent_source_id"])]
            program = None if condition == "whole_lora500" else programs[str(row["parent_source_id"])]
            record, alignments, judges = _program_record(
                row, reference, condition=condition, program=program
            )
            records.append(record)
            alignment_rows.extend(alignments)
            judge_rows.extend(judges)
        write_jsonl(output / filenames[condition], records)
        condition_rows[condition] = records
        all_rows.extend(records)
    write_jsonl(output / "symbol_alignment.jsonl", alignment_rows)
    write_jsonl(output / "judge_queue.jsonl", judge_rows)

    summaries = {name: _aggregate(rows) for name, rows in condition_rows.items()}
    primary_metrics = {
        "entity": ("semantic", "entity", "f1"),
        "attribute": ("semantic", "attribute", "f1"),
        "qualifier": ("semantic", "qualifier", "f1"),
        "path_reuse": ("semantic", "workspace_path_reuse_rate", "accuracy"),
        "operator": ("semantic", "operator", "f1"),
        "argument_order": ("semantic", "argument_order", "accuracy"),
        "relation_direction": ("semantic", "relation_direction", "accuracy"),
        "source_fact": ("semantic", "source_fact_attachment", "f1"),
        "dependency": ("semantic", "dependency", "f1"),
        "aggregation_cardinality": ("semantic", "aggregation_cardinality", "accuracy"),
        "temporal": ("semantic", "temporal", "accuracy"),
        "return_target": ("semantic", "return_target", "accuracy"),
    }
    deltas = {}
    whole = summaries["whole_lora500"]
    for candidate in ("incremental_predicted", "incremental_oracle"):
        deltas[candidate] = {}
        for name, path in primary_metrics.items():
            left: Any = whole
            right: Any = summaries[candidate]
            for part in path:
                left = left[part]
                right = right[part]
            deltas[candidate][name] = None if left is None or right is None else right - left

    error_examples: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        for error in row["metrics"]["errors"]:
            entry = error_examples.setdefault(error["type"], {"count": 0, "examples": []})
            entry["count"] += 1
            if len(entry["examples"]) < 5:
                entry["examples"].append(
                    {
                        "example_id": row["example_id"],
                        "source_id": row["source_id"],
                        "dataset": row["dataset"],
                        "condition": row["condition"],
                    }
                )
    summary = {
        "schema_version": "ccpu.paper1.semantic_failure_analysis.v1",
        "conditions": summaries,
        "incremental_minus_whole": deltas,
        "teacher_consistency": _teacher_consistency(list(teacher_paths)),
        "judge_queue_count": len(judge_rows),
        "input_sha256": {
            "eval": file_sha256(eval_path),
            "programs": file_sha256(programs_path),
            "whole_scored": file_sha256(whole_scored_path),
            "predicted_scored": file_sha256(predicted_scored_path),
            "oracle_scored": file_sha256(oracle_scored_path),
            "teachers": {str(path): file_sha256(path) for path in teacher_paths},
        },
        "full_artifact_sha256": {
            filename: file_sha256(output / filename)
            for filename in [
                *filenames.values(),
                "symbol_alignment.jsonl",
                "judge_queue.jsonl",
            ]
        },
        "claim_boundary": (
            "deterministic 25-program diagnostic; symbol alignment is bounded and optional "
            "judge cases remain unresolved"
        ),
    }
    by_dataset = {
        condition: _by_groups(rows, lambda row: [row["dataset"]])
        for condition, rows in condition_rows.items()
    }
    by_pattern = {
        condition: _by_groups(rows, lambda row: row["pattern_families"])
        for condition, rows in condition_rows.items()
    }
    by_complexity = {
        condition: _by_groups(rows, _complexity_groups)
        for condition, rows in condition_rows.items()
    }
    write_json(output / "summary.json", summary)
    write_json(output / "by_dataset.json", by_dataset)
    write_json(output / "by_pattern.json", by_pattern)
    write_json(output / "by_complexity.json", by_complexity)
    write_json(output / "error_examples.json", dict(sorted(error_examples.items())))
    return summary
