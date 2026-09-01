"""Deterministic representation diagnostics for matched ASL and functor runs."""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

from .functor_runtime import (
    Call,
    functor_registry,
    lower_f1,
    lower_f2,
    parse_functor_program,
    validate_functor_program,
)
from .semantic_failure import _path_features, _program_view, analyze_program

_CALL_NAME = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)
_ASYMMETRIC = {
    "offset",
    "difference",
    "quotient",
    "multiple",
    "fraction_of",
    "percent_of",
    "percentage_ratio",
    "increase_percent",
    "decrease_percent",
    "rate_total",
    "per_unit_total",
    "remaining",
}
_ROLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "given": ("target", "amount"),
    "same": ("target", "source"),
    "offset": ("target", "source", "amount"),
    "difference": ("target", "minuend", "subtrahend"),
    "absolute_difference": ("target", "left", "right"),
    "sum_of": ("target", "member"),
    "product_of": ("target", "member"),
    "quotient": ("target", "numerator", "denominator"),
    "multiple": ("target", "base", "factor"),
    "fraction_of": ("target", "base", "numerator", "denominator"),
    "percent_of": ("target", "base", "percent"),
    "percentage_ratio": ("target", "part", "whole"),
    "increase_percent": ("target", "base", "percent"),
    "decrease_percent": ("target", "base", "percent"),
    "rate_total": ("target", "rate", "duration"),
    "per_unit_total": ("target", "per_unit", "count"),
    "remaining": ("target", "whole", "used"),
    "mean_of": ("target", "member"),
    "minimum_of": ("target", "member"),
    "maximum_of": ("target", "member"),
    "query": ("query_target",),
}


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["parent_source_id"])


def _rate(correct: int, support: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "support": support,
        "rate": correct / support if support else None,
    }


def _prf(overlap: int, reference: int, predicted: int) -> dict[str, Any]:
    precision = overlap / predicted if predicted else float(reference == 0)
    recall = overlap / reference if reference else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "overlap": overlap,
        "reference_count": reference,
        "predicted_count": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _counter_prf(reference: Iterable[Any], predicted: Iterable[Any]) -> dict[str, Any]:
    left, right = Counter(reference), Counter(predicted)
    return _prf(sum((left & right).values()), sum(left.values()), sum(right.values()))


def _wilson(correct: int, support: int) -> list[float] | None:
    if not support:
        return None
    z = 1.959963984540054
    p = correct / support
    denominator = 1 + z * z / support
    center = (p + z * z / (2 * support)) / denominator
    margin = z * math.sqrt(p * (1 - p) / support + z * z / (4 * support**2)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _paired_exact(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if not discordant:
        return None
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def _mean(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return mean(selected) if selected else None


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _calls(program: str, condition: str) -> tuple[list[Call], str | None]:
    try:
        return parse_functor_program(program, condition), None
    except (TypeError, ValueError) as error:
        return [], str(error)


def _call_paths(calls: Iterable[Call]) -> set[str]:
    return {
        value
        for call in calls
        for value in call.args
        if isinstance(value, str)
    }


def _role_names(call: Call) -> list[str]:
    schema = _ROLE_SCHEMAS.get(call.name, ("target",))
    if len(schema) == len(call.args):
        return list(schema)
    if len(schema) == 2 and schema[-1] in {"member", "used"}:
        return [schema[0], *[schema[1]] * (len(call.args) - 1)]
    return [*schema, *["extra"] * max(0, len(call.args) - len(schema))]


def _surface(row: dict[str, Any], condition: str) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    program_key = "predicted_asl" if condition == "f0" else "predicted_program"
    program = str(row.get(program_key, ""))
    raw = str(row.get("generated_text", program))
    balanced = raw.count("(") == raw.count(")") and raw.count("[") == raw.count("]")
    if condition == "f0":
        statements = [line for line in program.splitlines() if line.strip()]
        malformed = int(not metrics.get("parse_valid"))
        return {
            "parse_valid": bool(metrics.get("parse_valid")),
            "ast_constructed": bool(metrics.get("parse_valid")),
            "balanced_delimiters": balanced,
            "canonical_statement_count": int(metrics.get("ast_record_count", len(statements))),
            "malformed_construct_count": malformed,
            "candidate_token_count": len(statements),
            "unknown_token_count": sum("unsupported" in str(error).casefold() for error in metrics.get("errors", [])),
            "invalid_token_rate": malformed / max(1, len(statements)),
        }
    registry = functor_registry(condition)
    names = _CALL_NAME.findall(raw)
    unknown = [name for name in names if name not in registry]
    calls, error = _calls(program, condition)
    arity_error = bool(error and "expects" in error)
    malformed = int(not metrics.get("parse_valid"))
    return {
        "parse_valid": bool(metrics.get("parse_valid")),
        "ast_constructed": bool(calls),
        "balanced_delimiters": balanced,
        "registered_functor_syntax": bool(names) and not unknown,
        "arity_valid": bool(calls) or not arity_error,
        "positional_argument_syntax": not bool(error and "positional" in error),
        "canonical_statement_count": len(calls),
        "malformed_construct_count": malformed,
        "candidate_token_count": len(names),
        "unknown_functor_count": len(unknown),
        "invalid_token_rate": len(unknown) / len(names) if names else float(malformed),
        "parse_error": error,
    }


def _workspace(asl: str, scope: dict[str, Any]) -> dict[str, str]:
    if not asl:
        return {}
    validation = validate_asl(asl, effective_scope=scope)
    execution = validation.get("execution") or {}
    workspace = execution.get("workspace", {}).get(str(scope["id"]), {})
    values = workspace.get("values", {})
    return {str(path): str(value) for path, value in values.items()} if isinstance(values, dict) else {}


def _state_metrics(reference_asl: str, predicted_asl: str, scope: dict[str, Any]) -> dict[str, Any]:
    reference = _workspace(reference_asl, scope)
    predicted = _workspace(predicted_asl, scope)
    facts = _counter_prf(reference.items(), predicted.items())
    entity_facts = _counter_prf(
        ((_path_features(path)["entity"], value) for path, value in reference.items()),
        ((_path_features(path)["entity"], value) for path, value in predicted.items()),
    )
    attribute_facts = _counter_prf(
        (
            (_path_features(path)["attributes"], _path_features(path)["qualifiers"], value)
            for path, value in reference.items()
        ),
        (
            (_path_features(path)["attributes"], _path_features(path)["qualifiers"], value)
            for path, value in predicted.items()
        ),
    )
    view = _program_view(predicted_asl, scope) if predicted_asl else {"relations": [], "return": None}
    targets = [relation["target"] for relation in view["relations"]]
    duplicate_slots = sum(count - 1 for count in Counter(targets).values() if count > 1)
    source_values: dict[str, set[str]] = defaultdict(set)
    for relation in view["relations"]:
        if relation["operator"] == "CONST":
            source_values[relation["target"]].update(relation["constants"])
    conflicts = sum(len(values) > 1 for values in source_values.values())
    entities = {_path_features(path)["entity"] for path in predicted}
    return {
        "blackboard_precision": facts["precision"],
        "blackboard_recall": facts["recall"],
        "blackboard_f1": facts["f1"],
        "entity_state": entity_facts,
        "attribute_state": attribute_facts,
        "value_attachment": facts,
        "canonical_entity_count": len(entities),
        "canonical_slot_count": len(predicted),
        "duplicate_entity_rate": 0.0,
        "duplicate_slot_rate": duplicate_slots / max(1, len(targets)),
        "conflicting_fact_count": conflicts,
        "blackboard_completeness": facts["recall"],
        "spurious_state_rate": 1 - facts["precision"],
        "semantic_state_equivalent": facts["f1"] == 1.0,
        "reference_facts": len(reference),
        "predicted_facts": len(predicted),
    }


def _semantic_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    names = (
        "entity_metrics",
        "attribute_metrics",
        "qualifier_metrics",
        "operator_metrics",
        "dependency_metrics",
        "relation_participant_accuracy",
        "relation_direction_accuracy",
        "argument_order_accuracy",
        "relation_constant_accuracy",
        "return_target_accuracy",
        "source_literal_metrics",
        "source_fact_attachment_metrics",
        "path_exact_metrics",
    )
    return {name.removesuffix("_metrics"): metrics[name] for name in names if name in metrics}


def _f1_isomorphism(reference_asl: str, predicted_asl: str, scope: dict[str, Any]) -> dict[str, Any]:
    metrics = analyze_program(reference_asl, predicted_asl, scope, condition="f1_isomorphism")
    checks = {
        "target_path": metrics["path_exact_metrics"]["f1"] == 1.0,
        "operator": metrics["operator_metrics"]["f1"] == 1.0,
        "source_reference": metrics["relation_participant_accuracy"]["accuracy"] == 1.0,
        "dependency_graph": metrics["dependency_metrics"]["f1"] == 1.0,
        "return_target": metrics["return_target_accuracy"]["accuracy"] == 1.0,
    }
    return {"checks": checks, "f1_asl_isomorphic_accuracy": all(checks.values())}


def _translated(value: Any, reverse: dict[str, str]) -> Any:
    if isinstance(value, str):
        return reverse.get(value, f"UNALIGNED:{value}")
    return str(value)


def _f2_call_metrics(gold_program: str, predicted_program: str, scope: dict[str, Any]) -> dict[str, Any]:
    gold, _ = _calls(gold_program, "f2")
    predicted, parse_error = _calls(predicted_program, "f2")
    if not predicted:
        gold_relations = [call for call in gold if call.name != "query"]
        return {
            "gold_count": len(gold),
            "predicted_count": 0,
            "class_pairs": [(call.name, "MISSING") for call in gold_relations],
            "role_correct": 0,
            "role_support": sum(len(call.args) for call in gold_relations),
            "binding_correct": 0,
            "binding_support": len(gold_relations),
            "direction_correct": 0,
            "direction_support": sum(call.name in _ASYMMETRIC for call in gold_relations),
            "query_correct": False,
            "call_records": [
                {
                    "gold_functor": call.name,
                    "predicted_functor": "MISSING",
                    "class_correct": False,
                    "binding_exact": False,
                    "direction_correct": False if call.name in _ASYMMETRIC else None,
                }
                for call in gold_relations
            ],
            "parse_error": parse_error,
        }
    gold_asl, predicted_asl = lower_f2(gold), lower_f2(predicted)
    alignment = analyze_program(gold_asl, predicted_asl, scope)["symbol_alignment"]["mapping"]
    reverse = {predicted_path: gold_path for gold_path, predicted_path in alignment.items()}
    predicted_by_target = {
        str(call.args[0]): call for call in predicted if call.name != "query" and call.args
    }
    class_pairs: list[tuple[str, str]] = []
    role_correct = role_support = binding_correct = binding_support = 0
    direction_correct = direction_support = 0
    role_confusion: Counter[str] = Counter()
    call_records = []
    matched_targets = set()
    for gold_call in (call for call in gold if call.name != "query"):
        target = str(gold_call.args[0])
        predicted_target = alignment.get(target, target)
        candidate = predicted_by_target.get(predicted_target)
        class_pairs.append((gold_call.name, candidate.name if candidate else "MISSING"))
        binding_support += 1
        if candidate is None:
            role_support += len(gold_call.args)
            call_records.append(
                {
                    "gold_functor": gold_call.name,
                    "predicted_functor": "MISSING",
                    "class_correct": False,
                    "binding_exact": False,
                    "direction_correct": False
                    if gold_call.name in _ASYMMETRIC
                    else None,
                }
            )
            continue
        matched_targets.add(predicted_target)
        gold_roles = _role_names(gold_call)
        predicted_roles = _role_names(candidate)
        candidate_by_role: dict[str, list[Any]] = defaultdict(list)
        for role, value in zip(predicted_roles, candidate.args):
            candidate_by_role[role].append(_translated(value, reverse))
        exact = gold_call.name == candidate.name
        for role, value in zip(gold_roles, gold_call.args):
            role_support += 1
            translated = value if isinstance(value, str) else str(value)
            options = candidate_by_role.get(role, [])
            if translated in options:
                role_correct += 1
                options.remove(translated)
            else:
                observed = next(
                    (name for name, values in candidate_by_role.items() if translated in values),
                    "missing",
                )
                role_confusion[f"{role}->{observed}"] += 1
                exact = False
        binding_correct += int(exact and len(gold_call.args) == len(candidate.args))
        binding_exact = exact and len(gold_call.args) == len(candidate.args)
        if gold_call.name in _ASYMMETRIC:
            direction_support += 1
            direction_correct += int(exact)
        call_records.append(
            {
                "gold_functor": gold_call.name,
                "predicted_functor": candidate.name,
                "class_correct": gold_call.name == candidate.name,
                "binding_exact": binding_exact,
                "direction_correct": exact if gold_call.name in _ASYMMETRIC else None,
            }
        )
    for call in predicted:
        if call.name != "query" and str(call.args[0]) not in matched_targets:
            class_pairs.append(("EXTRA", call.name))
    gold_query = gold[-1]
    predicted_query = predicted[-1]
    query_correct = (
        predicted_query.name == "query"
        and _translated(predicted_query.args[0], reverse) == gold_query.args[0]
    )
    class_correct = sum(left == right for left, right in class_pairs)
    gold_classes = sum(left != "EXTRA" for left, _ in class_pairs)
    predicted_classes = sum(right != "MISSING" for _, right in class_pairs)
    return {
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "class_pairs": class_pairs,
        "class_metrics": _prf(class_correct, gold_classes, predicted_classes),
        "role_correct": role_correct,
        "role_support": role_support,
        "binding_correct": binding_correct,
        "binding_support": binding_support,
        "direction_correct": direction_correct,
        "direction_support": direction_support,
        "query_correct": query_correct,
        "call_records": call_records,
        "role_confusion": dict(sorted(role_confusion.items())),
        "parse_error": parse_error,
    }


def _runtime(
    program: str,
    condition: str,
    scope: dict[str, Any],
    calls: list[Call],
) -> dict[str, Any]:
    if condition == "f0":
        validation = validate_asl(program, effective_scope=scope)
        operations = len((validation.get("ccir") or {}).get("operations", []))
        errors = [str(error) for error in validation.get("errors", [])]
        return {
            "lowerable": bool(validation["lower_verified"]),
            "type_valid": bool(validation["type_verified"]),
            "executable": bool(validation["execution_verified"]),
            "registered_functor": None,
            "runtime_operation_count": operations,
            "runtime_operations_per_model_statement": operations / max(1, operations),
            "unresolved_constraint_count": 0,
            "unsupported_relation_count": sum("unsupported" in error.casefold() for error in errors),
            "ambiguous_lowering_count": 0,
            "errors": errors,
        }
    validation = validate_functor_program(program, condition, effective_scope=scope)
    lowered = str(validation["lowered_asl"])
    lowered_view = _program_view(lowered, scope) if lowered else {"relations": []}
    model_targets = {
        str(call.args[0]) for call in calls if call.name not in {"given", "value", "query"}
    }
    runtime_targets = {relation["target"] for relation in lowered_view["relations"]}
    errors = [str(error) for error in validation["errors"]]
    semantic_decisions = sum(call.name != "query" for call in calls)
    return {
        "lowerable": bool(validation["lowerable"]),
        "type_valid": bool(validation["type_valid"]),
        "executable": bool(validation["executable"]),
        "registered_functor": bool(calls),
        "deterministic_lowering_coverage": float(bool(validation["lowerable"])),
        "model_semantic_decisions": semantic_decisions,
        "runtime_lowered_decisions": len(runtime_targets),
        "runtime_operation_count": len(lowered_view["relations"]),
        "runtime_operations_per_functor": len(lowered_view["relations"])
        / max(1, semantic_decisions),
        "runtime_created_canonical_slots": len(runtime_targets - model_targets),
        "unresolved_constraint_count": sum("underdetermined" in error for error in errors),
        "unsupported_relation_count": sum("unsupported" in error.casefold() for error in errors),
        "ambiguous_lowering_count": 0,
        "errors": errors,
        "lowered_asl_bytes": len(lowered.encode("utf-8")),
    }


def _complexity(program: str, condition: str, generated_tokens: int) -> dict[str, Any]:
    if condition == "f0":
        statements = [line for line in program.splitlines() if line.strip()]
        paths = set(re.findall(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+\b", program))
        return {
            "generated_tokens": generated_tokens,
            "statement_count": len(statements),
            "ast_node_count": len(statements),
            "mean_arity": None,
            "max_nesting": max((line.count("(") for line in statements), default=0),
            "distinct_path_count": len(paths),
        }
    calls, _ = _calls(program, condition)
    nodes = 0
    for line in program.splitlines():
        try:
            nodes += sum(1 for _ in ast.walk(ast.parse(line, mode="eval")))
        except SyntaxError:
            pass
    paths = _call_paths(calls)
    return {
        "generated_tokens": generated_tokens,
        "statement_count": len(calls),
        "ast_node_count": nodes,
        "mean_arity": _mean(len(call.args) for call in calls),
        "max_nesting": 1 if calls else 0,
        "distinct_path_count": len(paths),
        "functor_count": sum(call.name != "query" for call in calls),
    }


def _record(
    condition: str,
    scored: dict[str, Any],
    eval_row: dict[str, Any],
    common_reference_asl: str,
    f2_reference_program: str,
) -> dict[str, Any]:
    scope = eval_row["effective_scope"]
    key = "predicted_asl" if condition == "f0" else "predicted_program"
    program = str(scored.get(key, ""))
    calls, _ = _calls(program, condition) if condition != "f0" else ([], None)
    if condition == "f0":
        structural_asl = executable_asl = program
    elif calls:
        structural_asl = lower_f1(calls) if condition == "f1" else lower_f2(calls)
        validated = validate_functor_program(program, condition, effective_scope=scope)
        executable_asl = str(validated["lowered_asl"])
    else:
        structural_asl = executable_asl = ""
    semantic = analyze_program(
        common_reference_asl,
        structural_asl,
        scope,
        condition=f"common_{condition}",
    )
    state = _state_metrics(common_reference_asl, executable_asl, scope)
    state["dependency_f1"] = semantic["dependency_metrics"]["f1"]
    state["canonical_slot_correctness"] = state["value_attachment"]["f1"]
    state["query_target_correctness"] = semantic["return_target_accuracy"]["accuracy"]
    state["unresolved_symbol_rate"] = len(
        semantic["symbol_alignment"]["unresolved_reference_paths"]
    ) / max(1, semantic["path_exact_metrics"]["reference_count"])
    metrics = scored["metrics"]
    result = {
        "schema_version": "ccpu.paper1.functor_normalized.v2",
        "example_id": scored["example_id"],
        "parent_source_id": scored["parent_source_id"],
        "semantic_pattern_id": scored["semantic_pattern_id"],
        "dataset": scored["dataset"],
        "condition": condition,
        "surface": _surface(scored, condition),
        "semantic_structure": _semantic_subset(semantic),
        "semantic_name_equivalent": semantic["semantic_name_equivalent"],
        "runtime": _runtime(program, condition, scope, calls),
        "blackboard": state,
        "final": {
            "executable": bool(metrics.get("executable")),
            "answer_correct": bool(metrics.get("final_answer_correct")),
            "exact_answer": bool(metrics.get("final_answer_correct")),
            "fail_closed": not bool(metrics.get("executable")),
        },
        "complexity": _complexity(program, condition, int(scored.get("generated_tokens", 0))),
        "prompt_tokens": int(scored.get("prompt_tokens", 0)),
        "generated_tokens": int(scored.get("generated_tokens", 0)),
        "error_categories": sorted({item["type"] for item in semantic["errors"]}),
    }
    if condition == "f1":
        result["f1_isomorphism"] = _f1_isomorphism(
            common_reference_asl, structural_asl, scope
        )
    if condition == "f2":
        result["f2_semantics"] = _f2_call_metrics(f2_reference_program, program, scope)
    return _json_value(result)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    rates = {}
    for name, predicate in {
        "parse": lambda row: row["surface"]["parse_valid"],
        "lowerable": lambda row: row["runtime"]["lowerable"],
        "type_valid": lambda row: row["runtime"]["type_valid"],
        "executable": lambda row: row["final"]["executable"],
        "semantic_structure": lambda row: row["semantic_name_equivalent"],
        "semantic_state": lambda row: row["blackboard"]["semantic_state_equivalent"],
        "final_answer": lambda row: row["final"]["answer_correct"],
        "fail_closed": lambda row: row["final"]["fail_closed"],
    }.items():
        correct = sum(bool(predicate(row)) for row in records)
        rates[name] = {**_rate(correct, count), "wilson95": _wilson(correct, count)}
    parsed = [row for row in records if row["surface"]["parse_valid"]]
    lowerable = [row for row in records if row["runtime"]["lowerable"]]
    semantic_correct = [row for row in records if row["semantic_name_equivalent"]]
    semantic_wrong = [row for row in records if not row["semantic_name_equivalent"]]
    state_correct = [row for row in records if row["blackboard"]["semantic_state_equivalent"]]
    state_wrong = [row for row in records if not row["blackboard"]["semantic_state_equivalent"]]
    probability = lambda rows: _rate(
        sum(row["final"]["answer_correct"] for row in rows), len(rows)
    )
    output = {
        "count": count,
        "rates": rates,
        "mean_semantic": {
            name: _mean(row["semantic_structure"].get(name, {}).get("f1") for row in records)
            for name in ("entity", "attribute", "qualifier", "operator", "dependency")
        },
        "mean_blackboard": {
            name: _mean(row["blackboard"][name] for row in records)
            for name in (
                "blackboard_precision",
                "blackboard_recall",
                "blackboard_f1",
                "spurious_state_rate",
                "dependency_f1",
            )
        },
        "conditional": {
            "semantic_structure_given_parse": _rate(
                sum(row["semantic_name_equivalent"] for row in parsed), len(parsed)
            ),
            "semantic_structure_given_lowerable": _rate(
                sum(row["semantic_name_equivalent"] for row in lowerable), len(lowerable)
            ),
            "answer_given_semantic_structure_correct": probability(semantic_correct),
            "answer_given_semantic_structure_incorrect": probability(semantic_wrong),
            "answer_given_runtime_state_correct": probability(state_correct),
            "answer_given_runtime_state_incorrect": probability(state_wrong),
        },
        "cost": {
            "mean_prompt_tokens": _mean(row["prompt_tokens"] for row in records),
            "mean_generated_tokens": _mean(row["generated_tokens"] for row in records),
            "mean_statement_count": _mean(
                row["complexity"]["statement_count"] for row in records
            ),
            "mean_ast_nodes": _mean(row["complexity"]["ast_node_count"] for row in records),
            "generated_tokens_per_correct_answer": sum(
                row["generated_tokens"] for row in records
            )
            / max(1, sum(row["final"]["answer_correct"] for row in records)),
            "semantic_success_per_generated_token": sum(
                row["semantic_name_equivalent"] for row in records
            )
            / max(1, sum(row["generated_tokens"] for row in records)),
            "final_success_per_generated_token": sum(
                row["final"]["answer_correct"] for row in records
            )
            / max(1, sum(row["generated_tokens"] for row in records)),
        },
        "error_counts": dict(
            sorted(Counter(error for row in records for error in row["error_categories"]).items())
        ),
    }
    if records and records[0]["condition"] == "f1":
        output["f1_asl_isomorphic_accuracy"] = _rate(
            sum(row["f1_isomorphism"]["f1_asl_isomorphic_accuracy"] for row in records),
            count,
        )
    if records and records[0]["condition"] == "f2":
        output["f2_semantics"] = _aggregate_f2(records)
        output["f2_semantics_given_parse"] = _aggregate_f2(parsed)
    return output


def _aggregate_f2(records: list[dict[str, Any]]) -> dict[str, Any]:
    items = [row["f2_semantics"] for row in records]
    pairs = [pair for item in items for pair in item["class_pairs"]]
    class_correct = sum(left == right for left, right in pairs)
    classes = sorted(
        ({left for left, _ in pairs} | {right for _, right in pairs}) - {"EXTRA", "MISSING"}
    )
    per_class = {}
    for name in classes:
        per_class[name] = _prf(
            sum(left == right == name for left, right in pairs),
            sum(left == name for left, _ in pairs),
            sum(right == name for _, right in pairs),
        )
    return {
        "functor_class_micro": _prf(
            class_correct,
            sum(left != "EXTRA" for left, _ in pairs),
            sum(right != "MISSING" for _, right in pairs),
        ),
        "functor_class_macro_f1": _mean(value["f1"] for value in per_class.values()),
        "per_functor_class": per_class,
        "argument_role_accuracy": _rate(
            sum(item["role_correct"] for item in items),
            sum(item["role_support"] for item in items),
        ),
        "argument_binding_exact": _rate(
            sum(item["binding_correct"] for item in items),
            sum(item["binding_support"] for item in items),
        ),
        "relation_direction_accuracy": _rate(
            sum(item["direction_correct"] for item in items),
            sum(item["direction_support"] for item in items),
        ),
        "query_target_accuracy": _rate(
            sum(item["query_correct"] for item in items), len(items)
        ),
    }


def _vocabulary(
    f1_train: list[dict[str, Any]],
    f1_dev: list[dict[str, Any]],
    f1_eval: list[dict[str, Any]],
    f2_train: list[dict[str, Any]],
    f2_dev: list[dict[str, Any]],
    f2_eval: list[dict[str, Any]],
) -> dict[str, Any]:
    def view(rows: list[dict[str, Any]], condition: str, key: str) -> dict[str, Any]:
        calls = [
            call
            for row in rows
            for call in _calls(str(row[key]), condition)[0]
        ]
        paths = _call_paths(calls)
        attributes = {
            token for path in paths for token in _path_features(path)["attributes"]
        }
        qualifiers = {
            item for path in paths for item in _path_features(path)["qualifiers"]
        }
        return {
            "functors": sorted({call.name for call in calls}),
            "functor_occurrences": dict(sorted(Counter(call.name for call in calls).items())),
            "distinct_paths": len(paths),
            "distinct_attributes": len(attributes),
            "distinct_qualifiers": len(qualifiers),
        }

    f1 = {
        "train": view(f1_train, "f1", "target"),
        "dev": view(f1_dev, "f1", "target"),
        "test": view(f1_eval, "f1", "reference_program"),
    }
    f2 = {
        "train": view(f2_train, "f2", "target"),
        "dev": view(f2_dev, "f2", "target"),
        "test": view(f2_eval, "f2", "reference_program"),
    }
    train_functors, test_functors = set(f2["train"]["functors"]), set(f2["test"]["functors"])
    test_counts = Counter(f2["test"]["functor_occurrences"])
    unseen_occurrences = sum(test_counts[name] for name in test_functors - train_functors)
    return {
        "registered_vocabulary": {
            "f1": sorted(functor_registry("f1")),
            "f2": sorted(functor_registry("f2")),
        },
        "f1": f1,
        "f2": f2,
        "f2_dev_only_functors": sorted(set(f2["dev"]["functors"]) - train_functors),
        "f2_test_only_functors": sorted(test_functors - train_functors),
        "f2_unseen_functor_rate": unseen_occurrences / max(1, sum(test_counts.values())),
    }


def _gold_runtime(
    f0_eval: dict[tuple[str, str], dict[str, Any]],
    f1_eval: dict[tuple[str, str], dict[str, Any]],
    f2_eval: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    records = {name: [] for name in ("F0", "F1", "F2")}
    for identity in sorted(f0_eval):
        scope = f0_eval[identity]["effective_scope"]
        reference_asl = f0_eval[identity]["reference_asl"]
        f0 = validate_asl(reference_asl, effective_scope=scope)
        records["F0"].append(
            {
                "identity": identity,
                "parse": bool(f0["syntax_verified"]),
                "lowerable": bool(f0["lower_verified"]),
                "type_valid": bool(f0["type_verified"]),
                "executable": bool(f0["execution_verified"]),
                "answer_correct": bool(f0["execution_verified"]),
            }
        )
        for name, condition, source in (
            ("F1", "f1", f1_eval[identity]),
            ("F2", "f2", f2_eval[identity]),
        ):
            validation = validate_functor_program(
                source["reference_program"], condition, effective_scope=scope
            )
            reference_return = _program_view(reference_asl, scope)["validation"].get(
                "execution", {}
            ).get("workspace", {}).get(str(scope["id"]), {}).get("returned")
            candidate_return = (validation.get("validation") or {}).get("execution", {}).get(
                "workspace", {}
            ).get(str(scope["id"]), {}).get("returned")
            records[name].append(
                {
                    "identity": identity,
                    "parse": validation["parse_valid"],
                    "lowerable": validation["lowerable"],
                    "type_valid": validation["type_valid"],
                    "executable": validation["executable"],
                    "answer_correct": validation["executable"]
                    and str(candidate_return) == str(reference_return),
                    "errors": validation["errors"],
                }
            )
    return {
        name: {
            metric: _rate(sum(row[metric] for row in rows), len(rows))
            for metric in ("parse", "lowerable", "type_valid", "executable", "answer_correct")
        }
        for name, rows in records.items()
    } | {"records": records}


def _by_functor(
    records: list[dict[str, Any]], train_occurrences: dict[str, int]
) -> dict[str, Any]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        for call in row["f2_semantics"]["call_records"]:
            occurrences[call["gold_functor"]].append(
                {**call, "answer_correct": row["final"]["answer_correct"]}
            )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        direction = [item for item in items if item["direction_correct"] is not None]
        return {
            "support": len(items),
            "class_accuracy": _rate(sum(item["class_correct"] for item in items), len(items)),
            "argument_binding_exact": _rate(
                sum(item["binding_exact"] for item in items), len(items)
            ),
            "direction_accuracy": _rate(
                sum(item["direction_correct"] for item in direction), len(direction)
            ),
            "final_answer_association": _rate(
                sum(item["answer_correct"] for item in items), len(items)
            ),
        }

    per_functor = {
        name: {
            "train_frequency": int(train_occurrences.get(name, 0)),
            **summarize(items),
        }
        for name, items in sorted(occurrences.items())
    }
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, items in occurrences.items():
        frequency = int(train_occurrences.get(name, 0))
        bucket = "unseen" if frequency == 0 else "rare" if frequency < 10 else "medium" if frequency < 50 else "frequent"
        buckets[bucket].extend(items)
    return {
        "bucket_definition": {
            "unseen": "0 train occurrences",
            "rare": "1-9 train occurrences",
            "medium": "10-49 train occurrences",
            "frequent": "50+ train occurrences",
        },
        "per_functor": per_functor,
        "by_frequency_bucket": {
            name: summarize(items) for name, items in sorted(buckets.items())
        },
    }


def _target_complexity(
    f0_eval: list[dict[str, Any]],
    f1_eval: list[dict[str, Any]],
    f2_eval: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = {
        "F0": [(row["reference_asl"], "f0") for row in f0_eval],
        "F1": [(row["reference_program"], "f1") for row in f1_eval],
        "F2": [(row["reference_program"], "f2") for row in f2_eval],
    }
    return {
        name: {
            metric: _mean(item[metric] for item in (_complexity(program, condition, 0) for program, condition in rows))
            for metric in (
                "statement_count",
                "ast_node_count",
                "mean_arity",
                "max_nesting",
                "distinct_path_count",
            )
        }
        for name, rows in sources.items()
    }


def analyze_functor_metrics(
    *,
    f0_eval_path: str | Path,
    f0_scored_path: str | Path,
    f1_eval_path: str | Path,
    f1_scored_path: str | Path,
    f2_eval_path: str | Path,
    f2_scored_path: str | Path,
    f1_train_path: str | Path,
    f1_dev_path: str | Path,
    f2_train_path: str | Path,
    f2_dev_path: str | Path,
    output_dir: str | Path,
    model_label: str,
) -> dict[str, Any]:
    """Build the complete frozen-prediction representation analysis."""

    output = Path(output_dir)
    f0_eval_rows, f1_eval_rows, f2_eval_rows = (
        read_jsonl(f0_eval_path),
        read_jsonl(f1_eval_path),
        read_jsonl(f2_eval_path),
    )
    eval_maps = [
        {_identity(row): row for row in rows}
        for rows in (f0_eval_rows, f1_eval_rows, f2_eval_rows)
    ]
    scored_maps = [
        {_identity(row): row for row in rows}
        for rows in (
            read_jsonl(f0_scored_path),
            read_jsonl(f1_scored_path),
            read_jsonl(f2_scored_path),
        )
    ]
    identities = set(eval_maps[0])
    if len(identities) != 25 or any(set(rows) != identities for rows in [*eval_maps, *scored_maps]):
        raise ValueError("F0/F1/F2 analysis requires the identical frozen 25 source identities")
    normalized = {name: [] for name in ("f0", "f1", "f2")}
    cross = []
    for identity in sorted(identities):
        common_reference = lower_f1(
            parse_functor_program(eval_maps[1][identity]["reference_program"], "f1")
        )
        records = {}
        for index, name in enumerate(("f0", "f1", "f2")):
            records[name] = _record(
                name,
                scored_maps[index][identity],
                eval_maps[index][identity],
                common_reference,
                eval_maps[2][identity]["reference_program"],
            )
            normalized[name].append(records[name])
        cross.append(
            {
                "dataset": identity[0],
                "parent_source_id": identity[1],
                "outcome": {
                    name.upper(): records[name]["final"]["answer_correct"] for name in records
                },
                "semantic_structure": {
                    name.upper(): records[name]["semantic_name_equivalent"] for name in records
                },
                "blackboard_f1": {
                    name.upper(): records[name]["blackboard"]["blackboard_f1"] for name in records
                },
                "errors": {
                    name.upper(): records[name]["error_categories"] for name in records
                },
            }
        )
    for name, rows in normalized.items():
        write_jsonl(output / f"{name}_normalized.jsonl", rows)
    write_jsonl(output / "cross_representation.jsonl", cross)
    aggregates = {name.upper(): _aggregate(rows) for name, rows in normalized.items()}
    by_dataset = {
        dataset: {
            name.upper(): _aggregate([row for row in rows if row["dataset"] == dataset])
            for name, rows in normalized.items()
        }
        for dataset in sorted({row["dataset"] for row in normalized["f0"]})
    }
    patterns = Counter(
        "".join(name for name in ("F0", "F1", "F2") if row["outcome"][name]) or "NONE"
        for row in cross
    )
    pairwise = {}
    for left, right in (("F1", "F0"), ("F2", "F0"), ("F2", "F1")):
        left_only = sum(row["outcome"][left] and not row["outcome"][right] for row in cross)
        right_only = sum(row["outcome"][right] and not row["outcome"][left] for row in cross)
        pairwise[f"{left}_minus_{right}"] = {
            "left_only": left_only,
            "right_only": right_only,
            "paired_exact_p_exploratory": _paired_exact(left_only, right_only),
        }
    paired_flips = {
        "outcome_patterns": dict(sorted(patterns.items())),
        "pairwise": pairwise,
        "records": cross,
    }
    vocabulary = _vocabulary(
        read_jsonl(f1_train_path),
        read_jsonl(f1_dev_path),
        f1_eval_rows,
        read_jsonl(f2_train_path),
        read_jsonl(f2_dev_path),
        f2_eval_rows,
    )
    gold_runtime = _gold_runtime(*eval_maps)
    by_functor = _by_functor(
        normalized["f2"], vocabulary["f2"]["train"]["functor_occurrences"]
    )
    target_complexity = _target_complexity(f0_eval_rows, f1_eval_rows, f2_eval_rows)
    f2_pairs = [
        pair
        for row in normalized["f2"]
        for pair in row["f2_semantics"]["class_pairs"]
    ]
    confusion = dict(sorted(Counter(f"{left}->{right}" for left, right in f2_pairs).items()))
    role_confusion = dict(
        sorted(
            sum(
                (
                    Counter(row["f2_semantics"].get("role_confusion", {}))
                    for row in normalized["f2"]
                ),
                Counter(),
            ).items()
        )
    )
    blackboard = {
        name.upper(): aggregate["mean_blackboard"] for name, aggregate in aggregates.items()
    }
    cost = {name.upper(): aggregate["cost"] for name, aggregate in aggregates.items()}
    interpretation = {
        "syntax_only_explanation_supported": (
            aggregates["F2"]["rates"]["parse"]["rate"]
            < aggregates["F0"]["rates"]["parse"]["rate"]
            and (
                aggregates["F2"]["conditional"]["semantic_structure_given_parse"]["rate"] or 0
            )
            >= (
                aggregates["F0"]["conditional"]["semantic_structure_given_parse"]["rate"] or 0
            )
        ),
        "semantic_functor_errors_present": (
            aggregates["F2"]["f2_semantics"]["functor_class_micro"]["f1"] < 1.0
            or aggregates["F2"]["f2_semantics"]["argument_role_accuracy"]["rate"] < 1.0
        ),
        "runtime_bottleneck_present": gold_runtime["F2"]["answer_correct"]["rate"] < 1.0,
        "representation_success": (
            aggregates["F2"]["mean_blackboard"]["blackboard_f1"]
            > aggregates["F0"]["mean_blackboard"]["blackboard_f1"]
            and aggregates["F2"]["rates"]["final_answer"]["rate"]
            >= aggregates["F0"]["rates"]["final_answer"]["rate"]
        ),
    }
    input_paths = {
        "f0_eval": f0_eval_path,
        "f0_scored": f0_scored_path,
        "f1_eval": f1_eval_path,
        "f1_scored": f1_scored_path,
        "f2_eval": f2_eval_path,
        "f2_scored": f2_scored_path,
        "f1_train": f1_train_path,
        "f1_dev": f1_dev_path,
        "f2_train": f2_train_path,
        "f2_dev": f2_dev_path,
    }
    summary = {
        "schema_version": "ccpu.paper1.functor_metrics.v2",
        "model_label": model_label,
        "frozen_identity_count": len(identities),
        "identical_frozen_source_ids": True,
        "common_reference": "query-computational F1 graph shared across F0/F1/F2",
        "conditions": aggregates,
        "by_dataset": by_dataset,
        "paired": {key: value for key, value in paired_flips.items() if key != "records"},
        "vocabulary": vocabulary,
        "gold_runtime": {key: value for key, value in gold_runtime.items() if key != "records"},
        "functor_confusion": confusion,
        "argument_role_confusion": role_confusion,
        "blackboard": blackboard,
        "representation_cost": cost,
        "target_complexity": target_complexity,
        "interpretation_gates": interpretation,
        "training_size_limitation": {"F0": 450, "F1": 434, "F2": 434},
        "statistical_boundary": "25 programs; exact counts primary, Wilson and paired exact exploratory",
        "input_sha256": {name: file_sha256(path) for name, path in input_paths.items()},
    }
    artifacts = {
        "paired_flips.json": paired_flips,
        "by_dataset.json": by_dataset,
        "by_functor.json": by_functor,
        "argument_roles.json": {
            "aggregate": aggregates["F2"]["f2_semantics"],
            "confusion": role_confusion,
        },
        "confusion_matrix.json": confusion,
        "blackboard_metrics.json": blackboard,
        "runtime_lowering.json": gold_runtime,
        "representation_cost.json": cost,
        "model_size_comparison.json": {
            "status": "pending_larger_model",
            "model_label": model_label,
        },
        "summary.json": summary,
    }
    for name, value in artifacts.items():
        write_json(output / name, value)
    return summary


def compare_functor_model_sizes(
    small_summary: str | Path, large_summary: str | Path, output: str | Path
) -> dict[str, Any]:
    """Compare matched representation metrics across two model capacities."""

    small, large = read_json(small_summary), read_json(large_summary)
    if small["frozen_identity_count"] != 25 or large["frozen_identity_count"] != 25:
        raise ValueError("model-size comparison requires both frozen 25-program analyses")
    fields = {
        "parse": ("rates", "parse", "rate"),
        "semantic_structure": ("rates", "semantic_structure", "rate"),
        "dependency_f1": ("mean_semantic", "dependency"),
        "blackboard_f1": ("mean_blackboard", "blackboard_f1"),
        "final_answer": ("rates", "final_answer", "rate"),
    }

    def get(source: dict[str, Any], condition: str, path: tuple[str, ...]) -> float:
        value: Any = source["conditions"][condition]
        for part in path:
            value = value[part]
        return float(value)

    conditions = {}
    for condition in ("F0", "F1", "F2"):
        conditions[condition] = {}
        for name, path in fields.items():
            before, after = get(small, condition, path), get(large, condition, path)
            conditions[condition][name] = {
                "small": before,
                "large": after,
                "delta": after - before,
            }
    for name, path in {
        "functor_class_micro_f1": ("f2_semantics", "functor_class_micro", "f1"),
        "argument_role_accuracy": ("f2_semantics", "argument_role_accuracy", "rate"),
        "argument_binding_exact": ("f2_semantics", "argument_binding_exact", "rate"),
        "relation_direction_accuracy": (
            "f2_semantics",
            "relation_direction_accuracy",
            "rate",
        ),
    }.items():
        before, after = get(small, "F2", path), get(large, "F2", path)
        conditions["F2"][name] = {"small": before, "large": after, "delta": after - before}
    final_deltas = {name: values["final_answer"]["delta"] for name, values in conditions.items()}
    comparison = {
        "schema_version": "ccpu.paper1.functor_model_size.v1",
        "small_model": small["model_label"],
        "large_model": large["model_label"],
        "frozen_identity_count": 25,
        "conditions": conditions,
        "representation_by_capacity_interaction": {
            "F2_minus_F0_final_delta": final_deltas["F2"] - final_deltas["F0"],
            "F2_minus_F1_final_delta": final_deltas["F2"] - final_deltas["F1"],
            "capacity_bottleneck_supported": (
                final_deltas["F2"] > final_deltas["F0"]
                and final_deltas["F2"] > final_deltas["F1"]
            ),
            "rule": "capacity explanation requires F2 to improve more than both F0 and F1",
        },
        "input_sha256": {
            "small": file_sha256(small_summary),
            "large": file_sha256(large_summary),
        },
    }
    write_json(output, comparison)
    return comparison
