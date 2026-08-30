"""Model execution and semantic component metrics for the Paper 1 ASL pilot."""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import parse_asl, validate_asl
from ccpu.dsl_dataset.semantic import semantic_lint

from .asl_pilot_data import asl_prompt, semantic_family
from .generation import HuggingFaceBackend, HuggingFaceGenerationConfig

_FENCE = re.compile(r"```(?:asl|text|ini|python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_STATEMENT = re.compile(
    r"^\s*(?:[a-z_][a-z0-9_.]*\s*=\s*.+|RETURN\s+.+)\s*$", re.IGNORECASE
)


def extract_asl(text: str) -> str:
    matches = _FENCE.findall(text)
    candidates = [match.strip() for match in matches if match.strip()]
    candidates.append(text.strip())
    for candidate in candidates:
        lines = []
        for raw_line in candidate.splitlines():
            line = raw_line.strip().rstrip(";")
            if _STATEMENT.match(line):
                lines.append(line)
            elif lines and line:
                break
        if lines:
            return "\n".join(lines)
    return ""


def _multiset_f1(reference: Counter[Any], predicted: Counter[Any]) -> dict[str, float]:
    overlap = sum((reference & predicted).values())
    reference_count = sum(reference.values())
    predicted_count = sum(predicted.values())
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _expr_ops(node: dict[str, Any]) -> list[str]:
    operation = str(node["op"])
    values = [] if operation in {"CONST", "REF"} else [operation]
    for argument in node.get("args", []):
        values.extend(_expr_ops(argument))
    return values


def _expr_refs(node: dict[str, Any]) -> list[str]:
    if node["op"] == "REF":
        return [str(node["path"])]
    return [reference for argument in node.get("args", []) for reference in _expr_refs(argument)]


def _components(validation: dict[str, Any]) -> dict[str, Counter[Any]]:
    operations = validation["ccir"]["operations"]
    paths = []
    source_facts = []
    operators = []
    edges = []
    target_families = {
        str(item["operation"]["target"]): semantic_family(str(item["operation"]["target"]))
        for item in operations
        if item["operation"]["op"] == "SET"
    }
    for item in operations:
        operation = item["operation"]
        if operation["op"] != "SET":
            if operation["op"] == "RETURN":
                operators.extend(_expr_ops(operation["expr"]))
            continue
        target = str(operation["target"])
        family = semantic_family(target)
        paths.append((family, target.count(".") + 1))
        expression = operation["expr"]
        operators.extend(_expr_ops(expression))
        if expression["op"] == "CONST":
            source_facts.append((family, str(expression.get("value"))))
        for reference in _expr_refs(expression):
            edges.append(
                (
                    target_families.get(reference, semantic_family(reference)),
                    family,
                    expression["op"],
                )
            )
    return {
        "paths": Counter(paths),
        "source_facts": Counter(source_facts),
        "operators": Counter(operators),
        "edges": Counter(edges),
    }


def _expression_signature(
    node: dict[str, Any], assignments: dict[str, dict[str, Any]], stack: frozenset[str]
) -> Any:
    operation = str(node["op"])
    if operation == "CONST":
        return ("CONST", str(node.get("value")))
    if operation == "REF":
        path = str(node["path"])
        if path in stack:
            return ("CYCLE", semantic_family(path))
        if path in assignments:
            return (
                "STATE",
                semantic_family(path),
                _expression_signature(assignments[path], assignments, stack | {path}),
            )
        return ("EXTERNAL", semantic_family(path))
    arguments = [
        _expression_signature(argument, assignments, stack)
        for argument in node.get("args", [])
    ]
    if operation in {"ADD", "MUL", "SUM", "MIN", "MAX"}:
        flattened = []
        equivalent = "ADD" if operation == "SUM" else operation
        for argument in arguments:
            if isinstance(argument, tuple) and argument and argument[0] == equivalent:
                flattened.extend(argument[1:])
            else:
                flattened.append(argument)
        return (equivalent, *sorted(flattened, key=repr))
    return (operation, *arguments)


def _state_signatures(validation: dict[str, Any]) -> tuple[Counter[Any], Any]:
    assignments = {
        str(item["operation"]["target"]): item["operation"]["expr"]
        for item in validation["ccir"]["operations"]
        if item["operation"]["op"] == "SET"
    }
    states = Counter(
        (
            semantic_family(path),
            _expression_signature(expression, assignments, frozenset({path})),
        )
        for path, expression in assignments.items()
    )
    return_operation = next(
        (
            item["operation"]["expr"]
            for item in reversed(validation["ccir"]["operations"])
            if item["operation"]["op"] == "RETURN"
        ),
        None,
    )
    returned = (
        _expression_signature(return_operation, assignments, frozenset())
        if return_operation is not None
        else None
    )
    return states, returned


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def score_asl(reference: str, predicted: str, scope: dict[str, Any]) -> dict[str, Any]:
    reference_validation = validate_asl(reference, effective_scope=scope)
    if not reference_validation["execution_verified"]:
        raise ValueError("reference ASL must execute")
    result: dict[str, Any] = {
        "exact_asl": "\n".join(reference.split()) == "\n".join(predicted.split()),
        "parse_valid": False,
        "lowerable_to_ccir": False,
        "type_valid": False,
        "semantic_lint_valid": False,
        "executable": False,
        "dependency_correct": False,
        "semantic_return_equivalent": False,
        "semantic_state_equivalent": False,
        "final_answer_correct": False,
        "errors": [],
    }
    if not predicted.strip():
        result["errors"].append("no ASL statements extracted")
        return result
    try:
        parsed = parse_asl(predicted, effective_scope=scope)
        result["parse_valid"] = True
        mappings = [
            {
                "asl": [
                    line
                    for line in predicted.splitlines()
                    if line.strip() and not line.strip().upper().startswith("RETURN ")
                ]
                + [
                    line
                    for line in predicted.splitlines()
                    if line.strip().upper().startswith("RETURN ")
                ]
            }
        ]
        lint_errors = semantic_lint(mappings)
        result["semantic_lint_valid"] = not lint_errors
        result["errors"].extend(lint_errors)
        result["ast_record_count"] = len(parsed["records"])
    except (KeyError, TypeError, ValueError) as error:
        result["errors"].append(str(error))
        return result
    predicted_validation = validate_asl(predicted, effective_scope=scope)
    result["errors"].extend(predicted_validation["errors"])
    result["lowerable_to_ccir"] = bool(predicted_validation["lower_verified"])
    result["type_valid"] = bool(predicted_validation["type_verified"])
    if not result["lowerable_to_ccir"]:
        return result
    reference_components = _components(reference_validation)
    predicted_components = _components(predicted_validation)
    for name in ("paths", "source_facts", "operators", "edges"):
        result[f"{name}_metrics"] = _multiset_f1(
            reference_components[name], predicted_components[name]
        )
    result["executable"] = predicted_validation["execution_verified"]
    result["dependency_correct"] = (
        result["executable"] and result["edges_metrics"]["f1"] == 1.0
    )
    if not result["type_valid"]:
        return result
    reference_states, reference_return = _state_signatures(reference_validation)
    predicted_states, predicted_return = _state_signatures(predicted_validation)
    state_metrics = _multiset_f1(reference_states, predicted_states)
    result["semantic_state_metrics"] = state_metrics
    result["semantic_state_equivalent"] = state_metrics["f1"] == 1.0
    result["semantic_return_equivalent"] = reference_return == predicted_return
    if result["executable"]:
        scope_id = str(scope["id"])
        expected = reference_validation["execution"]["workspace"][scope_id]["returned"]
        actual = predicted_validation["execution"]["workspace"][scope_id]["returned"]
        try:
            result["final_answer_correct"] = abs(_decimal(actual) - _decimal(expected)) <= Decimal(
                "0.011"
            )
        except (InvalidOperation, TypeError):
            result["final_answer_correct"] = actual == expected
        result["predicted_return"] = actual
        result["reference_return"] = expected
    return result


def _select_demos(train_rows: list[dict[str, Any]], shots: int, *, seed: int) -> list[dict[str, Any]]:
    if shots == 0:
        return []
    ordered = sorted(
        train_rows,
        key=lambda row: fingerprint(f"{seed}:{row['semantic_pattern_id']}:{row['record_sha256']}"),
    )
    selected = []
    seen_patterns = set()
    for row in ordered:
        if row["semantic_pattern_id"] in seen_patterns:
            continue
        selected.append(row)
        seen_patterns.add(row["semantic_pattern_id"])
        if len(selected) == shots:
            return selected
    raise ValueError(f"not enough distinct train patterns for {shots}-shot ICL")


def run_asl_pilot(
    *,
    eval_path: str | Path,
    train_split_path: str | Path,
    model_config: dict[str, Any],
    condition: str,
    shots: int,
    output_dir: str | Path,
    seed: int = 44017,
    checkpoint_every: int = 5,
) -> dict[str, Any]:
    if condition not in {"base", "icl", "lora", "lora_icl"}:
        raise ValueError(f"unsupported ASL condition: {condition}")
    model = dict(model_config["model"])
    if condition in {"lora", "lora_icl"}:
        model["adapter_path"] = model_config["adapter_path"]
        model["adapter_id"] = model_config["adapter_id"]
    else:
        model.pop("adapter_path", None)
        model.pop("adapter_id", None)
    backend = HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_new_tokens=int(model.get("max_new_tokens", 384)),
            device=str(model.get("device", "xpu")),
            dtype=str(model.get("dtype", "float16")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            adapter_path=model.get("adapter_path"),
            adapter_id=model.get("adapter_id"),
            cached_generation=True,
        )
    )
    eval_rows = read_jsonl(eval_path)
    train_rows = read_jsonl(train_split_path)
    demos = _select_demos(train_rows, shots, seed=seed)
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    expected_model_id = str(model["model_id"])
    eval_suites = {item["suite"] for item in eval_rows}
    if any(
        row.get("condition") != condition
        or int(row.get("shots", -1)) != shots
        or row.get("model_id") != expected_model_id
        or row.get("suite") not in eval_suites
        for row in predictions
    ):
        raise ValueError("resume output does not match the requested ASL pilot run")
    completed = {row["example_id"] for row in predictions}
    for index, row in enumerate(
        [row for row in eval_rows if row["example_id"] not in completed], 1
    ):
        prompt_row = {
            "question": row["question"],
            "source_context": row.get("source_context"),
            "asl": row["reference_asl"],
        }
        prompt = asl_prompt(prompt_row, demos)
        generation = backend.generate(prompt, seed=seed)
        predicted_asl = extract_asl(generation.generated_text)
        predictions.append(
            {
                "schema_version": "ccpu.paper1.asl_prediction.v1",
                "example_id": row["example_id"],
                "parent_source_id": row["parent_source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "suite": row["suite"],
                "condition": condition,
                "shots": shots,
                "model_id": backend.model_id,
                "seed": seed,
                "generated_text": generation.generated_text,
                "predicted_asl": predicted_asl,
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "wall_time_ns": generation.wall_time_ns,
                "backend_metadata": generation.metadata,
            }
        )
        if index % checkpoint_every == 0:
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint {condition}/{shots}: {len(predictions)}/{len(eval_rows)}")
    write_jsonl(predictions_path, predictions)
    report = analyze_asl_predictions(eval_path, predictions_path, output)
    report["run"] = {
        "condition": condition,
        "shots": shots,
        "seed": seed,
        "model": model,
        "demo_source_ids": [row["source_id"] for row in demos],
        "eval_sha256": file_sha256(eval_path),
        "train_split_sha256": file_sha256(train_split_path),
    }
    write_json(output / "summary.json", report)
    return report


def analyze_asl_predictions(
    eval_path: str | Path, predictions_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    references = {row["example_id"]: row for row in read_jsonl(eval_path)}
    predictions = read_jsonl(predictions_path)
    scored = []
    for prediction in predictions:
        reference = references[prediction["example_id"]]
        metrics = score_asl(
            reference["reference_asl"],
            prediction["predicted_asl"],
            reference["effective_scope"],
        )
        scored.append({**prediction, "metrics": metrics})
    metric_names = (
        "exact_asl",
        "parse_valid",
        "lowerable_to_ccir",
        "type_valid",
        "semantic_lint_valid",
        "executable",
        "dependency_correct",
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
    )
    component_names = ("paths", "source_facts", "operators", "edges", "semantic_state")
    summary = {
        "schema_version": "ccpu.paper1.asl_evaluation.v1",
        "prediction_count": len(scored),
        "rates": {
            name: sum(bool(row["metrics"].get(name)) for row in scored) / len(scored)
            if scored
            else 0.0
            for name in metric_names
        },
        "component_mean_f1": {
            name: sum(
                float(row["metrics"].get(f"{name}_metrics", {}).get("f1", 0.0))
                for row in scored
            )
            / len(scored)
            if scored
            else 0.0
            for name in component_names
        },
        "by_dataset": {},
        "eval_sha256": file_sha256(eval_path),
        "predictions_sha256": file_sha256(predictions_path),
    }
    for dataset in sorted({row["dataset"] for row in scored}):
        members = [row for row in scored if row["dataset"] == dataset]
        summary["by_dataset"][dataset] = {
            "count": len(members),
            "rates": {
                name: sum(bool(row["metrics"].get(name)) for row in members) / len(members)
                for name in metric_names
            },
        }
    scored_path = write_jsonl(Path(output_dir) / "scored_predictions.jsonl", scored)
    summary["scored_predictions_sha256"] = file_sha256(scored_path)
    return summary
