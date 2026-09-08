"""Deterministic O0/O1 registry and matched GSM8K-OC dataset generator."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl import validate_asl

OPERATOR_REGISTRY_VERSION = "paper1-operator-registry-v1"
OPERATOR_DATASET_VERSION = "gsm8k-oc-v1"

OPERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "add": {"level": "O0", "arity": 2, "ccir": "ADD", "exactness": "EXACT_RATIONAL"},
    "subtract": {"level": "O0", "arity": 2, "ccir": "SUB", "exactness": "EXACT_RATIONAL"},
    "multiply": {"level": "O0", "arity": 2, "ccir": "MUL", "exactness": "EXACT_RATIONAL"},
    "divide": {"level": "O0", "arity": 2, "ccir": "DIV", "exactness": "EXACT_RATIONAL"},
    "square": {"level": "O1", "arity": 1, "ccir": "SQUARE", "exactness": "EXACT_INTEGER"},
    "cube": {"level": "O1", "arity": 1, "ccir": "CUBE", "exactness": "EXACT_INTEGER"},
    "sqrt": {
        "level": "O1",
        "arity": 1,
        "ccir": "SQRT",
        "exactness": "EXACT_INTEGER",
        "domain": "non-negative perfect squares",
    },
}

_TEMPLATES = (
    ("tile_square_area", "square", "A square tile has side length {value} centimeters. What is its area in square centimeters?"),
    ("garden_square_area", "square", "Each side of a square garden is {value} meters long. How many square meters does the garden cover?"),
    ("poster_square_area", "square", "A square poster is {value} inches wide. What is the poster's area in square inches?"),
    ("plaza_square_area", "square", "A square plaza measures {value} meters along one edge. Find its area in square meters."),
    ("mat_square_side", "sqrt", "A square mat has area {derived} square feet. How many feet long is one side?"),
    ("field_square_side", "sqrt", "The area of a square field is {derived} square meters. What is the field's side length in meters?"),
    ("window_square_side", "sqrt", "A square window covers {derived} square centimeters. Find the length of one side in centimeters."),
    ("court_square_side", "sqrt", "A square court has an area of {derived} square yards. How long is each side in yards?"),
    ("crate_cube_volume", "cube", "A cube-shaped crate has edge length {value} feet. What is its volume in cubic feet?"),
    ("block_cube_volume", "cube", "Every edge of a solid cube is {value} centimeters long. Find its volume in cubic centimeters."),
    ("tank_cube_volume", "cube", "A cubical tank measures {value} meters on each edge. How many cubic meters does it hold?"),
    ("box_cube_volume", "cube", "A cube-shaped box has sides of length {value} inches. What is its volume in cubic inches?"),
)

_SPLIT_FAMILIES = {
    "train": {
        "tile_square_area", "garden_square_area", "mat_square_side",
        "field_square_side", "crate_cube_volume", "block_cube_volume",
    },
    "dev": {"poster_square_area", "window_square_side", "tank_cube_volume"},
    "test": {"plaza_square_area", "court_square_side", "box_cube_volume"},
}


def _programs(operator: str, value: int) -> tuple[str, str, str, int]:
    if operator == "square":
        answer = value * value
        return (
            f"side = {value}\narea = square(side)\nRETURN area",
            f"side = {value}\narea = area_of_square(side)\nRETURN area",
            f"{value} * {value}",
            answer,
        )
    if operator == "sqrt":
        area = value * value
        return (
            f"area = {area}\nside = sqrt(area)\nRETURN side",
            f"area = {area}\nside = side_of_square(area)\nRETURN side",
            f"sqrt({area})",
            value,
        )
    answer = value * value * value
    return (
        f"edge = {value}\nvolume = cube(edge)\nRETURN volume",
        f"edge = {value}\nvolume = volume_of_cube(edge)\nRETURN volume",
        f"{value} * {value} * {value}",
        answer,
    )


def _returned(validation: dict[str, Any], scope_id: str) -> Any:
    return validation["execution"]["workspace"][scope_id]["returned"]


def _record(
    *, split: str, index: int, template: tuple[str, str, str], rng: random.Random
) -> dict[str, Any]:
    template_id, operator, surface = template
    value = rng.randint(2, 99)
    ap, semantic, expression, answer = _programs(operator, value)
    derived = value * value if operator == "sqrt" else answer
    question = surface.format(value=value, derived=derived)
    example_id = f"gsm8k-oc-o1-{split}-{index:05d}"
    scope = {
        "id": example_id,
        "parent": None,
        "kind": "benchmark_case",
        "source": OPERATOR_DATASET_VERSION,
    }
    ap_validation = validate_asl(ap, effective_scope=scope)
    semantic_validation = validate_asl(semantic, effective_scope=scope)
    required = ("syntax_verified", "lower_verified", "type_verified", "execution_verified")
    if not all(ap_validation[key] and semantic_validation[key] for key in required):
        raise AssertionError(f"generated O1 program failed validation: {example_id}")
    if _returned(ap_validation, example_id) != answer:
        raise AssertionError(f"AP answer mismatch: {example_id}")
    if _returned(semantic_validation, example_id) != answer:
        raise AssertionError(f"AS answer mismatch: {example_id}")
    ap_ccir = ap_validation["ccir"]["operations"][-2]["operation"]["expr"]["op"]
    semantic_ccir = semantic_validation["ccir"]["operations"][-2]["operation"]["expr"]["op"]
    if ap_ccir != semantic_ccir:
        raise AssertionError(f"AP/AS CCIR mismatch: {example_id}")
    return {
        "schema_version": "ccpu.paper1.gsm8k_oc.v1",
        "dataset": OPERATOR_DATASET_VERSION,
        "split": split,
        "example_id": example_id,
        "source_row": index,
        "parent_template_id": template_id,
        "operator_level": "O1",
        "difficulty_steps": 1,
        "difficulty_stratum": "low",
        "operator_family": operator,
        "operator_signature": [operator],
        "semantic_signature": f"{operator}(CONST)->RETURN",
        "surface_variant_id": template_id,
        "numeric_variant_id": f"n{value}",
        "generator_version": OPERATOR_DATASET_VERSION,
        "validation_status": "execution_verified",
        "question": question,
        "question_sha256": fingerprint(question),
        "reference_return": str(answer),
        "effective_scope": scope,
        "source_fields_visible_to_model": ["question"],
        "targets": {
            "ap_asl": ap,
            "as_asl": semantic,
            "generic_expression": expression,
        },
        "complexity": {
            "primitive_operation_count": 1,
            "dependency_depth": 1,
            "function_nesting_depth": 1,
            "entity_count": 1,
            "operand_magnitude": len(str(value)),
        },
        "runtime": {
            "registry_version": OPERATOR_REGISTRY_VERSION,
            "ccir_operator": ap_ccir,
            "exactness_class": "EXACT_INTEGER",
        },
    }


def freeze_o1_dataset(
    output_dir: str | Path,
    *,
    train_count: int = 2000,
    dev_count: int = 100,
    test_count: int = 250,
    seed: int = 81001,
) -> dict[str, Any]:
    """Generate matched AP/AS O1 records with template-family-disjoint splits."""

    counts = {"train": train_count, "dev": dev_count, "test": test_count}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all O1 split counts must be positive")
    templates = {item[0]: item for item in _TEMPLATES}
    if set().union(*_SPLIT_FAMILIES.values()) != set(templates):
        raise AssertionError("O1 split registry does not cover every template exactly once")
    if sum((Counter(families) for families in _SPLIT_FAMILIES.values()), Counter()).most_common(1)[0][1] != 1:
        raise AssertionError("O1 template families overlap splits")

    output = Path(output_dir)
    paths: dict[str, Path] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        rng = random.Random(f"{seed}:{split}")
        eligible = [templates[name] for name in sorted(_SPLIT_FAMILIES[split])]
        rows = [
            _record(split=split, index=index, template=eligible[index % len(eligible)], rng=rng)
            for index in range(count)
        ]
        split_rows[split] = rows
        paths[split] = write_jsonl(output / f"{split}.jsonl", rows)

    registry_path = write_json(
        output / "operator_registry.json",
        {"version": OPERATOR_REGISTRY_VERSION, "operators": OPERATOR_REGISTRY},
    )
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_oc_manifest.v1",
        "dataset_version": OPERATOR_DATASET_VERSION,
        "operator_level": "O1",
        "seed": seed,
        "counts": counts,
        "template_families": {
            split: sorted(families) for split, families in _SPLIT_FAMILIES.items()
        },
        "operator_counts": {
            split: dict(sorted(Counter(row["operator_family"] for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "gold_runtime_ceiling": {
            "ap_parse": 1.0,
            "ap_execute": 1.0,
            "as_parse": 1.0,
            "as_execute": 1.0,
            "answer": 1.0,
            "ccir_equivalent": 1.0,
        },
        "output_sha256": {
            **{split: file_sha256(path) for split, path in paths.items()},
            "operator_registry": file_sha256(registry_path),
        },
        "prompt_fields": ["question"],
        "hidden_fields": ["targets", "reference_return", "runtime"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _pilot_prompt(question: str, representation: str) -> str:
    if representation == "ap":
        instruction = (
            "Use the primitive O1 operators square(x), cube(x), and sqrt(x). "
            "Use sqrt only for exact roots."
        )
    else:
        instruction = (
            "Use the semantic O1 operators area_of_square(side), "
            "side_of_square(area), and volume_of_cube(edge)."
        )
    return (
        "Compile the quantitative problem into executable ASL-Arith. "
        f"{instruction} Return only ASL, one statement per line; do not explain.\n\n"
        f"Problem: {question}\nASL:"
    )


def _balanced_prefix(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_operator: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_operator.setdefault(str(row["operator_family"]), []).append(row)
    operators = sorted(by_operator)
    selected = []
    offsets = {operator: 0 for operator in operators}
    while len(selected) < count:
        progressed = False
        for operator in operators:
            offset = offsets[operator]
            if offset < len(by_operator[operator]) and len(selected) < count:
                selected.append(by_operator[operator][offset])
                offsets[operator] += 1
                progressed = True
        if not progressed:
            raise ValueError(f"cannot select {count} balanced records from {len(rows)} rows")
    return selected


def freeze_o1_pilot(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    train_count: int = 250,
    dev_count: int = 30,
    test_count: int = 100,
    seed: int = 99173,
) -> dict[str, Any]:
    """Freeze a one-seed pilot before promoting O1 to full-scale training."""

    source = Path(source_dir)
    selected = {
        split: _balanced_prefix(read_jsonl(source / f"{split}.jsonl"), count)
        for split, count in (
            ("train", train_count),
            ("dev", dev_count),
            ("test", test_count),
        )
    }
    # Older frozen O1 rows predate the generic evaluator's bookkeeping fields.
    for split, rows in selected.items():
        for index, row in enumerate(rows):
            row.setdefault("source_row", index)
            row.setdefault("difficulty_steps", 1)
            row.setdefault("difficulty_stratum", "low")
    output = Path(output_dir)
    eval_path = write_jsonl(output / "test.jsonl", selected["test"])
    output_paths: dict[str, dict[str, Path]] = {}
    for representation, target_key in (("ap", "ap_asl"), ("as", "as_asl")):
        output_paths[representation] = {}
        for split in ("train", "dev"):
            rows = [
                {
                    "schema_version": "ccpu.paper1.gsm8k_oc_sft.v1",
                    "dataset": OPERATOR_DATASET_VERSION,
                    "dataset_id": f"GSM8K_OC_O1_{representation.upper()}_PILOT",
                    "example_id": f"{row['example_id']}:{representation}",
                    "parent_example_id": row["example_id"],
                    "parent_template_id": row["parent_template_id"],
                    "operator_family": row["operator_family"],
                    "representation_id": representation.upper(),
                    "objective_id": "L0",
                    "prompt": _pilot_prompt(str(row["question"]), representation),
                    "target": row["targets"][target_key],
                    "source_fields_visible_to_model": ["question"],
                }
                for row in selected[split]
            ]
            path = write_jsonl(output / representation / f"{split}.jsonl", rows)
            output_paths[representation][split] = path
        evaluation_rows = [
            {
                **row,
                "prompt": _pilot_prompt(str(row["question"]), representation),
                "evaluation_representation": representation.upper(),
            }
            for row in selected["test"]
        ]
        output_paths[representation]["test"] = write_jsonl(
            output / representation / "test.jsonl", evaluation_rows
        )
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_oc_pilot_manifest.v1",
        "dataset_version": OPERATOR_DATASET_VERSION,
        "operator_level": "O1",
        "seed": seed,
        "counts": {"train": train_count, "dev": dev_count, "test": test_count},
        "training": {
            "epochs": 3,
            "unique_records": train_count,
            "exposures_per_adapter": train_count * 3,
            "adapter_representations": ["AP", "AS"],
        },
        "test_operator_counts": dict(
            sorted(Counter(row["operator_family"] for row in selected["test"]).items())
        ),
        "source": {
            "directory": str(source),
            "manifest_sha256": file_sha256(source / "manifest.json"),
        },
        "output_sha256": {
            "test": file_sha256(eval_path),
            **{
                f"{representation}_{split}": file_sha256(path)
                for representation, paths in output_paths.items()
                for split, path in paths.items()
            },
        },
        "promotion_rule": (
            "do not train the 2000-record O1 condition unless the 250-record one-seed pilot "
            "shows a useful accuracy, robustness, representation, or token signal"
        ),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
