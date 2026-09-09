"""Frozen O0 and O3 datasets for the categorical operator ladder."""

from __future__ import annotations

import random
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

LEVEL_VERSION = "gsm8k-oc-levels-v1"
LEVEL_REGISTRY_VERSION = "paper1-operator-registry-v2"
LEVEL_REGISTRY = {
    "O0": {
        "add": {"ccir": "ADD", "exactness": "EXACT_RATIONAL"},
        "subtract": {"ccir": "SUB", "exactness": "EXACT_RATIONAL"},
        "multiply": {"ccir": "MUL", "exactness": "EXACT_RATIONAL"},
        "divide": {"ccir": "DIV", "exactness": "EXACT_RATIONAL"},
    },
    "O3": {
        "sin_degrees": {
            "ccir": "SIN_DEGREES",
            "domain": [0, 30, 45, 60, 90],
            "exactness": "FROZEN_DECIMAL_28",
        },
        "cos_degrees": {
            "ccir": "COS_DEGREES",
            "domain": [0, 30, 45, 60, 90],
            "exactness": "FROZEN_DECIMAL_28",
        },
        "log10": {
            "ccir": "LOG10",
            "domain": "positive integer powers of ten",
            "exactness": "EXACT_INTEGER",
        },
    },
    "O5": {
        "solve_linear": {
            "ccir": "SOLVE_LINEAR",
            "form": "a*x+b=c",
            "exactness": "EXACT_RATIONAL",
        }
    },
    "O6": {
        "positive_quadratic_root": {
            "ccir": "POSITIVE_QUADRATIC_ROOT",
            "domain": "one positive integer root and one negative integer root",
            "exactness": "EXACT_INTEGER",
        },
        "larger_quadratic_root": {
            "ccir": "LARGER_QUADRATIC_ROOT",
            "domain": "two positive integer roots",
            "exactness": "EXACT_INTEGER",
        },
    },
}

_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "O0": {
        "add_train_a": ("add", "Lina has {left} beads and receives {right} more. How many beads does she have?"),
        "add_train_b": ("add", "One shelf holds {left} books and another holds {right}. How many books are there altogether?"),
        "sub_train_a": ("subtract", "A box starts with {left} cards and {right} are removed. How many remain?"),
        "sub_train_b": ("subtract", "A bus carried {left} passengers before {right} got off. How many remain?"),
        "mul_train_a": ("multiply", "There are {left} bags with {right} marbles each. How many marbles are there?"),
        "mul_train_b": ("multiply", "A hall has {left} rows of {right} chairs. How many chairs are there?"),
        "div_train_a": ("divide", "A teacher shares {left} pencils equally among {right} students. How many does each receive?"),
        "div_train_b": ("divide", "Pack {left} bottles equally into {right} crates. How many bottles go in each crate?"),
        "add_dev": ("add", "A jar contains {left} red buttons and {right} blue buttons. How many buttons does it contain?"),
        "sub_dev": ("subtract", "A store had {left} lamps and sold {right}. How many lamps remain?"),
        "mul_dev": ("multiply", "A garden has {left} lines with {right} plants per line. How many plants are there?"),
        "div_dev": ("divide", "Divide {left} stickers equally among {right} albums. How many go in each album?"),
        "add_test": ("add", "Noah walked {left} meters before lunch and {right} afterward. How far did he walk?"),
        "sub_test": ("subtract", "A tank held {left} liters and used {right} liters. How many liters remain?"),
        "mul_test": ("multiply", "A workshop makes {left} boxes with {right} parts in each. How many parts are used?"),
        "div_test": ("divide", "Arrange {left} tiles into {right} equal rows. How many tiles are in each row?"),
    },
    "O3": {
        "sin_train_a": ("sin_degrees", "A unit-length pointer is {value} degrees above horizontal. What is its vertical component? Give a decimal to three places if needed."),
        "sin_train_b": ("sin_degrees", "A unit ramp rises at {value} degrees. What fraction of its length is vertical? Give a decimal to three places if needed."),
        "cos_train_a": ("cos_degrees", "A unit-length beam points {value} degrees above horizontal. What is its horizontal component? Give a decimal to three places if needed."),
        "cos_train_b": ("cos_degrees", "A unit vector forms a {value}-degree angle with the x-axis. What is its x-component? Give a decimal to three places if needed."),
        "log_train_a": ("log10", "What is the base-10 logarithm of {derived}?"),
        "log_train_b": ("log10", "Ten raised to what power equals {derived}?"),
        "sin_dev": ("sin_degrees", "For a unit rope at {value} degrees above level ground, what is the vertical component? Give a decimal to three places if needed."),
        "cos_dev": ("cos_degrees", "For a unit rod at {value} degrees from horizontal, what is the horizontal component? Give a decimal to three places if needed."),
        "log_dev": ("log10", "Find log base 10 of {derived}."),
        "sin_test": ("sin_degrees", "A unit arrow makes a {value}-degree angle above the horizon. Find its vertical component. Give a decimal to three places if needed."),
        "cos_test": ("cos_degrees", "A unit cable is inclined {value} degrees above horizontal. Find its horizontal component. Give a decimal to three places if needed."),
        "log_test": ("log10", "How many factors of 10 are multiplied to obtain {derived}?"),
    },
    "O5": {
        "linear_train_a": ("solve_linear", "A number is multiplied by {a}, then {b} is added, giving {c}. What is the number?"),
        "linear_train_b": ("solve_linear", "After {b} is added to {a} times an unknown quantity, the result is {c}. Find the unknown."),
        "linear_dev": ("solve_linear", "The balance {a} times a number plus {b} equals {c}. What is the number?"),
        "linear_test": ("solve_linear", "A machine multiplies an unknown input by {a} and adds {b}; its output is {c}. Find the input."),
    },
    "O6": {
        "positive_train_a": ("positive_quadratic_root", "A quantity x satisfies x squared plus {b} times x plus {c} equals zero and has exactly one positive solution. Find it."),
        "positive_train_b": ("positive_quadratic_root", "The equation x^2 + ({b})x + ({c}) = 0 has one positive root. What is that root?"),
        "larger_train_a": ("larger_quadratic_root", "A quantity x satisfies x squared plus {b} times x plus {c} equals zero and has two positive solutions. Find the larger one."),
        "larger_train_b": ("larger_quadratic_root", "The equation x^2 + ({b})x + {c} = 0 has two positive roots. What is the larger root?"),
        "positive_dev": ("positive_quadratic_root", "Find the positive solution of x squared plus {b}x plus {c} equals zero."),
        "larger_dev": ("larger_quadratic_root", "For x^2 + ({b})x + {c} = 0, report the larger of its two positive solutions."),
        "positive_test": ("positive_quadratic_root", "A physical length x obeys x squared plus {b}x plus {c} equals zero. Only one root is positive. Find that length."),
        "larger_test": ("larger_quadratic_root", "Two positive values solve x squared plus {b}x plus {c} equals zero. What is the larger value?"),
    },
}


def _split_templates(level: str, split: str) -> list[tuple[str, str, str]]:
    return [
        (template_id, operator, text)
        for template_id, (operator, text) in _TEMPLATES[level].items()
        if template_id.endswith(f"_{split}") or (split == "train" and "_train_" in template_id)
    ]


def _o0_program(operator: str, rng: random.Random) -> tuple[dict[str, Any], str, str, str, Decimal]:
    left = rng.randint(3, 40)
    right = rng.randint(2, 20)
    if operator == "add":
        answer = Decimal(left + right)
        symbol, semantic = "+", "total_of"
    elif operator == "subtract":
        left += right
        answer = Decimal(left - right)
        symbol, semantic = "-", "difference"
    elif operator == "multiply":
        answer = Decimal(left * right)
        symbol, semantic = "*", "product"
    else:
        quotient = rng.randint(2, 20)
        divisor = right
        left = quotient * divisor
        answer = Decimal(quotient)
        symbol, semantic = "/", "quotient"
    values = {"left": left, "right": right, "derived": left}
    ap = f"left = {left}\nright = {right}\nanswer = left {symbol} right\nRETURN answer"
    semantic_program = f"left = {left}\nright = {right}\nanswer = {semantic}(left, right)\nRETURN answer"
    expression = f"{left} {symbol} {right}"
    return values, ap, semantic_program, expression, answer


def _o3_program(operator: str, rng: random.Random) -> tuple[dict[str, Any], str, str, str, Decimal]:
    if operator == "log10":
        exponent = rng.randint(1, 8)
        derived = 10**exponent
        values = {"value": exponent, "derived": derived}
        ap = f"value = {derived}\nanswer = log10(value)\nRETURN answer"
        semantic = f"value = {derived}\nanswer = decimal_order(value)\nRETURN answer"
        return values, ap, semantic, f"log10({derived})", Decimal(exponent)
    angle = rng.choice((0, 30, 45, 60, 90))
    values_table = {
        "sin_degrees": {
            0: "0", 30: "0.5", 45: "0.7071067811865475244008443621",
            60: "0.8660254037844386467637231708", 90: "1",
        },
        "cos_degrees": {
            0: "1", 30: "0.8660254037844386467637231708",
            45: "0.7071067811865475244008443621", 60: "0.5", 90: "0",
        },
    }
    semantic_name = {
        "sin_degrees": "vertical_component_of_unit",
        "cos_degrees": "horizontal_component_of_unit",
    }[operator]
    values = {"value": angle, "derived": angle}
    ap = f"angle_degrees = {angle}\nanswer = {operator}(angle_degrees)\nRETURN answer"
    semantic = f"angle_degrees = {angle}\nanswer = {semantic_name}(angle_degrees)\nRETURN answer"
    return values, ap, semantic, f"{operator}({angle})", Decimal(values_table[operator][angle])


def _o5_program(operator: str, rng: random.Random) -> tuple[dict[str, Any], str, str, str, Decimal]:
    if operator != "solve_linear":
        raise ValueError(f"unsupported O5 operator: {operator}")
    coefficient = rng.randint(2, 12)
    answer = rng.randint(2, 50)
    offset = rng.randint(1, 30)
    total = coefficient * answer + offset
    values = {"a": coefficient, "b": offset, "c": total}
    prefix = f"a = {coefficient}\nb = {offset}\nc = {total}"
    ap = f"{prefix}\nanswer = solve_linear(a, b, c)\nRETURN answer"
    semantic = f"{prefix}\nanswer = unknown_from_linear_balance(a, b, c)\nRETURN answer"
    return values, ap, semantic, f"({total} - {offset}) / {coefficient}", Decimal(answer)


def _o6_program(operator: str, rng: random.Random) -> tuple[dict[str, Any], str, str, str, Decimal]:
    first = rng.randint(3, 25)
    if operator == "positive_quadratic_root":
        second = -rng.randint(2, 15)
        semantic_name = "positive_solution_of_quadratic"
    elif operator == "larger_quadratic_root":
        second = rng.randint(1, first - 1)
        semantic_name = "larger_solution_of_quadratic"
    else:
        raise ValueError(f"unsupported O6 operator: {operator}")
    coefficient = 1
    linear = -(first + second)
    constant = first * second
    values = {"a": coefficient, "b": linear, "c": constant}
    prefix = f"a = {coefficient}\nb = {linear}\nc = {constant}"
    ap = f"{prefix}\nanswer = {operator}(a, b, c)\nRETURN answer"
    semantic = f"{prefix}\nanswer = {semantic_name}(a, b, c)\nRETURN answer"
    expression = f"root({coefficient}*x^2 + {linear}*x + {constant})"
    return values, ap, semantic, expression, Decimal(first)


def _program(level: str, operator: str, rng: random.Random):
    builders = {
        "O0": _o0_program,
        "O3": _o3_program,
        "O5": _o5_program,
        "O6": _o6_program,
    }
    return builders[level](operator, rng)


def _record(level: str, split: str, index: int, template: tuple[str, str, str], rng: random.Random) -> dict[str, Any]:
    template_id, operator, surface = template
    values, ap, semantic, expression, answer = _program(level, operator, rng)
    question = surface.format(**values)
    example_id = f"gsm8k-oc-{level.lower()}-{split}-{index:05d}"
    scope = {"id": example_id, "parent": None, "kind": "benchmark_case", "source": LEVEL_VERSION}
    validations = [validate_asl(program, effective_scope=scope) for program in (ap, semantic)]
    required = ("syntax_verified", "lower_verified", "type_verified", "execution_verified")
    if not all(all(validation[key] for key in required) for validation in validations):
        raise AssertionError(f"generated {level} program failed validation: {example_id}")
    returned = [validation["execution"]["workspace"][example_id]["returned"] for validation in validations]
    if any(Decimal(str(value)) != answer for value in returned):
        raise AssertionError(f"generated {level} answer mismatch: {example_id}")
    ccir = [validation["ccir"]["operations"][-2]["operation"]["expr"]["op"] for validation in validations]
    if ccir[0] != ccir[1]:
        raise AssertionError(f"AP/AS CCIR mismatch: {example_id}")
    return {
        "schema_version": "ccpu.paper1.gsm8k_oc.v1",
        "dataset": LEVEL_VERSION,
        "split": split,
        "example_id": example_id,
        "source_row": index,
        "parent_template_id": template_id,
        "operator_level": level,
        "difficulty_steps": 1,
        "difficulty_stratum": "low",
        "operator_family": operator,
        "operator_signature": [operator],
        "semantic_signature": f"{operator}(CONST)->RETURN",
        "question": question,
        "question_sha256": fingerprint(question),
        "reference_return": str(answer),
        "effective_scope": scope,
        "source_fields_visible_to_model": ["question"],
        "targets": {"ap_asl": ap, "as_asl": semantic, "generic_expression": expression},
        "complexity": {"primitive_operation_count": 1, "dependency_depth": 1, "function_nesting_depth": 1, "entity_count": 1},
        "runtime": {"registry_version": LEVEL_REGISTRY_VERSION, "ccir_operator": ccir[0]},
    }


def freeze_operator_level(output_dir: str | Path, *, level: str, train_count: int = 2000, dev_count: int = 100, test_count: int = 250, seed: int = 81001) -> dict[str, Any]:
    if level not in _TEMPLATES:
        raise ValueError(f"unsupported implemented operator level: {level}")
    counts = {"train": train_count, "dev": dev_count, "test": test_count}
    output = Path(output_dir)
    paths = {}
    split_rows = {}
    for split, count in counts.items():
        templates = _split_templates(level, split)
        rng = random.Random(f"{seed}:{level}:{split}")
        rows = [_record(level, split, index, templates[index % len(templates)], rng) for index in range(count)]
        split_rows[split] = rows
        paths[split] = write_jsonl(output / f"{split}.jsonl", rows)
    registry_path = write_json(
        output / "operator_registry.json",
        {
            "version": LEVEL_REGISTRY_VERSION,
            "level": level,
            "operators": LEVEL_REGISTRY[level],
        },
    )
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_oc_manifest.v1",
        "dataset_version": LEVEL_VERSION,
        "operator_level": level,
        "seed": seed,
        "counts": counts,
        "template_families": {split: [item[0] for item in _split_templates(level, split)] for split in counts},
        "operator_counts": {split: dict(sorted(Counter(row["operator_family"] for row in rows).items())) for split, rows in split_rows.items()},
        "output_sha256": {
            **{split: file_sha256(path) for split, path in paths.items()},
            "operator_registry": file_sha256(registry_path),
        },
        "prompt_fields": ["question"],
        "hidden_fields": ["targets", "reference_return", "runtime"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _prompt(question: str, level: str, representation: str) -> str:
    instructions = {
        ("O0", "ap"): "Use ordinary ASL arithmetic with +, -, *, and /.",
        ("O0", "as"): "Use the semantic operators total_of, difference, product, and quotient.",
        ("O3", "ap"): "Use sin_degrees(x), cos_degrees(x), or log10(x). Angles are degrees.",
        ("O3", "as"): "Use vertical_component_of_unit(angle_degrees), horizontal_component_of_unit(angle_degrees), or decimal_order(value).",
        ("O5", "ap"): "Use solve_linear(a, b, c) for a*x + b = c.",
        ("O5", "as"): "Use unknown_from_linear_balance(a, b, c) for a*x + b = c.",
        ("O6", "ap"): "Use positive_quadratic_root(a, b, c) or larger_quadratic_root(a, b, c).",
        ("O6", "as"): "Use positive_solution_of_quadratic(a, b, c) or larger_solution_of_quadratic(a, b, c).",
    }
    return (
        "Compile the quantitative problem into executable ASL-Arith. "
        f"{instructions[(level, representation)]} Return only ASL, one statement per line; do not explain.\n\n"
        f"Problem: {question}\nASL:"
    )


def _balanced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["operator_family"]), []).append(row)
    selected = []
    offsets = {key: 0 for key in groups}
    while len(selected) < count:
        progressed = False
        for key in sorted(groups):
            if len(selected) == count:
                break
            if offsets[key] < len(groups[key]):
                selected.append(groups[key][offsets[key]])
                offsets[key] += 1
                progressed = True
        if not progressed:
            raise ValueError(f"cannot select {count} balanced records from {len(rows)} rows")
    return selected


def freeze_operator_pilot(source_dir: str | Path, output_dir: str | Path, *, level: str, train_count: int = 250, dev_count: int = 30, test_count: int = 100, seed: int = 99173) -> dict[str, Any]:
    source = Path(source_dir)
    selected = {split: _balanced(read_jsonl(source / f"{split}.jsonl"), count) for split, count in (("train", train_count), ("dev", dev_count), ("test", test_count))}
    output = Path(output_dir)
    base_test = write_jsonl(output / "test.jsonl", selected["test"])
    outputs: dict[str, Path] = {"test": base_test}
    for representation, target in (("ap", "ap_asl"), ("as", "as_asl")):
        for split in ("train", "dev"):
            rows = [
                {
                    "schema_version": "ccpu.paper1.gsm8k_oc_sft.v1",
                    "dataset_id": f"GSM8K_OC_{level}_{representation.upper()}_PILOT",
                    "example_id": f"{row['example_id']}:{representation}",
                    "parent_example_id": row["example_id"],
                    "operator_family": row["operator_family"],
                    "representation_id": representation.upper(),
                    "objective_id": "L0",
                    "prompt": _prompt(str(row["question"]), level, representation),
                    "target": row["targets"][target],
                    "source_fields_visible_to_model": ["question"],
                }
                for row in selected[split]
            ]
            outputs[f"{representation}_{split}"] = write_jsonl(output / representation / f"{split}.jsonl", rows)
        eval_rows = [{**row, "prompt": _prompt(str(row["question"]), level, representation), "evaluation_representation": representation.upper()} for row in selected["test"]]
        outputs[f"{representation}_test"] = write_jsonl(output / representation / "test.jsonl", eval_rows)
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_oc_pilot_manifest.v1",
        "dataset_version": LEVEL_VERSION,
        "operator_level": level,
        "seed": seed,
        "counts": {"train": train_count, "dev": dev_count, "test": test_count},
        "training": {"epochs": 3, "unique_records": train_count, "exposures_per_adapter": train_count * 3, "adapter_representations": ["AP", "AS"]},
        "test_operator_counts": dict(sorted(Counter(row["operator_family"] for row in selected["test"]).items())),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "output_sha256": {name: file_sha256(path) for name, path in outputs.items()},
    }
    write_json(output / "manifest.json", manifest)
    return manifest
