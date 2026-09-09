"""Matched GSM8K-derived numeric and operator interventions for Paper 1."""

from __future__ import annotations

import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl

PROTOCOL_ID = "paper1_gsm8k_matched_interventions_v2"
OPERATOR_LEVELS = ("O1", "O3", "O5", "O6")
DEFAULT_FACTORS = (1, 100, 1000, 10000, 1000000)
DEFAULT_JITTER_SEEDS = (17011, 17023, 17037)

_QUESTION_FROM_PROMPT = re.compile(r"\nProblem:\s*(?P<question>.*?)\nASL:\s*$", re.DOTALL)
_LITERAL_ASSIGNMENT = re.compile(
    r"^(?P<path>[A-Za-z][A-Za-z0-9_.]*)\s*=\s*(?P<value>-?\d+)$"
)
_QUESTION_NUMBER = re.compile(r"(?<![\d.])(?P<number>\d[\d,]*)(?![\d.])")


class InterventionExclusion(ValueError):
    """Expected conservative exclusion from the matched intervention panel."""


def _question(row: dict[str, Any]) -> str:
    match = _QUESTION_FROM_PROMPT.search(str(row["prompt"]))
    if not match:
        raise InterventionExclusion("source prompt does not expose one Problem block")
    return match.group("question").strip()


def _returned(program: str, scope_id: str) -> Decimal:
    validation = validate_asl(
        program,
        effective_scope={
            "id": scope_id,
            "parent": None,
            "kind": "benchmark_case",
            "source": PROTOCOL_ID,
        },
    )
    required = ("syntax_verified", "lower_verified", "type_verified", "execution_verified")
    if not all(validation[key] for key in required):
        raise InterventionExclusion("ASL program does not validate and execute")
    return Decimal(str(validation["execution"]["workspace"][scope_id]["returned"]))


def _number_matches(question: str, value: int) -> list[re.Match[str]]:
    return [
        match
        for match in _QUESTION_NUMBER.finditer(question)
        if int(match.group("number").replace(",", "")) == value
    ]


def _eligible_binding(question: str, program: str) -> tuple[str, int, str]:
    candidates = []
    for line in program.splitlines():
        match = _LITERAL_ASSIGNMENT.fullmatch(line.strip())
        if not match:
            continue
        value = int(match.group("value"))
        if not 4 <= value <= 9:
            continue
        occurrences = _number_matches(question, value)
        if len(occurrences) != 1:
            continue
        occurrence = occurrences[0]
        prefix = question[max(0, occurrence.start() - 1) : occurrence.start()]
        suffix = question[occurrence.end() : occurrence.end() + 9].lower()
        if prefix == "$" or re.match(r"\s*(?:%|percent)\b", suffix):
            continue
        candidates.append((match.group("path"), value, line))
    if not candidates:
        raise InterventionExclusion("no unique integer source binding in [4,9]")
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0]


def _replace_question_value(question: str, original: int, replacement: str) -> str:
    matches = _number_matches(question, original)
    if len(matches) != 1:
        raise InterventionExclusion("designated source value is not unique in question")
    match = matches[0]
    return question[: match.start()] + replacement + question[match.end() :]


def _replace_assignment(program: str, original_line: str, replacement: str) -> str:
    lines = program.splitlines()
    indices = [index for index, line in enumerate(lines) if line.strip() == original_line.strip()]
    if len(indices) != 1:
        raise InterventionExclusion("designated source assignment is not unique in ASL")
    lines[indices[0] : indices[0] + 1] = replacement.splitlines()
    return "\n".join(lines)


def _operator_injection(
    level: str, path: str, value: int, original_line: str, program: str
) -> tuple[str, str, str]:
    if level == "O1":
        operand = value * value
        definition = f"the exact square root of {operand}"
        replacement = (
            f"cogcop_probe.o1.square_value = {operand}\n"
            f"{path} = sqrt(cogcop_probe.o1.square_value)"
        )
        family = "sqrt"
    elif level == "O3":
        operand = 10**value
        definition = f"the base-10 logarithm of {operand}"
        replacement = (
            f"cogcop_probe.o3.power_of_ten = {operand}\n"
            f"{path} = log10(cogcop_probe.o3.power_of_ten)"
        )
        family = "log10"
    elif level == "O5":
        coefficient = 2 + value % 4
        offset = 3 + value % 5
        total = coefficient * value + offset
        definition = (
            f"the unique number x satisfying {coefficient} times x plus {offset} "
            f"equals {total}"
        )
        replacement = (
            f"cogcop_probe.o5.a = {coefficient}\n"
            f"cogcop_probe.o5.b = {offset}\n"
            f"cogcop_probe.o5.c = {total}\n"
            f"{path} = solve_linear(cogcop_probe.o5.a, cogcop_probe.o5.b, "
            "cogcop_probe.o5.c)"
        )
        family = "solve_linear"
    elif level == "O6":
        negative_root = 2 + value % 5
        linear = negative_root - value
        constant = -(value * negative_root)
        definition = (
            f"the positive solution of x squared plus {linear} times x plus "
            f"{constant} equals zero"
        )
        replacement = (
            "cogcop_probe.o6.a = 1\n"
            f"cogcop_probe.o6.b = {linear}\n"
            f"cogcop_probe.o6.c = {constant}\n"
            f"{path} = positive_quadratic_root(cogcop_probe.o6.a, "
            "cogcop_probe.o6.b, cogcop_probe.o6.c)"
        )
        family = "positive_quadratic_root"
    else:
        raise ValueError(f"unsupported operator level: {level}")
    return _replace_assignment(program, original_line, replacement), definition, family


def _operator_prompt(question: str) -> str:
    return (
        "Compile the quantitative problem into semantically grounded ASL-Arith. "
        "Available exact operators include sqrt(x), log10(x), solve_linear(a, b, c) "
        "for a*x+b=c, and positive_quadratic_root(a, b, c). Preserve all entities, "
        "source facts, dependencies, and the requested RETURN. Select operators from "
        "the language semantics; no operator has been preselected. Return only ASL, "
        "one statement per line; do not explain.\n\n"
        f"Problem: {question}\nASL:"
    )


def _base_eval_record(
    *, row: dict[str, Any], split: str, question: str, program: str, condition: str
) -> dict[str, Any]:
    parent = f"gsm8k:{row['parent_source_id']}"
    example_id = f"{parent}:{condition}"
    answer = _returned(program, example_id)
    steps = sum(1 for line in program.splitlines() if "=" in line)
    return {
        "schema_version": "ccpu.paper1.gsm8k_intervention.v2",
        "protocol_id": PROTOCOL_ID,
        "dataset": "gsm8k",
        "split": split,
        "condition": condition,
        "example_id": example_id,
        "parent_example_id": parent,
        "source_row": int(row["parent_source_id"]),
        "difficulty_steps": steps,
        "difficulty_stratum": "low" if steps <= 3 else "medium" if steps <= 5 else "high",
        "question": question,
        "question_sha256": fingerprint(question),
        "reference_return": str(answer),
        "source_fields_visible_to_model": ["question"],
        "gold_asl": program,
    }


def _numeric_variant(
    *, row: dict[str, Any], split: str, factor: int, jitter_seed: int | None
) -> dict[str, Any]:
    question = _question(row)
    program = str(row["target"])
    path, value, original_line = _eligible_binding(question, program)
    scaled = value * factor
    jitter = None
    transformed = scaled
    if jitter_seed is not None:
        rng = random.Random(f"{jitter_seed}:{row['parent_source_id']}:{path}:{factor}")
        jitter = rng.uniform(-0.30, 0.30)
        transformed = max(1, round(scaled * (1.0 + jitter)))
        if transformed == scaled:
            transformed = scaled + (1 if jitter >= 0 else -1)
    transformed_program = _replace_assignment(program, original_line, f"{path} = {transformed}")
    transformed_question = _replace_question_value(question, value, f"{transformed:,}")
    label = f"magnitude-x{factor}" if jitter_seed is None else f"jitter-s{jitter_seed}-x{factor}"
    result = _base_eval_record(
        row=row,
        split=split,
        question=transformed_question,
        program=transformed_program,
        condition=label,
    )
    result["transformation"] = {
        "kind": "strict_magnitude" if jitter_seed is None else "uniform_jitter",
        "factor": factor,
        "jitter_seed": jitter_seed,
        "jitter_requested": jitter,
        "jitter_realized": transformed / scaled - 1,
        "binding_path": path,
        "original_value": value,
        "transformed_value": transformed,
        "answer_recomputed_by_runtime": True,
    }
    return result


def _operator_variant(row: dict[str, Any], split: str, level: str) -> dict[str, Any]:
    question = _question(row)
    program = str(row["target"])
    path, value, original_line = _eligible_binding(question, program)
    transformed_program, definition, family = _operator_injection(
        level, path, value, original_line, program
    )
    transformed_question = _replace_question_value(question, value, "this derived quantity")
    transformed_question = (
        f"First determine a quantity defined as {definition}. In the problem below, "
        f"'this derived quantity' refers to that value. {transformed_question}"
    )
    result = _base_eval_record(
        row=row,
        split=split,
        question=transformed_question,
        program=transformed_program,
        condition=f"operator-{level.lower()}",
    )
    result.update(
        {
            "operator_level": level,
            "operator_family": family,
            "operator_signature": [family],
            "prompt": _operator_prompt(transformed_question),
            "transformation": {
                "kind": "operator_injection",
                "binding_path": path,
                "preserved_value": value,
                "preserves_original_program_suffix": True,
            },
        }
    )
    return result


def _source_id(row: dict[str, Any]) -> str:
    return str(row.get("parent_source_id", row.get("parent_example_id", "")))


def freeze_gsm8k_matched_interventions(
    *,
    source_corpus_path: str | Path,
    excluded_training_path: str | Path,
    output_dir: str | Path,
    train_count: int = 250,
    dev_count: int = 30,
    test_count: int = 100,
    factors: tuple[int, ...] = DEFAULT_FACTORS,
    jitter_seeds: tuple[int, ...] = DEFAULT_JITTER_SEEDS,
    seed: int = 99173,
) -> dict[str, Any]:
    """Freeze one held-out GSM8K parent panel for all matched interventions."""

    if factors != tuple(sorted(set(factors))) or not factors or factors[0] != 1:
        raise ValueError("factors must be unique, increasing, and start at 1")
    if any(factor < 1 for factor in factors) or len(set(jitter_seeds)) != len(jitter_seeds):
        raise ValueError("invalid factors or duplicate jitter seeds")
    counts = {"train": train_count, "dev": dev_count, "test": test_count}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all split counts must be positive")

    excluded_ids = {_source_id(row) for row in read_jsonl(excluded_training_path)}
    unique: dict[str, dict[str, Any]] = {}
    exclusions = Counter()
    for row in read_jsonl(source_corpus_path):
        source_id = _source_id(row)
        if not source_id or source_id in excluded_ids or source_id in unique:
            continue
        try:
            question = _question(row)
            program = str(row["target"])
            _eligible_binding(question, program)
            original = _base_eval_record(
                row=row, split="candidate", question=question, program=program, condition="original"
            )
            for factor in factors:
                _numeric_variant(row=row, split="candidate", factor=factor, jitter_seed=None)
                for jitter_seed in jitter_seeds:
                    _numeric_variant(
                        row=row,
                        split="candidate",
                        factor=factor,
                        jitter_seed=jitter_seed,
                    )
            for level in OPERATOR_LEVELS:
                variant = _operator_variant(row, "candidate", level)
                if Decimal(variant["reference_return"]) != Decimal(original["reference_return"]):
                    raise InterventionExclusion("operator injection changed the original answer")
        except (InterventionExclusion, KeyError, ValueError) as error:
            exclusions[str(error)] += 1
            continue
        unique[source_id] = row

    needed = sum(counts.values())
    if len(unique) < needed:
        raise ValueError(f"only {len(unique)} eligible held-out parents; need {needed}")
    ordered = sorted(
        unique.values(),
        key=lambda row: fingerprint([seed, _source_id(row)]),
    )[:needed]
    split_rows: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split, count in counts.items():
        split_rows[split] = ordered[offset : offset + count]
        offset += count

    output = Path(output_dir)
    paths: dict[str, Path] = {}
    parent_ids: dict[str, list[str]] = {}
    for split, rows in split_rows.items():
        originals = [
            _base_eval_record(
                row=row,
                split=split,
                question=_question(row),
                program=str(row["target"]),
                condition="original",
            )
            for row in rows
        ]
        parent_ids[split] = [str(row["parent_example_id"]) for row in originals]
        paths[f"original_{split}"] = write_jsonl(output / "original" / f"{split}.jsonl", originals)
        for factor in factors:
            numeric = [
                _numeric_variant(row=row, split=split, factor=factor, jitter_seed=None)
                for row in rows
            ]
            paths[f"magnitude_x{factor}_{split}"] = write_jsonl(
                output / "magnitude" / f"x{factor}" / f"{split}.jsonl", numeric
            )
            for jitter_seed in jitter_seeds:
                jittered = [
                    _numeric_variant(
                        row=row,
                        split=split,
                        factor=factor,
                        jitter_seed=jitter_seed,
                    )
                    for row in rows
                ]
                paths[f"jitter_s{jitter_seed}_x{factor}_{split}"] = write_jsonl(
                    output / "jitter" / f"seed_{jitter_seed}" / f"x{factor}" / f"{split}.jsonl",
                    jittered,
                )

    operator_sft = {"train": [], "dev": []}
    for level in OPERATOR_LEVELS:
        for split, rows in split_rows.items():
            variants = [_operator_variant(row, split, level) for row in rows]
            paths[f"operator_{level}_{split}"] = write_jsonl(
                output / "operators" / level.lower() / f"{split}.jsonl", variants
            )
            if split in operator_sft:
                operator_sft[split].extend(
                    {
                        "schema_version": "ccpu.paper1.gsm8k_intervention_sft.v2",
                        "dataset_id": "GSM8K_OC_NATURAL_V2",
                        "example_id": row["example_id"],
                        "parent_example_id": row["parent_example_id"],
                        "operator_level": level,
                        "representation_id": "AP",
                        "objective_id": "L0",
                        "prompt": row["prompt"],
                        "target": row["gold_asl"],
                        "source_fields_visible_to_model": ["question"],
                    }
                    for row in variants
                )
    for split, rows in operator_sft.items():
        rows.sort(key=lambda row: fingerprint([seed, split, row["example_id"]]))
        paths[f"operator_sft_{split}"] = write_jsonl(
            output / "operators" / "sft" / f"{split}.jsonl", rows
        )

    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_matched_interventions_manifest.v2",
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "counts": counts,
        "factors": list(factors),
        "jitter": {
            "distribution": "deterministic_uniform[-0.30,+0.30]",
            "seeds": list(jitter_seeds),
            "binding_policy": "one shared source binding per parent across all interventions",
        },
        "operator_levels": list(OPERATOR_LEVELS),
        "source": {
            "corpus": str(source_corpus_path),
            "corpus_sha256": file_sha256(source_corpus_path),
            "excluded_training": str(excluded_training_path),
            "excluded_training_sha256": file_sha256(excluded_training_path),
        },
        "leakage_audit": {
            "excluded_training_source_ids": len(excluded_ids),
            "selected_source_overlap": 0,
            "parent_ids_sha256": {
                split: fingerprint(ids) for split, ids in parent_ids.items()
            },
        },
        "eligibility": {
            "eligible": len(unique),
            "excluded_by_reason": dict(sorted(exclusions.items())),
        },
        "comparison_contract": (
            "Direct and ASL consume the same frozen file for each condition; comparisons "
            "must verify identical ordered example_id and question_sha256 values."
        ),
        "legacy_synthetic_role": "sanity baseline only; not primary crossover evidence",
        "output_sha256": {name: file_sha256(path) for name, path in sorted(paths.items())},
    }
    write_json(output / "manifest.json", manifest)
    return manifest
