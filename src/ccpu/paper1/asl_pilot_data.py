"""Frozen grouped splits and controlled data augmentation for the Paper 1 ASL pilot."""

from __future__ import annotations

import copy
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl import parse_asl, validate_asl

_FAMILY_TERMS = {
    "age": {"age", "aged", "old"},
    "money": {
        "amount",
        "cost",
        "expense",
        "fee",
        "income",
        "price",
        "profit",
        "revenue",
        "sales",
        "value",
    },
    "percent": {"percent", "percentage", "pct", "rate"},
    "time": {"day", "days", "duration", "hour", "hours", "minute", "minutes", "year"},
    "distance": {"distance", "feet", "kilometer", "kilometers", "miles", "yards"},
    "mass": {"grams", "kilograms", "ounces", "pounds", "weight"},
    "volume": {"cups", "gallons", "liters", "milliliters", "volume"},
    "average": {"average", "mean"},
    "difference": {"change", "difference", "decrease", "increase"},
    "total": {"all", "combined", "overall", "population", "sum", "total"},
    "count": {
        "bags",
        "bears",
        "books",
        "boxes",
        "cartons",
        "chickens",
        "clips",
        "count",
        "customers",
        "flowers",
        "games",
        "items",
        "kids",
        "members",
        "number",
        "people",
        "petals",
        "stamps",
        "students",
    },
}
_NAME_POOL = (
    "maria",
    "helen",
    "amina",
    "lucas",
    "diego",
    "sofia",
    "noah",
    "priya",
    "mateo",
    "leila",
    "owen",
    "zara",
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9_])")


def _path_tokens(path: str) -> list[str]:
    return [token for token in re.split(r"[._]+", path.casefold()) if token]


def semantic_family(path: str) -> str:
    tokens = set(_path_tokens(path))
    for family, terms in _FAMILY_TERMS.items():
        if tokens & terms:
            return family
    return "other"


def _normalized_expr(
    node: dict[str, Any], targets: dict[str, str], *, retain_values: bool
) -> dict[str, Any]:
    operation = str(node["op"])
    if operation == "CONST":
        value = node.get("value")
        return {
            "op": "CONST",
            "value": value if retain_values else type(value).__name__,
        }
    if operation == "REF":
        path = str(node["path"])
        return {
            "op": "REF",
            "target": targets.get(path, f"external:{semantic_family(path)}"),
        }
    arguments = [
        _normalized_expr(argument, targets, retain_values=retain_values)
        for argument in node.get("args", [])
    ]
    if operation in {"ADD", "MUL", "SUM", "MIN", "MAX"}:
        arguments = sorted(arguments, key=fingerprint)
    return {"op": operation, "args": arguments}


def semantic_pattern(row: dict[str, Any]) -> dict[str, Any]:
    operations = list(row["ccir"]["operations"])
    target_paths = [
        str(item["operation"]["target"]) for item in operations if item["operation"]["op"] == "SET"
    ]
    targets = {path: f"v{index}" for index, path in enumerate(target_paths)}
    normalized = []
    for item in operations:
        operation = item["operation"]
        if operation["op"] == "SET":
            path = str(operation["target"])
            normalized.append(
                {
                    "op": "SET",
                    "target": targets[path],
                    "family": semantic_family(path),
                    "expr": _normalized_expr(operation["expr"], targets, retain_values=False),
                }
            )
        elif operation["op"] == "RETURN":
            normalized.append(
                {
                    "op": "RETURN",
                    "expr": _normalized_expr(operation["expr"], targets, retain_values=False),
                }
            )
    return {
        "operations": normalized,
        "part_statement_counts": [len(mapping["asl"]) for mapping in row["part_mappings"]],
    }


def pattern_id(row: dict[str, Any]) -> str:
    return f"asl-pattern-{fingerprint(semantic_pattern(row), 16)}"


def _group_stats(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    return (
        len(rows),
        sum(row["dataset"] == "tatqa" for row in rows),
        sum(int(row["provenance"]["repair_round"]) > 0 for row in rows),
    )


def _subset_options(
    group_ids: list[str], groups: dict[str, list[dict[str, Any]]], target: int
) -> list[tuple[tuple[int, int], tuple[str, ...]]]:
    states: dict[tuple[int, int, int], tuple[str, ...]] = {(0, 0, 0): ()}
    for group_id in group_ids:
        size, tatqa, repaired = _group_stats(groups[group_id])
        for state, selected in list(states.items()):
            count, tatqa_count, repaired_count = state
            next_state = (count + size, tatqa_count + tatqa, repaired_count + repaired)
            if next_state[0] <= target and next_state not in states:
                states[next_state] = (*selected, group_id)
    target_tatqa = round(target * 50 / 150)
    target_repaired = round(target * 52 / 150)
    options = [
        (
            (abs(tatqa - target_tatqa) + abs(repaired - target_repaired), len(selected)),
            selected,
        )
        for (count, tatqa, repaired), selected in states.items()
        if count == target
    ]
    return sorted(options, key=lambda item: (item[0], fingerprint(item[1])))


def _choose_grouped_splits(
    groups: dict[str, list[dict[str, Any]]], *, seed: int
) -> dict[str, set[str]]:
    group_ids = sorted(groups)
    best: tuple[tuple[int, str], dict[str, set[str]]] | None = None
    for attempt in range(200):
        order = list(group_ids)
        random.Random(seed + attempt).shuffle(order)
        for test_penalty, test_ids_tuple in _subset_options(order, groups, 25)[:24]:
            test_ids = set(test_ids_tuple)
            remaining = [group_id for group_id in order if group_id not in test_ids]
            dev_options = _subset_options(remaining, groups, 25)
            if not dev_options:
                continue
            dev_penalty, dev_ids_tuple = dev_options[0]
            dev_ids = set(dev_ids_tuple)
            train_ids = set(group_ids) - test_ids - dev_ids
            score = (
                test_penalty[0] + dev_penalty[0],
                fingerprint(
                    {
                        "train": sorted(train_ids),
                        "dev": sorted(dev_ids),
                        "test": sorted(test_ids),
                    }
                ),
            )
            candidate = {"train": train_ids, "dev": dev_ids, "test": test_ids}
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None and best[0][0] == 0:
            break
    if best is None:
        raise ValueError("could not construct exact grouped 100/25/25 split")
    return best[1]


def freeze_asl_pilot(
    accepted_path: str | Path, output_dir: str | Path, *, seed: int = 731993
) -> dict[str, Any]:
    rows = read_jsonl(accepted_path)
    if len(rows) != 150:
        raise ValueError(f"expected 150 accepted semantic programs, found {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row = {**row, "semantic_pattern_id": pattern_id(row)}
        groups[row["semantic_pattern_id"]].append(row)
    assignment = _choose_grouped_splits(groups, seed=seed)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    output = Path(output_dir)
    split_hashes = {}
    ledger_rows = []
    for split in ("train", "dev", "test"):
        members = sorted(
            [row for group_id in assignment[split] for row in groups[group_id]],
            key=lambda row: fingerprint(f"{seed}:{split}:{row['record_sha256']}"),
        )
        if len(members) != {"train": 100, "dev": 25, "test": 25}[split]:
            raise AssertionError(f"unexpected {split} size: {len(members)}")
        split_rows[split] = members
        path = write_jsonl(output / "splits" / f"{split}.jsonl", members)
        split_hashes[split] = file_sha256(path)
        for row in members:
            ledger_rows.append(
                {
                    "dataset": row["dataset"],
                    "source_id": row["source_id"],
                    "record_sha256": row["record_sha256"],
                    "semantic_pattern_id": row["semantic_pattern_id"],
                    "split": split,
                    "repair_round": row["provenance"]["repair_round"],
                    "statement_count": len(row["ccir"]["operations"]),
                }
            )
    ledger_path = write_jsonl(output / "split_ledger.jsonl", ledger_rows)
    pattern_sets = {
        split: {row["semantic_pattern_id"] for row in members}
        for split, members in split_rows.items()
    }
    overlap = {
        "train_dev": sorted(pattern_sets["train"] & pattern_sets["dev"]),
        "train_test": sorted(pattern_sets["train"] & pattern_sets["test"]),
        "dev_test": sorted(pattern_sets["dev"] & pattern_sets["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"semantic pattern leakage: {overlap}")
    manifest = {
        "schema_version": "ccpu.paper1.asl_pilot_freeze.v1",
        "seed": seed,
        "source_sha256": file_sha256(accepted_path),
        "split_sha256": split_hashes,
        "ledger_sha256": file_sha256(ledger_path),
        "counts": {split: len(members) for split, members in split_rows.items()},
        "by_split": {
            split: {
                "datasets": dict(Counter(row["dataset"] for row in members)),
                "repair_rounds": dict(
                    Counter(str(row["provenance"]["repair_round"]) for row in members)
                ),
                "pattern_count": len(pattern_sets[split]),
            }
            for split, members in split_rows.items()
        },
        "semantic_pattern_overlap": overlap,
        "grouping_policy": (
            "renaming-invariant CCIR topology plus semantic quantity family and clause layout"
        ),
        "test_frozen_before_model_runs": True,
    }
    write_json(output / "freeze_manifest.json", manifest)
    return manifest


def _compact_context(row: dict[str, Any]) -> str:
    context = row.get("source_context") or {}
    table = list(context.get("table", []))
    paragraphs = list(context.get("paragraphs", []))
    if not table and not paragraphs:
        return ""
    question_tokens = {
        token
        for token in re.findall(r"[a-z]{4,}", row["question"].casefold())
        if token not in {"what", "which", "between", "from", "average", "total"}
    }
    asl_numbers = set(_NUMBER.findall(row["asl"]))
    selected_table = []
    for index, table_row in enumerate(table):
        text = " | ".join(str(cell) for cell in table_row)
        lowered = text.casefold()
        relevant = bool(question_tokens & set(re.findall(r"[a-z]{4,}", lowered)))
        relevant = relevant or any(
            number.replace(",", "") in text.replace(",", "") for number in asl_numbers
        )
        if index < 3 or relevant:
            selected_table.append(text)
    selected_paragraphs = []
    for paragraph in paragraphs:
        text = str(paragraph.get("text", ""))
        lowered = text.casefold()
        if question_tokens & set(re.findall(r"[a-z]{4,}", lowered)):
            selected_paragraphs.append(text)
    lines = [*(f"TABLE: {line}" for line in selected_table[:12])]
    lines.extend(f"TEXT: {line}" for line in selected_paragraphs[:3])
    return "\n".join(lines)


def asl_prompt(row: dict[str, Any], demonstrations: list[dict[str, Any]] | None = None) -> str:
    instruction = (
        "Compile the quantitative problem into semantically grounded ASL-Arith. "
        "Preserve entities, measured quantities, source facts, relations, temporal state, "
        "dependencies, and the requested RETURN. Use meaningful lowercase paths. "
        "Return only ASL, one statement per line; do not explain."
    )
    sections = [instruction]
    for index, demo in enumerate(demonstrations or [], 1):
        context = _compact_context(demo)
        sections.append(
            f"\nExample {index} input:\n"
            + (f"Evidence:\n{context}\n" if context else "")
            + f"Problem: {demo['question']}\nExample {index} ASL:\n{demo['asl']}"
        )
    context = _compact_context(row)
    sections.append(
        "\nInput:\n"
        + (f"Evidence:\n{context}\n" if context else "")
        + f"Problem: {row['question']}\nASL:"
    )
    return "\n".join(sections)


def _number_literals(asl: str) -> list[str]:
    program = parse_asl(asl)
    values = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "number":
                values.append(str(node["value"]))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(program)
    return list(dict.fromkeys(values))


def _replace_number(text: str, old: str, new: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])")
    return pattern.sub(new, text)


def _replace_context(context: Any, replacements: dict[str, str]) -> Any:
    if isinstance(context, str):
        result = context
        for old, new in replacements.items():
            result = _replace_number(result, old, new)
        return result
    if isinstance(context, list):
        return [_replace_context(item, replacements) for item in context]
    if isinstance(context, dict):
        return {key: _replace_context(value, replacements) for key, value in context.items()}
    return context


def _paraphrase(question: str, variant: int) -> str:
    replacements = (
        (r"\bHow many\b", "What is the number of"),
        (r"\bWhat is\b", "Determine"),
        (r"\bcalculate\b", "determine"),
        (r"\bEach\b", "Every"),
        (r"\baltogether\b", "in total"),
        (r"\bin all\b", "altogether"),
    )
    result = question
    start = variant % len(replacements)
    changed = False
    for offset in range(len(replacements)):
        pattern, replacement = replacements[(start + offset) % len(replacements)]
        updated, count = re.subn(pattern, replacement, result, count=1, flags=re.IGNORECASE)
        if count:
            result = updated
            changed = True
            if variant % 2 == 0:
                break
    return result if changed else f"Solve this equivalent quantitative task: {question}"


def _rename_entities(row: dict[str, Any], variant: int) -> tuple[str, str, dict[str, Any], bool]:
    question = row["question"]
    asl = row["asl"]
    context = copy.deepcopy(row.get("source_context"))
    roots = list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(r"(?m)^([a-z_][a-z0-9_]*)\.", asl)
            if re.search(rf"\b{re.escape(match.group(1))}\b", question, flags=re.IGNORECASE)
        )
    )
    changed = False
    for index, root in enumerate(roots[:2]):
        replacement = _NAME_POOL[(variant * 2 + index) % len(_NAME_POOL)]
        if replacement == root:
            replacement = _NAME_POOL[(variant * 2 + index + 1) % len(_NAME_POOL)]
        question = re.sub(
            rf"\b{re.escape(root)}\b", replacement.title(), question, flags=re.IGNORECASE
        )
        asl = re.sub(rf"\b{re.escape(root)}(?=\.)", replacement, asl)
        if context:
            context_text = str(context)
            if re.search(rf"\b{re.escape(root)}\b", context_text, flags=re.IGNORECASE):
                context = _replace_entity_context(context, root, replacement.title())
        changed = True
    return question, asl, context, changed


def _replace_entity_context(context: Any, old: str, new: str) -> Any:
    if isinstance(context, str):
        return re.sub(rf"\b{re.escape(old)}\b", new, context, flags=re.IGNORECASE)
    if isinstance(context, list):
        return [_replace_entity_context(item, old, new) for item in context]
    if isinstance(context, dict):
        return {key: _replace_entity_context(value, old, new) for key, value in context.items()}
    return context


def _percentage_literals(asl: str, question: str) -> set[str]:
    protected = {
        match.group(1)
        for match in re.finditer(r"(?i)(-?\d+(?:\.\d+)?)\s*(?:%|percent\b|percentage\b)", question)
    }
    protected.update(
        match.group(1)
        for match in re.finditer(
            r"(?im)^[a-z_][a-z0-9_.]*(?:pct|percent|rate)\s*=\s*(-?\d+(?:\.\d+)?)\s*$",
            asl,
        )
    )
    protected.update(
        match.group(1)
        for match in re.finditer(
            r"(?i)(?:percent_of|inc_pct|dec_pct)\([^,\n]+,\s*(-?\d+(?:\.\d+)?)\s*\)",
            asl,
        )
    )
    if re.search(r"(?i)(?:percent_of|inc_pct|dec_pct|(?:pct|percent|rate)\s*=)", asl):
        protected.add("100")
    return protected


def perturb_row(row: dict[str, Any], *, variant: int, mode: str) -> dict[str, Any]:
    transformed = copy.deepcopy(row)
    if mode == "augmented":
        question, asl, context, entity_changed = _rename_entities(transformed, variant)
    else:
        question = str(transformed["question"])
        asl = str(transformed["asl"])
        context = copy.deepcopy(transformed.get("source_context"))
        entity_changed = False
    replacements = {}
    surface = f"{question}\n{context or ''}"
    protected_percentages = _percentage_literals(asl, question)
    rng = random.Random(int(fingerprint(f"{row['record_sha256']}:{mode}:{variant}", 12), 16))
    if mode in {"numeric", "large", "augmented"}:
        for literal in _number_literals(asl):
            try:
                numeric = float(literal)
            except ValueError:
                continue
            if (
                literal in protected_percentages
                or 1900 <= abs(numeric) <= 2100
                or not re.search(rf"(?<![A-Za-z0-9_]){re.escape(literal)}(?![A-Za-z0-9_])", surface)
            ):
                continue
            if mode == "large" or (mode == "augmented" and variant % 4 == 0):
                replacement = str(rng.randint(10_000, 999_999))
            else:
                replacement = str(rng.randint(3, 999))
            if "." in literal:
                replacement = f"{rng.randint(30, 9999) / 10:.1f}"
            replacements[literal] = replacement
        if mode in {"numeric", "large"} and not replacements:
            raise ValueError("no safely surface-matched numeric literal")
    for old, new in replacements.items():
        question = _replace_number(question, old, new)
        asl = _replace_number(asl, old, new)
    if context:
        context = _replace_context(context, replacements)
    if mode in {"paraphrase", "augmented"}:
        question = _paraphrase(question, variant)
    transformed.update({"question": question, "asl": asl, "source_context": context})
    validation = validate_asl(asl, effective_scope=row["effective_scope"])
    if not validation["execution_verified"]:
        raise ValueError("perturbed ASL is not executable: " + "; ".join(validation["errors"]))
    transformed["perturbation"] = {
        "mode": mode,
        "variant": variant,
        "numeric_replacements": replacements,
        "entity_renamed": entity_changed,
        "question_changed": question != row["question"],
    }
    transformed["reference_return"] = validation["execution"]["workspace"][
        str(row["effective_scope"]["id"])
    ]["returned"]
    transformed["parent_source_id"] = row["source_id"]
    return transformed


def _sft_record(row: dict[str, Any], *, variant_id: str) -> dict[str, Any]:
    identity = f"{row['dataset']}:{row['source_id']}:{variant_id}"
    return {
        "schema_version": "ccpu.paper1.asl_sft.v1",
        "example_id": f"asl-sft-{fingerprint(identity, 16)}",
        "parent_source_id": row["source_id"],
        "semantic_pattern_id": row["semantic_pattern_id"],
        "dataset": row["dataset"],
        "prompt": asl_prompt(row),
        "target": row["asl"],
        "augmentation": row.get("perturbation"),
    }


def _diverse_order(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], int(row["provenance"]["repair_round"]) > 0)
        buckets[key].append(row)
    for key, members in buckets.items():
        members.sort(key=lambda row: fingerprint(f"{seed}:{key}:{row['record_sha256']}"))
    ordered = []
    keys = sorted(buckets)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return ordered


def build_asl_pilot_data(
    freeze_dir: str | Path,
    output_dir: str | Path,
    *,
    augmentation_variants: int = 9,
    seed: int = 912733,
) -> dict[str, Any]:
    freeze_dir = Path(freeze_dir)
    output = Path(output_dir)
    splits = {
        split: read_jsonl(freeze_dir / "splits" / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    ordered_train = _diverse_order(list(splits["train"]), seed=seed)
    files = {}
    for count in (25, 50, 100):
        rows = [_sft_record(row, variant_id="original") for row in ordered_train[:count]]
        path = write_jsonl(output / "sft" / f"train_{count}.jsonl", rows)
        files[f"train_{count}"] = {"rows": len(rows), "sha256": file_sha256(path)}
    dev_sft = [_sft_record(row, variant_id="original") for row in splits["dev"]]
    dev_path = write_jsonl(output / "sft" / "dev.jsonl", dev_sft)
    files["dev"] = {"rows": len(dev_sft), "sha256": file_sha256(dev_path)}

    augmented = [_sft_record(row, variant_id="original") for row in ordered_train]
    augmentation_failures = []
    for row in ordered_train:
        created = 0
        for variant in range(1, augmentation_variants * 4 + 1):
            try:
                transformed = perturb_row(row, variant=variant, mode="augmented")
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                augmentation_failures.append(
                    {"source_id": row["source_id"], "variant": variant, "reason": str(error)}
                )
                continue
            augmented.append(_sft_record(transformed, variant_id=f"augmented-{variant}"))
            created += 1
            if created == augmentation_variants:
                break
        if created < augmentation_variants:
            raise ValueError(
                f"only generated {created}/{augmentation_variants} variants for {row['source_id']}"
            )
    augmented_path = write_jsonl(output / "sft" / "train_augmented.jsonl", augmented)
    files["train_augmented"] = {
        "rows": len(augmented),
        "sha256": file_sha256(augmented_path),
    }

    eval_sets: dict[str, list[dict[str, Any]]] = {"original": list(splits["test"])}
    eval_perturbation_failures = []
    for mode in ("numeric", "large", "paraphrase"):
        eval_sets[mode] = []
        for index, row in enumerate(splits["test"]):
            try:
                eval_sets[mode].append(perturb_row(row, variant=10_000 + index, mode=mode))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                eval_perturbation_failures.append(
                    {"suite": mode, "source_id": row["source_id"], "reason": str(error)}
                )
    for name, rows in eval_sets.items():
        records = []
        for row in rows:
            validation = validate_asl(row["asl"], effective_scope=row["effective_scope"])
            records.append(
                {
                    "schema_version": "ccpu.paper1.asl_eval.v1",
                    "example_id": f"asl-eval-{name}-{fingerprint(row['record_sha256'], 12)}",
                    "parent_source_id": row["source_id"],
                    "semantic_pattern_id": row["semantic_pattern_id"],
                    "dataset": row["dataset"],
                    "suite": name,
                    "question": row["question"],
                    "source_context": row.get("source_context"),
                    "prompt": asl_prompt(row),
                    "reference_asl": row["asl"],
                    "effective_scope": row["effective_scope"],
                    "reference_return": validation["execution"]["workspace"][
                        str(row["effective_scope"]["id"])
                    ]["returned"],
                    "perturbation": row.get("perturbation"),
                }
            )
        path = write_jsonl(output / "eval" / f"{name}.jsonl", records)
        files[f"eval_{name}"] = {"rows": len(records), "sha256": file_sha256(path)}

    train_parents = {row["source_id"] for row in splits["train"]}
    dev_parents = {row["source_id"] for row in splits["dev"]}
    test_parents = {row["source_id"] for row in splits["test"]}
    audit = {
        "schema_version": "ccpu.paper1.asl_pilot_leakage_audit.v1",
        "train_dev_parent_overlap": sorted(train_parents & dev_parents),
        "train_test_parent_overlap": sorted(train_parents & test_parents),
        "dev_test_parent_overlap": sorted(dev_parents & test_parents),
        "augmented_parent_outside_train": sorted(
            {row["parent_source_id"] for row in augmented} - train_parents
        ),
        "evaluation_parent_outside_test": sorted(
            {
                row["source_id"]
                for name, members in eval_sets.items()
                if name != "original"
                for row in members
            }
            - test_parents
        ),
    }
    audit["passed"] = not any(
        value for key, value in audit.items() if key.endswith(("overlap", "train", "test"))
    )
    if not audit["passed"]:
        raise AssertionError(f"ASL pilot leakage audit failed: {audit}")
    audit_path = write_json(output / "leakage_audit.json", audit)
    manifest = {
        "schema_version": "ccpu.paper1.asl_pilot_data.v1",
        "freeze_manifest_sha256": file_sha256(freeze_dir / "freeze_manifest.json"),
        "seed": seed,
        "augmentation_variants_per_train_record": augmentation_variants,
        "files": files,
        "leakage_audit_sha256": file_sha256(audit_path),
        "augmentation_failure_count": len(augmentation_failures),
        "evaluation_perturbation_failure_count": len(eval_perturbation_failures),
        "evaluation_coverage": {
            name: {"rows": len(rows), "eligible_fraction": len(rows) / len(splits["test"])}
            for name, rows in eval_sets.items()
        },
        "test_semantic_programs_untouched_by_training": True,
    }
    write_json(output / "data_manifest.json", manifest)
    write_json(output / "augmentation_failures.json", augmentation_failures)
    write_json(output / "evaluation_perturbation_failures.json", eval_perturbation_failures)
    return manifest


def build_asl_expansion_data(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 912734,
) -> dict[str, Any]:
    """Materialize the 500-original checkpoint while preserving frozen evaluation."""

    freeze = Path(freeze_dir)
    expansion_train_path = Path(expansion_train_path)
    original = {
        split: read_jsonl(freeze / "splits" / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    expansion = read_jsonl(expansion_train_path)
    train = [*original["train"], *expansion]
    train_ids = {(row["dataset"], row["source_id"]) for row in train}
    dev_ids = {(row["dataset"], row["source_id"]) for row in original["dev"]}
    test_ids = {(row["dataset"], row["source_id"]) for row in original["test"]}
    train_patterns = {row["semantic_pattern_id"] for row in train}
    dev_patterns = {row["semantic_pattern_id"] for row in original["dev"]}
    test_patterns = {row["semantic_pattern_id"] for row in original["test"]}
    audit = {
        "schema_version": "ccpu.paper1.asl_expansion_leakage_audit.v1",
        "train_dev_parent_overlap": sorted(train_ids & dev_ids),
        "train_test_parent_overlap": sorted(train_ids & test_ids),
        "train_dev_pattern_overlap": sorted(train_patterns & dev_patterns),
        "train_test_pattern_overlap": sorted(train_patterns & test_patterns),
        "duplicate_train_parents": len(train) - len(train_ids),
    }
    audit["passed"] = not any(
        value for key, value in audit.items() if key.endswith(("overlap", "parents"))
    )
    if not audit["passed"]:
        raise AssertionError(f"ASL expansion leakage audit failed: {audit}")

    output = Path(output_dir)
    ordered_train = _diverse_order(train, seed=seed)
    train_path = write_jsonl(
        output / "sft" / "train_450.jsonl",
        (_sft_record(row, variant_id="original") for row in ordered_train),
    )
    dev_path = write_jsonl(
        output / "sft" / "dev.jsonl",
        (_sft_record(row, variant_id="original") for row in original["dev"]),
    )
    audit_path = write_json(output / "leakage_audit.json", audit)
    manifest = {
        "schema_version": "ccpu.paper1.asl_expansion_data.v1",
        "seed": seed,
        "train_rows": len(train),
        "dev_rows": len(original["dev"]),
        "test_rows_unchanged": len(original["test"]),
        "train_pattern_count": len(train_patterns),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in train).items())),
        "repair_round_counts": dict(
            sorted(Counter(str(row["provenance"]["repair_round"]) for row in train).items())
        ),
        "input_sha256": {
            "freeze_manifest": file_sha256(freeze / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
        },
        "output_sha256": {
            "train_450": file_sha256(train_path),
            "dev": file_sha256(dev_path),
            "leakage_audit": file_sha256(audit_path),
        },
    }
    write_json(output / "data_manifest.json", manifest)
    return manifest


def incremental_prompt(row: dict[str, Any], part: dict[str, Any], state: dict[str, Any]) -> str:
    """Render one causal transition prompt from runtime-visible information."""

    context = _compact_context(row)
    prompt_parts = [
        "Compile only the next quantitative clause into a grounded ASL-Arith delta.",
        "Use the current runtime state; preserve entities, quantities, and dependencies.",
        "Return only new ASL statements, one per line; do not repeat prior statements.",
        f"Scope: {row['effective_scope']['id']}",
    ]
    if context:
        prompt_parts.append(context)
    prompt_parts.extend(
        [
            f"Current runtime state: {canonical_json(state)}",
            f"Next clause: {part['text']}",
            "ASL delta:",
        ]
    )
    return "\n\n".join(prompt_parts)


def _incremental_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    parts = {int(part["part_id"]): part for part in row["parts"]}
    statements: list[str] = []
    state: dict[str, Any] = {"values": {}, "unresolved": []}
    records = []
    for mapping in sorted(row["part_mappings"], key=lambda item: int(item["part_id"])):
        part_id = int(mapping["part_id"])
        target = "\n".join(mapping["asl"])
        identity = f"{row['dataset']}:{row['source_id']}:{part_id}:incremental"
        records.append(
            {
                "schema_version": "ccpu.paper1.asl_incremental_sft.v1",
                "example_id": f"asl-inc-{fingerprint(identity, 16)}",
                "parent_source_id": row["source_id"],
                "semantic_pattern_id": row["semantic_pattern_id"],
                "dataset": row["dataset"],
                "part_id": part_id,
                "prompt": incremental_prompt(row, parts[part_id], state),
                "target": target,
                "prior_statement_count": len(statements),
            }
        )
        statements.extend(mapping["asl"])
        validation = validate_asl("\n".join(statements), effective_scope=row["effective_scope"])
        if not all(
            validation[key] for key in ("syntax_verified", "lower_verified", "type_verified")
        ):
            raise ValueError(
                f"incremental prefix failed for {row['source_id']} part {part_id}: "
                + "; ".join(validation["errors"])
            )
        workspace = validation["execution"]["workspace"][str(row["effective_scope"]["id"])]
        state = {
            "values": workspace["values"],
            "returned": workspace["returned"],
            "unresolved": validation["execution"]["unresolved"],
        }
    return records


def build_asl_incremental_data(
    freeze_dir: str | Path,
    expansion_train_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 912735,
) -> dict[str, Any]:
    """Derive clause-local NL+executed-state transitions without new supervision."""

    freeze = Path(freeze_dir)
    original_train = read_jsonl(freeze / "splits" / "train.jsonl")
    dev = read_jsonl(freeze / "splits" / "dev.jsonl")
    expansion = read_jsonl(expansion_train_path)
    train_programs = _diverse_order([*original_train, *expansion], seed=seed)
    train_records = [record for row in train_programs for record in _incremental_records(row)]
    dev_records = [record for row in dev for record in _incremental_records(row)]
    output = Path(output_dir)
    train_path = write_jsonl(output / "sft" / "train_incremental.jsonl", train_records)
    dev_path = write_jsonl(output / "sft" / "dev_incremental.jsonl", dev_records)
    manifest = {
        "schema_version": "ccpu.paper1.asl_incremental_data.v1",
        "seed": seed,
        "train_programs": len(train_programs),
        "train_transitions": len(train_records),
        "dev_programs": len(dev),
        "dev_transitions": len(dev_records),
        "mean_train_transitions_per_program": len(train_records) / len(train_programs),
        "causal_prompt_boundary": (
            "current clause plus compact evidence, scope, prior executed values, and unresolved prior dependencies"
        ),
        "answer_and_future_clauses_hidden": True,
        "input_sha256": {
            "freeze_manifest": file_sha256(freeze / "freeze_manifest.json"),
            "expansion_train": file_sha256(expansion_train_path),
        },
        "output_sha256": {
            "train_incremental": file_sha256(train_path),
            "dev_incremental": file_sha256(dev_path),
        },
    }
    write_json(output / "data_manifest.json", manifest)
    return manifest
