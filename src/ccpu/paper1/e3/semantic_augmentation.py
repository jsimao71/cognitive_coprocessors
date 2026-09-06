"""Leakage-safe incremental semantic augmentation for GSM8K ASL training."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl
from ccpu.paper1.asl_matrix.qwen import autonomous_asl_prompt


@dataclass(frozen=True)
class ParaphraseRule:
    rule_id: str
    relation_class: str
    pattern: str
    replacement: str


# Rules alter one explicit relation phrase at a time. Broad random synonym
# replacement is excluded because it cannot establish semantic preservation.
RELATION_PARAPHRASE_RULES = (
    ParaphraseRule(
        "multiple_twice_as_many",
        "multiplicative_comparison",
        r"\btwice as many\b",
        "two times as many",
    ),
    ParaphraseRule(
        "multiple_twice_the", "multiplicative_comparison", r"\btwice the\b", "two times the"
    ),
    ParaphraseRule("multiple_twice", "multiplicative_comparison", r"\btwice\b", "two times"),
    ParaphraseRule("ratio_half_as_many", "ratio_fraction", r"\bhalf as many\b", "one-half as many"),
    ParaphraseRule("ratio_half_the", "ratio_fraction", r"\bhalf the\b", "one-half the"),
    ParaphraseRule("ratio_half_of", "ratio_fraction", r"\bhalf of\b", "one-half of"),
    ParaphraseRule("multiple_triple", "multiplicative_comparison", r"\btriple\b", "three times"),
    ParaphraseRule(
        "allocation_sentence_initial_each",
        "equal_allocation",
        r"^Each (?!of\b)",
        "Every ",
    ),
    ParaphraseRule(
        "allocation_sentence_each",
        "equal_allocation",
        r"(?<=[.!?] )Each (?!of\b)",
        "Every ",
    ),
    ParaphraseRule("allocation_each_of", "equal_allocation", r"\beach of\b", "every one of"),
    ParaphraseRule("allocation_for_each", "equal_allocation", r"\bfor each (?!of\b)", "for every "),
    ParaphraseRule("allocation_apiece", "equal_allocation", r"\beach(?=\s*[,.;?!])", "apiece"),
    ParaphraseRule(
        "rate_per_time",
        "rate",
        r"\bper (minute|hour|day|week|month|year)\b",
        r"for each \1",
    ),
    ParaphraseRule(
        "allocation_equally_among", "equal_allocation", r"\bequally among\b", "evenly among"
    ),
    ParaphraseRule(
        "allocation_divided_equally", "equal_allocation", r"\bdivided equally\b", "split evenly"
    ),
    ParaphraseRule(
        "allocation_split_equally", "equal_allocation", r"\bsplit equally\b", "divided evenly"
    ),
    ParaphraseRule(
        "allocation_shared_equally", "equal_allocation", r"\bshared equally\b", "distributed evenly"
    ),
    ParaphraseRule("percent_more", "percentage", r"\bpercent more than\b", "percent above"),
    ParaphraseRule("percent_less", "percentage", r"\bpercent less than\b", "percent below"),
    ParaphraseRule("percent_symbol_more", "percentage", r"% more than\b", "% above"),
    ParaphraseRule("percent_symbol_less", "percentage", r"% less than\b", "% below"),
    ParaphraseRule("remaining_the_rest", "remaining_difference", r"\bthe rest\b", "the remainder"),
    ParaphraseRule(
        "remaining_left_over_inverse", "remaining_difference", r"\bleft over\b", "remaining"
    ),
    ParaphraseRule(
        "difference_gap", "remaining_difference", r"\bdifference between\b", "gap between"
    ),
    ParaphraseRule("aggregate_altogether", "aggregation", r"\baltogether\b", "in total"),
    ParaphraseRule("aggregate_in_all", "aggregation", r"\bin all\b", "altogether"),
    ParaphraseRule(
        "aggregate_total_number", "aggregation", r"\btotal number of\b", "combined number of"
    ),
    ParaphraseRule("average_mean", "average", r"\baverage\b", "arithmetic mean"),
)


def _case_aware_replacement(match: re.Match[str], replacement: str) -> str:
    expanded = match.expand(replacement)
    if match.group(0)[:1].isupper():
        return expanded[:1].upper() + expanded[1:]
    return expanded


def relation_paraphrases(question: str) -> list[dict[str, str]]:
    """Return unique one-rule semantic paraphrases in registry order."""

    variants: list[dict[str, str]] = []
    observed = {question}
    for rule in RELATION_PARAPHRASE_RULES:
        pattern = re.compile(rule.pattern, flags=re.IGNORECASE)
        transformed, count = pattern.subn(
            lambda match, replacement=rule.replacement: _case_aware_replacement(match, replacement),
            question,
            count=1,
        )
        if not count or transformed in observed:
            continue
        observed.add(transformed)
        variants.append(
            {
                "question": transformed,
                "rule_id": rule.rule_id,
                "relation_class": rule.relation_class,
            }
        )
    return variants


def _unique_parent_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("dataset")) != "gsm8k":
            raise ValueError("augmentation parent training data must be GSM8K-only")
        source_id = str(row["parent_source_id"])
        previous = parents.setdefault(source_id, row)
        if previous["target"] != row["target"] or previous["prompt"] != row["prompt"]:
            raise ValueError(f"inconsistent repeated parent training row: {source_id}")
    return parents


def build_relation_paraphrase_increment(
    *,
    parent_train_path: str | Path,
    eligible_path: str | Path,
    output_dir: str | Path,
    variants_per_parent: int = 1,
    max_rows: int | None = None,
    seed: int = 73031,
) -> dict[str, Any]:
    """Build an incremental AUG1 dataset from frozen U2000 parent identities."""

    if variants_per_parent < 1:
        raise ValueError("variants_per_parent must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")

    parent_train_path = Path(parent_train_path)
    eligible_path = Path(eligible_path)
    parents = _unique_parent_rows(read_jsonl(parent_train_path))
    eligible_rows = read_jsonl(eligible_path)
    eligible = {str(row["source_id"]): row for row in eligible_rows}
    missing = sorted(set(parents) - set(eligible))
    if missing:
        raise ValueError(f"eligible source is missing {len(missing)} training parents")

    candidates: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for source_id, training_row in sorted(parents.items(), key=lambda item: item[0]):
        source = eligible[source_id]
        if str(source.get("dataset")) != "gsm8k":
            raise ValueError(f"eligible parent is not GSM8K: {source_id}")
        if str(source["target"]) != str(training_row["target"]):
            raise ValueError(f"eligible target differs from frozen training target: {source_id}")
        validation = validate_asl(
            str(source["target"]), effective_scope=dict(source["effective_scope"])
        )
        if not validation["execution_verified"]:
            raise ValueError(f"parent ASL is not executable: {source_id}")

        variants = relation_paraphrases(str(source["question"]))
        if not variants:
            unmatched.append(source_id)
            continue
        ordered = sorted(
            variants,
            key=lambda row: fingerprint(f"{seed}:{source_id}:{row['rule_id']}"),
        )[:variants_per_parent]
        for variant in ordered:
            candidates.append(
                {
                    "source": source,
                    "training": training_row,
                    "variant": variant,
                    "selection_key": fingerprint(f"{seed}:select:{source_id}:{variant['rule_id']}"),
                }
            )

    candidates.sort(key=lambda row: row["selection_key"])
    if max_rows is not None:
        candidates = candidates[:max_rows]
    dataset_id = f"GSM8K_SEM_AUG1_RELATION_V1_N{len(candidates)}"
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        source = candidate["source"]
        training_row = candidate["training"]
        variant = candidate["variant"]
        source_id = str(source["source_id"])
        transformation_id = fingerprint(
            {
                "parent_source_id": source_id,
                "question": variant["question"],
                "rule_id": variant["rule_id"],
                "stage": "relation_paraphrase",
            },
            16,
        )
        example_id = f"gsm8k-sem-aug1-{index:05d}-{transformation_id[:10]}"
        augmentation = {
            "stage": "AUG1",
            "transformation": "relation_paraphrase",
            "transformation_id": transformation_id,
            "rule_id": variant["rule_id"],
            "relation_class": variant["relation_class"],
            "meaning_preserving": True,
            "target_unchanged": True,
            "parent_source_id": source_id,
            "parent_semantic_pattern_id": source["semantic_pattern_id"],
            "parent_question_sha256": fingerprint(source["question"]),
            "transformed_question_sha256": fingerprint(variant["question"]),
        }
        rows.append(
            {
                **training_row,
                "schema_version": "ccpu.paper1.gsm8k_semantic_augmentation_sft.v1",
                "example_id": example_id,
                "parent_example_id": training_row["parent_example_id"],
                "parent_source_id": source_id,
                "dataset_id": dataset_id,
                "epoch_view": 0,
                "prompt": autonomous_asl_prompt(str(variant["question"])),
                "target": source["target"],
                "source_fields_visible_to_model": ["question"],
                "augmentation": augmentation,
            }
        )
        ledger.append(
            {
                **augmentation,
                "example_id": example_id,
                "original_question": source["question"],
                "transformed_question": variant["question"],
                "target_sha256": fingerprint(source["target"]),
                "execution_verified": True,
            }
        )

    rule_counts = Counter(row["augmentation"]["rule_id"] for row in rows)
    relation_counts = Counter(row["augmentation"]["relation_class"] for row in rows)
    output = Path(output_dir)
    increment_path = write_jsonl(output / "increment.jsonl", rows)
    ledger_path = write_jsonl(output / "transformation_ledger.jsonl", ledger)
    unmatched_path = write_jsonl(
        output / "unmatched_parents.jsonl",
        ({"parent_source_id": source_id} for source_id in unmatched),
    )
    manifest = {
        "schema_version": "ccpu.paper1.gsm8k_semantic_augmentation.v1",
        "dataset_id": dataset_id,
        "stage": "AUG1",
        "transformation": "relation_paraphrase",
        "seed": seed,
        "variants_per_parent": variants_per_parent,
        "max_rows": max_rows,
        "counts": {
            "frozen_parent_exposures": len(read_jsonl(parent_train_path)),
            "unique_parent_ids": len(parents),
            "matched_parent_ids": len({str(row["parent_source_id"]) for row in rows}),
            "unmatched_parent_ids": len(unmatched),
            "increment_rows": len(rows),
            "unique_transformed_questions": len(
                {row["augmentation"]["transformed_question_sha256"] for row in rows}
            ),
            "execution_verified": len(rows),
        },
        "rule_counts": dict(sorted(rule_counts.items())),
        "relation_class_counts": dict(sorted(relation_counts.items())),
        "protocol": {
            "offline_only": True,
            "fixed_zero_shot_prompt": True,
            "hidden_answer_visible": False,
            "hidden_rationale_visible": False,
            "problem_dependent_icl": False,
            "target_program_changed": False,
            "split_inherited_from_parent": True,
        },
        "input_sha256": {
            "parent_train": file_sha256(parent_train_path),
            "eligible": file_sha256(eligible_path),
        },
        "output_sha256": {
            "increment": file_sha256(increment_path),
            "transformation_ledger": file_sha256(ledger_path),
            "unmatched_parents": file_sha256(unmatched_path),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
