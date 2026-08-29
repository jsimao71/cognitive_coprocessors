"""Deterministic oversized candidates for the Paper 1.5 frozen replication."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .dataset import RetrievalExample


@dataclass(frozen=True)
class NextBenchmarkConfig:
    seed: int = 15027
    dev_per_design: int = 8
    test_per_design: int = 48
    target_per_quadrant: int = 20

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NextBenchmarkConfig:
        value = raw.get("next_iteration", raw)
        return cls(
            seed=int(value.get("seed", 15027)),
            dev_per_design=int(value.get("dev_per_design", 8)),
            test_per_design=int(value.get("test_per_design", 48)),
            target_per_quadrant=int(value.get("target_per_quadrant", 20)),
        )


_KNOWN_FACTS = (
    ("France", "capital", "Paris"),
    ("Japan", "capital", "Tokyo"),
    ("Germany", "capital", "Berlin"),
    ("Italy", "capital", "Rome"),
    ("Spain", "capital", "Madrid"),
    ("Portugal", "capital", "Lisbon"),
    ("Canada", "capital", "Ottawa"),
    ("Australia", "capital", "Canberra"),
    ("Brazil", "capital", "Brasilia"),
    ("Egypt", "capital", "Cairo"),
    ("India", "capital", "New Delhi"),
    ("Greece", "capital", "Athens"),
    ("Norway", "capital", "Oslo"),
    ("Sweden", "capital", "Stockholm"),
    ("Finland", "capital", "Helsinki"),
    ("Ireland", "capital", "Dublin"),
    ("Austria", "capital", "Vienna"),
    ("Belgium", "capital", "Brussels"),
    ("Poland", "capital", "Warsaw"),
    ("Mexico", "capital", "Mexico City"),
    ("Argentina", "capital", "Buenos Aires"),
    ("Chile", "capital", "Santiago"),
    ("Thailand", "capital", "Bangkok"),
    ("Kenya", "capital", "Nairobi"),
    ("gold", "chemical symbol", "Au"),
    ("silver", "chemical symbol", "Ag"),
    ("oxygen", "chemical symbol", "O"),
    ("water", "chemical formula", "H2O"),
)

_LOW_SUBCLASSES = (
    "stable_familiar",
    "answer_in_context",
    "freshness_distractor",
    "quotation",
    "hypothetical",
    "non_factual_prose",
    "computational_need",
)
_HIGH_SUBCLASSES = (
    "fresh_current",
    "source_version_specific",
    "private_store",
    "changed_familiar",
    "unavailable_evidence",
    "conflicting_evidence",
    "exact_source_attribution",
    "structured_source_value",
)


def _code(rng: random.Random, prefix: str, index: int) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{prefix}-{''.join(rng.choice(letters) for _ in range(4))}-{index:03d}"


def build_next_candidates(
    config: NextBenchmarkConfig,
) -> tuple[dict[str, Any], list[RetrievalExample], dict[str, Any]]:
    """Build candidates designed to populate all measured confidence/risk quadrants."""

    rng = random.Random(config.seed)
    records: list[dict[str, Any]] = []
    examples: list[RetrievalExample] = []
    split_counts = {"dev": config.dev_per_design, "test": config.test_per_design}
    fact_cursor = {"dev": 0, "test": 8}
    record_index = 0

    def add_record(**row: Any) -> None:
        nonlocal record_index
        record_index += 1
        records.append({"record_id": f"next-r{record_index:04d}", **row})

    for split, count in split_counts.items():
        for design in (
            "likely_high_not_required",
            "likely_low_not_required",
            "likely_high_required",
            "likely_low_required",
        ):
            for index in range(count):
                suffix = f"{split}-{design}-{index:03d}"
                if design == "likely_high_not_required":
                    entity, attribute, answer = _KNOWN_FACTS[
                        (fact_cursor[split] + index) % len(_KNOWN_FACTS)
                    ]
                    question = f"Return only the well-known {attribute} of {entity}."
                    category = _LOW_SUBCLASSES[index % len(_LOW_SUBCLASSES)]
                    add_record(
                        entity=f"standard::{entity}",
                        attribute=attribute,
                        value=answer,
                        valid_from="1900-01-01",
                    )
                    example_entity = f"standard::{entity}"
                    evidence_required = False
                elif design == "likely_low_not_required":
                    answer = _code(rng, "CTX", index + (0 if split == "dev" else 100))
                    category = _LOW_SUBCLASSES[index % len(_LOW_SUBCLASSES)]
                    if category == "quotation":
                        question = f"Quote exactly the supplied label '{answer}' and nothing else."
                    elif category == "freshness_distractor":
                        question = f"The word 'latest' is decorative. Repeat the supplied code {answer}."
                    elif category == "computational_need":
                        question = f"A calculator, not retrieval, produced token {answer}. Repeat it."
                    else:
                        question = f"The active context explicitly supplies value {answer}. Return it only."
                    attribute = "supplied value"
                    example_entity = f"context::{suffix}"
                    evidence_required = False
                elif design == "likely_high_required":
                    entity, attribute, familiar = _KNOWN_FACTS[
                        (fact_cursor[split] + index) % len(_KNOWN_FACTS)
                    ]
                    answer = _code(rng, "REV", index + (0 if split == "dev" else 100))
                    category = _HIGH_SUBCLASSES[index % len(_HIGH_SUBCLASSES)]
                    example_entity = f"atlas::{split}::{entity}::{index:03d}"
                    add_record(
                        entity=example_entity,
                        attribute=attribute,
                        value=familiar,
                        valid_from="1900-01-01",
                        valid_to="2025-12-31",
                    )
                    add_record(
                        entity=example_entity,
                        attribute=attribute,
                        value=answer,
                        valid_from="2026-01-01",
                    )
                    question = (
                        f"According to Atlas source version 2026, what is the currently designated "
                        f"{attribute} for {entity}? Return only the value."
                    )
                    evidence_required = True
                else:
                    category = _HIGH_SUBCLASSES[index % len(_HIGH_SUBCLASSES)]
                    entity = f"Private {split.title()} Entity {index:03d}"
                    attribute = "assigned code"
                    example_entity = f"atlas::{entity}"
                    answer = _code(rng, "SRC", index + (0 if split == "dev" else 100))
                    question = (
                        f"Using the controlled Atlas source as of 2026, give the latest assigned "
                        f"code for {entity}. Return only the value."
                    )
                    if category == "unavailable_evidence":
                        answer = "ABSTAIN"
                        add_record(
                            entity=example_entity,
                            attribute=attribute,
                            value=_code(rng, "HIDDEN", index),
                            valid_from="2026-01-01",
                            available=False,
                        )
                    elif category == "conflicting_evidence":
                        answer = "ABSTAIN"
                        add_record(
                            entity=example_entity,
                            attribute=attribute,
                            value=_code(rng, "LEFT", index),
                            valid_from="2026-01-01",
                        )
                        add_record(
                            entity=example_entity,
                            attribute=attribute,
                            value=_code(rng, "RIGHT", index),
                            valid_from="2026-01-01",
                        )
                    else:
                        add_record(
                            entity=example_entity,
                            attribute=attribute,
                            value=answer,
                            valid_from="2026-01-01",
                        )
                    evidence_required = True

                examples.append(
                    RetrievalExample(
                        example_id=f"p15-next-{suffix}",
                        split=split,
                        question=question,
                        answer=answer,
                        entity=example_entity,
                        attribute=attribute,
                        as_of="2026-08-28",
                        evidence_required=evidence_required,
                        category=category,
                        retrieval_subclass=category,
                        design_group=design,
                    )
                )

    source = {
        "source_id": "atlas-controlled-registry",
        "version": "2026.08.2-frozen-candidates",
        "records": records,
    }
    audit = {
        "schema_version": "ccpu.paper1_5.candidate_audit.v1",
        "seed": config.seed,
        "candidate_count": len(examples),
        "record_count": len(records),
        "dev_count": sum(example.split == "dev" for example in examples),
        "test_count": sum(example.split == "test" for example in examples),
        "retrieval_required_subclasses": list(_HIGH_SUBCLASSES),
        "retrieval_not_required_subclasses": list(_LOW_SUBCLASSES),
        "unique_record_ids": len({row["record_id"] for row in records}) == len(records),
    }
    return source, examples, audit


def select_measured_quadrants(
    examples: list[RetrievalExample],
    minimum_probabilities: dict[str, float],
    *,
    threshold: float,
    target_per_quadrant: int,
) -> tuple[list[RetrievalExample], dict[str, Any]]:
    groups: dict[str, list[RetrievalExample]] = {}
    for example in examples:
        if example.split != "test":
            continue
        low = minimum_probabilities[example.example_id] < threshold
        quadrant = (
            f"{'low' if low else 'high'}_confidence__"
            f"{'retrieval_required' if example.evidence_required else 'retrieval_not_required'}"
        )
        groups.setdefault(quadrant, []).append(example)
    counts = {name: len(rows) for name, rows in sorted(groups.items())}
    required = (
        "high_confidence__retrieval_not_required",
        "low_confidence__retrieval_not_required",
        "high_confidence__retrieval_required",
        "low_confidence__retrieval_required",
    )
    shortages = {name: counts.get(name, 0) for name in required if counts.get(name, 0) < target_per_quadrant}
    if shortages:
        raise ValueError(f"insufficient measured quadrant candidates: {shortages}; all={counts}")
    selected_test = [
        example
        for name in required
        for example in sorted(groups[name], key=lambda item: item.example_id)[:target_per_quadrant]
    ]
    selected = [example for example in examples if example.split == "dev"] + selected_test
    audit = {
        "schema_version": "ccpu.paper1_5.quadrant_freeze.v1",
        "threshold": threshold,
        "target_per_quadrant": target_per_quadrant,
        "candidate_counts": counts,
        "selected_counts": {name: target_per_quadrant for name in required},
        "selected_test_ids": [example.example_id for example in selected_test],
    }
    return selected, audit
