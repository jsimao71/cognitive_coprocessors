from decimal import Decimal

import pytest

from ccpu.common.artifacts import read_json, read_jsonl, write_json, write_jsonl
from ccpu.paper1.f3 import parse_f3_program, validate_f3_program
from ccpu.paper1.f3.data import (
    f3_prompt,
    prepare_f3_annotation_batches,
    validate_f3_annotations,
)
from ccpu.paper1.f3.normalize import semantic_signature

SCOPE = {"id": "test:f3:1", "kind": "benchmark_case", "parent": None}


def returned(result: dict) -> Decimal:
    value = result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"]
    return Decimal(str(value))


def test_hats_event_history_and_intent_query() -> None:
    question = (
        "In a truck, there are 26 pink, 15 green, and 24 yellow hats. "
        "Carl removes 4 pink hats. John removes 6 pink hats and twice that many green hats. "
        "How many remain?"
    )
    program = """
collection("truck.inventory.hats", "hat", "truck", source("26 pink, 15 green, and 24 yellow hats"))
observe(at("truck.inventory.hats.pink.count", "initial"), 26, "count", source("26 pink"))
observe(at("truck.inventory.hats.green.count", "initial"), 15, "count", source("15 green"))
observe(at("truck.inventory.hats.yellow.count", "initial"), 24, "count", source("24 yellow hats"))
remove("e1", "carl", at("truck.inventory.hats.pink.count", "current"), 4, source("Carl removes 4 pink hats"))
remove("e2", "john", at("truck.inventory.hats.pink.count", "current"), 6, source("John removes 6 pink hats"))
remove("e3", "john", at("truck.inventory.hats.green.count", "current"), scale(event_field("e2", "quantity"), 2), source("twice that many green hats"))
query("remaining_count", "truck.inventory.hats", "current")
""".strip()
    result = validate_f3_program(
        program, question=question, source_context=None, effective_scope=SCOPE, mode="r1"
    )
    assert result["evidence_valid"]
    assert result["executable"], result["errors"]
    assert returned(result) == 43
    assert len(result["runtime_edges"]) == 3


def test_tatqa_cell_grounding_and_percentage_query() -> None:
    context = {
        "paragraphs": [],
        "table": [
            ["", "2019", "2018"],
            ["Acquisition and integration costs", "$17", "$8"],
            ["Total", "$175", "$45"],
        ],
    }
    program = """
observe(at("company.acquisition_integration_cost", "2019"), 17, "million_usd", cell("Acquisition and integration costs", "2019"))
observe(at("company.acquisition_integration_cost", "2018"), 8, "million_usd", cell("Acquisition and integration costs", "2018"))
query("percentage_change", at("company.acquisition_integration_cost", "2018"), at("company.acquisition_integration_cost", "2019"))
""".strip()
    result = validate_f3_program(
        program,
        question="What was the percentage change in 2019 from 2018?",
        source_context=context,
        effective_scope=SCOPE,
        mode="r1",
    )
    assert result["executable"], result["errors"]
    assert returned(result) == Decimal("112.5")


def test_age_requires_constraint_closure() -> None:
    question = (
        "Jessica is six years older than Claire. "
        "In two years, Claire will be 20 years old. How old is Jessica now?"
    )
    program = """
older_than(at("jessica.age", "now"), at("claire.age", "now"), 6, "year", source("Jessica is six years older than Claire"))
observe(at("claire.age", "plus_2_year"), 20, "year", source("In two years, Claire will be 20 years old"))
query("value", at("jessica.age", "now"))
""".strip()
    r1 = validate_f3_program(
        program, question=question, source_context=None, effective_scope=SCOPE, mode="r1"
    )
    r2 = validate_f3_program(
        program, question=question, source_context=None, effective_scope=SCOPE, mode="r2"
    )
    assert not r1["executable"]
    assert r2["executable"], r2["errors"]
    assert returned(r2) == 24


def test_unknown_event_quantity_is_solved_from_final_state() -> None:
    question = (
        "A trader has 55 bags of rice in stock. She sells off some bags of rice and "
        "restocks 132 bags of rice. How many bags did she sell if she now has 164 bags?"
    )
    program = """
observe(at("trader.rice.count", "initial"), 55, "count", source("55 bags of rice in stock"))
remove("sale", "trader", at("trader.rice.count", "current"), "trader.rice.sold.count", source("sells off some bags of rice"))
add("restock", "trader", at("trader.rice.count", "current"), 132, source("restocks 132 bags of rice"))
observe(at("trader.rice.count", "now"), 164, "count", source("if she now has 164 bags"))
query("value", event_field("sale", "quantity"))
""".strip()
    result = validate_f3_program(
        program, question=question, source_context=None, effective_scope=SCOPE, mode="r2"
    )
    assert result["executable"], result["errors"]
    assert returned(result) == 23


def test_event_quantity_can_feed_a_relation() -> None:
    question = "There are 10 potatoes. She cuts 2 potatoes into 8 wedges each."
    program = """
observe(at("potatoes.count", "initial"), 10, "count", source("10 potatoes"))
remove("cut", "she", at("potatoes.count", "current"), 2, source("cuts 2 potatoes"))
product_relation(at("wedges.count", "current"), event_field("cut", "quantity"), 8, source("8 wedges each"))
query("value", at("wedges.count", "current"))
""".strip()
    result = validate_f3_program(
        program, question=question, source_context=None, effective_scope=SCOPE, mode="r2"
    )
    assert result["executable"], result["errors"]
    assert returned(result) == 16


def test_collection_sum_resolves_custom_times_and_declared_members() -> None:
    question = "Payments of 123 and 344 were made for two acquisitions in 2018."
    program = """
collection("company.acquisitions", "payment", "company", source("two acquisitions"))
observe(at("company.acquisitions.first.amount", "august_2018"), 123, "million", source("123"))
observe(at("company.acquisitions.second.amount", "october_2018"), 344, "million", source("344"))
member("company.acquisitions", "company.acquisitions.first.amount", source("two acquisitions"))
member("company.acquisitions", "company.acquisitions.second.amount", source("two acquisitions"))
query("sum", "company.acquisitions", "2018")
""".strip()
    result = validate_f3_program(
        program, question=question, source_context=None, effective_scope=SCOPE, mode="r2"
    )
    assert result["executable"], result["errors"]
    assert returned(result) == 467


def test_mean_query_represents_requested_aggregation_intent() -> None:
    question = "What is the average of the values 98, 183, and 377?"
    program = """
observe(at("shares.nonvested", "2017"), 98, "count", source("98"))
observe(at("shares.nonvested", "2018"), 183, "count", source("183"))
observe(at("shares.nonvested", "2019"), 377, "count", source("377"))
query("mean", at("shares.nonvested", "2017"), at("shares.nonvested", "2018"), at("shares.nonvested", "2019"))
""".strip()
    result = validate_f3_program(
        program, question=question, source_context=None, effective_scope=SCOPE, mode="r2"
    )
    assert result["executable"], result["errors"]
    assert returned(result) == Decimal("219.3333333333333333333333333")


def test_evidence_must_match_supplied_source() -> None:
    program = """
observe(at("apples.count", "initial"), 5, "count", source("six apples"))
query("value", at("apples.count", "current"))
""".strip()
    result = validate_f3_program(
        program,
        question="There are five apples.",
        source_context=None,
        effective_scope=SCOPE,
    )
    assert result["parse_valid"]
    assert not result["evidence_valid"]
    assert not result["lowerable"]


def test_parser_rejects_python_and_requires_terminal_query() -> None:
    with pytest.raises(ValueError, match="unsupported top-level"):
        parse_f3_program('__import__("os")\nquery("value", at("x", "now"))')
    with pytest.raises(ValueError, match="exactly one query"):
        parse_f3_program('observe(at("x", "now"), 1, "count", source("one"))')


def test_semantic_signature_abstracts_labels_and_constants() -> None:
    left = parse_f3_program(
        'observe(at("a.count", "initial"), 5, "count", source("five"))\n'
        'query("value", at("a.count", "current"))'
    )
    right = parse_f3_program(
        'observe(at("b.count", "initial"), 900, "items", source("nine hundred"))\n'
        'query("value", at("b.count", "current"))'
    )
    assert semantic_signature(left) == semantic_signature(right)


def test_student_prompt_hides_supervision() -> None:
    prompt = f3_prompt(
        {
            "question": "There are five apples. How many apples are there?",
            "answer": "5",
            "rationale": "The answer is five.",
            "asl_program": "apples.count = 5\nRETURN apples.count",
        }
    )
    assert "There are five apples" in prompt
    assert "The answer is five" not in prompt
    assert "apples.count = 5" not in prompt


def test_train_only_pilot_records_selected_identities(tmp_path) -> None:
    freeze = tmp_path / "freeze"

    def row(split: str, index: int) -> dict:
        return {
            "dataset": "gsm8k",
            "source_id": f"{split}-{index}",
            "question": f"Question {split} {index}",
            "answer": index,
            "rationale": f"Hidden rationale {index}",
        }

    write_jsonl(freeze / "splits" / "train.jsonl", [row("train", i) for i in range(100)])
    write_jsonl(freeze / "splits" / "dev.jsonl", [row("dev", i) for i in range(25)])
    write_jsonl(freeze / "splits" / "test.jsonl", [row("test", i) for i in range(25)])
    write_json(freeze / "freeze_manifest.json", {"frozen": True})
    expansion = write_jsonl(tmp_path / "expansion.jsonl", [row("expansion", i) for i in range(350)])

    manifest = prepare_f3_annotation_batches(
        freeze,
        expansion,
        tmp_path / "pilot",
        batch_size=5,
        max_train_examples=7,
    )
    batch = read_json(tmp_path / "pilot" / "requests" / "batch_000.json")
    serialized = str(batch)
    assert manifest["example_count"] == 7
    assert manifest["batch_count"] == 2
    assert manifest["selection"] == {"kind": "train_prefix_pilot", "max_examples": 7}
    assert len(manifest["selected_identities"]) == 7
    assert all(item["source_id"].startswith("train-") for item in manifest["selected_identities"])
    assert "Hidden rationale" not in serialized
    assert "answer" not in batch["items"][0]


def test_partial_validation_separates_unattempted_sources(tmp_path) -> None:
    freeze = tmp_path / "freeze"

    def row(split: str, index: int) -> dict:
        scope_id = f"scope:{split}:{index}"
        return {
            "dataset": "gsm8k",
            "source_id": f"{split}-{index}",
            "question": "There are five apples.",
            "effective_scope": {"id": scope_id, "kind": "benchmark_case", "parent": None},
            "state_after": {scope_id: {"returned": 5}},
            "asl": "apples.count = 5\nRETURN apples.count",
            "semantic_pattern_id": "observe-value",
            "record_sha256": f"sha-{split}-{index}",
        }

    train_rows = [row("train", i) for i in range(100)]
    write_jsonl(freeze / "splits" / "train.jsonl", train_rows)
    write_jsonl(freeze / "splits" / "dev.jsonl", [row("dev", i) for i in range(25)])
    write_jsonl(freeze / "splits" / "test.jsonl", [row("test", i) for i in range(25)])
    write_json(freeze / "freeze_manifest.json", {"frozen": True})
    expansion = write_jsonl(tmp_path / "expansion.jsonl", [row("expansion", i) for i in range(350)])
    annotations = write_jsonl(
        tmp_path / "annotations.jsonl",
        [
            {
                "dataset": "gsm8k",
                "source_id": "train-0",
                "f3_status": "ok",
                "f3_program": [
                    'observe(at("apples.count", "initial"), 5, "count", source("five apples"))',
                    'query("value", at("apples.count", "current"))',
                ],
            }
        ],
    )
    summary = validate_f3_annotations(
        freeze,
        expansion,
        [annotations],
        tmp_path / "validated",
    )
    assert summary["attempted_count"] == 1
    assert summary["accepted_count"] == 1
    assert summary["acceptance_rate"] == 1.0
    assert summary["unattempted_count"] == 499
    assert read_jsonl(tmp_path / "validated" / "rejected.jsonl") == []
