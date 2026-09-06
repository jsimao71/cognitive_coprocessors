from ccpu.common.artifacts import read_jsonl, write_jsonl
from ccpu.paper1.e3.semantic_augmentation import (
    build_entity_rename_increment,
    build_relation_paraphrase_increment,
    entity_rename_variant,
    freeze_augmentation_selection_gate,
    relation_paraphrases,
)


def _eligible(source_id: str = "1") -> dict:
    return {
        "annotation_sha256": "a" * 64,
        "dataset": "gsm8k",
        "effective_scope": {
            "id": f"gsm8k:train:{source_id}",
            "kind": "benchmark_case",
            "parent": None,
            "source": "dataset",
        },
        "question": "John has twice as many green hats as Carl. How many hats does John have?",
        "recovery": None,
        "semantic_pattern_id": "asl-pattern-test",
        "source_id": source_id,
        "source_record_sha256": "b" * 64,
        "target": "carl.green_hats = 4\njohn.green_hats = carl.green_hats * 2\nRETURN john.green_hats",
        "teacher": {"provider": "test"},
    }


def _training(source_id: str = "1", exposure: int = 0) -> dict:
    source = _eligible(source_id)
    return {
        "dataset": "gsm8k",
        "dataset_id": "G1_GSM8K_U1_E2",
        "epoch_view": exposure,
        "example_id": f"parent-{exposure}",
        "parent_example_id": f"gsm8k:{source_id}",
        "parent_source_id": source_id,
        "prompt": f"Problem: {source['question']}\nASL:",
        "semantic_pattern_id": source["semantic_pattern_id"],
        "target": source["target"],
    }


def test_relation_paraphrases_are_one_rule_unique_variants():
    variants = relation_paraphrases(
        "Each box has twice as many red balls as blue balls, with half the rest set aside."
    )
    questions = [row["question"] for row in variants]

    assert len(questions) == len(set(questions))
    assert any("Every box" in question for question in questions)
    assert any("two times as many" in question for question in questions)
    assert any("one-half the" in question for question in questions)


def test_relation_increment_deduplicates_exposures_and_preserves_target(tmp_path):
    parent = write_jsonl(
        tmp_path / "parent.jsonl",
        [_training(exposure=0), _training(exposure=1)],
    )
    eligible = write_jsonl(tmp_path / "eligible.jsonl", [_eligible()])

    manifest = build_relation_paraphrase_increment(
        parent_train_path=parent,
        eligible_path=eligible,
        output_dir=tmp_path / "output",
        seed=7,
    )
    rows = read_jsonl(tmp_path / "output" / "increment.jsonl")
    ledger = read_jsonl(tmp_path / "output" / "transformation_ledger.jsonl")

    assert manifest["counts"]["frozen_parent_exposures"] == 2
    assert manifest["counts"]["unique_parent_ids"] == 1
    assert manifest["counts"]["increment_rows"] == 1
    assert manifest["counts"]["execution_verified"] == 1
    assert rows[0]["target"] == _eligible()["target"]
    assert rows[0]["prompt"] != _training()["prompt"]
    assert rows[0]["source_fields_visible_to_model"] == ["question"]
    assert rows[0]["augmentation"]["target_unchanged"] is True
    assert ledger[0]["execution_verified"] is True
    assert manifest["protocol"]["problem_dependent_icl"] is False


def test_relation_increment_is_deterministic(tmp_path):
    parent = write_jsonl(tmp_path / "parent.jsonl", [_training()])
    eligible = write_jsonl(tmp_path / "eligible.jsonl", [_eligible()])

    first = build_relation_paraphrase_increment(
        parent_train_path=parent,
        eligible_path=eligible,
        output_dir=tmp_path / "first",
        seed=13,
    )
    second = build_relation_paraphrase_increment(
        parent_train_path=parent,
        eligible_path=eligible,
        output_dir=tmp_path / "second",
        seed=13,
    )

    assert first["output_sha256"] == second["output_sha256"]


def test_entity_rename_is_synchronized_reversible_and_excludes_calendar_roots():
    variant = entity_rename_variant(
        "Yesterday, John had twice as many hats as Carl.",
        "yesterday.day = 1\n"
        "carl.hats = 4\n"
        "john.hats = carl.hats * 2\n"
        "RETURN john.hats",
        source_id="42",
        seed=7,
    )

    assert variant is not None
    assert set(variant["entity_mapping"]) == {"carl", "john"}
    assert "Yesterday" in variant["question"]
    assert "yesterday.day" in variant["target"]
    assert "john.hats" not in variant["target"]
    assert "carl.hats" not in variant["target"]

    nested = entity_rename_variant(
        "Teresa is older than Michiko.",
        "teresa.age_at_michiko_birth = 26\nRETURN teresa.age_at_michiko_birth",
        source_id="44",
        seed=7,
    )
    assert nested is not None
    assert "michiko" not in nested["target"]

    unsafe = entity_rename_variant(
        "The Tampa Bay Bucs have cards for Valentine's day.",
        "bucs.players = 13\nvalentine.cards = 20\nRETURN bucs.players",
        source_id="43",
        seed=7,
    )
    assert unsafe is None


def test_entity_increment_preserves_execution_and_changes_paths(tmp_path):
    parent = write_jsonl(tmp_path / "parent.jsonl", [_training()])
    eligible = write_jsonl(tmp_path / "eligible.jsonl", [_eligible()])

    manifest = build_entity_rename_increment(
        parent_train_path=parent,
        eligible_path=eligible,
        output_dir=tmp_path / "entity",
        seed=11,
    )
    rows = read_jsonl(tmp_path / "entity" / "increment.jsonl")

    assert manifest["counts"]["increment_rows"] == 1
    assert manifest["counts"]["renamed_entities"] == 2
    assert rows[0]["target"] != _eligible()["target"]
    assert rows[0]["augmentation"]["target_alpha_equivalent"] is True
    assert manifest["protocol"]["execution_return_preserved"] is True


def test_augmentation_gate_is_stratified_and_disjoint(tmp_path):
    full = write_jsonl(
        tmp_path / "full.jsonl",
        [
            {
                "example_id": f"case-{index}",
                "source_row": index,
                "difficulty_stratum": ("low", "medium", "high")[index % 3],
            }
            for index in range(12)
        ],
    )
    excluded = write_jsonl(
        tmp_path / "excluded.jsonl",
        [
            {
                "example_id": f"case-{index}",
                "source_row": index,
                "difficulty_stratum": ("low", "medium", "high")[index % 3],
            }
            for index in range(3)
        ],
    )

    manifest = freeze_augmentation_selection_gate(
        full_eval_path=full,
        excluded_eval_path=excluded,
        output_dir=tmp_path / "selection",
        count=6,
        seed=19,
    )
    selected = read_jsonl(tmp_path / "selection" / "gate.jsonl")

    assert manifest["counts"]["selected"] == 6
    assert manifest["overlap_with_final_confirmation"] == []
    assert {row["example_id"] for row in selected}.isdisjoint(
        {"case-0", "case-1", "case-2"}
    )
    assert manifest["counts"]["selected_by_difficulty"] == {
        "high": 2,
        "low": 2,
        "medium": 2,
    }
