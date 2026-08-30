from ccpu.dsl import validate_asl
from ccpu.paper1.asl_pilot_data import (
    _choose_grouped_splits,
    pattern_id,
    perturb_row,
)
from ccpu.paper1.asl_pilot_eval import extract_asl, score_asl


def _scope(source_id="one"):
    return {
        "id": f"gsm8k:train:{source_id}",
        "parent": None,
        "kind": "benchmark_case",
        "source": "dataset",
    }


def _row(source_id: str, asl: str, question: str):
    validation = validate_asl(asl, effective_scope=_scope(source_id))
    assert validation["execution_verified"]
    return {
        "dataset": "gsm8k",
        "source_id": source_id,
        "record_sha256": source_id.rjust(64, "0"),
        "question": question,
        "asl": asl,
        "ccir": validation["ccir"],
        "part_mappings": [{"asl": asl.splitlines()}],
        "provenance": {"repair_round": 0},
        "effective_scope": _scope(source_id),
    }


def test_pattern_signature_ignores_names_and_literal_values():
    first = _row(
        "1",
        "claire.age_now = 18\njessica.age_now = claire.age_now + 6\nRETURN jessica.age_now",
        "Claire is 18. Jessica is six years older. How old is Jessica?",
    )
    second = _row(
        "2",
        "helen.age_now = 900\nmaria.age_now = helen.age_now + 77\nRETURN maria.age_now",
        "Helen is 900. Maria is 77 years older. How old is Maria?",
    )
    assert pattern_id(first) == pattern_id(second)


def test_grouped_split_keeps_exact_counts_without_group_overlap():
    groups = {
        f"p{index}": [
            {
                "dataset": "tatqa" if index % 3 == 0 else "gsm8k",
                "provenance": {"repair_round": 1 if index % 4 == 0 else 0},
            }
        ]
        for index in range(150)
    }
    splits = _choose_grouped_splits(groups, seed=7)
    assert [len(splits[name]) for name in ("train", "dev", "test")] == [100, 25, 25]
    assert not splits["train"] & splits["dev"]
    assert not splits["train"] & splits["test"]
    assert not splits["dev"] & splits["test"]


def test_numeric_perturbation_updates_nl_and_asl_and_remains_executable():
    row = _row(
        "3",
        "claire.age_now = 18\njessica.age_now = claire.age_now + 6\nRETURN jessica.age_now",
        "Claire is 18 years old. Jessica is 6 years older. How old is Jessica?",
    )
    transformed = perturb_row(row, variant=11, mode="large")
    assert transformed["perturbation"]["numeric_replacements"]
    assert transformed["question"] != row["question"]
    assert transformed["asl"] != row["asl"]
    assert validate_asl(transformed["asl"], effective_scope=row["effective_scope"])[
        "execution_verified"
    ]


def test_large_perturbation_preserves_percentage_parameters():
    row = _row(
        "pct",
        "item.price = 250\nitem.discount_pct = 20\n"
        "item.sale = dec_pct(item.price, item.discount_pct)\nRETURN item.sale",
        "A store discounts a 250 euro item by 20 percent.",
    )
    transformed = perturb_row(row, variant=1, mode="large")
    assert "discount_pct = 20" in transformed["asl"]
    assert "20 percent" in transformed["question"]
    assert transformed["perturbation"]["numeric_replacements"]["250"] != "250"


def test_semantic_scoring_accepts_equivalent_path_names():
    reference = """claire.age_in_2y = 20
claire.age_now = claire.age_in_2y - 2
jessica.age_now = claire.age_now + 6
RETURN jessica.age_now"""
    predicted = """claire.future_age = 20
claire.age_now = claire.future_age - 2
jessica.age_now = claire.age_now + 6
RETURN jessica.age_now"""
    metrics = score_asl(reference, predicted, _scope())
    assert metrics["exact_asl"] is False
    assert metrics["parse_valid"] is True
    assert metrics["executable"] is True
    assert metrics["semantic_return_equivalent"] is True
    assert metrics["semantic_state_equivalent"] is True
    assert metrics["final_answer_correct"] is True


def test_extract_asl_ignores_fenced_explanation():
    output = "Here is the program:\n```asl\na.count = 4\nRETURN a.count\n```\nDone."
    assert extract_asl(output) == "a.count = 4\nRETURN a.count"
