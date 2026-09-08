from decimal import Decimal

import pytest

from ccpu.common.artifacts import read_jsonl
from ccpu.dsl import validate_asl
from ccpu.dsl.registry import ARITHMETIC_FUNCTIONS
from ccpu.paper1.e3.operator_complexity import freeze_o1_dataset, freeze_o1_pilot


def test_o1_exact_runtime_and_semantic_aliases():
    assert ARITHMETIC_FUNCTIONS["square"]([Decimal(7)]) == 49
    assert ARITHMETIC_FUNCTIONS["cube"]([Decimal(3)]) == 27
    assert ARITHMETIC_FUNCTIONS["sqrt"]([Decimal(81)]) == 9
    assert ARITHMETIC_FUNCTIONS["area_of_square"]([Decimal(7)]) == 49
    with pytest.raises(ValueError, match="exact roots"):
        ARITHMETIC_FUNCTIONS["sqrt"]([Decimal(2)])


def test_o1_ap_and_as_lower_to_same_ccir():
    ap = validate_asl("side = 12\narea = square(side)\nRETURN area")
    semantic = validate_asl("side = 12\narea = area_of_square(side)\nRETURN area")

    assert ap["execution_verified"] and semantic["execution_verified"]
    assert ap["ccir"]["operations"][1]["operation"]["expr"]["op"] == "SQUARE"
    assert semantic["ccir"]["operations"][1]["operation"]["expr"]["op"] == "SQUARE"


def test_o1_generator_freezes_disjoint_matched_splits(tmp_path):
    manifest = freeze_o1_dataset(tmp_path, train_count=12, dev_count=6, test_count=6)

    assert manifest["gold_runtime_ceiling"]["ccir_equivalent"] == 1.0
    families = [set(value) for value in manifest["template_families"].values()]
    assert not (families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
    for split, expected in (("train", 12), ("dev", 6), ("test", 6)):
        rows = read_jsonl(tmp_path / f"{split}.jsonl")
        assert len(rows) == expected
        assert all(row["source_fields_visible_to_model"] == ["question"] for row in rows)
        assert all(row["targets"]["ap_asl"] != row["targets"]["as_asl"] for row in rows)


def test_o1_pilot_is_balanced_and_uses_fixed_representation_prompts(tmp_path):
    full = tmp_path / "full"
    pilot = tmp_path / "pilot"
    freeze_o1_dataset(full, train_count=30, dev_count=15, test_count=15)
    manifest = freeze_o1_pilot(
        full, pilot, train_count=12, dev_count=6, test_count=6
    )

    assert manifest["training"]["exposures_per_adapter"] == 36
    assert manifest["test_operator_counts"] == {"cube": 2, "sqrt": 2, "square": 2}
    ap = read_jsonl(pilot / "ap" / "train.jsonl")
    semantic = read_jsonl(pilot / "as" / "train.jsonl")
    assert {row["parent_example_id"] for row in ap} == {
        row["parent_example_id"] for row in semantic
    }
    assert all("square(x), cube(x), and sqrt(x)" in row["prompt"] for row in ap)
    assert all("area_of_square(side)" in row["prompt"] for row in semantic)
    assert all(
        row["evaluation_representation"] == "AP"
        for row in read_jsonl(pilot / "ap" / "test.jsonl")
    )
    assert [row["source_row"] for row in read_jsonl(pilot / "test.jsonl")] == list(
        range(6)
    )
