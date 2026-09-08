from decimal import Decimal

import pytest

from ccpu.common.artifacts import read_jsonl
from ccpu.dsl import validate_asl
from ccpu.dsl.registry import ARITHMETIC_FUNCTIONS
from ccpu.paper1.e3.operator_complexity import freeze_o1_dataset


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
