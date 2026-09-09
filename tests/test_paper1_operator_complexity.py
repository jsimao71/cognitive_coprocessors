from decimal import Decimal

import pytest

from ccpu.common.artifacts import read_jsonl, write_jsonl
from ccpu.dsl import validate_asl
from ccpu.dsl.registry import ARITHMETIC_FUNCTIONS
from ccpu.paper1.e3.operator_complexity import freeze_o1_dataset, freeze_o1_pilot
from ccpu.paper1.e3.operator_analysis import analyze_operator_pilot
from ccpu.paper1.e3.operator_levels import freeze_operator_level, freeze_operator_pilot


def test_o1_exact_runtime_and_semantic_aliases():
    assert ARITHMETIC_FUNCTIONS["square"]([Decimal(7)]) == 49
    assert ARITHMETIC_FUNCTIONS["cube"]([Decimal(3)]) == 27
    assert ARITHMETIC_FUNCTIONS["sqrt"]([Decimal(81)]) == 9
    assert ARITHMETIC_FUNCTIONS["area_of_square"]([Decimal(7)]) == 49
    with pytest.raises(ValueError, match="exact roots"):
        ARITHMETIC_FUNCTIONS["sqrt"]([Decimal(2)])


def test_o0_and_o3_runtime_aliases_are_deterministic():
    assert ARITHMETIC_FUNCTIONS["total_of"]([Decimal(9), Decimal(4)]) == 13
    assert ARITHMETIC_FUNCTIONS["difference"]([Decimal(9), Decimal(4)]) == 5
    assert ARITHMETIC_FUNCTIONS["product"]([Decimal(9), Decimal(4)]) == 36
    assert ARITHMETIC_FUNCTIONS["quotient"]([Decimal(36), Decimal(4)]) == 9
    assert ARITHMETIC_FUNCTIONS["sin_degrees"]([Decimal(30)]) == Decimal("0.5")
    assert ARITHMETIC_FUNCTIONS["horizontal_component_of_unit"](
        [Decimal(60)]
    ) == Decimal("0.5")
    assert ARITHMETIC_FUNCTIONS["decimal_order"]([Decimal(100000)]) == 5
    with pytest.raises(ValueError, match="common angles"):
        ARITHMETIC_FUNCTIONS["sin_degrees"]([Decimal(12)])
    assert ARITHMETIC_FUNCTIONS["solve_linear"](
        [Decimal(3), Decimal(4), Decimal(25)]
    ) == 7
    assert ARITHMETIC_FUNCTIONS["positive_quadratic_root"](
        [Decimal(1), Decimal(-3), Decimal(-10)]
    ) == 5
    assert ARITHMETIC_FUNCTIONS["larger_solution_of_quadratic"](
        [Decimal(1), Decimal(-9), Decimal(20)]
    ) == 5


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


@pytest.mark.parametrize(
    ("level", "operator_count"), (("O0", 4), ("O3", 3), ("O5", 1), ("O6", 2))
)
def test_additional_operator_levels_freeze_matched_pilots(
    tmp_path, level, operator_count
):
    full = tmp_path / level / "full"
    pilot = tmp_path / level / "pilot"
    manifest = freeze_operator_level(
        full, level=level, train_count=40, dev_count=20, test_count=20
    )
    pilot_manifest = freeze_operator_pilot(
        full,
        pilot,
        level=level,
        train_count=20,
        dev_count=operator_count * 2,
        test_count=operator_count * 2,
    )

    assert manifest["operator_level"] == level
    assert len(pilot_manifest["test_operator_counts"]) == operator_count
    ap = read_jsonl(pilot / "ap" / "test.jsonl")
    semantic = read_jsonl(pilot / "as" / "test.jsonl")
    assert [row["example_id"] for row in ap] == [row["example_id"] for row in semantic]
    assert all(row["evaluation_representation"] == "AP" for row in ap)
    assert all(row["evaluation_representation"] == "AS" for row in semantic)


def test_operator_pilot_analysis_is_paired_and_audits_bindings(tmp_path):
    evaluation = tmp_path / "eval.jsonl"
    conditions = {name: tmp_path / f"{name}.jsonl" for name in ("direct", "ap", "as")}
    rows = [
        {"example_id": "a", "operator_family": "square"},
        {"example_id": "b", "operator_family": "cube"},
    ]
    write_jsonl(evaluation, rows)
    for name, path in conditions.items():
        predictions = []
        for index, row in enumerate(rows):
            correct = name == "ap" or (name == "direct" and index == 0)
            errors = ["unresolved reference 'edge'"] if name == "as" else []
            predictions.append(
                {
                    "example_id": row["example_id"],
                    "generated_tokens": 10 + index,
                    "metrics": {
                        "final_answer_correct": correct,
                        "parse_valid": True,
                        "lowerable_to_ccir": True,
                        "type_valid": True,
                        "executable": correct,
                        "errors": errors,
                    },
                }
            )
        write_jsonl(path, predictions)

    report = analyze_operator_pilot(
        eval_path=evaluation,
        direct_predictions_path=conditions["direct"],
        ap_predictions_path=conditions["ap"],
        as_predictions_path=conditions["as"],
        output_dir=tmp_path / "analysis",
    )

    assert report["conditions"]["ap"]["correct"] == 2
    assert report["paired"]["ap_vs_direct"]["left_only"] == 1
    assert report["as_failure_audit"]["unresolved_reference"] == 2
    assert (tmp_path / "analysis" / "operator_o1_pilot.png").exists()
