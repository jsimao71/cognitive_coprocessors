from decimal import Decimal

import pytest

from ccpu.common.artifacts import (
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl import validate_asl
from ccpu.dsl.registry import ARITHMETIC_FUNCTIONS
from ccpu.paper1.e3.gsm8k_interventions import (
    freeze_gsm8k_matched_interventions,
    freeze_gsm8k_operator_jitter_matrix,
)
from ccpu.paper1.e3.intervention_analysis import analyze_matched_interventions
from ccpu.paper1.e3.intervention_audit import (
    audit_intervention_panel,
    finalize_intervention_review,
)
from ccpu.paper1.e3.operator_analysis import analyze_operator_ladder, analyze_operator_pilot
from ccpu.paper1.e3.operator_complexity import freeze_o1_dataset, freeze_o1_pilot
from ccpu.paper1.e3.operator_levels import freeze_operator_level, freeze_operator_pilot


def _audit_row(example_id, question, program, answer, binding_path, transformed_value):
    return {
        "example_id": example_id,
        "parent_example_id": example_id.split(":cell", 1)[0],
        "condition": "audit-test",
        "question": question,
        "question_sha256": fingerprint(question),
        "reference_return": str(answer),
        "gold_asl": program,
        "transformation": {
            "factor": 1,
            "jitter_seed": 17,
            "binding_path": binding_path,
            "transformed_value": transformed_value,
        },
    }


def test_intervention_audit_replays_and_flags_without_automatic_exclusion(tmp_path):
    root = tmp_path / "panel"
    rows = [
        _audit_row(
            "gsm8k:1:cell-a",
            (
                "A box held 5 hats. Ana removed 9 hats. "
                "How many more hats are needed to restore the box?"
            ),
            (
                "box.hats.initial = 5\n"
                "ana.hats.removed = 9\n"
                "box.hats.remaining = box.hats.initial - ana.hats.removed\n"
                "RETURN box.hats.remaining"
            ),
            -4,
            "ana.hats.removed",
            9,
        ),
        _audit_row(
            "gsm8k:2:cell-a",
            "Mike watches television for 30 hours per day. How many hours is that?",
            "mike.hours_per_day = 30\nRETURN mike.hours_per_day",
            30,
            "mike.hours_per_day",
            30,
        ),
    ]
    write_jsonl(root / "cell_a" / "test.jsonl", rows)
    write_jsonl(
        root / "cell_b" / "test.jsonl",
        [
            {**row, "example_id": row["example_id"].replace("cell-a", "cell-b")}
            for row in rows
        ],
    )

    report = audit_intervention_panel(
        input_dirs=[root], output_dir=tmp_path / "audit", split="test"
    )
    records = read_jsonl(tmp_path / "audit" / "audit_records.jsonl")

    assert report["counts"] == {
        "files": 2,
        "rows": 4,
        "unique_parent_examples": 2,
        "integrity_failures": 0,
        "labeled_rows": 4,
        "review_required_rows": 4,
        "review_required_unique_parents": 2,
        "strict_common_support_parents": 0,
        "labeled_unique_parents": 2,
    }
    assert report["pairing_audit"]["identical_parent_sets_across_files"] is True
    assert report["policy"]["automatic_exclusion"] is False
    first_codes = {flag["code"] for flag in records[0]["semantic_flags"]}
    assert "negative_requested_quantity" in first_codes
    assert "negative_world_quantity" in first_codes
    assert "component_exceeds_available_total" in first_codes
    second_codes = {flag["code"] for flag in records[1]["semantic_flags"]}
    assert "bounded_rate_or_duration_exceeded" in second_codes
    assert all(row["analysis_disposition"] == "manual_review_required" for row in records)

    decisions = write_jsonl(
        tmp_path / "decisions.jsonl",
        [
            {
                "parent_example_id": "gsm8k:1",
                "decision": "exclude_from_strict_common_support",
                "rationale": "negative inventory is inconsistent",
            },
            {
                "parent_example_id": "gsm8k:2",
                "decision": "retain",
                "rationale": "retain for test coverage",
            },
        ],
    )
    frozen = finalize_intervention_review(
        audit_records_path=tmp_path / "audit" / "audit_records.jsonl",
        decisions_path=decisions,
        output_path=tmp_path / "audit" / "frozen.json",
    )
    assert frozen["counts"]["strict_common_support_parents"] == 1
    assert frozen["parent_example_ids"] == ["gsm8k:2"]
    assert frozen["blinding_disclosure"]["reviewer_had_prior_aggregate_result_knowledge"]


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
        {"example_id": "a", "operator_family": "square", "operator_level": "O1"},
        {"example_id": "b", "operator_family": "cube", "operator_level": "O1"},
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


def test_operator_ladder_analysis_preserves_categorical_level_order(tmp_path):
    summaries = []
    for level, direct in (("O3", 0.4), ("O0", 0.9)):
        path = tmp_path / f"{level}.json"
        conditions = {
            condition: {
                "correct": round(10 * accuracy),
                "count": 10,
                "accuracy": accuracy,
                "generated_tokens": {"mean": tokens},
            }
            for condition, accuracy, tokens in (
                ("direct", direct, 100), ("ap", 1.0, 12), ("as", 0.8, 14)
            )
        }
        write_json(
            path,
            {
                "schema_version": "ccpu.paper1.operator_pilot_analysis.v1",
                "operator_level": level,
                "conditions": conditions,
            },
        )
        summaries.append((level, path))

    report = analyze_operator_ladder(summaries, tmp_path / "ladder")

    assert report["level_order"] == ["O0", "O3"]
    assert "not equally spaced" in report["categorical_level_warning"]
    assert (tmp_path / "ladder" / "operator_ladder_pilots.png").exists()


def test_matched_interventions_keep_real_parent_graph_and_exclude_training(tmp_path):
    source = tmp_path / "source.jsonl"
    training = tmp_path / "training.jsonl"
    rows = []
    for index in range(8):
        value = 4 + index % 6
        rows.append(
            {
                "parent_source_id": str(index),
                "prompt": (
                    "Compile.\n\nInput:\nProblem: Mina has "
                    f"{value} red boxes and 3 blue boxes. Each box contains 2 cards. "
                    "How many cards are in all the boxes?\nASL:"
                ),
                "target": (
                    f"mina.red.boxes = {value}\n"
                    "mina.blue.boxes = 3\n"
                    "mina.total.boxes = mina.red.boxes + mina.blue.boxes\n"
                    "box.cards = 2\n"
                    "mina.total.cards = mina.total.boxes * box.cards\n"
                    "RETURN mina.total.cards"
                ),
            }
        )
    write_jsonl(source, rows)
    write_jsonl(training, [{"parent_source_id": "0"}])

    manifest = freeze_gsm8k_matched_interventions(
        source_corpus_path=source,
        excluded_training_path=training,
        output_dir=tmp_path / "out",
        train_count=3,
        dev_count=2,
        test_count=2,
        factors=(1, 100, 1000),
        jitter_seeds=(17,),
    )

    assert manifest["leakage_audit"]["selected_source_overlap"] == 0
    original = read_jsonl(tmp_path / "out" / "original" / "test.jsonl")
    o6 = read_jsonl(tmp_path / "out" / "operators" / "o6" / "test.jsonl")
    assert [row["parent_example_id"] for row in original] == [
        row["parent_example_id"] for row in o6
    ]
    assert [row["reference_return"] for row in original] == [
        row["reference_return"] for row in o6
    ]
    assert all("mina.total.cards" in row["gold_asl"] for row in o6)
    assert all("positive_quadratic_root" in row["gold_asl"] for row in o6)
    jittered = read_jsonl(
        tmp_path / "out" / "jitter" / "seed_17" / "x100" / "test.jsonl"
    )
    assert all(
        -0.30 <= row["transformation"]["jitter_realized"] <= 0.30
        for row in jittered
    )

    combined = freeze_gsm8k_operator_jitter_matrix(
        panel_dir=tmp_path / "out",
        output_dir=tmp_path / "combined",
        jitter_seeds=(17, 23),
    )
    assert [stage["jitter_seed"] for stage in combined["execution_order"]] == [17, 23]
    assert combined["execution_order"][0]["cells"] == [
        "x1/o1",
        "x1/o5",
        "x1/o6",
        "x1000/o1",
        "x1000/o5",
        "x1000/o6",
    ]
    combined_o5 = read_jsonl(
        tmp_path / "combined" / "seed_17" / "x1000" / "o5" / "test.jsonl"
    )
    standalone = read_jsonl(
        tmp_path / "out" / "jitter" / "seed_17" / "x1000" / "test.jsonl"
    )
    assert [row["parent_example_id"] for row in combined_o5] == [
        row["parent_example_id"] for row in standalone
    ]
    assert all("solve_linear" in row["gold_asl"] for row in combined_o5)
    assert all(row["transformation"]["factor"] == 1000 for row in combined_o5)
    assert all(row["transformation"]["answer_recomputed_by_runtime"] for row in combined_o5)
    assert [row["transformation"]["transformed_value"] for row in combined_o5] == [
        row["transformation"]["transformed_value"] for row in standalone
    ]


def test_intervention_analysis_rejects_unpaired_rows_and_reports_only_matched(tmp_path):
    evaluation = tmp_path / "eval.jsonl"
    write_jsonl(
        evaluation,
        [
            {"example_id": "a", "question_sha256": "qa"},
            {"example_id": "b", "question_sha256": "qb"},
        ],
    )
    paths = {}
    for label, correctness in (("direct", (True, False)), ("asl", (True, True))):
        directory = tmp_path / label
        predictions = directory / "predictions.jsonl"
        write_jsonl(
            predictions,
            [
                {
                    "example_id": example_id,
                    "question_sha256": question_sha,
                    "generated_tokens": 20 if label == "asl" else 100,
                    "metrics": {"final_answer_correct": correct},
                }
                for example_id, question_sha, correct in zip(
                    ("a", "b"), ("qa", "qb"), correctness, strict=True
                )
            ],
        )
        write_json(directory / "summary.json", {"eval_sha256": file_sha256(evaluation)})
        paths[label] = predictions

    report = analyze_matched_interventions(
        cells=[("O5", evaluation, paths["direct"], paths["asl"])],
        output_dir=tmp_path / "analysis",
    )

    assert report["cells"][0]["direct_accuracy"] == 0.5
    assert report["cells"][0]["asl_accuracy"] == 1.0
    assert report["cells"][0]["asl_minus_direct"] == 0.5
    assert report["cells"][0]["paired"]["asl_only"] == 1

    diagnostics = write_jsonl(
        tmp_path / "direct_v2.jsonl",
        [
            {
                "example_id": example_id,
                "question_sha256": question_sha,
                "v2_correct": correct,
            }
            for example_id, question_sha, correct in zip(
                ("a", "b"), ("qa", "qb"), (True, True), strict=True
            )
        ],
    )
    write_json(
        diagnostics.with_name("summary.json"),
        {
            "inputs": {
                "eval": {"sha256": file_sha256(evaluation)},
                "predictions": {"sha256": file_sha256(paths["direct"])},
            }
        },
    )
    support = write_json(tmp_path / "support.json", {"parent_example_ids": ["a"]})
    audited = analyze_matched_interventions(
        cells=[("O5", evaluation, paths["direct"], paths["asl"])],
        output_dir=tmp_path / "audited",
        common_support_path=support,
        direct_diagnostic_paths={"O5": diagnostics},
    )
    cell = audited["cells"][0]
    assert cell["scorer_v2_all"]["direct_accuracy"] == 1.0
    assert cell["registered_common_support"]["count"] == 1
    assert cell["scorer_v2_common_support"]["paired"]["both_correct"] == 1

    rows = read_jsonl(paths["asl"])
    rows[0]["question_sha256"] = "wrong"
    write_jsonl(paths["asl"], rows)
    with pytest.raises(ValueError, match="question hash mismatch"):
        analyze_matched_interventions(
            cells=[("O5", evaluation, paths["direct"], paths["asl"])],
            output_dir=tmp_path / "bad",
        )
