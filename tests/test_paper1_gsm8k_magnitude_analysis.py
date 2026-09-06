from ccpu.common.artifacts import read_json, write_jsonl
from ccpu.paper1.e3.magnitude_analysis import (
    analyze_magnitude_curve,
    analyze_magnitude_failures,
    project_magnitude_predictions,
)


def _eval_row(index, factor=100):
    parent = f"gsm8k-official-test-{index:04d}"
    return {
        "example_id": f"{parent}:magnitude-x{factor}" if factor != 1 else parent,
        "parent_example_id": parent,
        "reference_return": "700" if factor != 1 else "7",
        "effective_scope": {"id": f"case:{index}:x{factor}"},
        "transformation": {
            "factor": factor,
            "source_value_mapping": {"3": "300", "4": "400"},
            "hidden_execution_trace": [
                {"original_result": "7", "transformed_result": "700"}
            ],
        },
    }


def _prediction(example_id, asl, correct):
    return {
        "example_id": example_id,
        "predicted_asl": asl,
        "metrics": {"final_answer_correct": correct},
    }


def test_magnitude_failure_partition_and_literal_oracle(tmp_path):
    eval_rows = [_eval_row(index) for index in range(4)]
    original_asl = "a.left = 3\na.right = 4\na.total = a.left + a.right\nRETURN a.total"
    original = [
        _prediction(row["parent_example_id"], original_asl, True) for row in eval_rows
    ]
    transformed = [
        _prediction(
            eval_rows[0]["example_id"],
            "a.left = 300\na.right = 400\na.total = a.left + a.right\nRETURN a.total",
            True,
        ),
        _prediction(
            eval_rows[1]["example_id"],
            "b.left = 30\nb.right = 40\nb.total = b.left + b.right\nRETURN b.total",
            False,
        ),
        _prediction(eval_rows[2]["example_id"], "a.left =", False),
        _prediction(
            eval_rows[3]["example_id"],
            "a.left = 300\na.right = 400\na.total = a.left * a.right\nRETURN a.total",
            False,
        ),
    ]
    report = analyze_magnitude_failures(
        transformed_eval_path=write_jsonl(tmp_path / "eval.jsonl", eval_rows),
        original_predictions_path=write_jsonl(tmp_path / "original.jsonl", original),
        transformed_predictions_path=write_jsonl(
            tmp_path / "transformed.jsonl", transformed
        ),
        output_dir=tmp_path / "analysis",
    )

    assert report["counts"] == {
        "total": 4,
        "correct": 1,
        "failures": 3,
        "wrong_semantic_structure": 1,
        "literal_copy_error": 1,
        "runtime_failure": 1,
    }
    assert report["rates"]["oracle_corrected_accuracy"] == 0.5
    assert report["runtime_failure_stages"] == {"parse": 1}
    assert read_json(tmp_path / "analysis" / "summary.json") == report


def test_magnitude_curve_requires_and_reports_common_parents(tmp_path):
    original_eval = [_eval_row(index, factor=1) for index in range(2)]
    transformed_eval = [_eval_row(index, factor=100) for index in range(2)]
    b1_original = [
        _prediction(original_eval[0]["example_id"], "", True),
        _prediction(original_eval[1]["example_id"], "", True),
    ]
    b1_transformed = [
        _prediction(transformed_eval[0]["example_id"], "", True),
        _prediction(transformed_eval[1]["example_id"], "", False),
    ]
    report = analyze_magnitude_curve(
        eval_paths=[
            (1, write_jsonl(tmp_path / "factor1.jsonl", original_eval)),
            (100, write_jsonl(tmp_path / "factor100.jsonl", transformed_eval)),
        ],
        prediction_paths=[
            ("B1", 1, write_jsonl(tmp_path / "b1-1.jsonl", b1_original)),
            ("B1", 100, write_jsonl(tmp_path / "b1-100.jsonl", b1_transformed)),
        ],
        output_dir=tmp_path / "curve",
    )

    assert report["common_parent_count"] == 2
    assert [row["accuracy"] for row in report["results"]] == [1.0, 0.5]
    assert (tmp_path / "curve" / "accuracy_by_magnitude.csv").exists()
    assert (tmp_path / "curve" / "accuracy_by_magnitude.svg").exists()


def test_projection_reuses_only_identical_questions_and_answers(tmp_path):
    source_eval = [_eval_row(index, factor=100) for index in range(2)]
    target_eval = [_eval_row(index, factor=100) for index in range(2)]
    for index, row in enumerate(source_eval):
        row["question_sha256"] = f"question-{index}"
        target_eval[index]["question_sha256"] = f"question-{index}"
        source_eval[index]["difficulty_stratum"] = "low"
        target_eval[index]["difficulty_stratum"] = "low"
        source_eval[index]["source_row"] = index
        target_eval[index]["source_row"] = index
        source_eval[index]["example_id"] += ":old-protocol"
    predictions = [
        _prediction(row["example_id"], "RETURN 700", True) for row in source_eval
    ]
    manifest = project_magnitude_predictions(
        source_eval_path=write_jsonl(tmp_path / "source-eval.jsonl", source_eval),
        target_eval_path=write_jsonl(tmp_path / "target-eval.jsonl", target_eval),
        source_predictions_path=write_jsonl(
            tmp_path / "source-predictions.jsonl", predictions
        ),
        output_dir=tmp_path / "projected",
    )

    assert manifest["count"] == 2
    projected = read_json(tmp_path / "projected" / "projection_manifest.json")
    assert projected == manifest
