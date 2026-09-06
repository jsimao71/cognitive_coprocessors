from ccpu.common.artifacts import read_json, write_json, write_jsonl
from ccpu.paper1.e3.augmentation_analysis import analyze_gsm8k_augmentation_stage


def _prediction(example_id: str, correct: bool) -> dict:
    return {"example_id": example_id, "metrics": {"final_answer_correct": correct}}


def _summary(path, count: int, answer: float, path_f1: float):
    return write_json(
        path,
        {
            "eval_sha256": "frozen-eval",
            "prediction_count": count,
            "rates": {"final_answer_correct": answer, "parse_valid": 1.0},
            "component_mean_f1": {"paths": path_f1},
        },
    )


def test_augmentation_gate_accepts_strict_positive_selection_delta(tmp_path):
    selection_eval = write_jsonl(
        tmp_path / "selection.jsonl", [{"example_id": "a"}, {"example_id": "b"}]
    )
    historical_eval = write_jsonl(tmp_path / "historical.jsonl", [{"example_id": "h"}])
    baseline_selection = write_jsonl(
        tmp_path / "baseline-selection.jsonl",
        [_prediction("a", True), _prediction("b", False)],
    )
    candidate_selection = write_jsonl(
        tmp_path / "candidate-selection.jsonl",
        [_prediction("a", True), _prediction("b", True)],
    )
    baseline_historical = write_jsonl(
        tmp_path / "baseline-historical.jsonl", [_prediction("h", False)]
    )
    candidate_historical = write_jsonl(
        tmp_path / "candidate-historical.jsonl", [_prediction("h", True)]
    )
    report = analyze_gsm8k_augmentation_stage(
        stage="AUG1",
        selection_eval_path=selection_eval,
        historical_eval_path=historical_eval,
        baseline_selection_predictions=baseline_selection,
        candidate_selection_predictions=candidate_selection,
        baseline_historical_predictions=baseline_historical,
        candidate_historical_predictions=candidate_historical,
        baseline_historical_summary=_summary(tmp_path / "base-summary.json", 1, 0.0, 0.25),
        candidate_historical_summary=_summary(tmp_path / "candidate-summary.json", 1, 1.0, 0.75),
        output_path=tmp_path / "report.json",
    )

    assert report["decision"] == {
        "accepted": True,
        "criterion": "candidate selection correct count must strictly exceed baseline",
        "delta_correct": 1,
        "next_parent": "candidate",
    }
    assert report["augmentation_selection"]["candidate_only"] == 1
    assert (
        report["historical_diagnostic"]["semantic_diagnostics"]["component_mean_f1_deltas"]["paths"]
        == 0.5
    )
    assert read_json(tmp_path / "report.json")["stage"] == "AUG1"


def test_augmentation_gate_rejects_tie(tmp_path):
    evaluation = write_jsonl(tmp_path / "eval.jsonl", [{"example_id": "a"}])
    predictions = write_jsonl(tmp_path / "predictions.jsonl", [_prediction("a", True)])
    summary = _summary(tmp_path / "summary.json", 1, 1.0, 1.0)

    report = analyze_gsm8k_augmentation_stage(
        stage="AUG1",
        selection_eval_path=evaluation,
        historical_eval_path=evaluation,
        baseline_selection_predictions=predictions,
        candidate_selection_predictions=predictions,
        baseline_historical_predictions=predictions,
        candidate_historical_predictions=predictions,
        baseline_historical_summary=summary,
        candidate_historical_summary=summary,
        output_path=tmp_path / "report.json",
    )

    assert report["decision"]["accepted"] is False
    assert report["decision"]["next_parent"] == "baseline"
