import pytest

from ccpu.common.artifacts import write_json, write_jsonl
from ccpu.paper1.e3.cross_dataset_analysis import analyze_cross_dataset_transfer


def _prediction(identity, correct):
    return {
        "example_id": identity,
        "difficulty_stratum": "low",
        "generated_tokens": 10,
        "metrics": {
            "parse_valid": True,
            "lowerable_to_ccir": True,
            "type_valid": True,
            "executable": True,
            "final_answer_correct": correct,
        },
    }


def _fixture(tmp_path):
    root = tmp_path / "asdiv"
    diagnostic = write_jsonl(
        root / "diagnostic.jsonl",
        [{"example_id": "a"}, {"example_id": "b"}, {"example_id": "c"}],
    )
    del diagnostic
    manifest = write_json(root / "manifest.json", {"dataset": "asdiv"})
    e0 = write_jsonl(
        root / "e0.jsonl",
        [_prediction("a", True), _prediction("b", False), _prediction("c", False)],
    )
    e1 = write_jsonl(
        root / "e1.jsonl",
        [_prediction("a", True), _prediction("b", True), _prediction("c", False)],
    )
    return manifest, e0, e1


def test_cross_dataset_analysis_is_paired(tmp_path):
    manifest, e0, e1 = _fixture(tmp_path)
    report = analyze_cross_dataset_transfer(
        manifest_paths={"asdiv": manifest},
        prediction_paths={("asdiv", "E0"): e0, ("asdiv", "E1"): e1},
        output_path=tmp_path / "analysis.json",
    )
    assert report["datasets"]["asdiv"]["E0"]["counts"]["final_answer_correct"] == 1
    assert report["datasets"]["asdiv"]["E1"]["counts"]["final_answer_correct"] == 2
    assert report["paired_comparisons"]["asdiv:E0_to_E1"]["delta_correct"] == 1


def test_cross_dataset_analysis_rejects_incomplete_condition(tmp_path):
    manifest, e0, _ = _fixture(tmp_path)
    rows = [_prediction("a", True), _prediction("b", False)]
    partial = write_jsonl(tmp_path / "partial.jsonl", rows)
    with pytest.raises(ValueError, match="incomplete asdiv/E1"):
        analyze_cross_dataset_transfer(
            manifest_paths={"asdiv": manifest},
            prediction_paths={("asdiv", "E0"): e0, ("asdiv", "E1"): partial},
            output_path=tmp_path / "analysis.json",
        )
