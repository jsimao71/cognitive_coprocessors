import pytest

from ccpu.common.artifacts import read_json, write_jsonl
from ccpu.paper1.e3.contribution_analysis import analyze_gsm8k_contribution


def _predictions(path, outcomes):
    return write_jsonl(
        path,
        [
            {
                "example_id": example_id,
                "metrics": {"final_answer_correct": correct},
            }
            for example_id, correct in outcomes.items()
        ],
    )


def _inputs(tmp_path):
    original = write_jsonl(
        tmp_path / "original.jsonl",
        [{"example_id": f"o{index}"} for index in range(1, 5)],
    )
    large = write_jsonl(
        tmp_path / "large.jsonl",
        [
            {"example_id": "l1", "parent_example_id": "o1"},
            {"example_id": "l2", "parent_example_id": "o2"},
        ],
    )
    paths = {
        "original_direct": _predictions(
            tmp_path / "original_direct.jsonl",
            {"o1": True, "o2": False, "o3": True, "o4": False},
        ),
        "original_asl": _predictions(
            tmp_path / "original_asl.jsonl",
            {"o1": True, "o2": True, "o3": False, "o4": False},
        ),
        "large_direct": _predictions(
            tmp_path / "large_direct.jsonl", {"l1": False, "l2": False}
        ),
        "large_asl": _predictions(
            tmp_path / "large_asl.jsonl", {"l1": True, "l2": True}
        ),
    }
    return original, large, paths


def _analyze(tmp_path, original, large, paths):
    return analyze_gsm8k_contribution(
        original_eval_path=original,
        large_eval_path=large,
        original_direct_paths=[("direct_reasoning", paths["original_direct"])],
        original_asl_paths=[("seed11", paths["original_asl"])],
        large_direct_paths=[("direct_reasoning", paths["large_direct"])],
        large_asl_paths=[("seed11", paths["large_asl"])],
        output_path=tmp_path / "report.json",
        bootstrap_seed=7,
        bootstrap_samples=1000,
    )


def test_contribution_analysis_preserves_pairing_and_differential(tmp_path):
    original, large, paths = _inputs(tmp_path)
    report = _analyze(tmp_path, original, large, paths)

    assert report["identity_counts"] == {"original": 4, "large_number": 2}
    paired = report["original_asl_vs_direct"]["seed11__vs__direct_reasoning"]
    assert paired["both_correct"] == 1
    assert paired["left_only"] == 1
    assert paired["right_only"] == 1
    assert paired["both_wrong"] == 1

    direct = report["large_number_robustness"]["direct_reasoning"]
    assert direct["original_eligible_rate"] == 0.5
    assert direct["large_rate"] == 0.0
    assert direct["large_minus_original"] == -0.5
    assert direct["lost_on_large"] == 1

    asl = report["large_number_robustness"]["seed11"]
    assert asl["original_eligible_rate"] == 1.0
    assert asl["large_rate"] == 1.0
    assert asl["retained_correct"] == 2
    differential = report["large_number_differential_degradation"][
        "seed11__vs__direct_reasoning"
    ]
    assert differential["estimate"] == 0.5
    assert differential["bootstrap_identity_95"] == [0, 1]
    assert read_json(tmp_path / "report.json") == report


def test_contribution_analysis_is_deterministic(tmp_path):
    original, large, paths = _inputs(tmp_path)
    first = _analyze(tmp_path, original, large, paths)
    second = _analyze(tmp_path, original, large, paths)
    assert first == second


def test_contribution_analysis_rejects_identity_mismatch(tmp_path):
    original, large, paths = _inputs(tmp_path)
    paths["original_direct"] = _predictions(
        tmp_path / "bad.jsonl", {"o1": True, "o2": False, "o3": True}
    )
    with pytest.raises(ValueError, match="prediction identities differ"):
        _analyze(tmp_path, original, large, paths)


def test_contribution_analysis_rejects_overlapping_labels(tmp_path):
    original, large, paths = _inputs(tmp_path)
    with pytest.raises(ValueError, match="must be distinct"):
        analyze_gsm8k_contribution(
            original_eval_path=original,
            large_eval_path=large,
            original_direct_paths=[("same", paths["original_direct"])],
            original_asl_paths=[("same", paths["original_asl"])],
            large_direct_paths=[("same", paths["large_direct"])],
            large_asl_paths=[("same", paths["large_asl"])],
            output_path=tmp_path / "report.json",
        )
