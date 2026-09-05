import pytest

from ccpu.common.artifacts import read_json, write_json
from ccpu.paper1.e3.model_size_analysis import analyze_model_size_interaction


def _report(path, *, pair, answer_delta, robustness, original_hash="o", large_hash="l"):
    return write_json(
        path,
        {
            "identity_counts": {"original": 250, "large_number": 59},
            "original_asl_vs_direct": {pair: {"delta": answer_delta}},
            "large_number_differential_degradation": {
                pair: {"estimate": robustness}
            },
            "sources": {
                "original_eval": {"sha256": original_hash},
                "large_eval": {"sha256": large_hash},
            },
        },
    )


def test_model_size_analysis_computes_matched_interactions(tmp_path):
    small = _report(
        tmp_path / "small.json", pair="s__vs__d", answer_delta=-0.2, robustness=0.1
    )
    large = _report(
        tmp_path / "large.json", pair="l__vs__d", answer_delta=-0.05, robustness=0.18
    )
    output = tmp_path / "interaction.json"
    report = analyze_model_size_interaction(
        small_report_path=small,
        large_report_path=large,
        small_pair="s__vs__d",
        large_pair="l__vs__d",
        output_path=output,
    )
    assert report["original_answer_contribution"]["model_size_interaction"] == pytest.approx(
        0.15
    )
    assert report["large_number_robustness_contribution"][
        "model_size_interaction"
    ] == pytest.approx(0.08)
    assert report["gates"]["larger_model_benefits_asl_answer_contribution"]
    assert read_json(output) == report


def test_model_size_analysis_rejects_different_eval_freezes(tmp_path):
    small = _report(tmp_path / "small.json", pair="s", answer_delta=0, robustness=0)
    large = _report(
        tmp_path / "large.json",
        pair="l",
        answer_delta=0,
        robustness=0,
        original_hash="different",
    )
    with pytest.raises(ValueError, match="different original_eval"):
        analyze_model_size_interaction(
            small_report_path=small,
            large_report_path=large,
            small_pair="s",
            large_pair="l",
            output_path=tmp_path / "out.json",
        )
