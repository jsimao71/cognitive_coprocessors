from __future__ import annotations

import json

import pytest

from ccpu.paper1.asl_replication_analysis import analyze_asl_replications


def _write_rows(path, values):
    rows = []
    for index, (answer, execute, alpha_return, alpha_f1) in enumerate(values):
        metrics = {
            "parse_valid": True,
            "lowerable_to_ccir": True,
            "type_valid": True,
            "executable": execute,
            "semantic_state_equivalent": False,
            "semantic_return_equivalent": False,
            "alpha_state_equivalent": alpha_f1 == 1.0,
            "alpha_return_equivalent": alpha_return,
            "final_answer_correct": answer,
            "alpha_state_metrics": {"f1": alpha_f1},
        }
        rows.append({"example_id": f"e{index}", "metrics": metrics})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_replication_analysis_preserves_seed_boundaries(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    seed_a = tmp_path / "seed_a.jsonl"
    seed_b = tmp_path / "seed_b.jsonl"
    _write_rows(baseline, [(True, True, False, 0.5), (False, False, False, 0.0)])
    _write_rows(seed_a, [(True, True, True, 1.0), (True, True, False, 0.5)])
    _write_rows(seed_b, [(False, True, False, 0.5), (True, True, True, 1.0)])

    report = analyze_asl_replications(
        baseline_path=baseline,
        candidate_paths=[("a", seed_a), ("b", seed_b)],
    )

    answers = report["aggregate"]["final_answer_correct"]
    assert answers["baseline_count"] == 1
    assert answers["candidate_counts"] == [2, 1]
    assert answers["candidate_mean_count"] == pytest.approx(1.5)
    assert answers["all_seeds_above_baseline"] is False
    assert report["per_seed"]["b"]["endpoints"]["final_answer_correct"]["gains"] == 1
    assert report["per_seed"]["b"]["endpoints"]["final_answer_correct"]["losses"] == 1
    assert report["per_seed"]["b"]["endpoints"]["final_answer_correct"][
        "two_sided_exact_mcnemar_p"
    ] == 1.0
    assert report["alpha_state_f1"]["candidate_mean"] == pytest.approx(0.75)


def test_replication_analysis_rejects_mismatched_identities(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_rows(baseline, [(True, True, True, 1.0)])
    candidate.write_text(
        json.dumps(
            {
                "example_id": "different",
                "metrics": {"alpha_state_metrics": {"f1": 1.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match baseline identities"):
        analyze_asl_replications(
            baseline_path=baseline,
            candidate_paths=[("candidate", candidate)],
        )
