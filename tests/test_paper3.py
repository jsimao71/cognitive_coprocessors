from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl, write_json, write_jsonl


def test_public_control_registry_is_bounded_and_blocks_unannotated_headlines(tmp_path):
    compute = write_jsonl(
        tmp_path / "compute.jsonl",
        [
            {
                "benchmark": "gsm8k",
                "example_id": f"g-{index}",
                "content_sha256": str(index).zfill(64),
                "selection_key": str(10 - index).zfill(64),
            }
            for index in range(3)
        ],
    )
    crag = write_jsonl(
        tmp_path / "crag.jsonl",
        [
            {
                "benchmark": "crag",
                "example_id": "c-1",
                "content_sha256": "c" * 64,
                "selection_key": "c" * 64,
            }
        ],
    )
    tatqa = write_jsonl(
        tmp_path / "tatqa.jsonl",
        [
            {
                "benchmark": "tatqa",
                "example_id": "t-1",
                "content_sha256": "d" * 64,
                "selection_key": "d" * 64,
                "answer_type": "arithmetic",
            }
        ],
    )
    crag_diagnostics = write_json(
        tmp_path / "crag_diagnostics.json",
        {"interpretation": {"paper3_5_gate": "closed_pending_model_and_evidence_runs"}},
    )
    paper2_5_readiness = write_json(
        tmp_path / "paper2_5_readiness.json",
        {"headline_ready": False, "claim_boundary": "readiness ledger only"},
    )
    output = tmp_path / "output"
    assert main(
        [
            "paper3",
            "freeze-public",
            "--compute-selection",
            str(compute),
            "--crag-selection",
            str(crag),
            "--tatqa-selection",
            str(tatqa),
            "--per-benchmark",
            "2",
            "--crag-diagnostics",
            str(crag_diagnostics),
            "--paper2-5-readiness",
            str(paper2_5_readiness),
            "--output-dir",
            str(output),
        ]
    ) == 0
    manifest = read_json(output / "manifest.json")
    rows = read_jsonl(output / "selection.jsonl")
    assert manifest["record_count"] == 4
    assert manifest["headline_ready"] is False
    assert "audited_causal_span_annotations" in manifest["blocking_requirements"]
    assert "matched_model_facing_public_condition_results" in manifest[
        "blocking_requirements"
    ]
    assert manifest["upstream_diagnostics"]["paper2_5_public_suite"]["headline_ready"] is False
    tatqa_row = next(row for row in rows if row["benchmark"] == "tatqa")
    assert tatqa_row["assistance_type"] == "RETRIEVE"
    assert tatqa_row["secondary_assistance_type"] == "COMPUTE"
