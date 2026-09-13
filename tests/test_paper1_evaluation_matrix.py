import pytest

from ccpu.common.artifacts import read_json, write_json, write_jsonl
from ccpu.paper1.e3.evaluation_matrix import build_evaluation_matrix


def _manifest(tmp_path, prediction_paths):
    return write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": "ccpu.paper1.evaluation_matrix_manifest.v1",
            "columns": [
                {"id": "direct", "label": "Direct"},
                {"id": "asl_base", "label": "ASL no LoRA"},
            ],
            "rows": [
                {
                    "id": "toy",
                    "label": "Toy",
                    "eval_path": "eval.jsonl",
                    "cells": {
                        "direct": {"prediction_paths": prediction_paths},
                    },
                }
            ],
        },
    )


def test_matrix_reports_complete_partial_and_missing_cells(tmp_path):
    write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {"example_id": "a", "question_sha256": "ha"},
            {"example_id": "b", "question_sha256": "hb"},
        ],
    )
    write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {
                "example_id": "a",
                "question_sha256": "ha",
                "generated_tokens": 10,
                "metrics": {"final_answer_correct": True},
            }
        ],
    )
    report = build_evaluation_matrix(
        manifest_path=_manifest(tmp_path, ["predictions.jsonl"]),
        repository_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    direct = report["rows"][0]["cells"]["direct"]
    assert direct["status"] == "partial"
    assert direct["prediction_count"] == 1
    assert direct["rates"]["final_answer_correct"] == 1.0
    assert report["rows"][0]["cells"]["asl_base"]["status"] == "missing"
    assert "100.0% (1/1 partial; N=2)" in (tmp_path / "out" / "matrix.md").read_text()
    assert read_json(tmp_path / "out" / "matrix.json") == report


def test_matrix_rejects_question_hash_mismatch(tmp_path):
    write_jsonl(tmp_path / "eval.jsonl", [{"example_id": "a", "question_sha256": "gold"}])
    write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {
                "example_id": "a",
                "question_sha256": "wrong",
                "metrics": {"final_answer_correct": True},
            }
        ],
    )

    with pytest.raises(ValueError, match="question hash mismatch"):
        build_evaluation_matrix(
            manifest_path=_manifest(tmp_path, ["predictions.jsonl"]),
            repository_root=tmp_path,
            output_dir=tmp_path / "out",
        )
