from click.testing import CliRunner

from ccpu.common.artifacts import read_json, read_jsonl, write_json, write_jsonl
from ccpu.dsl_dataset.chop import chop_example
from ccpu.dsl_dataset.cli import main


def test_chopper_preserves_decimals_and_common_abbreviations():
    parts = chop_example(
        {
            "dataset": "gsm8k",
            "question": "Mr. Lee pays $10.00. What remains?",
            "gold_reasoning": "",
        }
    )
    assert [part["text"] for part in parts] == ["Mr. Lee pays $10.00.", "What remains?"]


def test_click_pipeline_mines_audits_selects_and_prepares_teacher_requests(tmp_path):
    gsm = write_jsonl(
        tmp_path / "gsm8k.jsonl",
        [
            {
                "id": "one",
                "question": "Mia has 6 books. She buys twice as many. How many books now?",
                "answer": "She buys 6 * 2 = 12. 6 + 12 = 18.\n#### 18",
            },
            {
                "id": "two",
                "question": "A jar has 30 beads and loses 20 percent. How many remain?",
                "answer": "30 * .8 = 24.\n#### 24",
            },
        ],
    )
    tatqa = write_json(
        tmp_path / "tatqa.json",
        [
            {
                "table": {"table": [["Metric", "2025"], ["Sales", "50"]]},
                "paragraphs": [],
                "questions": [
                    {
                        "uid": "tat-one",
                        "question": "What is twice the sales value?",
                        "answer": 100,
                        "derivation": "50 * 2",
                        "answer_type": "arithmetic",
                        "answer_from": "table",
                        "scale": "",
                    }
                ],
            }
        ],
    )
    raw = tmp_path / "raw"
    runner = CliRunner()
    mined = runner.invoke(
        main,
        [
            "mine",
            "--source",
            f"gsm8k={gsm}",
            "--source",
            f"tatqa={tatqa}",
            "--source-split",
            "tatqa=development",
            "--output-dir",
            str(raw),
        ],
    )
    assert mined.exit_code == 0, mined.output
    manifest = read_json(raw / "manifest.json")
    assert manifest["datasets"]["gsm8k"]["record_count"] == 2
    assert manifest["datasets"]["tatqa"]["arithmetic_compatible_count"] == 1
    assert manifest["datasets"]["tatqa"]["declared_split"] == "development"
    gsm_rows = read_jsonl(raw / "gsm8k.jsonl")
    assert len({row["effective_scope"]["id"] for row in gsm_rows}) == 2
    assert any(not part["teacher_input_default"] for part in gsm_rows[0]["parts"])

    audit = raw / "chop_audit.json"
    audited = runner.invoke(
        main,
        ["audit-chops", "--input-dir", str(raw), "--output", str(audit)],
    )
    assert audited.exit_code == 0, audited.output
    assert read_json(audit)["hard_errors"] == []

    seed = tmp_path / "bootstrap" / "gsm8k_seed.jsonl"
    selected = runner.invoke(
        main,
        [
            "select",
            "--input",
            str(raw / "gsm8k.jsonl"),
            "--max-examples",
            "2",
            "--output",
            str(seed),
        ],
    )
    assert selected.exit_code == 0, selected.output
    assert len(read_jsonl(seed)) == 2
    ledger = read_jsonl(seed.with_name("gsm8k_seed_ledger.jsonl"))
    assert len(ledger) == 2
    assert not any("question" in row or "answer" in row for row in ledger)

    skill = tmp_path / "SKILL.md"
    skill.write_text("compile ASL", encoding="utf-8")
    requests = tmp_path / "bootstrap" / "requests.jsonl"
    prepared = runner.invoke(
        main,
        [
            "prepare-teacher",
            "--input",
            str(seed),
            "--skill",
            str(skill),
            "--output",
            str(requests),
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    request_rows = read_jsonl(requests)
    assert request_rows
    assert all(row["profile"] == "asl-arith-v0" for row in request_rows)
    assert all("API_KEY" not in str(row) for row in request_rows)


def test_annotation_bootstrap_creates_execution_verified_low_grade_asl(tmp_path):
    raw = write_jsonl(
        tmp_path / "gsm8k.jsonl",
        [
            {
                "dataset": "gsm8k",
                "split": "train",
                "source_id": "one",
                "record_sha256": "a" * 64,
                "question": "Mia has 6 books and buys twice as many. How many bought?",
                "answer": "12",
                "gold_reasoning": "She buys <<6*2=12>>12 books.",
                "metadata": {"arithmetic_compatible": True},
                "effective_scope": {
                    "id": "gsm8k:train:one",
                    "parent": None,
                    "kind": "benchmark_case",
                    "source": "dataset",
                },
                "parts": [],
            }
        ],
    )
    output = tmp_path / "annotated"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bootstrap-annotated",
            "--input",
            str(raw),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    accepted = read_jsonl(output / "accepted.jsonl")
    assert accepted[0]["asl"] == "step_1 = 6*2\nRETURN step_1"
    assert accepted[0]["validation"]["final_answer_verified"] is True
    assert accepted[0]["quality_grade"].startswith("Q0_")
