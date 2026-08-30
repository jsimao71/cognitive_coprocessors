from click.testing import CliRunner

from ccpu.common.artifacts import read_json, read_jsonl, write_json, write_jsonl
from ccpu.dsl_dataset.chop import chop_example
from ccpu.dsl_dataset.cli import main
from ccpu.dsl_dataset.expansion import finalize_asl_expansion
from ccpu.dsl_dataset.local_codex import run_local_codex_batches
from ccpu.paper1.asl_pilot_data import pattern_id


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
    tatqa_rows = read_jsonl(raw / "tatqa.jsonl")
    assert tatqa_rows[0]["source_context"]["table"][1] == ["Sales", "50"]

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
    assert all("correct_answer" not in row for row in request_rows)
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


def test_diverse_selector_excludes_sources_and_records_relation_cues(tmp_path):
    rows = []
    for source_id, question in (
        ("ratio", "Ava has half as many books as Ben. How many altogether?"),
        ("percent", "A store discounts a price by 20 percent. What remains?"),
        ("rate", "A car travels 30 miles per hour for 2 hours. How far?"),
        ("time", "In two years Mia will be 20. How old is she now?"),
    ):
        rows.append(
            {
                "dataset": "gsm8k",
                "split": "train",
                "source_id": source_id,
                "record_sha256": source_id.ljust(64, "0"),
                "question": question,
                "metadata": {"arithmetic_compatible": True},
                "parts": [{"part_id": 0, "teacher_input_default": True}],
            }
        )
    raw = write_jsonl(tmp_path / "raw.jsonl", rows)
    excluded = write_jsonl(
        tmp_path / "excluded.jsonl", [{"dataset": "gsm8k", "source_id": "ratio"}]
    )
    output = tmp_path / "diverse.jsonl"
    result = CliRunner().invoke(
        main,
        [
            "select-diverse",
            "--input",
            str(raw),
            "--exclude",
            str(excluded),
            "--dataset-target",
            "gsm8k=3",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    selected = read_jsonl(output)
    assert {row["source_id"] for row in selected} == {"percent", "rate", "time"}
    assert all(row["selection_relation_classes"] for row in selected)
    manifest = read_json(output.with_suffix(".manifest.json"))
    assert manifest["excluded_source_count"] == 1
    assert manifest["post_annotation_pattern_filter_required"] is True


def test_semantic_validator_accepts_forward_state_and_rejects_operation_ledgers(tmp_path):
    seeds = write_jsonl(
        tmp_path / "seed.jsonl",
        [
            {
                "dataset": "gsm8k",
                "split": "train",
                "source_id": source_id,
                "record_sha256": source_id * 64,
                "question": "Jessica is six years older than Claire. Claire is 18. How old is Jessica?",
                "answer": "24",
                "gold_reasoning": "",
                "metadata": {"arithmetic_compatible": True},
                "effective_scope": {
                    "id": f"gsm8k:train:{source_id}",
                    "parent": None,
                    "kind": "benchmark_case",
                    "source": "dataset",
                },
                "parts": [
                    {
                        "part_id": 0,
                        "text": "Jessica is six years older than Claire.",
                        "teacher_input_default": True,
                    },
                    {
                        "part_id": 1,
                        "text": "Claire is 18.",
                        "teacher_input_default": True,
                    },
                    {
                        "part_id": 2,
                        "text": "How old is Jessica?",
                        "teacher_input_default": True,
                    },
                ],
            }
            for source_id in ("a", "b")
        ],
    )
    annotations = write_jsonl(
        tmp_path / "annotations.jsonl",
        [
            {
                "dataset": "gsm8k",
                "source_id": "a",
                "part_mappings": [
                    {
                        "part_id": 0,
                        "status": "ok",
                        "asl": ["jessica.age_now = claire.age_now + 6"],
                    },
                    {"part_id": 1, "status": "ok", "asl": ["claire.age_now = 18"]},
                    {"part_id": 2, "status": "ok", "asl": ["RETURN jessica.age_now"]},
                ],
            },
            {
                "dataset": "gsm8k",
                "source_id": "b",
                "part_mappings": [
                    {"part_id": 0, "status": "ok", "asl": ["step_1 = 18"]},
                    {"part_id": 1, "status": "ok", "asl": ["step_2 = step_1 + 6"]},
                    {"part_id": 2, "status": "ok", "asl": ["RETURN step_2"]},
                ],
            },
        ],
    )
    output = tmp_path / "semantic"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "validate-semantic",
            "--seed",
            str(seeds),
            "--annotations",
            str(annotations),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert read_json(output / "summary.json")["accepted_count"] == 1
    assert "anonymous operation-ledger target" in read_jsonl(output / "rejected.jsonl")[0]["reason"]


def _expansion_row(source_id, target, question):
    scope = {"id": f"gsm8k:train:{source_id}", "kind": "benchmark_case", "parent": None}
    return {
        "dataset": "gsm8k",
        "split": "train",
        "source_id": source_id,
        "record_sha256": source_id.ljust(64, "0"),
        "question": question,
        "part_mappings": [{"part_id": 0, "asl": [f"{target} = 1", f"RETURN {target}"]}],
        "ccir": {
            "operations": [
                {
                    "scope": scope,
                    "source_line": 1,
                    "operation": {
                        "op": "SET",
                        "target": target,
                        "expr": {"op": "CONST", "value": 1},
                    },
                },
                {
                    "scope": scope,
                    "source_line": 2,
                    "operation": {"op": "RETURN", "expr": {"op": "REF", "path": target}},
                },
            ]
        },
    }


def test_expansion_finalizer_quarantines_frozen_eval_patterns(tmp_path):
    frozen_template = _expansion_row("frozen", "person.age", "A person's age is 1.")
    collision = _expansion_row("collision", "other.age", "Another person's age is 9.")
    eligible_a = _expansion_row("eligible_a", "shop.total", "A shop has 1 item total.")
    eligible_b = _expansion_row("eligible_b", "trip.distance", "A trip is 1 mile long.")
    existing = write_jsonl(
        tmp_path / "existing.jsonl", [_expansion_row("existing", "box.count", "One box.")]
    )
    candidates = write_jsonl(tmp_path / "candidates.jsonl", [collision, eligible_a, eligible_b])
    ledger = write_jsonl(
        tmp_path / "ledger.jsonl",
        [{"split": "test", "semantic_pattern_id": pattern_id(frozen_template)}],
    )
    manifest = finalize_asl_expansion(candidates, existing, ledger, tmp_path / "final", target=2)
    assert manifest["selected_count"] == 2
    assert manifest["frozen_eval_pattern_overlap"] == []
    assert manifest["quarantine_reason_counts"] == {"frozen_eval_semantic_pattern": 1}
    assert {
        row["source_id"] for row in read_jsonl(tmp_path / "final" / "expansion_train.jsonl")
    } == {
        "eligible_a",
        "eligible_b",
    }


def test_local_codex_runner_resumes_valid_completed_batches(tmp_path):
    requests = tmp_path / "requests"
    write_json(requests / "batch_000.json", {"items": []})
    output = tmp_path / "run"
    write_json(output / "annotations" / "batch_000.json", {"annotations": [{"source_id": "a"}]})
    prompt = write_json(tmp_path / "prompt.json", {"instruction": "test"})
    schema = write_json(tmp_path / "schema.json", {"type": "object"})
    manifest = run_local_codex_batches(
        requests,
        output,
        prompt_path=prompt,
        schema_path=schema,
        repo_root=tmp_path,
        executable="intentionally-missing-codex",
        concurrency=1,
    )
    assert manifest["completed_count"] == 1
    assert manifest["annotation_count"] == 1
    assert manifest["batches"][0]["status"] == "resumed"
