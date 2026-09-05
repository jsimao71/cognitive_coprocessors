from types import SimpleNamespace

import pytest

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_jsonl
from ccpu.paper1.e3.gsm8k_confirmatory import (
    freeze_official_gsm8k,
    merge_official_gsm8k_shards,
    run_official_gsm8k_shard,
)


def _source(tmp_path):
    source = write_jsonl(
        tmp_path / "official.jsonl",
        [
            {"question": "One plus one?", "answer": "1 + 1 = <<1+1=2>>2\n#### 2"},
            {"question": "Three plus two?", "answer": "3 + 2 = <<3+2=5>>5\n#### 5"},
            {
                "question": "Ten minus four, then times two?",
                "answer": "10 - 4 = <<10-4=6>>6\n6 * 2 = <<6*2=12>>12\n#### 12",
            },
        ],
    )
    train = write_jsonl(tmp_path / "train.jsonl", [{"question": "A different problem."}])
    return source, train


def test_freeze_official_gsm8k_hides_rationales_and_audits_overlap(tmp_path):
    source, train = _source(tmp_path)
    output = tmp_path / "frozen"
    report = freeze_official_gsm8k(
        source_path=source,
        train_paths=[train],
        output_dir=output,
        expected_sha256=file_sha256(source),
        expected_rows=3,
        confirmatory_size=2,
    )
    assert report["counts"]["full"] == 3
    assert report["leakage_audit"]["passed"]
    rows = read_jsonl(output / "full.jsonl")
    assert all(row["source_fields_visible_to_model"] == ["question"] for row in rows)
    assert all("gold_reasoning" not in row for row in rows)
    assert {row["reference_return"] for row in rows} == {"2", "5", "12"}


def test_freeze_official_gsm8k_rejects_exact_train_question_overlap(tmp_path):
    source, _ = _source(tmp_path)
    train = write_jsonl(tmp_path / "overlap.jsonl", [{"question": " ONE  PLUS ONE? "}])
    with pytest.raises(ValueError, match="exact train-question overlaps"):
        freeze_official_gsm8k(
            source_path=source,
            train_paths=[train],
            output_dir=tmp_path / "frozen",
            expected_sha256=file_sha256(source),
            expected_rows=3,
            confirmatory_size=2,
        )


def test_freeze_official_gsm8k_audits_fixed_autonomous_prompt(tmp_path):
    source, _ = _source(tmp_path)
    train = write_jsonl(
        tmp_path / "prompt-train.jsonl",
        [{"prompt": "Fixed instruction\n\nInput:\nProblem: One plus one?\nASL:"}],
    )
    with pytest.raises(ValueError, match="exact train-question overlaps"):
        freeze_official_gsm8k(
            source_path=source,
            train_paths=[train],
            output_dir=tmp_path / "frozen",
            expected_sha256=file_sha256(source),
            expected_rows=3,
            confirmatory_size=2,
        )


class _Backend:
    model_id = "fake"

    def generate(self, prompt, *, seed):
        del seed
        value = 2 if "One plus one" in prompt else 5 if "Three plus two" in prompt else 12
        text = f"result.value = {value}\nRETURN result.value"
        return SimpleNamespace(
            generated_text=text,
            prompt_tokens=10,
            generated_tokens=5,
            wall_time_ns=1,
            metadata={},
        )


def test_sharded_answer_only_run_and_complete_merge(tmp_path):
    source, train = _source(tmp_path)
    frozen = tmp_path / "frozen"
    freeze_official_gsm8k(
        source_path=source,
        train_paths=[train],
        output_dir=frozen,
        expected_sha256=file_sha256(source),
        expected_rows=3,
        confirmatory_size=2,
    )
    config = {"model": {"model_id": "fake", "revision": "test"}}
    shard_dirs = []
    for shard_index in range(2):
        output = tmp_path / f"shard-{shard_index}"
        shard_dirs.append(output)
        report = run_official_gsm8k_shard(
            eval_path=frozen / "full.jsonl",
            model_config=config,
            adapter_path="adapter",
            adapter_id="test-adapter",
            output_dir=output,
            shard_index=shard_index,
            shard_count=2,
            checkpoint_every=1,
            backend_override=_Backend(),
        )
        assert report["rates"]["final_answer_correct"] == 1.0
    merged = merge_official_gsm8k_shards(
        eval_path=frozen / "full.jsonl",
        shard_dirs=shard_dirs,
        output_dir=tmp_path / "merged",
    )
    assert merged["prediction_count"] == 3
    assert merged["counts"]["final_answer_correct"] == 3
    assert read_json(tmp_path / "merged" / "summary.json") == merged


def test_merge_rejects_incomplete_shards(tmp_path):
    source, train = _source(tmp_path)
    frozen = tmp_path / "frozen"
    freeze_official_gsm8k(
        source_path=source,
        train_paths=[train],
        output_dir=frozen,
        expected_sha256=file_sha256(source),
        expected_rows=3,
        confirmatory_size=2,
    )
    output = tmp_path / "shard"
    run_official_gsm8k_shard(
        eval_path=frozen / "full.jsonl",
        model_config={"model": {"model_id": "fake", "revision": "test"}},
        adapter_path="adapter",
        adapter_id="test-adapter",
        output_dir=output,
        shard_index=0,
        shard_count=2,
        backend_override=_Backend(),
    )
    with pytest.raises(ValueError, match="incomplete shard merge"):
        merge_official_gsm8k_shards(
            eval_path=frozen / "full.jsonl",
            shard_dirs=[output],
            output_dir=tmp_path / "merged",
        )


def test_shard_rejects_concurrent_output_writer(tmp_path):
    source, train = _source(tmp_path)
    frozen = tmp_path / "frozen"
    freeze_official_gsm8k(
        source_path=source,
        train_paths=[train],
        output_dir=frozen,
        expected_sha256=file_sha256(source),
        expected_rows=3,
        confirmatory_size=2,
    )
    output = tmp_path / "locked"
    output.mkdir()
    (output / ".run.lock").write_text('{"pid":123}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="already locked"):
        run_official_gsm8k_shard(
            eval_path=frozen / "full.jsonl",
            model_config={"model": {"model_id": "fake", "revision": "test"}},
            adapter_path="adapter",
            adapter_id="test-adapter",
            output_dir=output,
            shard_index=0,
            shard_count=1,
            backend_override=_Backend(),
        )
