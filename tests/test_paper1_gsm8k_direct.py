from types import SimpleNamespace

import pytest

from ccpu.common.artifacts import (
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.paper1.e3.direct_answer_eval import (
    DIRECT_SCORER_ID,
    direct_prompt,
    freeze_direct_gsm8k_protocol,
    merge_direct_gsm8k_shards,
    run_direct_gsm8k_shard,
)
from ccpu.paper1.e3.gsm8k_confirmatory import freeze_official_gsm8k


class _DirectBackend:
    model_id = "fake-direct"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, seed):
        self.prompts.append((prompt, seed))
        value = "2" if "One plus one" in prompt else "5"
        return SimpleNamespace(
            generated_text=f"Short work.\nAnswer: {value}",
            prompt_tokens=12,
            generated_tokens=6,
            wall_time_ns=10,
            metadata={"backend": "fake"},
        )


def _frozen(tmp_path):
    source = write_jsonl(
        tmp_path / "official.jsonl",
        [
            {"question": "One plus one?", "answer": "1 + 1 = <<1+1=2>>2\n#### 2"},
            {"question": "Three plus two?", "answer": "3 + 2 = <<3+2=5>>5\n#### 5"},
        ],
    )
    train = write_jsonl(tmp_path / "train.jsonl", [{"question": "Different."}])
    output = tmp_path / "frozen"
    freeze_official_gsm8k(
        source_path=source,
        train_paths=[train],
        output_dir=output,
        expected_sha256=file_sha256(source),
        expected_rows=2,
        confirmatory_size=2,
    )
    return output / "full.jsonl"


def test_direct_prompt_is_fixed_question_only_contract():
    prompt = direct_prompt("How many remain?", "direct_reasoning")
    assert "How many remain?" in prompt
    assert "Answer: <number>" in prompt
    assert "ASL" in prompt
    with pytest.raises(ValueError, match="unsupported direct condition"):
        direct_prompt("Question", "unknown")


def test_freeze_direct_protocol_pins_matched_provenance(tmp_path):
    eval_path = _frozen(tmp_path)
    paths = []
    for condition, thinking, tokens in (
        ("direct_concise", False, 128),
        ("direct_reasoning", True, 1024),
    ):
        paths.append(
            write_json(
                tmp_path / f"{condition}.json",
                {
                    "condition": condition,
                    "model": {
                        "model_id": "Qwen/Qwen3-0.6B",
                        "revision": "pinned",
                        "device": "xpu",
                        "dtype": "float16",
                        "use_chat_template": True,
                        "enable_thinking": thinking,
                        "max_new_tokens": tokens,
                    },
                },
            )
        )
    manifest = freeze_direct_gsm8k_protocol(
        eval_path=eval_path,
        config_paths=paths,
        output_dir=tmp_path / "protocol",
    )
    assert manifest["identity_count"] == 2
    assert manifest["prompt_fields"] == ["question"]
    assert manifest["conditions"]["direct_reasoning"]["enable_thinking"]
    assert manifest["conditions"]["direct_reasoning"]["max_new_tokens"] == 1024
    assert manifest["answer_scorer"]["policy_id"] == DIRECT_SCORER_ID
    assert manifest["answer_scorer"]["condition_independent"]
    assert read_json(tmp_path / "protocol" / "manifest.json") == manifest


def test_direct_shard_scores_and_records_no_gold_visibility(tmp_path):
    eval_path = _frozen(tmp_path)
    backend = _DirectBackend()
    output = tmp_path / "direct"
    summary = run_direct_gsm8k_shard(
        eval_path=eval_path,
        model_config={
            "model": {
                "model_id": "fake",
                "revision": "test",
                "max_new_tokens": 32,
            }
        },
        condition="direct_reasoning",
        output_dir=output,
        shard_index=0,
        shard_count=1,
        checkpoint_every=1,
        backend_override=backend,
    )
    assert summary["counts"] == {"answer_scorable": 2, "final_answer_correct": 2}
    assert summary["run"]["prompt_fields"] == ["question"]
    assert not summary["run"]["rationales_visible_to_model"]
    assert not summary["run"]["answers_visible_to_model"]
    assert all("####" not in prompt for prompt, _ in backend.prompts)
    rows = read_jsonl(output / "predictions.jsonl")
    assert [row["predicted_answer"] for row in rows] == ["2", "5"]
    assert {row["scorer_id"] for row in rows} == {DIRECT_SCORER_ID}
    assert read_json(output / "summary.json")["predictions_sha256"] == summary[
        "predictions_sha256"
    ]


def test_direct_resume_rejects_different_condition(tmp_path):
    eval_path = _frozen(tmp_path)
    output = tmp_path / "direct"
    config = {
        "model": {"model_id": "fake", "revision": "test", "max_new_tokens": 32}
    }
    run_direct_gsm8k_shard(
        eval_path=eval_path,
        model_config=config,
        condition="direct_concise",
        output_dir=output,
        shard_index=0,
        shard_count=1,
        backend_override=_DirectBackend(),
    )
    with pytest.raises(ValueError, match="resume output does not match"):
        run_direct_gsm8k_shard(
            eval_path=eval_path,
            model_config=config,
            condition="direct_reasoning",
            output_dir=output,
            shard_index=0,
            shard_count=1,
            backend_override=_DirectBackend(),
        )


def test_direct_shards_merge_with_complete_matched_provenance(tmp_path):
    eval_path = _frozen(tmp_path)
    shard_dirs = []
    for shard_index in range(2):
        output = tmp_path / f"shard{shard_index}"
        run_direct_gsm8k_shard(
            eval_path=eval_path,
            model_config={
                "model": {"model_id": "fake", "revision": "test", "max_new_tokens": 32}
            },
            condition="direct_concise",
            output_dir=output,
            shard_index=shard_index,
            shard_count=2,
            backend_override=_DirectBackend(),
        )
        shard_dirs.append(output)

    summary = merge_direct_gsm8k_shards(
        eval_path=eval_path,
        shard_dirs=shard_dirs,
        output_dir=tmp_path / "merged",
    )
    assert summary["prediction_count"] == 2
    assert summary["counts"]["final_answer_correct"] == 2
    assert summary["shard_index"] is None
    assert summary["run"]["shard_count"] == 2
    assert len(summary["run"]["source_prediction_sha256"]) == 2
    assert [row["example_id"] for row in read_jsonl(tmp_path / "merged" / "predictions.jsonl")] == [
        "gsm8k-official-test-0000",
        "gsm8k-official-test-0001",
    ]


def test_direct_merge_rejects_duplicate_shards(tmp_path):
    eval_path = _frozen(tmp_path)
    output = tmp_path / "shard0"
    run_direct_gsm8k_shard(
        eval_path=eval_path,
        model_config={
            "model": {"model_id": "fake", "revision": "test", "max_new_tokens": 32}
        },
        condition="direct_concise",
        output_dir=output,
        shard_index=0,
        shard_count=2,
        backend_override=_DirectBackend(),
    )
    with pytest.raises(ValueError, match="incomplete direct shard merge"):
        merge_direct_gsm8k_shards(
            eval_path=eval_path,
            shard_dirs=[output, output],
            output_dir=tmp_path / "merged",
        )


def test_direct_shard_rejects_concurrent_writer(tmp_path):
    eval_path = _frozen(tmp_path)
    output = tmp_path / "direct"
    output.mkdir()
    (output / ".run.lock").write_text('{"pid":123}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="already locked"):
        run_direct_gsm8k_shard(
            eval_path=eval_path,
            model_config={
                "model": {
                    "model_id": "fake",
                    "revision": "test",
                    "max_new_tokens": 32,
                }
            },
            condition="direct_concise",
            output_dir=output,
            shard_index=0,
            shard_count=1,
            backend_override=_DirectBackend(),
        )
