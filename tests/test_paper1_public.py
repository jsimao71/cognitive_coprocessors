from types import SimpleNamespace

from ccpu.common.artifacts import canonical_json, write_jsonl
from ccpu.common.schema import GenerationResult
from ccpu.paper1.public_gsm8k import (
    freeze_gsm8k_slice,
    run_gsm8k_example,
    summarize_gsm8k,
)


class FakeBackend:
    model_id = "fake-model"

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, prompt, *, controller=None, seed=0):
        del prompt, seed
        generated = next(self.outputs)
        rendered = controller.feed(generated).rendered_text if controller else generated
        return GenerationResult(
            generated_text=generated,
            rendered_text=rendered,
            prompt_tokens=10,
            generated_tokens=5,
            reinjected_tokens=1 if controller and controller.state else 0,
            model_calls=1,
            wall_time_ns=100,
            metadata={"device": "test"},
        )


def _example():
    return {
        "example_id": "gsm8k:test:1",
        "content_sha256": "a" * 64,
        "difficulty": 2,
        "difficulty_stratum": "2_steps",
        "target_label": "12",
        "question": "There are 3 groups of 4. How many?",
        "opportunities": [{"expression": "3 * 4", "result": "12"}],
    }


def test_generic_compute_executes_through_gateway_and_returns_final_answer():
    backend = FakeBackend(
        ['__compute({"expression":"3 * 4"})', "Answer: 12"]
    )
    row = run_gsm8k_example(
        _example(), backend, condition="generic_compute", seed=1, max_assistance_episodes=2
    )
    assert row["correct"] is True
    assert row["assistance_calls"] == 1
    assert row["matched_gold_operations"] == 1
    assert row["calls"][0]["result"] == "12"


def test_runtime_block_records_trace_and_automatic_rescue():
    runtime = run_gsm8k_example(
        _example(),
        FakeBackend(["```calculator\n3 * 4\n```\nAnswer: 12"]),
        condition="calculator_block",
        seed=1,
    )
    assert runtime["assistance_valid"] is True
    assert runtime["correct"] is True
    assert any(event["stage"] == "execution" for event in runtime["runtime_trace"])

    generic_miss = {
        **runtime,
        "condition": "generic_compute",
        "assistance_valid": False,
        "correct": False,
    }
    automatic = {**runtime, "condition": "runtime_trigger"}
    summary = summarize_gsm8k([generic_miss, automatic])
    assert summary["automatic_rescue"]["eligible_voluntary_misses"] == 1
    assert summary["automatic_rescue"]["rescued_correctly"] == 1
    assert summary["automatic_rescue"]["rate"] == 1.0
    assert summary["paired_vs_llm_only"] == {}


def test_oracle_ledger_is_executed_by_the_bounded_calculator():
    example = _example()
    example["opportunities"][0]["result"] = "999"
    backend = FakeBackend(["Answer: 12"])
    row = run_gsm8k_example(example, backend, condition="oracle_calculator", seed=1)

    assert row["correct"] is True
    assert row["assistance_valid"] is True
    assert row["matched_gold_operations"] == 1
    assert row["calls"] == [
        {"expression": "3 * 4", "canonical_expression": "3*4", "result": "12"}
    ]


def test_materializer_verifies_frozen_content(monkeypatch, tmp_path):
    import hashlib

    from ccpu.common import gsm8k

    raw = {
        "question": "There are 3 groups of 4. How many?",
        "answer": "Multiply. <<3*4=12>> #### 12",
    }
    source = SimpleNamespace(benchmark="gsm8k")
    monkeypatch.setattr(gsm8k, "load_config", lambda path: (1, [source], {}))
    monkeypatch.setattr(gsm8k, "read_verified_parquet", lambda source, cache: [raw])
    content_sha = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    selection = write_jsonl(
        tmp_path / "selection.jsonl",
        [
            {
                "benchmark": "gsm8k",
                "source_row": 0,
                "example_id": "gsm8k:test:0",
                "content_sha256": content_sha,
                "target_label": "12",
                "difficulty": 1,
                "difficulty_stratum": "2_steps",
            }
        ],
    )
    rows = gsm8k.materialize_gsm8k("config.json", tmp_path, selection)
    assert rows[0]["question"] == raw["question"]
    assert rows[0]["opportunities"] == [{"expression": "3*4", "result": "12"}]


def test_balanced_gsm8k_slice_is_frozen_by_existing_selection_key(tmp_path):
    source = write_jsonl(
        tmp_path / "source.jsonl",
        [
            {
                "benchmark": "gsm8k",
                "example_id": f"{stratum}-{index}",
                "difficulty_stratum": stratum,
                "selection_key": str(index).zfill(64),
            }
            for stratum in ("2_steps", "3_4_steps", "5plus_steps")
            for index in range(3)
        ],
    )
    manifest = freeze_gsm8k_slice(source, tmp_path / "frozen", per_stratum=2)
    assert manifest["record_count"] == 6
    assert manifest["counts"] == {"2_steps": 2, "3_4_steps": 2, "5plus_steps": 2}
