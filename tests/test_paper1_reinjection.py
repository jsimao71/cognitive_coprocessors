from types import SimpleNamespace

from ccpu.common.artifacts import read_jsonl, write_jsonl
from ccpu.paper1.e3.reinjection import reinjection_prompt, run_reinjection


class _Backend:
    model_id = "fake"

    def generate(self, prompt, *, seed):
        del seed
        value = prompt.split("VALUE: ", 1)[1].splitlines()[0]
        return SimpleNamespace(
            generated_text=f"Answer: {value}",
            prompt_tokens=20,
            generated_tokens=4,
            wall_time_ns=10,
            metadata={"backend": "fake"},
        )


def test_reinjection_keeps_runtime_truth_and_value_uptake_separate(tmp_path):
    evaluation = write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {
                "example_id": "a",
                "parent_example_id": "pa",
                "question": "One plus one?",
                "question_sha256": "ha",
                "reference_return": "2",
            },
            {
                "example_id": "b",
                "parent_example_id": "pb",
                "question": "Two plus two?",
                "question_sha256": "hb",
                "reference_return": "4",
            },
        ],
    )
    source = write_jsonl(
        tmp_path / "asl.jsonl",
        [
            {
                "example_id": "a",
                "question_sha256": "ha",
                "predicted_asl": "RETURN 2",
                "metrics": {"predicted_return": "2"},
            },
            {
                "example_id": "b",
                "question_sha256": "hb",
                "predicted_asl": "RETURN 5",
                "metrics": {"predicted_return": "5"},
            },
        ],
    )
    report = run_reinjection(
        eval_path=evaluation,
        asl_predictions_path=source,
        model_config={"model": {"model_id": "fake", "revision": "test", "max_new_tokens": 16}},
        condition="typed_result_text",
        output_dir=tmp_path / "out",
        backend_override=_Backend(),
    )
    rows = read_jsonl(tmp_path / "out" / "predictions.jsonl")
    assert report["counts"]["runtime_value_correct"] == 1
    assert report["counts"]["final_answer_correct"] == 1
    assert report["counts"]["value_uptake"] == 2
    assert report["counts"]["wrong_value_uptake"] == 1
    assert rows[1]["metrics"]["wrong_value_uptake"] is True


def test_reinjection_prompt_marks_oracle_and_corrupted_controls():
    oracle, value, grounding = reinjection_prompt(
        question="Question", condition="oracle_value", runtime_value="3/2", predicted_asl=""
    )
    assert value == "3/2"
    assert grounding == "ORACLE_CONTROL"
    assert "SEMANTIC_GROUNDING: ORACLE_CONTROL" in oracle
