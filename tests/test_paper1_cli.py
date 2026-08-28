import importlib.util
import json

from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl
from ccpu.paper1.cli import _smoke_examples
from ccpu.paper1.dataset import ArithmeticDatasetConfig, iter_dataset


def test_cli_generate_validate_simulate_and_evaluate(tmp_path):
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset.jsonl"
    run_dir = tmp_path / "run"
    recomputed = tmp_path / "recomputed.json"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "seed": 3,
                    "examples_per_cell": 1,
                    "operator_counts": [1],
                    "operand_digits": [1],
                    "operations": ["+", "*"],
                    "control_examples": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(["paper1", "generate", "--config", str(config), "--output", str(dataset)]) == 0
    assert main(["paper1", "validate", "--dataset", str(dataset)]) == 0
    assert (
        main(
            [
                "paper1",
                "simulate",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "paper1",
                "evaluate",
                "--dataset",
                str(dataset),
                "--predictions",
                str(run_dir / "predictions.jsonl"),
                "--output",
                str(recomputed),
            ]
        )
        == 0
    )

    assert len(read_jsonl(dataset)) == 3
    assert len(read_jsonl(run_dir / "predictions.jsonl")) == 21
    assert read_json(run_dir / "summary.json")["empirical"] is False
    assert read_json(run_dir / "manifest.json")["prediction_count"] == 21
    assert read_json(recomputed)["schema_version"] == "ccpu.paper1.evaluation.v1"


def test_xpu_smoke_selection_includes_arithmetic_and_controls():
    examples = list(
        iter_dataset(
            ArithmeticDatasetConfig(
                examples_per_cell=4,
                operator_counts=(1,),
                operand_digits=(1,),
                control_examples=3,
            )
        )
    )

    selected = _smoke_examples(examples)

    assert sum(example.task_kind == "arithmetic" for example in selected) == 4
    assert sum(example.task_kind == "control" for example in selected) == 2


def test_cli_replay_applies_reflex_to_saved_completion(tmp_path):
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset.jsonl"
    completions = tmp_path / "completions.jsonl"
    run_dir = tmp_path / "replay"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "examples_per_cell": 1,
                    "operator_counts": [1],
                    "operand_digits": [1],
                    "control_examples": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    main(["paper1", "generate", "--config", str(config), "--output", str(dataset)])
    example = read_jsonl(dataset)[0]
    completions.write_text(
        json.dumps(
            {
                "example_id": example["example_id"],
                "condition": "reflex",
                "model_id": "saved-model",
                "seed": 11,
                "generated_text": example["reference_completion"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "paper1",
                "replay",
                "--dataset",
                str(dataset),
                "--completions",
                str(completions),
                "--output-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    prediction = read_jsonl(run_dir / "predictions.jsonl")[0]
    assert prediction["interventions"] == 1
    assert prediction["predicted_answer"] == example["answer"]


def test_cli_endpoint_rescore_preserves_reported_labels(tmp_path):
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset.jsonl"
    source = tmp_path / "source.jsonl"
    output = tmp_path / "rescore"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "examples_per_cell": 1,
                    "operator_counts": [1],
                    "operand_digits": [1],
                    "control_examples": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    main(["paper1", "generate", "--config", str(config), "--output", str(dataset)])
    example = read_jsonl(dataset)[0]
    rows = []
    for condition in ("llm_only", "matched_prompt"):
        rows.append(
            {
                "example_id": example["example_id"],
                "model_id": "saved-model",
                "condition": condition,
                "seed": 17,
                "generated_text": f"Response: {example['answer']}",
                "rendered_text": f"Response: {example['answer']}",
                "predicted_answer": None,
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert main(
        [
            "paper1",
            "rescore-endpoints",
            "--dataset",
            str(dataset),
            "--predictions",
            str(source),
            "--output-dir",
            str(output),
        ]
    ) == 0

    rescored = read_jsonl(output / "predictions.jsonl")
    assert all(row["predicted_answer"] is None for row in rescored)
    assert all(row["endpoint_predicted_answer"] == example["answer"] for row in rescored)
    assert read_json(output / "manifest.json")["reported_labels_preserved"] is True


def test_cli_plot_writes_scaling_figure_when_analysis_extra_is_available(tmp_path):
    if importlib.util.find_spec("matplotlib") is None:
        return
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset.jsonl"
    run_dir = tmp_path / "run"
    figure = tmp_path / "scaling.png"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "examples_per_cell": 1,
                    "operator_counts": [1, 2],
                    "operand_digits": [1],
                    "control_examples": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    main(["paper1", "generate", "--config", str(config), "--output", str(dataset)])
    main(
        [
            "paper1",
            "simulate",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(run_dir),
        ]
    )

    assert (
        main(
            [
                "paper1",
                "plot",
                "--summary",
                str(run_dir / "summary.json"),
                "--output",
                str(figure),
            ]
        )
        == 0
    )
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
