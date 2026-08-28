import importlib.util
import json

from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl
from ccpu.paper1.lora_data import LoRAProtocolDataConfig, generate_protocol_data
from ccpu.paper1.placement_analysis import build_placement_comparison


def _excluded_dataset(tmp_path):
    path = tmp_path / "heldout.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "heldout-1",
                "task_kind": "arithmetic",
                "expression": "123 + 456",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_protocol_data_is_deterministic_balanced_and_leakage_audited(tmp_path):
    config = LoRAProtocolDataConfig(
        seed=9,
        train_arithmetic=8,
        train_controls=8,
        dev_arithmetic=4,
        dev_controls=4,
    )
    excluded = _excluded_dataset(tmp_path)

    first, audit = generate_protocol_data(config, excluded_dataset=excluded)
    second, second_audit = generate_protocol_data(config, excluded_dataset=excluded)

    assert first == second
    assert audit == second_audit
    assert audit["passed"]
    assert len(first["train"]) == 16
    assert len(first["dev"]) == 8
    assert {row["task_kind"] for row in first["train"]} == {"arithmetic", "control"}
    arithmetic = next(row for row in first["train"] if row["task_kind"] == "arithmetic")
    control = next(row for row in first["train"] if row["task_kind"] == "control")
    assert arithmetic["target"].startswith("```calculator\n")
    assert "```calculator" not in control["target"]
    assert not ({123, 456} & set(arithmetic["operands"]))


def test_cli_generates_versioned_protocol_data(tmp_path):
    config = tmp_path / "config.json"
    output = tmp_path / "protocol"
    excluded = _excluded_dataset(tmp_path)
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "seed": 7,
                    "train_arithmetic": 4,
                    "train_controls": 4,
                    "dev_arithmetic": 2,
                    "dev_controls": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "paper1",
                "generate-lora-data",
                "--config",
                str(config),
                "--excluded-dataset",
                str(excluded),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert len(read_jsonl(output / "train.jsonl")) == 8
    assert read_json(output / "leakage_audit.json")["passed"]
    assert read_json(output / "manifest.json")["train_rows"] == 8


def test_placement_analysis_writes_hashed_rows_and_figure(tmp_path):
    if importlib.util.find_spec("matplotlib") is None:
        return
    summary = tmp_path / "summary.json"
    training = tmp_path / "training.json"
    config = tmp_path / "config.json"
    output = tmp_path / "output"
    summary.write_text(
        json.dumps(
            {
                "by_run": [
                    {
                        "model_id": "test/model",
                        "condition": "calculator_block_minimal",
                        "accuracy": 1.0,
                        "arithmetic_count": 4,
                        "block_execution_rate": 1.0,
                        "false_block_rate": 0.0,
                        "result_use_rate": 1.0,
                        "mean_prompt_tokens": 20.0,
                        "mean_generated_tokens": 5.0,
                        "mean_wall_time_ms": 10.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    training.write_text(
        json.dumps(
            {
                "adapter_id": "test/adapter",
                "trainable_parameters": 10,
                "trainable_fraction": 0.01,
                "training_target_tokens": 100,
                "wall_time_seconds": 2.0,
                "peak_memory_bytes": 1000,
            }
        ),
        encoding="utf-8",
    )
    config.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "model_label": "Test",
                        "placement": "weights",
                        "condition": "calculator_block_minimal",
                        "summary": "summary.json",
                    }
                ],
                "training_reports": [
                    {"model_label": "Test", "report": "training.json"}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_placement_comparison(
        read_json(config), config_path=config, output_dir=output
    )

    assert result["rows"][0]["interface_success_rate"] == 1.0
    assert result["training"][0]["trainable_parameters"] == 10
    assert (output / "placement_reliability_cost.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
