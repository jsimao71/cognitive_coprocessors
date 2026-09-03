import importlib.util
import json

import pytest

from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl
from ccpu.paper1.lora_data import LoRAProtocolDataConfig, generate_protocol_data
from ccpu.paper1.lora_train import (
    LoRATrainingConfig,
    _model_loss,
    _tokenize_record,
    pairwise_rank_terms,
    semantic_weight_spans,
)
from ccpu.paper1.placement_analysis import build_placement_comparison


class _CharacterChatTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        prefix = f"USER: {messages[0]['content']}\nASSISTANT: "
        if add_generation_prompt:
            return prefix
        return prefix + messages[1]["content"] + "<eos>"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_lora_training_config_parses_restart_and_truncation_guards():
    config = LoRATrainingConfig.from_dict(
        {
            "training": {
                "checkpoint_every_optimizer_steps": 50,
                "reject_truncation": True,
                "evaluate_each_epoch": True,
                "restore_best_dev": True,
                "save_epoch_adapters": True,
                "logical_epoch_field": "epoch_view",
            }
        }
    )

    assert config.checkpoint_every_optimizer_steps == 50
    assert config.reject_truncation is True
    assert config.restore_best_dev is True
    assert config.save_epoch_adapters is True
    assert config.logical_epoch_field == "epoch_view"


def test_best_dev_restoration_requires_epoch_evaluation():
    with pytest.raises(ValueError, match="requires evaluation after every epoch"):
        LoRATrainingConfig(
            evaluate_each_epoch=False,
            restore_best_dev=True,
        ).validate()


def test_lora_tokenization_can_reject_instead_of_hiding_truncation():
    tokenizer = _CharacterChatTokenizer()
    row = {"example_id": "long-1", "prompt": "short", "target": "x" * 80}

    truncated = _tokenize_record(tokenizer, row, 40)
    assert truncated["was_truncated"] is True
    assert truncated["full_tokens"] > 40

    with pytest.raises(ValueError, match="record exceeds max_length"):
        _tokenize_record(tokenizer, row, 40, reject_truncation=True)


def test_semantic_weight_spans_prioritize_binding_and_query_tokens():
    weights = {"default": 0.35, "path": 4.0, "operator": 3.0, "literal": 2.0, "return": 5.0}
    target = "box.remaining = box.initial - 4\nRETURN box.remaining"
    spans = semantic_weight_spans(target, weights)
    observed = {(target[start:end], weight) for start, end, weight in spans}
    assert ("box.remaining", 4.0) in observed
    assert ("box.initial", 4.0) in observed
    assert ("-", 3.0) in observed
    assert ("4", 2.0) in observed
    assert ("RETURN box.remaining", 5.0) in observed


def test_semantic_weight_spans_understand_f4_json_decisions():
    weights = {"default": 0.35, "path": 4.0, "operator": 3.0, "literal": 2.0, "return": 5.0}
    target = json.dumps(
        {
            "bindings": [
                {"path": "john.green_removed", "slot": "s0"},
                {"path": "john.pink_removed", "slot": "s1"},
            ],
            "schema_version": "ccpu.paper1.semantic_bottleneck.v1",
            "steps": [
                {
                    "expression": {
                        "arguments": [
                            {"kind": "ref", "slot": "s1"},
                            {"kind": "literal", "literal_type": "number", "value": 2},
                        ],
                        "kind": "apply",
                        "operator": "MUL",
                    },
                    "kind": "set",
                    "target": "s0",
                },
                {"expression": {"kind": "ref", "slot": "s0"}, "kind": "return"},
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    spans = semantic_weight_spans(target, weights)
    observed = {(target[start:end], weight) for start, end, weight in spans}
    assert ('"john.green_removed"', 4.0) in observed
    assert ('"s1"', 4.0) in observed
    assert ('"MUL"', 3.0) in observed
    assert ("2", 2.0) in observed
    assert ('"return"', 5.0) in observed
    assert ('"ccpu.paper1.semantic_bottleneck.v1"', 4.0) not in observed


def test_semantic_weight_config_is_explicit_and_positive():
    config = LoRATrainingConfig(
        semantic_token_weights={
            "default": 0.35,
            "path": 4.0,
            "operator": 3.0,
            "literal": 2.0,
            "return": 5.0,
        }
    )
    config.validate()
    with pytest.raises(ValueError, match="exactly"):
        LoRATrainingConfig(semantic_token_weights={"path": 4.0}).validate()


def test_weighted_causal_loss_emphasizes_high_weight_errors():
    torch = pytest.importorskip("torch")

    class Model:
        def __call__(self, **_kwargs):
            # Position zero predicts label zero correctly; position one predicts label one poorly.
            logits = torch.tensor(
                [[[4.0, 0.0], [4.0, 0.0], [0.0, 0.0]]], requires_grad=True
            )
            return type("Output", (), {"logits": logits})()

    batch = {
        "input_ids": torch.tensor([[0, 0, 0]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 0, 1]]),
        "loss_weights": torch.tensor([[0.0, 1.0, 10.0]]),
    }
    weighted, ordinary = _model_loss(Model(), torch, batch)
    assert weighted > ordinary
    weighted.backward()


def test_pairwise_rank_terms_match_direct_logistic_gradient():
    torch = pytest.importorskip("torch")
    positive = torch.tensor(-0.7, requires_grad=True)
    negative = torch.tensor(-0.2, requires_grad=True)
    direct = torch.nn.functional.softplus(negative - positive)
    direct.backward()
    expected = (positive.grad.item(), negative.grad.item())

    positive = torch.tensor(-0.7, requires_grad=True)
    negative = torch.tensor(-0.2, requires_grad=True)
    loss, coefficient = pairwise_rank_terms(torch, positive, negative, 1.0)
    (-coefficient * positive + coefficient * negative).backward()
    assert (positive.grad.item(), negative.grad.item()) == pytest.approx(expected)
    assert loss.item() == pytest.approx(direct.item())


def test_pairwise_tokenization_preserves_prompt_and_adds_negative_view():
    tokenizer = _CharacterChatTokenizer()
    weights = {"default": 0.35, "path": 4.0, "operator": 3.0, "literal": 2.0, "return": 5.0}
    row = {
        "example_id": "pair-1",
        "prompt": "Two values are related.",
        "target": "john.green = john.pink * 2\nRETURN john.green",
        "negative_target": "john.green = carl.pink * 2\nRETURN john.green",
        "negative_type": "dependency_rebind",
    }
    tokenized = _tokenize_record(
        tokenizer, row, 256, reject_truncation=True, semantic_token_weights=weights
    )
    assert tokenized["negative_type"] == "dependency_rebind"
    assert tokenized["negative"]["target_tokens"] > 0
    assert tokenized["prefix_tokens"] == tokenized["negative"]["prefix_tokens"]


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
