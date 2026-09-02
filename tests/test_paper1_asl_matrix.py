from dataclasses import asdict, replace

import pytest

from ccpu.cli import build_parser
from ccpu.common.artifacts import read_json, read_jsonl, write_json, write_jsonl
from ccpu.paper1.asl_matrix.data import (
    CORRUPTION_POLICIES,
    MatrixExample,
    RegimeBuilder,
    StaticMixture,
    assert_view_is_leakage_safe,
    canonicalize_asl,
    corrupt_asl,
)
from ccpu.paper1.asl_matrix.eval import analyze_matrix_runs
from ccpu.paper1.asl_matrix.qwen import build_qwen_patch_data
from ccpu.paper1.asl_matrix.train import MatrixTrainingConfig

ASL = """box.initial_count = 12
box.removed_count = 4
box.remaining_count = box.initial_count - box.removed_count
RETURN box.remaining_count"""


def _example() -> MatrixExample:
    return MatrixExample(
        example_id="matrix-train-one",
        dataset="gsm8k",
        parent_source_id="one",
        semantic_pattern_id="pattern-one",
        nl_input="A box has 12 items and four are removed. How many remain?",
        target_asl=ASL,
        effective_scope={
            "id": "gsm8k:matrix:one",
            "parent": None,
            "kind": "benchmark_case",
            "source": "dataset",
        },
        split="train",
    )


def test_canonical_reference_executes_and_is_stable():
    example = _example()
    first = canonicalize_asl(example.target_asl, example.effective_scope)
    second = canonicalize_asl(example.target_asl + "\n", example.effective_scope)
    assert first == second
    assert first["record_count"] == 4


@pytest.mark.parametrize("policy", CORRUPTION_POLICIES)
def test_every_corruption_is_deterministic_and_audited(policy):
    kwargs = {
        "policy": policy,
        "severity": 0.5,
        "seed": 17,
        "noise_asl": "noise.unrelated_count = 99\nRETURN noise.unrelated_count",
    }
    first, metadata = corrupt_asl(ASL, **kwargs)
    second, repeated = corrupt_asl(ASL, **kwargs)
    assert first == second
    assert metadata == repeated
    assert first != ASL
    assert metadata["policy"] == policy
    assert metadata["selected"]
    assert 0 <= metadata["external_asl_fraction"] <= 1


def test_autonomous_view_exposes_only_natural_language():
    view = RegimeBuilder(seed=3).make_view(_example(), regime="autonomous")
    assert view["external_asl_input"] is None
    assert view["has_external_asl"] is False
    assert view["external_asl_fraction"] == 0
    assert view["source_fields_visible_to_model"] == ["nl_input"]
    assert "target_asl" not in view["source_fields_visible_to_model"]


def test_leakage_assertion_rejects_external_asl_in_autonomous_view():
    view = RegimeBuilder(seed=3).make_view(_example(), regime="autonomous")
    view["external_asl_input"] = ASL
    view["has_external_asl"] = True
    with pytest.raises(AssertionError, match="cannot contain external ASL"):
        assert_view_is_leakage_safe(view)


def test_full_and_partial_views_preserve_target_but_change_teacher():
    builder = RegimeBuilder(
        mixture=StaticMixture(full_teacher=0.2, partial_teacher=0.4, autonomous=0.4),
        corruption_policies=("value_mask",),
        seed=9,
    )
    full = builder.make_view(_example(), regime="full")
    partial = builder.make_view(_example(), regime="partial")
    assert full["external_asl_input"] == ASL
    assert partial["external_asl_input"] != ASL
    assert full["target_asl"] == partial["target_asl"] == ASL
    assert partial["external_asl_corruption"]["policy"] == "value_mask"


def test_runtime_regime_sampling_is_epoch_deterministic():
    builder = RegimeBuilder(seed=21)
    example = _example()
    assert builder.sample_regime(example, epoch=2) == builder.sample_regime(example, epoch=2)
    renamed = replace(example, example_id="matrix-train-two")
    assert builder.sample_regime(renamed, epoch=2) in {"full", "partial", "autonomous"}


def test_cli_exposes_matrix_data_gate():
    args = build_parser().parse_args(
        [
            "paper1",
            "prepare-asl-matrix-data",
            "--train",
            "train.jsonl",
            "--dev",
            "dev.jsonl",
            "--test",
            "test.jsonl",
            "--output-dir",
            "matrix",
        ]
    )
    assert args.seed == 912736


def test_training_config_rejects_invalid_optimizer():
    config = MatrixTrainingConfig(learning_rate=0)
    with pytest.raises(ValueError, match="optimizer"):
        config.validate()


def test_matrix_analysis_computes_seed_matched_grounding_gain(tmp_path):
    def summary(run_id, answer):
        rates = {
            "exact_asl": 0.0,
            "parse_valid": 1.0,
            "semantic_return_equivalent": answer,
            "final_answer_correct": answer,
        }
        return {
            "run_id": run_id,
            "seed": 11,
            "encoder_architecture": "separate",
            "attention_mode": "cross",
            "training_mixture": {},
            "by_condition": {
                "autonomous": {"rates": rates},
                "full_teacher": {"rates": rates},
            },
            "teacher_gap": {
                "semantic_return_equivalent": 0.0,
                "final_answer_correct": 0.0,
            },
        }

    b0 = write_json(tmp_path / "b0.json", summary("B0-separate-cross-t3", 0.2))
    b1 = write_json(tmp_path / "b1.json", summary("B1-separate-cross-mixed", 0.4))
    report = analyze_matrix_runs([b0, b1], tmp_path / "analysis")
    assert report["aggregates"]["B1-separate-cross-mixed"][
        "grounding_gain_semantic_mean"
    ] == pytest.approx(0.2)
    assert read_json(tmp_path / "analysis" / "summary.json") == report


@pytest.mark.parametrize(
    ("command", "required"),
    [
        (
            "train-asl-matrix",
            ["--config", "c.json", "--data-dir", "d", "--output-dir", "o"],
        ),
        (
            "evaluate-asl-matrix",
            [
                "--config",
                "c.json",
                "--data-dir",
                "d",
                "--checkpoint",
                "best.pt",
                "--output-dir",
                "o",
            ],
        ),
        ("analyze-asl-matrix", ["--summary", "s.json", "--output-dir", "o"]),
    ],
)
def test_cli_exposes_matrix_run_commands(command, required):
    args = build_parser().parse_args(["paper1", command, *required])
    assert callable(args.handler)


def test_qwen_q0_unfolds_matched_epochs_without_teacher_leakage(tmp_path):
    data = tmp_path / "data" / "source"
    train = [_example(), replace(_example(), example_id="two", parent_source_id="two")]
    dev = [replace(_example(), example_id="dev", parent_source_id="dev", split="dev")]
    write_jsonl(data / "train.jsonl", [asdict(row) for row in train])
    write_jsonl(data / "dev.jsonl", [asdict(row) for row in dev])
    report = build_qwen_patch_data(
        tmp_path / "data", tmp_path / "q0", condition="q0_t3", epochs=3, seed=11
    )
    rows = read_jsonl(tmp_path / "q0" / "train.jsonl")
    assert report["train_views"] == 6
    assert report["regime_counts"] == {"autonomous": 6}
    assert all(not row["has_external_asl"] for row in rows)
    assert all("External ASL teacher:" not in row["prompt"] for row in rows)
