from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_jsonl
from ccpu.paper1.e3.cli import main
from ccpu.paper1.e3.compositional_generator import (
    TIER_CONTROLS,
    build_compositional_curriculum,
    freeze_compositional_pilots,
    generate_compositional_record,
)
from ccpu.paper1.e3.direct_answer_eval import run_direct_gsm8k_shard
from ccpu.paper1.e3.gsm8k_confirmatory import run_official_gsm8k_shard


class _StaticBackend:
    model_id = "static-compositional-test"

    def __init__(self, generated_text: str) -> None:
        self.generated_text = generated_text

    def generate(self, prompt: str, *, seed: int):
        del prompt, seed
        return SimpleNamespace(
            generated_text=self.generated_text,
            prompt_tokens=10,
            generated_tokens=5,
            wall_time_ns=1,
            metadata={"backend": "static"},
        )


@pytest.mark.parametrize("tier", ("C1", "C2", "C3", "C4"))
def test_records_are_deterministic_executable_and_self_describing(tier: str) -> None:
    first = generate_compositional_record(tier=tier, split="test", index=2, seed=7101)
    second = generate_compositional_record(tier=tier, split="test", index=2, seed=7101)

    assert first == second
    assert first["question_hash"] == first["question_sha256"]
    assert first["generation_seed"] == 7101
    assert first["record_seed"] > 0
    assert first["source_row"] == 2
    assert first["difficulty_stratum"] == tier.lower()
    assert first["reference_return"] == first["reference_answer"]
    assert first["target"] == first["gold_asl"]
    assert first["prompt"].endswith(f"Problem: {first['question']}\nASL:")
    assert first["lineage"]["tier_id"] == tier
    assert first["gold_asl"].endswith(
        f"RETURN {first['typed_symbolic_graph']['answer_path']}"
    )
    assert first["validation_status"]["valid"] is True
    assert first["validation_status"]["execution_verified"] is True
    assert first["validation_status"]["answer_verified"] is True
    assert first["validation_status"]["returned"] == first["reference_answer"]
    assert all(
        fact["value_type"]
        for fact in first["typed_symbolic_graph"]["facts"]
    )
    assert all(
        operation["value_type"]
        for operation in first["typed_symbolic_graph"]["operations"]
    )


def test_frozen_tier_complexity_definitions() -> None:
    c1 = generate_compositional_record(tier="C1", split="train", index=0, seed=11)
    c2 = generate_compositional_record(tier="C2", split="train", index=0, seed=11)
    c3_rows = [
        generate_compositional_record(tier="C3", split="train", index=index, seed=11)
        for index in range(2)
    ]
    c4_rows = [
        generate_compositional_record(tier="C4", split="train", index=index, seed=11)
        for index in range(3)
    ]

    c1_complexity = c1["complexity"]["observed"]
    assert c1_complexity["operator_count"] == 1
    assert c1_complexity["dependency_depth"] == 1
    assert c1_complexity["distractor_count"] == 0

    c2_complexity = c2["complexity"]["observed"]
    assert c2_complexity["operator_count"] == 2
    assert c2_complexity["dependency_depth"] == 2
    assert c2_complexity["entity_count"] >= 2
    assert c2_complexity["binding_count"] >= 2

    assert {row["complexity"]["observed"]["operator_count"] for row in c3_rows} == {3, 4}
    for row in c3_rows:
        observed = row["complexity"]["observed"]
        assert observed["graph_structure"] == "branch_merge"
        assert observed["distractor_count"] == 1
        assert len(observed["disconnected_distractor_paths"]) == 1

    assert {row["lineage"]["graph_schema_id"] for row in c4_rows} == {
        "cube_from_face_area",
        "square_from_linear_side",
        "square_from_quadratic_side",
    }
    for row in c4_rows:
        observed = row["complexity"]["observed"]
        assert observed["operator_count"] == 4
        assert observed["dependency_depth"] >= 4
        assert observed["function_nesting_depth"] == 2
        assert observed["reference_count"] >= 5
        assert observed["distractor_count"] == 2
        assert "O0" in observed["operator_families"]
        assert {"O1", "O5", "O6"} & set(observed["operator_families"])


def test_freeze_is_byte_reproducible_and_templates_are_disjoint(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = freeze_compositional_pilots(
        first, tiers=("C1", "C4"), train_count=6, dev_count=3, test_count=3, seed=29
    )
    second_manifest = freeze_compositional_pilots(
        second, tiers=("C1", "C4"), train_count=6, dev_count=3, test_count=3, seed=29
    )

    assert first_manifest == second_manifest
    for tier in ("C1", "C4"):
        for split in ("train", "dev", "test"):
            assert file_sha256(first / tier / f"{split}.jsonl") == file_sha256(
                second / tier / f"{split}.jsonl"
            )
        manifest = read_json(first / tier / "manifest.json")
        assert manifest["train_test_template_families_disjoint"] is True
        assert set(manifest["template_families"]["train"]).isdisjoint(
            manifest["template_families"]["test"]
        )
        assert all(
            result["passed"] == result["total"] == 12
            for result in manifest["validation_counts"].values()
        )


def test_freeze_retries_numeric_collisions_without_duplicate_questions(tmp_path: Path) -> None:
    freeze_compositional_pilots(
        tmp_path, tiers=("C1",), train_count=250, dev_count=10, test_count=10, seed=124001
    )
    rows = read_jsonl(tmp_path / "C1" / "train.jsonl")

    assert len({row["question_sha256"] for row in rows}) == len(rows)
    assert any(row["generation_retry"] > 0 for row in rows)


def test_seed_changes_values_without_changing_frozen_structure() -> None:
    first = generate_compositional_record(tier="C4", split="test", index=1, seed=101)
    second = generate_compositional_record(tier="C4", split="test", index=1, seed=102)

    assert first["question_hash"] != second["question_hash"]
    assert first["lineage"]["graph_schema_id"] == second["lineage"]["graph_schema_id"]
    assert first["complexity"] == second["complexity"]


def test_cli_freezes_repeatable_named_tiers(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    result = main(
        [
            "prepare-compositional-pilots",
            "--tier",
            "C2",
            "--tier",
            "C3",
            "--train-count",
            "4",
            "--dev-count",
            "2",
            "--test-count",
            "2",
            "--seed",
            "83",
            "--output-dir",
            str(output),
        ]
    )

    assert result == 0
    assert read_json(output / "manifest.json")["tiers"] == ["C2", "C3"]
    assert len(read_jsonl(output / "C2" / "train.jsonl")) == 4
    assert len(read_jsonl(output / "C3" / "test.jsonl")) == 2


def test_cli_all_expands_the_complete_ladder(tmp_path: Path) -> None:
    output = tmp_path / "all"
    result = main(
        [
            "prepare-compositional-pilots",
            "--tier",
            "all",
            "--train-count",
            "1",
            "--dev-count",
            "1",
            "--test-count",
            "1",
            "--output-dir",
            str(output),
        ]
    )

    assert result == 0
    assert read_json(output / "manifest.json")["tiers"] == ["C1", "C2", "C3", "C4"]
    assert all((output / tier / "test.jsonl").is_file() for tier in ("C1", "C2", "C3", "C4"))


def test_curriculum_combines_only_train_and_dev_without_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "curriculum"
    freeze_compositional_pilots(
        source, tiers=("C1", "C3"), train_count=4, dev_count=2, test_count=3, seed=91
    )

    manifest = build_compositional_curriculum(
        source, output, tiers=("C1", "C3"), seed=97
    )

    assert manifest["counts"] == {"train": 8, "dev": 4}
    assert manifest["counts_by_tier"]["train"] == {"C1": 4, "C3": 4}
    assert manifest["test_rows_included"] is False
    assert not (output / "test.jsonl").exists()
    train = read_jsonl(output / "train.jsonl")
    dev = read_jsonl(output / "dev.jsonl")
    assert {row["split"] for row in train} == {"train"}
    assert {row["split"] for row in dev} == {"dev"}
    assert {row["example_id"] for row in train}.isdisjoint(
        row["example_id"] for row in dev
    )


def test_cli_builds_compositional_curriculum(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "curriculum"
    freeze_compositional_pilots(
        source, tiers=("C2", "C4"), train_count=3, dev_count=2, test_count=1, seed=101
    )

    result = main(
        [
            "prepare-compositional-curriculum",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--tier",
            "C2",
            "--tier",
            "C4",
            "--seed",
            "103",
        ]
    )

    assert result == 0
    assert read_json(output / "manifest.json")["tiers"] == ["C2", "C4"]
    assert len(read_jsonl(output / "train.jsonl")) == 6


def test_frozen_row_runs_through_existing_direct_and_asl_evaluators(tmp_path: Path) -> None:
    row = generate_compositional_record(tier="C4", split="test", index=1, seed=124001)
    evaluation = write_jsonl(tmp_path / "test.jsonl", [row])
    config = {"model": {"model_id": "fake", "revision": "test", "max_new_tokens": 32}}

    direct = run_direct_gsm8k_shard(
        eval_path=evaluation,
        model_config=config,
        condition="direct_reasoning",
        output_dir=tmp_path / "direct",
        shard_index=0,
        shard_count=1,
        checkpoint_every=1,
        backend_override=_StaticBackend(f"Answer: {row['reference_return']}"),
    )
    asl = run_official_gsm8k_shard(
        eval_path=evaluation,
        model_config=config,
        adapter_path="unused-with-static-backend",
        adapter_id="static-test-adapter",
        output_dir=tmp_path / "asl",
        shard_index=0,
        shard_count=1,
        checkpoint_every=1,
        backend_override=_StaticBackend(row["gold_asl"]),
    )

    assert direct["counts"]["final_answer_correct"] == 1
    assert asl["counts"]["final_answer_correct"] == 1


def test_tier_registry_is_ascending() -> None:
    assert TIER_CONTROLS["C1"].operator_count == (1, 1)
    assert TIER_CONTROLS["C2"].dependency_depth == (2, 2)
    assert TIER_CONTROLS["C3"].graph_structure == "branch_merge"
    assert TIER_CONTROLS["C4"].function_nesting_depth == (2, 2)


def test_freeze_rejects_nonpositive_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split counts"):
        freeze_compositional_pilots(tmp_path, train_count=0)
