from copy import deepcopy
from types import SimpleNamespace

import pytest

from ccpu.common.artifacts import read_jsonl, write_jsonl
from ccpu.dsl import validate_asl
from ccpu.dsl_dataset.remote_analysis import _program_metrics
from ccpu.paper1.asl_pilot_eval import score_asl
from ccpu.paper1.e3.bottleneck import (
    asl_to_bottleneck,
    lower_bottleneck_to_asl,
    parse_bottleneck,
    render_bottleneck,
)
from ccpu.paper1.e3.components import component_labels
from ccpu.paper1.e3.data import build_bottleneck_preference_data
from ccpu.paper1.e3.data_scale import build_d1_f0_data, build_gsm8k_f0_data
from ccpu.paper1.e3.eval import (
    extract_bottleneck,
    run_bottleneck_condition,
    score_bottleneck,
)
from ccpu.paper1.e3.negatives import generate_hard_negatives
from ccpu.paper1.e3.selection import select_semantic_checkpoint

ASL = """box.initial_count = 12
box.removed_count = 4
box.remaining_count = box.initial_count - box.removed_count
RETURN box.remaining_count"""
SCOPE = {
    "id": "gsm8k:e3:test",
    "parent": None,
    "kind": "benchmark_case",
    "source": "dataset",
}


def test_bottleneck_roundtrip_preserves_runtime_semantics():
    program = asl_to_bottleneck(ASL, effective_scope=SCOPE)
    rendered = render_bottleneck(program)
    assert render_bottleneck(parse_bottleneck(rendered)) == rendered
    lowered = lower_bottleneck_to_asl(rendered)
    score = score_asl(ASL, lowered, SCOPE)
    assert validate_asl(lowered, effective_scope=SCOPE)["execution_verified"]
    assert score["semantic_state_equivalent"]
    assert score["semantic_return_equivalent"]
    assert score["dependency_correct"]
    assert score["final_answer_correct"]


def test_bottleneck_factorizes_paths_into_ordered_slots():
    program = asl_to_bottleneck(ASL, effective_scope=SCOPE)
    assert program["bindings"] == [
        {"slot": "s0", "path": "box.initial_count"},
        {"slot": "s1", "path": "box.removed_count"},
        {"slot": "s2", "path": "box.remaining_count"},
    ]
    labels = component_labels(program)
    assert labels["operator_counts"] == {"SUB": 1}
    assert labels["query"] == {"kind": "ref", "slot": "s2"}
    assert labels["dependencies"] == [
        {"source": "s0", "target": "s2"},
        {"source": "s1", "target": "s2"},
    ]


def test_bottleneck_rejects_unknown_slot_and_nonfinal_return():
    program = asl_to_bottleneck(ASL, effective_scope=SCOPE)
    broken = deepcopy(program)
    broken["steps"][-1]["expression"]["slot"] = "s99"
    with pytest.raises(ValueError, match="unknown slot"):
        render_bottleneck(broken)
    broken = deepcopy(program)
    broken["steps"][-2], broken["steps"][-1] = broken["steps"][-1], broken["steps"][-2]
    with pytest.raises(ValueError, match="final return"):
        render_bottleneck(broken)


def test_bottleneck_extraction_and_scoring_use_deterministic_lowering():
    rendered = render_bottleneck(asl_to_bottleneck(ASL, effective_scope=SCOPE))
    extracted = extract_bottleneck(f"Result:\n```json\n{rendered}\n```")
    assert extracted == rendered
    metrics = score_bottleneck(
        reference_program=rendered,
        reference_asl=ASL,
        predicted_program=extracted,
        effective_scope=SCOPE,
    )
    assert metrics["exact_program"]
    assert metrics["parse_valid"]
    assert metrics["lowerable_to_asl"]
    assert metrics["semantic_state_equivalent"]
    assert metrics["final_answer_correct"]


def test_bottleneck_run_is_autonomous_and_checkpointed(tmp_path):
    rendered = render_bottleneck(asl_to_bottleneck(ASL, effective_scope=SCOPE))
    eval_path = write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {
                "example_id": "f4-test-1",
                "parent_source_id": "source-1",
                "semantic_pattern_id": "pattern-1",
                "dataset": "gsm8k",
                "prompt": "Compile this problem without demonstrations.",
                "target": rendered,
                "target_asl": ASL,
                "effective_scope": SCOPE,
            }
        ],
    )

    class Backend:
        model_id = "Qwen/Qwen3-0.6B"

        def generate(self, prompt, *, seed):
            assert prompt == "Compile this problem without demonstrations."
            assert seed == 44017
            return SimpleNamespace(
                generated_text=rendered,
                prompt_tokens=7,
                generated_tokens=11,
                wall_time_ns=13,
                metadata={"device": "test"},
            )

    output = tmp_path / "output"
    summary = run_bottleneck_condition(
        eval_path=eval_path,
        model_config={
            "model": {
                "model_id": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            }
        },
        adapter_path=tmp_path / "adapter",
        adapter_id="f4-test-adapter",
        output_dir=output,
        backend_override=Backend(),
        checkpoint_every=1,
    )
    assert summary["rates"]["final_answer_correct"] == 1.0
    prediction = read_jsonl(output / "predictions.jsonl")[0]
    assert prediction["representation_id"] == "F4"
    assert prediction["objective_id"] == "L0"


def _d1_source(source_id, target, answer):
    return {
        "dataset": "gsm8k",
        "source_id": source_id,
        "question": f"What is the value for {source_id}?",
        "answer": str(answer),
        "effective_scope": {
            "id": f"gsm8k:train:{source_id}",
            "kind": "benchmark_case",
            "parent": None,
            "source": "dataset",
        },
        "record_sha256": source_id.ljust(64, "0"),
        "target_fixture": target,
    }


def _d1_annotation(source):
    lines = source["target_fixture"].splitlines()
    return {
        "dataset": source["dataset"],
        "source_id": source["source_id"],
        "full_asl": source["target_fixture"],
        "part_mappings": [
            {
                "part_id": 0,
                "asl": lines,
                "status": "ok",
                "confidence": 1.0,
                "assumptions": [],
                "semantic_notes": [],
            }
        ],
        "teacher": {"model": "test/model", "attempt": 1},
    }


def test_d1_freeze_matches_exposure_and_excludes_frozen_leakage(tmp_path):
    sources = [
        _d1_source("a", "a.count = 1\nRETURN a.count", 1),
        _d1_source("b", "shared.count = 2\nRETURN shared.count", 2),
        _d1_source("c", "c.left = 3\nc.total = c.left + 1\nRETURN c.total", 4),
        _d1_source("d", "d.left = 8\nd.total = d.left - 2\nRETURN d.total", 6),
    ]
    annotations = [_d1_annotation(source) for source in sources]
    frozen_pattern = _program_metrics(
        sources[1], annotations[1]["full_asl"], annotations[1]["part_mappings"]
    )["semantic_pattern_id"]
    source_path = write_jsonl(tmp_path / "sources.jsonl", sources)
    strict_path = write_jsonl(tmp_path / "strict.jsonl", annotations)
    frozen = tmp_path / "frozen"
    write_jsonl(
        frozen / "dev.jsonl",
        [
            {
                "dataset": "gsm8k",
                "parent_source_id": "a",
                "semantic_pattern_id": "dev-pattern",
            }
        ],
    )
    write_jsonl(
        frozen / "test.jsonl",
        [
            {
                "dataset": "gsm8k",
                "parent_source_id": "frozen-test",
                "semantic_pattern_id": frozen_pattern,
            }
        ],
    )

    manifest = build_d1_f0_data(
        strict_path=strict_path,
        source_paths=[source_path],
        frozen_data_dir=frozen,
        output_dir=tmp_path / "d1",
        target=2,
        epochs=2,
        seed=11,
    )
    train = read_jsonl(tmp_path / "d1" / "train.jsonl")
    assert {row["parent_source_id"] for row in train} == {"c", "d"}
    assert {row["epoch_view"] for row in train} == {0, 1}
    assert all(row["source_fields_visible_to_model"] == ["question"] for row in train)
    assert manifest["counts"]["strict_pre_exclusion"] == 4
    assert manifest["counts"]["excluded"] == 2
    assert manifest["counts"]["selected_dataset_quotas"] == {"gsm8k": 2}
    assert manifest["leakage_audit"]["passed"]


def test_gsm8k_freeze_rejects_other_datasets_at_boundary(tmp_path):
    gsm_sources = [
        _d1_source("a", "a.count = 1\nRETURN a.count", 1),
        _d1_source("b", "b.count = 2\nRETURN b.count", 2),
    ]
    tatqa_source = _d1_source("t", "t.count = 3\nRETURN t.count", 3)
    tatqa_source["dataset"] = "tatqa"
    strict = [_d1_annotation(source) for source in [*gsm_sources, tatqa_source]]
    source_path = write_jsonl(tmp_path / "gsm8k.jsonl", gsm_sources)
    strict_path = write_jsonl(tmp_path / "strict.jsonl", strict)
    frozen = tmp_path / "frozen"
    for split in ("dev", "test"):
        write_jsonl(
            frozen / f"{split}.jsonl",
            [
                {
                    "dataset": "tatqa",
                    "parent_source_id": f"tatqa-{split}",
                    "semantic_pattern_id": f"tatqa-{split}-pattern",
                },
                {
                    "dataset": "gsm8k",
                    "parent_source_id": f"gsm8k-{split}",
                    "semantic_pattern_id": f"gsm8k-{split}-pattern",
                },
            ],
        )

    manifest = build_gsm8k_f0_data(
        strict_path=strict_path,
        source_path=source_path,
        frozen_data_dir=frozen,
        output_dir=tmp_path / "g1",
        target=2,
        epochs=2,
    )
    train = read_jsonl(tmp_path / "g1" / "train.jsonl")
    assert {row["dataset"] for row in train} == {"gsm8k"}
    assert {row["dataset_id"] for row in train} == {"G1_GSM8K"}
    assert manifest["dataset_scope"] == ["gsm8k"]
    assert manifest["counts"]["strict_input"] == 3
    assert manifest["counts"]["strict_scope_rejected"] == 1
    assert manifest["leakage_audit"]["dataset_scope_enforced"]


def test_hard_negatives_are_executable_and_semantically_different():
    program = asl_to_bottleneck(ASL, effective_scope=SCOPE)
    negatives = generate_hard_negatives(program, reference_asl=ASL, effective_scope=SCOPE)
    kinds = {item["negative_type"] for item in negatives}
    assert {"operator_swap", "query_target_swap", "path_binding_swap", "source_fact_swap"} <= kinds
    for item in negatives:
        assert validate_asl(item["lowered_asl"], effective_scope=SCOPE)["execution_verified"]
        assert item["binding_changed"] or not (
            item["semantic_state_equivalent"] and item["semantic_return_equivalent"]
        )


def test_f4_preference_data_rotates_native_negatives_without_changing_positive(tmp_path):
    program = asl_to_bottleneck(ASL, effective_scope=SCOPE)
    rendered = render_bottleneck(program)
    generated = generate_hard_negatives(
        program, reference_asl=ASL, effective_scope=SCOPE
    )[:2]
    source = tmp_path / "bottleneck"
    positive = {
        "example_id": "f4-positive",
        "prompt": "fixed prompt",
        "target": rendered,
    }
    for split in ("train", "dev"):
        write_jsonl(source / "sft" / f"{split}.jsonl", [positive])
        write_jsonl(
            source / "negatives" / f"{split}.jsonl",
            [
                {
                    **negative,
                    "example_id": f"f4-negative-{index}",
                    "parent_example_id": "f4-positive",
                }
                for index, negative in enumerate(generated)
            ],
        )
    report = build_bottleneck_preference_data(
        source, tmp_path / "preference", epochs=2
    )
    train = read_jsonl(tmp_path / "preference" / "train.jsonl")
    assert len(train) == 2
    assert {row["epoch_view"] for row in train} == {0, 1}
    assert all(row["prompt"] == positive["prompt"] for row in train)
    assert all(row["target"] == positive["target"] for row in train)
    assert len({row["negative_target"] for row in train}) == 2
    assert all(parse_bottleneck(row["negative_target"]) for row in train)
    assert report["test_rows_generated"] == 0


def _metrics(epoch, *, executable, semantic_state, loss):
    return {
        "split": "dev",
        "examples": 25,
        "checkpoint": f"epoch-{epoch}.pt",
        "epoch": epoch,
        "parse_rate": 1.0,
        "lowerable_rate": 1.0,
        "type_valid_rate": 1.0,
        "executable_rate": executable,
        "semantic_return_rate": semantic_state,
        "semantic_state_rate": semantic_state,
        "dependency_rate": semantic_state,
        "operator_f1": semantic_state,
        "path_f1": semantic_state,
        "source_fact_f1": semantic_state,
        "answer_rate": semantic_state,
        "dev_loss": loss,
    }


def test_checkpoint_selection_prioritizes_semantics_over_token_loss():
    low_loss_wrong = _metrics(1, executable=0.8, semantic_state=0.8, loss=0.1)
    safe = _metrics(2, executable=1.0, semantic_state=0.4, loss=0.5)
    report = select_semantic_checkpoint([low_loss_wrong, safe])
    assert report["selected_checkpoint"] == "epoch-2.pt"


def test_checkpoint_selection_rejects_test_metrics():
    row = _metrics(1, executable=1.0, semantic_state=1.0, loss=0.1)
    row["split"] = "test"
    with pytest.raises(ValueError, match="development split"):
        select_semantic_checkpoint([row])
