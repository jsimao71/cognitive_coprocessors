from copy import deepcopy

import pytest

from ccpu.dsl import validate_asl
from ccpu.paper1.asl_pilot_eval import score_asl
from ccpu.paper1.e3.bottleneck import (
    asl_to_bottleneck,
    lower_bottleneck_to_asl,
    parse_bottleneck,
    render_bottleneck,
)
from ccpu.paper1.e3.components import component_labels
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
