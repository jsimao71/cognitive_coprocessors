import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ccpu.paper1.asl_matrix.qwen import _prompt
from ccpu.paper1.asl_matrix.qwen_patch import install_qwen_memory_patches
from ccpu.paper1.asl_matrix.qwen_patch_train import split_patch_record
from ccpu.paper1.asl_pilot_data import asl_prompt


def _tiny_qwen():
    config = transformers.Qwen3Config(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize("mode", ["cross", "native_kv"])
def test_qwen_patch_is_exactly_inert_without_external_memory(mode):
    torch.manual_seed(7)
    model = _tiny_qwen()
    input_ids = torch.randint(2, 100, (1, 6))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits

    controller = install_qwen_memory_patches(model, mode=mode)
    with torch.no_grad():
        patched = model(input_ids=input_ids, attention_mask=attention_mask).logits

    torch.testing.assert_close(patched, baseline, rtol=0, atol=0)
    assert controller.parameter_report()["patched_layers"] == 2


def test_qwen_cross_patch_starts_near_zero_and_counts_only_new_capacity():
    model = _tiny_qwen()
    controller = install_qwen_memory_patches(model, mode="cross")
    external_ids = torch.randint(2, 100, (1, 5))
    local_ids = torch.randint(2, 100, (1, 6))

    with torch.no_grad():
        controller.capture_external(external_ids, torch.ones_like(external_ids))
        output = model(input_ids=local_ids, attention_mask=torch.ones_like(local_ids))

    assert torch.isfinite(output.logits).all()
    assert all(item["external_gate"] < 0.02 for item in controller.diagnostics())
    report = controller.parameter_report()
    expected_per_layer = 32 * 16 + 32 * 16 + 32 * 32 + 1
    assert report["memory_patch_trainable_parameters"] == 32 + 2 * expected_per_layer
    assert report["other_trainable_parameters"] > 0
    assert report["total_trainable_parameters"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


@pytest.mark.parametrize("mode", ["cross", "native_kv"])
def test_qwen_external_memory_backpropagates_without_a_second_backbone(mode):
    model = _tiny_qwen().train()
    for parameter in model.parameters():
        parameter.requires_grad = False
    controller = install_qwen_memory_patches(model, mode=mode)
    external_ids = torch.randint(2, 100, (1, 5))
    local_ids = torch.randint(2, 100, (1, 6))

    controller.capture_external(external_ids, torch.ones_like(external_ids))
    loss = model(
        input_ids=local_ids,
        attention_mask=torch.ones_like(local_ids),
        labels=local_ids,
    ).loss
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert len({id(patch.original) for patch in controller.patches}) == 2


def test_qwen_native_patch_uses_only_a_source_type_embedding():
    model = _tiny_qwen()
    for parameter in model.parameters():
        parameter.requires_grad = False
    controller = install_qwen_memory_patches(model, mode="native_kv")

    assert controller.parameter_report() == {
        "mode": "native_kv",
        "patched_layers": 2,
        "memory_patch_trainable_parameters": 32,
        "other_trainable_parameters": 0,
        "total_trainable_parameters": 32,
    }


def test_serialized_q1_view_splits_without_changing_autonomous_prompt():
    autonomous = {
        "example_id": "a",
        "has_external_asl": False,
        "prompt": "Instruction\n\nInput:\nProblem: P\nASL:",
    }
    assisted = {
        "example_id": "b",
        "has_external_asl": True,
        "prompt": (
            "Instruction\n\nInput:\nProblem: P\n\nExternal ASL teacher:\n"
            "x.value = 3\nRETURN x.value\nASL:"
        ),
    }

    assert split_patch_record(autonomous) == (autonomous["prompt"], None)
    assert split_patch_record(assisted) == (
        "Instruction\n\nInput:\nProblem: P\nASL:",
        "x.value = 3\nRETURN x.value",
    )


def test_qwen_autonomous_prompt_is_byte_identical_to_historical_f0():
    question = "There are 3 objects. How many objects are there?"
    matrix_prompt = _prompt({"nl_input": question, "external_asl_input": None})
    historical_prompt = asl_prompt(
        {"question": question, "source_context": None}, demonstrations=[]
    )

    assert matrix_prompt == historical_prompt


def test_qwen_prompt_preserves_authoritative_evidence_before_memory():
    base = (
        "Instruction\n\nInput:\nEvidence:\nTABLE: Total | 302 | 148\n"
        "Problem: What changed?\nASL:"
    )
    view = {
        "nl_input": "What changed?",
        "external_asl_input": "total.current = 302\nRETURN total.current",
    }

    assert _prompt({**view, "external_asl_input": None}, base) == base
    assert _prompt(view, base) == (
        "Instruction\n\nInput:\nEvidence:\nTABLE: Total | 302 | 148\n"
        "Problem: What changed?\n\nExternal ASL teacher:\n"
        "total.current = 302\nRETURN total.current\nASL:"
    )
