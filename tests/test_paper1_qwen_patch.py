import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ccpu.paper1.asl_matrix.qwen_patch import install_qwen_memory_patches


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
