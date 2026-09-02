import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ccpu.paper1.asl_matrix.model import (
    ASLMatrixModel,
    adapt_pretrained_backbone,
    representation_alignment,
)


def _backbone():
    config = transformers.T5Config(
        vocab_size=101,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return transformers.T5ForConditionalGeneration(config)


@pytest.mark.parametrize("architecture", ASLMatrixModel.ARCHITECTURES)
@pytest.mark.parametrize("attention_mode", ASLMatrixModel.ATTENTION_MODES)
def test_all_matrix_architectures_run_autonomous_forward(architecture, attention_mode):
    model = ASLMatrixModel(
        _backbone(),
        encoder_architecture=architecture,
        attention_mode=attention_mode,
        hybrid_shared_top_layers=1,
    )
    batch = 2
    output = model(
        nl_input_ids=torch.randint(2, 100, (batch, 7)),
        nl_attention_mask=torch.ones(batch, 7, dtype=torch.long),
        asl_input_ids=torch.zeros(batch, 3, dtype=torch.long),
        asl_attention_mask=torch.zeros(batch, 3, dtype=torch.long),
        labels=torch.randint(2, 100, (batch, 5)),
        output_attentions=True,
    )
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.logits.shape == (batch, 5, 101)
    assert output.diagnostics["mode"] == attention_mode
    assert len(output.diagnostics["layers"]) == 2


def test_encoder_sharing_matches_architecture_contract():
    separate = ASLMatrixModel(
        _backbone(), encoder_architecture="separate", attention_mode="merged_kv"
    )
    shared = ASLMatrixModel(_backbone(), encoder_architecture="shared", attention_mode="merged_kv")
    hybrid = ASLMatrixModel(
        _backbone(),
        encoder_architecture="hybrid",
        attention_mode="merged_kv",
        hybrid_shared_top_layers=1,
    )
    assert separate.nl_encoder is not separate.asl_encoder
    assert shared.nl_encoder is shared.asl_encoder
    assert hybrid.nl_encoder.block[0] is not hybrid.asl_encoder.block[0]
    assert hybrid.nl_encoder.block[1] is hybrid.asl_encoder.block[1]
    assert (
        separate.parameter_report()["total_parameters"]
        > hybrid.parameter_report()["total_parameters"]
        > shared.parameter_report()["total_parameters"]
    )


def test_m1_asl_branch_is_disabled_for_autonomous_view():
    model = ASLMatrixModel(_backbone(), encoder_architecture="separate", attention_mode="cross")
    output = model(
        nl_input_ids=torch.randint(2, 100, (1, 5)),
        nl_attention_mask=torch.ones(1, 5, dtype=torch.long),
        asl_input_ids=torch.zeros(1, 2, dtype=torch.long),
        asl_attention_mask=torch.zeros(1, 2, dtype=torch.long),
        labels=torch.randint(2, 100, (1, 4)),
    )
    assert all(layer["asl_available_fraction"] == 0 for layer in output.diagnostics["layers"])
    assert all(layer["asl_output_norm"] == 0 for layer in output.diagnostics["layers"])
    assert all(layer["asl_gate"] < 0.02 for layer in output.diagnostics["layers"])


def test_lora_patch_freezes_pretrained_t5_and_reports_budget():
    pytest.importorskip("peft")
    adaptation = {
        "method": "lora",
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q", "k", "v", "o"],
        "bias": "none",
    }
    backbone = adapt_pretrained_backbone(_backbone(), adaptation)
    model = ASLMatrixModel(
        backbone,
        encoder_architecture="separate",
        attention_mode="cross",
        adaptation=adaptation,
    )
    report = model.parameter_report()
    assert report["adaptation"]["method"] == "lora"
    assert report["lora_parameters"] > 0
    assert report["frozen_parameters"] > report["trainable_parameters"]
    assert report["source_type_parameters"] == 64
    assert report["gate_parameters"] == 4
    assert report["other_trainable_parameters"] == 0


def test_representation_alignment_recovers_identical_pairs():
    states = torch.eye(3).unsqueeze(1)
    mask = torch.ones(3, 1, dtype=torch.long)
    metrics = representation_alignment(states, states.clone(), mask, mask)
    assert metrics["paired_cosine_mean"] == pytest.approx(1.0)
    assert metrics["paired_retrieval_accuracy"] == 1.0
