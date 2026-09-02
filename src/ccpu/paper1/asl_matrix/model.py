"""Composable pretrained encoder-decoder models for the ASL 3D matrix."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SourceMemory:
    """Architecture-independent encoded source memory."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    source_type: str
    layer_states: dict[int, torch.Tensor]
    metadata: dict[str, Any]


@dataclass
class MatrixModelOutput:
    """Training/evaluation output without depending on a Transformers output class."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    diagnostics: dict[str, Any]
    past_key_values: Any = None


def has_repeated_generation_suffix(token_ids: list[int]) -> bool:
    """Detect only sustained token loops that cannot be useful ASL continuations."""

    repeats = 3
    for ngram in range(1, 33):
        width = ngram * repeats
        if len(token_ids) < width:
            continue
        tail = token_ids[-width:]
        if all(tail[:ngram] == tail[index : index + ngram] for index in range(ngram, width, ngram)):
            return True
    return len(token_ids) >= 64 and len(set(token_ids[-64:])) <= 4


class DualSourceT5CrossAttention(nn.Module):
    """Two separately normalized T5 cross-attention branches (matrix M1)."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.nl_attention = original.EncDecAttention
        self.asl_attention = copy.deepcopy(original.EncDecAttention)
        self.layer_norm = original.layer_norm
        self.dropout = original.dropout
        self.nl_gate_logit = nn.Parameter(torch.tensor(4.0))
        self.asl_gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.nl_width: int | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def set_nl_width(self, width: int) -> None:
        self.nl_width = width

    @staticmethod
    def _availability(mask: torch.Tensor | None, states: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones(states.shape[0], 1, 1, device=states.device, dtype=states.dtype)
        available = (mask == 0).any(dim=-1).to(dtype=states.dtype)
        while available.ndim > 3:
            available = available.any(dim=1)
        return available.reshape(states.shape[0], 1, 1)

    @staticmethod
    def _entropy(weights: torch.Tensor) -> float:
        probabilities = weights.float().clamp_min(1e-12)
        return float((-(probabilities * probabilities.log()).sum(dim=-1).mean()).detach().cpu())

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        query_length: int | None = None,
        output_attentions: bool = False,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        del position_bias
        if self.nl_width is None:
            raise RuntimeError("M1 NL memory width was not set")
        nl_states = key_value_states[:, : self.nl_width]
        asl_states = key_value_states[:, self.nl_width :]
        nl_mask = attention_mask[..., : self.nl_width] if attention_mask is not None else None
        asl_mask = attention_mask[..., self.nl_width :] if attention_mask is not None else None
        normed = self.layer_norm(hidden_states)
        kwargs = {
            "layer_head_mask": layer_head_mask,
            "query_length": query_length,
            "output_attentions": output_attentions,
            "cache_position": cache_position,
        }
        nl_output = self.nl_attention(
            normed,
            mask=nl_mask,
            key_value_states=nl_states,
            position_bias=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        asl_output = self.asl_attention(
            normed,
            mask=asl_mask,
            key_value_states=asl_states,
            position_bias=None,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        )
        nl_gate = torch.sigmoid(self.nl_gate_logit)
        asl_gate = torch.sigmoid(self.asl_gate_logit)
        nl_available = self._availability(nl_mask, nl_states)
        asl_available = self._availability(asl_mask, asl_states)
        nl_branch = nl_output[0] * nl_gate * nl_available
        asl_branch = asl_output[0] * asl_gate * asl_available
        layer_output = hidden_states + self.dropout(nl_branch + asl_branch)
        combined_bias = torch.cat((nl_output[1], asl_output[1]), dim=-1)
        self.last_diagnostics = {
            "nl_gate": float(nl_gate.detach().cpu()),
            "asl_gate": float(asl_gate.detach().cpu()),
            "nl_output_norm": float(nl_branch.float().norm(dim=-1).mean().detach().cpu()),
            "asl_output_norm": float(asl_branch.float().norm(dim=-1).mean().detach().cpu()),
            "nl_available_fraction": float(nl_available.mean().detach().cpu()),
            "asl_available_fraction": float(asl_available.mean().detach().cpu()),
        }
        outputs: tuple[torch.Tensor, ...] = (layer_output, combined_bias)
        if output_attentions:
            nl_weights = nl_output[2] * nl_gate * nl_available.unsqueeze(1)
            asl_weights = asl_output[2] * asl_gate * asl_available.unsqueeze(1)
            denominator = (
                nl_gate * nl_available.unsqueeze(1) + asl_gate * asl_available.unsqueeze(1)
            ).clamp_min(1e-8)
            weights = torch.cat((nl_weights, asl_weights), dim=-1) / denominator
            self.last_diagnostics.update(
                {
                    "nl_attention_entropy": self._entropy(nl_output[2]),
                    "asl_attention_entropy": self._entropy(asl_output[2]),
                }
            )
            outputs += (weights,)
        return outputs


class ASLMatrixModel(nn.Module):
    """T5-backed A1/A2/A3 encoder and M1/M2 memory composition."""

    ARCHITECTURES = ("separate", "shared", "hybrid")
    ATTENTION_MODES = ("cross", "merged_kv")

    def __init__(
        self,
        backbone: nn.Module,
        *,
        encoder_architecture: str,
        attention_mode: str,
        hybrid_shared_top_layers: int = 2,
        adaptation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if encoder_architecture not in self.ARCHITECTURES:
            raise ValueError(f"unsupported encoder architecture: {encoder_architecture}")
        if attention_mode not in self.ATTENTION_MODES:
            raise ValueError(f"unsupported attention mode: {attention_mode}")
        self.config = backbone.config
        self.shared = backbone.shared
        self.decoder = backbone.decoder
        self.lm_head = backbone.lm_head
        self.encoder_architecture = encoder_architecture
        self.attention_mode = attention_mode
        self.hybrid_shared_top_layers = hybrid_shared_top_layers
        self.adaptation = adaptation or {"method": "full"}
        self.last_generation_diagnostics: dict[str, Any] = {}
        self.source_type_embeddings = nn.Embedding(2, self.config.d_model)
        nn.init.zeros_(self.source_type_embeddings.weight)

        self.nl_encoder = backbone.encoder
        if encoder_architecture == "shared":
            self.asl_encoder = self.nl_encoder
        else:
            self.asl_encoder = copy.deepcopy(backbone.encoder)
            self.asl_encoder.set_input_embeddings(self.shared)
        if encoder_architecture == "hybrid":
            layer_count = len(self.nl_encoder.block)
            if not 0 < hybrid_shared_top_layers < layer_count:
                raise ValueError("hybrid shared top layers must be between zero and encoder depth")
            for index in range(layer_count - hybrid_shared_top_layers, layer_count):
                self.asl_encoder.block[index] = self.nl_encoder.block[index]

        self._m1_layers: list[DualSourceT5CrossAttention] = []
        if attention_mode == "cross":
            for block in self.decoder.block:
                layer = DualSourceT5CrossAttention(block.layer[1])
                block.layer[1] = layer
                self._m1_layers.append(layer)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        encoder_architecture: str,
        attention_mode: str,
        hybrid_shared_top_layers: int = 2,
        adaptation: dict[str, Any] | None = None,
    ) -> ASLMatrixModel:
        from transformers import T5ForConditionalGeneration

        backbone = T5ForConditionalGeneration.from_pretrained(model_id, revision=revision)
        backbone = adapt_pretrained_backbone(backbone, adaptation)
        return cls(
            backbone,
            encoder_architecture=encoder_architecture,
            attention_mode=attention_mode,
            hybrid_shared_top_layers=hybrid_shared_top_layers,
            adaptation=adaptation,
        )

    def _encode(
        self,
        encoder: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_type: str,
        *,
        output_hidden_states: bool,
    ) -> SourceMemory:
        result = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        type_index = 0 if source_type == "nl" else 1
        typed = result.last_hidden_state + self.source_type_embeddings.weight[type_index]
        layer_states = {}
        if output_hidden_states and result.hidden_states:
            layer_states = {
                index: state for index, state in enumerate(result.hidden_states) if index > 0
            }
        return SourceMemory(
            hidden_states=typed,
            attention_mask=attention_mask,
            source_type=source_type,
            layer_states=layer_states,
            metadata={"encoder_architecture": self.encoder_architecture},
        )

    def encode_sources(
        self,
        nl_input_ids: torch.Tensor,
        nl_attention_mask: torch.Tensor,
        asl_input_ids: torch.Tensor,
        asl_attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> tuple[SourceMemory, SourceMemory]:
        nl_memory = self._encode(
            self.nl_encoder,
            nl_input_ids,
            nl_attention_mask,
            "nl",
            output_hidden_states=output_hidden_states,
        )
        asl_memory = self._encode(
            self.asl_encoder,
            asl_input_ids,
            asl_attention_mask,
            "asl",
            output_hidden_states=output_hidden_states,
        )
        return nl_memory, asl_memory

    def _memory(
        self, nl_memory: SourceMemory, asl_memory: SourceMemory
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.cat((nl_memory.hidden_states, asl_memory.hidden_states), dim=1)
        mask = torch.cat((nl_memory.attention_mask, asl_memory.attention_mask), dim=1)
        for layer in self._m1_layers:
            layer.set_nl_width(nl_memory.hidden_states.shape[1])
        return hidden, mask

    def forward(
        self,
        *,
        nl_input_ids: torch.Tensor,
        nl_attention_mask: torch.Tensor,
        asl_input_ids: torch.Tensor,
        asl_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> MatrixModelOutput:
        if labels is None and decoder_input_ids is None:
            raise ValueError("labels or decoder_input_ids are required")
        nl_memory, asl_memory = self.encode_sources(
            nl_input_ids,
            nl_attention_mask,
            asl_input_ids,
            asl_attention_mask,
            output_hidden_states=output_hidden_states,
        )
        return self._decode_memories(
            nl_memory,
            asl_memory,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def _decode_memories(
        self,
        nl_memory: SourceMemory,
        asl_memory: SourceMemory,
        *,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> MatrixModelOutput:
        memory, memory_mask = self._memory(nl_memory, asl_memory)
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        decoder = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )
        sequence_output = decoder.last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.config.d_model**-0.5)
        logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        diagnostics = self.attention_diagnostics(decoder.cross_attentions, nl_memory, asl_memory)
        return MatrixModelOutput(
            loss=loss,
            logits=logits,
            diagnostics=diagnostics,
            past_key_values=decoder.past_key_values,
        )

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        decoder_start = self.config.decoder_start_token_id
        pad = self.config.pad_token_id
        if decoder_start is None or pad is None:
            raise ValueError("backbone must define decoder_start_token_id and pad_token_id")
        shifted = labels.new_full(labels.shape, pad)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = decoder_start
        shifted.masked_fill_(shifted == -100, pad)
        return shifted

    def attention_diagnostics(
        self,
        cross_attentions: tuple[torch.Tensor, ...] | None,
        nl_memory: SourceMemory,
        asl_memory: SourceMemory,
    ) -> dict[str, Any]:
        if self.attention_mode == "cross":
            return {
                "mode": "cross",
                "layers": [dict(layer.last_diagnostics) for layer in self._m1_layers],
            }
        layers = []
        if cross_attentions:
            split = nl_memory.hidden_states.shape[1]
            nl_mask = nl_memory.attention_mask[:, None, None, :]
            asl_mask = asl_memory.attention_mask[:, None, None, :]
            for weights in cross_attentions:
                nl_mass = (weights[..., :split] * nl_mask).sum(dim=-1).mean()
                asl_mass = (weights[..., split:] * asl_mask).sum(dim=-1).mean()
                probabilities = weights.float().clamp_min(1e-12)
                entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
                layers.append(
                    {
                        "nl_attention_mass": float(nl_mass.detach().cpu()),
                        "asl_attention_mass": float(asl_mass.detach().cpu()),
                        "joint_attention_entropy": float(entropy.detach().cpu()),
                    }
                )
        return {"mode": "merged_kv", "layers": layers}

    @torch.no_grad()
    def greedy_generate(
        self,
        *,
        nl_input_ids: torch.Tensor,
        nl_attention_mask: torch.Tensor,
        asl_input_ids: torch.Tensor,
        asl_attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        batch = nl_input_ids.shape[0]
        nl_memory, asl_memory = self.encode_sources(
            nl_input_ids,
            nl_attention_mask,
            asl_input_ids,
            asl_attention_mask,
        )
        decoder_ids = torch.full(
            (batch, 1),
            int(self.config.decoder_start_token_id),
            dtype=torch.long,
            device=nl_input_ids.device,
        )
        complete = torch.zeros(batch, dtype=torch.bool, device=nl_input_ids.device)
        repeated = torch.zeros(batch, dtype=torch.bool, device=nl_input_ids.device)
        past_key_values = None
        for _ in range(max_new_tokens):
            step_ids = decoder_ids if past_key_values is None else decoder_ids[:, -1:]
            output = self._decode_memories(
                nl_memory,
                asl_memory,
                decoder_input_ids=step_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1)
            halted = complete | repeated
            next_token = torch.where(
                halted,
                torch.full_like(next_token, int(self.config.eos_token_id)),
                next_token,
            )
            decoder_ids = torch.cat((decoder_ids, next_token[:, None]), dim=1)
            complete |= next_token == int(self.config.eos_token_id)
            repeated |= torch.tensor(
                [has_repeated_generation_suffix(row[1:].tolist()) for row in decoder_ids],
                device=decoder_ids.device,
            )
            if bool((complete | repeated).all()):
                break
        self.last_generation_diagnostics = {
            "generated_tokens": [int(len(row) - 1) for row in decoder_ids],
            "eos_reached": [bool(value) for value in complete.tolist()],
            "repetition_stopped": [bool(value) for value in repeated.tolist()],
            "max_new_tokens": max_new_tokens,
        }
        return decoder_ids[:, 1:]

    def parameter_report(self) -> dict[str, Any]:
        named_parameters = list(self.named_parameters())
        parameters = [parameter for _, parameter in named_parameters]
        total = sum(parameter.numel() for parameter in parameters)
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        lora = sum(
            parameter.numel()
            for name, parameter in named_parameters
            if parameter.requires_grad and "lora_" in name
        )
        source_type = sum(
            parameter.numel()
            for name, parameter in named_parameters
            if parameter.requires_grad and "source_type_embeddings" in name
        )
        gates = sum(
            parameter.numel()
            for name, parameter in named_parameters
            if parameter.requires_grad and name.endswith("_gate_logit")
        )
        nl_ids = {id(parameter) for parameter in self.nl_encoder.parameters()}
        asl_ids = {id(parameter) for parameter in self.asl_encoder.parameters()}
        return {
            "encoder_architecture": self.encoder_architecture,
            "attention_mode": self.attention_mode,
            "adaptation": self.adaptation,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "lora_parameters": lora,
            "source_type_parameters": source_type,
            "gate_parameters": gates,
            "other_trainable_parameters": trainable - lora - source_type - gates,
            "shared_encoder_parameter_tensors": len(nl_ids & asl_ids),
            "hybrid_shared_top_layers": (
                self.hybrid_shared_top_layers if self.encoder_architecture == "hybrid" else 0
            ),
        }


def adapt_pretrained_backbone(backbone: nn.Module, adaptation: dict[str, Any] | None) -> nn.Module:
    """Apply a budgeted PEFT patch while preserving pretrained base weights."""

    spec = adaptation or {"method": "full"}
    method = str(spec.get("method", "full"))
    if method == "full":
        return backbone
    if method != "lora":
        raise ValueError(f"unsupported matrix adaptation method: {method}")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("LoRA matrix adaptation requires peft") from error
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(spec.get("rank", 8)),
        lora_alpha=int(spec.get("alpha", 16)),
        lora_dropout=float(spec.get("dropout", 0.0)),
        target_modules=list(spec.get("target_modules", ("q", "k", "v", "o"))),
        bias=str(spec.get("bias", "none")),
    )
    return get_peft_model(backbone, config)


def representation_alignment(
    nl_states: torch.Tensor, asl_states: torch.Tensor, nl_mask: torch.Tensor, asl_mask: torch.Tensor
) -> dict[str, float]:
    """Compute stable pooled cosine and paired-retrieval diagnostics."""

    nl_denom = nl_mask.sum(dim=1, keepdim=True).clamp_min(1)
    asl_denom = asl_mask.sum(dim=1, keepdim=True).clamp_min(1)
    nl_pooled = (nl_states * nl_mask.unsqueeze(-1)).sum(dim=1) / nl_denom
    asl_pooled = (asl_states * asl_mask.unsqueeze(-1)).sum(dim=1) / asl_denom
    cosine = F.cosine_similarity(nl_pooled.float(), asl_pooled.float(), dim=-1)
    normalized_nl = F.normalize(nl_pooled.float(), dim=-1)
    normalized_asl = F.normalize(asl_pooled.float(), dim=-1)
    similarities = normalized_nl @ normalized_asl.T
    retrieval = (
        similarities.argmax(dim=-1) == torch.arange(len(similarities), device=similarities.device)
    ).float()
    return {
        "paired_cosine_mean": float(cosine.mean().detach().cpu()),
        "paired_retrieval_accuracy": float(retrieval.mean().detach().cpu()),
        "batch_size": float(len(similarities)),
    }
