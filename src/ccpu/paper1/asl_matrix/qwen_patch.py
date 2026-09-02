"""LoRA-compatible external-ASL memory patches for pretrained Qwen3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return states
    batch, heads, length, width = states.shape
    states = states[:, :, None, :, :].expand(batch, heads, groups, length, width)
    return states.reshape(batch, heads * groups, length, width)


class QwenExternalMemoryAttention(nn.Module):
    """Wrap one Qwen attention layer with Q2 cross or Q3 merged-native memory."""

    MODES = ("cross", "native_kv")

    def __init__(self, original: nn.Module, *, mode: str) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported Qwen memory patch mode: {mode}")
        self.original = original
        self.mode = mode
        self.config = original.config
        self.layer_idx = original.layer_idx
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.scaling = original.scaling
        self.attention_dropout = original.attention_dropout
        self.is_causal = True
        self.sliding_window = original.sliding_window
        hidden_size = self.config.hidden_size
        kv_width = self.config.num_key_value_heads * self.head_dim
        if mode == "cross":
            self.external_k_proj = nn.Linear(hidden_size, kv_width, bias=False)
            self.external_v_proj = nn.Linear(hidden_size, kv_width, bias=False)
            self.external_o_proj = nn.Linear(
                self.config.num_attention_heads * self.head_dim, hidden_size, bias=False
            )
            nn.init.normal_(self.external_k_proj.weight, std=self.config.initializer_range)
            nn.init.normal_(self.external_v_proj.weight, std=self.config.initializer_range)
            nn.init.zeros_(self.external_o_proj.weight)
            self.external_gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.capture_mode = False
        self.external_attention_mask: torch.Tensor | None = None
        self.external_hidden_states: torch.Tensor | None = None
        self.external_key_states: torch.Tensor | None = None
        self.external_value_states: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}

    def begin_capture(self, attention_mask: torch.Tensor) -> None:
        self.capture_mode = True
        self.external_attention_mask = attention_mask
        self.external_hidden_states = None
        self.external_key_states = None
        self.external_value_states = None

    def end_capture(self) -> None:
        self.capture_mode = False

    def clear_external(self) -> None:
        self.capture_mode = False
        self.external_attention_mask = None
        self.external_hidden_states = None
        self.external_key_states = None
        self.external_value_states = None
        self.last_diagnostics = {}

    def _project_native(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.original.q_norm(self.original.q_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        key = self.original.k_norm(self.original.k_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        value = self.original.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value

    def _capture(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.external_hidden_states = hidden_states
        if self.mode == "native_kv":
            _, key, value = self._project_native(hidden_states, position_embeddings)
            self.external_key_states = key
            self.external_value_states = value

    def _external_validity(self, dtype: torch.dtype, query_length: int) -> torch.Tensor:
        if self.external_attention_mask is None:
            raise RuntimeError("external attention mask is unavailable")
        mask = self.external_attention_mask[:, None, None, :].to(dtype=dtype)
        mask = (1.0 - mask) * torch.finfo(dtype).min
        return mask.expand(-1, 1, query_length, -1)

    def _cross_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.external_hidden_states is None:
            return torch.zeros_like(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.original.q_norm(
            self.original.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        external_shape = (*self.external_hidden_states.shape[:-1], -1, self.head_dim)
        external_hidden = self.external_hidden_states.to(self.external_k_proj.weight.dtype)
        key = self.original.k_norm(
            self.external_k_proj(external_hidden).view(external_shape)
        ).transpose(1, 2)
        value = self.external_v_proj(external_hidden).view(external_shape).transpose(1, 2)
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        query = query.to(key.dtype)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        scores = scores + self._external_validity(scores.dtype, query.shape[-2])
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = output.reshape(*input_shape, -1)
        output = self.external_o_proj(output)
        gate = torch.sigmoid(self.external_gate_logit)
        available = self.external_attention_mask.any(dim=-1, keepdim=True).to(output.dtype)
        output = output * gate * available.unsqueeze(-1)
        self.last_diagnostics = {
            "external_gate": float(gate.detach().cpu()),
            "external_output_norm": float(output.float().norm(dim=-1).mean().detach().cpu()),
            "external_attention_entropy": float(
                (-(weights.float().clamp_min(1e-12) * weights.float().clamp_min(1e-12).log()))
                .sum(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
        }
        return output.to(hidden_states.dtype)

    def _native_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any,
        cache_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.external_key_states is None or self.external_value_states is None:
            return self.original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
        input_shape = hidden_states.shape[:-1]
        query, local_key, local_value = self._project_native(hidden_states, position_embeddings)
        if past_key_values is not None:
            cos, sin = position_embeddings
            local_key, local_value = past_key_values.update(
                local_key,
                local_value,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        external_key = _repeat_kv(self.external_key_states, self.num_key_value_groups)
        external_value = _repeat_kv(self.external_value_states, self.num_key_value_groups)
        local_key = _repeat_kv(local_key, self.num_key_value_groups)
        local_value = _repeat_kv(local_value, self.num_key_value_groups)
        key = torch.cat((external_key, local_key), dim=-2)
        value = torch.cat((external_value, local_value), dim=-2)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        external_mask = self._external_validity(scores.dtype, query.shape[-2])
        if attention_mask is None:
            local_mask = torch.zeros(
                (*external_mask.shape[:-1], local_key.shape[-2]),
                dtype=scores.dtype,
                device=scores.device,
            )
        else:
            local_mask = attention_mask[:, :, :, : local_key.shape[-2]]
        scores = scores + torch.cat((external_mask, local_mask), dim=-1)
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = self.original.o_proj(output.reshape(*input_shape, -1))
        external_width = external_key.shape[-2]
        external_mass = weights[..., :external_width].sum(dim=-1).mean()
        self.last_diagnostics = {
            "external_attention_mass": float(external_mass.detach().cpu()),
            "joint_attention_entropy": float(
                (-(weights.float().clamp_min(1e-12) * weights.float().clamp_min(1e-12).log()))
                .sum(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
        }
        return output, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.capture_mode:
            self._capture(hidden_states, position_embeddings)
            return self.original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        if self.external_attention_mask is None:
            return self.original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        if self.mode == "cross":
            local, weights = self.original(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            return local + self._cross_attention(hidden_states), weights
        return self._native_attention(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
        )


@dataclass
class QwenPatchController:
    """Controls capture/visibility for installed per-layer patches."""

    model: nn.Module
    mode: str
    patches: list[QwenExternalMemoryAttention]
    source_type_embedding: nn.Parameter

    def clear(self) -> None:
        for patch in self.patches:
            patch.clear_external()

    def capture_external(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> None:
        self.clear()
        if not bool(attention_mask.any()):
            return
        for patch in self.patches:
            patch.begin_capture(attention_mask)
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings = embeddings + self.source_type_embedding.to(embeddings.dtype)
        self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        )
        for patch in self.patches:
            patch.end_capture()

    def diagnostics(self) -> list[dict[str, float]]:
        return [dict(patch.last_diagnostics) for patch in self.patches]

    def parameter_report(self) -> dict[str, int | str]:
        memory_parameters = [self.source_type_embedding]
        for patch in self.patches:
            if patch.mode == "cross":
                memory_parameters.extend(
                    [
                        *patch.external_k_proj.parameters(),
                        *patch.external_v_proj.parameters(),
                        *patch.external_o_proj.parameters(),
                        patch.external_gate_logit,
                    ]
                )
        memory_ids = {id(parameter) for parameter in memory_parameters}
        all_trainable = {
            id(parameter): parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        memory_count = sum(
            parameter.numel()
            for identifier, parameter in all_trainable.items()
            if identifier in memory_ids
        )
        return {
            "mode": self.mode,
            "patched_layers": len(self.patches),
            "memory_patch_trainable_parameters": memory_count,
            "other_trainable_parameters": sum(
                parameter.numel()
                for identifier, parameter in all_trainable.items()
                if identifier not in memory_ids
            ),
            "total_trainable_parameters": sum(
                parameter.numel() for parameter in all_trainable.values()
            ),
        }


def install_qwen_memory_patches(
    model: nn.Module,
    *,
    mode: str,
    layer_indices: tuple[int, ...] | None = None,
) -> QwenPatchController:
    """Install Q2/Q3 patches without creating another Qwen backbone."""

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Qwen3Attention"
    ]
    if not candidates:
        raise ValueError("no Qwen3Attention modules found")
    available = sorted({int(module.layer_idx) for _, module in candidates})
    selected = set(layer_indices if layer_indices is not None else available[:: max(1, len(available) // 4)])
    patches = []
    for name, module in candidates:
        if int(module.layer_idx) not in selected:
            continue
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        patch = QwenExternalMemoryAttention(module, mode=mode)
        setattr(parent, attribute, patch)
        patches.append(patch)
    if not patches:
        raise ValueError(f"no Qwen layers selected from {available}")
    hidden_size = int(patches[0].config.hidden_size)
    embedding_weight = model.get_input_embeddings().weight
    source_type_embedding = nn.Parameter(
        torch.zeros(hidden_size, device=embedding_weight.device, dtype=torch.float32)
    )
    model.register_parameter("asl_external_source_embedding", source_type_embedding)
    return QwenPatchController(
        model=model,
        mode=mode,
        patches=patches,
        source_type_embedding=source_type_embedding,
    )
