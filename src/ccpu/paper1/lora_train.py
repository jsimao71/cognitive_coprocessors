"""Minimal PEFT LoRA training for calculator-block interface behavior."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json

from .generation import select_device


@dataclass(frozen=True)
class LoRATrainingConfig:
    epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_length: int = 192
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    seed: int = 99173
    device: str = "xpu"
    dtype: str = "float16"
    gradient_checkpointing: bool = True
    evaluate_each_epoch: bool = True
    checkpoint_every_optimizer_steps: int = 0
    reject_truncation: bool = False
    restore_best_dev: bool = False
    logical_epoch_field: str | None = None
    semantic_token_weights: dict[str, float] | None = None
    pairwise_rank_weight: float = 0.0
    pairwise_temperature: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LoRATrainingConfig:
        data = value.get("training", value)
        return cls(
            epochs=int(data.get("epochs", 2)),
            batch_size=int(data.get("batch_size", 1)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 8)),
            learning_rate=float(data.get("learning_rate", 2e-4)),
            max_length=int(data.get("max_length", 192)),
            rank=int(data.get("rank", 8)),
            alpha=int(data.get("alpha", 16)),
            dropout=float(data.get("dropout", 0.05)),
            target_modules=tuple(
                str(item)
                for item in data.get(
                    "target_modules", ("q_proj", "k_proj", "v_proj", "o_proj")
                )
            ),
            seed=int(data.get("seed", 99173)),
            device=str(data.get("device", "xpu")),
            dtype=str(data.get("dtype", "float16")),
            gradient_checkpointing=bool(data.get("gradient_checkpointing", True)),
            evaluate_each_epoch=bool(data.get("evaluate_each_epoch", True)),
            checkpoint_every_optimizer_steps=int(
                data.get("checkpoint_every_optimizer_steps", 0)
            ),
            reject_truncation=bool(data.get("reject_truncation", False)),
            restore_best_dev=bool(data.get("restore_best_dev", False)),
            logical_epoch_field=(
                str(data["logical_epoch_field"])
                if data.get("logical_epoch_field") is not None
                else None
            ),
            semantic_token_weights=(
                {str(key): float(weight) for key, weight in data["semantic_token_weights"].items()}
                if data.get("semantic_token_weights") is not None
                else None
            ),
            pairwise_rank_weight=float(data.get("pairwise_rank_weight", 0.0)),
            pairwise_temperature=float(data.get("pairwise_temperature", 1.0)),
        )

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, batch size, and accumulation steps must be positive")
        if self.max_length < 32 or self.rank < 1 or self.alpha < 1:
            raise ValueError("max length, rank, and alpha are invalid")
        if not 0 <= self.dropout < 1 or not self.target_modules:
            raise ValueError("dropout or target modules are invalid")
        if self.checkpoint_every_optimizer_steps < 0:
            raise ValueError("checkpoint interval cannot be negative")
        if self.restore_best_dev and not self.evaluate_each_epoch:
            raise ValueError("best-dev restoration requires evaluation after every epoch")
        if self.pairwise_rank_weight < 0 or self.pairwise_temperature <= 0:
            raise ValueError("pairwise rank weight must be non-negative and temperature positive")
        if self.semantic_token_weights is not None:
            required = {"default", "path", "operator", "literal", "return"}
            if set(self.semantic_token_weights) != required:
                raise ValueError(f"semantic token weights must have exactly {sorted(required)}")
            if any(weight <= 0 for weight in self.semantic_token_weights.values()):
                raise ValueError("semantic token weights must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_modules"] = list(self.target_modules)
        return value


def _chat_text(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> str:
    arguments = {"tokenize": False, **kwargs, "enable_thinking": False}
    try:
        return str(tokenizer.apply_chat_template(messages, **arguments))
    except TypeError:
        arguments.pop("enable_thinking")
        return str(tokenizer.apply_chat_template(messages, **arguments))


def _tokenize_record(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    *,
    reject_truncation: bool = False,
    semantic_token_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    user = {"role": "user", "content": str(row["prompt"])}
    prefix = _chat_text(tokenizer, [user], add_generation_prompt=True)
    complete = _chat_text(
        tokenizer,
        [user, {"role": "assistant", "content": str(row["target"])}],
        add_generation_prompt=False,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        complete,
        add_special_tokens=False,
        **({"return_offsets_mapping": True} if semantic_token_weights else {}),
    )
    full_input_ids = encoded["input_ids"]
    common = 0
    for prefix_token, full_token in zip(prefix_ids, full_input_ids):
        if prefix_token != full_token:
            break
        common += 1
    if common < max(1, len(prefix_ids) - 2):
        raise ValueError(f"chat template prefix mismatch for {row['example_id']}")
    if reject_truncation and len(full_input_ids) > max_length:
        raise ValueError(
            f"record exceeds max_length for {row['example_id']}: "
            f"{len(full_input_ids)} > {max_length}"
        )
    input_ids = full_input_ids[:max_length]
    labels = [-100] * min(common, len(input_ids)) + input_ids[common:]
    if not any(label != -100 for label in labels):
        raise ValueError(f"target truncated for {row['example_id']}")
    result = {
        "example_id": row["example_id"],
        "input_ids": input_ids,
        "labels": labels,
        "target_tokens": sum(label != -100 for label in labels),
        "full_tokens": len(full_input_ids),
        "prefix_tokens": len(prefix_ids),
        "was_truncated": len(full_input_ids) > max_length,
    }
    if semantic_token_weights is not None:
        target = str(row["target"])
        target_start = complete.rfind(target)
        if target_start < 0:
            raise ValueError(f"target is absent from rendered chat for {row['example_id']}")
        spans = semantic_weight_spans(target, semantic_token_weights)
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("semantic token weighting requires a fast tokenizer with offsets")
        weights = []
        for index, (start, end) in enumerate(offsets[:max_length]):
            if index < common or end <= target_start:
                weights.append(0.0)
                continue
            local_start = max(0, start - target_start)
            local_end = max(0, end - target_start)
            matched = [
                weight
                for span_start, span_end, weight in spans
                if local_start < span_end and local_end > span_start
            ]
            weights.append(max(matched, default=semantic_token_weights["default"]))
        result["loss_weights"] = weights
    if row.get("negative_target") is not None:
        negative_row = {
            "example_id": f"{row['example_id']}:negative",
            "prompt": row["prompt"],
            "target": row["negative_target"],
        }
        result["negative"] = _tokenize_record(
            tokenizer,
            negative_row,
            max_length,
            reject_truncation=reject_truncation,
            semantic_token_weights=semantic_token_weights,
        )
        result["negative_type"] = row.get("negative_type")
    return result


_SEMANTIC_SPANS = {
    "path": re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"),
    "operator": re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()|(?<!\w)[+*/-](?!\w)"),
    "literal": re.compile(r'(?<![A-Za-z0-9_])(?:\d+(?:\.\d+)?|"(?:\\.|[^"\\])*")'),
    "return": re.compile(r"\bRETURN\b[^\n]*", re.IGNORECASE),
}

_JSON_STRING = r'"(?:\\.|[^"\\])*"'
_F4_SEMANTIC_SPANS = {
    "path": (
        re.compile(rf'"path":(?P<value>{_JSON_STRING})'),
        re.compile(r'"(?:slot|target)":(?P<value>"s\d+")'),
    ),
    "operator": (re.compile(rf'"operator":(?P<value>{_JSON_STRING})'),),
    "literal": (
        re.compile(
            rf'"value":(?P<value>{_JSON_STRING}|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'
        ),
    ),
    "return": (re.compile(r'"kind":(?P<value>"return")'),),
}


def _is_f4_target(target: str) -> bool:
    try:
        value = json.loads(target)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "ccpu.paper1.semantic_bottleneck.v1"
    )


def semantic_weight_spans(
    target: str, weights: dict[str, float]
) -> list[tuple[int, int, float]]:
    """Locate semantic decisions without changing any existing F0 target."""

    spans = []
    if _is_f4_target(target):
        for component, patterns in _F4_SEMANTIC_SPANS.items():
            for pattern in patterns:
                spans.extend(
                    (*match.span("value"), weights[component])
                    for match in pattern.finditer(target)
                )
        return spans
    for component, pattern in _SEMANTIC_SPANS.items():
        spans.extend((match.start(), match.end(), weights[component]) for match in pattern.finditer(target))
    return spans


def _batch(torch: Any, records: list[dict[str, Any]], pad_token_id: int, device: str):
    length = max(len(row["input_ids"]) for row in records)
    input_ids = []
    attention_mask = []
    labels = []
    loss_weights = []
    weighted = any("loss_weights" in row for row in records)
    for row in records:
        padding = length - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
        if weighted:
            if "loss_weights" not in row:
                raise ValueError("cannot mix weighted and unweighted training records")
            loss_weights.append(row["loss_weights"] + [0.0] * padding)
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }
    if weighted:
        batch["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32, device=device)
    return batch


def _loss_and_score(
    torch: Any, logits: Any, labels: Any, weights: Any | None
) -> tuple[Any, Any, Any]:
    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = labels[..., 1:].contiguous()
    token_losses = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    active = shifted_labels.ne(-100)
    ordinary = token_losses[active].mean()
    effective = (
        active.float()
        if weights is None
        else weights[..., 1:].contiguous() * active.to(dtype=weights.dtype)
    )
    weighted_loss = (token_losses * effective).sum() / effective.sum().clamp_min(1e-12)
    return weighted_loss, ordinary, -weighted_loss


def _model_loss_and_score(model: Any, torch: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
    weights = batch.pop("loss_weights", None)
    labels = batch.pop("labels")
    logits = model(**batch).logits
    return _loss_and_score(torch, logits, labels, weights)


def _model_loss(model: Any, torch: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    weighted, ordinary, _ = _model_loss_and_score(model, torch, batch)
    return weighted, ordinary


def _negative_batch(
    torch: Any, records: list[dict[str, Any]], pad_token_id: int, device: str
) -> dict[str, Any]:
    if any("negative" not in row for row in records):
        raise ValueError("pairwise ranking requires one negative target per record")
    return _batch(torch, [row["negative"] for row in records], pad_token_id, device)


def pairwise_rank_terms(
    torch: Any, positive_score: Any, negative_score: Any, temperature: float
) -> tuple[Any, Any]:
    """Return logistic rank loss and its detached score-gradient coefficient."""

    delta = (negative_score - positive_score) / temperature
    return torch.nn.functional.softplus(delta), torch.sigmoid(delta).detach() / temperature


def _mean_loss(
    model: Any,
    torch: Any,
    rows: list[dict[str, Any]],
    pad: int,
    device: str,
    *,
    pairwise_rank_weight: float = 0.0,
    pairwise_temperature: float = 1.0,
) -> tuple[float, float]:
    model.eval()
    losses = []
    ordinary_losses = []
    with torch.no_grad():
        for row in rows:
            loss, ordinary, positive_score = _model_loss_and_score(
                model, torch, _batch(torch, [row], pad, device)
            )
            if pairwise_rank_weight:
                _, _, negative_score = _model_loss_and_score(
                    model, torch, _negative_batch(torch, [row], pad, device)
                )
                rank_loss = torch.nn.functional.softplus(
                    (negative_score - positive_score) / pairwise_temperature
                )
                loss = loss + pairwise_rank_weight * rank_loss
            value = float(loss.detach().cpu())
            ordinary_value = float(ordinary.detach().cpu())
            if not math.isfinite(value) or not math.isfinite(ordinary_value):
                raise FloatingPointError("non-finite development loss")
            losses.append(value)
            ordinary_losses.append(ordinary_value)
    model.train()
    return sum(losses) / len(losses), sum(ordinary_losses) / len(ordinary_losses)


def train_lora(
    *,
    model: dict[str, Any],
    training: LoRATrainingConfig,
    train_path: str | Path,
    dev_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train and save one protocol adapter without teaching answer values."""

    training.validate()
    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists() and any(adapter_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite adapter: {adapter_dir}")
    try:
        import torch
        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("LoRA training requires torch, transformers, and peft") from error

    model_id = str(model["model_id"])
    revision = str(model["revision"])
    if len(revision) != 40:
        raise ValueError("model revision must be a pinned 40-character SHA")
    device = select_device(torch, training.device)
    if training.device == "xpu" and device != "xpu":
        raise RuntimeError("the requested XPU training device is unavailable")
    if training.device == "directml" and not str(device).startswith("privateuseone"):
        raise RuntimeError("the requested DirectML training device is unavailable")
    model_dtype = str(model.get("dtype", training.dtype))
    dtype = getattr(torch, model_dtype)
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = []
    for source_row in read_jsonl(train_path):
        if training.pairwise_rank_weight and source_row.get("negative_target") is None:
            raise ValueError(
                f"pairwise ranking is enabled but {source_row['example_id']} has no negative"
            )
        tokenized = _tokenize_record(
            tokenizer,
            source_row,
            training.max_length,
            reject_truncation=training.reject_truncation,
            semantic_token_weights=training.semantic_token_weights,
        )
        if training.logical_epoch_field is not None:
            if training.logical_epoch_field not in source_row:
                raise ValueError(
                    f"logical epoch field {training.logical_epoch_field!r} is absent from "
                    f"{source_row['example_id']}"
                )
            tokenized["logical_epoch"] = int(source_row[training.logical_epoch_field])
        train_rows.append(tokenized)
    dev_rows = [
        _tokenize_record(
            tokenizer,
            row,
            training.max_length,
            reject_truncation=training.reject_truncation,
            semantic_token_weights=training.semantic_token_weights,
        )
        for row in read_jsonl(dev_path)
    ]
    if training.pairwise_rank_weight and any("negative" not in row for row in dev_rows):
        raise ValueError("pairwise ranking requires development negatives")
    logical_epoch_rows: dict[int, list[dict[str, Any]]] | None = None
    if training.logical_epoch_field is not None:
        logical_epoch_rows = {
            epoch: [row for row in train_rows if row["logical_epoch"] == epoch]
            for epoch in range(training.epochs)
        }
        observed = {row["logical_epoch"] for row in train_rows}
        expected = set(range(training.epochs))
        if observed != expected or any(not rows for rows in logical_epoch_rows.values()):
            raise ValueError(
                f"logical epochs must be exactly {sorted(expected)}; observed={sorted(observed)}"
            )

    model_kwargs: dict[str, Any] = {"revision": revision, "dtype": dtype}
    if model.get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = str(model["attn_implementation"])
    base = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    base.config.use_cache = False
    peft_config = LoraConfig(
        r=training.rank,
        lora_alpha=training.alpha,
        lora_dropout=training.dropout,
        target_modules=list(training.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_instance = get_peft_model(base, peft_config).to(device)
    if training.gradient_checkpointing:
        model_instance.enable_input_require_grads()
        model_instance.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model_instance.train()
    trainable_parameters = sum(
        parameter.numel() for parameter in model_instance.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model_instance.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model_instance.parameters() if parameter.requires_grad),
        lr=training.learning_rate,
    )
    checkpoint_path = output_dir / "checkpoint_last.pt"
    train_sha256 = file_sha256(train_path)
    dev_sha256 = file_sha256(dev_path)
    checkpoint_identity = {
        "model_id": model_id,
        "model_revision": revision,
        "adapter_id": str(model["adapter_id"]),
        "training": training.to_dict(),
        "train_sha256": train_sha256,
        "dev_sha256": dev_sha256,
    }
    start_epoch = 0
    start_batch_index = 0
    resume_losses: list[float] = []
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    resumed_from_optimizer_step = 0
    previous_wall_time_seconds = 0.0
    best_dev_loss = math.inf
    best_epoch: int | None = None
    best_adapter_state: dict[str, Any] | None = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != checkpoint_identity:
            raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
        set_peft_model_state_dict(model_instance, checkpoint["adapter_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch_index = int(checkpoint["next_batch_index"])
        resume_losses = [float(value) for value in checkpoint["losses"]]
        history = list(checkpoint["history"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        resumed_from_optimizer_step = optimizer_steps
        previous_wall_time_seconds = float(checkpoint.get("wall_time_seconds", 0.0))
        best_dev_loss = float(checkpoint.get("best_dev_loss", math.inf))
        best_epoch = checkpoint.get("best_epoch")
        best_adapter_state = checkpoint.get("best_adapter_state")
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if device == "xpu" and checkpoint.get("xpu_rng_state") is not None:
            torch.xpu.set_rng_state(checkpoint["xpu_rng_state"])
        print(
            f"resuming at epoch {start_epoch + 1}, batch {start_batch_index}, "
            f"optimizer step {optimizer_steps}"
        )
    if device == "xpu":
        torch.xpu.reset_peak_memory_stats()
    started = time.perf_counter()

    def save_resume(epoch: int, next_batch_index: int, losses: list[float]) -> None:
        adapter_state = {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(model_instance).items()
        }
        state = {
            "schema_version": "ccpu.paper1.lora_checkpoint.v1",
            "identity": checkpoint_identity,
            "adapter_state": adapter_state,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "losses": losses,
            "history": history,
            "optimizer_steps": optimizer_steps,
            "wall_time_seconds": previous_wall_time_seconds
            + (time.perf_counter() - started),
            "best_dev_loss": best_dev_loss,
            "best_epoch": best_epoch,
            "best_adapter_state": best_adapter_state,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "xpu_rng_state": torch.xpu.get_rng_state() if device == "xpu" else None,
        }
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(state, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, training.epochs):
        epoch_rows = logical_epoch_rows[epoch] if logical_epoch_rows is not None else train_rows
        order = list(range(len(epoch_rows)))
        random.Random(training.seed + epoch).shuffle(order)
        losses = resume_losses if epoch == start_epoch else []
        ordinary_losses = []
        rank_losses = []
        rank_correct = []
        batch_starts = list(range(0, len(order), training.batch_size))
        for batch_index, start in enumerate(batch_starts):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            members = [epoch_rows[index] for index in order[start : start + training.batch_size]]
            loss, ordinary_loss, positive_score = _model_loss_and_score(
                model_instance,
                torch,
                _batch(torch, members, tokenizer.pad_token_id, device),
            )
            rank_loss = None
            if training.pairwise_rank_weight:
                cpu_rng_state = torch.get_rng_state()
                xpu_rng_state = torch.xpu.get_rng_state() if device == "xpu" else None
                with torch.no_grad():
                    _, _, negative_score_probe = _model_loss_and_score(
                        model_instance,
                        torch,
                        _negative_batch(torch, members, tokenizer.pad_token_id, device),
                    )
                torch.set_rng_state(cpu_rng_state)
                if device == "xpu" and xpu_rng_state is not None:
                    torch.xpu.set_rng_state(xpu_rng_state)
                rank_loss, coefficient = pairwise_rank_terms(
                    torch,
                    positive_score.detach(),
                    negative_score_probe,
                    training.pairwise_temperature,
                )
                positive_objective = (
                    loss
                    - training.pairwise_rank_weight * coefficient * positive_score
                )
                (positive_objective / training.gradient_accumulation_steps).backward()
                _, _, negative_score = _model_loss_and_score(
                    model_instance,
                    torch,
                    _negative_batch(torch, members, tokenizer.pad_token_id, device),
                )
                negative_objective = (
                    training.pairwise_rank_weight * coefficient * negative_score
                )
                (negative_objective / training.gradient_accumulation_steps).backward()
                objective_for_report = loss + training.pairwise_rank_weight * rank_loss
                rank_losses.append(float(rank_loss.detach().cpu()))
                rank_correct.append(
                    float((positive_score.detach() > negative_score_probe).detach().cpu())
                )
            else:
                (loss / training.gradient_accumulation_steps).backward()
                objective_for_report = loss
            loss_value = float(objective_for_report.detach().cpu())
            ordinary_loss_value = float(ordinary_loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch + 1}, batch {batch_index}"
                )
            losses.append(loss_value)
            ordinary_losses.append(ordinary_loss_value)
            final_batch = start + training.batch_size >= len(order)
            if (batch_index + 1) % training.gradient_accumulation_steps == 0 or final_batch:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model_instance.parameters(), 1.0, foreach=False
                )
                if not math.isfinite(float(gradient_norm.detach().cpu())):
                    raise FloatingPointError(
                        f"non-finite gradient norm at epoch {epoch + 1}, "
                        f"batch {batch_index}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                interval = training.checkpoint_every_optimizer_steps
                if interval and optimizer_steps % interval == 0:
                    save_resume(epoch, batch_index + 1, losses)
                    print(
                        f"checkpoint: epoch {epoch + 1}, batch {batch_index + 1}/"
                        f"{len(batch_starts)}, optimizer step {optimizer_steps}"
                    )
        should_evaluate = training.evaluate_each_epoch or epoch + 1 == training.epochs
        dev_losses = (
                _mean_loss(model_instance, torch, dev_rows, tokenizer.pad_token_id, device)
                if not training.pairwise_rank_weight
                else _mean_loss(
                    model_instance,
                    torch,
                    dev_rows,
                    tokenizer.pad_token_id,
                    device,
                    pairwise_rank_weight=training.pairwise_rank_weight,
                    pairwise_temperature=training.pairwise_temperature,
                )
            if should_evaluate
            else None
        )
        dev_loss = dev_losses[0] if dev_losses is not None else None
        ordinary_dev_loss = dev_losses[1] if dev_losses is not None else None
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": sum(losses) / len(losses),
                "mean_dev_loss": dev_loss,
                "mean_unweighted_train_loss": sum(ordinary_losses) / len(ordinary_losses),
                "mean_unweighted_dev_loss": ordinary_dev_loss,
                "mean_pairwise_rank_loss": (
                    sum(rank_losses) / len(rank_losses) if rank_losses else None
                ),
                "pairwise_train_accuracy": (
                    sum(rank_correct) / len(rank_correct) if rank_correct else None
                ),
            }
        )
        if (
            training.restore_best_dev
            and dev_loss is not None
            and dev_loss < best_dev_loss
        ):
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
            best_adapter_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(model_instance).items()
            }
        dev_text = f"{dev_loss:.4f}" if dev_loss is not None else "deferred"
        print(
            f"epoch {epoch + 1}/{training.epochs}: "
            f"train={history[-1]['mean_train_loss']:.4f} dev={dev_text}"
        )
        if training.checkpoint_every_optimizer_steps:
            save_resume(epoch + 1, 0, [])
        start_batch_index = 0
        resume_losses = []

    if device == "xpu":
        torch.xpu.synchronize()
    wall_time_seconds = previous_wall_time_seconds + (time.perf_counter() - started)
    if training.restore_best_dev:
        if best_adapter_state is None or best_epoch is None:
            raise RuntimeError("best-dev restoration did not observe a development loss")
        set_peft_model_state_dict(model_instance, best_adapter_state)
    adapter_dir.mkdir(parents=True, exist_ok=False)
    adapter_state = {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(model_instance).items()
    }
    model_instance.save_pretrained(
        adapter_dir, state_dict=adapter_state, safe_serialization=True
    )
    adapter_files = {
        path.name: file_sha256(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()
    }
    effective_training = training.to_dict()
    effective_training["dtype"] = model_dtype
    report = {
        "schema_version": "ccpu.paper1.lora_training.v1",
        "model_id": model_id,
        "model_revision": revision,
        "adapter_id": str(model["adapter_id"]),
        "device": str(device),
        "dtype": model_dtype,
        "training": effective_training,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_target_tokens_per_epoch": (
            [
                sum(row["target_tokens"] for row in logical_epoch_rows[epoch])
                for epoch in range(training.epochs)
            ]
            if logical_epoch_rows is not None
            else sum(row["target_tokens"] for row in train_rows)
        ),
        "training_target_tokens": (
            sum(row["target_tokens"] for row in train_rows)
            if logical_epoch_rows is not None
            else training.epochs * sum(row["target_tokens"] for row in train_rows)
        ),
        "optimizer_steps": optimizer_steps,
        "resumed_from_optimizer_step": resumed_from_optimizer_step,
        "selected_epoch": best_epoch if training.restore_best_dev else training.epochs,
        "selection_policy": (
            "minimum_weighted_dev_loss"
            if training.restore_best_dev
            else "final_epoch"
        ),
        "selected_dev_loss": (
            best_dev_loss if training.restore_best_dev else history[-1]["mean_dev_loss"]
        ),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "wall_time_seconds": wall_time_seconds,
        "peak_memory_bytes": (
            int(torch.xpu.max_memory_allocated()) if device == "xpu" else None
        ),
        "history": history,
        "token_audit": {
            "train_max_full_tokens": max(row["full_tokens"] for row in train_rows),
            "train_max_prefix_tokens": max(row["prefix_tokens"] for row in train_rows),
            "train_truncated_rows": sum(row["was_truncated"] for row in train_rows),
            "dev_max_full_tokens": max(row["full_tokens"] for row in dev_rows),
            "dev_max_prefix_tokens": max(row["prefix_tokens"] for row in dev_rows),
            "dev_truncated_rows": sum(row["was_truncated"] for row in dev_rows),
        },
        "train_sha256": train_sha256,
        "dev_sha256": dev_sha256,
        "adapter_files": adapter_files,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
        },
    }
    write_json(output_dir / "training_report.json", report)
    return report
