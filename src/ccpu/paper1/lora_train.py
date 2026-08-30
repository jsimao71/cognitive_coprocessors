"""Minimal PEFT LoRA training for calculator-block interface behavior."""

from __future__ import annotations

import importlib.metadata
import random
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
        )

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, batch size, and accumulation steps must be positive")
        if self.max_length < 32 or self.rank < 1 or self.alpha < 1:
            raise ValueError("max length, rank, and alpha are invalid")
        if not 0 <= self.dropout < 1 or not self.target_modules:
            raise ValueError("dropout or target modules are invalid")

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


def _tokenize_record(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    user = {"role": "user", "content": str(row["prompt"])}
    prefix = _chat_text(tokenizer, [user], add_generation_prompt=True)
    complete = _chat_text(
        tokenizer,
        [user, {"role": "assistant", "content": str(row["target"])}],
        add_generation_prompt=False,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(complete, add_special_tokens=False)["input_ids"]
    common = 0
    for prefix_token, full_token in zip(prefix_ids, input_ids):
        if prefix_token != full_token:
            break
        common += 1
    if common < max(1, len(prefix_ids) - 2):
        raise ValueError(f"chat template prefix mismatch for {row['example_id']}")
    input_ids = input_ids[:max_length]
    labels = [-100] * min(common, len(input_ids)) + input_ids[common:]
    if not any(label != -100 for label in labels):
        raise ValueError(f"target truncated for {row['example_id']}")
    return {
        "example_id": row["example_id"],
        "input_ids": input_ids,
        "labels": labels,
        "target_tokens": sum(label != -100 for label in labels),
    }


def _batch(torch: Any, records: list[dict[str, Any]], pad_token_id: int, device: str):
    length = max(len(row["input_ids"]) for row in records)
    input_ids = []
    attention_mask = []
    labels = []
    for row in records:
        padding = length - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _mean_loss(model: Any, torch: Any, rows: list[dict[str, Any]], pad: int, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for row in rows:
            loss = model(**_batch(torch, [row], pad, device)).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


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
        from peft import LoraConfig, get_peft_model
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
    model_dtype = str(model.get("dtype", training.dtype))
    dtype = getattr(torch, model_dtype)
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = [
        _tokenize_record(tokenizer, row, training.max_length) for row in read_jsonl(train_path)
    ]
    dev_rows = [
        _tokenize_record(tokenizer, row, training.max_length) for row in read_jsonl(dev_path)
    ]

    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
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
    if device == "xpu":
        torch.xpu.reset_peak_memory_stats()
    started = time.perf_counter()
    history = []
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(training.epochs):
        order = list(range(len(train_rows)))
        random.Random(training.seed + epoch).shuffle(order)
        losses = []
        batch_starts = range(0, len(order), training.batch_size)
        for batch_index, start in enumerate(batch_starts):
            members = [train_rows[index] for index in order[start : start + training.batch_size]]
            loss = model_instance(
                **_batch(torch, members, tokenizer.pad_token_id, device)
            ).loss
            (loss / training.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))
            final_batch = start + training.batch_size >= len(order)
            if (batch_index + 1) % training.gradient_accumulation_steps == 0 or final_batch:
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        should_evaluate = training.evaluate_each_epoch or epoch + 1 == training.epochs
        dev_loss = (
            _mean_loss(model_instance, torch, dev_rows, tokenizer.pad_token_id, device)
            if should_evaluate
            else None
        )
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": sum(losses) / len(losses),
                "mean_dev_loss": dev_loss,
            }
        )
        dev_text = f"{dev_loss:.4f}" if dev_loss is not None else "deferred"
        print(
            f"epoch {epoch + 1}/{training.epochs}: "
            f"train={history[-1]['mean_train_loss']:.4f} dev={dev_text}"
        )

    if device == "xpu":
        torch.xpu.synchronize()
    wall_time_seconds = time.perf_counter() - started
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model_instance.save_pretrained(adapter_dir, safe_serialization=True)
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
        "device": device,
        "dtype": model_dtype,
        "training": effective_training,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_target_tokens_per_epoch": sum(row["target_tokens"] for row in train_rows),
        "training_target_tokens": training.epochs
        * sum(row["target_tokens"] for row in train_rows),
        "optimizer_steps": optimizer_steps,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "wall_time_seconds": wall_time_seconds,
        "peak_memory_bytes": (
            int(torch.xpu.max_memory_allocated()) if device == "xpu" else None
        ),
        "history": history,
        "train_sha256": file_sha256(train_path),
        "dev_sha256": file_sha256(dev_path),
        "adapter_files": adapter_files,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
        },
    }
    write_json(output_dir / "training_report.json", report)
    return report
