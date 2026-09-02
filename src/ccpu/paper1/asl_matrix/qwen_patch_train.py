"""Restartable LoRA training for Qwen external-ASL memory patches."""

from __future__ import annotations

import importlib.metadata
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json
from ccpu.paper1.generation import select_device
from ccpu.paper1.lora_train import LoRATrainingConfig, _batch, _tokenize_record

from .qwen_patch import QwenPatchController, install_qwen_memory_patches

_EXTERNAL_MARKER = "\n\nExternal ASL teacher:\n"
_TARGET_MARKER = "\n\nASL:"


def split_patch_record(row: dict[str, Any]) -> tuple[str, str | None]:
    """Separate the fixed NL prompt from serialized Q1 external memory."""

    prompt = str(row["prompt"])
    has_external = bool(row["has_external_asl"])
    if not has_external:
        if _EXTERNAL_MARKER in prompt:
            raise ValueError(f"autonomous row contains external memory: {row['example_id']}")
        return prompt, None
    if _EXTERNAL_MARKER not in prompt or not prompt.endswith(_TARGET_MARKER):
        raise ValueError(f"teacher row has malformed markers: {row['example_id']}")
    local_prompt, external = prompt.split(_EXTERNAL_MARKER, 1)
    external = external[: -len(_TARGET_MARKER)]
    if not external.strip():
        raise ValueError(f"teacher row has empty external memory: {row['example_id']}")
    return local_prompt + _TARGET_MARKER, external


def _tokenize_external(
    tokenizer: Any,
    external: str | None,
    *,
    max_length: int,
    example_id: str,
) -> tuple[list[int], list[int], int]:
    if external is None:
        return [tokenizer.pad_token_id], [0], 0
    text = f"External ASL teacher:\n{external}"
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(input_ids) > max_length:
        raise ValueError(
            f"external memory exceeds max_length for {example_id}: "
            f"{len(input_ids)} > {max_length}"
        )
    return input_ids, [1] * len(input_ids), len(input_ids)


def _tokenize_rows(
    tokenizer: Any,
    path: str | Path,
    training: LoRATrainingConfig,
    *,
    max_external_length: int,
) -> list[dict[str, Any]]:
    records = []
    for row in read_jsonl(path):
        local_prompt, external = split_patch_record(row)
        record = _tokenize_record(
            tokenizer,
            {
                "example_id": row["example_id"],
                "prompt": local_prompt,
                "target": row["target"],
            },
            training.max_length,
            reject_truncation=training.reject_truncation,
        )
        ids, mask, tokens = _tokenize_external(
            tokenizer,
            external,
            max_length=max_external_length,
            example_id=str(row["example_id"]),
        )
        record.update(
            {
                "external_input_ids": ids,
                "external_attention_mask": mask,
                "external_tokens": tokens,
                "regime": row["regime"],
            }
        )
        records.append(record)
    return records


def _external_batch(
    torch: Any,
    rows: list[dict[str, Any]],
    pad_token_id: int,
    device: str,
) -> tuple[Any, Any]:
    length = max(len(row["external_input_ids"]) for row in rows)
    ids = []
    masks = []
    for row in rows:
        padding = length - len(row["external_input_ids"])
        ids.append(row["external_input_ids"] + [pad_token_id] * padding)
        masks.append(row["external_attention_mask"] + [0] * padding)
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(model: Any, state: dict[str, Any]) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(
            f"trainable checkpoint keys differ; missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in parameters.items():
        parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def _mean_autonomous_loss(
    model: Any,
    controller: QwenPatchController,
    torch: Any,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    training: LoRATrainingConfig,
    device: str,
) -> float:
    model.eval()
    controller.clear()
    losses = []
    with torch.no_grad():
        for start in range(0, len(rows), training.batch_size):
            members = rows[start : start + training.batch_size]
            loss = model(**_batch(torch, members, tokenizer.pad_token_id, device)).loss
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError("non-finite autonomous development loss")
            losses.append(value)
    model.train()
    return sum(losses) / len(losses)


def train_qwen_patch(
    *,
    config_path: str | Path,
    train_path: str | Path,
    dev_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train one Q2/Q3 cell from the exact serialized Q1 runtime views."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Qwen patch training requires torch, transformers, peft, and safetensors"
        ) from error

    config_path = Path(config_path)
    config = read_json(config_path)
    training = LoRATrainingConfig.from_dict(config)
    training.validate()
    if training.epochs != 1:
        raise ValueError("the persisted mixed-view stream requires epochs=1")
    if training.gradient_checkpointing:
        raise ValueError("Qwen memory patches require gradient_checkpointing=false")
    patch_spec = config["patch"]
    patch_mode = str(patch_spec["mode"])
    layer_indices = tuple(int(value) for value in patch_spec["layer_indices"])
    max_external_length = int(patch_spec.get("max_external_length", 640))
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("patch layer_indices must be non-empty and unique")
    if min(layer_indices) < 0 or max_external_length < 16:
        raise ValueError("patch layer indices or external length are invalid")
    model_spec = config["model"]
    model_id = str(model_spec["model_id"])
    revision = str(model_spec["revision"])
    if len(revision) != 40:
        raise ValueError("model revision must be a pinned 40-character SHA")

    output = Path(output_dir)
    adapter_dir = output / "adapter"
    report_path = output / "training_report.json"
    if report_path.exists() or (adapter_dir.exists() and any(adapter_dir.iterdir())):
        raise FileExistsError(f"completed Qwen patch run already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(torch, training.device)
    if training.device == "xpu" and device != "xpu":
        raise RuntimeError("the requested XPU training device is unavailable")
    dtype_name = str(model_spec.get("dtype", training.dtype))
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    if device == "xpu":
        torch.xpu.manual_seed_all(training.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = _tokenize_rows(
        tokenizer,
        train_path,
        training,
        max_external_length=max_external_length,
    )
    dev_rows = _tokenize_rows(
        tokenizer,
        dev_path,
        training,
        max_external_length=max_external_length,
    )
    if any(row["regime"] != "autonomous" for row in dev_rows):
        raise ValueError("Qwen patch development rows must be autonomous")

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
    model = get_peft_model(base, peft_config)
    controller = install_qwen_memory_patches(
        model,
        mode=patch_mode,
        layer_indices=layer_indices,
    )
    model = model.to(device)
    controller.model = model
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training.learning_rate,
    )

    train_sha256 = file_sha256(train_path)
    dev_sha256 = file_sha256(dev_path)
    identity = {
        "config_sha256": file_sha256(config_path),
        "train_sha256": train_sha256,
        "dev_sha256": dev_sha256,
    }
    checkpoint_path = output / "checkpoint_last.pt"
    start_batch_index = 0
    losses: list[float] = []
    optimizer_steps = 0
    resumed_from_optimizer_step = 0
    previous_wall_time = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != identity:
            raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
        _load_trainable_state(model, checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_batch_index = int(checkpoint["next_batch_index"])
        losses = [float(value) for value in checkpoint["losses"]]
        optimizer_steps = int(checkpoint["optimizer_steps"])
        resumed_from_optimizer_step = optimizer_steps
        previous_wall_time = float(checkpoint.get("wall_time_seconds", 0.0))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if device == "xpu" and checkpoint.get("xpu_rng_state") is not None:
            torch.xpu.set_rng_state(checkpoint["xpu_rng_state"])
        print(
            f"resuming at batch {start_batch_index}, optimizer step {optimizer_steps}"
        )
    if device == "xpu":
        torch.xpu.reset_peak_memory_stats()
    started = time.perf_counter()

    def save_resume(next_batch_index: int) -> None:
        state = {
            "schema_version": "ccpu.paper1.qwen_patch_checkpoint.v1",
            "identity": identity,
            "trainable_state": _trainable_state(model),
            "optimizer_state": optimizer.state_dict(),
            "next_batch_index": next_batch_index,
            "losses": losses,
            "optimizer_steps": optimizer_steps,
            "wall_time_seconds": previous_wall_time + time.perf_counter() - started,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "xpu_rng_state": torch.xpu.get_rng_state() if device == "xpu" else None,
        }
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, checkpoint_path)

    order = list(range(len(train_rows)))
    random.Random(training.seed).shuffle(order)
    batch_starts = list(range(0, len(order), training.batch_size))
    optimizer.zero_grad(set_to_none=True)
    for batch_index, start in enumerate(batch_starts):
        if batch_index < start_batch_index:
            continue
        members = [train_rows[index] for index in order[start : start + training.batch_size]]
        external_ids, external_mask = _external_batch(
            torch, members, tokenizer.pad_token_id, device
        )
        controller.capture_external(external_ids, external_mask)
        loss = model(**_batch(torch, members, tokenizer.pad_token_id, device)).loss
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}")
        (loss / training.gradient_accumulation_steps).backward()
        losses.append(value)
        final_batch = start + training.batch_size >= len(order)
        if (batch_index + 1) % training.gradient_accumulation_steps == 0 or final_batch:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(gradient_norm.detach().cpu())):
                raise FloatingPointError(f"non-finite gradient norm at batch {batch_index}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            interval = training.checkpoint_every_optimizer_steps
            if interval and optimizer_steps % interval == 0:
                save_resume(batch_index + 1)
                print(
                    f"checkpoint: batch {batch_index + 1}/{len(batch_starts)}, "
                    f"optimizer step {optimizer_steps}"
                )
        controller.clear()

    autonomous_dev_loss = _mean_autonomous_loss(
        model, controller, torch, dev_rows, tokenizer, training, device
    )
    if training.checkpoint_every_optimizer_steps:
        save_resume(len(batch_starts))
    if device == "xpu":
        torch.xpu.synchronize()
    wall_time = previous_wall_time + time.perf_counter() - started
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    trainable_path = output / "trainable_patch_state.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in _trainable_state(model).items()},
        trainable_path,
    )
    parameter_report = controller.parameter_report()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_report.update(
        {
            "lora_trainable_parameters": parameter_report["other_trainable_parameters"],
            "frozen_parameters": total_parameters
            - parameter_report["total_trainable_parameters"],
            "total_parameters": total_parameters,
        }
    )
    adapter_files = {
        path.name: file_sha256(path)
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    }
    report = {
        "schema_version": "ccpu.paper1.qwen_patch_training.v1",
        "run_id": config["run_id"],
        "model": model_spec,
        "patch": patch_spec,
        "training": training.to_dict(),
        "device": device,
        "dtype": dtype_name,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "regime_counts": {
            name: sum(row["regime"] == name for row in train_rows)
            for name in ("full", "partial", "autonomous")
        },
        "training_target_tokens": sum(row["target_tokens"] for row in train_rows),
        "teacher_tokens_processed": sum(row["external_tokens"] for row in train_rows),
        "mean_train_loss": sum(losses) / len(losses),
        "autonomous_dev_loss": autonomous_dev_loss,
        "optimizer_steps": optimizer_steps,
        "resumed_from_optimizer_step": resumed_from_optimizer_step,
        "wall_time_seconds": wall_time,
        "peak_memory_bytes": (
            int(torch.xpu.max_memory_allocated()) if device == "xpu" else None
        ),
        "parameter_report": parameter_report,
        "token_audit": {
            "train_max_full_tokens": max(row["full_tokens"] for row in train_rows),
            "train_max_external_tokens": max(
                row["external_tokens"] for row in train_rows
            ),
            "train_truncated_rows": sum(row["was_truncated"] for row in train_rows),
            "dev_max_full_tokens": max(row["full_tokens"] for row in dev_rows),
            "dev_truncated_rows": sum(row["was_truncated"] for row in dev_rows),
        },
        "config_sha256": identity["config_sha256"],
        "train_sha256": train_sha256,
        "dev_sha256": dev_sha256,
        "trainable_patch_state_sha256": file_sha256(trainable_path),
        "adapter_files": adapter_files,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "safetensors")
        },
    }
    write_json(report_path, report)
    return report


def load_qwen_patch_backend(
    config_path: str | Path,
    state_path: str | Path,
    *,
    max_new_tokens: int = 384,
) -> Any:
    """Reconstruct one trained patch behind the common generation backend."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Qwen patch evaluation requires torch, transformers, peft, and safetensors"
        ) from error
    from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig

    config = read_json(config_path)
    training = LoRATrainingConfig.from_dict(config)
    model_spec = config["model"]
    revision = str(model_spec["revision"])
    dtype_name = str(model_spec.get("dtype", training.dtype))
    tokenizer = AutoTokenizer.from_pretrained(model_spec["model_id"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_spec["model_id"],
        revision=revision,
        dtype=getattr(torch, dtype_name),
    )
    peft_config = LoraConfig(
        r=training.rank,
        lora_alpha=training.alpha,
        lora_dropout=training.dropout,
        target_modules=list(training.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, peft_config)
    patch_spec = config["patch"]
    controller = install_qwen_memory_patches(
        model,
        mode=str(patch_spec["mode"]),
        layer_indices=tuple(int(value) for value in patch_spec["layer_indices"]),
    )
    _load_trainable_state(model, load_file(state_path, device="cpu"))
    controller.clear()
    generation = HuggingFaceGenerationConfig(
        model_id=str(model_spec["model_id"]),
        revision=revision,
        max_new_tokens=max_new_tokens,
        device=training.device,
        dtype=dtype_name,
        use_chat_template=True,
        enable_thinking=False,
        adapter_path=str(state_path),
        adapter_id=str(model_spec["adapter_id"]),
        cached_generation=True,
    )
    return HuggingFaceBackend.from_components(
        generation,
        tokenizer=tokenizer,
        model=model,
    )


def evaluate_qwen_patch(
    *,
    config_path: str | Path,
    state_path: str | Path,
    eval_path: str | Path,
    train_split_path: str | Path,
    output_dir: str | Path,
    seed: int = 44017,
    checkpoint_every: int = 5,
) -> dict[str, Any]:
    """Run the frozen autonomous ASL evaluation with a Q2/Q3 patch."""

    from ccpu.paper1.asl_pilot_eval import run_asl_pilot

    config = read_json(config_path)
    model_spec = config["model"]
    backend = load_qwen_patch_backend(config_path, state_path)
    model_config = {
        "model": {
            **model_spec,
            "device": config["training"]["device"],
            "max_new_tokens": 384,
            "use_chat_template": True,
            "enable_thinking": False,
        },
        "adapter_path": str(state_path),
        "adapter_id": str(model_spec["adapter_id"]),
    }
    report = run_asl_pilot(
        eval_path=eval_path,
        train_split_path=train_split_path,
        model_config=model_config,
        condition="lora",
        shots=0,
        output_dir=output_dir,
        seed=seed,
        checkpoint_every=checkpoint_every,
        backend_override=backend,
    )
    report["qwen_patch"] = {
        "run_id": config["run_id"],
        "mode": config["patch"]["mode"],
        "layer_indices": config["patch"]["layer_indices"],
        "config_sha256": file_sha256(config_path),
        "state_sha256": file_sha256(state_path),
    }
    write_json(Path(output_dir) / "summary.json", report)
    return report
