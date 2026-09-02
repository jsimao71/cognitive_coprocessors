"""Resumable full-model training for the staged ASL architecture ladder."""

from __future__ import annotations

import importlib.metadata
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json
from ccpu.paper1.generation import select_device

from .data import MatrixExample, RegimeBuilder, StaticMixture
from .model import ASLMatrixModel


@dataclass(frozen=True)
class MatrixTrainingConfig:
    epochs: int = 8
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_nl_length: int = 256
    max_asl_length: int = 256
    max_target_length: int = 256
    seed: int = 11
    device: str = "xpu"
    dtype: str = "float16"
    early_stopping_patience: int = 2

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MatrixTrainingConfig:
        data = value.get("training", value)
        return cls(
            epochs=int(data.get("epochs", 8)),
            batch_size=int(data.get("batch_size", 1)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 8)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            weight_decay=float(data.get("weight_decay", 0.01)),
            max_nl_length=int(data.get("max_nl_length", 256)),
            max_asl_length=int(data.get("max_asl_length", 256)),
            max_target_length=int(data.get("max_target_length", 256)),
            seed=int(data.get("seed", 11)),
            device=str(data.get("device", "xpu")),
            dtype=str(data.get("dtype", "float16")),
            early_stopping_patience=int(data.get("early_stopping_patience", 2)),
        )

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, batch size, and accumulation steps must be positive")
        if min(self.max_nl_length, self.max_asl_length, self.max_target_length) < 16:
            raise ValueError("matrix sequence lengths must be at least 16")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")


def _examples(path: str | Path) -> list[MatrixExample]:
    return [MatrixExample(**row) for row in read_jsonl(path)]


def _mixture(config: dict[str, Any]) -> StaticMixture:
    values = config["training"]["static"]
    return StaticMixture(
        full_teacher=float(values["full_teacher"]),
        partial_teacher=float(values["partial_teacher"]),
        autonomous=float(values["autonomous"]),
    )


def _regime_builder(config: dict[str, Any], seed: int) -> RegimeBuilder:
    corruption = config["training"]["corruption"]
    return RegimeBuilder(
        mixture=_mixture(config),
        corruption_policies=tuple(str(item) for item in corruption["policy"]),
        corruption_severity=float(corruption["severity"]),
        seed=seed,
    )


def _token_length_audit(
    tokenizer: Any,
    examples: list[MatrixExample],
    training: MatrixTrainingConfig,
    *,
    split: str,
) -> dict[str, Any]:
    """Fail closed when any scientific input or target would be truncated."""

    lengths = {"nl": [], "external_asl": [], "target": []}
    for example in examples:
        lengths["nl"].append(
            len(tokenizer(f"translate natural language to ASL: {example.nl_input}").input_ids)
        )
        lengths["external_asl"].append(
            len(tokenizer(f"external ASL teacher: {example.target_asl}").input_ids)
        )
        lengths["target"].append(len(tokenizer(example.target_asl).input_ids))
    limits = {
        "nl": training.max_nl_length,
        "external_asl": training.max_asl_length,
        "target": training.max_target_length,
    }
    report = {"split": split, "rows": len(examples), "fields": {}}
    violations = []
    for name, values in lengths.items():
        ordered = sorted(values)
        over_limit = sum(value > limits[name] for value in values)
        report["fields"][name] = {
            "limit": limits[name],
            "maximum": max(values),
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "over_limit": over_limit,
        }
        if over_limit:
            violations.append(f"{name}={over_limit}/{len(values)}")
    if violations:
        details = ", ".join(violations)
        raise ValueError(f"{split} matrix rows would be truncated: {details}")
    return report


def _views(
    examples: list[MatrixExample],
    builder: RegimeBuilder,
    *,
    epoch: int,
    forced_regime: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(examples):
        regime = forced_regime or builder.sample_regime(example, epoch=epoch)
        noise = examples[(index + 1) % len(examples)].target_asl
        rows.append(builder.make_view(example, regime=regime, epoch=epoch, noise_asl=noise))
    return rows


def _batch(
    tokenizer: Any,
    views: list[dict[str, Any]],
    training: MatrixTrainingConfig,
    device: str,
) -> dict[str, Any]:
    nl = tokenizer(
        [f"translate natural language to ASL: {row['nl_input']}" for row in views],
        padding=True,
        truncation=True,
        max_length=training.max_nl_length,
        return_tensors="pt",
    )
    external_text = [
        f"external ASL teacher: {row['external_asl_input']}"
        if row["external_asl_input"] is not None
        else ""
        for row in views
    ]
    external = tokenizer(
        external_text,
        padding=True,
        truncation=True,
        max_length=training.max_asl_length,
        return_tensors="pt",
    )
    for index, row in enumerate(views):
        if not row["has_external_asl"]:
            external["attention_mask"][index].zero_()
            external["input_ids"][index].fill_(tokenizer.pad_token_id)
    target = tokenizer(
        text_target=[row["target_asl"] for row in views],
        padding=True,
        truncation=True,
        max_length=training.max_target_length,
        return_tensors="pt",
    )
    labels = target["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    if any(not (labels[index] != -100).any() for index in range(len(views))):
        raise ValueError("matrix target was fully truncated")
    return {
        "nl_input_ids": nl["input_ids"].to(device),
        "nl_attention_mask": nl["attention_mask"].to(device),
        "asl_input_ids": external["input_ids"].to(device),
        "asl_attention_mask": external["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def _mean_loss(
    model: ASLMatrixModel,
    tokenizer: Any,
    views: list[dict[str, Any]],
    training: MatrixTrainingConfig,
    device: str,
) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(views), training.batch_size):
            output = model(
                **_batch(tokenizer, views[start : start + training.batch_size], training, device)
            )
            loss = float(output.loss.detach().float().cpu())
            if not torch.isfinite(output.loss).all():
                raise FloatingPointError(f"non-finite autonomous development loss at row {start}")
            losses.append(loss)
    model.train()
    return sum(losses) / len(losses)


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _model_state_cpu(model: ASLMatrixModel) -> dict[str, Any]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def train_matrix(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Train one matrix cell with resumable epoch checkpoints."""

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("matrix training requires torch and transformers") from error

    config_path = Path(config_path)
    config = read_json(config_path)
    training = MatrixTrainingConfig.from_dict(config)
    if seed_override is not None:
        training = MatrixTrainingConfig(**{**training.__dict__, "seed": seed_override})
    training.validate()
    model_spec = config["model"]
    revision = str(model_spec["revision"])
    if len(revision) != 40:
        raise ValueError("matrix model revision must be a pinned 40-character SHA")
    output = Path(output_dir)
    report_path = output / "training_report.json"
    if report_path.exists():
        raise FileExistsError(f"completed matrix run already exists: {report_path}")
    device = select_device(torch, training.device)
    if training.device == "xpu" and device != "xpu":
        raise RuntimeError("the requested XPU matrix training device is unavailable")
    dtype = getattr(torch, training.dtype)
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    if device == "xpu":
        torch.xpu.manual_seed_all(training.seed)
        torch.xpu.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_spec["model_id"], revision=revision)
    train_examples = _examples(Path(data_dir) / "source" / "train.jsonl")
    dev_examples = _examples(Path(data_dir) / "source" / "dev.jsonl")
    length_audit = {
        "train": _token_length_audit(tokenizer, train_examples, training, split="train"),
        "dev": _token_length_audit(tokenizer, dev_examples, training, split="dev"),
    }
    model = ASLMatrixModel.from_pretrained(
        model_spec["model_id"],
        revision=revision,
        encoder_architecture=config["encoder"]["architecture"],
        attention_mode=config["attention"]["mode"],
        hybrid_shared_top_layers=int(
            config["encoder"].get("hybrid", {}).get("shared_top_layers", 2)
        ),
    ).to(device=device, dtype=dtype)
    model.config.use_cache = False
    builder = _regime_builder(config, training.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    last_path = output / "checkpoint_last.pt"
    best_path = output / "checkpoint_best.pt"
    start_epoch = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    best_dev_loss = float("inf")
    best_epoch = 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        history = list(checkpoint["history"])
        best_dev_loss = float(checkpoint["best_dev_loss"])
        best_epoch = int(checkpoint["best_epoch"])

    started = time.perf_counter()
    model.train()
    for epoch in range(start_epoch, training.epochs):
        train_views = _views(train_examples, builder, epoch=epoch)
        order = list(range(len(train_views)))
        random.Random(training.seed + epoch).shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, start in enumerate(range(0, len(order), training.batch_size)):
            indexes = order[start : start + training.batch_size]
            batch_views = [train_views[index] for index in indexes]
            result = model(**_batch(tokenizer, batch_views, training, device))
            if not torch.isfinite(result.loss).all():
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch + 1}, batch {batch_index + 1}"
                )
            (result.loss / training.gradient_accumulation_steps).backward()
            losses.append(float(result.loss.detach().cpu()))
            final_batch = start + training.batch_size >= len(order)
            if (batch_index + 1) % training.gradient_accumulation_steps == 0 or final_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        dev_views = _views(dev_examples, builder, epoch=epoch, forced_regime="autonomous")
        dev_loss = _mean_loss(model, tokenizer, dev_views, training, device)
        row = {
            "epoch": epoch + 1,
            "mean_train_loss": sum(losses) / len(losses),
            "autonomous_dev_loss": dev_loss,
            "regime_counts": dict(
                sorted(
                    {
                        name: sum(v["regime"] == name for v in train_views)
                        for name in ("full", "partial", "autonomous")
                    }.items()
                )
            ),
        }
        history.append(row)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
            _atomic_torch_save(
                torch,
                {
                    "model_state": _model_state_cpu(model),
                    "epoch": best_epoch,
                    "dev_loss": best_dev_loss,
                },
                best_path,
            )
        _atomic_torch_save(
            torch,
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "optimizer_steps": optimizer_steps,
                "history": history,
                "best_dev_loss": best_dev_loss,
                "best_epoch": best_epoch,
            },
            last_path,
        )
        print(
            f"epoch {epoch + 1}/{training.epochs}: train={row['mean_train_loss']:.4f} "
            f"autonomous_dev={dev_loss:.4f} best={best_dev_loss:.4f}@{best_epoch}"
        )
        if epoch + 1 - best_epoch >= training.early_stopping_patience:
            break

    if device == "xpu":
        torch.xpu.synchronize()
    wall_time_seconds = time.perf_counter() - started
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    parameter_report = model.parameter_report()
    report = {
        "schema_version": "ccpu.paper1.asl_matrix.training.v1",
        "run_id": config["run_id"],
        "seed": training.seed,
        "model": model_spec,
        "encoder": config["encoder"],
        "attention": config["attention"],
        "training": config["training"],
        "device": device,
        "dtype": training.dtype,
        "train_rows": len(train_examples),
        "dev_rows": len(dev_examples),
        "history": history,
        "best_epoch": best["epoch"],
        "best_autonomous_dev_loss": best["dev_loss"],
        "optimizer_steps": optimizer_steps,
        "wall_time_seconds": wall_time_seconds,
        "parameter_report": parameter_report,
        "peak_memory_bytes": int(torch.xpu.max_memory_allocated()) if device == "xpu" else None,
        "config_sha256": file_sha256(config_path),
        "data_sha256": {
            "train": file_sha256(Path(data_dir) / "source" / "train.jsonl"),
            "dev": file_sha256(Path(data_dir) / "source" / "dev.jsonl"),
        },
        "token_length_audit": length_audit,
        "checkpoint_best_sha256": file_sha256(best_path),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "resumed_from_epoch": start_epoch,
    }
    write_json(report_path, report)
    return report
