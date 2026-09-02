"""Behavioral and representation evaluation for ASL matrix cells."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from ccpu.paper1.asl_pilot_eval import extract_asl, score_asl

from .data import MatrixExample, RegimeBuilder, StaticMixture
from .model import ASLMatrixModel, representation_alignment
from .train import MatrixTrainingConfig, _batch

EVALUATION_CONDITIONS = (
    "autonomous",
    "full_teacher",
    "partial_20",
    "partial_50",
    "partial_80",
    "teacher_only",
    "wrong_teacher",
)


def _example(row: dict[str, Any]) -> MatrixExample:
    return MatrixExample(**row)


def _condition_view(
    example: MatrixExample,
    *,
    condition: str,
    builder: RegimeBuilder,
    wrong_asl: str,
) -> dict[str, Any]:
    if condition == "autonomous":
        return builder.make_view(example, regime="autonomous")
    if condition in {"full_teacher", "teacher_only"}:
        return builder.make_view(example, regime="full")
    if condition == "wrong_teacher":
        view = builder.make_view(example, regime="full")
        view["external_asl_input"] = wrong_asl
        view["external_asl_fraction"] = 1.0
        view["external_asl_corruption"] = {
            "policy": "unrelated_teacher",
            "severity": 1.0,
            "source_fields_visible_to_model": ["external_asl_input"],
        }
        return view
    severity = int(condition.rsplit("_", 1)[1]) / 100
    return builder.make_view(
        example,
        regime="partial",
        corruption_policy="record_dropout",
        corruption_severity=severity,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _linear_cka(first: Any, second: Any) -> float:
    import torch

    first = first.float() - first.float().mean(dim=0, keepdim=True)
    second = second.float() - second.float().mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(first.T @ second) ** 2
    first_norm = torch.linalg.matrix_norm(first.T @ first)
    second_norm = torch.linalg.matrix_norm(second.T @ second)
    denominator = (first_norm * second_norm).clamp_min(1e-12)
    return float((cross / denominator).detach().cpu())


def _pool(states: Any, mask: Any) -> Any:
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return (states * mask.unsqueeze(-1)).sum(dim=1) / denominator


def _representation_diagnostics(
    model: ASLMatrixModel,
    tokenizer: Any,
    examples: list[MatrixExample],
    builder: RegimeBuilder,
    training: MatrixTrainingConfig,
    device: str,
) -> dict[str, Any]:
    import torch

    views = [builder.make_view(example, regime="full") for example in examples]
    batch = _batch(tokenizer, views, training, device)
    with torch.no_grad():
        nl_memory, asl_memory = model.encode_sources(
            batch["nl_input_ids"],
            batch["nl_attention_mask"],
            batch["asl_input_ids"],
            batch["asl_attention_mask"],
            output_hidden_states=True,
        )
    final = representation_alignment(
        nl_memory.hidden_states,
        asl_memory.hidden_states,
        nl_memory.attention_mask,
        asl_memory.attention_mask,
    )
    layerwise = []
    common_layers = sorted(set(nl_memory.layer_states) & set(asl_memory.layer_states))
    for layer in common_layers:
        nl_pooled = _pool(nl_memory.layer_states[layer], nl_memory.attention_mask)
        asl_pooled = _pool(asl_memory.layer_states[layer], asl_memory.attention_mask)
        cosine = torch.nn.functional.cosine_similarity(
            nl_pooled.float(), asl_pooled.float(), dim=-1
        )
        layerwise.append(
            {
                "layer": layer,
                "paired_cosine_mean": float(cosine.mean().detach().cpu()),
                "linear_cka": _linear_cka(nl_pooled, asl_pooled),
            }
        )
    return {"final": final, "layerwise": layerwise}


def run_matrix_evaluation(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
    conditions: tuple[str, ...] = EVALUATION_CONDITIONS,
) -> dict[str, Any]:
    """Evaluate one trained cell under every compatible teacher condition."""

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("matrix evaluation requires torch and transformers") from error
    unknown = set(conditions) - set(EVALUATION_CONDITIONS)
    if unknown:
        raise ValueError(f"unsupported matrix evaluation conditions: {sorted(unknown)}")
    config_path = Path(config_path)
    config = read_json(config_path)
    training = MatrixTrainingConfig.from_dict(config)
    if seed_override is not None:
        training = MatrixTrainingConfig(**{**training.__dict__, "seed": seed_override})
    from ccpu.paper1.generation import select_device

    device = select_device(torch, training.device)
    dtype = getattr(torch, training.dtype)
    model_spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"], revision=model_spec["revision"]
    )
    tokenizer.model_max_length = max(
        training.max_nl_length, training.max_asl_length, training.max_target_length
    )
    model = ASLMatrixModel.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        encoder_architecture=config["encoder"]["architecture"],
        attention_mode=config["attention"]["mode"],
        hybrid_shared_top_layers=int(
            config["encoder"].get("hybrid", {}).get("shared_top_layers", 2)
        ),
        adaptation=config.get("adaptation"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device=device, dtype=dtype).eval()
    examples = [_example(row) for row in read_jsonl(Path(data_dir) / "source" / "test.jsonl")]
    builder = RegimeBuilder(
        mixture=StaticMixture(),
        corruption_policies=tuple(config["training"]["corruption"]["policy"]),
        corruption_severity=float(config["training"]["corruption"]["severity"]),
        seed=training.seed,
    )
    output = Path(output_dir)
    predictions_path = output / "predictions.jsonl"
    predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed = {(row["condition"], row["example_id"]) for row in predictions}
    started = time.perf_counter()
    for condition in conditions:
        for index, example in enumerate(examples):
            key = (condition, example.example_id)
            if key in completed:
                continue
            wrong = examples[(index + 1) % len(examples)]
            view = _condition_view(
                example,
                condition=condition,
                builder=builder,
                wrong_asl=wrong.target_asl,
            )
            batch = _batch(tokenizer, [view], training, device)
            if condition == "teacher_only":
                batch["nl_attention_mask"].zero_()
                batch["nl_input_ids"].fill_(tokenizer.pad_token_id)
            with torch.no_grad():
                generated_ids = model.greedy_generate(
                    nl_input_ids=batch["nl_input_ids"],
                    nl_attention_mask=batch["nl_attention_mask"],
                    asl_input_ids=batch["asl_input_ids"],
                    asl_attention_mask=batch["asl_attention_mask"],
                    max_new_tokens=training.max_target_length,
                )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predicted_asl = extract_asl(generated_text)
            metrics = score_asl(example.target_asl, predicted_asl, example.effective_scope)
            followed_wrong = False
            if condition == "wrong_teacher" and predicted_asl:
                wrong_metrics = score_asl(wrong.target_asl, predicted_asl, example.effective_scope)
                followed_wrong = bool(wrong_metrics["semantic_return_equivalent"])
            diagnostic_view = dict(view)
            diagnostic_view["target_asl"] = predicted_asl or example.target_asl
            diagnostic_batch = _batch(tokenizer, [diagnostic_view], training, device)
            if condition == "teacher_only":
                diagnostic_batch["nl_attention_mask"].zero_()
            with torch.no_grad():
                diagnostic = model(**diagnostic_batch, output_attentions=True).diagnostics
            predictions.append(
                {
                    "schema_version": "ccpu.paper1.asl_matrix.prediction.v1",
                    "run_id": config["run_id"],
                    "seed": training.seed,
                    "condition": condition,
                    "example_id": example.example_id,
                    "dataset": example.dataset,
                    "parent_source_id": example.parent_source_id,
                    "semantic_pattern_id": example.semantic_pattern_id,
                    "generated_text": generated_text,
                    "predicted_asl": predicted_asl,
                    "metrics": metrics,
                    "followed_wrong_teacher": followed_wrong,
                    "attention_diagnostics": diagnostic,
                    "external_asl_corruption": view["external_asl_corruption"],
                    "source_fields_visible_to_model": view["source_fields_visible_to_model"],
                }
            )
            completed.add(key)
            write_jsonl(predictions_path, predictions)
            print(f"checkpoint matrix eval: {condition} {index + 1}/{len(examples)}")

    metric_names = (
        "exact_asl",
        "parse_valid",
        "lowerable_to_ccir",
        "type_valid",
        "executable",
        "dependency_correct",
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
    )
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        members = [row for row in predictions if row["condition"] == condition]
        by_condition[condition] = {
            "count": len(members),
            "rates": {
                name: _mean([float(row["metrics"].get(name, False)) for row in members])
                for name in metric_names
            },
            "followed_wrong_teacher_rate": _mean(
                [float(row["followed_wrong_teacher"]) for row in members]
            ),
        }
    autonomous = by_condition["autonomous"]["rates"]
    full = by_condition.get("full_teacher", {"rates": autonomous})["rates"]
    representation = _representation_diagnostics(
        model, tokenizer, examples, builder, training, device
    )
    summary = {
        "schema_version": "ccpu.paper1.asl_matrix.evaluation.v1",
        "run_id": config["run_id"],
        "seed": training.seed,
        "encoder_architecture": config["encoder"]["architecture"],
        "attention_mode": config["attention"]["mode"],
        "adaptation": config.get("adaptation", {"method": "full"}),
        "training_mixture": config["training"]["static"],
        "test_rows": len(examples),
        "by_condition": by_condition,
        "teacher_gap": {
            name: full[name] - autonomous[name]
            for name in ("semantic_return_equivalent", "final_answer_correct")
        },
        "representation_diagnostics": representation,
        "wall_time_seconds": time.perf_counter() - started,
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_sha256": file_sha256(Path(data_dir) / "source" / "test.jsonl"),
        "predictions_sha256": file_sha256(predictions_path),
    }
    write_json(output / "summary.json", summary)
    return summary


def analyze_matrix_runs(summary_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Aggregate seeds and compute matched grounding gains for ladder cells."""

    rows = []
    for path in summary_paths:
        summary = read_json(path)
        autonomous = summary["by_condition"]["autonomous"]["rates"]
        full = summary["by_condition"]["full_teacher"]["rates"]
        rows.append(
            {
                "run_id": summary["run_id"],
                "seed": summary["seed"],
                "encoder_architecture": summary["encoder_architecture"],
                "attention_mode": summary["attention_mode"],
                "training_mixture": summary["training_mixture"],
                "autonomous_exact": autonomous["exact_asl"],
                "autonomous_semantic": autonomous["semantic_return_equivalent"],
                "autonomous_answer": autonomous["final_answer_correct"],
                "autonomous_syntax": autonomous["parse_valid"],
                "full_teacher_semantic": full["semantic_return_equivalent"],
                "full_teacher_answer": full["final_answer_correct"],
                "teacher_gap_semantic": summary["teacher_gap"]["semantic_return_equivalent"],
                "teacher_gap_answer": summary["teacher_gap"]["final_answer_correct"],
            }
        )
    baseline_by_seed = {
        row["seed"]: row for row in rows if row["run_id"].casefold().startswith("b0")
    }
    for row in rows:
        baseline = baseline_by_seed.get(row["seed"])
        row["grounding_gain_semantic"] = (
            row["autonomous_semantic"] - baseline["autonomous_semantic"] if baseline else None
        )
        row["grounding_gain_answer"] = (
            row["autonomous_answer"] - baseline["autonomous_answer"] if baseline else None
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)
    aggregates = {}
    for run_id, members in grouped.items():
        aggregates[run_id] = {
            "seed_count": len(members),
            "autonomous_semantic_mean": _mean([row["autonomous_semantic"] for row in members]),
            "autonomous_answer_mean": _mean([row["autonomous_answer"] for row in members]),
            "grounding_gain_semantic_mean": _mean(
                [
                    row["grounding_gain_semantic"]
                    for row in members
                    if row["grounding_gain_semantic"] is not None
                ]
            ),
        }
    output = Path(output_dir)
    rows_path = write_jsonl(output / "matrix_rows.jsonl", rows)
    report = {
        "schema_version": "ccpu.paper1.asl_matrix.analysis.v1",
        "run_count": len(rows),
        "aggregates": aggregates,
        "rows_sha256": file_sha256(rows_path),
        "single_seed_results_are_exploratory": any(
            value["seed_count"] < 3 for value in aggregates.values()
        ),
    }
    write_json(output / "summary.json", report)
    return report
