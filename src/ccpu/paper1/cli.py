"""Command-line workflow for Paper 1 artifacts and model runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.gsm8k import materialize_gsm8k
from ccpu.dsl_dataset.local_codex import run_local_codex_batches

from .asl_incremental_analysis import (
    analyze_adapter_capacity_interventions,
    analyze_incremental_capacity,
)
from .asl_incremental_eval import run_asl_incremental
from .asl_pilot_analysis import build_asl_checkpoint_report
from .asl_pilot_data import (
    build_asl_expansion_data,
    build_asl_incremental_data,
    build_asl_pilot_data,
    freeze_asl_pilot,
)
from .asl_pilot_eval import analyze_asl_predictions, run_asl_pilot
from .dataset import (
    ArithmeticDatasetConfig,
    ArithmeticExample,
    HardArithmeticDatasetConfig,
    iter_dataset,
    iter_hard_dataset,
)
from .evaluate import evaluate, paired_comparisons, rescore_endpoint_predictions
from .experiment import run_huggingface, run_replay, run_scripted
from .f3.data import (
    build_f3_data,
    prepare_f3_annotation_batches,
    prepare_f3_retry_batches,
    validate_f3_annotations,
)
from .f3.eval import analyze_f3_predictions, run_f3_condition
from .functor_data import (
    build_functor_data,
    prepare_functor_annotation_batches,
    prepare_functor_retry_batches,
    validate_functor_annotations,
)
from .functor_eval import (
    analyze_functor_predictions,
    compare_functor_conditions,
    run_functor_condition,
)
from .functor_metrics import analyze_functor_metrics, compare_functor_model_sizes
from .generation import HuggingFaceBackend, HuggingFaceGenerationConfig
from .lora_data import LoRAProtocolDataConfig, generate_protocol_data
from .lora_train import LoRATrainingConfig, train_lora
from .model_analysis import build_model_comparison
from .placement_analysis import build_placement_comparison
from .plot import plot_interface_diagnostics, plot_scaling
from .prompts import (
    CONDITIONS,
    CORE_CONDITIONS,
    ICL_ORDER_CONTROL_PROMPT_VERSION,
    ICL_PROMPT_VERSION,
    MINIMAL_BLOCK_PROMPT_VERSION,
    PROMPT_VERSION,
)
from .public_gsm8k import (
    PUBLIC_GSM8K_CONDITIONS,
    analyze_gsm8k_runs,
    freeze_gsm8k_slice,
    run_gsm8k_example,
    write_gsm8k_run,
)
from .semantic_failure import analyze_saved_semantic_failures

PROTOCOL_VERSIONS = {
    "prompt": PROMPT_VERSION,
    "calculator_block_icl_prompt": ICL_PROMPT_VERSION,
    "calculator_block_icl_order_control_prompt": ICL_ORDER_CONTROL_PROMPT_VERSION,
    "calculator_block_minimal_prompt": MINIMAL_BLOCK_PROMPT_VERSION,
    "strict_detector": "strict_arithmetic_v1",
    "normalized_detector": "normalized_arithmetic_v1",
    "surface_normalizer": "surface_normalizer_v1",
    "calculator_block_grammar": "calculator_block_v1",
    "calculator_ir": "ccpu.arithmetic.postfix.v1",
    "hard_generator": "hard_arithmetic_v1",
}


def _examples(path: str | Path) -> list[ArithmeticExample]:
    return [ArithmeticExample.from_dict(row) for row in read_jsonl(path)]


def _smoke_examples(
    examples: list[ArithmeticExample], *, arithmetic_count: int = 4, control_count: int = 2
) -> list[ArithmeticExample]:
    arithmetic = [example for example in examples if example.task_kind == "arithmetic"]
    controls = [example for example in examples if example.task_kind == "control"]
    if len(arithmetic) < arithmetic_count or len(controls) < control_count:
        raise ValueError(
            f"smoke gate requires {arithmetic_count} arithmetic and {control_count} controls"
        )
    return [*arithmetic[:arithmetic_count], *controls[:control_count]]


def _write_run(
    output_dir: str | Path,
    *,
    dataset_path: str | Path,
    predictions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    empirical: bool,
    run_config: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    predictions_path = write_jsonl(output_dir / "predictions.jsonl", predictions)
    traces_path = write_jsonl(output_dir / "traces.jsonl", traces)
    summary = evaluate(_examples(dataset_path), predictions)
    summary["empirical"] = empirical
    if not empirical:
        summary["warning"] = "Scripted protocol smoke results are not model evidence."
    summary_path = write_json(output_dir / "summary.json", summary)
    block_failures_path = write_json(
        output_dir / "block_failures.json",
        {
            "schema_version": "ccpu.paper1.block_failures.v1",
            "by_run": [
                {
                    "model_id": row["model_id"],
                    "condition": row["condition"],
                    "seed": row["seed"],
                    "arithmetic_count": row["arithmetic_count"],
                    "failure_counts": row["block_failure_counts"],
                }
                for row in summary["by_run"]
                if row.get("block_failure_counts") is not None
            ],
        },
    )
    runtime_rows = []
    for model_id in sorted({str(row["model_id"]) for row in predictions}):
        members = [row for row in predictions if str(row["model_id"]) == model_id]
        metadata = [dict(row.get("backend_metadata", {})) for row in members]
        peaks = [
            int(item["peak_memory_bytes"]) for item in metadata if item.get("peak_memory_bytes")
        ]
        runtime_rows.append(
            {
                "model_id": model_id,
                "prediction_count": len(members),
                "devices": sorted({str(item.get("device", "unknown")) for item in metadata}),
                "dtypes": sorted({str(item.get("dtype", "unknown")) for item in metadata}),
                "revisions": sorted({str(item.get("revision", "unknown")) for item in metadata}),
                "chat_template_settings": sorted(
                    {
                        (
                            bool(item.get("use_chat_template", False)),
                            bool(item.get("used_chat_template", False)),
                            bool(item.get("enable_thinking", False)),
                        )
                        for item in metadata
                    }
                ),
                "wall_time_seconds": sum(int(row.get("wall_time_ns", 0)) for row in members) / 1e9,
                "model_memory_bytes": next(
                    (
                        item.get("model_memory_bytes")
                        for item in metadata
                        if item.get("model_memory_bytes")
                    ),
                    None,
                ),
                "peak_memory_bytes": max(peaks) if peaks else None,
            }
        )
    runtime_report_path = write_json(
        output_dir / "runtime_report.json",
        {
            "schema_version": "ccpu.paper1.runtime_report.v1",
            "empirical": empirical,
            "by_model": runtime_rows,
        },
    )
    repository_root = Path(__file__).resolve().parents[3]
    write_json(
        output_dir / "manifest.json",
        {
            "paper": "Paper 1",
            "schema_version": "ccpu.paper1.run_manifest.v1",
            "empirical": empirical,
            "dataset_sha256": file_sha256(dataset_path),
            "predictions_sha256": file_sha256(predictions_path),
            "traces_sha256": file_sha256(traces_path),
            "summary_sha256": file_sha256(summary_path),
            "block_failures_sha256": file_sha256(block_failures_path),
            "runtime_report_sha256": file_sha256(runtime_report_path),
            "prediction_count": len(predictions),
            "trace_count": len(traces),
            "run_config": run_config,
            "environment": environment_manifest(repository_root),
        },
    )


def generate_command(args: argparse.Namespace) -> int:
    raw_config = read_json(args.config)
    if raw_config.get("dataset", raw_config).get("mode") == "hard_v1":
        config = HardArithmeticDatasetConfig.from_dict(raw_config)
        examples = list(iter_hard_dataset(config))
    else:
        config = ArithmeticDatasetConfig.from_dict(raw_config)
        examples = list(iter_dataset(config))
    output = write_jsonl(args.output, (example.to_dict() for example in examples))
    write_json(
        Path(output).with_suffix(".manifest.json"),
        {
            "paper": "Paper 1",
            "schema_version": "ccpu.paper1.dataset_manifest.v1",
            "config": config.to_dict(),
            "config_fingerprint": fingerprint(config.to_dict()),
            "protocol_versions": PROTOCOL_VERSIONS,
            "record_count": len(examples),
            "dataset_sha256": file_sha256(output),
        },
    )
    print(f"generated {len(examples)} Paper 1 examples -> {output}")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    examples = _examples(args.dataset)
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset contains duplicate example IDs")
    if any(example.should_trigger != (example.task_kind == "arithmetic") for example in examples):
        raise ValueError("task_kind and should_trigger disagree")
    schema = examples[0].schema_version if examples else "empty"
    print(f"valid {len(examples)}-item {schema} dataset")
    return 0


def simulate_command(args: argparse.Namespace) -> int:
    examples = _examples(args.dataset)
    conditions = tuple(args.condition or CORE_CONDITIONS)
    predictions, traces = run_scripted(examples, conditions=conditions, seed=args.seed)
    _write_run(
        args.output_dir,
        dataset_path=args.dataset,
        predictions=predictions,
        traces=traces,
        empirical=False,
        run_config={
            "backend": "scripted_protocol_smoke",
            "protocol_versions": PROTOCOL_VERSIONS,
            "conditions": conditions,
            "seed": args.seed,
        },
    )
    print(f"completed non-empirical protocol smoke -> {args.output_dir}")
    return 0


def replay_command(args: argparse.Namespace) -> int:
    predictions, traces = run_replay(_examples(args.dataset), read_jsonl(args.completions))
    _write_run(
        args.output_dir,
        dataset_path=args.dataset,
        predictions=predictions,
        traces=traces,
        empirical=not args.non_empirical,
        run_config={"backend": "replay", "completions_sha256": file_sha256(args.completions)},
    )
    print(f"replayed {len(predictions)} completions -> {args.output_dir}")
    return 0


def hf_command(args: argparse.Namespace) -> int:
    raw_config = read_json(args.config)
    model_entries = list(raw_config.get("models", []))
    if args.model:
        model_entries = [entry for entry in model_entries if entry["model_id"] in set(args.model)]
    if not model_entries:
        raise ValueError("no matching pinned models in configuration")
    examples = _examples(args.dataset)
    if args.smoke and args.limit is not None:
        raise ValueError("--smoke and --limit are mutually exclusive")
    if args.smoke:
        examples = _smoke_examples(examples)
    elif args.limit is not None:
        examples = examples[: args.limit]
    conditions = tuple(args.condition or CORE_CONDITIONS)
    seeds = tuple(int(seed) for seed in raw_config.get("seeds", (17,)))
    all_predictions: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    for entry in model_entries:
        revision = str(entry["revision"])
        invalid_revision = len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        )
        if invalid_revision:
            raise ValueError(f"model revision must be a pinned 40-character SHA: {entry}")
        backend = HuggingFaceBackend(
            HuggingFaceGenerationConfig(
                model_id=str(entry["model_id"]),
                revision=revision,
                max_new_tokens=int(entry.get("max_new_tokens", 96)),
                device=str(args.device or entry.get("device", "auto")),
                dtype=str(entry.get("dtype", "auto")),
                trust_remote_code=bool(entry.get("trust_remote_code", False)),
                use_chat_template=bool(entry.get("use_chat_template", True)),
                enable_thinking=bool(entry.get("enable_thinking", False)),
                adapter_path=entry.get("adapter_path"),
                adapter_id=entry.get("adapter_id"),
            )
        )
        predictions, traces = run_huggingface(examples, backend, conditions=conditions, seeds=seeds)
        all_predictions.extend(predictions)
        all_traces.extend(traces)
    _write_run(
        args.output_dir,
        dataset_path=args.dataset,
        predictions=all_predictions,
        traces=all_traces,
        empirical=True,
        run_config={
            "backend": "huggingface",
            "protocol_versions": PROTOCOL_VERSIONS,
            "models": model_entries,
            "conditions": conditions,
            "seeds": seeds,
            "limit": args.limit,
            "smoke_gate": args.smoke,
        },
    )
    print(f"completed {len(all_predictions)} Hugging Face generations -> {args.output_dir}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    result = evaluate(_examples(args.dataset), read_jsonl(args.predictions))
    write_json(args.output, result)
    print(f"evaluated predictions -> {args.output}")
    return 0


def rescore_endpoints_command(args: argparse.Namespace) -> int:
    examples = _examples(args.dataset)
    source = read_jsonl(args.predictions)
    rescored = rescore_endpoint_predictions(source)
    output_dir = Path(args.output_dir)
    predictions_path = write_jsonl(output_dir / "predictions.jsonl", rescored)
    summary = evaluate(examples, rescored, answer_field="endpoint_predicted_answer")
    summary["schema_version"] = "ccpu.paper1.endpoint_rescore.v1"
    summary["preserves_reported_metrics"] = True
    summary["source_predictions_sha256"] = file_sha256(args.predictions)
    write_json(output_dir / "summary.json", summary)
    paired = paired_comparisons(
        examples,
        rescored,
        baseline=args.baseline,
        answer_field="endpoint_predicted_answer",
    )
    paired["schema_version"] = "ccpu.paper1.endpoint_rescore_paired.v1"
    write_json(output_dir / "paired_analysis.json", paired)
    write_json(
        output_dir / "manifest.json",
        {
            "paper": "Paper 1",
            "schema_version": "ccpu.paper1.endpoint_rescore_manifest.v1",
            "extractor_version": "paper1_condition_independent_endpoint_v2",
            "dataset_sha256": file_sha256(args.dataset),
            "source_predictions_sha256": file_sha256(args.predictions),
            "rescored_predictions_sha256": file_sha256(predictions_path),
            "prediction_count": len(rescored),
            "reported_labels_preserved": True,
        },
    )
    changed = sum(bool(row["endpoint_answer_changed"]) for row in rescored)
    print(f"rescored {len(rescored)} endpoints ({changed} changed labels) -> {output_dir}")
    return 0


def plot_command(args: argparse.Namespace) -> int:
    output = plot_scaling(read_json(args.summary), args.output)
    print(f"wrote scaling figure -> {output}")
    return 0


def paired_command(args: argparse.Namespace) -> int:
    prediction_paths = [Path(path) for path in args.predictions]
    predictions = [row for path in prediction_paths for row in read_jsonl(path)]
    result = paired_comparisons(_examples(args.dataset), predictions, baseline=args.baseline)
    result["dataset_sha256"] = file_sha256(args.dataset)
    result["prediction_sources"] = [
        {"path": str(path), "sha256": file_sha256(path)} for path in prediction_paths
    ]
    write_json(args.output, result)
    print(f"wrote paired comparisons -> {args.output}")
    return 0


def interface_plot_command(args: argparse.Namespace) -> int:
    output = plot_interface_diagnostics(read_json(args.summary), args.output)
    print(f"wrote interface figure -> {output}")
    return 0


def compare_models_command(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    result = build_model_comparison(config, config_path=args.config, output_dir=args.output_dir)
    print(f"wrote {len(result['rows'])} cross-model rows and figures -> {args.output_dir}")
    return 0


def generate_lora_data_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite protocol data: {output_dir}")
    config = LoRAProtocolDataConfig.from_dict(read_json(args.config))
    splits, audit = generate_protocol_data(config, excluded_dataset=args.excluded_dataset)
    train_path = write_jsonl(output_dir / "train.jsonl", splits["train"])
    dev_path = write_jsonl(output_dir / "dev.jsonl", splits["dev"])
    audit_path = write_json(output_dir / "leakage_audit.json", audit)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ccpu.paper1.lora_data_manifest.v1",
            "config": config.to_dict(),
            "config_sha256": file_sha256(args.config),
            "excluded_dataset_sha256": file_sha256(args.excluded_dataset),
            "train_sha256": file_sha256(train_path),
            "dev_sha256": file_sha256(dev_path),
            "leakage_audit_sha256": file_sha256(audit_path),
            "train_rows": len(splits["train"]),
            "dev_rows": len(splits["dev"]),
        },
    )
    print(f"generated {len(splits['train'])} train and {len(splits['dev'])} dev rows")
    return 0


def train_lora_command(args: argparse.Namespace) -> int:
    raw_config = read_json(args.config)
    entries = [entry for entry in raw_config.get("models", []) if entry["model_id"] == args.model]
    if len(entries) != 1:
        raise ValueError(f"expected one pinned training model for {args.model}")
    report = train_lora(
        model=entries[0],
        training=LoRATrainingConfig.from_dict(raw_config),
        train_path=args.train,
        dev_path=args.dev,
        output_dir=args.output_dir,
    )
    print(
        f"trained {report['adapter_id']} with {report['trainable_parameters']} parameters "
        f"in {report['wall_time_seconds']:.1f}s"
    )
    return 0


def freeze_asl_pilot_command(args: argparse.Namespace) -> int:
    manifest = freeze_asl_pilot(args.accepted, args.output_dir, seed=args.seed)
    print(
        f"froze grouped ASL split {manifest['counts']} with zero pattern overlap "
        f"-> {args.output_dir}"
    )
    return 0


def build_asl_pilot_data_command(args: argparse.Namespace) -> int:
    manifest = build_asl_pilot_data(
        args.freeze_dir,
        args.output_dir,
        augmentation_variants=args.augmentation_variants,
        seed=args.seed,
    )
    print(
        f"built ASL pilot data with {manifest['files']['train_augmented']['rows']} "
        f"augmented train rows -> {args.output_dir}"
    )
    return 0


def build_asl_expansion_data_command(args: argparse.Namespace) -> int:
    manifest = build_asl_expansion_data(
        args.freeze_dir,
        args.expansion_train,
        args.output_dir,
        seed=args.seed,
    )
    print(
        f"built {manifest['train_rows']} expansion training rows with "
        f"{manifest['train_pattern_count']} patterns -> {args.output_dir}"
    )
    return 0


def build_asl_incremental_data_command(args: argparse.Namespace) -> int:
    manifest = build_asl_incremental_data(
        args.freeze_dir,
        args.expansion_train,
        args.output_dir,
        seed=args.seed,
    )
    print(
        f"built {manifest['train_transitions']} incremental transitions from "
        f"{manifest['train_programs']} programs -> {args.output_dir}"
    )
    return 0


def prepare_functor_annotations_command(args: argparse.Namespace) -> int:
    manifest = prepare_functor_annotation_batches(
        args.freeze_dir,
        args.expansion_train,
        args.output_dir,
        batch_size=args.batch_size,
    )
    print(
        f"prepared {manifest['example_count']} answer-blind functor requests in "
        f"{manifest['batch_count']} batches -> {args.output_dir}"
    )
    return 0


def run_functor_annotations_command(args: argparse.Namespace) -> int:
    manifest = run_local_codex_batches(
        args.requests_dir,
        args.output_dir,
        prompt_path=args.prompt,
        schema_path=args.schema,
        repo_root=args.repo_root,
        executable=args.executable,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
    )
    print(
        f"completed {manifest['annotation_count']} functor annotations; "
        f"failed batches={manifest['failed_count']} -> {args.output_dir}"
    )
    return 0


def prepare_functor_retries_command(args: argparse.Namespace) -> int:
    manifest = prepare_functor_retry_batches(
        args.freeze_dir,
        args.expansion_train,
        args.rejected,
        args.output_dir,
        batch_size=args.batch_size,
        retry_round=args.retry_round,
    )
    print(
        f"prepared {manifest['example_count']} raw-only retry requests in "
        f"{manifest['batch_count']} batches -> {args.output_dir}"
    )
    return 0


def validate_functor_annotations_command(args: argparse.Namespace) -> int:
    report = validate_functor_annotations(
        args.freeze_dir,
        args.expansion_train,
        args.annotations,
        args.output_dir,
    )
    print(
        f"validated {report['paired_accepted_count']}/{report['source_count']} paired "
        f"functor labels -> {args.output_dir}"
    )
    return 0


def build_functor_data_command(args: argparse.Namespace) -> int:
    manifest = build_functor_data(
        args.freeze_dir,
        args.expansion_train,
        args.accepted,
        args.output_dir,
    )
    print(f"built matched functor data {manifest['split_counts']} -> {args.output_dir}")
    return 0


def prepare_f3_annotations_command(args: argparse.Namespace) -> int:
    manifest = prepare_f3_annotation_batches(
        args.freeze_dir,
        args.expansion_train,
        args.output_dir,
        batch_size=args.batch_size,
        max_train_examples=args.max_train_examples,
    )
    print(
        f"prepared {manifest['example_count']} answer-blind F3 requests in "
        f"{manifest['batch_count']} batches -> {args.output_dir}"
    )
    return 0


def run_f3_annotations_command(args: argparse.Namespace) -> int:
    manifest = run_local_codex_batches(
        args.requests_dir,
        args.output_dir,
        prompt_path=args.prompt,
        schema_path=args.schema,
        repo_root=args.repo_root,
        executable=args.executable,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
    )
    print(
        f"completed {manifest['annotation_count']} F3 annotations; "
        f"failed batches={manifest['failed_count']} -> {args.output_dir}"
    )
    return 0


def prepare_f3_retries_command(args: argparse.Namespace) -> int:
    manifest = prepare_f3_retry_batches(
        args.freeze_dir,
        args.expansion_train,
        args.rejected,
        args.output_dir,
        batch_size=args.batch_size,
    )
    print(
        f"prepared {manifest['example_count']} raw-only F3 retries in "
        f"{manifest['batch_count']} batches -> {args.output_dir}"
    )
    return 0


def validate_f3_annotations_command(args: argparse.Namespace) -> int:
    report = validate_f3_annotations(
        args.freeze_dir,
        args.expansion_train,
        args.annotations,
        args.output_dir,
    )
    print(
        f"validated {report['accepted_count']}/{report['source_count']} F3 labels "
        f"-> {args.output_dir}"
    )
    return 0


def build_f3_data_command(args: argparse.Namespace) -> int:
    manifest = build_f3_data(
        args.freeze_dir,
        args.expansion_train,
        args.accepted,
        args.output_dir,
    )
    print(f"built F3 data {manifest['retained_counts']} -> {args.output_dir}")
    return 0


def run_f3_command(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    config["adapter_path"] = args.adapter_path
    config["adapter_id"] = args.adapter_id or Path(args.adapter_path).name
    report = run_f3_condition(
        eval_path=args.eval,
        model_config=config,
        output_dir=args.output_dir,
        primary_mode=args.primary_mode,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        f"completed {report['prediction_count']} F3 generations; "
        f"answer={report['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
    )
    return 0


def evaluate_f3_command(args: argparse.Namespace) -> int:
    report = analyze_f3_predictions(
        args.eval,
        args.predictions,
        args.output_dir,
        primary_mode=args.primary_mode,
    )
    print(f"evaluated {report['prediction_count']} F3 predictions -> {args.output_dir}")
    return 0


def run_functor_command(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    config["adapter_path"] = args.adapter_path
    config["adapter_id"] = args.adapter_id or Path(args.adapter_path).name
    report = run_functor_condition(
        eval_path=args.eval,
        model_config=config,
        condition=args.condition,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        f"completed {report['prediction_count']} {args.condition.upper()} generations; "
        f"answer={report['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
    )
    return 0


def evaluate_functor_command(args: argparse.Namespace) -> int:
    report = analyze_functor_predictions(args.eval, args.predictions, args.output_dir)
    print(f"evaluated {report['prediction_count']} functor predictions -> {args.output_dir}")
    return 0


def compare_functors_command(args: argparse.Namespace) -> int:
    report = compare_functor_conditions(
        f0_summary=args.f0, f1_summary=args.f1, f2_summary=args.f2, output=args.output
    )
    print(
        "compared F0/F1/F2; semantic decomposition gate="
        f"{report['interpretation_gate']['semantic_decomposition_supported']} -> {args.output}"
    )
    return 0


def analyze_functor_metrics_command(args: argparse.Namespace) -> int:
    report = analyze_functor_metrics(
        f0_eval_path=args.f0_eval,
        f0_scored_path=args.f0_scored,
        f1_eval_path=args.f1_eval,
        f1_scored_path=args.f1_scored,
        f2_eval_path=args.f2_eval,
        f2_scored_path=args.f2_scored,
        f1_train_path=args.f1_train,
        f1_dev_path=args.f1_dev,
        f2_train_path=args.f2_train,
        f2_dev_path=args.f2_dev,
        output_dir=args.output_dir,
        model_label=args.model_label,
    )
    print(
        f"analyzed {report['frozen_identity_count']} matched F0/F1/F2 programs "
        f"for {report['model_label']} -> {args.output_dir}"
    )
    return 0


def compare_functor_model_sizes_command(args: argparse.Namespace) -> int:
    report = compare_functor_model_sizes(args.small, args.large, args.output)
    gate = report["representation_by_capacity_interaction"]["capacity_bottleneck_supported"]
    print(f"compared functor model sizes; capacity gate={gate} -> {args.output}")
    return 0


def run_asl_pilot_command(args: argparse.Namespace) -> int:
    model_config = read_json(args.config)
    if args.adapter_path:
        model_config["adapter_path"] = args.adapter_path
        model_config["adapter_id"] = args.adapter_id or Path(args.adapter_path).name
    report = run_asl_pilot(
        eval_path=args.eval,
        train_split_path=args.train_split,
        model_config=model_config,
        condition=args.condition,
        shots=args.shots,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        f"completed {report['prediction_count']} ASL generations; "
        f"execution={report['rates']['executable']:.3f} -> {args.output_dir}"
    )
    return 0


def run_asl_incremental_command(args: argparse.Namespace) -> int:
    model_config = read_json(args.config)
    model_config["adapter_path"] = args.adapter_path
    model_config["adapter_id"] = args.adapter_id or Path(args.adapter_path).name
    report = run_asl_incremental(
        programs_path=args.programs,
        model_config=model_config,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        state_mode=args.state_mode,
        context_mode=args.context_mode,
    )
    print(
        f"completed {report['prediction_count']} closed-loop ASL programs; "
        f"answer={report['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
    )
    return 0


def analyze_asl_incremental_command(args: argparse.Namespace) -> int:
    report = analyze_incremental_capacity(
        programs_path=args.programs,
        whole_scored_path=args.whole_scored,
        predicted_scored_path=args.predicted_scored,
        oracle_scored_path=args.oracle_scored,
        full_context_scored_path=args.full_context_scored,
        output_path=args.output,
    )
    print(
        f"analyzed incremental capacity; propagation gap="
        f"{report['state_error_propagation_gap']:.3f} -> {args.output}"
    )
    return 0


def compare_asl_adapter_capacity_command(args: argparse.Namespace) -> int:
    report = analyze_adapter_capacity_interventions(
        baseline_predicted_summary=args.baseline_predicted,
        baseline_oracle_summary=args.baseline_oracle,
        baseline_full_summary=args.baseline_full,
        candidate_predicted_summary=args.candidate_predicted,
        candidate_oracle_summary=args.candidate_oracle,
        candidate_full_summary=args.candidate_full,
        baseline_training_report=args.baseline_training,
        candidate_training_report=args.candidate_training,
        pilot_checkpoint=args.pilot_checkpoint,
        semantic_summary=args.semantic_summary,
        output_path=args.output,
    )
    delta = report["rank_deltas_r16_minus_r8"]["predicted"]["answer"]
    print(f"compared matched adapter capacity; answer delta={delta:.3f} -> {args.output}")
    return 0


def analyze_asl_semantic_failures_command(args: argparse.Namespace) -> int:
    report = analyze_saved_semantic_failures(
        eval_path=args.eval,
        programs_path=args.programs,
        whole_scored_path=args.whole_scored,
        predicted_scored_path=args.predicted_scored,
        oracle_scored_path=args.oracle_scored,
        output_dir=args.output_dir,
        teacher_paths=args.teacher or (),
    )
    count = report["conditions"]["whole_lora500"]["count"]
    print(
        f"analyzed fine-grained semantic failures for {count} frozen programs -> {args.output_dir}"
    )
    return 0


def evaluate_asl_pilot_command(args: argparse.Namespace) -> int:
    report = analyze_asl_predictions(args.eval, args.predictions, args.output_dir)
    write_json(Path(args.output_dir) / "summary.json", report)
    print(f"evaluated {report['prediction_count']} ASL predictions -> {args.output_dir}")
    return 0


def analyze_asl_checkpoint_command(args: argparse.Namespace) -> int:
    report = build_asl_checkpoint_report(
        freeze_dir=args.freeze_dir,
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        output=args.output,
    )
    print(
        f"analyzed {len(report['conditions'])} ASL conditions; "
        f"patterns={report['training_structure_coverage']['unique_normalized_semantic_patterns']} "
        f"-> {args.output}"
    )
    return 0


def analyze_placement_command(args: argparse.Namespace) -> int:
    result = build_placement_comparison(
        read_json(args.config), config_path=args.config, output_dir=args.output_dir
    )
    print(f"wrote {len(result['rows'])} interface-placement rows -> {args.output_dir}")
    return 0


def run_public_gsm8k_command(args: argparse.Namespace) -> int:
    config = read_json(args.model_config)
    if config.get("schema_version") != "ccpu.paper1.public_gsm8k_config.v1":
        raise ValueError("unsupported Paper 1 public GSM8K config schema")
    model_key = "lora_model" if args.condition == "lora_calculator_block" else "base_model"
    model = dict(config[model_key])
    revision = str(model["revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("public GSM8K model revision must be a pinned SHA")
    backend = HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=revision,
            max_new_tokens=int(model.get("max_new_tokens", 160)),
            device=str(args.device or model.get("device", "auto")),
            dtype=str(model.get("dtype", "auto")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            adapter_path=model.get("adapter_path"),
            adapter_id=model.get("adapter_id"),
            cached_generation=args.condition
            in {"llm_only", "matched_icl", "generic_compute", "oracle_calculator"},
        )
    )
    examples = materialize_gsm8k(args.public_config, args.cache_root, args.selection)
    if args.per_stratum is not None:
        selected = []
        for stratum in ("2_steps", "3_4_steps", "5plus_steps"):
            members = sorted(
                [row for row in examples if row["difficulty_stratum"] == stratum],
                key=lambda row: row["selection_key"],
            )
            if len(members) < args.per_stratum:
                raise ValueError(f"not enough GSM8K rows in {stratum}")
            selected.extend(members[: args.per_stratum])
        examples = sorted(selected, key=lambda row: row["selection_key"])
    examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    output = Path(args.output_dir)
    prediction_path = output / "predictions.jsonl"
    rows = read_jsonl(prediction_path) if prediction_path.exists() and not args.no_resume else []
    if any(row["condition"] != args.condition for row in rows):
        raise ValueError("resume output contains a different GSM8K condition")
    completed = {str(row["example_id"]) for row in rows}
    seed = int(config["seed"])
    checkpoint_every = max(1, int(args.checkpoint_every))
    pending = [example for example in examples if example["example_id"] not in completed]
    for index, example in enumerate(pending, 1):
        rows.append(
            run_gsm8k_example(
                example,
                backend,
                condition=args.condition,
                seed=seed,
                max_assistance_episodes=int(config.get("max_assistance_episodes", 4)),
            )
        )
        if index % checkpoint_every == 0:
            write_gsm8k_run(
                output,
                rows,
                config_path=args.model_config,
                selection_path=args.selection,
                condition=args.condition,
            )
            print(f"checkpoint {args.condition}: {len(rows)}/{len(examples)}")
    summary = write_gsm8k_run(
        output,
        rows,
        config_path=args.model_config,
        selection_path=args.selection,
        condition=args.condition,
    )
    print(
        f"completed {args.condition} on {summary['base_question_count']} GSM8K questions "
        f"-> {output}"
    )
    return 0


def freeze_public_gsm8k_command(args: argparse.Namespace) -> int:
    manifest = freeze_gsm8k_slice(
        args.source_selection, args.output_dir, per_stratum=args.per_stratum
    )
    print(f"froze {manifest['record_count']} balanced GSM8K questions -> {args.output_dir}")
    return 0


def analyze_public_gsm8k_command(args: argparse.Namespace) -> int:
    summary = analyze_gsm8k_runs(args.predictions, args.output_dir)
    print(
        f"analyzed {summary['base_question_count']} matched GSM8K questions; "
        f"automatic_rescue={summary['automatic_rescue']['rate']}"
    )
    return 0


def add_commands(papers: argparse._SubParsersAction) -> None:
    paper = papers.add_parser("paper1", help="strict reflex-calculator experiments")
    commands = paper.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate the deterministic benchmark")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.set_defaults(handler=generate_command)

    validate = commands.add_parser("validate", help="validate a benchmark JSONL")
    validate.add_argument("--dataset", required=True)
    validate.set_defaults(handler=validate_command)

    simulate = commands.add_parser("simulate", help="run a non-empirical protocol smoke test")
    simulate.add_argument("--dataset", required=True)
    simulate.add_argument("--output-dir", required=True)
    simulate.add_argument("--condition", action="append", choices=CONDITIONS)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.set_defaults(handler=simulate_command)

    replay = commands.add_parser("replay", help="apply controllers to saved model completions")
    replay.add_argument("--dataset", required=True)
    replay.add_argument("--completions", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--non-empirical", action="store_true")
    replay.set_defaults(handler=replay_command)

    hf = commands.add_parser("run-hf", help="run pinned Hugging Face checkpoints")
    hf.add_argument("--dataset", required=True)
    hf.add_argument("--config", required=True)
    hf.add_argument("--output-dir", required=True)
    hf.add_argument("--model", action="append")
    hf.add_argument("--condition", action="append", choices=CONDITIONS)
    hf.add_argument("--device")
    hf.add_argument("--limit", type=int)
    hf.add_argument(
        "--smoke",
        action="store_true",
        help="run the XPU smoke gate on four arithmetic examples and two controls",
    )
    hf.set_defaults(handler=hf_command)

    evaluation = commands.add_parser("evaluate", help="recompute summary metrics")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(handler=evaluate_command)

    rescore = commands.add_parser(
        "rescore-endpoints",
        help="write condition-independent endpoint labels without replacing reported labels",
    )
    rescore.add_argument("--dataset", required=True)
    rescore.add_argument("--predictions", required=True)
    rescore.add_argument("--output-dir", required=True)
    rescore.add_argument("--baseline", default="llm_only")
    rescore.set_defaults(handler=rescore_endpoints_command)

    plot = commands.add_parser("plot", help="plot accuracy scaling from a summary")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output", required=True)
    plot.set_defaults(handler=plot_command)

    paired = commands.add_parser("paired", help="compute paired exact condition comparisons")
    paired.add_argument("--dataset", required=True)
    paired.add_argument("--predictions", required=True, action="append")
    paired.add_argument("--baseline", default="llm_only")
    paired.add_argument("--output", required=True)
    paired.set_defaults(handler=paired_command)

    interface_plot = commands.add_parser(
        "plot-interfaces", help="plot held-out accuracy and interface diagnostics"
    )
    interface_plot.add_argument("--summary", required=True)
    interface_plot.add_argument("--output", required=True)
    interface_plot.set_defaults(handler=interface_plot_command)

    compare_models = commands.add_parser(
        "compare-models", help="merge model summaries and plot interface capability"
    )
    compare_models.add_argument("--config", required=True)
    compare_models.add_argument("--output-dir", required=True)
    compare_models.set_defaults(handler=compare_models_command)

    lora_data = commands.add_parser(
        "generate-lora-data", help="generate leakage-audited protocol-only SFT data"
    )
    lora_data.add_argument("--config", required=True)
    lora_data.add_argument("--excluded-dataset", required=True)
    lora_data.add_argument("--output-dir", required=True)
    lora_data.set_defaults(handler=generate_lora_data_command)

    lora_train = commands.add_parser("train-lora", help="train one pinned protocol adapter")
    lora_train.add_argument("--config", required=True)
    lora_train.add_argument("--model", required=True)
    lora_train.add_argument("--train", required=True)
    lora_train.add_argument("--dev", required=True)
    lora_train.add_argument("--output-dir", required=True)
    lora_train.set_defaults(handler=train_lora_command)

    asl_freeze = commands.add_parser(
        "freeze-asl-pilot", help="freeze grouped 100/25/25 semantic ASL splits"
    )
    asl_freeze.add_argument("--accepted", required=True)
    asl_freeze.add_argument("--output-dir", required=True)
    asl_freeze.add_argument("--seed", type=int, default=731993)
    asl_freeze.set_defaults(handler=freeze_asl_pilot_command)

    asl_data = commands.add_parser(
        "build-asl-pilot-data", help="build original, augmented, and perturbation ASL data"
    )
    asl_data.add_argument("--freeze-dir", required=True)
    asl_data.add_argument("--output-dir", required=True)
    asl_data.add_argument("--augmentation-variants", type=int, default=9)
    asl_data.add_argument("--seed", type=int, default=912733)
    asl_data.set_defaults(handler=build_asl_pilot_data_command)

    asl_expansion_data = commands.add_parser(
        "build-asl-expansion-data",
        help="build the leakage-audited 500-original checkpoint SFT data",
    )
    asl_expansion_data.add_argument("--freeze-dir", required=True)
    asl_expansion_data.add_argument("--expansion-train", required=True)
    asl_expansion_data.add_argument("--output-dir", required=True)
    asl_expansion_data.add_argument("--seed", type=int, default=912734)
    asl_expansion_data.set_defaults(handler=build_asl_expansion_data_command)

    asl_incremental_data = commands.add_parser(
        "build-asl-incremental-data",
        help="derive causal clause-local NL+executed-state ASL transitions",
    )
    asl_incremental_data.add_argument("--freeze-dir", required=True)
    asl_incremental_data.add_argument("--expansion-train", required=True)
    asl_incremental_data.add_argument("--output-dir", required=True)
    asl_incremental_data.add_argument("--seed", type=int, default=912735)
    asl_incremental_data.set_defaults(handler=build_asl_incremental_data_command)

    functor_prepare = commands.add_parser(
        "prepare-functor-annotations",
        help="prepare raw-NL, answer-blind F1/F2 teacher batches",
    )
    functor_prepare.add_argument("--freeze-dir", required=True)
    functor_prepare.add_argument("--expansion-train", required=True)
    functor_prepare.add_argument("--output-dir", required=True)
    functor_prepare.add_argument("--batch-size", type=int, default=5)
    functor_prepare.set_defaults(handler=prepare_functor_annotations_command)

    functor_annotate = commands.add_parser(
        "run-functor-annotations", help="run resumable local Codex F1/F2 annotation batches"
    )
    functor_annotate.add_argument("--requests-dir", required=True)
    functor_annotate.add_argument("--output-dir", required=True)
    functor_annotate.add_argument(
        "--prompt", default="configs/paper1/local_codex_functor_annotation_prompt.md"
    )
    functor_annotate.add_argument(
        "--schema", default="configs/paper1/functor_annotation_batch.schema.json"
    )
    functor_annotate.add_argument("--repo-root", default=".")
    functor_annotate.add_argument("--executable", default="codex")
    functor_annotate.add_argument("--model", default="gpt-5.4")
    functor_annotate.add_argument("--reasoning-effort", default="medium")
    functor_annotate.add_argument("--concurrency", type=int, default=4)
    functor_annotate.set_defaults(handler=run_functor_annotations_command)

    functor_retry = commands.add_parser(
        "prepare-functor-retries",
        help="retry rejected F1/F2 rows from the same raw-only fixed prompt",
    )
    functor_retry.add_argument("--freeze-dir", required=True)
    functor_retry.add_argument("--expansion-train", required=True)
    functor_retry.add_argument("--rejected", required=True)
    functor_retry.add_argument("--output-dir", required=True)
    functor_retry.add_argument("--batch-size", type=int, default=5)
    functor_retry.add_argument("--retry-round", type=int, default=1)
    functor_retry.set_defaults(handler=prepare_functor_retries_command)

    functor_validate = commands.add_parser(
        "validate-functor-annotations",
        help="parse, lower, execute, and answer-check primary functor labels",
    )
    functor_validate.add_argument("--freeze-dir", required=True)
    functor_validate.add_argument("--expansion-train", required=True)
    functor_validate.add_argument("--annotations", action="append", required=True)
    functor_validate.add_argument("--output-dir", required=True)
    functor_validate.set_defaults(handler=validate_functor_annotations_command)

    functor_data = commands.add_parser(
        "build-functor-data", help="build exact matched 450/25/25 F1/F2 SFT and evaluation data"
    )
    functor_data.add_argument("--freeze-dir", required=True)
    functor_data.add_argument("--expansion-train", required=True)
    functor_data.add_argument("--accepted", required=True)
    functor_data.add_argument("--output-dir", required=True)
    functor_data.set_defaults(handler=build_functor_data_command)

    f3_prepare = commands.add_parser(
        "prepare-f3-annotations", help="prepare raw-source-only F3 teacher batches"
    )
    f3_prepare.add_argument("--freeze-dir", required=True)
    f3_prepare.add_argument("--expansion-train", required=True)
    f3_prepare.add_argument("--output-dir", required=True)
    f3_prepare.add_argument("--batch-size", type=int, default=5)
    f3_prepare.add_argument(
        "--max-train-examples",
        type=int,
        help="prepare a deterministic train-only pilot instead of all frozen identities",
    )
    f3_prepare.set_defaults(handler=prepare_f3_annotations_command)

    f3_annotate = commands.add_parser(
        "run-f3-annotations", help="run resumable local Codex F3 annotation batches"
    )
    f3_annotate.add_argument("--requests-dir", required=True)
    f3_annotate.add_argument("--output-dir", required=True)
    f3_annotate.add_argument(
        "--prompt", default="configs/paper1/local_codex_f3_annotation_prompt.md"
    )
    f3_annotate.add_argument(
        "--schema", default="configs/paper1/f3_annotation_batch.schema.json"
    )
    f3_annotate.add_argument("--repo-root", default=".")
    f3_annotate.add_argument("--executable", default="codex")
    f3_annotate.add_argument("--model", default="gpt-5.4")
    f3_annotate.add_argument("--reasoning-effort", default="medium")
    f3_annotate.add_argument("--concurrency", type=int, default=4)
    f3_annotate.set_defaults(handler=run_f3_annotations_command)

    f3_retry = commands.add_parser(
        "prepare-f3-retries", help="retry rejected F3 rows from raw source only"
    )
    f3_retry.add_argument("--freeze-dir", required=True)
    f3_retry.add_argument("--expansion-train", required=True)
    f3_retry.add_argument("--rejected", required=True)
    f3_retry.add_argument("--output-dir", required=True)
    f3_retry.add_argument("--batch-size", type=int, default=5)
    f3_retry.set_defaults(handler=prepare_f3_retries_command)

    f3_validate = commands.add_parser(
        "validate-f3-annotations", help="ground, lower, execute, and answer-check F3 labels"
    )
    f3_validate.add_argument("--freeze-dir", required=True)
    f3_validate.add_argument("--expansion-train", required=True)
    f3_validate.add_argument("--annotations", action="append", required=True)
    f3_validate.add_argument("--output-dir", required=True)
    f3_validate.set_defaults(handler=validate_f3_annotations_command)

    f3_data = commands.add_parser(
        "build-f3-data", help="build exact matched F3 SFT and evaluation data"
    )
    f3_data.add_argument("--freeze-dir", required=True)
    f3_data.add_argument("--expansion-train", required=True)
    f3_data.add_argument("--accepted", required=True)
    f3_data.add_argument("--output-dir", required=True)
    f3_data.set_defaults(handler=build_f3_data_command)

    f3_run = commands.add_parser("run-f3", help="run one matched Qwen F3 LoRA condition")
    f3_run.add_argument("--eval", required=True)
    f3_run.add_argument("--config", required=True)
    f3_run.add_argument("--adapter-path", required=True)
    f3_run.add_argument("--adapter-id")
    f3_run.add_argument("--primary-mode", choices=("r0", "r1", "r2"), default="r1")
    f3_run.add_argument("--output-dir", required=True)
    f3_run.add_argument("--seed", type=int, default=44017)
    f3_run.add_argument("--checkpoint-every", type=int, default=5)
    f3_run.set_defaults(handler=run_f3_command)

    f3_evaluate = commands.add_parser(
        "evaluate-f3", help="score saved F3 predictions through all runtime modes"
    )
    f3_evaluate.add_argument("--eval", required=True)
    f3_evaluate.add_argument("--predictions", required=True)
    f3_evaluate.add_argument("--primary-mode", choices=("r0", "r1", "r2"), default="r1")
    f3_evaluate.add_argument("--output-dir", required=True)
    f3_evaluate.set_defaults(handler=evaluate_f3_command)

    functor_run = commands.add_parser(
        "run-functor", help="run one matched F1 or F2 Qwen LoRA condition"
    )
    functor_run.add_argument("--eval", required=True)
    functor_run.add_argument("--config", required=True)
    functor_run.add_argument("--adapter-path", required=True)
    functor_run.add_argument("--adapter-id")
    functor_run.add_argument("--condition", choices=("f1", "f2"), required=True)
    functor_run.add_argument("--output-dir", required=True)
    functor_run.add_argument("--seed", type=int, default=44017)
    functor_run.add_argument("--checkpoint-every", type=int, default=5)
    functor_run.set_defaults(handler=run_functor_command)

    functor_evaluate = commands.add_parser(
        "evaluate-functor", help="score saved F1/F2 predictions through runtime lowering"
    )
    functor_evaluate.add_argument("--eval", required=True)
    functor_evaluate.add_argument("--predictions", required=True)
    functor_evaluate.add_argument("--output-dir", required=True)
    functor_evaluate.set_defaults(handler=evaluate_functor_command)

    functor_compare = commands.add_parser(
        "compare-functors", help="compare matched F0 ASL, F1 low-level, and F2 semantic runs"
    )
    functor_compare.add_argument("--f0", required=True)
    functor_compare.add_argument("--f1", required=True)
    functor_compare.add_argument("--f2", required=True)
    functor_compare.add_argument("--output", required=True)
    functor_compare.set_defaults(handler=compare_functors_command)

    functor_metrics = commands.add_parser(
        "analyze-functor-metrics",
        help="build deterministic fine-grained F0/F1/F2 representation diagnostics",
    )
    functor_metrics.add_argument("--f0-eval", required=True)
    functor_metrics.add_argument("--f0-scored", required=True)
    functor_metrics.add_argument("--f1-eval", required=True)
    functor_metrics.add_argument("--f1-scored", required=True)
    functor_metrics.add_argument("--f2-eval", required=True)
    functor_metrics.add_argument("--f2-scored", required=True)
    functor_metrics.add_argument("--f1-train", required=True)
    functor_metrics.add_argument("--f1-dev", required=True)
    functor_metrics.add_argument("--f2-train", required=True)
    functor_metrics.add_argument("--f2-dev", required=True)
    functor_metrics.add_argument("--model-label", required=True)
    functor_metrics.add_argument("--output-dir", required=True)
    functor_metrics.set_defaults(handler=analyze_functor_metrics_command)

    functor_size = commands.add_parser(
        "compare-functor-model-sizes",
        help="compare matched F0/F1/F2 diagnostics across model capacities",
    )
    functor_size.add_argument("--small", required=True)
    functor_size.add_argument("--large", required=True)
    functor_size.add_argument("--output", required=True)
    functor_size.set_defaults(handler=compare_functor_model_sizes_command)

    asl_run = commands.add_parser(
        "run-asl-pilot", help="run one base, ICL, or LoRA semantic ASL condition"
    )
    asl_run.add_argument("--eval", required=True)
    asl_run.add_argument("--train-split", required=True)
    asl_run.add_argument("--config", required=True)
    asl_run.add_argument("--adapter-path")
    asl_run.add_argument("--adapter-id")
    asl_run.add_argument("--condition", choices=("base", "icl", "lora", "lora_icl"), required=True)
    asl_run.add_argument("--shots", type=int, default=0)
    asl_run.add_argument("--output-dir", required=True)
    asl_run.add_argument("--seed", type=int, default=44017)
    asl_run.add_argument("--checkpoint-every", type=int, default=5)
    asl_run.set_defaults(handler=run_asl_pilot_command)

    asl_incremental_run = commands.add_parser(
        "run-asl-incremental",
        help="run closed-loop clause-local ASL generation with predicted state",
    )
    asl_incremental_run.add_argument("--programs", required=True)
    asl_incremental_run.add_argument("--config", required=True)
    asl_incremental_run.add_argument("--adapter-path", required=True)
    asl_incremental_run.add_argument("--adapter-id")
    asl_incremental_run.add_argument("--output-dir", required=True)
    asl_incremental_run.add_argument("--seed", type=int, default=44017)
    asl_incremental_run.add_argument("--checkpoint-every", type=int, default=5)
    asl_incremental_run.add_argument(
        "--state-mode", choices=("predicted", "oracle"), default="predicted"
    )
    asl_incremental_run.add_argument(
        "--context-mode", choices=("causal", "full_question"), default="causal"
    )
    asl_incremental_run.set_defaults(handler=run_asl_incremental_command)

    asl_incremental_analysis = commands.add_parser(
        "analyze-asl-incremental",
        help="analyze paired, propagation, transition, and dataset diagnostics",
    )
    asl_incremental_analysis.add_argument("--programs", required=True)
    asl_incremental_analysis.add_argument("--whole-scored", required=True)
    asl_incremental_analysis.add_argument("--predicted-scored", required=True)
    asl_incremental_analysis.add_argument("--oracle-scored", required=True)
    asl_incremental_analysis.add_argument("--full-context-scored")
    asl_incremental_analysis.add_argument("--output", required=True)
    asl_incremental_analysis.set_defaults(handler=analyze_asl_incremental_command)

    capacity_comparison = commands.add_parser(
        "compare-asl-adapter-capacity",
        help="compare matched ASL adapter ranks and rank non-data interventions",
    )
    capacity_comparison.add_argument("--baseline-predicted", required=True)
    capacity_comparison.add_argument("--baseline-oracle", required=True)
    capacity_comparison.add_argument("--baseline-full", required=True)
    capacity_comparison.add_argument("--candidate-predicted", required=True)
    capacity_comparison.add_argument("--candidate-oracle", required=True)
    capacity_comparison.add_argument("--candidate-full", required=True)
    capacity_comparison.add_argument("--baseline-training", required=True)
    capacity_comparison.add_argument("--candidate-training", required=True)
    capacity_comparison.add_argument("--pilot-checkpoint", required=True)
    capacity_comparison.add_argument("--semantic-summary", required=True)
    capacity_comparison.add_argument("--output", required=True)
    capacity_comparison.set_defaults(handler=compare_asl_adapter_capacity_command)

    semantic_failures = commands.add_parser(
        "analyze-asl-semantic-failures",
        help="decompose saved whole and incremental ASL semantic failures",
    )
    semantic_failures.add_argument("--eval", required=True)
    semantic_failures.add_argument("--programs", required=True)
    semantic_failures.add_argument("--whole-scored", required=True)
    semantic_failures.add_argument("--predicted-scored", required=True)
    semantic_failures.add_argument("--oracle-scored", required=True)
    semantic_failures.add_argument("--teacher", action="append")
    semantic_failures.add_argument("--output-dir", required=True)
    semantic_failures.set_defaults(handler=analyze_asl_semantic_failures_command)

    asl_evaluate = commands.add_parser(
        "evaluate-asl-pilot", help="score saved ASL predictions with semantic components"
    )
    asl_evaluate.add_argument("--eval", required=True)
    asl_evaluate.add_argument("--predictions", required=True)
    asl_evaluate.add_argument("--output-dir", required=True)
    asl_evaluate.set_defaults(handler=evaluate_asl_pilot_command)

    asl_checkpoint = commands.add_parser(
        "analyze-asl-checkpoint", help="aggregate and pair the frozen ASL pilot results"
    )
    asl_checkpoint.add_argument("--freeze-dir", required=True)
    asl_checkpoint.add_argument("--data-dir", required=True)
    asl_checkpoint.add_argument("--runs-dir", required=True)
    asl_checkpoint.add_argument("--output", required=True)
    asl_checkpoint.set_defaults(handler=analyze_asl_checkpoint_command)

    placement = commands.add_parser(
        "analyze-placement", help="compare interface knowledge in context, weights, and runtime"
    )
    placement.add_argument("--config", required=True)
    placement.add_argument("--output-dir", required=True)
    placement.set_defaults(handler=analyze_placement_command)

    public_gsm8k = commands.add_parser(
        "run-public-gsm8k", help="run one checkpointed model-facing GSM8K condition"
    )
    public_gsm8k.add_argument("--public-config", required=True)
    public_gsm8k.add_argument("--cache-root", required=True)
    public_gsm8k.add_argument("--selection", required=True)
    public_gsm8k.add_argument("--model-config", required=True)
    public_gsm8k.add_argument("--condition", required=True, choices=PUBLIC_GSM8K_CONDITIONS)
    public_gsm8k.add_argument("--output-dir", required=True)
    public_gsm8k.add_argument("--device")
    public_gsm8k.add_argument("--offset", type=int, default=0)
    public_gsm8k.add_argument("--limit", type=int)
    public_gsm8k.add_argument("--per-stratum", type=int)
    public_gsm8k.add_argument("--checkpoint-every", type=int, default=10)
    public_gsm8k.add_argument("--no-resume", action="store_true")
    public_gsm8k.set_defaults(handler=run_public_gsm8k_command)

    public_gsm8k_freeze = commands.add_parser(
        "freeze-public-gsm8k", help="freeze a balanced developmental GSM8K slice"
    )
    public_gsm8k_freeze.add_argument("--source-selection", required=True)
    public_gsm8k_freeze.add_argument("--per-stratum", type=int, default=40)
    public_gsm8k_freeze.add_argument("--output-dir", required=True)
    public_gsm8k_freeze.set_defaults(handler=freeze_public_gsm8k_command)

    public_gsm8k_analysis = commands.add_parser(
        "analyze-public-gsm8k", help="analyze matched public GSM8K condition runs"
    )
    public_gsm8k_analysis.add_argument("--predictions", required=True, action="append")
    public_gsm8k_analysis.add_argument("--output-dir", required=True)
    public_gsm8k_analysis.set_defaults(handler=analyze_public_gsm8k_command)
