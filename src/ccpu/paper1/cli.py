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

from .dataset import (
    ArithmeticDatasetConfig,
    ArithmeticExample,
    HardArithmeticDatasetConfig,
    iter_dataset,
    iter_hard_dataset,
)
from .evaluate import evaluate, paired_comparisons, rescore_endpoint_predictions
from .experiment import run_huggingface, run_replay, run_scripted
from .generation import HuggingFaceBackend, HuggingFaceGenerationConfig
from .plot import plot_interface_diagnostics, plot_scaling
from .prompts import (
    CONDITIONS,
    CORE_CONDITIONS,
    ICL_ORDER_CONTROL_PROMPT_VERSION,
    ICL_PROMPT_VERSION,
    PROMPT_VERSION,
)

PROTOCOL_VERSIONS = {
    "prompt": PROMPT_VERSION,
    "calculator_block_icl_prompt": ICL_PROMPT_VERSION,
    "calculator_block_icl_order_control_prompt": ICL_ORDER_CONTROL_PROMPT_VERSION,
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
        peaks = [int(item["peak_memory_bytes"]) for item in metadata if item.get("peak_memory_bytes")]
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
                "wall_time_seconds": sum(int(row.get("wall_time_ns", 0)) for row in members)
                / 1e9,
                "model_memory_bytes": next(
                    (item.get("model_memory_bytes") for item in metadata if item.get("model_memory_bytes")),
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
            "extractor_version": "paper1_condition_independent_endpoint_v1",
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
    result = paired_comparisons(
        _examples(args.dataset), read_jsonl(args.predictions), baseline=args.baseline
    )
    result["dataset_sha256"] = file_sha256(args.dataset)
    result["predictions_sha256"] = file_sha256(args.predictions)
    write_json(args.output, result)
    print(f"wrote paired comparisons -> {args.output}")
    return 0


def interface_plot_command(args: argparse.Namespace) -> int:
    output = plot_interface_diagnostics(read_json(args.summary), args.output)
    print(f"wrote interface figure -> {output}")
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
    paired.add_argument("--predictions", required=True)
    paired.add_argument("--baseline", default="llm_only")
    paired.add_argument("--output", required=True)
    paired.set_defaults(handler=paired_command)

    interface_plot = commands.add_parser(
        "plot-interfaces", help="plot held-out accuracy and interface diagnostics"
    )
    interface_plot.add_argument("--summary", required=True)
    interface_plot.add_argument("--output", required=True)
    interface_plot.set_defaults(handler=interface_plot_command)
