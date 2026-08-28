"""Paper 1.5 command-line workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    read_json,
    write_json,
    write_jsonl,
)

from .dataset import load_benchmark
from .evaluate import evaluate
from .experiment import run_huggingface
from .generation import ConfidenceBackend
from .plot import plot_pareto


def validate_command(args: argparse.Namespace) -> int:
    store, examples = load_benchmark(read_json(args.config))
    print(f"valid Paper 1.5 source {store.source_id}@{store.version} with {len(examples)} examples")
    return 0


def run_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    store, examples = load_benchmark(raw)
    model = dict(raw["model"])
    revision = str(model["revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Paper 1.5 model revision must be a pinned 40-character SHA")
    if args.device:
        model["device"] = args.device
    backend = ConfidenceBackend(model)
    predictions, traces, threshold = run_huggingface(
        examples,
        store,
        backend,
        seed=int(raw.get("seed", 17)),
    )
    output = Path(args.output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", predictions)
    traces_path = write_jsonl(output / "traces.jsonl", traces)
    summary = evaluate(predictions)
    summary["empirical"] = True
    summary_path = write_json(output / "summary.json", summary)
    benchmark_path = write_jsonl(output / "benchmark.jsonl", (example.to_dict() for example in examples))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 1.5",
            "schema_version": "ccpu.paper1_5.run_manifest.v1",
            "empirical": True,
            "source_id": store.source_id,
            "source_version": store.version,
            "config_sha256": file_sha256(args.config),
            "benchmark_sha256": file_sha256(benchmark_path),
            "predictions_sha256": file_sha256(predictions_path),
            "traces_sha256": file_sha256(traces_path),
            "summary_sha256": file_sha256(summary_path),
            "prediction_count": len(predictions),
            "trace_count": len(traces),
            "fitted_confidence_threshold": threshold,
            "model": model,
            "environment": environment_manifest(root),
        },
    )
    print(f"completed {len(predictions)} Paper 1.5 generations -> {output}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    from ccpu.common.artifacts import read_jsonl

    summary = evaluate(read_jsonl(args.predictions))
    write_json(args.output, summary)
    print(f"evaluated Paper 1.5 predictions -> {args.output}")
    return 0


def plot_command(args: argparse.Namespace) -> int:
    output = plot_pareto(read_json(args.summary), args.output)
    print(f"wrote Paper 1.5 Pareto plot -> {output}")
    return 0


def add_commands(papers: argparse._SubParsersAction) -> None:
    paper = papers.add_parser("paper1.5", aliases=["paper1_5"], help="epistemic-risk retrieval")
    commands = paper.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate source and benchmark")
    validate.add_argument("--config", required=True)
    validate.set_defaults(handler=validate_command)

    run = commands.add_parser("run-hf", help="run the pinned checkpoint experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device")
    run.set_defaults(handler=run_command)

    evaluation = commands.add_parser("evaluate", help="recompute summary metrics")
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(handler=evaluate_command)

    plot = commands.add_parser("plot", help="plot reliability versus retrieval cost")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output", required=True)
    plot.set_defaults(handler=plot_command)
