"""Paper 2 deterministic protocol artifact workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

from .dataset import MixedBenchmarkConfig, MixedExample, iter_benchmark
from .evaluate import evaluate
from .experiment import run_scripted
from .plot import plot_scaling


def _examples(path: str | Path) -> list[MixedExample]:
    return [MixedExample.from_dict(row) for row in read_jsonl(path)]


def generate_command(args: argparse.Namespace) -> int:
    config = MixedBenchmarkConfig.from_dict(read_json(args.config))
    examples = list(iter_benchmark(config))
    output = write_jsonl(args.output, (example.to_dict() for example in examples))
    write_json(
        Path(output).with_suffix(".manifest.json"),
        {
            "paper": "Paper 2",
            "schema_version": "ccpu.paper2.dataset_manifest.v1",
            "config": config.to_dict(),
            "config_fingerprint": fingerprint(config.to_dict()),
            "record_count": len(examples),
            "dataset_sha256": file_sha256(output),
        },
    )
    print(f"generated {len(examples)} Paper 2 examples -> {output}")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    examples = _examples(args.dataset)
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Paper 2 example IDs must be unique")
    print(f"valid {len(examples)}-item Paper 2 mixed benchmark")
    return 0


def simulate_command(args: argparse.Namespace) -> int:
    examples = _examples(args.dataset)
    predictions, traces = run_scripted(examples)
    output = Path(args.output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", predictions)
    traces_path = write_jsonl(output / "traces.jsonl", traces)
    summary_path = write_json(output / "summary.json", evaluate(predictions))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2",
            "schema_version": "ccpu.paper2.run_manifest.v1",
            "empirical": False,
            "dataset_sha256": file_sha256(args.dataset),
            "predictions_sha256": file_sha256(predictions_path),
            "traces_sha256": file_sha256(traces_path),
            "summary_sha256": file_sha256(summary_path),
            "prediction_count": len(predictions),
            "trace_count": len(traces),
            "environment": environment_manifest(root),
        },
    )
    print(f"completed non-empirical Paper 2 protocol smoke -> {output}")
    return 0


def plot_command(args: argparse.Namespace) -> int:
    output = plot_scaling(read_json(args.summary), args.output)
    print(f"wrote Paper 2 scaling plot -> {output}")
    return 0


def add_commands(papers: argparse._SubParsersAction) -> None:
    paper = papers.add_parser("paper2", help="heterogeneous symbolic protocol")
    commands = paper.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate the mixed benchmark")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.set_defaults(handler=generate_command)

    validate = commands.add_parser("validate", help="validate the mixed benchmark")
    validate.add_argument("--dataset", required=True)
    validate.set_defaults(handler=validate_command)

    simulate = commands.add_parser("simulate", help="run the non-empirical protocol smoke")
    simulate.add_argument("--dataset", required=True)
    simulate.add_argument("--output-dir", required=True)
    simulate.set_defaults(handler=simulate_command)

    plot = commands.add_parser("plot", help="plot non-empirical scaling cells")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output", required=True)
    plot.set_defaults(handler=plot_command)
