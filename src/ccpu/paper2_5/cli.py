"""Paper 2.5 reproducible heterogeneous retrieval workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

from .benchmark import freeze_benchmark
from .composition import run_compositions
from .experiment import run_matrix, summarize
from .plot import decide_gate, plot_scaling


def freeze_command(args: argparse.Namespace) -> int:
    manifest = freeze_benchmark(args.output_dir)
    print(f"froze {manifest['count']} Paper 2.5 examples -> {args.output_dir}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    rows, traces = run_matrix(args.benchmark, source_count=args.source_count)
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    trace_path = write_jsonl(output / "traces.jsonl", traces)
    summary_path = write_json(output / "summary.json", summarize(rows))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2.5",
            "schema_version": "ccpu.paper2_5.run_manifest.v1",
            "source_count": args.source_count,
            "benchmark_sha256": file_sha256(args.benchmark),
            "predictions_sha256": file_sha256(predictions),
            "traces_sha256": file_sha256(trace_path),
            "summary_sha256": file_sha256(summary_path),
            "environment": environment_manifest(root),
        },
    )
    print(f"completed {len(rows)} Paper 2.5 matrix rows -> {output}")
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    rows = [row for path in args.predictions for row in read_jsonl(path)]
    summary = summarize(rows)
    output = Path(args.output_dir)
    write_json(output / "summary.json", summary)
    gate = decide_gate(summary)
    write_json(output / "paper3_5_gate.json", gate)
    plot_scaling(summary, output / "source_scaling.png")
    print(f"Paper 3.5 gate={gate['status']} from {len(rows)} matrix rows -> {output}")
    return 0


def plot_command(args: argparse.Namespace) -> int:
    output = plot_scaling(read_json(args.summary), args.output)
    print(f"wrote Paper 2.5 scaling plot -> {output}")
    return 0


def compositions_command(args: argparse.Namespace) -> int:
    rows, summary = run_compositions(args.count_per_family)
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary_path = write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2.5",
            "schema_version": "ccpu.paper2_5.composition_manifest.v1",
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary_path),
        },
    )
    print(f"completed {len(rows)} bounded Paper 2.5 compositions -> {output}")
    return 0


def add_commands(papers: argparse._SubParsersAction) -> None:
    paper = papers.add_parser("paper2.5", aliases=["paper2_5"], help="heterogeneous retrieval")
    commands = paper.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="freeze the source-optimal benchmark")
    freeze.add_argument("--output-dir", required=True)
    freeze.set_defaults(handler=freeze_command)

    run = commands.add_parser("run", help="run one source-count oracle matrix")
    run.add_argument("--benchmark", required=True)
    run.add_argument("--source-count", type=int, required=True, choices=(1, 2, 3, 4))
    run.add_argument("--output-dir", required=True)
    run.set_defaults(handler=run_command)

    analyze = commands.add_parser("analyze", help="merge scaling runs and decide Paper 3.5 gate")
    analyze.add_argument("--predictions", nargs="+", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.set_defaults(handler=analyze_command)

    plot = commands.add_parser("plot", help="plot source scaling")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output", required=True)
    plot.set_defaults(handler=plot_command)

    compositions = commands.add_parser(
        "compositions", help="run bounded entity/date resolver source compositions"
    )
    compositions.add_argument("--count-per-family", type=int, default=12)
    compositions.add_argument("--output-dir", required=True)
    compositions.set_defaults(handler=compositions_command)
