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
from .enterprise import create_enterprise_fixture, run_enterprise_evaluation
from .experiment import run_matrix, summarize
from .generic_tools import compare_enterprise_result_transports
from .plot import decide_gate, plot_scaling
from .production_analysis import analyze_substitution
from .public_benchmarks import (
    analyze_tatqa_composition,
    analyze_tatqa_retrieval,
    freeze_tatqa_subset,
)


def freeze_command(args: argparse.Namespace) -> int:
    manifest = freeze_benchmark(args.output_dir)
    print(f"froze {manifest['count']} Paper 2.5 examples -> {args.output_dir}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    rows, traces = run_matrix(
        args.benchmark, source_count=args.source_count, backend_suite=args.backend_suite
    )
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
            "backend_suite": args.backend_suite,
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


def analyze_production_command(args: argparse.Namespace) -> int:
    controlled = read_jsonl(args.controlled_predictions)
    production = read_jsonl(args.production_predictions)
    traces = read_jsonl(args.production_traces)
    output = Path(args.output_dir)
    summary_path = write_json(
        output / "substitution_summary.json",
        analyze_substitution(controlled, production, traces),
    )
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2.5",
            "schema_version": "ccpu.paper2_5.production_analysis_manifest.v1",
            "controlled_predictions_sha256": file_sha256(args.controlled_predictions),
            "production_predictions_sha256": file_sha256(args.production_predictions),
            "production_traces_sha256": file_sha256(args.production_traces),
            "substitution_summary_sha256": file_sha256(summary_path),
        },
    )
    print(f"analyzed Paper 2.5 backend substitution -> {output}")
    return 0


def prepare_enterprise_command(args: argparse.Namespace) -> int:
    manifest = create_enterprise_fixture(args.output_dir)
    print(
        "created Paper 2.5 enterprise Iceberg fixture "
        f"at snapshot {manifest['sales']['current_snapshot_id']} -> {args.output_dir}"
    )
    return 0


def run_enterprise_command(args: argparse.Namespace) -> int:
    rows, summary = run_enterprise_evaluation(args.fixture_root)
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary_path = write_json(output / "summary.json", summary)
    fixture_manifest = Path(args.fixture_root) / "manifest.json"
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2.5",
            "schema_version": "ccpu.paper2_5.enterprise_run_manifest.v1",
            "fixture_manifest_sha256": file_sha256(fixture_manifest),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary_path),
        },
    )
    print(f"completed {summary['count']} Paper 2.5 enterprise cases -> {output}")
    return 0


def freeze_public_tatqa_command(args: argparse.Namespace) -> int:
    manifest = freeze_tatqa_subset(args.config, args.cache_root, args.output_dir)
    print(
        f"froze {manifest['record_count']} pinned TAT-QA questions -> {args.output_dir}"
    )
    return 0


def analyze_public_tatqa_command(args: argparse.Namespace) -> int:
    summary = analyze_tatqa_composition(
        args.config,
        args.cache_root,
        args.selection,
        args.output_dir,
    )
    print(
        "analyzed TAT-QA retrieval-compute composition "
        f"for {summary['record_count']} questions -> {args.output_dir}"
    )
    return 0


def analyze_public_tatqa_retrieval_command(args: argparse.Namespace) -> int:
    summary = analyze_tatqa_retrieval(
        args.config,
        args.cache_root,
        args.selection,
        args.output_dir,
        limit=args.top_k,
    )
    print(
        f"analyzed {summary['evaluable_count']} TAT-QA evidence labels "
        f"at top-{summary['top_k']} -> {args.output_dir}"
    )
    return 0


def compare_generic_tools_command(args: argparse.Namespace) -> int:
    summary = compare_enterprise_result_transports(args.fixture_root, args.output_dir)
    print(
        f"compared {summary['row_count']} Paper 2.5 enterprise result transports; "
        f"agreement={summary['record_agreement']:.3f}"
    )
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
    run.add_argument(
        "--backend-suite",
        choices=("controlled", "local_production"),
        default="controlled",
    )
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

    production = commands.add_parser(
        "analyze-production", help="compare controlled and local production backends"
    )
    production.add_argument("--controlled-predictions", required=True)
    production.add_argument("--production-predictions", required=True)
    production.add_argument("--production-traces", required=True)
    production.add_argument("--output-dir", required=True)
    production.set_defaults(handler=analyze_production_command)

    enterprise = commands.add_parser(
        "prepare-enterprise", help="create the local Iceberg and governed-semantics fixture"
    )
    enterprise.add_argument("--output-dir", required=True)
    enterprise.set_defaults(handler=prepare_enterprise_command)

    enterprise_run = commands.add_parser(
        "run-enterprise", help="run native governed composition and universal baseline"
    )
    enterprise_run.add_argument("--fixture-root", required=True)
    enterprise_run.add_argument("--output-dir", required=True)
    enterprise_run.set_defaults(handler=run_enterprise_command)

    public_tatqa = commands.add_parser(
        "freeze-public-tatqa", help="freeze the pinned TAT-QA diagnostic subset"
    )
    public_tatqa.add_argument("--config", required=True)
    public_tatqa.add_argument("--cache-root", required=True)
    public_tatqa.add_argument("--output-dir", required=True)
    public_tatqa.set_defaults(handler=freeze_public_tatqa_command)

    public_tatqa_analysis = commands.add_parser(
        "analyze-public-tatqa", help="audit TAT-QA retrieval-compute composition"
    )
    public_tatqa_analysis.add_argument("--config", required=True)
    public_tatqa_analysis.add_argument("--cache-root", required=True)
    public_tatqa_analysis.add_argument("--selection", required=True)
    public_tatqa_analysis.add_argument("--output-dir", required=True)
    public_tatqa_analysis.set_defaults(handler=analyze_public_tatqa_command)

    public_tatqa_retrieval = commands.add_parser(
        "analyze-public-tatqa-retrieval",
        help="compare matched lexical TAT-QA evidence retrieval",
    )
    public_tatqa_retrieval.add_argument("--config", required=True)
    public_tatqa_retrieval.add_argument("--cache-root", required=True)
    public_tatqa_retrieval.add_argument("--selection", required=True)
    public_tatqa_retrieval.add_argument("--top-k", type=int, default=5)
    public_tatqa_retrieval.add_argument("--output-dir", required=True)
    public_tatqa_retrieval.set_defaults(handler=analyze_public_tatqa_retrieval_command)

    generic_tools = commands.add_parser(
        "compare-generic-tools", help="audit enterprise tool/CogCop result transport"
    )
    generic_tools.add_argument("--fixture-root", required=True)
    generic_tools.add_argument("--output-dir", required=True)
    generic_tools.set_defaults(handler=compare_generic_tools_command)
