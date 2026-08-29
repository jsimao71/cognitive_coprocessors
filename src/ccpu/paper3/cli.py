"""Paper 3 registered public-control workflow."""

from __future__ import annotations

import argparse

from .public_benchmarks import freeze_public_control_registry


def freeze_public_command(args: argparse.Namespace) -> int:
    manifest = freeze_public_control_registry(
        args.compute_selection,
        args.crag_selection,
        args.tatqa_selection,
        args.output_dir,
        per_benchmark=args.per_benchmark,
        crag_diagnostics=args.crag_diagnostics,
        paper2_5_readiness=args.paper2_5_readiness,
    )
    print(
        f"froze {manifest['record_count']} Paper 3 public control rows; "
        f"headline_ready={manifest['headline_ready']}"
    )
    return 0


def add_commands(papers: argparse._SubParsersAction) -> None:
    paper = papers.add_parser("paper3", help="latent cognitive control")
    commands = paper.add_subparsers(dest="command", required=True)
    public = commands.add_parser(
        "freeze-public", help="freeze the cross-paper public control registry"
    )
    public.add_argument("--compute-selection", required=True)
    public.add_argument("--crag-selection", required=True)
    public.add_argument("--tatqa-selection", required=True)
    public.add_argument("--per-benchmark", type=int, default=40)
    public.add_argument("--crag-diagnostics")
    public.add_argument("--paper2-5-readiness")
    public.add_argument("--output-dir", required=True)
    public.set_defaults(handler=freeze_public_command)
