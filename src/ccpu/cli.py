"""Top-level CLI for the cognitive-coprocessor series."""

from __future__ import annotations

import argparse

from .paper1.cli import add_commands as add_paper1_commands
from .paper1_5.cli import add_commands as add_paper1_5_commands
from .paper2.cli import add_commands as add_paper2_commands
from .paper2_5.cli import add_commands as add_paper2_5_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ccpu")
    papers = parser.add_subparsers(dest="paper", required=True)
    add_paper1_commands(papers)
    add_paper1_5_commands(papers)
    add_paper2_commands(papers)
    add_paper2_5_commands(papers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args) or 0)
