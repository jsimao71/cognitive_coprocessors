"""Command line entry points for the Paper 1 E3 ladder."""

from __future__ import annotations

import argparse

from ccpu.common.artifacts import read_jsonl, write_json

from .data import build_bottleneck_data, build_direct_preference_data
from .selection import select_semantic_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ccpu.paper1.e3")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-bottleneck")
    prepare.add_argument("--data-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    preference = commands.add_parser("prepare-direct-preference")
    preference.add_argument("--qwen-data-dir", required=True)
    preference.add_argument("--bottleneck-data-dir", required=True)
    preference.add_argument("--output-dir", required=True)
    select = commands.add_parser("select-checkpoint")
    select.add_argument("--metrics", required=True)
    select.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-bottleneck":
        report = build_bottleneck_data(args.data_dir, args.output_dir)
        print(
            f"round-trip {report['gold_roundtrip_passed']}/"
            f"{report['gold_roundtrip_total']} -> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-direct-preference":
        report = build_direct_preference_data(
            args.qwen_data_dir, args.bottleneck_data_dir, args.output_dir
        )
        print(
            f"built {report['files']['train']['rows']} train and "
            f"{report['files']['dev']['rows']} dev preference rows -> {args.output_dir}"
        )
        return 0
    report = select_semantic_checkpoint(read_jsonl(args.metrics))
    write_json(args.output, report)
    print(f"selected {report['selected_checkpoint']} -> {args.output}")
    return 0
