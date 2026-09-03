"""Command line entry points for the Paper 1 E3 ladder."""

from __future__ import annotations

import argparse

from ccpu.common.artifacts import read_json, read_jsonl, write_json

from .data import (
    build_bottleneck_data,
    build_bottleneck_preference_data,
    build_direct_preference_data,
)
from .data_scale import build_d1_f0_data
from .eval import analyze_bottleneck_predictions, run_bottleneck_condition
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
    f4_preference = commands.add_parser("prepare-bottleneck-preference")
    f4_preference.add_argument("--bottleneck-data-dir", required=True)
    f4_preference.add_argument("--output-dir", required=True)
    f4_preference.add_argument("--epochs", type=int, default=10)
    d1 = commands.add_parser("prepare-d1")
    d1.add_argument("--strict", required=True)
    d1.add_argument("--source", action="append", required=True)
    d1.add_argument("--frozen-data-dir", required=True)
    d1.add_argument("--output-dir", required=True)
    d1.add_argument("--target", type=int, default=4500)
    d1.add_argument("--epochs", type=int, default=10)
    d1.add_argument("--seed", type=int, default=11)
    select = commands.add_parser("select-checkpoint")
    select.add_argument("--metrics", required=True)
    select.add_argument("--output", required=True)
    run = commands.add_parser("run-bottleneck")
    run.add_argument("--eval", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--adapter-path", required=True)
    run.add_argument("--adapter-id", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--objective-id", default="L0")
    run.add_argument("--seed", type=int, default=44017)
    run.add_argument("--checkpoint-every", type=int, default=5)
    evaluate = commands.add_parser("evaluate-bottleneck")
    evaluate.add_argument("--eval", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output-dir", required=True)
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
    if args.command == "prepare-bottleneck-preference":
        report = build_bottleneck_preference_data(
            args.bottleneck_data_dir, args.output_dir, epochs=args.epochs
        )
        print(
            f"built {report['files']['train']['rows']} train and "
            f"{report['files']['dev']['rows']} dev native-F4 preference rows "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-d1":
        manifest = build_d1_f0_data(
            strict_path=args.strict,
            source_paths=args.source,
            frozen_data_dir=args.frozen_data_dir,
            output_dir=args.output_dir,
            target=args.target,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(
            f"D1 selected={manifest['counts']['selected']} "
            f"patterns={manifest['counts']['selected_patterns']} -> {args.output_dir}"
        )
        return 0
    if args.command == "run-bottleneck":
        summary = run_bottleneck_condition(
            eval_path=args.eval,
            model_config=read_json(args.config),
            adapter_path=args.adapter_path,
            adapter_id=args.adapter_id,
            output_dir=args.output_dir,
            objective_id=args.objective_id,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            f"F4/{args.objective_id} answer="
            f"{summary['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
        )
        return 0
    if args.command == "evaluate-bottleneck":
        summary = analyze_bottleneck_predictions(
            args.eval, args.predictions, args.output_dir
        )
        print(
            f"F4 answer={summary['rates']['final_answer_correct']:.3f} "
            f"-> {args.output_dir}"
        )
        return 0
    report = select_semantic_checkpoint(read_jsonl(args.metrics))
    write_json(args.output, report)
    print(f"selected {report['selected_checkpoint']} -> {args.output}")
    return 0
