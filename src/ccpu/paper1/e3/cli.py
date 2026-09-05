"""Command line entry points for the Paper 1 E3 ladder."""

from __future__ import annotations

import argparse

from ccpu.common.artifacts import read_json, read_jsonl, write_json

from .contribution_analysis import analyze_gsm8k_contribution
from .data import (
    build_bottleneck_data,
    build_bottleneck_preference_data,
    build_direct_preference_data,
)
from .data_scale import (
    build_d1_f0_data,
    build_gsm8k_exposure_scale,
    build_gsm8k_f0_data,
    freeze_gsm8k_eval_views,
)
from .direct_answer_eval import (
    DIRECT_CONDITIONS,
    freeze_direct_gsm8k_protocol,
    merge_direct_gsm8k_shards,
    run_direct_gsm8k_shard,
)
from .eval import analyze_bottleneck_predictions, run_bottleneck_condition
from .gsm8k_confirmatory import (
    analyze_official_gsm8k_replications,
    freeze_official_gsm8k,
    merge_official_gsm8k_shards,
    run_official_gsm8k_shard,
)
from .large_number_suite import freeze_large_number_gsm8k
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
    gsm8k = commands.add_parser("prepare-gsm8k")
    gsm8k.add_argument("--strict", required=True)
    gsm8k.add_argument("--source", required=True)
    gsm8k.add_argument("--frozen-data-dir", required=True)
    gsm8k.add_argument("--output-dir", required=True)
    gsm8k.add_argument("--target", type=int, default=4500)
    gsm8k.add_argument("--epochs", type=int, default=10)
    gsm8k.add_argument("--seed", type=int, default=11)
    gsm8k_eval = commands.add_parser("prepare-gsm8k-eval")
    gsm8k_eval.add_argument("--train", required=True)
    gsm8k_eval.add_argument("--dev", required=True)
    gsm8k_eval.add_argument("--test", required=True)
    gsm8k_eval.add_argument("--output-dir", required=True)
    gsm8k_scale = commands.add_parser("prepare-gsm8k-scale")
    gsm8k_scale.add_argument("--parent-dir", required=True)
    gsm8k_scale.add_argument("--output-dir", required=True)
    gsm8k_scale.add_argument("--unique-rows", type=int, required=True)
    gsm8k_scale.add_argument("--exposures", type=int, default=4500)
    gsm8k_scale.add_argument("--epochs", type=int, default=10)
    gsm8k_scale.add_argument("--seed", type=int, default=11)
    gsm8k_official = commands.add_parser("prepare-gsm8k-official")
    gsm8k_official.add_argument("--source", required=True)
    gsm8k_official.add_argument("--train", action="append", required=True)
    gsm8k_official.add_argument("--output-dir", required=True)
    gsm8k_official.add_argument("--expected-sha256", required=True)
    gsm8k_official.add_argument("--expected-rows", type=int, default=1319)
    gsm8k_official.add_argument("--confirmatory-size", type=int, default=250)
    gsm8k_official.add_argument("--seed", type=int, default=22901)
    gsm8k_run = commands.add_parser("run-gsm8k-official-shard")
    gsm8k_run.add_argument("--eval", required=True)
    gsm8k_run.add_argument("--config", required=True)
    gsm8k_run.add_argument("--adapter-path", required=True)
    gsm8k_run.add_argument("--adapter-id", required=True)
    gsm8k_run.add_argument("--output-dir", required=True)
    gsm8k_run.add_argument("--shard-index", type=int, required=True)
    gsm8k_run.add_argument("--shard-count", type=int, required=True)
    gsm8k_run.add_argument("--seed", type=int, default=44017)
    gsm8k_run.add_argument("--checkpoint-every", type=int, default=5)
    gsm8k_merge = commands.add_parser("merge-gsm8k-official")
    gsm8k_merge.add_argument("--eval", required=True)
    gsm8k_merge.add_argument("--shard-dir", action="append", required=True)
    gsm8k_merge.add_argument("--output-dir", required=True)
    gsm8k_analyze = commands.add_parser("analyze-gsm8k-official")
    gsm8k_analyze.add_argument("--candidate", action="append", required=True)
    gsm8k_analyze.add_argument("--output", required=True)
    gsm8k_direct_freeze = commands.add_parser("prepare-gsm8k-direct")
    gsm8k_direct_freeze.add_argument("--eval", required=True)
    gsm8k_direct_freeze.add_argument("--config", action="append", required=True)
    gsm8k_direct_freeze.add_argument("--output-dir", required=True)
    gsm8k_direct = commands.add_parser("run-gsm8k-direct-shard")
    gsm8k_direct.add_argument("--eval", required=True)
    gsm8k_direct.add_argument("--config", required=True)
    gsm8k_direct.add_argument("--condition", choices=DIRECT_CONDITIONS, required=True)
    gsm8k_direct.add_argument("--output-dir", required=True)
    gsm8k_direct.add_argument("--shard-index", type=int, required=True)
    gsm8k_direct.add_argument("--shard-count", type=int, required=True)
    gsm8k_direct.add_argument("--seed", type=int, default=44017)
    gsm8k_direct.add_argument("--checkpoint-every", type=int, default=5)
    gsm8k_direct_merge = commands.add_parser("merge-gsm8k-direct")
    gsm8k_direct_merge.add_argument("--eval", required=True)
    gsm8k_direct_merge.add_argument("--shard-dir", action="append", required=True)
    gsm8k_direct_merge.add_argument("--output-dir", required=True)
    gsm8k_large = commands.add_parser("prepare-gsm8k-large-numbers")
    gsm8k_large.add_argument("--source", required=True)
    gsm8k_large.add_argument("--eval", required=True)
    gsm8k_large.add_argument("--output-dir", required=True)
    gsm8k_large.add_argument("--expected-source-sha256", required=True)
    gsm8k_large.add_argument("--factor", type=int, default=1000)
    contribution = commands.add_parser("analyze-gsm8k-contribution")
    contribution.add_argument("--original-eval", required=True)
    contribution.add_argument("--large-eval", required=True)
    contribution.add_argument("--original-direct", action="append", required=True)
    contribution.add_argument("--original-asl", action="append", required=True)
    contribution.add_argument("--large-direct", action="append", required=True)
    contribution.add_argument("--large-asl", action="append", required=True)
    contribution.add_argument("--output", required=True)
    contribution.add_argument("--bootstrap-seed", type=int, default=22903)
    contribution.add_argument("--bootstrap-samples", type=int, default=10000)
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
    if args.command == "prepare-gsm8k":
        manifest = build_gsm8k_f0_data(
            strict_path=args.strict,
            source_path=args.source,
            frozen_data_dir=args.frozen_data_dir,
            output_dir=args.output_dir,
            target=args.target,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(
            f"G1_GSM8K selected={manifest['counts']['selected']} "
            f"patterns={manifest['counts']['selected_patterns']} "
            f"rejected_non_gsm8k={manifest['counts']['strict_scope_rejected']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-eval":
        manifest = freeze_gsm8k_eval_views(
            train_path=args.train,
            dev_path=args.dev,
            test_path=args.test,
            output_dir=args.output_dir,
        )
        print(
            f"GSM8K dev={manifest['counts']['dev']['selected_gsm8k']} "
            f"test={manifest['counts']['test']['selected_gsm8k']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-scale":
        manifest = build_gsm8k_exposure_scale(
            parent_dir=args.parent_dir,
            output_dir=args.output_dir,
            unique_rows=args.unique_rows,
            exposures=args.exposures,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(
            f"GSM8K unique={manifest['counts']['unique_train_rows']} "
            f"exposures={manifest['counts']['exposures']} -> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-official":
        manifest = freeze_official_gsm8k(
            source_path=args.source,
            train_paths=args.train,
            output_dir=args.output_dir,
            expected_sha256=args.expected_sha256,
            expected_rows=args.expected_rows,
            confirmatory_size=args.confirmatory_size,
            seed=args.seed,
        )
        print(
            f"GSM8K official full={manifest['counts']['full']} "
            f"confirmatory={manifest['counts']['confirmatory']} -> {args.output_dir}"
        )
        return 0
    if args.command == "run-gsm8k-official-shard":
        summary = run_official_gsm8k_shard(
            eval_path=args.eval,
            model_config=read_json(args.config),
            adapter_path=args.adapter_path,
            adapter_id=args.adapter_id,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            f"GSM8K shard {args.shard_index}: answer="
            f"{summary['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
        )
        return 0
    if args.command == "merge-gsm8k-official":
        summary = merge_official_gsm8k_shards(
            eval_path=args.eval,
            shard_dirs=args.shard_dir,
            output_dir=args.output_dir,
        )
        print(
            f"GSM8K merged={summary['prediction_count']} answer="
            f"{summary['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-gsm8k-official":
        candidates = []
        for value in args.candidate:
            if "=" not in value:
                raise ValueError("--candidate must use LABEL=PATH")
            label, path = value.split("=", 1)
            candidates.append((label, path))
        report = analyze_official_gsm8k_replications(
            candidate_paths=candidates, output_path=args.output
        )
        print(
            f"GSM8K official seeds={report['seed_count']} "
            f"identities={report['identity_count']} -> {args.output}"
        )
        return 0
    if args.command == "run-gsm8k-direct-shard":
        summary = run_direct_gsm8k_shard(
            eval_path=args.eval,
            model_config=read_json(args.config),
            condition=args.condition,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
        )
        print(
            f"GSM8K {args.condition} shard {args.shard_index}: answer="
            f"{summary['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
        )
        return 0
    if args.command == "merge-gsm8k-direct":
        summary = merge_direct_gsm8k_shards(
            eval_path=args.eval,
            shard_dirs=args.shard_dir,
            output_dir=args.output_dir,
        )
        print(
            f"GSM8K direct merged={summary['prediction_count']} answer="
            f"{summary['rates']['final_answer_correct']:.3f} -> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-direct":
        manifest = freeze_direct_gsm8k_protocol(
            eval_path=args.eval,
            config_paths=args.config,
            output_dir=args.output_dir,
        )
        print(
            f"GSM8K direct protocol identities={manifest['identity_count']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-large-numbers":
        manifest = freeze_large_number_gsm8k(
            source_path=args.source,
            official_eval_path=args.eval,
            output_dir=args.output_dir,
            expected_source_sha256=args.expected_source_sha256,
            factor=args.factor,
        )
        print(
            f"GSM8K large-number eligible={manifest['counts']['eligible']} "
            f"excluded={manifest['counts']['excluded']} -> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-gsm8k-contribution":
        for values in (
            args.original_direct,
            args.original_asl,
            args.large_direct,
            args.large_asl,
        ):
            if any("=" not in value for value in values):
                raise ValueError("contribution paths must use LABEL=PATH")
        report = analyze_gsm8k_contribution(
            original_eval_path=args.original_eval,
            large_eval_path=args.large_eval,
            original_direct_paths=[value.split("=", 1) for value in args.original_direct],
            original_asl_paths=[value.split("=", 1) for value in args.original_asl],
            large_direct_paths=[value.split("=", 1) for value in args.large_direct],
            large_asl_paths=[value.split("=", 1) for value in args.large_asl],
            output_path=args.output,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(
            f"GSM8K contribution original={report['identity_counts']['original']} "
            f"large={report['identity_counts']['large_number']} -> {args.output}"
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
