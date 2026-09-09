"""Command line entry points for the Paper 1 E3 ladder."""

from __future__ import annotations

import argparse

from ccpu.common.artifacts import read_json, read_jsonl, write_json

from .augmentation_analysis import analyze_gsm8k_augmentation_stage
from .contribution_analysis import analyze_gsm8k_contribution
from .cross_dataset import (
    build_cross_dataset_sft_data,
    build_cross_dataset_teacher_seed,
    freeze_cross_dataset_benchmark,
)
from .cross_dataset_analysis import (
    analyze_cross_dataset_transfer,
    plot_cross_dataset_transfer,
)
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
    prepare_long_budget_resume,
    run_direct_gsm8k_shard,
)
from .eval import analyze_bottleneck_predictions, run_bottleneck_condition
from .gsm8k_confirmatory import (
    analyze_official_gsm8k_replications,
    freeze_official_gsm8k,
    merge_official_gsm8k_shards,
    run_official_gsm8k_shard,
)
from .gsm8k_interventions import (
    freeze_gsm8k_matched_interventions,
    freeze_gsm8k_operator_jitter_matrix,
)
from .intervention_analysis import analyze_matched_interventions
from .large_number_suite import freeze_large_number_gsm8k, freeze_magnitude_ladder_gsm8k
from .magnitude_analysis import (
    analyze_magnitude_curve,
    analyze_magnitude_failures,
    project_magnitude_predictions,
)
from .model_size_analysis import analyze_model_size_interaction
from .operator_analysis import analyze_operator_ladder, analyze_operator_pilot
from .operator_complexity import freeze_o1_dataset, freeze_o1_pilot
from .operator_levels import freeze_operator_level, freeze_operator_pilot
from .result_plots import build_gsm8k_result_plots
from .selection import select_semantic_checkpoint
from .semantic_augmentation import (
    build_entity_rename_increment,
    build_relation_paraphrase_increment,
    freeze_augmentation_selection_gate,
)


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
    gsm8k_augmentation = commands.add_parser("prepare-gsm8k-semantic-augmentation")
    gsm8k_augmentation.add_argument("--parent-train", required=True)
    gsm8k_augmentation.add_argument("--eligible", required=True)
    gsm8k_augmentation.add_argument("--output-dir", required=True)
    gsm8k_augmentation.add_argument(
        "--stage", choices=("relation_paraphrase",), default="relation_paraphrase"
    )
    gsm8k_augmentation.add_argument("--variants-per-parent", type=int, default=1)
    gsm8k_augmentation.add_argument("--max-rows", type=int)
    gsm8k_augmentation.add_argument("--seed", type=int, default=73031)
    entity_augmentation = commands.add_parser("prepare-gsm8k-entity-augmentation")
    entity_augmentation.add_argument("--parent-train", required=True)
    entity_augmentation.add_argument("--eligible", required=True)
    entity_augmentation.add_argument("--output-dir", required=True)
    entity_augmentation.add_argument("--max-rows", type=int)
    entity_augmentation.add_argument("--seed", type=int, default=73035)
    augmentation_gate = commands.add_parser("prepare-gsm8k-augmentation-gate")
    augmentation_gate.add_argument("--full-eval", required=True)
    augmentation_gate.add_argument("--excluded-eval", required=True)
    augmentation_gate.add_argument("--output-dir", required=True)
    augmentation_gate.add_argument("--count", type=int, default=250)
    augmentation_gate.add_argument("--seed", type=int, default=73033)
    augmentation_analysis = commands.add_parser("analyze-gsm8k-augmentation")
    augmentation_analysis.add_argument("--stage", required=True)
    augmentation_analysis.add_argument("--selection-eval", required=True)
    augmentation_analysis.add_argument("--historical-eval", required=True)
    augmentation_analysis.add_argument("--baseline-selection", required=True)
    augmentation_analysis.add_argument("--candidate-selection", required=True)
    augmentation_analysis.add_argument("--baseline-historical", required=True)
    augmentation_analysis.add_argument("--candidate-historical", required=True)
    augmentation_analysis.add_argument("--baseline-historical-summary", required=True)
    augmentation_analysis.add_argument("--candidate-historical-summary", required=True)
    augmentation_analysis.add_argument("--output", required=True)
    gsm8k_official = commands.add_parser("prepare-gsm8k-official")
    gsm8k_official.add_argument("--source", required=True)
    gsm8k_official.add_argument("--train", action="append", required=True)
    gsm8k_official.add_argument("--output-dir", required=True)
    gsm8k_official.add_argument("--expected-sha256", required=True)
    gsm8k_official.add_argument("--expected-rows", type=int, default=1319)
    gsm8k_official.add_argument("--confirmatory-size", type=int, default=250)
    gsm8k_official.add_argument("--seed", type=int, default=22901)
    cross_dataset = commands.add_parser("prepare-cross-dataset")
    cross_dataset.add_argument(
        "--dataset", choices=("asdiv", "svamp", "mawps", "gsm_plus"), required=True
    )
    cross_dataset.add_argument("--source", action="append", required=True)
    cross_dataset.add_argument("--expected-sha256", action="append", required=True)
    cross_dataset.add_argument("--gsm-train", action="append", required=True)
    cross_dataset.add_argument("--gsm-adapter", required=True)
    cross_dataset.add_argument("--output-dir", required=True)
    cross_dataset.add_argument("--diagnostic-size", type=int, default=250)
    cross_dataset.add_argument("--dev-size", type=int, default=100)
    cross_dataset.add_argument("--seed", type=int, default=93001)
    cross_seed = commands.add_parser("prepare-cross-dataset-teacher-seed")
    cross_seed.add_argument("--frozen-dir", required=True)
    cross_seed.add_argument("--output", required=True)
    cross_seed.add_argument(
        "--source-role", choices=("train_source", "dev"), default="train_source"
    )
    cross_seed.add_argument("--max-records", type=int)
    cross_sft = commands.add_parser("prepare-cross-dataset-sft")
    cross_sft.add_argument("--frozen-dir", required=True)
    cross_sft.add_argument("--accepted-train", required=True)
    cross_sft.add_argument("--accepted-dev", required=True)
    cross_sft.add_argument("--output-dir", required=True)
    cross_sft.add_argument("--seed", type=int, default=93001)
    cross_analysis = commands.add_parser("analyze-cross-dataset-transfer")
    cross_analysis.add_argument("--manifest", action="append", required=True)
    cross_analysis.add_argument("--prediction", action="append", required=True)
    cross_analysis.add_argument("--output", required=True)
    cross_plot = commands.add_parser("plot-cross-dataset-transfer")
    cross_plot.add_argument("--analysis", required=True)
    cross_plot.add_argument("--output", required=True)
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
    gsm8k_long_resume = commands.add_parser("prepare-gsm8k-long-budget-resume")
    gsm8k_long_resume.add_argument("--source-predictions", required=True)
    gsm8k_long_resume.add_argument("--output-dir", required=True)
    gsm8k_long_resume.add_argument("--source-ceiling", type=int, required=True)
    gsm8k_long_resume.add_argument("--target-ceiling", type=int, required=True)
    gsm8k_large = commands.add_parser("prepare-gsm8k-large-numbers")
    gsm8k_large.add_argument("--source", required=True)
    gsm8k_large.add_argument("--eval", required=True)
    gsm8k_large.add_argument("--output-dir", required=True)
    gsm8k_large.add_argument("--expected-source-sha256", required=True)
    gsm8k_large.add_argument("--factor", type=int, default=1000)
    magnitude_ladder = commands.add_parser("prepare-gsm8k-magnitude-ladder")
    magnitude_ladder.add_argument("--source", required=True)
    magnitude_ladder.add_argument("--eval", required=True)
    magnitude_ladder.add_argument("--output-dir", required=True)
    magnitude_ladder.add_argument("--expected-source-sha256", required=True)
    magnitude_ladder.add_argument("--factor", type=int, action="append", required=True)
    magnitude_failures = commands.add_parser("analyze-gsm8k-magnitude-failures")
    magnitude_failures.add_argument("--eval", required=True)
    magnitude_failures.add_argument("--original-predictions", required=True)
    magnitude_failures.add_argument("--transformed-predictions", required=True)
    magnitude_failures.add_argument("--output-dir", required=True)
    magnitude_curve = commands.add_parser("analyze-gsm8k-magnitude-curve")
    magnitude_curve.add_argument("--eval", action="append", required=True)
    magnitude_curve.add_argument("--prediction", action="append", required=True)
    magnitude_curve.add_argument("--output-dir", required=True)
    magnitude_projection = commands.add_parser("project-gsm8k-magnitude-predictions")
    magnitude_projection.add_argument("--source-eval", required=True)
    magnitude_projection.add_argument("--target-eval", required=True)
    magnitude_projection.add_argument("--source-predictions", required=True)
    magnitude_projection.add_argument("--output-dir", required=True)
    result_plots = commands.add_parser("plot-gsm8k-main-results")
    result_plots.add_argument("--contribution", required=True)
    result_plots.add_argument("--large-eval", required=True)
    result_plots.add_argument("--small-original", required=True)
    result_plots.add_argument("--small-large", required=True)
    result_plots.add_argument("--large-original", required=True)
    result_plots.add_argument("--large-large", required=True)
    result_plots.add_argument("--failure-summary", action="append", required=True)
    result_plots.add_argument("--output-dir", required=True)
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
    model_size = commands.add_parser("analyze-gsm8k-model-size")
    model_size.add_argument("--small-report", required=True)
    model_size.add_argument("--large-report", required=True)
    model_size.add_argument("--small-pair", required=True)
    model_size.add_argument("--large-pair", required=True)
    model_size.add_argument("--output", required=True)
    operator_o1 = commands.add_parser("prepare-operator-o1")
    operator_o1.add_argument("--output-dir", required=True)
    operator_o1.add_argument("--train-count", type=int, default=2000)
    operator_o1.add_argument("--dev-count", type=int, default=100)
    operator_o1.add_argument("--test-count", type=int, default=250)
    operator_o1.add_argument("--seed", type=int, default=81001)
    operator_o1_pilot = commands.add_parser("prepare-operator-o1-pilot")
    operator_o1_pilot.add_argument("--source-dir", required=True)
    operator_o1_pilot.add_argument("--output-dir", required=True)
    operator_o1_pilot.add_argument("--train-count", type=int, default=250)
    operator_o1_pilot.add_argument("--dev-count", type=int, default=30)
    operator_o1_pilot.add_argument("--test-count", type=int, default=100)
    operator_o1_pilot.add_argument("--seed", type=int, default=99173)
    operator_analysis = commands.add_parser("analyze-operator-pilot")
    operator_analysis.add_argument("--eval", required=True)
    operator_analysis.add_argument("--direct-predictions", required=True)
    operator_analysis.add_argument("--ap-predictions", required=True)
    operator_analysis.add_argument("--as-predictions", required=True)
    operator_analysis.add_argument("--output-dir", required=True)
    operator_ladder = commands.add_parser("analyze-operator-ladder")
    operator_ladder.add_argument("--summary", action="append", required=True)
    operator_ladder.add_argument("--output-dir", required=True)
    operator_level = commands.add_parser("prepare-operator-level")
    operator_level.add_argument("--level", choices=("O0", "O3", "O5", "O6"), required=True)
    operator_level.add_argument("--output-dir", required=True)
    operator_level.add_argument("--train-count", type=int, default=2000)
    operator_level.add_argument("--dev-count", type=int, default=100)
    operator_level.add_argument("--test-count", type=int, default=250)
    operator_level.add_argument("--seed", type=int, default=81001)
    operator_level_pilot = commands.add_parser("prepare-operator-level-pilot")
    operator_level_pilot.add_argument(
        "--level", choices=("O0", "O3", "O5", "O6"), required=True
    )
    operator_level_pilot.add_argument("--source-dir", required=True)
    operator_level_pilot.add_argument("--output-dir", required=True)
    operator_level_pilot.add_argument("--train-count", type=int, default=250)
    operator_level_pilot.add_argument("--dev-count", type=int, default=30)
    operator_level_pilot.add_argument("--test-count", type=int, default=100)
    operator_level_pilot.add_argument("--seed", type=int, default=99173)
    matched_interventions = commands.add_parser("prepare-gsm8k-matched-interventions")
    matched_interventions.add_argument("--source-corpus", required=True)
    matched_interventions.add_argument("--excluded-training", required=True)
    matched_interventions.add_argument("--output-dir", required=True)
    matched_interventions.add_argument("--train-count", type=int, default=250)
    matched_interventions.add_argument("--dev-count", type=int, default=30)
    matched_interventions.add_argument("--test-count", type=int, default=100)
    matched_interventions.add_argument("--factor", type=int, action="append")
    matched_interventions.add_argument("--jitter-seed", type=int, action="append")
    matched_interventions.add_argument("--seed", type=int, default=99173)
    combined_interventions = commands.add_parser("prepare-gsm8k-operator-jitter-matrix")
    combined_interventions.add_argument("--panel-dir", required=True)
    combined_interventions.add_argument("--output-dir", required=True)
    combined_interventions.add_argument("--factor", type=int, action="append")
    combined_interventions.add_argument("--jitter-seed", type=int, action="append")
    combined_interventions.add_argument(
        "--operator-level", choices=("O1", "O5", "O6"), action="append"
    )
    matched_analysis = commands.add_parser("analyze-gsm8k-matched-interventions")
    matched_analysis.add_argument(
        "--cell",
        action="append",
        required=True,
        help="LABEL=EVAL|DIRECT_PREDICTIONS|ASL_PREDICTIONS",
    )
    matched_analysis.add_argument("--output-dir", required=True)
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
    if args.command == "prepare-gsm8k-semantic-augmentation":
        manifest = build_relation_paraphrase_increment(
            parent_train_path=args.parent_train,
            eligible_path=args.eligible,
            output_dir=args.output_dir,
            variants_per_parent=args.variants_per_parent,
            max_rows=args.max_rows,
            seed=args.seed,
        )
        print(
            f"GSM8K {manifest['stage']} rows={manifest['counts']['increment_rows']} "
            f"parents={manifest['counts']['matched_parent_ids']} -> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-entity-augmentation":
        manifest = build_entity_rename_increment(
            parent_train_path=args.parent_train,
            eligible_path=args.eligible,
            output_dir=args.output_dir,
            max_rows=args.max_rows,
            seed=args.seed,
        )
        print(
            f"GSM8K {manifest['stage']} rows={manifest['counts']['increment_rows']} "
            f"renamed-entities={manifest['counts']['renamed_entities']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-augmentation-gate":
        manifest = freeze_augmentation_selection_gate(
            full_eval_path=args.full_eval,
            excluded_eval_path=args.excluded_eval,
            output_dir=args.output_dir,
            count=args.count,
            seed=args.seed,
        )
        print(
            f"GSM8K augmentation selection={manifest['counts']['selected']} "
            f"excluded-confirmation={manifest['counts']['excluded_final_confirmation']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-gsm8k-augmentation":
        report = analyze_gsm8k_augmentation_stage(
            stage=args.stage,
            selection_eval_path=args.selection_eval,
            historical_eval_path=args.historical_eval,
            baseline_selection_predictions=args.baseline_selection,
            candidate_selection_predictions=args.candidate_selection,
            baseline_historical_predictions=args.baseline_historical,
            candidate_historical_predictions=args.candidate_historical,
            baseline_historical_summary=args.baseline_historical_summary,
            candidate_historical_summary=args.candidate_historical_summary,
            output_path=args.output,
        )
        decision = "ACCEPT" if report["decision"]["accepted"] else "REJECT"
        print(
            f"GSM8K {args.stage} {decision} "
            f"delta={report['decision']['delta_correct']:+d}/"
            f"{report['augmentation_selection']['count']} -> {args.output}"
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
    if args.command == "prepare-cross-dataset":
        def keyed(values: list[str], option: str) -> dict[str, str]:
            result = {}
            for value in values:
                if "=" not in value:
                    raise ValueError(f"{option} must use ROLE=VALUE")
                role, item = value.split("=", 1)
                if role in result:
                    raise ValueError(f"duplicate {option} role: {role}")
                result[role] = item
            return result

        manifest = freeze_cross_dataset_benchmark(
            dataset=args.dataset,
            source_paths=keyed(args.source, "--source"),
            expected_sha256=keyed(args.expected_sha256, "--expected-sha256"),
            gsm_train_paths=args.gsm_train,
            gsm_adapter_path=args.gsm_adapter,
            output_dir=args.output_dir,
            diagnostic_size=args.diagnostic_size,
            dev_size=args.dev_size,
            seed=args.seed,
        )
        print(
            f"{args.dataset} diagnostic={manifest['counts']['diagnostic']} "
            f"train={manifest['counts']['train_source']} -> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-cross-dataset-teacher-seed":
        manifest = build_cross_dataset_teacher_seed(
            frozen_dir=args.frozen_dir,
            output_path=args.output,
            source_role=args.source_role,
            max_records=args.max_records,
        )
        print(
            f"{manifest['dataset']} teacher seeds={manifest['count']} -> {args.output}"
        )
        return 0
    if args.command == "prepare-cross-dataset-sft":
        manifest = build_cross_dataset_sft_data(
            frozen_dir=args.frozen_dir,
            accepted_train_path=args.accepted_train,
            accepted_dev_path=args.accepted_dev,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(
            f"{manifest['dataset']} SFT train={manifest['counts']['train']} "
            f"dev={manifest['counts']['dev']} -> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-cross-dataset-transfer":
        manifests = {}
        for value in args.manifest:
            if "=" not in value:
                raise ValueError("--manifest must use DATASET=PATH")
            dataset, path = value.split("=", 1)
            manifests[dataset] = path
        predictions = {}
        for value in args.prediction:
            if "=" not in value or ":" not in value.split("=", 1)[0]:
                raise ValueError("--prediction must use DATASET:CONDITION=PATH")
            key, path = value.split("=", 1)
            dataset, condition = key.split(":", 1)
            predictions[(dataset, condition)] = path
        report = analyze_cross_dataset_transfer(
            manifest_paths=manifests,
            prediction_paths=predictions,
            output_path=args.output,
        )
        print(
            f"cross-dataset conditions={sorted(report['macro_answer_accuracy'])} "
            f"-> {args.output}"
        )
        return 0
    if args.command == "plot-cross-dataset-transfer":
        output = plot_cross_dataset_transfer(
            analysis_path=args.analysis, output_path=args.output
        )
        print(f"cross-dataset transfer plot -> {output}")
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
    if args.command == "prepare-gsm8k-long-budget-resume":
        manifest = prepare_long_budget_resume(
            source_predictions_path=args.source_predictions,
            output_dir=args.output_dir,
            source_ceiling=args.source_ceiling,
            target_ceiling=args.target_ceiling,
        )
        print(
            f"GSM8K long-budget reuse={manifest['reused_count']} "
            f"regenerate={manifest['regenerate_count']} -> {args.output_dir}"
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
    if args.command == "prepare-gsm8k-magnitude-ladder":
        manifest = freeze_magnitude_ladder_gsm8k(
            source_path=args.source,
            official_eval_path=args.eval,
            output_dir=args.output_dir,
            expected_source_sha256=args.expected_source_sha256,
            factors=tuple(args.factor),
        )
        print(
            f"GSM8K magnitude ladder factors={manifest['factors']} "
            f"common={manifest['counts']['common_parents']} -> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-gsm8k-magnitude-failures":
        report = analyze_magnitude_failures(
            transformed_eval_path=args.eval,
            original_predictions_path=args.original_predictions,
            transformed_predictions_path=args.transformed_predictions,
            output_dir=args.output_dir,
        )
        print(
            f"GSM8K magnitude factor={report['factor']} "
            f"correct={report['counts']['correct']}/{report['counts']['total']} "
            f"literal-rescues={report['counts']['literal_copy_error']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-gsm8k-magnitude-curve":
        eval_paths = []
        for value in args.eval:
            if "=" not in value:
                raise ValueError("--eval must use FACTOR=PATH")
            factor, path = value.split("=", 1)
            eval_paths.append((int(factor), path))
        prediction_paths = []
        for value in args.prediction:
            if "=" not in value or ":" not in value.split("=", 1)[0]:
                raise ValueError("--prediction must use CONDITION:FACTOR=PATH")
            label_factor, path = value.split("=", 1)
            label, factor = label_factor.rsplit(":", 1)
            prediction_paths.append((label, int(factor), path))
        report = analyze_magnitude_curve(
            eval_paths=eval_paths,
            prediction_paths=prediction_paths,
            output_dir=args.output_dir,
        )
        print(
            f"GSM8K magnitude curve conditions="
            f"{len({row['condition'] for row in report['results']})} "
            f"parents={report['common_parent_count']} -> {args.output_dir}"
        )
        return 0
    if args.command == "project-gsm8k-magnitude-predictions":
        manifest = project_magnitude_predictions(
            source_eval_path=args.source_eval,
            target_eval_path=args.target_eval,
            source_predictions_path=args.source_predictions,
            output_dir=args.output_dir,
        )
        print(
            f"GSM8K projected magnitude predictions={manifest['count']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "plot-gsm8k-main-results":
        failures = []
        for value in args.failure_summary:
            if "=" not in value:
                raise ValueError("--failure-summary must use LABEL=PATH")
            failures.append(tuple(value.split("=", 1)))
        report = build_gsm8k_result_plots(
            contribution_path=args.contribution,
            large_eval_path=args.large_eval,
            small_original_predictions=args.small_original,
            small_large_predictions=args.small_large,
            large_original_predictions=args.large_original,
            large_large_predictions=args.large_large,
            failure_summaries=failures,
            output_dir=args.output_dir,
        )
        print(f"GSM8K result plots={len(report['outputs'])} -> {args.output_dir}")
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
    if args.command == "analyze-gsm8k-model-size":
        report = analyze_model_size_interaction(
            small_report_path=args.small_report,
            large_report_path=args.large_report,
            small_pair=args.small_pair,
            large_pair=args.large_pair,
            output_path=args.output,
        )
        print(
            "GSM8K model-size answer interaction="
            f"{report['original_answer_contribution']['model_size_interaction']:.3f} "
            "robustness interaction="
            f"{report['large_number_robustness_contribution']['model_size_interaction']:.3f} "
            f"-> {args.output}"
        )
        return 0
    if args.command == "prepare-operator-o1":
        manifest = freeze_o1_dataset(
            args.output_dir,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
            seed=args.seed,
        )
        print(
            f"GSM8K-OC O1 train={manifest['counts']['train']} "
            f"dev={manifest['counts']['dev']} test={manifest['counts']['test']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-operator-o1-pilot":
        manifest = freeze_o1_pilot(
            args.source_dir,
            args.output_dir,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
            seed=args.seed,
        )
        print(
            f"GSM8K-OC O1 pilot train={manifest['counts']['train']} "
            f"dev={manifest['counts']['dev']} test={manifest['counts']['test']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-operator-pilot":
        report = analyze_operator_pilot(
            eval_path=args.eval,
            direct_predictions_path=args.direct_predictions,
            ap_predictions_path=args.ap_predictions,
            as_predictions_path=args.as_predictions,
            output_dir=args.output_dir,
        )
        print(
            "operator pilot "
            f"direct={report['conditions']['direct']['correct']}/{report['identity_count']} "
            f"AP={report['conditions']['ap']['correct']}/{report['identity_count']} "
            f"AS={report['conditions']['as']['correct']}/{report['identity_count']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-operator-ladder":
        if any("=" not in value for value in args.summary):
            raise ValueError("operator summaries must use LEVEL=PATH")
        report = analyze_operator_ladder(
            [value.split("=", 1) for value in args.summary], args.output_dir
        )
        print(f"operator ladder levels={','.join(report['level_order'])} -> {args.output_dir}")
        return 0
    if args.command == "prepare-operator-level":
        manifest = freeze_operator_level(
            args.output_dir,
            level=args.level,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
            seed=args.seed,
        )
        print(
            f"GSM8K-OC {args.level} train={manifest['counts']['train']} "
            f"dev={manifest['counts']['dev']} test={manifest['counts']['test']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-operator-level-pilot":
        manifest = freeze_operator_pilot(
            args.source_dir,
            args.output_dir,
            level=args.level,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
            seed=args.seed,
        )
        print(
            f"GSM8K-OC {args.level} pilot train={manifest['counts']['train']} "
            f"dev={manifest['counts']['dev']} test={manifest['counts']['test']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-matched-interventions":
        keyword = {
            "source_corpus_path": args.source_corpus,
            "excluded_training_path": args.excluded_training,
            "output_dir": args.output_dir,
            "train_count": args.train_count,
            "dev_count": args.dev_count,
            "test_count": args.test_count,
            "seed": args.seed,
        }
        if args.factor:
            keyword["factors"] = tuple(args.factor)
        if args.jitter_seed:
            keyword["jitter_seeds"] = tuple(args.jitter_seed)
        manifest = freeze_gsm8k_matched_interventions(**keyword)
        print(
            "GSM8K matched interventions "
            f"train={manifest['counts']['train']} dev={manifest['counts']['dev']} "
            f"test={manifest['counts']['test']} eligible={manifest['eligibility']['eligible']} "
            f"-> {args.output_dir}"
        )
        return 0
    if args.command == "prepare-gsm8k-operator-jitter-matrix":
        keyword = {
            "panel_dir": args.panel_dir,
            "output_dir": args.output_dir,
        }
        if args.factor:
            keyword["factors"] = tuple(args.factor)
        if args.jitter_seed:
            keyword["jitter_seeds"] = tuple(args.jitter_seed)
        if args.operator_level:
            keyword["operator_levels"] = tuple(args.operator_level)
        manifest = freeze_gsm8k_operator_jitter_matrix(**keyword)
        print(
            "GSM8K operator-jitter matrix "
            f"seeds={len(manifest['jitter']['seeds'])} "
            f"cells-per-seed={len(manifest['execution_order'][0]['cells'])} "
            f"test={manifest['counts']['test']} -> {args.output_dir}"
        )
        return 0
    if args.command == "analyze-gsm8k-matched-interventions":
        cells = []
        for value in args.cell:
            if "=" not in value or len(value.split("=", 1)[1].split("|")) != 3:
                raise ValueError(
                    "matched cells must use LABEL=EVAL|DIRECT_PREDICTIONS|ASL_PREDICTIONS"
                )
            label, paths = value.split("=", 1)
            cells.append((label, *paths.split("|")))
        report = analyze_matched_interventions(cells=cells, output_dir=args.output_dir)
        print(f"matched Direct-vs-ASL cells={len(report['cells'])} -> {args.output_dir}")
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
