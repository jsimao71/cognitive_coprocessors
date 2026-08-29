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
from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig
from ccpu.paper1.lora_train import LoRATrainingConfig, train_lora

from .benchmark_next import NextBenchmarkConfig, generate_next_benchmark
from .composition import run_compositions
from .dataset import MixedBenchmarkConfig, MixedExample, iter_benchmark
from .diagnostic import (
    DiagnosticBenchmarkConfig,
    analyze_trigger_ladder,
    generate_diagnostic_benchmark,
    lexical_audit,
)
from .evaluate import evaluate
from .experiment import run_scripted
from .next_analysis import analyze_runs
from .next_experiment import run_model_condition, run_oracle_condition, summarize_next
from .plot import plot_scaling
from .twil_analysis import analyze_twil_runs
from .twil_benchmark import TwILBenchmarkConfig, generate_twil_benchmark
from .twil_experiment import (
    load_twil_dataset,
    rescore_twil_predictions,
    run_reuse_workload,
    run_twil_condition,
    summarize_twil,
)


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


def generate_next_command(args: argparse.Namespace) -> int:
    result = generate_next_benchmark(
        NextBenchmarkConfig.from_dict(read_json(args.config)), args.output_dir
    )
    print(f"generated Paper 2 five-engine benchmark {result['counts']} -> {args.output_dir}")
    return 0


def _model(raw: dict, label: str) -> dict:
    for model in raw["models"]:
        if label in {model.get("label"), model.get("model_id")}:
            return model
    raise ValueError(f"unknown configured model: {label}")


def train_next_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    model = _model(raw, args.model)
    report = train_lora(
        model=model,
        training=LoRATrainingConfig.from_dict(
            {"training": {**raw["training"], "dtype": model.get("dtype", "float16")}}
        ),
        train_path=args.train,
        dev_path=args.dev,
        output_dir=args.output_dir,
    )
    print(
        f"trained {report['adapter_id']} on {report['train_rows']} rows -> {args.output_dir}"
    )
    return 0


def run_next_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    if args.condition in {"runtime", "oracle"}:
        rows = run_oracle_condition(
            args.dataset, condition=args.condition, catalog_size=args.catalog_size
        )
        model = None
    else:
        model = _model(raw, args.model)
        generation = raw.get("generation", {})
        backend = HuggingFaceBackend(
            HuggingFaceGenerationConfig(
                model_id=str(model["model_id"]),
                revision=str(model["revision"]),
                max_new_tokens=int(generation.get("max_new_tokens", 72)),
                device=str(generation.get("device", "xpu")),
                dtype=str(model.get("dtype", generation.get("dtype", "float16"))),
                adapter_path=args.adapter_path,
                adapter_id=str(model["adapter_id"]) if args.adapter_path else None,
            )
        )
        rows = run_model_condition(
            dataset_path=args.dataset,
            backend=backend,
            condition=args.condition,
            catalog_size=args.catalog_size,
            seed=int(raw.get("seed", 22051)),
            assess_use=not args.skip_use,
        )
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = write_json(output / "summary.json", summarize_next(rows))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2",
            "schema_version": "ccpu.paper2.next_run_manifest.v1",
            "condition": args.condition,
            "catalog_size": args.catalog_size,
            "model": model,
            "dataset_sha256": file_sha256(args.dataset),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary),
            "adapter_path": args.adapter_path,
            "environment": environment_manifest(root),
        },
    )
    print(f"completed {len(rows)} Paper 2 next-iteration rows -> {output}")
    return 0


def evaluate_next_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.predictions)
    output = write_json(args.output, summarize_next(rows))
    manifest_path = Path(output).parent / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest["summary_sha256"] = file_sha256(output)
        write_json(manifest_path, manifest)
    print(f"evaluated {len(rows)} Paper 2 next-iteration rows -> {args.output}")
    return 0


def compositions_command(args: argparse.Namespace) -> int:
    rows, summary = run_compositions(args.count_per_family)
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary_path = write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2",
            "schema_version": "ccpu.paper2.composition_manifest.v1",
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary_path),
        },
    )
    print(f"completed {len(rows)} bounded Paper 2 compositions -> {output}")
    return 0


def analyze_next_command(args: argparse.Namespace) -> int:
    result = analyze_runs(args.config, args.output_dir)
    print(
        f"merged {len(result['rows'])} Paper 2 scaling cells; "
        f"Paper 3 gate={result['paper3_gate']['status']}"
    )
    return 0


def generate_twil_command(args: argparse.Namespace) -> int:
    result = generate_twil_benchmark(
        TwILBenchmarkConfig.from_dict(read_json(args.config)), args.output_dir
    )
    print(f"generated {result['record_count']} frozen TwIL comparison examples -> {args.output_dir}")
    return 0


def run_twil_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    model = None
    backend = None
    if args.condition != "oracle":
        model = _model(raw, args.model)
        generation = raw.get("generation", {})
        backend = HuggingFaceBackend(
            HuggingFaceGenerationConfig(
                model_id=str(model["model_id"]),
                revision=str(model["revision"]),
                max_new_tokens=int(generation.get("max_new_tokens", 160)),
                device=str(generation.get("device", "xpu")),
                dtype=str(model.get("dtype", generation.get("dtype", "bfloat16"))),
                use_chat_template=bool(generation.get("use_chat_template", True)),
                enable_thinking=bool(generation.get("enable_thinking", False)),
                cached_generation=True,
            )
        )
    rows = run_twil_condition(
        load_twil_dataset(args.dataset),
        backend=backend,
        condition=args.condition,
        seed=int(raw.get("seed", 22501)),
    )
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = write_json(output / "summary.json", summarize_twil(rows))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2",
            "schema_version": "ccpu.paper2.twil_run_manifest.v1",
            "condition": args.condition,
            "model": model,
            "dataset_sha256": file_sha256(args.dataset),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary),
            "generation": raw.get("generation", {}),
            "environment": environment_manifest(root),
        },
    )
    print(f"completed {len(rows)} TwIL comparison rows -> {output}")
    return 0


def reuse_twil_command(args: argparse.Namespace) -> int:
    rows = run_reuse_workload(tuple(args.query_counts))
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = write_json(
        output / "summary.json",
        {
            "schema_version": "ccpu.paper2.twil_reuse_summary.v1",
            "rows": rows,
        },
    )
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2",
            "schema_version": "ccpu.paper2.twil_reuse_manifest.v1",
            "query_counts": args.query_counts,
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary),
        },
    )
    print(f"completed {len(rows)} persistent-state TwIL workload cells -> {output}")
    return 0


def analyze_twil_command(args: argparse.Namespace) -> int:
    result = analyze_twil_runs(args.config, args.output_dir)
    print(f"merged {result['prediction_count']} TwIL comparison rows -> {args.output_dir}")
    return 0


def generate_diagnostic_command(args: argparse.Namespace) -> int:
    result = generate_diagnostic_benchmark(
        DiagnosticBenchmarkConfig.from_dict(read_json(args.config)), args.output_dir
    )
    print(f"generated Paper 2 diagnostic benchmark {result['config']} -> {args.output_dir}")
    return 0


def audit_diagnostic_command(args: argparse.Namespace) -> int:
    result = lexical_audit(args.train, args.test)
    write_json(args.output, result)
    print(
        f"completed Paper 2 lexical audit; maximum accuracy={result['maximum_accuracy']:.4f}"
    )
    return 0


def analyze_diagnostic_command(args: argparse.Namespace) -> int:
    result = analyze_trigger_ladder(args.train, args.test, args.output_dir)
    print(
        f"completed {len(result['trigger_ladder'])}-condition trigger ladder; "
        f"decision={result['decision']['status']}"
    )
    return 0


def evaluate_twil_command(args: argparse.Namespace) -> int:
    source_rows = read_jsonl(args.predictions)
    rows = rescore_twil_predictions(source_rows, max_new_tokens=args.max_new_tokens)
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = write_json(output / "summary.json", summarize_twil(rows))
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 2",
            "schema_version": "ccpu.paper2.twil_rescore_manifest.v2",
            "source_predictions": str(Path(args.predictions).resolve()),
            "source_predictions_sha256": file_sha256(args.predictions),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary),
            "max_new_tokens": args.max_new_tokens,
            "credit_rule": "exact formalization AND exact engine execution",
        },
    )
    print(f"strictly rescored {len(rows)} TwIL comparison rows -> {output}")
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

    next_data = commands.add_parser(
        "generate-next", help="generate leakage-audited five-engine adapter data"
    )
    next_data.add_argument("--config", required=True)
    next_data.add_argument("--output-dir", required=True)
    next_data.set_defaults(handler=generate_next_command)

    next_train = commands.add_parser("train-next-lora", help="train a multi-engine adapter")
    next_train.add_argument("--config", required=True)
    next_train.add_argument("--model", required=True)
    next_train.add_argument("--train", required=True)
    next_train.add_argument("--dev", required=True)
    next_train.add_argument("--output-dir", required=True)
    next_train.set_defaults(handler=train_next_command)

    next_run = commands.add_parser(
        "run-next", help="run a model or deterministic capability-count condition"
    )
    next_run.add_argument("--config", required=True)
    next_run.add_argument("--dataset", required=True)
    next_run.add_argument(
        "--condition",
        required=True,
        choices=("no_engine", "context", "weights", "explicit_tools", "runtime", "oracle"),
    )
    next_run.add_argument("--catalog-size", required=True, type=int, choices=(1, 2, 3, 5))
    next_run.add_argument("--model")
    next_run.add_argument("--adapter-path")
    next_run.add_argument("--skip-use", action="store_true")
    next_run.add_argument("--output-dir", required=True)
    next_run.set_defaults(handler=run_next_command)

    next_evaluate = commands.add_parser(
        "evaluate-next", help="recompute factorized multi-engine metrics"
    )
    next_evaluate.add_argument("--predictions", required=True)
    next_evaluate.add_argument("--output", required=True)
    next_evaluate.set_defaults(handler=evaluate_next_command)

    compositions = commands.add_parser(
        "compositions", help="run bounded date-calculator and graph-Datalog compositions"
    )
    compositions.add_argument("--count-per-family", type=int, default=20)
    compositions.add_argument("--output-dir", required=True)
    compositions.set_defaults(handler=compositions_command)

    next_analysis = commands.add_parser(
        "analyze-next", help="merge capability runs and decide the Paper 3 gate"
    )
    next_analysis.add_argument("--config", required=True)
    next_analysis.add_argument("--output-dir", required=True)
    next_analysis.set_defaults(handler=analyze_next_command)

    twil_data = commands.add_parser(
        "generate-twil", help="generate the frozen TwIL/SmolLM3 comparison benchmark"
    )
    twil_data.add_argument("--config", required=True)
    twil_data.add_argument("--output-dir", required=True)
    twil_data.set_defaults(handler=generate_twil_command)

    twil_run = commands.add_parser(
        "run-twil", help="run neural, hybrid, or oracle TwIL comparison conditions"
    )
    twil_run.add_argument("--config", required=True)
    twil_run.add_argument("--dataset", required=True)
    twil_run.add_argument("--condition", required=True, choices=("neural", "hybrid", "oracle"))
    twil_run.add_argument("--model")
    twil_run.add_argument("--output-dir", required=True)
    twil_run.set_defaults(handler=run_twil_command)

    twil_reuse = commands.add_parser(
        "reuse-twil", help="measure persistent exact-state amortization"
    )
    twil_reuse.add_argument("--query-counts", nargs="+", type=int, default=[1, 5, 20, 100])
    twil_reuse.add_argument("--output-dir", required=True)
    twil_reuse.set_defaults(handler=reuse_twil_command)

    twil_analysis = commands.add_parser(
        "analyze-twil", help="merge TwIL comparison runs and render cost/frontier plots"
    )
    twil_analysis.add_argument("--config", required=True)
    twil_analysis.add_argument("--output-dir", required=True)
    twil_analysis.set_defaults(handler=analyze_twil_command)

    twil_evaluate = commands.add_parser(
        "evaluate-twil", help="strictly rescore preserved TwIL generations"
    )
    twil_evaluate.add_argument("--predictions", required=True)
    twil_evaluate.add_argument("--max-new-tokens", type=int, default=160)
    twil_evaluate.add_argument("--output-dir", required=True)
    twil_evaluate.set_defaults(handler=evaluate_twil_command)

    diagnostic_data = commands.add_parser(
        "generate-diagnostic", help="generate the richer six-way diagnostic benchmark"
    )
    diagnostic_data.add_argument("--config", required=True)
    diagnostic_data.add_argument("--output-dir", required=True)
    diagnostic_data.set_defaults(handler=generate_diagnostic_command)

    diagnostic_audit = commands.add_parser(
        "audit-diagnostic", help="run shallow six-way lexical baselines"
    )
    diagnostic_audit.add_argument("--train", required=True)
    diagnostic_audit.add_argument("--test", required=True)
    diagnostic_audit.add_argument("--output", required=True)
    diagnostic_audit.set_defaults(handler=audit_diagnostic_command)

    diagnostic_analysis = commands.add_parser(
        "analyze-diagnostic", help="run the CPU trigger and parser factorization ladder"
    )
    diagnostic_analysis.add_argument("--train", required=True)
    diagnostic_analysis.add_argument("--test", required=True)
    diagnostic_analysis.add_argument("--output-dir", required=True)
    diagnostic_analysis.set_defaults(handler=analyze_diagnostic_command)
