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

from .attention_diagnostic import (
    plot_attention_diagnostic,
    run_and_write_attention_diagnostic,
)
from .benchmark_next import NextBenchmarkConfig, generate_next_benchmark
from .composition import run_compositions
from .dataset import MixedBenchmarkConfig, MixedExample, iter_benchmark
from .diagnostic import (
    DiagnosticBenchmarkConfig,
    analyze_tokenizer_trigger_ladder,
    analyze_trigger_ladder,
    generate_diagnostic_benchmark,
    lexical_audit,
)
from .evaluate import evaluate
from .experiment import run_scripted
from .generic_tools import compare_oracle_transports
from .next_analysis import analyze_runs
from .next_experiment import run_model_condition, run_oracle_condition, summarize_next
from .plot import plot_scaling
from .public_benchmarks import analyze_public_coverage, freeze_public_suite
from .public_compute import (
    PUBLIC_COMPUTE_CONDITIONS,
    analyze_public_compute_runs,
    freeze_executable_public_slice,
    freeze_public_bm25_routes,
    materialize_executable_public_slice,
    run_public_compute_example,
    write_public_compute_run,
)
from .result_use import (
    ResultUseConfig,
    analyze_result_use,
    generate_result_use_benchmark,
    write_result_use_run,
)
from .router import analyze_router_runs, prepare_router_data, run_and_write_router
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
    print(f"trained {report['adapter_id']} on {report['train_rows']} rows -> {args.output_dir}")
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
    print(
        f"generated {result['record_count']} frozen TwIL comparison examples -> {args.output_dir}"
    )
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
    print(f"completed Paper 2 lexical audit; maximum accuracy={result['maximum_accuracy']:.4f}")
    return 0


def analyze_diagnostic_command(args: argparse.Namespace) -> int:
    result = analyze_trigger_ladder(args.train, args.test, args.output_dir)
    print(
        f"completed {len(result['trigger_ladder'])}-condition trigger ladder; "
        f"decision={result['decision']['status']}"
    )
    return 0


def analyze_tokenizer_triggers_command(args: argparse.Namespace) -> int:
    result = analyze_tokenizer_trigger_ladder(
        args.train,
        args.dev,
        args.test,
        args.tokenizer_config,
        args.neural_predictions,
        args.output_dir,
    )
    print(
        f"completed Paper 2 tokenizer trigger ladder; "
        f"decision={result['paper2_decision']['status']}"
    )
    return 0


def prepare_router_command(args: argparse.Namespace) -> int:
    result = prepare_router_data(args.source_dir, args.output_dir)
    print(f"prepared {result['counts']} Paper 2 router rows -> {args.output_dir}")
    return 0


def run_router_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    model = _model(raw, args.model)
    generation = raw.get("generation", {})
    backend = HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_new_tokens=int(generation.get("max_new_tokens", 6)),
            device=str(generation.get("device", "xpu")),
            dtype=str(model.get("dtype", generation.get("dtype", "float16"))),
            adapter_path=args.adapter_path,
            adapter_id=str(model["adapter_id"]) if args.adapter_path else None,
            cached_generation=True,
        )
    )
    summary = run_and_write_router(
        dataset_path=args.dataset,
        backend=backend,
        condition=args.condition,
        seed=int(raw.get("seed", 22631)),
        output_dir=args.output_dir,
        model=model,
        adapter_path=args.adapter_path,
    )
    print(
        f"completed Paper 2 router {args.condition}; "
        f"select={summary['engine_selection_accuracy']:.4f} "
        f"FAR={summary['false_activation_rate']:.4f}"
    )
    return 0


def analyze_router_command(args: argparse.Namespace) -> int:
    result = analyze_router_runs(args.config, args.output_dir)
    print(
        f"analyzed {len(result['rows'])} Paper 2 router conditions; "
        f"decision={result['decision']['status']}"
    )
    return 0


def generate_result_use_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    result = generate_result_use_benchmark(ResultUseConfig.from_dict(raw), args.output_dir)
    print(f"generated {result['counts']} Paper 2 result-use rows -> {args.output_dir}")
    return 0


def run_result_use_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    if args.condition == "runtime_copy":
        model = None
        backend = None
    else:
        model = _model(raw, args.model)
        generation = raw.get("generation", {})
        backend = HuggingFaceBackend(
            HuggingFaceGenerationConfig(
                model_id=str(model["model_id"]),
                revision=str(model["revision"]),
                max_new_tokens=int(generation.get("max_new_tokens", 24)),
                device=str(generation.get("device", "xpu")),
                dtype=str(model.get("dtype", generation.get("dtype", "float16"))),
                cached_generation=True,
            )
        )
    result = write_result_use_run(
        dataset_path=args.dataset,
        backend=backend,
        condition=args.condition,
        seed=int(raw.get("seed", 22701)),
        output_dir=args.output_dir,
        model=model,
    )
    print(
        f"completed Paper 2 result use {args.condition}; "
        f"exact={result['overall']['exact_rate']:.4f}"
    )
    return 0


def analyze_result_use_command(args: argparse.Namespace) -> int:
    result = analyze_result_use(args.config, args.output_dir)
    print(f"analyzed Paper 2 result use; attention={result['decision']['attention_diagnostics']}")
    return 0


def run_attention_diagnostic_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    model = _model(raw, args.model)
    attention = raw.get("attention", {})
    result = run_and_write_attention_diagnostic(
        dataset_path=args.dataset,
        baseline_predictions_path=args.baseline_predictions,
        model=model,
        device=str(attention.get("device", "xpu")),
        max_new_tokens=int(attention.get("max_new_tokens", 24)),
        per_task=int(attention.get("per_task", 12)),
        output_dir=args.output_dir,
    )
    print(
        "completed Paper 2 attention diagnostic; "
        f"fixed_bias_gate={result['decision']['fixed_bias_gate']}"
    )
    return 0


def plot_attention_diagnostic_command(args: argparse.Namespace) -> int:
    output = plot_attention_diagnostic(args.summary, args.output)
    print(f"plotted Paper 2 attention diagnostic -> {output}")
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


def freeze_public_command(args: argparse.Namespace) -> int:
    result = freeze_public_suite(args.config, args.cache_root, args.output_dir)
    print(
        f"froze {result['record_count']} public Paper 2 examples; "
        f"counts={result['counts']}"
    )
    return 0


def analyze_public_command(args: argparse.Namespace) -> int:
    result = analyze_public_coverage(
        args.config, args.cache_root, args.selection, args.output_dir
    )
    print(
        f"analyzed {result['record_count']} public Paper 2 examples; "
        f"status={result['interpretation']['status']}"
    )
    return 0


def compare_generic_tools_command(args: argparse.Namespace) -> int:
    result = compare_oracle_transports(args.benchmark, args.output_dir)
    print(
        f"compared {result['row_count']} Paper 2 generic-tool/block transports; "
        f"agreement={result['backend_result_agreement']:.3f}"
    )
    return 0


def freeze_public_executable_command(args: argparse.Namespace) -> int:
    result = freeze_executable_public_slice(
        args.config,
        args.cache_root,
        args.source_selection,
        args.output_dir,
        per_benchmark=args.per_benchmark,
    )
    print(
        f"froze {result['record_count']} executable public Paper 2 rows; "
        f"counts={result['selected_counts']}"
    )
    return 0


def freeze_public_bm25_command(args: argparse.Namespace) -> int:
    result = freeze_public_bm25_routes(
        args.config,
        args.cache_root,
        args.source_selection,
        args.executable_selection,
        args.output_dir,
        k=args.k,
    )
    print(
        f"froze {result['test_count']} public BM25 routes from "
        f"{result['train_count']} disjoint examples; accuracy={result['route_accuracy']:.3f}"
    )
    return 0


def run_public_compute_command(args: argparse.Namespace) -> int:
    config = read_json(args.model_config)
    if config.get("schema_version") != "ccpu.paper2.public_compute_config.v1":
        raise ValueError("unsupported Paper 2 public compute config schema")
    model = dict(config["model"])
    revision = str(model["revision"])
    invalid_revision = len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    )
    if invalid_revision:
        raise ValueError("public compute model revision must be a pinned SHA")
    if args.condition == "cpu_bm25" and not args.route_predictions:
        raise ValueError("cpu_bm25 requires --route-predictions")
    backend = HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=revision,
            max_new_tokens=int(model.get("max_new_tokens", 128)),
            device=str(args.device or model.get("device", "auto")),
            dtype=str(model.get("dtype", "auto")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            cached_generation=True,
        )
    )
    examples = materialize_executable_public_slice(
        args.public_config,
        args.cache_root,
        args.selection,
        args.route_predictions,
    )
    examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    output = Path(args.output_dir)
    prediction_path = output / "predictions.jsonl"
    rows = read_jsonl(prediction_path) if prediction_path.exists() and not args.no_resume else []
    if any(row["condition"] != args.condition for row in rows):
        raise ValueError("resume output contains a different public compute condition")
    completed = {str(row["example_id"]) for row in rows}
    pending = [row for row in examples if str(row["example_id"]) not in completed]
    checkpoint_every = max(1, int(args.checkpoint_every))
    seed = int(config["seed"])
    for index, example in enumerate(pending, 1):
        rows.append(
            run_public_compute_example(
                example,
                backend,
                condition=args.condition,
                seed=seed,
            )
        )
        if index % checkpoint_every == 0:
            write_public_compute_run(
                output,
                rows,
                model_config=args.model_config,
                selection_path=args.selection,
                condition=args.condition,
                route_predictions=args.route_predictions,
            )
            print(f"checkpoint {args.condition}: {len(rows)}/{len(examples)}")
    summary = write_public_compute_run(
        output,
        rows,
        model_config=args.model_config,
        selection_path=args.selection,
        condition=args.condition,
        route_predictions=args.route_predictions,
    )
    print(
        f"completed {args.condition} on {summary['base_question_count']} public questions "
        f"-> {output}"
    )
    return 0


def analyze_public_compute_command(args: argparse.Namespace) -> int:
    result = analyze_public_compute_runs(args.predictions, args.output_dir)
    print(
        f"analyzed {result['base_question_count']} matched public Paper 2 questions "
        f"-> {args.output_dir}"
    )
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

    tokenizer_analysis = commands.add_parser(
        "analyze-tokenizer-triggers",
        help="run matched token-space CPU routers and a hierarchical fallback",
    )
    tokenizer_analysis.add_argument("--train", required=True)
    tokenizer_analysis.add_argument("--dev", required=True)
    tokenizer_analysis.add_argument("--test", required=True)
    tokenizer_analysis.add_argument("--tokenizer-config", required=True)
    tokenizer_analysis.add_argument("--neural-predictions", required=True)
    tokenizer_analysis.add_argument("--output-dir", required=True)
    tokenizer_analysis.set_defaults(handler=analyze_tokenizer_triggers_command)

    router_data = commands.add_parser(
        "prepare-router", help="prepare six-way classification-only LoRA data"
    )
    router_data.add_argument("--source-dir", required=True)
    router_data.add_argument("--output-dir", required=True)
    router_data.set_defaults(handler=prepare_router_command)

    router_run = commands.add_parser(
        "run-router", help="evaluate a base or adapted six-way neural router"
    )
    router_run.add_argument("--config", required=True)
    router_run.add_argument("--model", required=True)
    router_run.add_argument("--dataset", required=True)
    router_run.add_argument("--condition", required=True)
    router_run.add_argument("--adapter-path")
    router_run.add_argument("--output-dir", required=True)
    router_run.set_defaults(handler=run_router_command)

    router_analysis = commands.add_parser(
        "analyze-router", help="compare CPU and neural six-way router conditions"
    )
    router_analysis.add_argument("--config", required=True)
    router_analysis.add_argument("--output-dir", required=True)
    router_analysis.set_defaults(handler=analyze_router_command)

    result_data = commands.add_parser(
        "generate-result-use", help="generate COPY/INTERPRET/CONTINUE result-use rows"
    )
    result_data.add_argument("--config", required=True)
    result_data.add_argument("--output-dir", required=True)
    result_data.set_defaults(handler=generate_result_use_command)

    result_run = commands.add_parser(
        "run-result-use", help="run neural or deterministic result-use conditions"
    )
    result_run.add_argument("--config", required=True)
    result_run.add_argument("--dataset", required=True)
    result_run.add_argument("--condition", required=True, choices=("qwen_base", "runtime_copy"))
    result_run.add_argument("--model")
    result_run.add_argument("--output-dir", required=True)
    result_run.set_defaults(handler=run_result_use_command)

    result_analysis = commands.add_parser(
        "analyze-result-use", help="compare result-use formats and gate SA experiments"
    )
    result_analysis.add_argument("--config", required=True)
    result_analysis.add_argument("--output-dir", required=True)
    result_analysis.set_defaults(handler=analyze_result_use_command)

    attention_run = commands.add_parser(
        "run-attention-diagnostic",
        help="measure result/question attention and causal content ablations",
    )
    attention_run.add_argument("--config", required=True)
    attention_run.add_argument("--model", required=True)
    attention_run.add_argument("--dataset", required=True)
    attention_run.add_argument("--baseline-predictions", required=True)
    attention_run.add_argument("--output-dir", required=True)
    attention_run.set_defaults(handler=run_attention_diagnostic_command)

    attention_plot = commands.add_parser(
        "plot-attention-diagnostic", help="plot and finalize an attention diagnostic"
    )
    attention_plot.add_argument("--summary", required=True)
    attention_plot.add_argument("--output", required=True)
    attention_plot.set_defaults(handler=plot_attention_diagnostic_command)

    public_freeze = commands.add_parser(
        "freeze-public", help="verify and freeze the public heterogeneous benchmark suite"
    )
    public_freeze.add_argument("--config", required=True)
    public_freeze.add_argument("--cache-root", required=True)
    public_freeze.add_argument("--output-dir", required=True)
    public_freeze.set_defaults(handler=freeze_public_command)

    public_analysis = commands.add_parser(
        "analyze-public", help="factor public routing, formalization, and backend coverage"
    )
    public_analysis.add_argument("--config", required=True)
    public_analysis.add_argument("--cache-root", required=True)
    public_analysis.add_argument("--selection", required=True)
    public_analysis.add_argument("--output-dir", required=True)
    public_analysis.set_defaults(handler=analyze_public_command)

    generic_tools = commands.add_parser(
        "compare-generic-tools", help="audit four-tool and CogCop transport equivalence"
    )
    generic_tools.add_argument("--benchmark", required=True)
    generic_tools.add_argument("--output-dir", required=True)
    generic_tools.set_defaults(handler=compare_generic_tools_command)

    public_executable = commands.add_parser(
        "freeze-public-executable",
        help="freeze a validated executable slice of all five public benchmarks",
    )
    public_executable.add_argument("--config", required=True)
    public_executable.add_argument("--cache-root", required=True)
    public_executable.add_argument("--source-selection", required=True)
    public_executable.add_argument("--per-benchmark", type=int, default=12)
    public_executable.add_argument("--output-dir", required=True)
    public_executable.set_defaults(handler=freeze_public_executable_command)

    public_bm25 = commands.add_parser(
        "freeze-public-bm25",
        help="fit and freeze disjoint shared-token BM25 public routes",
    )
    public_bm25.add_argument("--config", required=True)
    public_bm25.add_argument("--cache-root", required=True)
    public_bm25.add_argument("--source-selection", required=True)
    public_bm25.add_argument("--executable-selection", required=True)
    public_bm25.add_argument("--k", type=int, default=3)
    public_bm25.add_argument("--output-dir", required=True)
    public_bm25.set_defaults(handler=freeze_public_bm25_command)

    public_run = commands.add_parser(
        "run-public-compute",
        help="run one matched model-facing public compute condition",
    )
    public_run.add_argument("--public-config", required=True)
    public_run.add_argument("--cache-root", required=True)
    public_run.add_argument("--selection", required=True)
    public_run.add_argument("--model-config", required=True)
    public_run.add_argument("--route-predictions")
    public_run.add_argument("--condition", required=True, choices=PUBLIC_COMPUTE_CONDITIONS)
    public_run.add_argument("--device")
    public_run.add_argument("--checkpoint-every", type=int, default=10)
    public_run.add_argument("--no-resume", action="store_true")
    public_run.add_argument("--offset", type=int, default=0)
    public_run.add_argument("--limit", type=int)
    public_run.add_argument("--output-dir", required=True)
    public_run.set_defaults(handler=run_public_compute_command)

    public_compute_analysis = commands.add_parser(
        "analyze-public-compute",
        help="merge matched public compute conditions and render results",
    )
    public_compute_analysis.add_argument("--predictions", nargs="+", required=True)
    public_compute_analysis.add_argument("--output-dir", required=True)
    public_compute_analysis.set_defaults(handler=analyze_public_compute_command)
