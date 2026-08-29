"""Paper 1.5 command-line workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    read_json,
    write_json,
    write_jsonl,
)
from ccpu.paper1.generation import HuggingFaceBackend, HuggingFaceGenerationConfig
from ccpu.paper1.lora_train import LoRATrainingConfig, train_lora

from .benchmark_next import (
    NextBenchmarkConfig,
    build_next_candidates,
    select_measured_quadrants,
)
from .dataset import load_benchmark
from .evaluate import evaluate
from .experiment import base_prompt, run_huggingface
from .generation import ConfidenceBackend
from .natural_analysis import build_natural_analysis
from .natural_robustness import (
    NaturalRobustnessConfig,
    generate_natural_benchmark,
    lexical_audit,
    run_longform_opportunities,
    run_natural_model,
    summarize_natural,
    tokenizer_trigger_comparison,
)
from .next_analysis import build_next_analysis
from .plot import plot_pareto
from .policy_analysis import build_policy_placement
from .policy_lora import (
    PolicyDataConfig,
    generate_policy_data,
    run_policy_hf,
    summarize_policy,
)
from .public_benchmarks import analyze_crag_triggers, freeze_crag_subset
from .source import ControlledFactStore
from .triggers import fit_confidence_threshold


def _raw_config(path: str | Path) -> dict:
    path = Path(path)
    raw = dict(read_json(path))
    benchmark_path = raw.get("benchmark_path")
    if benchmark_path:
        benchmark_file = (path.parent / str(benchmark_path)).resolve()
        benchmark = read_json(benchmark_file)
        raw["source"] = benchmark["source"]
        raw["examples"] = benchmark["examples"]
        raw["benchmark_file"] = str(benchmark_file)
    return raw


def validate_command(args: argparse.Namespace) -> int:
    store, examples = load_benchmark(_raw_config(args.config))
    print(f"valid Paper 1.5 source {store.source_id}@{store.version} with {len(examples)} examples")
    return 0


def run_command(args: argparse.Namespace) -> int:
    raw = _raw_config(args.config)
    store, examples = load_benchmark(raw)
    model = dict(raw["model"])
    revision = str(model["revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Paper 1.5 model revision must be a pinned 40-character SHA")
    if args.device:
        model["device"] = args.device
    backend = ConfidenceBackend(model)
    predictions, traces, threshold = run_huggingface(
        examples,
        store,
        backend,
        seed=int(raw.get("seed", 17)),
    )
    output = Path(args.output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", predictions)
    traces_path = write_jsonl(output / "traces.jsonl", traces)
    summary = evaluate(predictions)
    summary["empirical"] = True
    summary_path = write_json(output / "summary.json", summary)
    benchmark_path = write_jsonl(output / "benchmark.jsonl", (example.to_dict() for example in examples))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 1.5",
            "schema_version": "ccpu.paper1_5.run_manifest.v1",
            "empirical": True,
            "source_id": store.source_id,
            "source_version": store.version,
            "config_sha256": file_sha256(args.config),
            "benchmark_sha256": file_sha256(benchmark_path),
            "predictions_sha256": file_sha256(predictions_path),
            "traces_sha256": file_sha256(traces_path),
            "summary_sha256": file_sha256(summary_path),
            "prediction_count": len(predictions),
            "trace_count": len(traces),
            "fitted_confidence_threshold": threshold,
            "model": model,
            "environment": environment_manifest(root),
        },
    )
    print(f"completed {len(predictions)} Paper 1.5 generations -> {output}")
    return 0


def freeze_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    config = NextBenchmarkConfig.from_dict(raw)
    source, candidates, candidate_audit = build_next_candidates(config)
    model = dict(raw["model"])
    backend = ConfidenceBackend(model)
    seed = int(raw.get("seed", 17))
    spans = {
        example.example_id: backend.complete(base_prompt(example), seed=seed)
        for example in candidates
    }
    development = [
        (spans[example.example_id].token_probabilities, example.evidence_required)
        for example in candidates
        if example.split == "dev"
    ]
    threshold = fit_confidence_threshold(development)
    probabilities = {
        example.example_id: min(spans[example.example_id].token_probabilities, default=1.0)
        for example in candidates
    }
    output = Path(args.output_dir)
    probe_path = write_jsonl(
        output / "confidence_probe.jsonl",
        (
            {
                "example_id": example.example_id,
                "split": example.split,
                "evidence_required": example.evidence_required,
                "design_group": example.design_group,
                "minimum_token_probability": probabilities[example.example_id],
                "forecast": spans[example.example_id].text,
                "token_probabilities": spans[example.example_id].token_probabilities,
            }
            for example in candidates
        ),
    )
    candidate_path = write_json(output / "candidate_audit.json", candidate_audit)
    selected, freeze_audit = select_measured_quadrants(
        candidates,
        probabilities,
        threshold=threshold,
        target_per_quadrant=config.target_per_quadrant,
    )
    benchmark_path = write_json(
        output / "benchmark_config.json",
        {
            "schema_version": "ccpu.paper1_5.frozen_benchmark.v2",
            "source": source,
            "examples": [example.to_dict() for example in selected],
            "freeze_model": model,
            "confidence_threshold": threshold,
        },
    )
    freeze_path = write_json(output / "freeze_audit.json", freeze_audit)
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 1.5",
            "schema_version": "ccpu.paper1_5.freeze_manifest.v1",
            "config_sha256": file_sha256(args.config),
            "benchmark_sha256": file_sha256(benchmark_path),
            "probe_sha256": file_sha256(probe_path),
            "candidate_audit_sha256": file_sha256(candidate_path),
            "freeze_audit_sha256": file_sha256(freeze_path),
            "source_id": ControlledFactStore.from_dict(source).source_id,
            "selected_examples": len(selected),
            "environment": environment_manifest(root),
        },
    )
    print(f"froze {len(selected)} examples across four measured quadrants -> {output}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    from ccpu.common.artifacts import read_jsonl

    summary = evaluate(read_jsonl(args.predictions))
    write_json(args.output, summary)
    print(f"evaluated Paper 1.5 predictions -> {args.output}")
    return 0


def plot_command(args: argparse.Namespace) -> int:
    output = plot_pareto(read_json(args.summary), args.output)
    print(f"wrote Paper 1.5 Pareto plot -> {output}")
    return 0


def generate_policy_data_command(args: argparse.Namespace) -> int:
    raw = _raw_config(args.config)
    _, examples = load_benchmark(raw)
    result = generate_policy_data(
        PolicyDataConfig.from_dict(raw),
        excluded_examples=examples,
        output_dir=args.output_dir,
    )
    print(f"generated {result['train_rows']} train and {result['dev_rows']} dev policy rows")
    return 0


def train_policy_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    matches = [entry for entry in raw["models"] if entry["model_id"] == args.model]
    if len(matches) != 1:
        raise ValueError(f"expected one configured policy model: {args.model}")
    result = train_lora(
        model=dict(matches[0]),
        training=LoRATrainingConfig.from_dict(raw),
        train_path=args.train,
        dev_path=args.dev,
        output_dir=args.output_dir,
    )
    print(f"trained {result['adapter_id']} with {result['trainable_parameters']} parameters")
    return 0


def run_policy_command(args: argparse.Namespace) -> int:
    raw = _raw_config(args.config)
    _, examples = load_benchmark(raw)
    if "model" in raw:
        model = dict(raw["model"])
    else:
        matches = [entry for entry in raw["models"] if entry["model_id"] == args.model]
        if len(matches) != 1:
            raise ValueError(f"expected one configured policy model: {args.model}")
        model = dict(matches[0])
    if args.adapter_path:
        model["adapter_path"] = args.adapter_path
    backend = HuggingFaceBackend(
        HuggingFaceGenerationConfig(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            max_new_tokens=int(model.get("max_new_tokens", 96)),
            device=str(model.get("device", "auto")),
            dtype=str(model.get("dtype", "auto")),
            use_chat_template=bool(model.get("use_chat_template", True)),
            enable_thinking=bool(model.get("enable_thinking", False)),
            adapter_path=model.get("adapter_path"),
            adapter_id=model.get("adapter_id") if model.get("adapter_path") else None,
        )
    )
    rows = run_policy_hf(
        examples,
        backend,
        mode=str(args.mode or raw.get("policy_mode", "minimal")),
        seed=int(raw.get("seed", 17)),
    )
    output = Path(args.output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", rows)
    summary_path = write_json(output / "summary.json", summarize_policy(rows))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 1.5",
            "schema_version": "ccpu.paper1_5.policy_run_manifest.v1",
            "config_sha256": file_sha256(args.config),
            "predictions_sha256": file_sha256(predictions_path),
            "summary_sha256": file_sha256(summary_path),
            "model": model,
            "policy_mode": args.mode or raw.get("policy_mode", "minimal"),
            "environment": environment_manifest(root),
        },
    )
    print(f"completed {len(rows)} retrieval-policy generations -> {output}")
    return 0


def analyze_policy_command(args: argparse.Namespace) -> int:
    result = build_policy_placement(args.config, args.output_dir)
    print(f"wrote {len(result['rows'])} retrieval-policy placement rows -> {args.output_dir}")
    return 0


def analyze_next_command(args: argparse.Namespace) -> int:
    result = build_next_analysis(args.config, args.output_dir)
    print(
        f"wrote {len(result['rows'])} Phase A rows; "
        f"Paper 2.5 gate={result['paper2_5_gate']['status']}"
    )
    return 0


def freeze_natural_command(args: argparse.Namespace) -> int:
    result = generate_natural_benchmark(
        NaturalRobustnessConfig.from_dict(read_json(args.config)), args.output_dir
    )
    print(f"froze {result['example_count']} natural-language examples -> {args.output_dir}")
    return 0


def audit_natural_command(args: argparse.Namespace) -> int:
    result = lexical_audit(args.dataset)
    write_json(args.output, result)
    print(f"natural lexical audit status={result['status']} -> {args.output}")
    return 0


def compare_natural_tokenizers_command(args: argparse.Namespace) -> int:
    result = tokenizer_trigger_comparison(
        args.benchmark,
        args.tokenizer_config,
        args.output_dir,
        train_path=args.train,
        dev_path=args.dev,
    )
    print(
        f"Paper 1.5 tokenizer comparison selected={result['selected_condition']}; "
        f"decision={result['paper1_5_decision']['status']}"
    )
    return 0


def run_natural_command(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    matches = [
        model
        for model in raw["models"]
        if args.model in {model.get("label"), model.get("model_id")}
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one configured natural-robustness model: {args.model}")
    model = dict(matches[0])
    backend = ConfidenceBackend(model)
    rows = run_natural_model(
        args.dataset,
        args.source,
        backend,
        seed=int(raw.get("seed", 15531)),
    )
    output = Path(args.output_dir)
    predictions = write_jsonl(output / "predictions.jsonl", rows)
    summary = write_json(output / "summary.json", summarize_natural(rows))
    root = Path(__file__).resolve().parents[3]
    write_json(
        output / "manifest.json",
        {
            "paper": "Paper 1.5",
            "schema_version": "ccpu.paper1_5.natural_run_manifest.v1",
            "model": model,
            "dataset_sha256": file_sha256(args.dataset),
            "source_sha256": file_sha256(args.source),
            "predictions_sha256": file_sha256(predictions),
            "summary_sha256": file_sha256(summary),
            "environment": environment_manifest(root),
        },
    )
    print(f"completed {len(rows)} natural robustness condition rows -> {output}")
    return 0


def longform_natural_command(args: argparse.Namespace) -> int:
    result = run_longform_opportunities(args.dataset)
    write_json(args.output, result)
    print(
        f"scored {result['opportunity_count']} long-form factual opportunities -> {args.output}"
    )
    return 0


def analyze_natural_command(args: argparse.Namespace) -> int:
    result = build_natural_analysis(args.config, args.output_dir)
    print(
        f"wrote {len(result['rows'])} natural robustness analysis rows; "
        f"learned policy={result['learned_policy_decision']['status']}"
    )
    return 0


def freeze_public_crag_command(args: argparse.Namespace) -> int:
    result = freeze_crag_subset(args.config, args.cache_root, args.output_dir)
    print(
        f"froze {result['record_count']} CRAG questions and "
        f"{result['matched_evaluation_rows']} matched rows"
    )
    return 0


def analyze_public_crag_command(args: argparse.Namespace) -> int:
    result = analyze_crag_triggers(
        args.config, args.cache_root, args.selection, args.output_dir
    )
    print(
        f"analyzed {result['matched_row_count']} matched CRAG rows; "
        f"status={result['interpretation']['status']}"
    )
    return 0


def add_commands(papers: argparse._SubParsersAction) -> None:
    paper = papers.add_parser("paper1.5", aliases=["paper1_5"], help="epistemic-risk retrieval")
    commands = paper.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate source and benchmark")
    validate.add_argument("--config", required=True)
    validate.set_defaults(handler=validate_command)

    freeze = commands.add_parser(
        "freeze-next", help="measure and freeze the expanded four-quadrant benchmark"
    )
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.set_defaults(handler=freeze_command)

    run = commands.add_parser("run-hf", help="run the pinned checkpoint experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device")
    run.set_defaults(handler=run_command)

    evaluation = commands.add_parser("evaluate", help="recompute summary metrics")
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(handler=evaluate_command)

    plot = commands.add_parser("plot", help="plot reliability versus retrieval cost")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output", required=True)
    plot.set_defaults(handler=plot_command)

    policy_data = commands.add_parser(
        "generate-policy-data", help="generate leakage-audited one-source policy data"
    )
    policy_data.add_argument("--config", required=True)
    policy_data.add_argument("--output-dir", required=True)
    policy_data.set_defaults(handler=generate_policy_data_command)

    policy_train = commands.add_parser(
        "train-policy-lora", help="train a protocol-only retrieval-policy adapter"
    )
    policy_train.add_argument("--config", required=True)
    policy_train.add_argument("--model", required=True)
    policy_train.add_argument("--train", required=True)
    policy_train.add_argument("--dev", required=True)
    policy_train.add_argument("--output-dir", required=True)
    policy_train.set_defaults(handler=train_policy_command)

    policy_run = commands.add_parser(
        "run-policy-hf", help="evaluate context or adapter retrieval-policy placement"
    )
    policy_run.add_argument("--config", required=True)
    policy_run.add_argument("--output-dir", required=True)
    policy_run.add_argument("--model")
    policy_run.add_argument("--mode", choices=("context", "minimal"))
    policy_run.add_argument("--adapter-path")
    policy_run.set_defaults(handler=run_policy_command)

    policy_analysis = commands.add_parser(
        "analyze-policy", help="compare retrieval policy in context, weights, and runtime"
    )
    policy_analysis.add_argument("--config", required=True)
    policy_analysis.add_argument("--output-dir", required=True)
    policy_analysis.set_defaults(handler=analyze_policy_command)

    next_analysis = commands.add_parser(
        "analyze-next", help="compare Phase A models and decide the Paper 2.5 gate"
    )
    next_analysis.add_argument("--config", required=True)
    next_analysis.add_argument("--output-dir", required=True)
    next_analysis.set_defaults(handler=analyze_next_command)

    natural_freeze = commands.add_parser(
        "freeze-natural", help="freeze the natural-language epistemic benchmark"
    )
    natural_freeze.add_argument("--config", required=True)
    natural_freeze.add_argument("--output-dir", required=True)
    natural_freeze.set_defaults(handler=freeze_natural_command)

    natural_audit = commands.add_parser(
        "audit-natural", help="run shallow lexical-triviality baselines"
    )
    natural_audit.add_argument("--dataset", required=True)
    natural_audit.add_argument("--output", required=True)
    natural_audit.set_defaults(handler=audit_natural_command)

    natural_tokenizers = commands.add_parser(
        "compare-natural-tokenizers",
        help="compare matched lexical token spaces with the semantic runtime policy",
    )
    natural_tokenizers.add_argument("--benchmark", required=True)
    natural_tokenizers.add_argument("--tokenizer-config", required=True)
    natural_tokenizers.add_argument("--train")
    natural_tokenizers.add_argument("--dev")
    natural_tokenizers.add_argument("--output-dir", required=True)
    natural_tokenizers.set_defaults(handler=compare_natural_tokenizers_command)

    natural_run = commands.add_parser(
        "run-natural", help="run one checkpoint on the natural-language benchmark"
    )
    natural_run.add_argument("--config", required=True)
    natural_run.add_argument("--dataset", required=True)
    natural_run.add_argument("--source", required=True)
    natural_run.add_argument("--model", required=True)
    natural_run.add_argument("--output-dir", required=True)
    natural_run.set_defaults(handler=run_natural_command)

    natural_longform = commands.add_parser(
        "longform-natural", help="score multi-opportunity long-form runtime behavior"
    )
    natural_longform.add_argument("--dataset", required=True)
    natural_longform.add_argument("--output", required=True)
    natural_longform.set_defaults(handler=longform_natural_command)

    natural_analysis = commands.add_parser(
        "analyze-natural", help="combine natural-language cross-family results and plots"
    )
    natural_analysis.add_argument("--config", required=True)
    natural_analysis.add_argument("--output-dir", required=True)
    natural_analysis.set_defaults(handler=analyze_natural_command)

    crag_freeze = commands.add_parser(
        "freeze-public-crag", help="freeze a pinned, stratified CRAG subset"
    )
    crag_freeze.add_argument("--config", required=True)
    crag_freeze.add_argument("--cache-root", required=True)
    crag_freeze.add_argument("--output-dir", required=True)
    crag_freeze.set_defaults(handler=freeze_public_crag_command)

    crag_analysis = commands.add_parser(
        "analyze-public-crag", help="audit CRAG triggers with matched context controls"
    )
    crag_analysis.add_argument("--config", required=True)
    crag_analysis.add_argument("--cache-root", required=True)
    crag_analysis.add_argument("--selection", required=True)
    crag_analysis.add_argument("--output-dir", required=True)
    crag_analysis.set_defaults(handler=analyze_public_crag_command)
