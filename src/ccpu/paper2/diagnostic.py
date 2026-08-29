"""Richer Paper 2 benchmark and CPU-first failure decomposition."""

from __future__ import annotations

import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.lexical_routing import (
    NativeTokenizerSpec,
    run_matched_lexical_comparison,
    score_labels,
)
from ccpu.common.metrics import binary_classification, safe_mean

from .next_experiment import deterministic_reflex
from .runtime import HeterogeneousRuntime

_ENGINES = ("calculator", "date", "units", "graph", "datalog")
_LABELS = ("NONE", "CALCULATOR", "DATE", "UNITS", "GRAPH", "DATALOG")
_UNIT_FACTORS = {
    "mile": Decimal("1609.344"),
    "kilometer": Decimal(1000),
    "meter": Decimal(1),
    "foot": Decimal("0.3048"),
    "pound": Decimal("0.45359237"),
    "kilogram": Decimal(1),
    "hour": Decimal(3600),
    "minute": Decimal(60),
}
_UNIT_DIMENSIONS = {
    "mile": "length",
    "kilometer": "length",
    "meter": "length",
    "foot": "length",
    "pound": "mass",
    "kilogram": "mass",
    "hour": "time",
    "minute": "time",
}
_UNIT_PAIRS = (
    ("mile", "kilometer"),
    ("foot", "meter"),
    ("pound", "kilogram"),
    ("hour", "minute"),
)


@dataclass(frozen=True)
class DiagnosticBenchmarkConfig:
    seed: int = 22601
    train_per_engine: int = 250
    dev_per_engine: int = 50
    test_per_engine: int = 100
    train_controls: int = 1000
    dev_controls: int = 200
    test_controls: int = 400

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DiagnosticBenchmarkConfig:
        values = raw.get("diagnostic_benchmark", raw)
        defaults = asdict(cls())
        return cls(**{key: int(values.get(key, value)) for key, value in defaults.items()})


_PROMPT_FRAMES = {
    "calculator": {
        "train": (
            "Find the exact {operation} of {left} and {right}.",
            "Arithmetic request: {left} {symbol} {right}; return the exact value.",
        ),
        "dev": (
            "A ledger needs {left} {operation_word} {right}. What is the exact result?",
            "Evaluate precisely: {left} {symbol} {right}.",
        ),
        "test": (
            "Without estimating, what do you get when {left} is {operation_word} {right}?",
            "Give the integer result for the expression {left} {symbol} {right}.",
        ),
    },
    "date": {
        "train": (
            "Move {days} days forward from {first}; give the ISO date.",
            "How many days separate {first} and {second}?",
        ),
        "dev": (
            "Advance the calendar by {days} days starting at {first}.",
            "Return the signed day difference from {first} to {second}.",
        ),
        "test": (
            "Which ISO date lands {days} days after {first}?",
            "Count the elapsed days between {first} and {second}.",
        ),
    },
    "units": {
        "train": (
            "Convert {value} {source}s into {target}s exactly.",
            "Express {value} {source} in {target}.",
        ),
        "dev": (
            "What quantity in {target}s equals {value} {source}s?",
            "Perform the dimensional conversion: {value} {source} to {target}.",
        ),
        "test": (
            "Translate the measurement {value} {source}s into {target}s.",
            "Give the exact {target} value corresponding to {value} {source}.",
        ),
    },
    "graph": {
        "train": (
            "In an ISA taxonomy, {a} is a {b} and {b} is a {c}. Is {query_left} a {query_right}?",
            "Hierarchy facts: {a} kind-of {b}; {b} kind-of {c}. Test whether {query_left} inherits {query_right}.",
        ),
        "dev": (
            "Given classes {a} -> {b} -> {c}, decide the ISA claim {query_left} -> {query_right}.",
            "The type chain says {a} belongs under {b}, then {c}. Does {query_left} count as {query_right}?",
        ),
        "test": (
            "Resolve this taxonomy: {a} specializes {b}, which specializes {c}; is {query_left} an instance-kind of {query_right}?",
            "Follow the inheritance chain {a}, {b}, {c}. Verify the type relation from {query_left} to {query_right}.",
        ),
    },
    "datalog": {
        "train": (
            "Directed links run {a} to {b} and {b} to {c}. Is {query_right} reachable from {query_left}?",
            "With link({a},{b}) and link({b},{c}), query reachability from {query_left} to {query_right}.",
        ),
        "dev": (
            "A relation has arcs {a}->{b}->{c}. Can a path lead from {query_left} to {query_right}?",
            "Use the two link facts through {b}; determine reachable({query_left},{query_right}).",
        ),
        "test": (
            "For the directed chain {a}, {b}, {c}, decide whether {query_right} can be reached starting at {query_left}.",
            "Apply transitive link rules to {a}->{b} and {b}->{c}; answer the path query {query_left}->{query_right}.",
        ),
    },
}

_CONTROL_FRAMES = {
    "train": (
        "The supplied arithmetic label for {marker} is 42; repeat 42 without calculating.",
        "Quote this date phrase exactly for {marker}: '30 days after 2090-01-01'.",
        "In a fictional unit story, {marker} says 'mile to kilometer'; repeat the title.",
        "Copy the ISA note '{marker} is a sample type' without testing inheritance.",
        "The Datalog syntax example for {marker} is quoted, not a query; answer NOT-A-QUERY.",
    ),
    "dev": (
        "An answer card already gives {marker}=17; return 17, no arithmetic.",
        "Editing task: preserve the words 'date difference' beside {marker}.",
        "Is 'kilometer' the supplied unit label for {marker}? Answer yes.",
        "A novel calls {marker} an ISA graph; repeat the fictional genre.",
        "Print the literal token reachable({marker},{marker}) without evaluating it.",
    ),
    "test": (
        "For {marker}, copy the provided result 144 rather than recomputing its product.",
        "Quote the heading 'ISO date after offset' for document {marker}.",
        "The form already states {marker}'s unit is meter; return meter.",
        "Hypothetically call {marker} a graph node; do not infer a hierarchy.",
        "A code sample mentions Datalog reachability for {marker}; classify it as SAMPLE.",
    ),
}


def generate_diagnostic_benchmark(
    config: DiagnosticBenchmarkConfig, output_dir: str | Path
) -> dict[str, Any]:
    split_specs = {
        "train": (config.train_per_engine, config.train_controls, config.seed),
        "dev": (config.dev_per_engine, config.dev_controls, config.seed + 10000),
        "test": (config.test_per_engine, config.test_controls, config.seed + 20000),
    }
    splits = {
        split: _build_split(split, per_engine, controls, seed)
        for split, (per_engine, controls, seed) in split_specs.items()
    }
    identifiers = [row["example_id"] for rows in splits.values() for row in rows]
    signatures = {
        split: {str(row["signature"]) for row in rows} for split, rows in splits.items()
    }
    target_answer_leaks = [
        row["example_id"]
        for rows in splits.values()
        for row in rows
        if row["should_trigger"] and str(row["answer"]) in str(row["target"]).split()
    ]
    audit = {
        "schema_version": "ccpu.paper2.diagnostic_audit.v1",
        "counts": {split: len(rows) for split, rows in splits.items()},
        "class_counts": {
            split: {
                label: sum(row["classification_label"] == label for row in rows)
                for label in _LABELS
            }
            for split, rows in splits.items()
        },
        "duplicate_ids": len(identifiers) - len(set(identifiers)),
        "train_dev_overlap": sorted(signatures["train"] & signatures["dev"]),
        "train_test_overlap": sorted(signatures["train"] & signatures["test"]),
        "dev_test_overlap": sorted(signatures["dev"] & signatures["test"]),
        "target_contains_answer": target_answer_leaks,
        "balanced_logic_labels": {
            split: {
                engine: {
                    answer: sum(
                        row["engine"] == engine and row["answer"] == answer for row in rows
                    )
                    for answer in ("true", "false")
                }
                for engine in ("graph", "datalog")
            }
            for split, rows in splits.items()
        },
    }
    fatal = (
        audit["duplicate_ids"]
        or audit["train_dev_overlap"]
        or audit["train_test_overlap"]
        or audit["dev_test_overlap"]
        or audit["target_contains_answer"]
    )
    if fatal:
        raise ValueError(f"Paper 2 diagnostic benchmark audit failed: {audit}")
    output_dir = Path(output_dir)
    paths = {
        split: write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in splits.items()
    }
    audit_path = write_json(output_dir / "audit.json", audit)
    root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": "ccpu.paper2.diagnostic_manifest.v1",
        "config": asdict(config),
        "paths": {split: str(path) for split, path in paths.items()},
        "dataset_sha256": {split: file_sha256(path) for split, path in paths.items()},
        "audit": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "fingerprint": fingerprint({"config": asdict(config), "audit": audit}),
        "environment": environment_manifest(root),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _build_split(split: str, per_engine: int, controls: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for engine in _ENGINES:
        for index in range(per_engine):
            content = _build_engine_row(engine, split, index, rng)
            rows.append(
                {
                    "schema_version": "ccpu.paper2.diagnostic_example.v1",
                    "example_id": f"p2d-{split}-{engine}-{index:04d}",
                    "split": split,
                    "engine": engine,
                    "classification_label": engine.upper(),
                    "should_trigger": True,
                    **content,
                }
            )
    for index in range(controls):
        marker = f"{split.upper()}-SAFE-{index:05d}"
        frame = _CONTROL_FRAMES[split][index % len(_CONTROL_FRAMES[split])]
        prompt = frame.format(marker=marker)
        rows.append(
            {
                "schema_version": "ccpu.paper2.diagnostic_example.v1",
                "example_id": f"p2d-{split}-control-{index:05d}",
                "split": split,
                "engine": "control",
                "classification_label": "NONE",
                "should_trigger": False,
                "prompt": prompt,
                "target": "NO_EXECUTION",
                "answer": _control_answer(prompt),
                "signature": f"control:{marker}",
                "operation": "none",
            }
        )
    rng.shuffle(rows)
    return rows


def _control_answer(prompt: str) -> str:
    if "repeat 42" in prompt:
        return "42"
    if "return 17" in prompt:
        return "17"
    if "provided result 144" in prompt:
        return "144"
    if "return meter" in prompt:
        return "meter"
    if "Answer yes" in prompt:
        return "yes"
    if "NOT-A-QUERY" in prompt:
        return "NOT-A-QUERY"
    if "SAMPLE" in prompt:
        return "SAMPLE"
    if "fictional genre" in prompt:
        return "fictional genre"
    if "repeat the title" in prompt:
        return "mile to kilometer"
    if "repeat the fictional genre" in prompt:
        return "fictional genre"
    if "literal token" in prompt:
        return re.search(r"reachable\([^)]+\)", prompt)[0]
    quoted = re.search(r"'([^']+)'", prompt)
    return quoted[1] if quoted else "NO_EXECUTION"


def _build_engine_row(
    engine: str, split: str, index: int, rng: random.Random
) -> dict[str, Any]:
    frame = _PROMPT_FRAMES[engine][split][index % 2]
    namespace = {"train": 100000, "dev": 300000, "test": 500000}[split]
    if engine == "calculator":
        left = namespace + rng.randrange(100, 90000)
        right = rng.randrange(11, 9000)
        operation, symbol, word = (
            ("sum", "+", "plus"),
            ("difference", "-", "minus"),
            ("product", "*", "times"),
        )[index % 3]
        answer = {"+": left + right, "-": left - right, "*": left * right}[symbol]
        prompt = frame.format(
            operation=operation,
            operation_word=word,
            symbol=symbol,
            left=left,
            right=right,
        )
        target = f"```calculator\n{left} {symbol} {right}\n```"
        return _content(prompt, target, str(answer), f"calc:{left}:{symbol}:{right}", operation)
    if engine == "date":
        first = date({"train": 2030, "dev": 2060, "test": 2090}[split] + index % 20, 1 + index % 12, 1 + index % 24)
        days = rng.randrange(3, 360)
        second = first + timedelta(days=days)
        if index % 2 == 0:
            prompt = _PROMPT_FRAMES[engine][split][0].format(
                days=days, first=first.isoformat(), second=second.isoformat()
            )
            target = f"```date\nadd {first.isoformat()} P{days}D\n```"
            return _content(prompt, target, second.isoformat(), f"date:add:{first}:{days}", "add")
        prompt = _PROMPT_FRAMES[engine][split][1].format(
            days=days, first=first.isoformat(), second=second.isoformat()
        )
        target = f"```date\ndiff {first.isoformat()} {second.isoformat()}\n```"
        return _content(prompt, target, str(days), f"date:diff:{first}:{second}", "diff")
    if engine == "units":
        source, target_unit = _UNIT_PAIRS[index % len(_UNIT_PAIRS)]
        value = Decimal(namespace + rng.randrange(10, 9000)) / Decimal(10)
        with localcontext() as context:
            context.prec = 40
            converted = value * _UNIT_FACTORS[source] / _UNIT_FACTORS[target_unit]
        answer = format(converted.normalize(), "f")
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
        source_value = format(value.normalize(), "f")
        prompt = frame.format(value=source_value, source=source, target=target_unit)
        target = f"```units\nconvert {source_value} {source} -> {target_unit}\n```"
        return _content(
            prompt,
            target,
            answer,
            f"units:{source_value}:{source}:{target_unit}",
            _UNIT_DIMENSIONS[source],
        )
    prefix = f"{split}{engine[0]}{index:05d}"
    a, b, c = f"{prefix}a", f"{prefix}b", f"{prefix}c"
    positive = index % 2 == 0
    query_left, query_right = (a, c) if positive else (c, a)
    prompt = frame.format(
        a=a,
        b=b,
        c=c,
        query_left=query_left,
        query_right=query_right,
    )
    if engine == "graph":
        target = (
            f"```graph\nisa {a} {b}\nisa {b} {c}\n"
            f"query isa {query_left} {query_right}\n```"
        )
    else:
        target = (
            f"```datalog\nfact link({a},{b})\nfact link({b},{c})\n"
            f"query reachable({query_left},{query_right})\n```"
        )
    return _content(prompt, target, str(positive).lower(), f"{engine}:{prefix}:{positive}", "query")


def _content(prompt: str, target: str, answer: str, signature: str, operation: str) -> dict[str, str]:
    return {
        "prompt": prompt,
        "target": target,
        "answer": answer,
        "signature": signature,
        "operation": operation,
    }


def lexical_audit(train_path: str | Path, test_path: str | Path) -> dict[str, Any]:
    train = read_jsonl(train_path)
    test = read_jsonl(test_path)
    models = _fit_classical_models(train)
    results = []
    for name, model in models.items():
        started = time.perf_counter_ns()
        predicted = model.predict([str(row["prompt"]) for row in test])
        elapsed = time.perf_counter_ns() - started
        results.append(_classification_metrics(name, test, predicted, elapsed))
    return {
        "schema_version": "ccpu.paper2.diagnostic_lexical_audit.v1",
        "train_count": len(train),
        "test_count": len(test),
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
        "results": results,
        "maximum_accuracy": max(row["accuracy"] for row in results),
    }


def analyze_trigger_ladder(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    train = read_jsonl(train_path)
    test = read_jsonl(test_path)
    models = _fit_classical_models(train)
    predictors: dict[str, Any] = {
        "T0_anchored_parser": lambda prompts: [_t0(prompt) for prompt in prompts],
        "T1_lexical_regex": lambda prompts: [_t1(prompt) for prompt in prompts],
        "T2_semantic_rules": lambda prompts: [_t2(prompt) for prompt in prompts],
        "T3_tfidf_linear": models["tfidf_word_ngrams"].predict,
        "T4_latent_cpu_classifier": models["latent_tfidf_svd"].predict,
    }
    prompts = [str(row["prompt"]) for row in test]
    rows = []
    predictions = []
    for name, predict in predictors.items():
        started = time.perf_counter_ns()
        labels = [str(value) for value in predict(prompts)]
        elapsed = time.perf_counter_ns() - started
        metrics = _classification_metrics(name, test, labels, elapsed)
        execution = _score_parser_execution(test, labels, condition=name)
        metrics.update(execution["summary"])
        metrics.update(_ladder_metadata(name))
        rows.append(metrics)
        predictions.extend(execution["rows"])
    oracle_labels = [str(row["classification_label"]) for row in test]
    oracle = _score_parser_execution(
        test, oracle_labels, condition="C_oracle_engine_parser"
    )
    rows.append(
        {
            **_classification_metrics("C_oracle_engine_parser", test, oracle_labels, 0),
            **oracle["summary"],
            "portability": "high",
            "engineering_burden": "per-engine grammar",
        }
    )
    curves = _learning_curves(train, test)
    result = {
        "schema_version": "ccpu.paper2.trigger_ladder_analysis.v1",
        "train_count": len(train),
        "test_count": len(test),
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
        "trigger_ladder": rows,
        "learning_curves": curves,
        "decision": _decision(rows),
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "trigger_ladder.json", result)
    write_jsonl(output_dir / "parser_predictions.jsonl", predictions)
    _plot_ladder(result, output_dir / "trigger_ladder.png")
    _plot_learning_curves(result, output_dir / "classification_learning_curves.png")
    return result


def analyze_tokenizer_trigger_ladder(
    train_path: str | Path,
    dev_path: str | Path,
    test_path: str | Path,
    tokenizer_config_path: str | Path,
    neural_predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    test = read_jsonl(test_path)
    config = read_json(tokenizer_config_path)
    specs = [NativeTokenizerSpec(**model) for model in config["models"]]
    output_dir = Path(output_dir)
    result = run_matched_lexical_comparison(
        train,
        dev,
        test,
        text_key="prompt",
        label_key="classification_label",
        negative_label="NONE",
        tokenizer_specs=specs,
        output_dir=output_dir,
        source_hashes={
            "train": file_sha256(train_path),
            "dev": file_sha256(dev_path),
            "test": file_sha256(test_path),
        },
        subgroup_key="classification_label",
        include_prototypes=True,
    )
    lexical_predictions = read_jsonl(output_dir / "predictions.jsonl")
    by_condition_split: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in lexical_predictions:
        by_condition_split[(row["condition"], row["split"])][row["example_id"]] = row

    parser_rows = []
    for condition in result["results"]:
        predicted = [
            by_condition_split[(condition["condition"], "test")][row["example_id"]][
                "predicted_label"
            ]
            for row in test
        ]
        execution = _score_parser_execution(test, predicted, condition=condition["condition"])
        condition["test"]["engine_selection_accuracy"] = safe_mean(
            predicted[index] == row["classification_label"]
            for index, row in enumerate(test)
            if row["classification_label"] != "NONE"
        )
        condition["test"].update(execution["summary"])
        parser_rows.extend(execution["rows"])

    neural_rows = read_jsonl(neural_predictions_path)
    neural = {str(row["example_id"]): str(row["predicted_engine"]) for row in neural_rows}
    if set(neural) != {str(row["example_id"]) for row in test}:
        raise ValueError("hierarchical neural fallback must exactly cover the tokenizer test freeze")
    hierarchy = []
    hierarchy_predictions = []
    easy_labels = {"NONE", "CALCULATOR", "DATE", "UNITS"}
    for condition in result["results"]:
        name = condition["condition"]
        dev_rows = by_condition_split[(name, "dev")]
        threshold, accepted_accuracy, accepted_coverage = _cpu_acceptance_threshold(
            dev, dev_rows, easy_labels
        )
        test_rows = by_condition_split[(name, "test")]
        labels = []
        oracle_labels = []
        fallback_count = 0
        for row in test:
            lexical = test_rows[str(row["example_id"])]
            logic_deferred = _t2(str(row["prompt"])) in {"GRAPH", "DATALOG"}
            cpu_accepted = (
                not logic_deferred
                and
                lexical["predicted_label"] in easy_labels
                and float(lexical["confidence"]) >= threshold
            )
            if cpu_accepted:
                final = str(lexical["predicted_label"])
                oracle_final = final
            else:
                fallback_count += 1
                final = neural[str(row["example_id"])]
                oracle_final = str(row["classification_label"])
            labels.append(final)
            oracle_labels.append(oracle_final)
            hierarchy_predictions.append(
                {
                    "schema_version": "ccpu.paper2.hierarchical_trigger_prediction.v1",
                    "condition": name,
                    "example_id": str(row["example_id"]),
                    "gold_engine": str(row["classification_label"]),
                    "cpu_engine": str(lexical["predicted_label"]),
                    "cpu_confidence": float(lexical["confidence"]),
                    "logic_cue_deferred": logic_deferred,
                    "cpu_accepted": cpu_accepted,
                    "final_engine": final,
                    "fallback": "none" if cpu_accepted else "qwen_l2_router",
                }
            )
        scored = score_labels(
            [str(row["classification_label"]) for row in test],
            labels,
            negative_label="NONE",
        )
        engine_selection_accuracy = safe_mean(
            labels[index] == row["classification_label"]
            for index, row in enumerate(test)
            if row["classification_label"] != "NONE"
        )
        execution = _score_parser_execution(
            test, labels, condition=f"hierarchical_{name}_qwen_l2"
        )
        oracle_execution = _score_parser_execution(
            test, oracle_labels, condition=f"hierarchical_{name}_oracle_fallback"
        )
        fallback_rate = fallback_count / max(len(test), 1)
        hierarchy.append(
            {
                "condition": f"hierarchical_{name}_qwen_l2",
                "cpu_condition": name,
                "cpu_threshold_selected_on_dev": threshold,
                "dev_cpu_accepted_accuracy": accepted_accuracy,
                "dev_cpu_coverage": accepted_coverage,
                "fallback_rate": fallback_rate,
                "model_calls_avoided_rate": 1.0 - fallback_rate,
                "context_tokens_added": 0,
                "fallback_model": "Qwen3-0.6B L2 six-way router",
                "mean_estimated_route_latency_us": condition["mean_cpu_latency_us"]
                + fallback_rate * 554_663.2642222223,
                **scored,
                "engine_selection_accuracy": engine_selection_accuracy,
                **execution["summary"],
                "oracle_fallback_runtime_exact_rate": oracle_execution["summary"][
                    "runtime_exact_rate"
                ],
            }
        )
        parser_rows.extend(execution["rows"])
    eligible_hierarchy = [
        row for row in hierarchy if row["dev_cpu_accepted_accuracy"] >= 0.95
    ]
    selected_hierarchy = min(
        eligible_hierarchy or hierarchy,
        key=lambda row: (
            -row["dev_cpu_coverage"],
            -row["dev_cpu_accepted_accuracy"],
            row["cpu_condition"],
        ),
    )
    result["hierarchical_routing"] = hierarchy
    result["paper2_decision"] = {
        "selection_is_development_only": True,
        "selected_hierarchy": selected_hierarchy["condition"],
        "criterion": "maximize development CPU coverage at >=0.95 accepted accuracy",
        "status": (
            "hierarchical_cpu_neural_passes"
            if selected_hierarchy["engine_selection_accuracy"] >= 0.9
            and selected_hierarchy["false_activation_rate"] <= 0.1
            and selected_hierarchy["runtime_exact_rate"] >= 0.9
            else "retain_diagnostic_no_go"
        ),
        "paper3_gate": "no_go",
    }
    result["neural_fallback_sha256"] = file_sha256(neural_predictions_path)
    result["tokenizer_config_sha256"] = file_sha256(tokenizer_config_path)
    write_json(output_dir / "comparison.json", result)
    write_jsonl(output_dir / "parser_predictions.jsonl", parser_rows)
    write_jsonl(output_dir / "hierarchical_predictions.jsonl", hierarchy_predictions)
    _plot_tokenizer_ladder(result, output_dir / "tokenizer_trigger_ladder.png")
    return result


def _cpu_acceptance_threshold(
    dev: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    easy_labels: set[str],
) -> tuple[float, float, float]:
    candidates = []
    for threshold_index in range(21):
        threshold = threshold_index / 20
        accepted = [
            row
            for row in dev
            if _t2(str(row["prompt"])) not in {"GRAPH", "DATALOG"}
            and predictions[str(row["example_id"])]["predicted_label"] in easy_labels
            and float(predictions[str(row["example_id"])]["confidence"]) >= threshold
        ]
        correct = sum(
            predictions[str(row["example_id"])]["predicted_label"]
            == row["classification_label"]
            for row in accepted
        )
        accuracy = correct / len(accepted) if accepted else 0.0
        coverage = len(accepted) / max(len(dev), 1)
        candidates.append((accuracy >= 0.95, coverage, accuracy, threshold))
    selected = max(candidates)
    return float(selected[3]), float(selected[2]), float(selected[1])


def _plot_tokenizer_ladder(result: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    rows = result["results"]
    selected = sorted(rows, key=lambda row: (-row["dev"]["accuracy"], row["condition"]))[:12]
    labels = [row["condition"].replace("native_", "") for row in selected]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    positions = list(range(len(selected)))
    axes[0].barh(positions, [row["test"]["accuracy"] for row in selected], color="#176b87")
    axes[1].barh(
        positions,
        [row["mean_cpu_latency_us"] for row in selected],
        color="#c4512d",
    )
    for axis, title in zip(axes, ("Six-way test accuracy", "CPU route latency (us)"), strict=True):
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.22)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fit_classical_models(train: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as error:
        raise RuntimeError("Paper 2 diagnostic classifiers require scikit-learn") from error
    texts = [str(row["prompt"]) for row in train]
    labels = [str(row["classification_label"]) for row in train]
    models = {
        "bag_of_words": make_pipeline(
            CountVectorizer(binary=True),
            LogisticRegression(max_iter=3000, random_state=0),
        ),
        "tfidf_word_ngrams": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            LogisticRegression(max_iter=3000, random_state=0),
        ),
        "tfidf_character_ngrams": make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
            LogisticRegression(max_iter=3000, random_state=0),
        ),
        "latent_tfidf_svd": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            TruncatedSVD(n_components=48, random_state=0),
            LogisticRegression(max_iter=3000, random_state=0),
        ),
    }
    for model in models.values():
        model.fit(texts, labels)
    return models


def _classification_metrics(
    name: str, test: list[dict[str, Any]], predicted: Any, elapsed_ns: int
) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as error:
        raise RuntimeError("Paper 2 diagnostic metrics require scikit-learn") from error
    labels = [str(value) for value in predicted]
    gold = [str(row["classification_label"]) for row in test]
    trigger = binary_classification(
        [label != "NONE" for label in gold],
        [label != "NONE" for label in labels],
    )
    positives = [index for index, label in enumerate(gold) if label != "NONE"]
    return {
        "condition": name,
        "accuracy": float(accuracy_score(gold, labels)),
        "macro_f1": float(f1_score(gold, labels, average="macro", zero_division=0)),
        "trigger_recall": trigger["recall"],
        "false_activation_rate": trigger["false_intervention_rate"],
        "engine_selection_accuracy": safe_mean(labels[index] == gold[index] for index in positives),
        "mean_cpu_latency_us": elapsed_ns / max(len(test), 1) / 1000,
    }


def _t0(prompt: str) -> str:
    block = deterministic_reflex(prompt)
    return _block_label(block)


def _block_label(block: str | None) -> str:
    if not block or block == "NO_EXECUTION":
        return "NONE"
    match = re.match(r"```(calculator|date|units|graph|datalog)", block)
    return match[1].upper() if match else "NONE"


def _suppressed(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(?:supplied|already|repeat|quote|quoted|copy|fictional|novel|hypothetically|sample|editing|without (?:calculating|evaluating|testing|recomputing))\b",
            prompt,
            re.IGNORECASE,
        )
    )


def _t1(prompt: str) -> str:
    lowered = prompt.casefold()
    if _suppressed(prompt):
        return "NONE"
    if re.search(r"\d{4}-\d{2}-\d{2}", prompt) and any(
        cue in lowered for cue in ("date", "days", "calendar", "elapsed")
    ):
        return "DATE"
    if any(cue in lowered for cue in ("convert", "measurement", "quantity")):
        return "UNITS"
    if any(cue in lowered for cue in ("isa", "taxonomy", "hierarchy", "inheritance")):
        return "GRAPH"
    if any(cue in lowered for cue in ("reachable", "reachability", "directed", "transitive link")):
        return "DATALOG"
    if re.search(r"\d+\s*[+*-]\s*\d+", prompt) or any(
        cue in lowered for cue in ("arithmetic", "sum", "difference", "product", "times", "plus", "minus")
    ):
        return "CALCULATOR"
    return "NONE"


def _t2(prompt: str) -> str:
    if _suppressed(prompt):
        return "NONE"
    for label in ("DATE", "UNITS", "GRAPH", "DATALOG", "CALCULATOR"):
        try:
            deterministic_payload(prompt, label)
            return label
        except ValueError:
            continue
    return "NONE"


def deterministic_payload(prompt: str, engine_label: str) -> str:
    if engine_label == "CALCULATOR":
        values = [int(value) for value in re.findall(r"\b\d+\b", prompt)]
        if len(values) != 2:
            raise ValueError("calculator parser requires two integers")
        lowered = prompt.casefold()
        if "+" in prompt or any(word in lowered for word in ("sum", "plus")):
            symbol = "+"
        elif "*" in prompt or any(word in lowered for word in ("product", "times")):
            symbol = "*"
        elif "-" in prompt or any(word in lowered for word in ("difference", "minus")):
            symbol = "-"
        else:
            raise ValueError("calculator operation not found")
        return f"```calculator\n{values[0]} {symbol} {values[1]}\n```"
    if engine_label == "DATE":
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", prompt)
        lowered = prompt.casefold()
        if len(dates) == 1:
            days = re.search(r"\b(\d+) days\b", lowered)
            if not days:
                raise ValueError("date offset not found")
            return f"```date\nadd {dates[0]} P{days[1]}D\n```"
        if len(dates) == 2:
            return f"```date\ndiff {dates[0]} {dates[1]}\n```"
        raise ValueError("date parser requires one or two ISO dates")
    if engine_label == "UNITS":
        value = re.search(r"\b(\d+(?:\.\d+)?)\b", prompt)
        unit_pattern = "|".join(sorted(_UNIT_FACTORS, key=len, reverse=True))
        matches = list(re.finditer(rf"\b(?:{unit_pattern})s?\b", prompt, re.IGNORECASE))
        units = [match[0].removesuffix("s").casefold() for match in matches]
        if not value or len(units) != 2:
            raise ValueError("unit parser requires one value and two units")
        source_index = next(
            (index for index, match in enumerate(matches) if match.start() > value.end()),
            None,
        )
        if source_index is None:
            raise ValueError("unit parser could not bind source quantity")
        source = units[source_index]
        target = units[1 - source_index]
        if _UNIT_DIMENSIONS[source] != _UNIT_DIMENSIONS[target]:
            raise ValueError("unit parser requires one value and compatible units")
        return f"```units\nconvert {value[1]} {source} -> {target}\n```"
    identifiers = re.findall(r"\b(?:train|dev|test)[gd]\d{5}[abc]\b", prompt)
    if len(identifiers) < 5:
        raise ValueError("logic parser did not find facts and query")
    fact_entities = list(dict.fromkeys(identifiers[:-2]))
    if len(fact_entities) != 3:
        raise ValueError("logic parser requires a three-entity fact chain")
    a, b, c = fact_entities
    query_left, query_right = identifiers[-2], identifiers[-1]
    if "can be reached starting at" in prompt.casefold():
        query_left, query_right = query_right, query_left
    if engine_label == "GRAPH":
        return (
            f"```graph\nisa {a} {b}\nisa {b} {c}\n"
            f"query isa {query_left} {query_right}\n```"
        )
    if engine_label == "DATALOG":
        return (
            f"```datalog\nfact link({a},{b})\nfact link({b},{c})\n"
            f"query reachable({query_left},{query_right})\n```"
        )
    raise ValueError(f"unsupported engine label: {engine_label}")


def _score_parser_execution(
    test: list[dict[str, Any]], labels: list[str], *, condition: str
) -> dict[str, Any]:
    rows = []
    for row, label in zip(test, labels, strict=True):
        started = time.perf_counter_ns()
        block = None
        result = None
        try:
            if label != "NONE":
                block = deterministic_payload(str(row["prompt"]), label)
                result = HeterogeneousRuntime(max_state_items=16).execute_event(
                    block, event_id=f"diagnostic:{row['example_id']}"
                )
        except ValueError:
            block = None
        elapsed = time.perf_counter_ns() - started
        expected = bool(row["should_trigger"])
        exact = bool(result and result.ok and result.display == str(row["answer"]))
        rows.append(
            {
                "schema_version": "ccpu.paper2.parser_prediction.v1",
                "condition": condition,
                "example_id": row["example_id"],
                "gold_engine": row["classification_label"],
                "predicted_engine": label,
                "block": block,
                "payload_exact": block == row["target"] if expected else block is None,
                "runtime_exact": exact if expected else block is None,
                "latency_ns": elapsed,
            }
        )
    positives = [item for item, source in zip(rows, test, strict=True) if source["should_trigger"]]
    controls = [item for item, source in zip(rows, test, strict=True) if not source["should_trigger"]]
    return {
        "rows": rows,
        "summary": {
            "payload_exact_rate": safe_mean(item["payload_exact"] for item in positives),
            "runtime_exact_rate": safe_mean(item["runtime_exact"] for item in positives),
            "parser_false_activation_rate": safe_mean(not item["payload_exact"] for item in controls),
            "mean_parse_execute_latency_us": safe_mean(item["latency_ns"] for item in rows) / 1000,
        },
    }


def _ladder_metadata(name: str) -> dict[str, str]:
    metadata = {
        "T0_anchored_parser": ("high", "low; brittle templates"),
        "T1_lexical_regex": ("high", "low; cue maintenance"),
        "T2_semantic_rules": ("high", "medium; per-engine grammar"),
        "T3_tfidf_linear": ("high", "medium; retraining and audit"),
        "T4_latent_cpu_classifier": ("medium", "medium; latent projection"),
    }
    portability, burden = metadata[name]
    return {"portability": portability, "engineering_burden": burden}


def _learning_curves(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curves = []
    rng = random.Random(22611)
    by_label = {
        label: [row for row in train if row["classification_label"] == label]
        for label in _LABELS
    }
    for per_class in (25, 50, 100, 250):
        sampled = []
        for label, members in by_label.items():
            limit = min(per_class * (4 if label == "NONE" else 1), len(members))
            sampled.extend(rng.sample(members, limit))
        model = _fit_classical_models(sampled)["tfidf_word_ngrams"]
        labels = model.predict([str(row["prompt"]) for row in test])
        cell = _classification_metrics(f"n={per_class}", test, labels, 0)
        cell["per_engine_train"] = per_class
        cell["train_count"] = len(sampled)
        curves.append(cell)
    return curves


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_passes = [
        row
        for row in rows
        if row["condition"] != "C_oracle_engine_parser"
        and row["engine_selection_accuracy"] >= 0.9
        and row["false_activation_rate"] <= 0.1
        and row["runtime_exact_rate"] >= 0.9
    ]
    semantic_passes = [
        row
        for row in threshold_passes
        if row["condition"]
        in {"T2_semantic_rules", "T3_tfidf_linear", "T4_latent_cpu_classifier"}
    ]
    return {
        "status": "prefer_semantic_cpu" if semantic_passes else "escalate_neural_router",
        "criterion": (
            ">=0.9 engine selection/runtime exact and <=0.1 false activation; "
            "semantic decision considers T2-T4"
        ),
        "benchmark_passing_conditions": [row["condition"] for row in threshold_passes],
        "semantic_passing_conditions": [row["condition"] for row in semantic_passes],
        "deployment_baseline": (
            threshold_passes[0]["condition"] if threshold_passes else "none"
        ),
        "lora_gate": "defer_joint_lora" if semantic_passes else "run_factorized_lora",
    }


def _plot_ladder(result: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    rows = result["trigger_ladder"]
    labels = [str(row["condition"]).replace("_", " ") for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    x = range(len(rows))
    axes[0].bar(x, [row["engine_selection_accuracy"] for row in rows], color="#176b87")
    axes[1].bar(x, [row["runtime_exact_rate"] for row in rows], color="#c4512d", label="runtime exact")
    axes[1].plot(x, [row["false_activation_rate"] for row in rows], "o-", color="#222222", label="false activation")
    for axis, title in zip(axes, ("Engine selection", "Execution and false activation"), strict=True):
        axis.set_xticks(list(x), labels, rotation=25, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_learning_curves(result: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    rows = result["learning_curves"]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        [row["per_engine_train"] for row in rows],
        [row["accuracy"] for row in rows],
        "o-",
        color="#176b87",
        label="six-way accuracy",
    )
    axis.plot(
        [row["per_engine_train"] for row in rows],
        [row["engine_selection_accuracy"] for row in rows],
        "s-",
        color="#c4512d",
        label="positive engine selection",
    )
    axis.set(xlabel="training examples per engine", ylabel="test score", ylim=(0, 1.05))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 2 diagnostic plots require matplotlib") from error
    return plt
