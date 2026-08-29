"""Natural-language robustness benchmark and factorized epistemic evaluation."""

from __future__ import annotations

import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
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

from .experiment import answers_equal, extract_answer
from .generation import ConfidenceBackend
from .source import ControlledFactStore
from .triggers import fit_confidence_threshold, semantic_risk


@dataclass(frozen=True)
class NaturalRobustnessConfig:
    seed: int = 15531
    train_per_category: int = 3
    dev_per_category: int = 1
    test_per_category: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NaturalRobustnessConfig:
        values = raw.get("natural_robustness", raw)
        return cls(
            seed=int(values.get("seed", cls.seed)),
            train_per_category=int(
                values.get("train_per_category", cls.train_per_category)
            ),
            dev_per_category=int(values.get("dev_per_category", cls.dev_per_category)),
            test_per_category=int(values.get("test_per_category", cls.test_per_category)),
        )


_REQUIRED_CATEGORIES = (
    "source_release",
    "private_identifier",
    "authoritative_attribution",
    "changed_familiar",
    "dynamic_assignment",
    "recorded_field",
    "opaque_private_field",
    "post_reorganization",
)
_CONTROL_CATEGORIES = (
    "context_sufficient",
    "quoted_freshness",
    "hypothetical",
    "electric_current",
    "historical_date",
    "fictional_source",
    "compute_not_retrieve",
    "stable_familiar",
)
_CATEGORY_CODES = {
    "source_release": "SR",
    "private_identifier": "PI",
    "authoritative_attribution": "AA",
    "changed_familiar": "CF",
    "dynamic_assignment": "DA",
    "recorded_field": "RF",
    "opaque_private_field": "OP",
    "post_reorganization": "PR",
    "context_sufficient": "CS",
    "quoted_freshness": "QF",
    "hypothetical": "HY",
    "electric_current": "EC",
    "historical_date": "HD",
    "fictional_source": "FS",
    "compute_not_retrieve": "CN",
    "stable_familiar": "ST",
}
_TEMPLATE_CONTROL_ANSWERS = {
    "quoted_freshness": ("latest custodian", "current registry", "updated owner"),
    "historical_date": ("18th", "yes", "1848"),
    "compute_not_retrieve": ("42", "120", "2026-08-29"),
    "stable_familiar": ("Au", "Lisbon", "H2O"),
}

_REQUIRED_TEMPLATES = {
    "source_release": (
        "Which custodian appears in release {release} of the Atlas dossier for {entity}?",
        "Read the {release} Atlas entry for {entity} and return its custodian.",
        "For {entity}, what custodian does Atlas publication {release} record?",
    ),
    "private_identifier": (
        "Complete the custodian field for case {entity}.",
        "Who owns the internal case {entity}?",
        "Return the assigned custodian for {entity}.",
    ),
    "authoritative_attribution": (
        "What custodian would the compliance office attest for {entity}?",
        "Name the custodian attributable to the compliance record for {entity}.",
        "For audit attribution, identify {entity}'s custodian.",
    ),
    "changed_familiar": (
        "The familiar custodian for {entity} was reassigned. Who fills the role now?",
        "After the handover, which custodian is responsible for {entity}?",
        "Give the replacement custodian for {entity} after its reassignment.",
    ),
    "dynamic_assignment": (
        "Who is on call for {entity}'s rotation?",
        "Name the person covering the active rotation for {entity}.",
        "Which custodian is handling {entity}'s present rotation?",
    ),
    "recorded_field": (
        "For filing {entity}, supply the recorded custodian.",
        "What name occupies the custodian field in filing {entity}?",
        "Return the custodian recorded against {entity}.",
    ),
    "opaque_private_field": (
        "Fill field 7B for {entity}.",
        "What value belongs in box 7B of {entity}?",
        "Provide {entity}'s internal 7B value.",
    ),
    "post_reorganization": (
        "Use the designation in force after reorganization for {entity}.",
        "Following the restructure, who is designated for {entity}?",
        "Which custodian applies to {entity} after organizational change?",
    ),
}

_CONTROL_TEMPLATES = {
    "context_sufficient": (
        "The active Atlas brief states that {entity}'s custodian is {answer}. Repeat that value.",
        "Use only this supplied filing context: {entity} -> {answer}. Return the custodian.",
        "The current answer for {entity} is already given as {answer}; repeat it for the audit.",
    ),
    "quoted_freshness": (
        "Compliance note {entity} uses the label 'latest custodian'. Quote that label exactly.",
        "For {entity}, repeat the quoted Atlas words 'current registry' without resolving it.",
        "Filing {entity} shows the text 'updated owner'. Copy that label exactly.",
    ),
    "hypothetical": (
        "If {entity} hypothetically had custodian {answer}, repeat {answer}.",
        "In a thought experiment, assume {entity} maps to {answer}; return {answer}.",
        "Suppose, without asserting fact, that {entity}'s value is {answer}. Repeat it.",
    ),
    "electric_current": (
        "Atlas training item {entity} asks: what SI unit measures electric current?",
        "In {entity}'s custodian compliance quiz, name the unit used for electric current.",
        "After filing {entity}'s update, answer the quiz: electric current uses which SI unit?",
    ),
    "historical_date": (
        "The old dossier for {entity} mentions 1776; which century contains that year?",
        "For {entity}'s audit timeline, was 1900 before 2000? Answer yes or no.",
        "After a reorganization note for {entity}, return the year printed here: 1848.",
    ),
    "fictional_source": (
        "In the fictional Atlas story, custodian {answer} guards {entity}. Name the character.",
        "A novel says registry keeper {answer} watches {entity}; repeat the fictional name.",
        "Within this invented dossier, {entity} belongs to {answer}. Return {answer}.",
    ),
    "compute_not_retrieve": (
        "For internal case {entity}, calculate 17 + 25 and return only the number.",
        "A custodian rotation lasts 2 hours; convert that duration to minutes.",
        "A filing changed on 2026-08-28; what ISO date is one day after it?",
    ),
    "stable_familiar": (
        "Compliance quiz {entity} asks for the chemical symbol for gold. What is it?",
        "Atlas example {entity} asks for the capital of Portugal. Name it.",
        "The custodian test in {entity} asks for the chemical formula for water.",
    ),
}


def _control_answer(category: str, index: int, template_index: int) -> str:
    values = {
        "context_sufficient": f"CTX-{index:03d}",
        "quoted_freshness": _TEMPLATE_CONTROL_ANSWERS["quoted_freshness"][template_index],
        "hypothetical": f"HYP-{index:03d}",
        "electric_current": "ampere",
        "historical_date": _TEMPLATE_CONTROL_ANSWERS["historical_date"][template_index],
        "fictional_source": f"FICTION-{index:03d}",
        "compute_not_retrieve": _TEMPLATE_CONTROL_ANSWERS["compute_not_retrieve"][
            template_index
        ],
        "stable_familiar": _TEMPLATE_CONTROL_ANSWERS["stable_familiar"][template_index],
    }
    return values[category]


def generate_natural_benchmark(
    config: NaturalRobustnessConfig, output_dir: str | Path
) -> dict[str, Any]:
    rng = random.Random(config.seed)
    rows = []
    records = []
    split_counts = {
        "train": config.train_per_category,
        "dev": config.dev_per_category,
        "test": config.test_per_category,
    }
    for split, count in split_counts.items():
        split_offset = {"train": 0, "dev": 100, "test": 200}[split]
        for category in (*_REQUIRED_CATEGORIES, *_CONTROL_CATEGORIES):
            required = category in _REQUIRED_CATEGORIES
            for local_index in range(count):
                index = split_offset + local_index
                template_index = split_offset // 100 % 3
                entity = f"CASE-{_CATEGORY_CODES[category]}{index:03d}"
                answer = (
                    f"CUST-{rng.randrange(1000, 9999)}"
                    if required
                    else _control_answer(category, index, template_index)
                )
                release = f"R{1 + local_index}.{2 + split_offset // 100}"
                templates = _REQUIRED_TEMPLATES if required else _CONTROL_TEMPLATES
                template = templates[category][template_index]
                question = template.format(entity=entity, answer=answer, release=release)
                suffixes = (
                    "Answer concisely.",
                    "Return only the requested value.",
                    "Do not add commentary.",
                    "Give the direct response.",
                )
                question = f"{question} {suffixes[local_index % len(suffixes)]}"
                example_id = f"p15-natural-{split}-{category}-{local_index:03d}"
                if required:
                    records.append(
                        {
                            "record_id": f"natural-record-{split}-{category}-{local_index:03d}",
                            "entity": entity,
                            "attribute": "custodian",
                            "value": answer,
                            "valid_from": "2026-01-01",
                        }
                    )
                rows.append(
                    {
                        "example_id": example_id,
                        "split": split,
                        "question": question,
                        "answer": answer,
                        "entity": entity,
                        "attribute": "custodian",
                        "as_of": "2026-08-29",
                        "evidence_required": required,
                        "category": category,
                        "template_index": template_index,
                        "cue_removed": required
                        and category in {"private_identifier", "opaque_private_field"},
                        "context_sufficient": category == "context_sufficient",
                    }
                )
    rng.shuffle(rows)
    record_keys = [(record["entity"], record["attribute"]) for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("natural benchmark source contains duplicate entity/attribute keys")
    answer_consistency_errors = sum(
        row["category"] in _TEMPLATE_CONTROL_ANSWERS
        and row["answer"]
        != _TEMPLATE_CONTROL_ANSWERS[str(row["category"])][int(row["template_index"])]
        for row in rows
    )
    if answer_consistency_errors:
        raise ValueError("natural benchmark contains template/gold answer mismatches")
    output_dir = Path(output_dir)
    dataset = write_jsonl(output_dir / "benchmark.jsonl", rows)
    source = write_json(
        output_dir / "source.json",
        {
            "source_id": "atlas-natural-controlled",
            "version": "2026.08.29-natural-v5",
            "records": records,
        },
    )
    signatures = {
        split: {fingerprint(row["question"].casefold().split()) for row in rows if row["split"] == split}
        for split in split_counts
    }
    normalized_signatures = {
        split: {
            fingerprint(_normalize_question(str(row["question"])).split())
            for row in rows
            if row["split"] == split
        }
        for split in split_counts
    }
    within_split_duplicates = {
        split: len([row for row in rows if row["split"] == split]) - len(signatures[split])
        for split in split_counts
    }
    audit = {
        "schema_version": "ccpu.paper1_5.natural_audit.v1",
        "counts": {split: sum(row["split"] == split for row in rows) for split in split_counts},
        "label_balance": {
            split: {
                "required": sum(row["split"] == split and row["evidence_required"] for row in rows),
                "not_required": sum(
                    row["split"] == split and not row["evidence_required"] for row in rows
                ),
            }
            for split in split_counts
        },
        "exact_signature_overlap": {
            "train_dev": sorted(signatures["train"] & signatures["dev"]),
            "train_test": sorted(signatures["train"] & signatures["test"]),
            "dev_test": sorted(signatures["dev"] & signatures["test"]),
        },
        "normalized_template_overlap": {
            "train_dev": sorted(normalized_signatures["train"] & normalized_signatures["dev"]),
            "train_test": sorted(normalized_signatures["train"] & normalized_signatures["test"]),
            "dev_test": sorted(normalized_signatures["dev"] & normalized_signatures["test"]),
        },
        "within_split_duplicate_questions": within_split_duplicates,
        "source_key_collisions": len(record_keys) - len(set(record_keys)),
        "answer_consistency_errors": answer_consistency_errors,
        "required_categories": list(_REQUIRED_CATEGORIES),
        "control_categories": list(_CONTROL_CATEGORIES),
    }
    audit_path = write_json(output_dir / "freeze_audit.json", audit)
    manifest = {
        "schema_version": "ccpu.paper1_5.natural_manifest.v1",
        "config": asdict(config),
        "record_count": len(records),
        "example_count": len(rows),
        "dataset": str(dataset),
        "source": str(source),
        "audit": str(audit_path),
        "fingerprint": fingerprint({"config": asdict(config), "rows": rows, "records": records}),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


_FEATURE_PATTERNS = {
    "temporal": re.compile(r"\b(?:now|active|after|following|replacement|reassigned|handover)\b", re.IGNORECASE),
    "source_specific": re.compile(r"\b(?:atlas|dossier|record|filing|publication|release)\b", re.IGNORECASE),
    "private_identifier": re.compile(r"\bCASE-[A-Z][A-Z]\d{3}\b|\b(?:field|box) 7B\b", re.IGNORECASE),
    "attribution": re.compile(r"\b(?:attest|attribut|audit|compliance)\w*\b", re.IGNORECASE),
    "version_change": re.compile(r"\b(?:reassign|handover|restructure|reorganization|change)\w*\b", re.IGNORECASE),
    "dynamic_field": re.compile(r"\b(?:on call|rotation|custodian|designated)\b", re.IGNORECASE),
}
_SUPPRESS_PATTERNS = {
    "active_context": re.compile(r"\b(?:active brief states|supplied context|already given|answer .* is)\b", re.IGNORECASE),
    "quote_hypothetical": re.compile(r"\b(?:quote|quoted|copy|hypothetical|thought experiment|suppose|fictional|novel|invented)\b", re.IGNORECASE),
    "compute": re.compile(r"\b(?:calculate|convert|one day after|hours to minutes)\b", re.IGNORECASE),
    "safe_stable": re.compile(r"\b(?:electric current|chemical symbol|capital of|chemical formula|historical year|century)\b", re.IGNORECASE),
}


def semantic_features(text: str) -> dict[str, bool]:
    features = {name: bool(pattern.search(text)) for name, pattern in _FEATURE_PATTERNS.items()}
    features.update({name: bool(pattern.search(text)) for name, pattern in _SUPPRESS_PATTERNS.items()})
    positive = any(features[name] for name in _FEATURE_PATTERNS)
    suppressed = any(features[name] for name in _SUPPRESS_PATTERNS)
    features["combined"] = positive and not suppressed
    return features


def _feature_trigger(features: dict[str, bool], condition: str) -> bool:
    if condition == "temporal_only":
        return features["temporal"] or features["version_change"]
    if condition == "source_only":
        return features["source_specific"] or features["attribution"]
    if condition == "context_sufficiency_only":
        return any(features[name] for name in _FEATURE_PATTERNS) and not features["active_context"]
    if condition == "combined_semantic":
        return features["combined"]
    raise ValueError(f"unknown natural semantic condition: {condition}")


def lexical_audit(dataset_path: str | Path) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from sklearn.pipeline import make_pipeline
    except ImportError as error:
        raise RuntimeError("Natural lexical audit requires scikit-learn") from error
    rows = [row for row in _read_jsonl(dataset_path)]
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    models = {
        "bag_of_words": TfidfVectorizer(binary=True, use_idf=False, norm=None),
        "tfidf_word_ngrams": TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        "tfidf_character_ngrams": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
    }
    results = []
    for name, vectorizer in models.items():
        pipeline = make_pipeline(vectorizer, LogisticRegression(max_iter=2000, random_state=0))
        pipeline.fit(
            [str(row["question"]) for row in train],
            [bool(row["evidence_required"]) for row in train],
        )
        predicted = pipeline.predict([str(row["question"]) for row in test])
        gold = [bool(row["evidence_required"]) for row in test]
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold, predicted, average="binary", zero_division=0
        )
        results.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(gold, predicted)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    maximum = max(row["accuracy"] for row in results)
    return {
        "schema_version": "ccpu.paper1_5.lexical_audit.v1",
        "train_count": len(train),
        "test_count": len(test),
        "results": results,
        "maximum_accuracy": maximum,
        "status": "revise_before_semantic_claim" if maximum >= 0.9 else "passes_triviality_screen",
        "threshold": 0.9,
    }


def tokenizer_trigger_comparison(
    benchmark_path: str | Path,
    tokenizer_config_path: str | Path,
    output_dir: str | Path,
    *,
    train_path: str | Path | None = None,
    dev_path: str | Path | None = None,
) -> dict[str, Any]:
    benchmark = read_jsonl(benchmark_path)
    if train_path and dev_path:
        train = _policy_trigger_rows(read_jsonl(train_path))
        dev = _policy_trigger_rows(read_jsonl(dev_path))
        test = _benchmark_trigger_rows(row for row in benchmark if row["split"] == "test")
        source_hashes = {
            "train": file_sha256(train_path),
            "dev": file_sha256(dev_path),
            "test": file_sha256(benchmark_path),
        }
    elif train_path or dev_path:
        raise ValueError("Paper 1.5 tokenizer comparison requires both train and dev paths")
    else:
        normalized = _benchmark_trigger_rows(benchmark)
        train = [row for row in normalized if row["split"] == "train"]
        dev = [row for row in normalized if row["split"] == "dev"]
        test = [row for row in normalized if row["split"] == "test"]
        source_hashes = {"benchmark": file_sha256(benchmark_path)}
    if not train or not dev or not test:
        raise ValueError("Paper 1.5 tokenizer comparison requires non-empty train/dev/test data")

    config = read_json(tokenizer_config_path)
    specs = [NativeTokenizerSpec(**model) for model in config["models"]]
    result = run_matched_lexical_comparison(
        train,
        dev,
        test,
        text_key="text",
        label_key="label",
        negative_label="NONE",
        tokenizer_specs=specs,
        output_dir=output_dir,
        source_hashes=source_hashes,
        subgroup_key="subgroup",
    )
    semantic_started = time.perf_counter_ns()
    semantic_policy = "legacy_semantic_risk" if train_path else "natural_combined_semantic"
    semantic_labels = []
    for row in test:
        triggered = (
            semantic_risk(row["text"])[0]
            if semantic_policy == "legacy_semantic_risk"
            else semantic_features(row["text"])["combined"]
        )
        semantic_labels.append("RETRIEVE" if triggered else "NONE")
    semantic_ns = time.perf_counter_ns() - semantic_started
    semantic = {
        "condition": "transparent_semantic_runtime",
        "classifier": "fixed_semantic_features",
        "representation": "interpretable_runtime_policy",
        "semantic_policy": semantic_policy,
        "fit_time_ms": 0.0,
        "mean_cpu_latency_us": semantic_ns / max(len(test), 1) / 1000,
        "dev": None,
        "test": {
            **score_labels(
                [row["label"] for row in test], semantic_labels, negative_label="NONE"
            ),
            "by_subgroup": _trigger_subgroups(test, semantic_labels),
        },
    }
    result["results"].append(semantic)
    for row in result["results"]:
        if row["condition"] != "transparent_semantic_runtime":
            row["test"]["runtime_enforced_ucr"] = 1.0 - row["test"]["trigger_recall"]
    semantic["test"]["runtime_enforced_ucr"] = 1.0 - semantic["test"]["trigger_recall"]
    selected = next(
        row for row in result["results"] if row["condition"] == result["selected_condition"]
    )
    lexical_results = [
        row for row in result["results"] if row["condition"] != "transparent_semantic_runtime"
    ]
    best_dev = max(row["dev"]["accuracy"] for row in lexical_results)
    dev_tied = [row for row in lexical_results if row["dev"]["accuracy"] == best_dev]
    aligned = [
        row
        for row in lexical_results
        if row["condition"].startswith("tfidf_token_ngrams_native_")
        and row["condition"].endswith("_raw")
    ]
    aligned_rows = [
        {
            "condition": row["condition"],
            "dev_accuracy": row["dev"]["accuracy"],
            "test_accuracy": row["test"]["accuracy"],
            "trigger_recall": row["test"]["trigger_recall"],
            "false_activation_rate": row["test"]["false_activation_rate"],
        }
        for row in aligned
    ]
    aligned_best = max(
        (row["test_accuracy"] for row in aligned_rows),
        default=selected["test"]["accuracy"],
    )
    descriptive_best = max(row["test"]["accuracy"] for row in lexical_results)
    result["paper1_5_decision"] = {
        "selection_is_development_only": True,
        "selected_lexical_condition": selected["condition"],
        "selected_lexical_test_accuracy": selected["test"]["accuracy"],
        "development_best_accuracy": best_dev,
        "development_tie_count": len(dev_tied),
        "model_aligned_native_ngram_conditions": aligned_rows,
        "best_model_aligned_test_accuracy": aligned_best,
        "best_lexical_test_accuracy_descriptive_only": descriptive_best,
        "semantic_test_accuracy": semantic["test"]["accuracy"],
        "semantic_accuracy_advantage": semantic["test"]["accuracy"]
        - selected["test"]["accuracy"],
        "semantic_trigger_recall_advantage": semantic["test"]["trigger_recall"]
        - selected["test"]["trigger_recall"],
        "semantic_false_activation_advantage": selected["test"]["false_activation_rate"]
        - semantic["test"]["false_activation_rate"],
        "status": (
            "model_aligned_lexical_matches_or_wins"
            if aligned_best >= semantic["test"]["accuracy"]
            else "semantic_runtime_retains_advantage"
        ),
    }
    result["tokenizer_config_sha256"] = file_sha256(tokenizer_config_path)
    write_json(Path(output_dir) / "comparison.json", result)
    return result


def _benchmark_trigger_rows(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(row["example_id"]),
            "split": str(row["split"]),
            "text": str(row["question"]),
            "label": "RETRIEVE" if row["evidence_required"] else "NONE",
            "subgroup": str(row.get("retrieval_subclass", row.get("category", "unknown"))),
        }
        for row in rows
    ]


def _policy_trigger_rows(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(row["example_id"]),
            "split": str(row["split"]),
            "text": str(row["prompt"]),
            "label": "NONE" if row["target"] == "NO_RETRIEVAL" else "RETRIEVE",
            "subgroup": str(row["kind"]),
        }
        for row in rows
    ]


def _trigger_subgroups(
    rows: list[dict[str, Any]], predicted: list[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["subgroup"]].append(index)
    return {
        subgroup: score_labels(
            [rows[index]["label"] for index in indices],
            [predicted[index] for index in indices],
            negative_label="NONE",
        )
        for subgroup, indices in sorted(grouped.items())
    }


def run_natural_model(
    dataset_path: str | Path,
    source_path: str | Path,
    backend: ConfidenceBackend,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rows = list(_read_jsonl(dataset_path))
    store = ControlledFactStore.from_dict(read_json(source_path))
    spans = {
        str(row["example_id"]): backend.complete(_natural_prompt(row), seed=seed)
        for row in rows
        if row["split"] in {"dev", "test"}
    }
    development = [
        (spans[str(row["example_id"])].token_probabilities, bool(row["evidence_required"]))
        for row in rows
        if row["split"] == "dev"
    ]
    threshold = fit_confidence_threshold(development)
    predictions = []
    semantic_conditions = (
        "temporal_only",
        "source_only",
        "context_sufficiency_only",
        "combined_semantic",
    )
    for row in (item for item in rows if item["split"] == "test"):
        span = spans[str(row["example_id"])]
        forecast = extract_answer(span.text)
        features = semantic_features(str(row["question"]))
        confidence = bool(span.token_probabilities) and min(span.token_probabilities) < threshold
        base_triggers = {name: _feature_trigger(features, name) for name in semantic_conditions}
        base_triggers.update(
            {
                "confidence_only": confidence,
                "confidence_or_semantic": confidence or features["combined"],
                "confidence_and_semantic": confidence and features["combined"],
            }
        )
        conditions = {
            "no_retrieval": False,
            "upfront_retrieval": True,
            **base_triggers,
            "retrospective_verification": features["combined"],
            "evidence_advisory": features["combined"],
            "support_contract": features["combined"],
            "runtime_enforcement": features["combined"],
            "oracle": bool(row["evidence_required"]),
        }
        for condition, retrieved in conditions.items():
            result = None
            retrieval_time_ns = 0
            if retrieved:
                request = store.request(
                    example_id=str(row["example_id"]),
                    entity=str(row["entity"]),
                    attribute=str(row["attribute"]),
                    as_of=str(row["as_of"]),
                    forecast=forecast,
                    candidate_answer=forecast,
                )
                started = time.perf_counter_ns()
                result = store.execute(request)
                retrieval_time_ns = time.perf_counter_ns() - started
            values = tuple(str(value) for value in result.value["values"]) if result else ()
            final = forecast
            prompt_tokens = span.prompt_tokens
            generated_tokens = span.generated_tokens
            model_calls = span.model_calls
            wall_time_ns = span.wall_time_ns
            assisted = condition in {
                "upfront_retrieval",
                "evidence_advisory",
                "support_contract",
            }
            if assisted and result:
                evidence_span = backend.complete(
                    _natural_evidence_prompt(
                        row,
                        result.display,
                        support_contract=condition == "support_contract",
                    ),
                    seed=seed,
                )
                final = extract_answer(evidence_span.text)
                prompt_tokens += evidence_span.prompt_tokens
                generated_tokens += evidence_span.generated_tokens
                model_calls += evidence_span.model_calls
                wall_time_ns += evidence_span.wall_time_ns
            enforced = condition in {"runtime_enforcement", "oracle"} and retrieved
            if enforced:
                final = values[0] if len(values) == 1 else "abstain"
            required = bool(row["evidence_required"])
            predictions.append(
                {
                    "schema_version": "ccpu.paper1_5.natural_prediction.v1",
                    **row,
                    "model_id": backend.model_id,
                    "revision": backend.revision,
                    "condition": condition,
                    "forecast_text": span.text,
                    "forecast_answer": forecast,
                    "predicted_answer": final,
                    "correct": answers_equal(final, str(row["answer"])),
                    "confidence_threshold": threshold,
                    "minimum_token_probability": min(span.token_probabilities, default=1.0),
                    "confidence_low": confidence,
                    "semantic_features": features,
                    "retrieved": retrieved,
                    "runtime_enforced": enforced,
                    "retrospective_detected": condition == "retrospective_verification"
                    and bool(result)
                    and len(values) == 1
                    and not answers_equal(forecast, values[0]),
                    "evidence_override": retrieved and not answers_equal(final, forecast),
                    "abstained": final.casefold() == "abstain",
                    "unsupported_commitment": required
                    and final.casefold() != "abstain"
                    and not answers_equal(final, str(row["answer"])),
                    "authorized_commitment": required
                    and answers_equal(final, str(row["answer"])),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "model_calls": model_calls,
                    "wall_time_ns": wall_time_ns,
                    "retrieval_time_ns": retrieval_time_ns,
                    "evidence": result.value if result else None,
                    "evidence_status": str(result.value["status"]) if result else None,
                }
            )
    return predictions


def summarize_natural(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["condition"]))].append(row)
    cells = []
    for (model_id, condition), members in sorted(grouped.items()):
        gold = [bool(row["evidence_required"]) for row in members]
        triggered = [bool(row["retrieved"]) for row in members]
        required = [row for row in members if row["evidence_required"]]
        controls = [row for row in members if not row["evidence_required"]]
        classification = binary_classification(gold, triggered)
        cells.append(
            {
                "model_id": model_id,
                "condition": condition,
                "count": len(members),
                "accuracy": safe_mean(row["correct"] for row in members),
                "trigger_precision": classification["precision"],
                "trigger_recall": classification["recall"],
                "false_activation_rate": classification["false_intervention_rate"],
                "unsupported_commitment_rate": safe_mean(
                    row["unsupported_commitment"] for row in required
                ),
                "authorized_commitment_coverage": safe_mean(
                    row["authorized_commitment"] for row in required
                ),
                "unnecessary_retrieval_rate": safe_mean(row["retrieved"] for row in controls),
                "evidence_override_rate": safe_mean(
                    row["evidence_override"] for row in members if row["retrieved"]
                ),
                "abstention_rate": safe_mean(row["abstained"] for row in members),
                "retrospective_detection_rate": safe_mean(
                    row["retrospective_detected"] for row in required
                ),
                "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in members),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in members) / 1e6,
                "mean_retrieval_time_ms": safe_mean(
                    row["retrieval_time_ns"] for row in members
                ) / 1e6,
            }
        )
    return {
        "schema_version": "ccpu.paper1_5.natural_summary.v1",
        "prediction_count": len(rows),
        "by_model_condition": cells,
    }


def run_longform_opportunities(dataset_path: str | Path) -> dict[str, Any]:
    test = [row for row in _read_jsonl(dataset_path) if row["split"] == "test"]
    required = [row for row in test if row["evidence_required"]][:4]
    controls = [row for row in test if not row["evidence_required"]][:4]
    opportunities = []
    for document_index, (control, need) in enumerate(zip(controls, required, strict=True)):
        sequence = (control, need, need)
        evidence_cache: set[tuple[str, str]] = set()
        retrieved_count = 0
        for opportunity_index, row in enumerate(sequence):
            prefix = (
                "I will first summarize the supplied material. "
                if opportunity_index == 0
                else "The response is already under way. "
            )
            features = semantic_features(prefix + str(row["question"]))
            trigger = features["combined"]
            key = (str(row["entity"]), str(row["attribute"]))
            reused = trigger and key in evidence_cache
            retrieval = trigger and not reused
            if retrieval:
                retrieved_count += 1
                evidence_cache.add(key)
            required_now = bool(row["evidence_required"])
            opportunities.append(
                {
                    "schema_version": "ccpu.paper1_5.longform_opportunity.v1",
                    "document_id": f"natural-long-{document_index:02d}",
                    "opportunity_index": opportunity_index,
                    "example_id": row["example_id"],
                    "evidence_required": required_now,
                    "triggered_before_value": trigger,
                    "late_trigger": required_now,
                    "retrieval_call": retrieval,
                    "evidence_reused": reused,
                    "repeated_unnecessary_retrieval": retrieval and not required_now,
                    "advisory_unsupported_commitment": required_now,
                    "support_contract_unsupported_commitment": required_now and not trigger,
                    "support_contract_authorized_commitment": required_now and trigger,
                    "runtime_unsupported_commitment": required_now and not trigger,
                    "runtime_authorized_commitment": required_now and trigger,
                }
            )
    required_ops = [row for row in opportunities if row["evidence_required"]]
    return {
        "schema_version": "ccpu.paper1_5.longform_summary.v1",
        "document_count": len({row["document_id"] for row in opportunities}),
        "opportunity_count": len(opportunities),
        "opportunities": opportunities,
        "early_catch_rate": safe_mean(row["triggered_before_value"] for row in required_ops),
        "late_catch_rate": safe_mean(row["late_trigger"] for row in required_ops),
        "evidence_reuse_rate": safe_mean(row["evidence_reused"] for row in required_ops),
        "advisory_ucr": safe_mean(row["advisory_unsupported_commitment"] for row in required_ops),
        "support_contract_ucr": safe_mean(
            row["support_contract_unsupported_commitment"] for row in required_ops
        ),
        "support_contract_authorized_coverage": safe_mean(
            row["support_contract_authorized_commitment"] for row in required_ops
        ),
        "runtime_ucr": safe_mean(row["runtime_unsupported_commitment"] for row in required_ops),
        "runtime_authorized_coverage": safe_mean(
            row["runtime_authorized_commitment"] for row in required_ops
        ),
        "repeated_unnecessary_retrieval_rate": safe_mean(
            row["repeated_unnecessary_retrieval"] for row in opportunities
        ),
    }


def _natural_prompt(row: dict[str, Any]) -> str:
    return f"Answer with only the requested value and no explanation.\nQuestion: {row['question']}\nAnswer:"


def _natural_evidence_prompt(
    row: dict[str, Any], evidence: str, *, support_contract: bool
) -> str:
    policy = (
        "Use only a uniquely supported evidence value. If evidence is missing, ambiguous, "
        "or conflicting, answer ABSTAIN."
        if support_contract
        else "Use the evidence if it helps answer the question."
    )
    return (
        f"{policy}\nQuestion: {row['question']}\n"
        f"Evidence: {evidence}\nAnswer with only the requested value:"
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    import json

    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _normalize_question(question: str) -> str:
    normalized = question.casefold()
    normalized = re.sub(r"\bcase-[a-z]{2}\d+\b", "case-id", normalized)
    normalized = re.sub(r"\b(?:cust|ctx|hyp|fiction)-\d+\b", "value-id", normalized)
    normalized = re.sub(r"\br\d+\.\d+\b", "release-id", normalized)
    return normalized
