"""Matched tokenizer, TF-IDF, and BM25 routing comparisons."""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import environment_manifest, write_json, write_jsonl
from ccpu.common.metrics import binary_classification

Tokenize = Callable[[str], list[str]]
_WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")
_WORD_PUNCT_PATTERN = re.compile(r"(?u)\w+|[^\w\s]")


@dataclass(frozen=True)
class NativeTokenizerSpec:
    label: str
    model_id: str
    revision: str


def current_word_tokens(text: str) -> list[str]:
    """Match scikit-learn's default lower-cased word tokenization."""
    return _WORD_PATTERN.findall(text.casefold())


def shared_nlp_tokens(text: str) -> list[str]:
    """Dependency-free Unicode word/punctuation tokenization shared by all models."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WORD_PUNCT_PATTERN.findall(normalized)


def character_word_boundary_ngrams(text: str) -> list[str]:
    """Approximate sklearn char_wb 3--5 grams for BM25's discrete terms."""
    normalized = " ".join(text.casefold().split())
    terms: list[str] = []
    for word in normalized.split(" "):
        padded = f" {word} "
        for size in range(3, 6):
            terms.extend(padded[index : index + size] for index in range(len(padded) - size + 1))
    return terms


def native_tokenizers(specs: Iterable[NativeTokenizerSpec]) -> dict[str, Tokenize]:
    specs = list(specs)
    if not specs:
        return {}
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Native-tokenizer comparisons require transformers") from error

    tokenizers: dict[str, Tokenize] = {}
    for spec in specs:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            local_files_only=True,
            use_fast=True,
        )
        raw = tokenizer.tokenize
        tokenizers[f"native_{spec.label}_raw"] = raw
        tokenizers[f"native_{spec.label}_normalized"] = (
            lambda text, tokenize=raw: [_normalize_piece(piece) for piece in tokenize(text)]
        )
    return tokenizers


def _normalize_piece(piece: str) -> str:
    normalized = unicodedata.normalize("NFKC", piece).casefold()
    normalized = normalized.lstrip("\u0120\u2581").replace("\u010a", "\n")
    return normalized or "<space>"


def token_ngrams(tokens: list[str]) -> list[str]:
    return tokens + [f"{left}\u241f{right}" for left, right in pairwise(tokens)]


class BM25ExemplarRouter:
    """Small inverted-index BM25 router with deterministic weighted voting."""

    def __init__(self, tokenize: Tokenize, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.tokenize = tokenize
        self.k1 = k1
        self.b = b
        self.labels: list[str] = []
        self.lengths: list[int] = []
        self.average_length = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.default_label = ""

    def fit(self, texts: list[str], labels: list[str]) -> BM25ExemplarRouter:
        self.labels = list(labels)
        self.default_label = Counter(labels).most_common(1)[0][0]
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, text in enumerate(texts):
            counts = Counter(self.tokenize(text))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        count = len(texts)
        self.postings = dict(postings)
        self.idf = {
            term: math.log(1.0 + (count - len(members) + 0.5) / (len(members) + 0.5))
            for term, members in postings.items()
        }
        return self

    def ranked(self, text: str, limit: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(self.tokenize(text)):
            for index, frequency in self.postings.get(term, ()):
                normalization = self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / max(self.average_length, 1.0)
                )
                scores[index] += self.idf[term] * frequency * (self.k1 + 1.0) / (
                    frequency + normalization
                )
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def class_scores(self, text: str, k: int) -> dict[str, float]:
        ranked = self.ranked(text, k)
        if not ranked:
            return {}
        by_label: dict[str, float] = defaultdict(float)
        for index, score in ranked:
            by_label[self.labels[index]] += score
        return dict(by_label)

    def predict(
        self,
        texts: list[str],
        k: int,
        *,
        negative_label: str,
        threshold: float,
    ) -> tuple[list[str], list[float]]:
        return _bm25_labels_from_scores(
            [self.class_scores(text, k) for text in texts], negative_label, threshold
        )

    def index_size_bytes(self) -> int:
        return sum(
            len(term.encode("utf-8")) + len(members) * 16
            for term, members in self.postings.items()
        )


def score_labels(
    gold: list[str], predicted: list[str], *, negative_label: str
) -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
    except ImportError as error:
        raise RuntimeError("Matched lexical comparisons require scikit-learn") from error
    trigger = binary_classification(
        [label != negative_label for label in gold],
        [label != negative_label for label in predicted],
    )
    classes = sorted(set(gold) | set(predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "trigger_precision": float(trigger["precision"]),
        "trigger_recall": float(trigger["recall"]),
        "false_activation_rate": float(trigger["false_intervention_rate"]),
        "labels": classes,
        "confusion_matrix": (
            [[len(gold)]]
            if len(classes) == 1
            else confusion_matrix(gold, predicted, labels=classes).tolist()
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(classes)
        },
    }


def run_matched_lexical_comparison(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    text_key: str,
    label_key: str,
    negative_label: str,
    tokenizer_specs: Iterable[NativeTokenizerSpec],
    output_dir: str | Path,
    source_hashes: dict[str, str],
    subgroup_key: str | None = None,
    include_prototypes: bool = False,
) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as error:
        raise RuntimeError("Matched lexical comparisons require scikit-learn") from error

    tokenizer_specs = list(tokenizer_specs)
    representations: dict[str, tuple[str, Any]] = {
        "current_word": ("tokens", current_word_tokens),
        "current_character": ("characters", "char_wb"),
        "shared_nlp": ("tokens", shared_nlp_tokens),
        **{
            name: ("tokens", tokenizer)
            for name, tokenizer in native_tokenizers(tokenizer_specs).items()
        },
    }
    train_text = [str(row[text_key]) for row in train]
    dev_text = [str(row[text_key]) for row in dev]
    test_text = [str(row[text_key]) for row in test]
    train_labels = [str(row[label_key]) for row in train]
    dev_labels = [str(row[label_key]) for row in dev]
    test_labels = [str(row[label_key]) for row in test]
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    tokenized_examples: list[dict[str, Any]] = []

    for representation, (kind, analyzer) in representations.items():
        tokenized_at = time.perf_counter_ns()
        test_terms = [
            character_word_boundary_ngrams(text) if kind == "characters" else analyzer(text)
            for text in test_text
        ]
        tokenization_ns = time.perf_counter_ns() - tokenized_at
        representation_stats = {
            "mean_sequence_length": sum(map(len, test_terms)) / max(len(test_terms), 1),
            "mean_unigram_bigram_count": sum(
                len(terms) + max(len(terms) - 1, 0) for terms in test_terms
            )
            / max(len(test_terms), 1),
            "mean_tokenization_latency_us": tokenization_ns / max(len(test), 1) / 1000,
            "unknown_piece_rate": _unknown_rate(test_terms),
        }
        tokenized_examples.extend(
            {
                "schema_version": "ccpu.tokenized_trigger_example.v1",
                "representation": representation,
                "split": "test",
                "example_id": str(row["example_id"]),
                "tokens": terms,
            }
            for row, terms in zip(test, test_terms, strict=True)
        )
        feature_ranges = (("character_ngrams", (3, 5)),) if kind == "characters" else (
            ("unigrams", (1, 1)),
            ("token_ngrams", (1, 2)),
        )
        ngram_vectorizer = None
        for feature_name, ngram_range in feature_ranges:
            if kind == "characters":
                vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
            else:
                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    tokenizer=analyzer,
                    preprocessor=None,
                    lowercase=False,
                    ngram_range=ngram_range,
                    token_pattern=None,
                )
            fitted = time.perf_counter_ns()
            pipeline = make_pipeline(
                vectorizer,
                LogisticRegression(max_iter=3000, random_state=0),
            )
            pipeline.fit(train_text, train_labels)
            fit_ns = time.perf_counter_ns() - fitted
            dev_predicted = [str(value) for value in pipeline.predict(dev_text)]
            dev_probabilities = pipeline.predict_proba(dev_text)
            test_started = time.perf_counter_ns()
            test_predicted = [str(value) for value in pipeline.predict(test_text)]
            test_probabilities = pipeline.predict_proba(test_text)
            test_ns = time.perf_counter_ns() - test_started
            fitted_vectorizer = pipeline[0]
            condition = f"tfidf_{feature_name}_{representation}"
            result = _result_row(
                condition,
                "tfidf_logistic",
                representation,
                dev,
                test,
                dev_labels,
                test_labels,
                dev_predicted,
                test_predicted,
                negative_label,
                subgroup_key,
                fit_ns,
                test_ns,
                {
                    **representation_stats,
                    "ngram_range": list(ngram_range),
                    "vocabulary_size": len(fitted_vectorizer.vocabulary_),
                    "index_size_bytes": _tfidf_size_bytes(pipeline),
                },
            )
            results.append(result)
            predictions.extend(
                _prediction_rows(condition, "dev", dev, dev_labels, dev_predicted, dev_probabilities.max(axis=1))
            )
            predictions.extend(
                _prediction_rows(condition, "test", test, test_labels, test_predicted, test_probabilities.max(axis=1))
            )
            if feature_name in {"token_ngrams", "character_ngrams"}:
                ngram_vectorizer = fitted_vectorizer

        if include_prototypes and ngram_vectorizer is not None:
            prototype = _run_prototype(
                ngram_vectorizer,
                train_text,
                train_labels,
                dev_text,
                dev_labels,
                test_text,
                negative_label,
            )
            condition = f"prototype_{representation}"
            result = _result_row(
                condition,
                "idf_class_prototype",
                representation,
                dev,
                test,
                dev_labels,
                test_labels,
                prototype["dev_labels"],
                prototype["test_labels"],
                negative_label,
                subgroup_key,
                prototype["fit_ns"],
                prototype["test_ns"],
                {
                    **representation_stats,
                    "selected_threshold_on_dev": prototype["threshold"],
                    "vocabulary_size": len(ngram_vectorizer.vocabulary_),
                    "index_size_bytes": prototype["index_size_bytes"],
                },
            )
            results.append(result)
            predictions.extend(_prediction_rows(condition, "dev", dev, dev_labels, prototype["dev_labels"], prototype["dev_confidence"]))
            predictions.extend(_prediction_rows(condition, "test", test, test_labels, prototype["test_labels"], prototype["test_confidence"]))

        bm25_tokenize = (
            character_word_boundary_ngrams
            if kind == "characters"
            else lambda text, tokenize=analyzer: token_ngrams(tokenize(text))
        )
        fitted = time.perf_counter_ns()
        bm25 = BM25ExemplarRouter(bm25_tokenize).fit(train_text, train_labels)
        bm25_fit_ns = time.perf_counter_ns() - fitted
        selected_k, selected_threshold = _tune_bm25(
            bm25, dev_text, dev_labels, negative_label
        )
        dev_predicted, dev_confidence = bm25.predict(
            dev_text,
            selected_k,
            negative_label=negative_label,
            threshold=selected_threshold,
        )
        test_started = time.perf_counter_ns()
        test_predicted, test_confidence = bm25.predict(
            test_text,
            selected_k,
            negative_label=negative_label,
            threshold=selected_threshold,
        )
        test_ns = time.perf_counter_ns() - test_started
        condition = f"bm25_{representation}"
        result = _result_row(
            condition,
            "bm25_exemplar",
            representation,
            dev,
            test,
            dev_labels,
            test_labels,
            dev_predicted,
            test_predicted,
            negative_label,
            subgroup_key,
            bm25_fit_ns,
            test_ns,
            {
                **representation_stats,
                "selected_k_on_dev": selected_k,
                "candidate_k": [1, 3, 5, 10],
                "selected_threshold_on_dev": selected_threshold,
                "uses_token_bigrams": kind != "characters",
                "vocabulary_size": len(bm25.postings),
                "index_size_bytes": bm25.index_size_bytes(),
            },
        )
        results.append(result)
        predictions.extend(_prediction_rows(condition, "dev", dev, dev_labels, dev_predicted, dev_confidence))
        predictions.extend(_prediction_rows(condition, "test", test, test_labels, test_predicted, test_confidence))

    best = min(results, key=lambda row: (-row["dev"]["accuracy"], row["condition"]))
    result = {
        "schema_version": "ccpu.matched_lexical_routing.v1",
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "source_sha256": source_hashes,
        "native_tokenizers": [spec.__dict__ for spec in tokenizer_specs],
        "bm25_tuning_split": "dev",
        "threshold_tuning_split": "dev",
        "selection_rule": "highest development accuracy; lexical condition breaks ties",
        "selected_condition": best["condition"],
        "results": results,
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "comparison.json", result)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "tokenized_test.jsonl", tokenized_examples)
    return result


def _result_row(
    condition: str,
    classifier: str,
    representation: str,
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    dev_gold: list[str],
    test_gold: list[str],
    dev_predicted: list[str],
    test_predicted: list[str],
    negative_label: str,
    subgroup_key: str | None,
    fit_ns: int,
    test_ns: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": condition,
        "classifier": classifier,
        "representation": representation,
        "fit_time_ms": fit_ns / 1_000_000,
        "mean_cpu_latency_us": test_ns / max(len(test_rows), 1) / 1000,
        "dev": score_labels(dev_gold, dev_predicted, negative_label=negative_label),
        "test": {
            **score_labels(test_gold, test_predicted, negative_label=negative_label),
            "by_subgroup": _subgroup_metrics(
                test_rows, test_gold, test_predicted, negative_label, subgroup_key
            ),
        },
        **metadata,
    }


def _subgroup_metrics(
    rows: list[dict[str, Any]],
    gold: list[str],
    predicted: list[str],
    negative_label: str,
    subgroup_key: str | None,
) -> dict[str, dict[str, Any]]:
    if not subgroup_key:
        return {}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row[subgroup_key])].append(index)
    return {
        group: score_labels(
            [gold[index] for index in indices],
            [predicted[index] for index in indices],
            negative_label=negative_label,
        )
        for group, indices in sorted(groups.items())
    }


def _tune_bm25(
    router: BM25ExemplarRouter,
    texts: list[str],
    gold: list[str],
    negative_label: str,
) -> tuple[int, float]:
    candidates = []
    for k in (1, 3, 5, 10):
        score_rows = [router.class_scores(text, k) for text in texts]
        for threshold_index in range(21):
            threshold = threshold_index / 20
            predicted, _ = _bm25_labels_from_scores(
                score_rows, negative_label, threshold
            )
            metrics = score_labels(gold, predicted, negative_label=negative_label)
            candidates.append(
                (metrics["accuracy"], metrics["macro_f1"], -metrics["false_activation_rate"], -k, threshold)
            )
    selected = max(candidates)
    return -int(selected[3]), float(selected[4])


def _bm25_labels_from_scores(
    score_rows: list[dict[str, float]],
    negative_label: str,
    threshold: float,
) -> tuple[list[str], list[float]]:
    labels = []
    confidence = []
    for scores in score_rows:
        total = sum(scores.values())
        nonnegative = [(label, score) for label, score in scores.items() if label != negative_label]
        if not nonnegative or not total:
            labels.append(negative_label)
            confidence.append(1.0)
            continue
        best_label, best_score = min(nonnegative, key=lambda item: (-item[1], item[0]))
        trigger_score = best_score / total
        if trigger_score >= threshold:
            labels.append(best_label)
            confidence.append(trigger_score)
        else:
            labels.append(negative_label)
            confidence.append(
                max(1.0 - trigger_score, scores.get(negative_label, 0.0) / total)
            )
    return labels, confidence


def _run_prototype(
    vectorizer: Any,
    train_text: list[str],
    train_labels: list[str],
    dev_text: list[str],
    dev_gold: list[str],
    test_text: list[str],
    negative_label: str,
) -> dict[str, Any]:
    import numpy as np

    fitted = time.perf_counter_ns()
    train_matrix = vectorizer.transform(train_text)
    classes = sorted(set(train_labels))
    prototypes = np.vstack(
        [np.asarray(train_matrix[[label == item for item in train_labels]].mean(axis=0)) for label in classes]
    )
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(norms, 1e-12)
    fit_ns = time.perf_counter_ns() - fitted
    dev_scores = vectorizer.transform(dev_text) @ prototypes.T
    dev_scores = np.asarray(dev_scores)
    threshold, dev_labels, dev_confidence = _tune_score_threshold(
        dev_scores, classes, dev_gold, negative_label
    )
    tested = time.perf_counter_ns()
    test_scores = np.asarray(vectorizer.transform(test_text) @ prototypes.T)
    test_labels, test_confidence = _labels_from_scores(
        test_scores, classes, negative_label, threshold
    )
    test_ns = time.perf_counter_ns() - tested
    return {
        "threshold": threshold,
        "dev_labels": dev_labels,
        "dev_confidence": dev_confidence,
        "test_labels": test_labels,
        "test_confidence": test_confidence,
        "fit_ns": fit_ns,
        "test_ns": test_ns,
        "index_size_bytes": int(prototypes.nbytes),
    }


def _tune_score_threshold(
    scores: Any,
    classes: list[str],
    gold: list[str],
    negative_label: str,
) -> tuple[float, list[str], list[float]]:
    candidates = []
    for threshold_index in range(21):
        threshold = threshold_index / 20
        predicted, confidence = _labels_from_scores(scores, classes, negative_label, threshold)
        metrics = score_labels(gold, predicted, negative_label=negative_label)
        candidates.append((metrics["accuracy"], metrics["macro_f1"], -metrics["false_activation_rate"], threshold, predicted, confidence))
    selected = max(candidates, key=lambda row: row[:4])
    return float(selected[3]), selected[4], selected[5]


def _labels_from_scores(
    scores: Any,
    classes: list[str],
    negative_label: str,
    threshold: float,
) -> tuple[list[str], list[float]]:
    nonnegative = [index for index, label in enumerate(classes) if label != negative_label]
    labels = []
    confidence = []
    for row in scores:
        best_index = max(nonnegative, key=lambda index: (float(row[index]), -index))
        score = float(row[best_index])
        if score >= threshold:
            labels.append(classes[best_index])
            confidence.append(score)
        else:
            labels.append(negative_label)
            confidence.append(1.0 - score)
    return labels, confidence


def _unknown_rate(documents: list[list[str]]) -> float:
    pieces = [piece for document in documents for piece in document]
    unknown = {"<unk>", "[unk]", "unk"}
    return sum(piece.casefold() in unknown for piece in pieces) / max(len(pieces), 1)


def _tfidf_size_bytes(pipeline: Any) -> int:
    vectorizer, classifier = pipeline
    vocabulary = sum(len(term.encode("utf-8")) + 8 for term in vectorizer.vocabulary_)
    return int(vocabulary + vectorizer.idf_.nbytes + classifier.coef_.nbytes + classifier.intercept_.nbytes)


def _prediction_rows(
    condition: str,
    split: str,
    rows: list[dict[str, Any]],
    gold: list[str],
    predicted: list[str],
    confidence: Iterable[float],
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "ccpu.matched_lexical_prediction.v1",
            "condition": condition,
            "split": split,
            "example_id": str(row["example_id"]),
            "gold_label": expected,
            "predicted_label": actual,
            "confidence": float(score),
        }
        for row, expected, actual, score in zip(rows, gold, predicted, confidence, strict=True)
    ]
