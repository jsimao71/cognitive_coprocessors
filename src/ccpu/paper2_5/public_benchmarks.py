"""Pinned TAT-QA selection and oracle composition diagnostics."""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    environment_manifest,
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.lexical_routing import (
    BM25ExemplarRouter,
    character_word_boundary_ngrams,
    current_word_tokens,
)
from ccpu.common.public_benchmarks import stratified_select


def _source_path(config: dict[str, Any], cache_root: str | Path) -> Path:
    return Path(cache_root) / "tatqa" / str(config["source"]["file"])


def _load_documents(config: dict[str, Any], cache_root: str | Path) -> list[dict[str, Any]]:
    source = config["source"]
    path = _source_path(config, cache_root)
    if file_sha256(path) != source["file_sha256"]:
        raise ValueError("TAT-QA source checksum mismatch")
    documents = json.loads(path.read_text(encoding="utf-8"))
    if len(documents) != int(source["expected_documents"]):
        raise ValueError("TAT-QA document count mismatch")
    question_count = sum(len(document["questions"]) for document in documents)
    if question_count != int(source["expected_questions"]):
        raise ValueError("TAT-QA question count mismatch")
    return documents


def _content_sha(document: dict[str, Any], question: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json({"document": document, "question": question}).encode("utf-8")
    ).hexdigest()


def freeze_tatqa_subset(
    config_path: str | Path, cache_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != "ccpu.paper2_5.public_tatqa_config.v1":
        raise ValueError("unsupported Paper 2.5 TAT-QA config schema")
    seed = int(config["selection_seed"])
    records = []
    source_row = 0
    for document_index, document in enumerate(_load_documents(config, cache_root)):
        for question_index, question in enumerate(document["questions"]):
            example_id = str(question["uid"])
            content_sha = _content_sha(document, question)
            stratum = "|".join(
                str(question[key])
                for key in ("answer_type", "answer_from", "scale", "req_comparison")
            )
            records.append(
                {
                    "benchmark": "tatqa",
                    "example_id": example_id,
                    "source_row": source_row,
                    "document_index": document_index,
                    "question_index": question_index,
                    "content_sha256": content_sha,
                    "selection_key": hashlib.sha256(
                        f"{seed}:{example_id}:{content_sha}".encode("ascii")
                    ).hexdigest(),
                    "difficulty": 2 if question["answer_from"] == "table-text" else 1,
                    "difficulty_stratum": stratum,
                    "answer_type": str(question["answer_type"]),
                    "answer_from": str(question["answer_from"]),
                    "scale": str(question["scale"]),
                    "req_comparison": bool(question["req_comparison"]),
                    "has_derivation": bool(str(question.get("derivation", "")).strip()),
                }
            )
            source_row += 1
    selected = stratified_select(records, int(config["max_rows"]), seed)
    output = Path(output_dir)
    selection_path = write_jsonl(output / "selection.jsonl", selected)
    counts = {}
    for key in ("answer_type", "answer_from", "scale", "req_comparison", "has_derivation"):
        values: dict[str, int] = defaultdict(int)
        for row in selected:
            values[str(row[key])] += 1
        counts[key] = dict(sorted(values.items()))
    manifest = {
        "schema_version": "ccpu.paper2_5.public_tatqa_manifest.v1",
        "config_fingerprint": fingerprint(config),
        "record_count": len(selected),
        "selection_sha256": file_sha256(selection_path),
        "counts": counts,
        "source": config["source"],
        "redistribution": "IDs, source coordinates, strata, and content hashes only",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _materialize(
    config_path: str | Path, cache_root: str | Path, selection_path: str | Path
) -> list[dict[str, Any]]:
    config = read_json(config_path)
    documents = _load_documents(config, cache_root)
    selected = read_jsonl(selection_path)
    rows = []
    for item in selected:
        document = documents[int(item["document_index"])]
        question = document["questions"][int(item["question_index"])]
        if question["uid"] != item["example_id"]:
            raise ValueError(f"TAT-QA ID changed for {item['example_id']}")
        if _content_sha(document, question) != item["content_sha256"]:
            raise ValueError(f"TAT-QA content changed for {item['example_id']}")
        rows.append({**item, "document": document, "question": question})
    return rows


_ALLOWED_BINARY = {ast.Add, ast.Sub, ast.Mult, ast.Div}
_ALLOWED_UNARY = {ast.UAdd, ast.USub}


def _decimal_expression(raw: str) -> Decimal:
    expression = re.sub(
        r"(?P<number>(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))\s*%",
        r"(\g<number>/100)",
        raw.strip(),
    )
    expression = expression.replace(",", "").replace("$", "")
    expression = expression.replace("[", "(").replace("]", ")")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            token = ast.get_source_segment(expression, node)
            if token is None:
                raise ValueError("missing numeric source token")
            return Decimal(token)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError(f"unsupported derivation syntax: {type(node).__name__}")

    with localcontext() as context:
        context.prec = 40
        return evaluate(tree.body)


def _numeric_answer(question: dict[str, Any]) -> Decimal:
    answer = question["answer"]
    if isinstance(answer, list):
        if len(answer) != 1:
            raise ValueError("answer is not a scalar")
        answer = answer[0]
    cleaned = re.sub(r"[^0-9.+-]", "", str(answer).replace(",", ""))
    if not cleaned:
        raise ValueError("answer has no numeric scalar")
    return Decimal(cleaned)


def _score_derivation(question: dict[str, Any]) -> dict[str, Any]:
    if question["answer_type"] != "arithmetic" or not str(question["derivation"]).strip():
        return {
            "oracle_compute_available": False,
            "oracle_compute_exact": None,
            "derivation_failure": "no arithmetic derivation",
        }
    try:
        computed = _decimal_expression(str(question["derivation"]))
        expected = _numeric_answer(question)
        candidates = (computed, computed * 100) if question["scale"] == "percent" else (computed,)
        error = min(abs(candidate - expected) for candidate in candidates)
        exact = error <= Decimal("0.011")
        return {
            "oracle_compute_available": True,
            "oracle_compute_exact": exact,
            "derivation_failure": None if exact else f"numeric mismatch {error}",
        }
    except (InvalidOperation, SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        return {
            "oracle_compute_available": False,
            "oracle_compute_exact": None,
            "derivation_failure": str(error),
        }


def analyze_tatqa_composition(
    config_path: str | Path,
    cache_root: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    rows = []
    for source in _materialize(config_path, cache_root, selection_path):
        question = source["question"]
        answer_type = str(question["answer_type"])
        rows.append(
            {
                "schema_version": "ccpu.paper2_5.public_tatqa_prediction.v1",
                "example_id": source["example_id"],
                "answer_type": answer_type,
                "answer_from": source["answer_from"],
                "scale": source["scale"],
                "req_comparison": source["req_comparison"],
                "retrieval_required": True,
                "compute_required": answer_type in {"arithmetic", "count"},
                "composition_depth": 2 if answer_type in {"arithmetic", "count"} else 1,
                "source_native_adapter_available": False,
                **_score_derivation(question),
            }
        )
    computable = [row for row in rows if row["oracle_compute_available"]]
    by_type = {}
    for answer_type in sorted({str(row["answer_type"]) for row in rows}):
        members = [row for row in rows if row["answer_type"] == answer_type]
        available = [row for row in members if row["oracle_compute_available"]]
        by_type[answer_type] = {
            "count": len(members),
            "compute_required_rate": sum(row["compute_required"] for row in members)
            / len(members),
            "oracle_compute_coverage": len(available) / len(members),
            "oracle_compute_exact_rate": (
                sum(row["oracle_compute_exact"] for row in available) / len(available)
                if available
                else None
            ),
        }
    output = Path(output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", rows)
    summary = {
        "schema_version": "ccpu.paper2_5.public_tatqa_analysis.v1",
        "record_count": len(rows),
        "selection_sha256": file_sha256(selection_path),
        "predictions_sha256": file_sha256(predictions_path),
        "compute_required_rate": sum(row["compute_required"] for row in rows) / len(rows),
        "table_text_rate": sum(row["answer_from"] == "table-text" for row in rows) / len(rows),
        "oracle_compute_coverage": len(computable) / len(rows),
        "oracle_compute_exact_rate": (
            sum(row["oracle_compute_exact"] for row in computable) / len(computable)
            if computable
            else None
        ),
        "source_native_adapter_coverage": 0.0,
        "by_answer_type": by_type,
        "interpretation": {
            "status": "adapter_gap",
            "claim": "gold arithmetic is executable, but retrieval/extraction is not integrated",
        },
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(output / "summary.json", summary)
    return summary


def _table_rows(document: dict[str, Any], *, structured: bool) -> list[tuple[str, str]]:
    table = document["table"]["table"]
    width = max((len(row) for row in table), default=0)
    headers = []
    for column in range(width):
        values = [str(row[column]).strip() for row in table[:2] if column < len(row)]
        headers.append(" ".join(value for value in values if value))
    rows = []
    for index, raw_row in enumerate(table):
        cells = [str(cell).strip() for cell in raw_row]
        if structured:
            row_name = cells[0] if cells else ""
            fields = [f"row {row_name}"]
            fields.extend(
                f"column {headers[column]} value {value}"
                for column, value in enumerate(cells)
                if value
            )
            text = " ; ".join(fields)
        else:
            text = " | ".join(cells)
        rows.append((f"table:{index}", text))
    return rows


def _document_chunks(
    document: dict[str, Any], *, structured: bool
) -> list[tuple[str, str]]:
    rows = _table_rows(document, structured=structured)
    paragraphs = [
        (f"paragraph:{paragraph['order']}", str(paragraph["text"]))
        for paragraph in document["paragraphs"]
    ]
    return rows + paragraphs


def _normalized_text(value: Any) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9.%+-]+", " ", str(value).casefold().replace(",", "")).split()
    )


def _gold_evidence_labels(
    document: dict[str, Any], question: dict[str, Any]
) -> set[str]:
    labels = {
        f"paragraph:{order}"
        for order in question.get("rel_paragraphs", [])
        if question["answer_from"] in {"text", "table-text"}
    }
    if question["answer_from"] not in {"table", "table-text"}:
        return labels

    targets: set[str] = set()
    if question["answer_type"] in {"span", "multi-span"}:
        answers = question["answer"] if isinstance(question["answer"], list) else [question["answer"]]
        targets.update(_normalized_text(answer) for answer in answers)
    if question["answer_type"] == "arithmetic":
        targets.update(
            _normalized_text(value)
            for value in re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?%?", question["derivation"])
        )
    targets.discard("")
    for label, text in _table_rows(document, structured=False):
        normalized = _normalized_text(text)
        padded = f" {normalized} "
        if any(f" {target} " in padded for target in targets):
            labels.add(label)
    return labels


def _rank_chunks(chunks: list[tuple[str, str]], query: str, limit: int) -> dict[str, list[str]]:
    labels = [label for label, _ in chunks]
    texts = [text for _, text in chunks]
    word = BM25ExemplarRouter(current_word_tokens).fit(texts, labels)
    char = BM25ExemplarRouter(character_word_boundary_ngrams).fit(texts, labels)
    word_rank = [labels[index] for index, _ in word.ranked(query, limit)]
    char_rank = [labels[index] for index, _ in char.ranked(query, limit)]
    scores: dict[str, float] = defaultdict(float)
    for ranking in (word_rank, char_rank):
        for position, label in enumerate(ranking, start=1):
            scores[label] += 1.0 / (60 + position)
    hybrid = sorted(scores, key=lambda label: (-scores[label], label))[:limit]
    return {"bm25_word": word_rank, "bm25_char": char_rank, "hybrid": hybrid}


def analyze_tatqa_retrieval(
    config_path: str | Path,
    cache_root: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    predictions = []
    for source in _materialize(config_path, cache_root, selection_path):
        question = source["question"]
        gold = _gold_evidence_labels(source["document"], question)
        universal = _rank_chunks(
            _document_chunks(source["document"], structured=False),
            str(question["question"]),
            limit,
        )
        structured = _rank_chunks(
            _document_chunks(source["document"], structured=True),
            str(question["question"]),
            limit,
        )["hybrid"]
        conditions = {**universal, "structured_hybrid": structured}
        predictions.append(
            {
                "schema_version": "ccpu.paper2_5.public_tatqa_retrieval.v1",
                "example_id": source["example_id"],
                "answer_type": source["answer_type"],
                "answer_from": source["answer_from"],
                "gold_evidence_labels": sorted(gold),
                "evaluable": bool(gold),
                "retrieved": conditions,
                "scores": {
                    condition: {
                        "recall": len(gold.intersection(labels)) / len(gold) if gold else None,
                        "complete": gold.issubset(labels) if gold else None,
                    }
                    for condition, labels in conditions.items()
                },
            }
        )
    evaluable = [row for row in predictions if row["evaluable"]]
    conditions = ("bm25_word", "bm25_char", "hybrid", "structured_hybrid")

    def summarize(members: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            condition: {
                "evaluable_count": len(members),
                "mean_evidence_recall_at_k": sum(
                    row["scores"][condition]["recall"] for row in members
                )
                / len(members),
                "complete_evidence_rate_at_k": sum(
                    row["scores"][condition]["complete"] for row in members
                )
                / len(members),
            }
            for condition in conditions
        }

    differences = [
        row["scores"]["structured_hybrid"]["recall"]
        - row["scores"]["hybrid"]["recall"]
        for row in evaluable
    ]
    generator = random.Random(22971)
    bootstrap = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(10_000)
    )
    structured_complete = [row["scores"]["structured_hybrid"]["complete"] for row in evaluable]
    hybrid_complete = [row["scores"]["hybrid"]["complete"] for row in evaluable]
    output = Path(output_dir)
    prediction_path = write_jsonl(output / "retrieval_predictions.jsonl", predictions)
    summary = {
        "schema_version": "ccpu.paper2_5.public_tatqa_retrieval_summary.v1",
        "record_count": len(predictions),
        "evaluable_count": len(evaluable),
        "top_k": limit,
        "selection_sha256": file_sha256(selection_path),
        "predictions_sha256": file_sha256(prediction_path),
        "by_condition": summarize(evaluable),
        "by_answer_from": {
            value: summarize([row for row in evaluable if row["answer_from"] == value])
            for value in sorted({row["answer_from"] for row in evaluable})
        },
        "by_answer_type": {
            value: summarize([row for row in evaluable if row["answer_type"] == value])
            for value in sorted({row["answer_type"] for row in evaluable})
        },
        "paired_structured_minus_flat_hybrid": {
            "mean_recall_difference": sum(differences) / len(differences),
            "bootstrap_95_ci": [bootstrap[249], bootstrap[9749]],
            "complete_wins": sum(
                left and not right for left, right in zip(structured_complete, hybrid_complete)
            ),
            "complete_losses": sum(
                right and not left for left, right in zip(structured_complete, hybrid_complete)
            ),
            "complete_ties": sum(
                left == right for left, right in zip(structured_complete, hybrid_complete)
            ),
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 22971,
        },
        "claim_boundary": {
            "ranking_uses_gold": False,
            "scoring_uses_gold": True,
            "operation_is_oracle": True,
            "final_answer_evaluated": False,
        },
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    write_json(output / "retrieval_summary.json", summary)
    return summary
