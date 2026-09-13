"""Comparable, coverage-aware result matrix for Paper 1 evaluations."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json


def _identity(row: dict[str, Any]) -> str:
    for key in ("example_id", "parent_example_id", "source_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("evaluation row has no stable identity")


def _metric(row: dict[str, Any], name: str) -> bool | None:
    metrics = row.get("metrics", {})
    value = metrics.get(name, row.get(name))
    return bool(value) if value is not None else None


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _summarize_cell(
    *,
    root: Path,
    eval_rows: list[dict[str, Any]],
    prediction_paths: list[str],
) -> dict[str, Any]:
    expected = {_identity(row): row for row in eval_rows}
    predictions: dict[str, dict[str, Any]] = {}
    present_paths: list[str] = []
    missing_paths: list[str] = []
    for value in prediction_paths:
        path = _resolve(root, value)
        if not path.exists():
            missing_paths.append(str(path))
            continue
        present_paths.append(str(path))
        for prediction in read_jsonl(path):
            identity = _identity(prediction)
            if identity in predictions:
                raise ValueError(f"duplicate prediction identity {identity}: {path}")
            predictions[identity] = prediction

    unexpected = sorted(set(predictions) - set(expected))
    if unexpected:
        raise ValueError(f"predictions outside frozen evaluation set: {unexpected[:5]}")
    hash_mismatches = []
    for identity, prediction in predictions.items():
        observed_hash = prediction.get("question_sha256")
        expected_hash = expected[identity].get("question_sha256")
        if observed_hash and expected_hash and observed_hash != expected_hash:
            hash_mismatches.append(identity)
    if hash_mismatches:
        raise ValueError(f"question hash mismatch: {hash_mismatches[:5]}")

    members = list(predictions.values())
    total = len(expected)
    observed = len(members)
    correct = sum(_metric(row, "final_answer_correct") is True for row in members)
    generated_tokens = [
        int(row["generated_tokens"])
        for row in members
        if row.get("generated_tokens") is not None
    ]
    status = "complete" if observed == total else "partial" if observed else "missing"
    result: dict[str, Any] = {
        "status": status,
        "expected_count": total,
        "prediction_count": observed,
        "missing_count": total - observed,
        "coverage": observed / total if total else 0.0,
        "counts": {"final_answer_correct": correct},
        "rates": {
            "final_answer_correct": correct / observed if observed else None,
        },
        "avg_generated_tokens": mean(generated_tokens) if generated_tokens else None,
        "prediction_paths": present_paths,
        "missing_prediction_paths": missing_paths,
    }
    for name in ("parse_valid", "lowerable_to_ccir", "type_valid", "executable"):
        available = [_metric(row, name) for row in members]
        values = [value for value in available if value is not None]
        result["counts"][name] = sum(value is True for value in values) if values else None
        result["rates"][name] = (
            sum(value is True for value in values) / len(values) if values else None
        )
    return result


def _cell_text(cell: dict[str, Any] | None) -> str:
    if not cell or cell["status"] == "missing":
        return "-"
    rate = cell["rates"]["final_answer_correct"]
    correct = cell["counts"]["final_answer_correct"]
    observed = cell["prediction_count"]
    expected = cell["expected_count"]
    suffix = " partial" if cell["status"] == "partial" else ""
    return f"{100 * rate:.1f}% ({correct}/{observed}{suffix}; N={expected})"


def _markdown(report: dict[str, Any]) -> str:
    columns = report["columns"]
    header = ["Dataset or intervention", *[column["label"] for column in columns]]
    lines = [
        "# Paper 1 Matched Evaluation Matrix",
        "",
        "Accuracy is final-answer correctness after the declared route. A partial cell is",
        "progress only and is not a publishable comparison. `-` means not run or not copied back.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", *["---:" for _ in columns]]) + " |",
    ]
    for row in report["rows"]:
        label = row["label"]
        if row.get("scientific_status") == "diagnostic_only":
            label += " [diagnostic only]"
        values = [_cell_text(row["cells"].get(column["id"])) for column in columns]
        lines.append("| " + " | ".join([label, *values]) + " |")
    notes = [row for row in report["rows"] if row.get("note")]
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- **{row['label']}:** {row['note']}" for row in notes)
    lines.extend(
        [
            "",
            "## Coverage Audit",
            "",
            "| Dataset or intervention | Eval SHA-256 | Complete cells | Partial cells |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in report["rows"]:
        statuses = [cell["status"] for cell in row["cells"].values()]
        lines.append(
            f"| {row['label']} | `{row['eval_sha256']}` | "
            f"{statuses.count('complete')} | {statuses.count('partial')} |"
        )
    return "\n".join(lines) + "\n"


def build_evaluation_matrix(
    *, manifest_path: str | Path, repository_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Build JSON and Markdown tables while enforcing frozen identity coverage."""

    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "ccpu.paper1.evaluation_matrix_manifest.v1":
        raise ValueError("unsupported evaluation-matrix manifest")
    root = Path(repository_root).resolve()
    columns = manifest["columns"]
    column_ids = [column["id"] for column in columns]
    if len(column_ids) != len(set(column_ids)):
        raise ValueError("duplicate matrix column id")

    report_rows = []
    row_ids: set[str] = set()
    for specification in manifest["rows"]:
        row_id = str(specification["id"])
        if row_id in row_ids:
            raise ValueError(f"duplicate matrix row id: {row_id}")
        row_ids.add(row_id)
        eval_path = _resolve(root, specification["eval_path"])
        eval_rows = read_jsonl(eval_path)
        identities = [_identity(row) for row in eval_rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate evaluation identity in {eval_path}")
        cells = {}
        for column in columns:
            cell_spec = specification.get("cells", {}).get(column["id"])
            paths = [] if cell_spec is None else list(cell_spec.get("prediction_paths", []))
            cells[column["id"]] = _summarize_cell(
                root=root,
                eval_rows=eval_rows,
                prediction_paths=paths,
            )
        report_rows.append(
            {
                "id": row_id,
                "label": specification["label"],
                "group": specification.get("group"),
                "scientific_status": specification.get("scientific_status", "primary"),
                "note": specification.get("note"),
                "eval_path": str(eval_path),
                "eval_sha256": file_sha256(eval_path),
                "eval_count": len(eval_rows),
                "cells": cells,
            }
        )
    report = {
        "schema_version": "ccpu.paper1.evaluation_matrix.v1",
        "manifest_path": str(Path(manifest_path).resolve()),
        "columns": columns,
        "rows": report_rows,
    }
    output = Path(output_dir)
    write_json(output / "matrix.json", report)
    markdown = _markdown(report)
    output.mkdir(parents=True, exist_ok=True)
    (output / "matrix.md").write_text(markdown, encoding="utf-8", newline="\n")
    return report


__all__ = ["build_evaluation_matrix"]
