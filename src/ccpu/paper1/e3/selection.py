"""Development-only semantic checkpoint ranking."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_RATES = (
    "parse_rate",
    "lowerable_rate",
    "type_valid_rate",
    "executable_rate",
    "semantic_return_rate",
    "semantic_state_rate",
    "dependency_rate",
    "operator_f1",
    "path_f1",
    "source_fact_f1",
    "answer_rate",
)


def semantic_checkpoint_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Rank safety first, semantic structure second, answer and loss last."""

    if row.get("split") != "dev":
        raise ValueError("semantic checkpoint selection is restricted to the development split")
    missing = [name for name in _RATES if name not in row]
    if missing:
        raise ValueError(f"checkpoint metrics are missing: {', '.join(missing)}")
    if int(row.get("examples", 0)) < 1:
        raise ValueError("checkpoint metrics require a positive example count")
    return (
        float(row["lowerable_rate"]),
        float(row["type_valid_rate"]),
        float(row["executable_rate"]),
        float(row["semantic_return_rate"]),
        float(row["semantic_state_rate"]),
        float(row["dependency_rate"]),
        float(row["operator_f1"]),
        float(row["path_f1"]),
        float(row["source_fact_f1"]),
        float(row["answer_rate"]),
        float(row["parse_rate"]),
        -float(row.get("dev_loss", float("inf"))),
        -float(row.get("epoch", 0)),
    )


def select_semantic_checkpoint(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select one checkpoint without consulting test behavior."""

    candidates = list(rows)
    if not candidates:
        raise ValueError("at least one development checkpoint is required")
    winner = max(candidates, key=semantic_checkpoint_key)
    return {
        "schema_version": "ccpu.paper1.semantic_checkpoint_selection.v1",
        "selection_policy": "lexicographic-safety-semantics-answer-loss",
        "candidate_count": len(candidates),
        "selected_checkpoint": winner.get("checkpoint"),
        "selected_epoch": winner.get("epoch"),
        "selected_key": list(semantic_checkpoint_key(winner)),
        "selected_metrics": winner,
    }
