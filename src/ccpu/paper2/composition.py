"""Bounded two-engine compositions and persistent-state reuse diagnostics."""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

from ccpu.common.metrics import safe_mean

from .runtime import HeterogeneousRuntime


def run_compositions(count_per_family: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count_per_family < 1:
        raise ValueError("composition count must be positive")
    rows = []
    for index in range(count_per_family):
        runtime = HeterogeneousRuntime(max_state_items=4096)
        base = date(2035 + index % 20, 1 + index % 12, 1 + index % 24)
        days = 30 + index * 3
        end = base + timedelta(days=days)
        rate = 7 + index
        started = time.perf_counter_ns()
        first = runtime.execute_event(
            f"```date\ndiff {base.isoformat()} {end.isoformat()}\n```",
            event_id=f"date-calc-{index}:date",
        )
        first_state = runtime.state.items[-1].state_id if runtime.state.items else ""
        runtime.state.add(
            "composition_input",
            {"family": "date_to_calculator", "value": first.display if first else None},
            dependencies=(first_state,) if first_state else (),
            provenance={"composition_id": f"date-calc-{index}"},
        )
        second = runtime.execute_event(
            f"```calculator\n{first.display if first else 0} * {rate}\n```",
            event_id=f"date-calc-{index}:calculator",
        )
        rows.append(
            {
                "schema_version": "ccpu.paper2.composition_prediction.v1",
                "composition_id": f"date-calc-{index}",
                "family": "date_to_calculator",
                "stage1_ok": bool(first and first.ok and first.display == str(days)),
                "stage2_ok": bool(second and second.ok),
                "answer": second.display if second else None,
                "gold_answer": str(days * rate),
                "correct": bool(second and second.display == str(days * rate)),
                "engine_calls": 2,
                "state_items": len(runtime.state.items),
                "state_dependency_recorded": bool(
                    runtime.state.by_kind("composition_input")[0].dependencies
                ),
                "wall_time_ns": time.perf_counter_ns() - started,
            }
        )

    for index in range(count_per_family):
        runtime = HeterogeneousRuntime(max_state_items=4096)
        entity = f"compose_entity_{index:04d}"
        middle = f"compose_middle_{index:04d}"
        parent = f"compose_parent_{index:04d}"
        started = time.perf_counter_ns()
        first = runtime.execute_event(
            f"```graph\nisa {entity} {middle}\nisa {middle} {parent}\n"
            f"query isa {entity} {parent}\n```",
            event_id=f"graph-datalog-{index}:graph",
        )
        first_state = runtime.state.items[-1].state_id if runtime.state.items else ""
        runtime.state.add(
            "composition_input",
            {"family": "graph_to_datalog", "entity": entity, "eligible": bool(first and first.ok)},
            dependencies=(first_state,) if first_state else (),
            provenance={"composition_id": f"graph-datalog-{index}"},
        )
        second = runtime.execute_event(
            f"```datalog\nfact eligible({entity})\nquery eligible({entity})\n```",
            event_id=f"graph-datalog-{index}:datalog",
        )
        reuse = runtime.execute_event(
            f"```graph\nquery isa {entity} {parent}\n```",
            event_id=f"graph-datalog-{index}:reuse",
        )
        rows.append(
            {
                "schema_version": "ccpu.paper2.composition_prediction.v1",
                "composition_id": f"graph-datalog-{index}",
                "family": "graph_to_datalog",
                "stage1_ok": bool(first and first.ok and first.display == "true"),
                "stage2_ok": bool(second and second.ok),
                "answer": second.display if second else None,
                "gold_answer": "true",
                "correct": bool(second and second.display == "true"),
                "engine_calls": 3,
                "state_items": len(runtime.state.items),
                "state_dependency_recorded": bool(
                    runtime.state.by_kind("composition_input")[0].dependencies
                ),
                "persistent_state_reused": bool(reuse and reuse.display == "true"),
                "wall_time_ns": time.perf_counter_ns() - started,
            }
        )

    by_family = []
    for family in ("date_to_calculator", "graph_to_datalog"):
        group = [row for row in rows if row["family"] == family]
        by_family.append(
            {
                "family": family,
                "count": len(group),
                "stage1_rate": safe_mean(row["stage1_ok"] for row in group),
                "stage2_rate": safe_mean(row["stage2_ok"] for row in group),
                "accuracy": safe_mean(row["correct"] for row in group),
                "state_dependency_rate": safe_mean(
                    row["state_dependency_recorded"] for row in group
                ),
                "state_reuse_rate": safe_mean(
                    row.get("persistent_state_reused", True) for row in group
                ),
                "mean_state_items": safe_mean(row["state_items"] for row in group),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in group) / 1e6,
            }
        )
    summary = {
        "schema_version": "ccpu.paper2.composition_evaluation.v1",
        "empirical": False,
        "count": len(rows),
        "by_family": by_family,
    }
    return rows, summary
