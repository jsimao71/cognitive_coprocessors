"""Non-empirical Paper 2 protocol simulation."""

from __future__ import annotations

import time
from typing import Any

from .dataset import MixedExample
from .runtime import HeterogeneousRuntime

CONDITIONS = (
    "no_engine",
    "explicit_tools",
    "single_calculator",
    "single_horn",
    "single_graph",
    "heterogeneous",
    "oracle_selection",
)

_ENABLED = {
    "explicit_tools": {"calculator", "horn", "frame_graph"},
    "single_calculator": {"calculator"},
    "single_horn": {"horn"},
    "single_graph": {"frame_graph"},
    "heterogeneous": {"calculator", "horn", "frame_graph"},
    "oracle_selection": {"calculator", "horn", "frame_graph"},
}


def run_scripted(examples: list[MixedExample]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtimes = {
        condition: HeterogeneousRuntime(enabled_engines=enabled, max_state_items=4096)
        for condition, enabled in _ENABLED.items()
    }
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for example in examples:
        for condition in CONDITIONS:
            started = time.perf_counter_ns()
            runtime = runtimes.get(condition)
            trace_start = len(runtime.trace) if runtime else 0
            result = (
                runtime.execute_event(example.event, event_id=f"{condition}:{example.example_id}")
                if runtime
                else None
            )
            elapsed = time.perf_counter_ns() - started
            predicted = result.display if result and result.ok else None
            correct = (
                predicted is None if not example.should_trigger else predicted == example.answer
            )
            predictions.append(
                {
                    "schema_version": "ccpu.paper2.prediction.v1",
                    "example_id": example.example_id,
                    "condition": condition,
                    "engine": example.engine,
                    "depth": example.depth,
                    "distractors": example.distractors,
                    "should_trigger": example.should_trigger,
                    "predicted_answer": predicted,
                    "gold_answer": example.answer,
                    "correct": correct,
                    "intervened": result is not None,
                    "engine_ok": result.ok if result else None,
                    "state_items": len(runtime.state.items) if runtime else 0,
                    "wall_time_ns": elapsed,
                    "empirical": False,
                }
            )
            if runtime:
                for row in runtime.trace[trace_start:]:
                    traces.append(
                        {
                            "schema_version": "ccpu.paper2.trace.v1",
                            "condition": condition,
                            "example_id": example.example_id,
                            **row,
                        }
                    )
    return predictions, traces
