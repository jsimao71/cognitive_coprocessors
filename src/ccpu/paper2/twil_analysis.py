"""Merge TwIL comparison runs and produce substrate/cost plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_json
from ccpu.common.metrics import safe_mean

from .twil_experiment import summarize_twil


def analyze_twil_runs(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_json(config_path)
    rows = []
    sources = []
    for relative in config["prediction_files"]:
        path = (config_path.parent / relative).resolve()
        rows.extend(read_jsonl(path))
        sources.append({"path": str(path), "sha256": file_sha256(path)})
    reuse_path = (config_path.parent / config["reuse_file"]).resolve()
    reuse = read_jsonl(reuse_path)
    sources.append({"path": str(reuse_path), "sha256": file_sha256(reuse_path)})
    summary = summarize_twil(rows)
    diagnostics = _diagnostics(rows)
    result = {
        **summary,
        "schema_version": "ccpu.paper2.twil_analysis.v1",
        "sources": sources,
        "reuse": reuse,
        "diagnostics": diagnostics,
        "headline_evidence": {
            "status": "diagnostic_only",
            "rankable": False,
            "reason": (
                "The run diagnoses typed-interface behavior but cannot rank internal reasoning "
                "because the logic suite is all-positive and the decoding protocol is not "
                "TwIL-aligned."
            ),
            "blockers": [
                "All 16 Horn/graph neural-only targets are true, so a constant predictor is perfect.",
                "The run forces no-thinking greedy decoding at 160 tokens; TwIL reports thinking-mode greedy evaluation at 2048 tokens with retry at 4096.",
                "Only four semantic controls are included and no FOL, Lean, or rule-induction lane is reproduced.",
                "Hybrid final answers use runtime displays directly; neural result integration is not tested.",
            ],
            "interface_finding": (
                "Under the matched low-latency ICL interface, TwIL improves over SmolLM3 on only "
                "one of 22 exact items; valid IR executes exactly, but formalization dominates "
                "end-to-end failure."
            ),
        },
        "config_sha256": file_sha256(config_path),
        "limitations": [
            "The local comparison is a bounded 26-item diagnostic, not a benchmark-wide TwIL reproduction.",
            "No local TwIL training cost is inferred; only released weights are evaluated.",
            "Hybrid integration uses the exact typed runtime result as the final answer.",
            "SMT and algebra remain outside the implemented bounded runtime gate.",
        ],
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "analysis.json", result)
    _plot_scaling(rows, output_dir / "accuracy_cost_scaling.png")
    _plot_reuse(reuse, output_dir / "persistent_state_amortization.png")
    _plot_frontier(rows, output_dir / "capability_pareto.png")
    return result


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["model_id"] != "oracle":
            grouped[(str(row["model_id"]), str(row["condition"]))].append(row)
    conditions = []
    for (model_id, condition), members in sorted(grouped.items()):
        exact = [row for row in members if row["should_trigger"]]
        failures: dict[str, int] = defaultdict(int)
        for row in members:
            failures[str(row.get("failure_type", "unlabeled"))] += 1
        conditions.append(
            {
                "model_id": model_id,
                "condition": condition,
                "count": len(members),
                "exact_count": len(exact),
                "exact_accuracy": safe_mean(row["final_correct"] for row in exact),
                "overall_accuracy": safe_mean(row["final_correct"] for row in members),
                "truncation_rate": safe_mean(row["truncated_at_budget"] for row in members),
                "mean_generated_tokens": safe_mean(row["generated_tokens"] for row in members),
                "mean_accelerator_time_ms": safe_mean(
                    row["accelerator_time_ns"] for row in members
                ) / 1e6,
                "failure_counts": dict(sorted(failures.items())),
            }
        )
    model_conditions = {
        ("twil" if "twil" in model.casefold() else "smollm3", condition): {
            str(row["example_id"]): row for row in members
        }
        for (model, condition), members in grouped.items()
    }
    paired = []
    for condition in ("neural", "hybrid"):
        smol = model_conditions.get(("smollm3", condition), {})
        twil = model_conditions.get(("twil", condition), {})
        for example_id in sorted(set(smol) & set(twil)):
            if (
                smol[example_id]["final_correct"] != twil[example_id]["final_correct"]
                or smol[example_id].get("failure_type") != twil[example_id].get("failure_type")
            ):
                paired.append(
                    {
                        "condition": condition,
                        "example_id": example_id,
                        "smollm3_correct": smol[example_id]["final_correct"],
                        "smollm3_failure": smol[example_id].get("failure_type"),
                        "twil_correct": twil[example_id]["final_correct"],
                        "twil_failure": twil[example_id].get("failure_type"),
                    }
                )
    return {"by_condition": conditions, "paired_differences": paired}


def _style() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("TwIL analysis plots require matplotlib") from error
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    return plt


def _logic_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] in {"datalog", "graph"}:
            grouped[(str(row["model_id"]), str(row["condition"]), int(row["depth"]))].append(row)
    return [
        {
            "model_id": model,
            "condition": condition,
            "depth": depth,
            "accuracy": safe_mean(row["final_correct"] for row in members),
            "tokens": safe_mean(row["generated_tokens"] for row in members),
            "accelerator_ms": safe_mean(row["accelerator_time_ns"] for row in members) / 1e6,
            "engine_ms": safe_mean(row["engine_time_ns"] for row in members) / 1e6,
        }
        for (model, condition, depth), members in sorted(grouped.items())
    ]


def _label(model_id: str, condition: str) -> str:
    model = "TwIL" if "twil" in model_id.casefold() else "SmolLM3" if "smollm3" in model_id.casefold() else model_id
    return f"{model} {condition}"


def _plot_scaling(rows: list[dict[str, Any]], output: Path) -> None:
    plt = _style()
    cells = _logic_cells(rows)
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    measures = (
        ("accuracy", "final accuracy", (0, 1.04)),
        ("tokens", "generated tokens", None),
        ("accelerator_ms", "XPU time / item (ms)", None),
        ("engine_ms", "CPU engine time / item (ms)", None),
    )
    groups = sorted({(row["model_id"], row["condition"]) for row in cells})
    for axis, (measure, ylabel, ylim) in zip(axes.flat, measures, strict=True):
        for model_id, condition in groups:
            points = [
                row for row in cells
                if row["model_id"] == model_id and row["condition"] == condition
            ]
            axis.plot(
                [row["depth"] for row in points],
                [row[measure] for row in points],
                marker="o",
                label=_label(model_id, condition),
            )
        axis.set(xlabel="derivation depth", ylabel=ylabel)
        if ylim:
            axis.set_ylim(*ylim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    figure.suptitle("Reasoning in weights versus exact coprocessors", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_reuse(rows: list[dict[str, Any]], output: Path) -> None:
    plt = _style()
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    for family in sorted({str(row["family"]) for row in rows}):
        points = [row for row in rows if row["family"] == family]
        axis.plot(
            [row["query_count"] for row in points],
            [row["persistent_amortized_ms"] for row in points],
            marker="o",
            label=f"{family} persistent",
        )
        axis.plot(
            [row["query_count"] for row in points],
            [row["fresh_amortized_ms"] for row in points],
            linestyle="--",
            label=f"{family} fresh",
        )
    axis.set_xscale("log")
    axis.set(xlabel="queries over one world", ylabel="amortized CPU time / query (ms)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_frontier(rows: list[dict[str, Any]], output: Path) -> None:
    plt = _style()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["condition"]))].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for (model_id, condition), members in sorted(grouped.items()):
        label = _label(model_id, condition)
        exact_families = sorted({str(row["family"]) for row in members if row["should_trigger"]})
        coverage = sum(
            safe_mean(row["final_correct"] for row in members if row["family"] == family) >= 0.8
            for family in exact_families
        )
        accuracy = safe_mean(row["final_correct"] for row in members)
        xpu_ms = safe_mean(row["accelerator_time_ns"] for row in members) / 1e6
        axes[0].bar(label, coverage)
        axes[1].scatter(xpu_ms, accuracy, s=65, label=label)
    axes[0].set(ylabel="exact families at >= 0.8", ylim=(0, 5.3))
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].set(xlabel="mean XPU time / item (ms)", ylabel="overall accuracy", ylim=(0, 1.04))
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
