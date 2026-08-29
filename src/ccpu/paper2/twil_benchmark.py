"""Frozen benchmark for reasoning-in-weights versus coprocessor execution."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import fingerprint, write_json, write_jsonl


@dataclass(frozen=True)
class TwILBenchmarkConfig:
    seed: int = 22501
    depths: tuple[int, ...] = (1, 2, 4, 8)
    distractors: tuple[int, ...] = (0, 4)
    examples_per_cell: int = 1
    nonlogic_per_engine: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TwILBenchmarkConfig:
        values = raw.get("benchmark", raw)
        return cls(
            seed=int(values.get("seed", cls.seed)),
            depths=tuple(map(int, values.get("depths", cls.depths))),
            distractors=tuple(map(int, values.get("distractors", cls.distractors))),
            examples_per_cell=int(values.get("examples_per_cell", cls.examples_per_cell)),
            nonlogic_per_engine=int(
                values.get("nonlogic_per_engine", cls.nonlogic_per_engine)
            ),
        )


def _logic_row(
    family: str, depth: int, distractors: int, replicate: int
) -> dict[str, Any]:
    prefix = f"{family[0]}d{depth}x{distractors}r{replicate}"
    nodes = [f"{prefix}n{index}" for index in range(depth + 1)]
    if family == "datalog":
        facts = [f"fact link({left},{right})" for left, right in pairwise(nodes)]
        facts.extend(
            f"fact link({prefix}z{index}a,{prefix}z{index}b)"
            for index in range(distractors)
        )
        target = "```datalog\n" + "\n".join(
            [*facts, f"query reachable({nodes[0]},{nodes[-1]})"]
        ) + "\n```"
        prompt = (
            "Directed links are "
            + ", ".join(
                f"{left} to {right}" for left, right in pairwise(nodes)
            )
            + ". "
            + (
                "Unrelated links are "
                + ", ".join(
                    f"{prefix}z{index}a to {prefix}z{index}b"
                    for index in range(distractors)
                )
                + ". "
                if distractors
                else ""
            )
            + f"Is {nodes[-1]} reachable from {nodes[0]}?"
        )
    else:
        facts = [f"isa {left} {right}" for left, right in pairwise(nodes)]
        facts.extend(
            f"isa {prefix}z{index}a {prefix}z{index}b" for index in range(distractors)
        )
        target = "```graph\n" + "\n".join(
            [*facts, f"query isa {nodes[0]} {nodes[-1]}"]
        ) + "\n```"
        prompt = (
            "An ISA hierarchy contains "
            + ", ".join(
                f"{left} is a {right}" for left, right in pairwise(nodes)
            )
            + ". "
            + (
                "Unrelated statements are "
                + ", ".join(
                    f"{prefix}z{index}a is a {prefix}z{index}b"
                    for index in range(distractors)
                )
                + ". "
                if distractors
                else ""
            )
            + f"Is {nodes[0]} a {nodes[-1]}?"
        )
    return {
        "example_id": f"twil-{family}-d{depth}-x{distractors}-r{replicate}",
        "family": family,
        "engine": family,
        "prompt": prompt,
        "target": target,
        "answer": "true",
        "should_trigger": True,
        "depth": depth,
        "fact_count": depth + distractors,
        "distractors": distractors,
        "semantic_class": "exact_deduction",
    }


def _nonlogic_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        left, right = rng.randrange(12000, 99000), rng.randrange(301, 9000)
        rows.append(
            {
                "example_id": f"twil-calculator-{index}",
                "family": "calculator",
                "engine": "calculator",
                "prompt": f"What is the exact product of {left} and {right}?",
                "target": f"```calculator\n{left} * {right}\n```",
                "answer": str(left * right),
                "should_trigger": True,
                "depth": 1,
                "fact_count": 0,
                "distractors": 0,
                "semantic_class": "exact_nonlogic",
            }
        )
        base = date(2070 + index, 2 + index, 10 + index)
        days = 137 + 37 * index
        rows.append(
            {
                "example_id": f"twil-date-{index}",
                "family": "date",
                "engine": "date",
                "prompt": f"What ISO date is {days} days after {base.isoformat()}?",
                "target": f"```date\nadd {base.isoformat()} P{days}D\n```",
                "answer": (base + timedelta(days=days)).isoformat(),
                "should_trigger": True,
                "depth": 1,
                "fact_count": 0,
                "distractors": 0,
                "semantic_class": "exact_nonlogic",
            }
        )
        value = Decimal(31 + 17 * index) / Decimal(10)
        answer = format((value * Decimal("1.609344")).normalize(), "f")
        rows.append(
            {
                "example_id": f"twil-units-{index}",
                "family": "units",
                "engine": "units",
                "prompt": f"Convert exactly {value} miles to kilometers.",
                "target": f"```units\nconvert {value} mile -> kilometer\n```",
                "answer": answer,
                "should_trigger": True,
                "depth": 1,
                "fact_count": 0,
                "distractors": 0,
                "semantic_class": "exact_nonlogic",
            }
        )
    return rows


def _semantic_rows() -> list[dict[str, Any]]:
    cases = (
        (
            "uncertain_premise",
            "Every glorp is a wug. Mira may be a glorp. Does it follow that Mira is a wug?",
            "unknown",
        ),
        (
            "converse_error",
            "Every painter is creative. Ivo is creative. Must Ivo be a painter?",
            "false",
        ),
        (
            "explicit_entailment",
            "Every botanist is patient. Ada is a botanist. Must Ada be patient?",
            "true",
        ),
        (
            "ambiguous_reference",
            "Lea told Mara that she won. Without more context, is Lea definitely the winner?",
            "unknown",
        ),
    )
    return [
        {
            "example_id": f"twil-semantic-{label}",
            "family": "semantic",
            "engine": "none",
            "prompt": prompt,
            "target": "NO_EXECUTION",
            "answer": answer,
            "should_trigger": False,
            "depth": 1,
            "fact_count": 2,
            "distractors": 0,
            "semantic_class": label,
        }
        for label, prompt, answer in cases
    ]


def generate_twil_benchmark(
    config: TwILBenchmarkConfig, output_dir: str | Path
) -> dict[str, Any]:
    rows = [
        _logic_row(family, depth, distractors, replicate)
        for family in ("datalog", "graph")
        for depth in config.depths
        for distractors in config.distractors
        for replicate in range(config.examples_per_cell)
    ]
    rows.extend(_nonlogic_rows(config.nonlogic_per_engine, config.seed))
    rows.extend(_semantic_rows())
    identifiers = [row["example_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("TwIL benchmark identifiers must be unique")
    output_dir = Path(output_dir)
    dataset = write_jsonl(output_dir / "test.jsonl", rows)
    manifest = {
        "schema_version": "ccpu.paper2.twil_benchmark.v1",
        "config": asdict(config),
        "record_count": len(rows),
        "families": sorted({str(row["family"]) for row in rows}),
        "model_independent": True,
        "fingerprint": fingerprint({"config": asdict(config), "rows": rows}),
        "dataset": str(dataset),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
