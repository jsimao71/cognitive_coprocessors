"""Leakage-audited five-engine benchmark and adapter data."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import fingerprint, write_json, write_jsonl

ENGINE_CATALOGS = {
    1: ("calculator",),
    2: ("calculator", "datalog"),
    3: ("calculator", "datalog", "graph"),
    5: ("calculator", "datalog", "graph", "date", "units"),
}


@dataclass(frozen=True)
class NextBenchmarkConfig:
    seed: int = 22051
    train_per_engine: int = 32
    dev_per_engine: int = 8
    test_per_engine: int = 16
    train_controls: int = 40
    dev_controls: int = 10
    test_controls: int = 20

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NextBenchmarkConfig:
        values = raw.get("benchmark", raw)
        defaults = asdict(cls())
        return cls(**{key: int(values.get(key, value)) for key, value in defaults.items()})


def _calculator(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    namespace = {"train": 100000, "dev": 200000, "test": 300000}[split]
    left = namespace + rng.randrange(1000, 90000)
    right = rng.randrange(101, 9000)
    expression = f"{left} * {right}"
    return {
        "engine": "calculator",
        "prompt": f"Compute the exact product of {left} and {right}.",
        "target": f"```calculator\n{expression}\n```",
        "answer": str(left * right),
        "signature": f"calc:{left}:{right}",
    }


def _datalog(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    del rng
    prefix = f"{split}n{index:04d}"
    target = (
        f"```datalog\nfact link({prefix}a,{prefix}b)\n"
        f"fact link({prefix}b,{prefix}c)\nquery reachable({prefix}a,{prefix}c)\n```"
    )
    return {
        "engine": "datalog",
        "prompt": (
            f"Given directed links {prefix}a to {prefix}b and {prefix}b to {prefix}c, "
            f"determine exactly whether {prefix}c is reachable from {prefix}a."
        ),
        "target": target,
        "answer": "true",
        "signature": f"datalog:{prefix}",
    }


def _graph(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    del rng
    prefix = f"{split}g{index:04d}"
    target = (
        f"```graph\nisa {prefix}_special {prefix}_middle\n"
        f"isa {prefix}_middle {prefix}_general\n"
        f"query isa {prefix}_special {prefix}_general\n```"
    )
    return {
        "engine": "graph",
        "prompt": (
            f"In an ISA hierarchy, {prefix}_special is a {prefix}_middle and "
            f"{prefix}_middle is a {prefix}_general. Is {prefix}_special a "
            f"{prefix}_general?"
        ),
        "target": target,
        "answer": "true",
        "signature": f"graph:{prefix}",
    }


def _date(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    start_year = {"train": 2030, "dev": 2060, "test": 2090}[split]
    base = date(start_year + index % 20, 1 + index % 12, 1 + index % 24)
    days = rng.randrange(25, 330)
    return {
        "engine": "date",
        "prompt": f"What ISO calendar date is {days} days after {base.isoformat()}?",
        "target": f"```date\nadd {base.isoformat()} P{days}D\n```",
        "answer": (base + timedelta(days=days)).isoformat(),
        "signature": f"date:{base.isoformat()}:{days}",
    }


def _units(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    del index
    namespace = {"train": 0, "dev": 1000, "test": 2000}[split]
    value = Decimal(namespace + rng.randrange(11, 990)) / Decimal(10)
    with localcontext() as context:
        context.prec = 40
        answer = value * Decimal("1.609344")
    display = format(answer.normalize(), "f").rstrip("0").rstrip(".")
    source = format(value.normalize(), "f")
    return {
        "engine": "units",
        "prompt": f"Convert exactly {source} miles to kilometers.",
        "target": f"```units\nconvert {source} mile -> kilometer\n```",
        "answer": display,
        "signature": f"units:{source}",
    }


_BUILDERS = {
    "calculator": _calculator,
    "datalog": _datalog,
    "graph": _graph,
    "date": _date,
    "units": _units,
}


def _rows(split: str, per_engine: int, controls: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for engine, builder in _BUILDERS.items():
        for index in range(per_engine):
            content = builder(split, index, rng)
            rows.append(
                {
                    "example_id": f"p2-{split}-{engine}-{index:04d}",
                    "split": split,
                    "should_trigger": True,
                    **content,
                }
            )
    for index in range(controls):
        marker = f"{split.upper()}-CONTROL-{index:04d}"
        rows.append(
            {
                "example_id": f"p2-{split}-control-{index:04d}",
                "split": split,
                "engine": "control",
                "prompt": f"Repeat the already supplied label {marker} without computation.",
                "target": "NO_EXECUTION",
                "answer": marker,
                "signature": f"control:{marker}",
                "should_trigger": False,
            }
        )
    random.Random(seed + 1).shuffle(rows)
    return rows


def generate_next_benchmark(
    config: NextBenchmarkConfig, output_dir: str | Path
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train = _rows("train", config.train_per_engine, config.train_controls, config.seed)
    dev = _rows("dev", config.dev_per_engine, config.dev_controls, config.seed + 10000)
    test = _rows("test", config.test_per_engine, config.test_controls, config.seed + 20000)
    splits = {"train": train, "dev": dev, "test": test}
    signatures = {name: {str(row["signature"]) for row in rows} for name, rows in splits.items()}
    audit = {
        "schema_version": "ccpu.paper2.leakage_audit.v1",
        "train_dev_overlap": sorted(signatures["train"] & signatures["dev"]),
        "train_test_overlap": sorted(signatures["train"] & signatures["test"]),
        "dev_test_overlap": sorted(signatures["dev"] & signatures["test"]),
        "target_contains_answer": [
            row["example_id"]
            for rows in splits.values()
            for row in rows
            if row["should_trigger"] and str(row["answer"]) in str(row["target"])
        ],
    }
    if any(audit[key] for key in audit if key != "schema_version"):
        raise ValueError(f"Paper 2 leakage audit failed: {audit}")
    paths = {
        name: write_jsonl(output_dir / f"{name}.jsonl", rows) for name, rows in splits.items()
    }
    manifest = {
        "schema_version": "ccpu.paper2.next_benchmark.v1",
        "config": asdict(config),
        "engine_catalogs": {str(key): list(value) for key, value in ENGINE_CATALOGS.items()},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "paths": {name: str(path) for name, path in paths.items()},
        "leakage_audit": audit,
        "fingerprint": fingerprint({"config": asdict(config), "audit": audit}),
    }
    write_json(output_dir / "leakage_audit.json", audit)
    write_json(output_dir / "manifest.json", manifest)
    return manifest
