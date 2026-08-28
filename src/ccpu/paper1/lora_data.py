"""Deterministic protocol-only SFT data with explicit leakage audits."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import fingerprint, read_jsonl

from .dataset import ArithmeticExample
from .prompts import minimal_block_prompt


@dataclass(frozen=True)
class LoRAProtocolDataConfig:
    seed: int = 880301
    train_arithmetic: int = 80
    train_controls: int = 80
    dev_arithmetic: int = 20
    dev_controls: int = 20

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LoRAProtocolDataConfig:
        data = value.get("dataset", value)
        return cls(
            seed=int(data.get("seed", 880301)),
            train_arithmetic=int(data.get("train_arithmetic", 80)),
            train_controls=int(data.get("train_controls", 80)),
            dev_arithmetic=int(data.get("dev_arithmetic", 20)),
            dev_controls=int(data.get("dev_controls", 20)),
        )

    def validate(self) -> None:
        counts = (
            self.train_arithmetic,
            self.train_controls,
            self.dev_arithmetic,
            self.dev_controls,
        )
        if any(count < 1 for count in counts):
            raise ValueError("all protocol split counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONTROL_TEMPLATES = (
    "Repeat this release label exactly: version {a}.{b}.{c}.",
    "Keep this date as prose: {a:04d}-{b:02d}-{c:02d}.",
    "Describe the range {a}-{b} without calculating it.",
    'Preserve the quoted example "{a} + {b}" without executing it.',
    "Explain that code `total = {a} * {b}` is only a quoted snippet.",
    "Room {a} and room {b} are identifiers on different floors.",
    "The catalog key ITEM-{a}-{b} is not an arithmetic request.",
    "Repeat: the equation x + {a} = {b} contains a variable.",
)


def _heldout_operands(path: str | Path) -> tuple[set[int], set[str]]:
    operands: set[int] = set()
    expressions: set[str] = set()
    for row in read_jsonl(path):
        expression = row.get("expression")
        if expression:
            expressions.add(re.sub(r"\s+", "", str(expression)))
            operands.update(int(value) for value in re.findall(r"\d+", str(expression)))
    return operands, expressions


def _new_operand(rng: random.Random, used: set[int], digits: int) -> int:
    lower = 10 ** (digits - 1)
    upper = 10**digits - 1
    for _ in range(10_000):
        value = rng.randint(lower, upper)
        if value not in used:
            used.add(value)
            return value
    raise RuntimeError(f"could not draw a unique {digits}-digit operand")


def _surface(expression: str, variant: str) -> str:
    if variant == "latex":
        return expression.replace("*", r"\times").replace("/", r"\div")
    if variant == "unicode":
        return expression.replace("*", "×").replace("/", "÷").replace("-", "−")
    if variant == "brackets":
        return expression.replace("(", "[").replace(")", "]")
    return expression


def _arithmetic_record(
    *, rng: random.Random, split: str, index: int, used_operands: set[int]
) -> dict[str, Any]:
    digits = (3, 4, 5, 6)[index % 4]
    operands = [_new_operand(rng, used_operands, digits) for _ in range(4)]
    a, b, c, d = operands
    pattern = index % 8
    if pattern == 0:
        expression, structure = f"{a} + {b}", "single"
    elif pattern == 1:
        expression, structure = f"{a} - {b}", "single"
    elif pattern == 2:
        expression, structure = f"{a} * {b}", "single"
    elif pattern == 3:
        expression, structure = f"({a} + {b}) * {c}", "nested"
    elif pattern == 4:
        expression, structure = f"{a} * {b} - {c}", "mixed"
    elif pattern == 5:
        expression, structure = f"({a} + {b}) * ({c} - {d})", "nested_mixed"
    elif pattern == 6:
        expression, structure = f"({a} * {b}) / {b}", "exact_division"
    else:
        expression, structure = f"{a} * {b} * {c}", "long_intermediate"
    variant = ("ascii", "latex", "unicode", "brackets")[index % 4]
    source_expression = _surface(expression, variant)
    request = (
        "Compute the exact value of the arithmetic expression below. Return an integer or "
        f"reduced fraction.\nExpression: {source_expression}"
    )
    example = ArithmeticExample(
        example_id="training-placeholder",
        schema_version="ccpu.paper1.lora_protocol_source.v1",
        task_kind="arithmetic",
        split=split,
        prompt=request,
        reference_completion="",
        should_trigger=True,
        expression=expression,
        answer=None,
        difficulty={},
        metadata={},
    )
    core = {
        "schema_version": "ccpu.paper1.lora_protocol_example.v1",
        "split": split,
        "task_kind": "arithmetic",
        "prompt": minimal_block_prompt(example),
        "target": f"```calculator\n{expression}\n```",
        "expression": expression,
        "source_expression": source_expression,
        "operands": operands,
        "structure": structure,
        "surface_variant": variant,
    }
    return {"example_id": f"p1-lora-{fingerprint(core, 16)}", **core}


def _control_record(*, rng: random.Random, split: str, index: int) -> dict[str, Any]:
    a, b, c = rng.randint(10, 9999), rng.randint(10, 9999), rng.randint(1, 28)
    sentence = _CONTROL_TEMPLATES[index % len(_CONTROL_TEMPLATES)].format(a=a, b=b, c=c)
    example = ArithmeticExample(
        example_id="training-placeholder",
        schema_version="ccpu.paper1.lora_protocol_source.v1",
        task_kind="control",
        split=split,
        prompt="Repeat the supplied control sentence exactly without executing it.",
        reference_completion=sentence,
        should_trigger=False,
        expression=None,
        answer=None,
        difficulty={},
        metadata={},
    )
    core = {
        "schema_version": "ccpu.paper1.lora_protocol_example.v1",
        "split": split,
        "task_kind": "control",
        "prompt": minimal_block_prompt(example),
        "target": sentence,
        "expression": None,
        "source_expression": None,
        "operands": [],
        "structure": "control",
        "surface_variant": "prose",
        "control_category": index % len(_CONTROL_TEMPLATES),
    }
    return {"example_id": f"p1-lora-{fingerprint(core, 16)}", **core}


def generate_protocol_data(
    config: LoRAProtocolDataConfig, *, excluded_dataset: str | Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Generate train/dev rows and prove benchmark/split operand separation."""

    config.validate()
    benchmark_operands, benchmark_expressions = _heldout_operands(excluded_dataset)
    used_operands = set(benchmark_operands)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, arithmetic_count, control_count, offset in (
        ("train", config.train_arithmetic, config.train_controls, 0),
        ("dev", config.dev_arithmetic, config.dev_controls, 10_000),
    ):
        rng = random.Random(config.seed + offset)
        rows = [
            _arithmetic_record(
                rng=rng,
                split=split,
                index=index + offset,
                used_operands=used_operands,
            )
            for index in range(arithmetic_count)
        ]
        rows.extend(
            _control_record(rng=rng, split=split, index=index + offset)
            for index in range(control_count)
        )
        rng.shuffle(rows)
        splits[split] = rows

    train_operands = {
        int(value) for row in splits["train"] for value in row.get("operands", [])
    }
    dev_operands = {
        int(value) for row in splits["dev"] for value in row.get("operands", [])
    }
    generated_expressions = {
        re.sub(r"\s+", "", str(row["expression"]))
        for rows in splits.values()
        for row in rows
        if row.get("expression")
    }
    audit = {
        "schema_version": "ccpu.paper1.lora_leakage_audit.v1",
        "excluded_dataset": str(Path(excluded_dataset)),
        "benchmark_operand_count": len(benchmark_operands),
        "benchmark_expression_count": len(benchmark_expressions),
        "train_dev_operand_overlap": sorted(train_operands & dev_operands),
        "train_benchmark_operand_overlap": sorted(train_operands & benchmark_operands),
        "dev_benchmark_operand_overlap": sorted(dev_operands & benchmark_operands),
        "generated_benchmark_expression_overlap": sorted(
            generated_expressions & benchmark_expressions
        ),
    }
    audit["passed"] = not any(
        audit[key]
        for key in (
            "train_dev_operand_overlap",
            "train_benchmark_operand_overlap",
            "dev_benchmark_operand_overlap",
            "generated_benchmark_expression_overlap",
        )
    )
    if not audit["passed"]:
        raise ValueError(f"protocol leakage audit failed: {audit}")
    return splits, audit
