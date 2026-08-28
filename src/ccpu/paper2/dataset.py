"""Deterministic mixed-engine benchmark generator."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from ccpu.common.artifacts import canonical_json, fingerprint


@dataclass(frozen=True)
class MixedBenchmarkConfig:
    depths: tuple[int, ...] = (1, 2, 4)
    distractors: tuple[int, ...] = (0, 4)
    examples_per_cell: int = 2
    control_examples: int = 6

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MixedBenchmarkConfig:
        data = raw.get("dataset", raw)
        return cls(
            depths=tuple(int(value) for value in data.get("depths", (1, 2, 4))),
            distractors=tuple(int(value) for value in data.get("distractors", (0, 4))),
            examples_per_cell=int(data.get("examples_per_cell", 2)),
            control_examples=int(data.get("control_examples", 6)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MixedExample:
    example_id: str
    engine: str
    event: str
    answer: str | None
    depth: int
    distractors: int
    should_trigger: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MixedExample:
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identifier(payload: dict[str, Any]) -> str:
    return fingerprint(payload, 16)


def _event(engine: str, payload: dict[str, Any]) -> str:
    return f"[{engine}] {canonical_json(payload)}"


def iter_benchmark(config: MixedBenchmarkConfig) -> Iterable[MixedExample]:
    for depth in config.depths:
        for distractors in config.distractors:
            for index in range(config.examples_per_cell):
                prefix = f"d{depth}_x{distractors}_i{index}"
                expression = " + ".join(str(value) for value in range(1, depth + 2))
                yield MixedExample(
                    example_id=_identifier({"engine": "calculator", "prefix": prefix}),
                    engine="calculator",
                    event=f"[calculator] {expression}",
                    answer=str(sum(range(1, depth + 2))),
                    depth=depth,
                    distractors=distractors,
                )

                facts = [
                    {"predicate": "link", "arguments": [f"{prefix}_n{step}", f"{prefix}_n{step + 1}"]}
                    for step in range(depth)
                ]
                facts.extend(
                    {"predicate": "noise", "arguments": [f"{prefix}_z{item}"]}
                    for item in range(distractors)
                )
                logic_payload = {
                    "facts": facts,
                    "rules": [
                        {
                            "head": {"predicate": "reach", "arguments": ["?x", "?y"]},
                            "body": [{"predicate": "link", "arguments": ["?x", "?y"]}],
                        },
                        {
                            "head": {"predicate": "reach", "arguments": ["?x", "?z"]},
                            "body": [
                                {"predicate": "reach", "arguments": ["?x", "?y"]},
                                {"predicate": "link", "arguments": ["?y", "?z"]},
                            ],
                        },
                    ],
                    "query": {"predicate": "reach", "arguments": [f"{prefix}_n0", f"{prefix}_n{depth}"]},
                }
                yield MixedExample(
                    example_id=_identifier({"engine": "horn", "prefix": prefix}),
                    engine="horn",
                    event=_event("horn", logic_payload),
                    answer="true",
                    depth=depth,
                    distractors=distractors,
                )

                isa = [[f"{prefix}_g{step}", f"{prefix}_g{step + 1}"] for step in range(depth)]
                isa.extend([f"{prefix}_q{item}", f"{prefix}_r{item}"] for item in range(distractors))
                graph_payload = {
                    "operation": "graph.isa",
                    "isa": isa,
                    "frames": [],
                    "query": [f"{prefix}_g0", f"{prefix}_g{depth}"],
                }
                yield MixedExample(
                    example_id=_identifier({"engine": "graph", "prefix": prefix}),
                    engine="frame_graph",
                    event=_event("graph", graph_payload),
                    answer="true",
                    depth=depth,
                    distractors=distractors,
                )
    controls = (
        "The text [calculator] 2 + 2 is quoted, not an event.",
        "prefix [horn] {}",
        "[unknown] payload",
        "ordinary prose with facts and numbers 2 3 4",
        "`[graph] {}`",
        "calculator: 4 + 5",
    )
    for index, text in enumerate(controls[: config.control_examples]):
        yield MixedExample(
            example_id=_identifier({"engine": "control", "index": index, "text": text}),
            engine="control",
            event=text,
            answer=None,
            depth=0,
            distractors=0,
            should_trigger=False,
        )
