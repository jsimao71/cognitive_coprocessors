"""Frozen F3-v1 vocabulary, signatures, and runtime capability classes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signature:
    minimum: int
    maximum: int | None
    category: str
    runtime_level: int = 0


NESTED: dict[str, Signature] = {
    "at": Signature(2, 2, "reference"),
    "event_field": Signature(2, 2, "reference"),
    "scale": Signature(2, 2, "quantity"),
    "fraction": Signature(3, 3, "quantity"),
    "source": Signature(1, 1, "evidence"),
    "cell": Signature(2, 2, "evidence"),
}

TOP_LEVEL: dict[str, Signature] = {
    "collection": Signature(4, 4, "declaration"),
    "observe": Signature(4, 4, "observation"),
    "same": Signature(3, 3, "relation", 1),
    "offset": Signature(5, 5, "relation", 1),
    "older_than": Signature(5, 5, "relation", 1),
    "younger_than": Signature(5, 5, "relation", 1),
    "multiple": Signature(4, 4, "relation", 1),
    "fraction_of": Signature(5, 5, "relation", 1),
    "difference_relation": Signature(4, 4, "relation", 1),
    "absolute_difference": Signature(4, 4, "relation", 1),
    "quotient_relation": Signature(4, 4, "relation", 1),
    "percent_of": Signature(4, 4, "relation", 1),
    "percent_more": Signature(4, 4, "relation", 1),
    "percent_less": Signature(4, 4, "relation", 1),
    "sum_relation": Signature(4, None, "relation", 1),
    "product_relation": Signature(4, None, "relation", 1),
    "rate_relation": Signature(4, 4, "relation", 1),
    "mean_relation": Signature(4, None, "relation", 1),
    "minimum_relation": Signature(4, None, "relation", 1),
    "maximum_relation": Signature(4, None, "relation", 1),
    "member": Signature(3, 3, "declaration"),
    "partition": Signature(3, 3, "declaration"),
    "remove": Signature(5, 5, "event", 1),
    "add": Signature(5, 5, "event", 1),
    "consume": Signature(5, 5, "event", 1),
    "produce": Signature(5, 5, "event", 1),
    "transfer": Signature(6, 6, "event", 1),
    "query": Signature(2, 4, "query"),
}

RUNTIME_MODES = {"r0": 0, "r1": 1, "r2": 2}


def registry_manifest() -> dict[str, object]:
    """Return a serializable registry manifest for artifact provenance."""

    def rows(values: dict[str, Signature]) -> dict[str, dict[str, object]]:
        return {
            name: {
                "minimum": item.minimum,
                "maximum": item.maximum,
                "category": item.category,
                "runtime_level": item.runtime_level,
            }
            for name, item in sorted(values.items())
        }

    return {
        "schema_version": "ccpu.paper1.f3.registry.v1",
        "nested": rows(NESTED),
        "top_level": rows(TOP_LEVEL),
        "runtime_modes": dict(RUNTIME_MODES),
    }
