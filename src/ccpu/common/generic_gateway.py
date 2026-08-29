"""Stable four-tool gateway over deployment-specific cognitive registries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

INTENTS = ("compute", "retrieve", "verify", "help")
TOOL_NAMES = tuple(f"__{intent}" for intent in INTENTS)


@dataclass(frozen=True)
class GenericToolCall:
    name: str
    payload: Any = None

    @property
    def intent(self) -> str:
        if self.name not in TOOL_NAMES:
            raise ValueError(f"unsupported generic cognitive tool: {self.name}")
        return self.name.removeprefix("__")


def generic_tool_schemas() -> list[dict[str, Any]]:
    """Return the fixed model-facing schemas; concrete capabilities stay in R2."""
    return [
        {
            "name": name,
            "description": f"Request {intent.upper()} cognitive assistance for the active task.",
            "parameters": {
                "type": "object",
                "properties": {"payload": {}},
                "additionalProperties": False,
            },
        }
        for name, intent in zip(TOOL_NAMES, INTENTS)
    ]


class GenericCognitiveGateway:
    """Validate stable tool names before handing intent and payload to shared R2."""

    def __init__(self, resolve: Callable[[str, Any, str], Any]) -> None:
        self._resolve = resolve

    def invoke(self, call: GenericToolCall, *, active_task: str) -> Any:
        return self._resolve(call.intent, call.payload, active_task)
