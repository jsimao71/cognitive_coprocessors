"""Hierarchical state workspace with explicit scope visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ScopeError(ValueError):
    pass


@dataclass
class ScopeState:
    scope_id: str
    parent_scope_id: str | None
    kind: str
    source: str
    values: dict[str, Any] = field(default_factory=dict)
    returned: Any = None


class Workspace:
    def __init__(self, root: dict[str, Any]) -> None:
        self.root_id = str(root["id"])
        self.scopes = {
            self.root_id: ScopeState(
                self.root_id,
                root.get("parent"),
                str(root.get("kind", "workspace")),
                str(root.get("source", "runtime_default")),
            )
        }
        self.scope_paths = {self.root_id: self.root_id}

    def add_scope(self, scope: dict[str, Any]) -> str:
        parent = str(scope["parent"])
        parent_path = self._scope_path(parent)
        path = f"{parent_path}/{scope['id']}"
        if path not in self.scopes:
            self.scopes[path] = ScopeState(
                str(scope["id"]),
                parent_path,
                str(scope.get("kind", "explicit")),
                str(scope.get("source", "model_asl")),
            )
        self.scope_paths["/".join(scope.get("path", []))] = path
        return path

    def _scope_path(self, scope: str) -> str:
        if scope in self.scopes:
            return scope
        if scope in self.scope_paths:
            return self.scope_paths[scope]
        matches = [path for path, state in self.scopes.items() if state.scope_id == scope]
        if len(matches) == 1:
            return matches[0]
        raise ScopeError(f"unknown or ambiguous scope: {scope}")

    def set(self, scope: str, path: str, value: Any) -> None:
        self.scopes[self._scope_path(scope)].values[path] = value

    def get(self, scope: str, path: str) -> Any:
        current = self._scope_path(scope)
        while current is not None:
            state = self.scopes[current]
            if path in state.values:
                return state.values[path]
            current = state.parent_scope_id
        parts = path.split(".")
        if len(parts) > 1:
            scope_name, local_path = parts[0], ".".join(parts[1:])
            current_path = self._scope_path(scope)
            parent = self.scopes[current_path].parent_scope_id
            candidates = [
                path_key
                for path_key, state in self.scopes.items()
                if state.scope_id == scope_name
                and state.parent_scope_id in {current_path, parent}
                and local_path in state.values
            ]
            if len(candidates) == 1:
                return self.scopes[candidates[0]].values[local_path]
        raise ScopeError(f"unresolved reference {path!r} from scope {scope!r}")

    def return_value(self, scope: str, value: Any) -> None:
        self.scopes[self._scope_path(scope)].returned = value

    def snapshot(self) -> dict[str, Any]:
        return {
            path: {"values": dict(state.values), "returned": state.returned}
            for path, state in sorted(self.scopes.items())
        }
