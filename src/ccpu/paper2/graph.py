"""Bounded ISA and frame-slot closure engine."""

from __future__ import annotations

from dataclasses import dataclass

from ccpu.common.schema import CoprocessorRequest, CoprocessorResult

from .state import StateLimitError, TypedMicroState


@dataclass(frozen=True)
class GraphLimits:
    max_edges: int = 512
    max_depth: int = 16


class FrameGraphEngine:
    name = "frame_graph"

    def __init__(self, state: TypedMicroState, limits: GraphLimits | None = None) -> None:
        self.state = state
        self.limits = limits or GraphLimits()

    def execute(self, request: CoprocessorRequest) -> CoprocessorResult:
        if request.operation not in {"graph.isa", "graph.frame"}:
            return self._failure(request, "unsupported_operation")
        try:
            edges = {tuple(map(str, edge)) for edge in request.payload.get("isa", [])}
            frames = {tuple(map(str, frame)) for frame in request.payload.get("frames", [])}
            if any(len(edge) != 2 for edge in edges) or any(len(frame) != 3 for frame in frames):
                raise ValueError("invalid graph tuple arity")
            for edge in edges:
                self.state.add("isa_edge", {"child": edge[0], "parent": edge[1]}, provenance={"request_id": request.request_id})
            for frame in frames:
                self.state.add(
                    "frame_slot",
                    {"entity": frame[0], "slot": frame[1], "value": frame[2]},
                    provenance={"request_id": request.request_id},
                )
            all_edges = {
                (str(item.payload["child"]), str(item.payload["parent"]))
                for item in self.state.by_kind("isa_edge")
            }
            if len(all_edges) > self.limits.max_edges:
                raise StateLimitError("graph edge budget exceeded")
            ancestors = self._closure(all_edges)
            if request.operation == "graph.isa":
                child, parent = map(str, request.payload["query"])
                answer: str | bool = child == parent or parent in ancestors.get(child, set())
            else:
                entity, slot = map(str, request.payload["query"])
                order = [entity, *sorted(ancestors.get(entity, set()))]
                values = []
                for node in order:
                    values.extend(
                        str(item.payload["value"])
                        for item in self.state.by_kind("frame_slot")
                        if item.payload["entity"] == node and item.payload["slot"] == slot
                    )
                    if values:
                        break
                answer = values[0] if len(set(values)) == 1 else "ambiguous" if values else "unknown"
            return CoprocessorResult(
                request_id=request.request_id,
                engine=self.name,
                ok=True,
                value=answer,
                display=str(answer).lower() if isinstance(answer, bool) else answer,
                metadata={"edge_count": len(all_edges), "node_count": len(ancestors)},
            )
        except (KeyError, TypeError, ValueError, StateLimitError) as error:
            return self._failure(request, "invalid_or_bounded_ir", str(error))

    def _closure(self, edges: set[tuple[str, str]]) -> dict[str, set[str]]:
        ancestors: dict[str, set[str]] = {}
        for child, parent in edges:
            ancestors.setdefault(child, set()).add(parent)
        for _ in range(self.limits.max_depth):
            changed = False
            for child, parents in list(ancestors.items()):
                expanded = set(parents)
                for parent in parents:
                    expanded.update(ancestors.get(parent, set()))
                if expanded != parents:
                    ancestors[child] = expanded
                    changed = True
            if not changed:
                return ancestors
        raise StateLimitError("graph closure depth budget exceeded")

    def _failure(self, request: CoprocessorRequest, code: str, message: str = "") -> CoprocessorResult:
        return CoprocessorResult(
            request_id=request.request_id,
            engine=self.name,
            ok=False,
            error_code=code,
            error_message=message,
        )
