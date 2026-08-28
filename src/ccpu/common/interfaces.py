"""Small protocols that keep detectors, engines, and model backends replaceable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .schema import (
    CoprocessorRequest,
    CoprocessorResult,
    DetectionCandidate,
    GenerationResult,
    MicroStateItem,
    RuntimeStep,
)


class IncrementalDetector(Protocol):
    name: str

    def feed(self, text: str) -> Sequence[DetectionCandidate]: ...

    def reset(self) -> None: ...


class Normalizer(Protocol):
    def normalize(self, candidate: DetectionCandidate) -> CoprocessorRequest: ...


class Coprocessor(Protocol):
    name: str

    def execute(self, request: CoprocessorRequest) -> CoprocessorResult: ...


class Materializer(Protocol):
    def materialize(
        self,
        request: CoprocessorRequest,
        result: CoprocessorResult,
        state: MicroStateItem,
    ) -> str: ...


class IncrementalController(Protocol):
    def feed(self, text: str) -> RuntimeStep: ...

    def reset(self, *, run_id: str | None = None) -> None: ...


class GenerationBackend(Protocol):
    model_id: str

    def generate(
        self,
        prompt: str,
        *,
        controller: IncrementalController | None = None,
        seed: int = 0,
    ) -> GenerationResult: ...
