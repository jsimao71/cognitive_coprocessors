"""Deterministic arithmetic surface normalization for Paper 1 interfaces."""

from __future__ import annotations

import re
from dataclasses import replace

from ccpu.common.schema import CoprocessorRequest, DetectionCandidate

from .arithmetic import (
    ArithmeticNormalizationError,
    ArithmeticNormalizer,
    CalculatorLimits,
)

_ALIASES = (
    (re.compile(r"\\left\b"), ""),
    (re.compile(r"\\right\b"), ""),
    (re.compile(r"\\times\b"), "*"),
    (re.compile(r"\\cdot\b"), "*"),
    (re.compile(r"\\div\b"), "/"),
)
_UNICODE_TRANSLATION = str.maketrans(
    {
        "×": "*",
        "÷": "/",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "﹣": "-",
        "－": "-",
        "[": "(",
        "]": ")",
        "{": "(",
        "}": ")",
    }
)
_ASCII_ARITHMETIC = re.compile(r"[0-9+\-*/%()\s]+\Z")


def normalize_arithmetic_surface(text: str) -> str:
    """Map a small allowlist of equivalent math notation to calculator syntax."""

    normalized = text.strip().translate(_UNICODE_TRANSLATION)
    for pattern, replacement in _ALIASES:
        normalized = pattern.sub(replacement, normalized)
    if not normalized or not _ASCII_ARITHMETIC.fullmatch(normalized):
        raise ArithmeticNormalizationError("surface contains unsupported arithmetic notation")
    return normalized


class ArithmeticSurfaceNormalizer:
    """Normalize notation, then delegate semantic validation to the existing IR."""

    name = "surface_normalizer_v1"

    def __init__(self, limits: CalculatorLimits | None = None) -> None:
        self.limits = limits or CalculatorLimits()
        self.delegate = ArithmeticNormalizer(self.limits)

    def normalize(self, candidate: DetectionCandidate) -> CoprocessorRequest:
        surface = candidate.raw_text
        normalized = normalize_arithmetic_surface(surface)
        normalized_candidate = replace(
            candidate,
            raw_text=normalized,
            metadata={**candidate.metadata, "surface_normalizer": self.name},
        )
        request = self.delegate.normalize(normalized_candidate)
        return replace(
            request,
            payload={
                **request.payload,
                "surface_expression": surface,
                "surface_normalizer": self.name,
            },
            metadata={**request.metadata, "surface_normalizer": self.name},
        )
