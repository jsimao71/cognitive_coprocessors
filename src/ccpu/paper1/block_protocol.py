"""Condition-independent decomposition of fenced calculator protocol behavior."""

from __future__ import annotations

import re
from typing import Any

from ccpu.common.schema import DetectionCandidate

from .arithmetic import ArithmeticNormalizationError
from .surface import ArithmeticSurfaceNormalizer

_OPENING = re.compile(r"(?:\A|\n)```calculator[ \t]*(?:\r?\n|\Z)")
_CLOSED = re.compile(
    r"(?:\A|\n)```calculator[ \t]*\r?\n([\s\S]*?)\r?\n```"
)


def _instructions(expression: str) -> Any:
    candidate = DetectionCandidate(
        candidate_id="block-protocol-analysis",
        family="compute",
        raw_text=expression,
        start_offset=0,
        end_offset=len(expression),
        detector="block_protocol_analysis",
    )
    return ArithmeticSurfaceNormalizer().normalize(candidate).payload.get("instructions")


def analyze_block_protocol(text: str, target_expression: str | None) -> dict[str, Any]:
    """Describe wrapper and payload quality without using final-answer correctness."""

    openings = list(_OPENING.finditer(text))
    closed = list(_CLOSED.finditer(text))
    payload: str | None = None
    if closed:
        payload = closed[0].group(1).strip()
    elif openings:
        payload = text[openings[0].end() :].strip()

    payload_present = bool(payload)
    payload_exact: bool | None = None
    payload_equivalent: bool | None = None
    if target_expression is not None:
        payload_exact = payload == target_expression.strip() if payload_present else False
        if payload_present:
            try:
                payload_equivalent = _instructions(payload) == _instructions(target_expression)
            except ArithmeticNormalizationError:
                payload_equivalent = False
        else:
            payload_equivalent = False

    return {
        "block_open": bool(openings),
        "block_close": bool(closed),
        "block_count": len(closed),
        "block_multiple": len(closed) > 1,
        "block_payload": payload,
        "block_payload_present": payload_present,
        "block_payload_exact": payload_exact,
        "block_payload_semantically_equivalent": payload_equivalent,
    }
