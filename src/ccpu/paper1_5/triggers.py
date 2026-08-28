"""Transparent confidence and semantic epistemic-risk triggers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_RISK_PATTERNS = (
    re.compile(r"\b(?:latest|currently|as of|version|registry|assigned|designated)\b", re.IGNORECASE),
    re.compile(r"\b(?:changed|updated|superseded|according to)\b", re.IGNORECASE),
    re.compile(r"\b20\d{2}\b"),
)
_SAFE_EXCEPTIONS = (re.compile(r"\belectric current\b", re.IGNORECASE),)


@dataclass(frozen=True)
class TriggerDecision:
    confidence: bool
    semantic: bool
    reasons: tuple[str, ...]

    def for_condition(self, condition: str) -> bool:
        if condition == "flare_like":
            return self.confidence
        if condition in {"semantic", "retrospective"}:
            return self.semantic
        if condition == "confidence_or_semantic":
            return self.confidence or self.semantic
        if condition == "confidence_and_semantic":
            return self.confidence and self.semantic
        raise ValueError(f"condition has no dynamic trigger: {condition}")


def semantic_risk(question: str, forecast: str = "") -> tuple[bool, tuple[str, ...]]:
    text = f"{question} {forecast}"
    if any(pattern.search(text) for pattern in _SAFE_EXCEPTIONS):
        return False, ("safe_exception:electric_current",)
    reasons = tuple(f"risk_pattern:{index}" for index, pattern in enumerate(_RISK_PATTERNS) if pattern.search(text))
    return bool(reasons), reasons


def decide(question: str, forecast: str, token_probabilities: tuple[float, ...], threshold: float) -> TriggerDecision:
    confidence = bool(token_probabilities) and min(token_probabilities) < threshold
    semantic, reasons = semantic_risk(question, forecast)
    return TriggerDecision(confidence=confidence, semantic=semantic, reasons=reasons)


def fit_confidence_threshold(
    rows: list[tuple[tuple[float, ...], bool]], *, retrieval_cost: float = 0.25
) -> float:
    """Choose a development threshold by risk F1 minus a fixed retrieval penalty."""
    observed = sorted({probability for probabilities, _ in rows for probability in probabilities})
    candidates = [0.0, *observed, 1.0]
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predictions = [bool(values) and min(values) < threshold for values, _ in rows]
        tp = sum(predicted and gold for predicted, (_, gold) in zip(predictions, rows, strict=True))
        fp = sum(predicted and not gold for predicted, (_, gold) in zip(predictions, rows, strict=True))
        fn = sum(not predicted and gold for predicted, (_, gold) in zip(predictions, rows, strict=True))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        retrieval_rate = sum(predictions) / len(predictions) if predictions else 0.0
        utility = f1 - retrieval_cost * retrieval_rate
        score = (utility, f1, -retrieval_rate, -threshold)
        if best is None or score > best:
            best = score
            selected = threshold
    return selected
