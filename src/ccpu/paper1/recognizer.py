"""Incremental strict-syntax detectors for reflex and explicit-tool conditions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ccpu.common.schema import DetectionCandidate

from .arithmetic import ArithmeticNormalizationError
from .surface import normalize_arithmetic_surface

_ARITHMETIC_CHARS = frozenset("0123456789+-*/%() \t")
_OPERATOR = re.compile(r"\*\*|//|[+\-*/%]")


def _candidate_id(detector: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha256(f"{detector}\0{start}\0{end}\0{text}".encode()).hexdigest()
    return f"candidate:{digest[:20]}"


@dataclass(frozen=True)
class RecognizerLimits:
    max_buffer_chars: int = 512
    max_expression_chars: int = 256
    suppress_double_quoted: bool = True


class StrictArithmeticRecognizer:
    """Detect an integer arithmetic suffix when a single equals sign completes it.

    Accepted lexical characters are deliberately narrower than Python arithmetic.
    Semantic validation and the binary-operation requirement remain the normalizer's
    responsibility so detection and normalization errors stay distinguishable.
    """

    name = "strict_arithmetic_v1"

    def __init__(self, limits: RecognizerLimits | None = None) -> None:
        self.limits = limits or RecognizerLimits()
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._buffer_start = 0
        self._offset = 0
        self._in_double_quote = False
        self._escaped = False

    def feed(self, text: str) -> tuple[DetectionCandidate, ...]:
        candidates: list[DetectionCandidate] = []
        for character in text:
            if character == '"' and not self._escaped:
                self._in_double_quote = not self._in_double_quote

            if character == "=" and not (
                self.limits.suppress_double_quoted and self._in_double_quote
            ):
                candidate = self._candidate_before_equals()
                if candidate is not None:
                    candidates.append(candidate)

            self._append(character)
            self._escaped = character == "\\" and not self._escaped
            if character != "\\":
                self._escaped = False
            self._offset += 1
        return tuple(candidates)

    def _append(self, character: str) -> None:
        self._buffer += character
        excess = len(self._buffer) - self.limits.max_buffer_chars
        if excess > 0:
            self._buffer = self._buffer[excess:]
            self._buffer_start += excess

    def _candidate_before_equals(self) -> DetectionCandidate | None:
        if self._buffer.endswith("="):
            return None
        index = len(self._buffer)
        while index > 0 and self._buffer[index - 1] in _ARITHMETIC_CHARS:
            index -= 1
        suffix = self._buffer[index:]
        leading = len(suffix) - len(suffix.lstrip())
        expression = suffix.strip()
        start = self._buffer_start + index + leading
        if not expression or len(expression) > self.limits.max_expression_chars:
            return None
        if expression[0] not in "0123456789+-(" or expression[-1] not in "0123456789)":
            return None
        if not _OPERATOR.search(expression):
            return None
        if expression.count("(") != expression.count(")"):
            return None
        if (
            leading == 0
            and index > 0
            and self._buffer[index - 1] in ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
        ):
            return None
        end = self._offset + 1
        return DetectionCandidate(
            candidate_id=_candidate_id(self.name, start, end, expression),
            family="compute",
            raw_text=expression,
            start_offset=start,
            end_offset=end,
            detector=self.name,
            metadata={"terminator": "=", "syntax": "integer_arithmetic_suffix_v1"},
        )


class ExplicitCalculatorToolRecognizer:
    """Textual conventional-tool baseline with an unambiguous authored call."""

    name = "explicit_calculator_tool_v1"
    _pattern = re.compile(r"<tool:calculator>\s*([^<>]+?)\s*</tool>\s*$")

    def __init__(self, max_buffer_chars: int = 512) -> None:
        self.max_buffer_chars = max_buffer_chars
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._buffer_start = 0
        self._offset = 0

    def feed(self, text: str) -> tuple[DetectionCandidate, ...]:
        candidates: list[DetectionCandidate] = []
        for character in text:
            self._buffer += character
            self._offset += 1
            excess = len(self._buffer) - self.max_buffer_chars
            if excess > 0:
                self._buffer = self._buffer[excess:]
                self._buffer_start += excess
            match = self._pattern.search(self._buffer)
            if match:
                expression = match.group(1).strip()
                start = self._buffer_start + match.start(1)
                candidates.append(
                    DetectionCandidate(
                        candidate_id=_candidate_id(self.name, start, self._offset, expression),
                        family="compute",
                        raw_text=expression,
                        start_offset=start,
                        end_offset=self._offset,
                        detector=self.name,
                        metadata={"syntax": "explicit_calculator_tool_v1"},
                    )
                )
        return tuple(candidates)


class NormalizedArithmeticRecognizer(StrictArithmeticRecognizer):
    """Detect complete arithmetic suffixes with allowlisted notation aliases."""

    name = "normalized_arithmetic_v1"

    def _candidate_before_equals(self) -> DetectionCandidate | None:
        if self._buffer.endswith("="):
            return None
        line_start = max(self._buffer.rfind("\n"), self._buffer.rfind(":"), -1) + 1
        line = self._buffer[line_start:]
        starts = [
            index
            for index, character in enumerate(line)
            if character.isdigit() or character in "+-([{−‐‑‒–﹣－"
        ]
        for relative_start in starts:
            surface = line[relative_start:].strip()
            if not surface or len(surface) > self.limits.max_expression_chars:
                continue
            try:
                normalized = normalize_arithmetic_surface(surface)
            except ArithmeticNormalizationError:
                continue
            if not _OPERATOR.search(normalized):
                continue
            if normalized.count("(") != normalized.count(")"):
                continue
            start = self._buffer_start + line_start + relative_start
            leading = len(line[relative_start:]) - len(line[relative_start:].lstrip())
            start += leading
            end = self._offset + 1
            return DetectionCandidate(
                candidate_id=_candidate_id(self.name, start, end, surface),
                family="compute",
                raw_text=surface,
                start_offset=start,
                end_offset=end,
                detector=self.name,
                metadata={"terminator": "=", "syntax": self.name},
            )
        return None


class CalculatorBlockRecognizer:
    """Emit one candidate only when a line-anchored calculator fence closes."""

    name = "calculator_block_v1"
    _pattern = re.compile(
        r"(?:\A|\n)```calculator[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*\Z"
    )

    def __init__(self, max_buffer_chars: int = 1024, max_expression_chars: int = 256) -> None:
        self.max_buffer_chars = max_buffer_chars
        self.max_expression_chars = max_expression_chars
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._buffer_start = 0
        self._offset = 0

    def feed(self, text: str) -> tuple[DetectionCandidate, ...]:
        candidates: list[DetectionCandidate] = []
        for character in text:
            self._buffer += character
            self._offset += 1
            excess = len(self._buffer) - self.max_buffer_chars
            if excess > 0:
                self._buffer = self._buffer[excess:]
                self._buffer_start += excess
            match = self._pattern.search(self._buffer)
            if match:
                expression = match.group(1).strip()
                if expression and len(expression) <= self.max_expression_chars:
                    start = self._buffer_start + match.start(1)
                    candidates.append(
                        DetectionCandidate(
                            candidate_id=_candidate_id(
                                self.name, start, self._offset, expression
                            ),
                            family="compute",
                            raw_text=expression,
                            start_offset=start,
                            end_offset=self._offset,
                            detector=self.name,
                            metadata={"syntax": self.name, "terminator": "closing_fence"},
                        )
                    )
                self._buffer = ""
                self._buffer_start = self._offset
        return tuple(candidates)


class OracleArithmeticRecognizer(StrictArithmeticRecognizer):
    """Trigger only when the completed strict expression equals the gold expression."""

    name = "oracle_arithmetic_v1"

    def __init__(self, expression: str, limits: RecognizerLimits | None = None) -> None:
        self.expression = "".join(expression.split())
        super().__init__(limits)

    def _candidate_before_equals(self) -> DetectionCandidate | None:
        candidate = super()._candidate_before_equals()
        if candidate is None or "".join(candidate.raw_text.split()) != self.expression:
            return None
        return DetectionCandidate(
            candidate_id=_candidate_id(
                self.name, candidate.start_offset, candidate.end_offset, candidate.raw_text
            ),
            family=candidate.family,
            raw_text=candidate.raw_text,
            start_offset=candidate.start_offset,
            end_offset=candidate.end_offset,
            detector=self.name,
            metadata={**candidate.metadata, "oracle": True},
        )
