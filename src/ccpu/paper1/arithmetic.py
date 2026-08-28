"""Strict arithmetic micro-IR and bounded deterministic calculator engine."""

from __future__ import annotations

import ast
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from ccpu.common.schema import (
    CoprocessorRequest,
    CoprocessorResult,
    DetectionCandidate,
    MicroStateItem,
)


class ArithmeticNormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class CalculatorLimits:
    max_expression_chars: int = 256
    max_ast_nodes: int = 64
    max_ast_depth: int = 16
    max_integer_digits: int = 64
    max_exponent: int = 12
    max_value_bits: int = 512
    max_stack_items: int = 64
    timeout_ms: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BINARY_OPS = {
    ast.Add: "ADD",
    ast.Sub: "SUB",
    ast.Mult: "MUL",
    ast.Div: "DIV",
    ast.FloorDiv: "FLOOR_DIV",
    ast.Mod: "MOD",
    ast.Pow: "POW",
}
_UNARY_OPS = {ast.UAdd: "POS", ast.USub: "NEG"}


class ArithmeticNormalizer:
    """Convert an allowlisted integer expression to canonical postfix micro-IR."""

    def __init__(self, limits: CalculatorLimits | None = None) -> None:
        self.limits = limits or CalculatorLimits()

    def normalize(self, candidate: DetectionCandidate) -> CoprocessorRequest:
        expression = candidate.raw_text.strip()
        if not expression:
            raise ArithmeticNormalizationError("empty expression")
        if len(expression) > self.limits.max_expression_chars:
            raise ArithmeticNormalizationError("expression exceeds character budget")
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError) as error:
            message = getattr(error, "msg", str(error))
            raise ArithmeticNormalizationError(f"invalid arithmetic syntax: {message}") from error

        instructions: list[dict[str, Any]] = []
        counters = {"nodes": 0, "binary": 0}
        self._emit(tree.body, instructions, counters, depth=1)
        if not counters["binary"]:
            raise ArithmeticNormalizationError("a reflex expression requires a binary operator")
        canonical = "".join(expression.split())
        return CoprocessorRequest(
            request_id=f"{candidate.candidate_id}:arithmetic",
            candidate_id=candidate.candidate_id,
            family="compute",
            operation="arithmetic.evaluate",
            engine="calculator",
            payload={
                "schema": "ccpu.arithmetic.postfix.v1",
                "expression": expression,
                "canonical_expression": canonical,
                "instructions": instructions,
            },
            confidence=1.0,
            budget=self.limits.to_dict(),
            metadata={"source_start": candidate.start_offset, "source_end": candidate.end_offset},
        )

    def _emit(
        self,
        node: ast.AST,
        instructions: list[dict[str, Any]],
        counters: dict[str, int],
        *,
        depth: int,
    ) -> None:
        counters["nodes"] += 1
        if counters["nodes"] > self.limits.max_ast_nodes:
            raise ArithmeticNormalizationError("expression exceeds AST node budget")
        if depth > self.limits.max_ast_depth:
            raise ArithmeticNormalizationError("expression exceeds AST depth budget")

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise ArithmeticNormalizationError("only integer literals are allowed")
            if len(str(abs(node.value))) > self.limits.max_integer_digits:
                raise ArithmeticNormalizationError("integer literal exceeds digit budget")
            instructions.append({"op": "PUSH", "value": str(node.value)})
            return

        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            self._emit(node.operand, instructions, counters, depth=depth + 1)
            instructions.append({"op": _UNARY_OPS[type(node.op)]})
            return

        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            counters["binary"] += 1
            self._emit(node.left, instructions, counters, depth=depth + 1)
            self._emit(node.right, instructions, counters, depth=depth + 1)
            instructions.append({"op": _BINARY_OPS[type(node.op)]})
            return

        raise ArithmeticNormalizationError(f"disallowed syntax: {type(node).__name__}")


class BoundedCalculator:
    name = "calculator"

    def __init__(
        self,
        limits: CalculatorLimits | None = None,
        timer_ns=time.perf_counter_ns,
    ) -> None:
        self.limits = limits or CalculatorLimits()
        self._timer_ns = timer_ns

    def execute(self, request: CoprocessorRequest) -> CoprocessorResult:
        if request.operation != "arithmetic.evaluate":
            return self._failure(request, "unsupported_operation", request.operation)
        instructions = request.payload.get("instructions")
        if not isinstance(instructions, list):
            return self._failure(request, "invalid_ir", "instructions must be a list")

        started = self._timer_ns()
        deadline = started + int(self.limits.timeout_ms * 1_000_000)
        stack: list[Fraction] = []
        try:
            for index, instruction in enumerate(instructions):
                if self._timer_ns() > deadline:
                    raise TimeoutError("calculator time budget exceeded")
                if not isinstance(instruction, dict) or "op" not in instruction:
                    raise ValueError(f"invalid instruction at index {index}")
                op = instruction["op"]
                if op == "PUSH":
                    stack.append(Fraction(int(instruction["value"]), 1))
                elif op in {"POS", "NEG"}:
                    if not stack:
                        raise ValueError("unary operation underflow")
                    value = stack.pop()
                    stack.append(value if op == "POS" else -value)
                else:
                    if len(stack) < 2:
                        raise ValueError("binary operation underflow")
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(self._binary(op, left, right))
                if len(stack) > self.limits.max_stack_items:
                    raise OverflowError("calculator stack budget exceeded")
                if stack:
                    self._check_size(stack[-1])
            if len(stack) != 1:
                raise ValueError("micro-IR did not reduce to one value")
            value = stack[0]
            display = (
                str(value.numerator)
                if value.denominator == 1
                else f"{value.numerator}/{value.denominator}"
            )
            return CoprocessorResult(
                request_id=request.request_id,
                engine=self.name,
                ok=True,
                value={"numerator": value.numerator, "denominator": value.denominator},
                display=display,
                metadata={
                    "exact": True,
                    "instruction_count": len(instructions),
                    "duration_ns": self._timer_ns() - started,
                },
            )
        except TimeoutError as error:
            return self._failure(request, "timeout", str(error), started)
        except ZeroDivisionError as error:
            return self._failure(request, "division_by_zero", str(error), started)
        except (KeyError, TypeError, ValueError) as error:
            return self._failure(request, "invalid_ir", str(error), started)
        except OverflowError as error:
            return self._failure(request, "resource_limit", str(error), started)

    def _binary(self, op: str, left: Fraction, right: Fraction) -> Fraction:
        if op == "ADD":
            return left + right
        if op == "SUB":
            return left - right
        if op == "MUL":
            return left * right
        if op == "DIV":
            return left / right
        if op == "FLOOR_DIV":
            return Fraction(left // right, 1)
        if op == "MOD":
            quotient = left // right
            return left - right * quotient
        if op == "POW":
            if right.denominator != 1 or abs(right.numerator) > self.limits.max_exponent:
                raise OverflowError("exponent exceeds bounded integer policy")
            return left**right.numerator
        raise ValueError(f"unknown arithmetic operation: {op}")

    def _check_size(self, value: Fraction) -> None:
        if (
            abs(value.numerator).bit_length() > self.limits.max_value_bits
            or value.denominator.bit_length() > self.limits.max_value_bits
        ):
            raise OverflowError("intermediate value exceeds bit budget")

    def _failure(
        self,
        request: CoprocessorRequest,
        code: str,
        message: str,
        started_ns: int | None = None,
    ) -> CoprocessorResult:
        metadata = {}
        if started_ns is not None:
            metadata["duration_ns"] = self._timer_ns() - started_ns
        return CoprocessorResult(
            request_id=request.request_id,
            engine=self.name,
            ok=False,
            error_code=code,
            error_message=message,
            metadata=metadata,
        )


class ArithmeticMaterializer:
    """Compact text baseline for exposing an exact result to future decoding."""

    def __init__(self, *, prefix: str = " ", suffix: str = "") -> None:
        self.prefix = prefix
        self.suffix = suffix

    def materialize(
        self,
        request: CoprocessorRequest,
        result: CoprocessorResult,
        state: MicroStateItem,
    ) -> str:
        del request, state
        if not result.ok:
            raise ValueError("cannot materialize a failed calculator result")
        return f"{self.prefix}{result.display}{self.suffix}"


class ExplicitToolMaterializer(ArithmeticMaterializer):
    def __init__(self) -> None:
        super().__init__(prefix="<tool_result>", suffix="</tool_result>")
