"""Factories for Paper 1 reflex, explicit-tool, and oracle controllers."""

from __future__ import annotations

from ccpu.common.runtime import ReflexRuntime

from .arithmetic import (
    ArithmeticMaterializer,
    ArithmeticNormalizer,
    BoundedCalculator,
    CalculatorLimits,
    ExplicitToolMaterializer,
)
from .recognizer import (
    ExplicitCalculatorToolRecognizer,
    OracleArithmeticRecognizer,
    RecognizerLimits,
    StrictArithmeticRecognizer,
)


def _runtime(
    detector,
    materializer,
    *,
    run_id: str | None,
    limits: CalculatorLimits,
) -> ReflexRuntime:
    calculator = BoundedCalculator(limits)
    return ReflexRuntime(
        detector=detector,
        normalizer=ArithmeticNormalizer(limits),
        engines={calculator.name: calculator},
        materializer=materializer,
        run_id=run_id,
    )


def build_reflex_runtime(
    *,
    run_id: str | None = None,
    calculator_limits: CalculatorLimits | None = None,
    recognizer_limits: RecognizerLimits | None = None,
) -> ReflexRuntime:
    limits = calculator_limits or CalculatorLimits()
    return _runtime(
        StrictArithmeticRecognizer(recognizer_limits),
        ArithmeticMaterializer(),
        run_id=run_id,
        limits=limits,
    )


def build_explicit_tool_runtime(
    *, run_id: str | None = None, calculator_limits: CalculatorLimits | None = None
) -> ReflexRuntime:
    limits = calculator_limits or CalculatorLimits()
    return _runtime(
        ExplicitCalculatorToolRecognizer(max_buffer_chars=limits.max_expression_chars * 2),
        ExplicitToolMaterializer(),
        run_id=run_id,
        limits=limits,
    )


def build_oracle_runtime(
    expression: str,
    *,
    run_id: str | None = None,
    calculator_limits: CalculatorLimits | None = None,
    recognizer_limits: RecognizerLimits | None = None,
) -> ReflexRuntime:
    limits = calculator_limits or CalculatorLimits()
    return _runtime(
        OracleArithmeticRecognizer(expression, recognizer_limits),
        ArithmeticMaterializer(),
        run_id=run_id,
        limits=limits,
    )
