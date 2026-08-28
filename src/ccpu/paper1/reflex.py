"""Factories for Paper 1 reflex, explicit-tool, and oracle controllers."""

from __future__ import annotations

from ccpu.common.runtime import ReflexRuntime

from .arithmetic import (
    ArithmeticMaterializer,
    ArithmeticNormalizer,
    BoundedCalculator,
    CalculatorBlockMaterializer,
    CalculatorLimits,
    ExplicitToolMaterializer,
)
from .recognizer import (
    CalculatorBlockRecognizer,
    ExplicitCalculatorToolRecognizer,
    NormalizedArithmeticRecognizer,
    OracleArithmeticRecognizer,
    RecognizerLimits,
    StrictArithmeticRecognizer,
)
from .surface import ArithmeticSurfaceNormalizer


def _runtime(
    detector,
    materializer,
    *,
    run_id: str | None,
    limits: CalculatorLimits,
    normalizer=None,
) -> ReflexRuntime:
    calculator = BoundedCalculator(limits)
    return ReflexRuntime(
        detector=detector,
        normalizer=normalizer or ArithmeticNormalizer(limits),
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


def build_normalized_reflex_runtime(
    *,
    run_id: str | None = None,
    calculator_limits: CalculatorLimits | None = None,
    recognizer_limits: RecognizerLimits | None = None,
) -> ReflexRuntime:
    limits = calculator_limits or CalculatorLimits()
    return _runtime(
        NormalizedArithmeticRecognizer(recognizer_limits),
        ArithmeticMaterializer(),
        run_id=run_id,
        limits=limits,
        normalizer=ArithmeticSurfaceNormalizer(limits),
    )


def build_calculator_block_runtime(
    *, run_id: str | None = None, calculator_limits: CalculatorLimits | None = None
) -> ReflexRuntime:
    limits = calculator_limits or CalculatorLimits()
    return _runtime(
        CalculatorBlockRecognizer(
            max_buffer_chars=limits.max_expression_chars * 4,
            max_expression_chars=limits.max_expression_chars,
        ),
        CalculatorBlockMaterializer(),
        run_id=run_id,
        limits=limits,
        normalizer=ArithmeticSurfaceNormalizer(limits),
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
