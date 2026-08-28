import pytest

from ccpu.common.schema import DetectionCandidate
from ccpu.paper1.arithmetic import (
    ArithmeticNormalizationError,
    ArithmeticNormalizer,
    BoundedCalculator,
    CalculatorLimits,
)


def candidate(expression: str) -> DetectionCandidate:
    return DetectionCandidate(
        candidate_id="candidate",
        family="compute",
        raw_text=expression,
        start_offset=0,
        end_offset=len(expression) + 1,
        detector="test",
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(7 + 5) * 3", "36"),
        ("7 / 2", "7/2"),
        ("-7 // 2", "-4"),
        ("-7 % 2", "1"),
        ("2 ** 10", "1024"),
        ("1 + 2 * 3", "7"),
    ],
)
def test_exact_bounded_arithmetic(expression, expected):
    request = ArithmeticNormalizer().normalize(candidate(expression))
    result = BoundedCalculator().execute(request)

    assert result.ok
    assert result.display == expected
    assert request.payload["schema"] == "ccpu.arithmetic.postfix.v1"


@pytest.mark.parametrize(
    "expression",
    [
        "2.5 + 1",
        "name + 1",
        "abs(-2)",
        "[1, 2][0] + 3",
        "__import__('os').system('echo unsafe')",
        "-2",
    ],
)
def test_normalizer_rejects_non_integer_or_non_arithmetic_syntax(expression):
    with pytest.raises(ArithmeticNormalizationError):
        ArithmeticNormalizer().normalize(candidate(expression))


def test_division_by_zero_is_a_typed_failure():
    request = ArithmeticNormalizer().normalize(candidate("4 / 0"))
    result = BoundedCalculator().execute(request)

    assert not result.ok
    assert result.error_code == "division_by_zero"


def test_exponent_policy_bounds_work_before_large_allocation():
    limits = CalculatorLimits(max_exponent=5)
    request = ArithmeticNormalizer(limits).normalize(candidate("2 ** 6"))
    result = BoundedCalculator(limits).execute(request)

    assert not result.ok
    assert result.error_code == "resource_limit"


def test_literal_and_intermediate_size_limits_are_distinct():
    literal_limits = CalculatorLimits(max_integer_digits=2)
    with pytest.raises(ArithmeticNormalizationError, match="digit budget"):
        ArithmeticNormalizer(literal_limits).normalize(candidate("100 + 1"))

    result_limits = CalculatorLimits(max_value_bits=8)
    request = ArithmeticNormalizer(result_limits).normalize(candidate("20 * 20"))
    result = BoundedCalculator(result_limits).execute(request)
    assert not result.ok
    assert result.error_code == "resource_limit"
