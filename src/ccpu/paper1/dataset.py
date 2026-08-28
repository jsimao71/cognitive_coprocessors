"""Deterministic arithmetic scaling benchmark and strict non-trigger controls."""

from __future__ import annotations

import ast
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from ccpu.common.artifacts import fingerprint
from ccpu.common.schema import DetectionCandidate


@dataclass(frozen=True)
class ArithmeticDatasetConfig:
    seed: int = 17
    examples_per_cell: int = 8
    operator_counts: tuple[int, ...] = (1, 2, 4)
    operand_digits: tuple[int, ...] = (1, 2, 3)
    operations: tuple[str, ...] = ("+", "-", "*", "/")
    control_examples: int = 12
    split: str = "test"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArithmeticDatasetConfig:
        dataset = value.get("dataset", value)
        return cls(
            seed=int(dataset.get("seed", 17)),
            examples_per_cell=int(dataset.get("examples_per_cell", 8)),
            operator_counts=tuple(int(item) for item in dataset.get("operator_counts", (1, 2, 4))),
            operand_digits=tuple(int(item) for item in dataset.get("operand_digits", (1, 2, 3))),
            operations=tuple(str(item) for item in dataset.get("operations", ("+", "-", "*", "/"))),
            control_examples=int(dataset.get("control_examples", 12)),
            split=str(dataset.get("split", "test")),
        )

    def validate(self) -> None:
        if self.examples_per_cell < 1 or self.control_examples < 0:
            raise ValueError("example counts must be non-negative and examples_per_cell positive")
        if not self.operator_counts or any(value < 1 for value in self.operator_counts):
            raise ValueError("operator_counts must contain positive values")
        if not self.operand_digits or any(not 1 <= value <= 9 for value in self.operand_digits):
            raise ValueError("operand_digits must be between 1 and 9")
        allowed = {"+", "-", "*", "/", "//", "%", "**"}
        if not self.operations or any(operation not in allowed for operation in self.operations):
            raise ValueError(f"operations must be a non-empty subset of {sorted(allowed)}")

    @property
    def arithmetic_count(self) -> int:
        return self.examples_per_cell * len(self.operator_counts) * len(self.operand_digits)

    @property
    def record_count(self) -> int:
        return self.arithmetic_count + self.control_examples

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardArithmeticDatasetConfig:
    """Factorial hard-arithmetic grid used only by the next-iteration protocol."""

    seed: int = 2718
    examples_per_cell: int = 4
    operator_counts: tuple[int, ...] = (1, 2, 4, 6)
    operand_digits: tuple[int, ...] = (2, 4, 6, 8)
    structures: tuple[str, ...] = ("left_chain", "nested", "multiplication_tree")
    operations: tuple[str, ...] = ("+", "-", "*", "/")
    surface_variants: tuple[str, ...] = ("ascii", "latex", "unicode", "brackets")
    selected_cells: tuple[tuple[int, int, str], ...] = ()
    control_examples: int = 16
    split: str = "adaptive_pilot"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HardArithmeticDatasetConfig:
        dataset = value.get("dataset", value)
        cells = tuple(
            (int(cell["operator_count"]), int(cell["operand_digits"]), str(cell["structure"]))
            for cell in dataset.get("selected_cells", ())
        )
        return cls(
            seed=int(dataset.get("seed", 2718)),
            examples_per_cell=int(dataset.get("examples_per_cell", 4)),
            operator_counts=tuple(int(item) for item in dataset.get("operator_counts", (1, 2, 4, 6))),
            operand_digits=tuple(int(item) for item in dataset.get("operand_digits", (2, 4, 6, 8))),
            structures=tuple(
                str(item)
                for item in dataset.get(
                    "structures", ("left_chain", "nested", "multiplication_tree")
                )
            ),
            operations=tuple(str(item) for item in dataset.get("operations", ("+", "-", "*", "/"))),
            surface_variants=tuple(
                str(item)
                for item in dataset.get("surface_variants", ("ascii", "latex", "unicode", "brackets"))
            ),
            selected_cells=cells,
            control_examples=int(dataset.get("control_examples", 16)),
            split=str(dataset.get("split", "adaptive_pilot")),
        )

    def validate(self) -> None:
        if self.examples_per_cell < 1 or self.control_examples < 0:
            raise ValueError("hard benchmark example counts are invalid")
        if len(self.operand_digits) < 4 or len(self.operator_counts) < 4:
            raise ValueError("hard benchmark requires at least four width and operator bands")
        if any(not 1 <= value <= 12 for value in self.operand_digits):
            raise ValueError("hard benchmark operand digits must be between 1 and 12")
        if any(not 1 <= value <= 7 for value in self.operator_counts):
            raise ValueError("hard benchmark operator counts must be between 1 and 7")
        allowed_structures = {"left_chain", "nested", "multiplication_tree"}
        if len(self.structures) < 3 or any(item not in allowed_structures for item in self.structures):
            raise ValueError(f"hard benchmark structures must cover {sorted(allowed_structures)}")
        if not self.operations or any(item not in {"+", "-", "*", "/"} for item in self.operations):
            raise ValueError("hard benchmark operations must be drawn from +, -, *, /")
        if any(item not in {"ascii", "latex", "unicode", "brackets"} for item in self.surface_variants):
            raise ValueError("unknown arithmetic surface variant")
        valid_cells = {
            (operators, digits, structure)
            for operators in self.operator_counts
            for digits in self.operand_digits
            for structure in self.structures
        }
        if any(cell not in valid_cells for cell in self.selected_cells):
            raise ValueError("selected_cells must belong to the configured factorial grid")

    @property
    def cells(self) -> tuple[tuple[int, int, str], ...]:
        if self.selected_cells:
            return self.selected_cells
        return tuple(
            (operators, digits, structure)
            for operators in self.operator_counts
            for digits in self.operand_digits
            for structure in self.structures
        )

    @property
    def arithmetic_count(self) -> int:
        return self.examples_per_cell * len(self.cells)

    @property
    def record_count(self) -> int:
        return self.arithmetic_count + self.control_examples

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_cells"] = [
            {"operator_count": operators, "operand_digits": digits, "structure": structure}
            for operators, digits, structure in self.selected_cells
        ]
        return value


@dataclass(frozen=True)
class ArithmeticExample:
    example_id: str
    schema_version: str
    task_kind: str
    split: str
    prompt: str
    reference_completion: str
    should_trigger: bool
    expression: str | None
    answer: str | None
    difficulty: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArithmeticExample:
        return cls(
            example_id=str(value["example_id"]),
            schema_version=str(value["schema_version"]),
            task_kind=str(value["task_kind"]),
            split=str(value["split"]),
            prompt=str(value["prompt"]),
            reference_completion=str(value["reference_completion"]),
            should_trigger=bool(value["should_trigger"]),
            expression=str(value["expression"]) if value.get("expression") is not None else None,
            answer=str(value["answer"]) if value.get("answer") is not None else None,
            difficulty=dict(value.get("difficulty", {})),
            metadata=dict(value.get("metadata", {})),
        )


_CONTROL_COMPLETIONS = (
    "The release year is 2026.",
    'The quoted fragment "2 + 2 =" is intentionally incomplete.',
    "The equation x + 2 = 5 contains a variable.",
    "Version 2.5 = stable is not valid arithmetic syntax.",
    "Use a == b when demonstrating equality in this pseudocode.",
    "Room 12 and room 7 are on different floors.",
    "The discount range is 10-20 percent.",
    "Parentheses (like these) do not imply a calculation.",
)

_HARD_CONTROL_COMPLETIONS = _CONTROL_COMPLETIONS + (
    "> ```calculator\n2 + 2\n``` is a quoted Markdown example.",
    "Inline text ```calculator 2 + 2 ``` is not an execution block.",
    r"The unrelated LaTeX commands \times and \div are vocabulary here.",
    'Quoted code "[7 × 8] =" must remain inert.',
)


def _operand(rng: random.Random, digits: int, *, exponent: bool = False) -> int:
    if exponent:
        return rng.randint(0, min(5, 10**digits - 1))
    lower = 1 if digits == 1 else 10 ** (digits - 1)
    return rng.randint(lower, 10**digits - 1)


def _expression(
    rng: random.Random,
    operator_count: int,
    digits: int,
    operations: tuple[str, ...],
) -> str:
    expression = str(_operand(rng, digits))
    for _ in range(operator_count):
        operation = rng.choice(operations)
        right = _operand(rng, digits, exponent=operation == "**")
        expression = f"({expression} {operation} {right})"
    return expression


def reference_answer(expression: str) -> str:
    """Independent exact-rational oracle; it does not call the tested engine."""

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value, 1)
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return Fraction(left // right, 1)
            if isinstance(node.op, ast.Mod):
                return left - right * (left // right)
            if isinstance(node.op, ast.Pow) and right.denominator == 1:
                return left**right.numerator
        raise ValueError(f"unsupported generated oracle syntax: {ast.dump(node)}")

    value = evaluate(ast.parse(expression, mode="eval").body)
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def iter_dataset(config: ArithmeticDatasetConfig) -> Iterable[ArithmeticExample]:
    config.validate()
    for operator_count in config.operator_counts:
        for digits in config.operand_digits:
            for replicate in range(config.examples_per_cell):
                cell_seed = int(
                    fingerprint(
                        {
                            "seed": config.seed,
                            "operator_count": operator_count,
                            "digits": digits,
                            "replicate": replicate,
                        },
                        15,
                    ),
                    16,
                )
                rng = random.Random(cell_seed)
                expression = _expression(rng, operator_count, digits, config.operations)
                answer = reference_answer(expression)
                identity = {
                    "schema": "ccpu.paper1.example.v1",
                    "seed": config.seed,
                    "operator_count": operator_count,
                    "digits": digits,
                    "replicate": replicate,
                    "expression": expression,
                }
                yield ArithmeticExample(
                    example_id=f"p1-arithmetic-{fingerprint(identity, 16)}",
                    schema_version="ccpu.paper1.example.v1",
                    task_kind="arithmetic",
                    split=config.split,
                    prompt=(
                        "Compute the exact value of the integer arithmetic expression below.\n"
                        f"Expression: {expression}"
                    ),
                    reference_completion=f"Calculation: {expression} =",
                    should_trigger=True,
                    expression=expression,
                    answer=answer,
                    difficulty={"operator_count": operator_count, "operand_digits": digits},
                    metadata={"generator_seed": cell_seed, "replicate": replicate},
                )

    for index in range(config.control_examples):
        completion = _CONTROL_COMPLETIONS[index % len(_CONTROL_COMPLETIONS)]
        identity = {"seed": config.seed, "control": index, "completion": completion}
        yield ArithmeticExample(
            example_id=f"p1-control-{fingerprint(identity, 16)}",
            schema_version="ccpu.paper1.example.v1",
            task_kind="control",
            split=config.split,
            prompt="Repeat the supplied control sentence exactly without adding arithmetic.",
            reference_completion=completion,
            should_trigger=False,
            expression=None,
            answer=None,
            difficulty={"control_family": index % len(_CONTROL_COMPLETIONS)},
            metadata={"control_index": index},
        )


def _hard_expression(
    rng: random.Random,
    operator_count: int,
    digits: int,
    operations: tuple[str, ...],
    structure: str,
) -> str:
    operands = [str(_operand(rng, digits)) for _ in range(operator_count + 1)]
    selected = [rng.choice(operations) for _ in range(operator_count)]
    if operator_count >= 2 and "*" in operations:
        selected[0] = "*"
    if operator_count >= 3 and "/" in operations:
        selected[-1] = "/"

    if structure == "multiplication_tree":
        selected = ["*" for _ in selected]

    if structure == "left_chain":
        expression = operands[0]
        for operation, operand in zip(selected, operands[1:], strict=True):
            expression = f"({expression} {operation} {operand})"
        return expression

    def tree(values: list[str], operators: list[str]) -> str:
        if not operators:
            return values[0]
        split = len(operators) // 2
        left = tree(values[: split + 1], operators[:split])
        right = tree(values[split + 1 :], operators[split + 1 :])
        return f"({left} {operators[split]} {right})"

    expression = tree(operands, selected)
    if structure == "nested" and operator_count > 1:
        return f"({expression})"
    return expression


def _surface_expression(expression: str, variant: str) -> str:
    if variant == "latex":
        return expression.replace(" * ", r" \times ").replace(" / ", r" \div ")
    if variant == "unicode":
        return expression.replace(" * ", " × ").replace(" / ", " ÷ ").replace(" - ", " − ")
    if variant == "brackets":
        return expression.replace("(", "[").replace(")", "]")
    return expression


def iter_hard_dataset(config: HardArithmeticDatasetConfig) -> Iterable[ArithmeticExample]:
    """Generate a bounded hard grid with deterministic retries for safe division."""

    from .arithmetic import ArithmeticNormalizer, BoundedCalculator

    config.validate()
    for operator_count, digits, structure in config.cells:
        for replicate in range(config.examples_per_cell):
            base_identity = {
                "schema": "ccpu.paper1.hard_example.v1",
                "seed": config.seed,
                "operator_count": operator_count,
                "digits": digits,
                "structure": structure,
                "replicate": replicate,
            }
            cell_seed = int(fingerprint(base_identity, 15), 16)
            expression = ""
            answer = ""
            for attempt in range(100):
                rng = random.Random(cell_seed + attempt)
                expression = _hard_expression(
                    rng, operator_count, digits, config.operations, structure
                )
                try:
                    answer = reference_answer(expression)
                    detector_candidate = DetectionCandidate(
                        candidate_id="hard-generator-check",
                        family="compute",
                        raw_text=expression,
                        start_offset=0,
                        end_offset=len(expression),
                        detector="hard_generator_v1",
                    )
                    request = ArithmeticNormalizer().normalize(detector_candidate)
                    result = BoundedCalculator().execute(request)
                    if result.ok and result.display == answer:
                        break
                except (ValueError, ZeroDivisionError):
                    pass
            else:
                raise RuntimeError(f"could not generate bounded hard expression: {base_identity}")

            variant = config.surface_variants[replicate % len(config.surface_variants)]
            surface = _surface_expression(expression, variant)
            identity = {**base_identity, "expression": expression, "surface": surface}
            yield ArithmeticExample(
                example_id=f"p1-hard-{fingerprint(identity, 16)}",
                schema_version="ccpu.paper1.hard_example.v1",
                task_kind="arithmetic",
                split=config.split,
                prompt=(
                    "Compute the exact value of the arithmetic expression below. "
                    "Return an integer or reduced fraction.\n"
                    f"Expression: {surface}"
                ),
                reference_completion=f"Calculation: {surface} =",
                should_trigger=True,
                expression=expression,
                answer=answer,
                difficulty={
                    "operator_count": operator_count,
                    "operand_digits": digits,
                    "structure": structure,
                    "surface_variant": variant,
                    "multiplication_depth": operator_count if structure == "multiplication_tree" else 0,
                },
                metadata={
                    "generator": "hard_arithmetic_v1",
                    "generator_seed": cell_seed,
                    "replicate": replicate,
                    "surface_expression": surface,
                },
            )

    for index in range(config.control_examples):
        completion = _HARD_CONTROL_COMPLETIONS[index % len(_HARD_CONTROL_COMPLETIONS)]
        identity = {"schema": "ccpu.paper1.hard_example.v1", "seed": config.seed, "control": index}
        yield ArithmeticExample(
            example_id=f"p1-hard-control-{fingerprint(identity, 16)}",
            schema_version="ccpu.paper1.hard_example.v1",
            task_kind="control",
            split=config.split,
            prompt="Repeat the supplied control sentence exactly without executing it.",
            reference_completion=completion,
            should_trigger=False,
            expression=None,
            answer=None,
            difficulty={"control_family": index % len(_HARD_CONTROL_COMPLETIONS)},
            metadata={"generator": "hard_arithmetic_v1", "control_index": index},
        )
