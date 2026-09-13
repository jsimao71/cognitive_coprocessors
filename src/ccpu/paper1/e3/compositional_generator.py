"""Deterministic graph-first word-problem generation for Paper 1.

The generator deliberately separates semantic construction from surface
realization: each example is a typed dependency graph before any prose is
rendered.  Gold ASL is lowered from that graph and accepted only after the
repository's ASL validator executes it to the independently computed answer.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl
from ccpu.dsl.registry import ARITHMETIC_FUNCTIONS

GENERATOR_VERSION = "paper1-compositional-word-problems-v1"
SCHEMA_VERSION = "ccpu.paper1.compositional_word_problem.v1"
PROTOCOL_ID = "paper1_compositional_complexity_v1"
Tier = Literal["C1", "C2", "C3", "C4"]
Split = Literal["train", "dev", "test"]


@dataclass(frozen=True)
class GenerationControls:
    """Requested semantic complexity for one frozen pilot tier."""

    tier: Tier
    operator_count: tuple[int, int]
    dependency_depth: tuple[int, int]
    function_nesting_depth: tuple[int, int]
    minimum_entity_count: int
    minimum_binding_count: int
    graph_structure: str
    distractor_count: int
    operator_families: tuple[str, ...]


TIER_CONTROLS: dict[Tier, GenerationControls] = {
    "C1": GenerationControls(
        tier="C1",
        operator_count=(1, 1),
        dependency_depth=(1, 1),
        function_nesting_depth=(0, 0),
        minimum_entity_count=2,
        minimum_binding_count=3,
        graph_structure="single",
        distractor_count=0,
        operator_families=("O0",),
    ),
    "C2": GenerationControls(
        tier="C2",
        operator_count=(2, 2),
        dependency_depth=(2, 2),
        function_nesting_depth=(0, 0),
        minimum_entity_count=2,
        minimum_binding_count=5,
        graph_structure="chain",
        distractor_count=0,
        operator_families=("O0",),
    ),
    "C3": GenerationControls(
        tier="C3",
        operator_count=(3, 4),
        dependency_depth=(2, 3),
        function_nesting_depth=(0, 0),
        minimum_entity_count=4,
        minimum_binding_count=8,
        graph_structure="branch_merge",
        distractor_count=1,
        operator_families=("O0",),
    ),
    "C4": GenerationControls(
        tier="C4",
        operator_count=(4, 4),
        dependency_depth=(4, 4),
        function_nesting_depth=(2, 2),
        minimum_entity_count=5,
        minimum_binding_count=8,
        graph_structure="nested",
        distractor_count=2,
        operator_families=("O0", "O1", "O5", "O6"),
    ),
}


@dataclass(frozen=True)
class Expr:
    kind: Literal["constant", "reference", "binary", "call"]
    value: Decimal | str
    arguments: tuple[Expr, ...] = ()

    def to_asl(self) -> str:
        if self.kind == "constant":
            return _decimal_text(Decimal(self.value))
        if self.kind == "reference":
            return str(self.value)
        if self.kind == "binary":
            left, right = self.arguments
            return f"{left.to_asl()} {self.value} {right.to_asl()}"
        return f"{self.value}({', '.join(item.to_asl() for item in self.arguments)})"

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "constant":
            return {"kind": self.kind, "value": _decimal_text(Decimal(self.value))}
        if self.kind == "reference":
            return {"kind": self.kind, "path": self.value}
        return {
            "kind": self.kind,
            "operator": self.value,
            "arguments": [argument.to_dict() for argument in self.arguments],
        }


@dataclass(frozen=True)
class Binding:
    path: str
    value: Decimal
    value_type: str
    entity: str
    disconnected_distractor: bool = False


@dataclass(frozen=True)
class Operation:
    path: str
    expression: Expr
    value_type: str
    entity: str


@dataclass(frozen=True)
class SemanticGraph:
    tier: Tier
    graph_schema_id: str
    graph_structure: str
    facts: tuple[Binding, ...]
    operations: tuple[Operation, ...]
    answer_path: str
    render_values: dict[str, int]


def constant(value: int | Decimal) -> Expr:
    return Expr("constant", Decimal(value))


def reference(path: str) -> Expr:
    return Expr("reference", path)


def binary(operator: str, left: Expr, right: Expr) -> Expr:
    return Expr("binary", operator, (left, right))


def call(operator: str, *arguments: Expr) -> Expr:
    return Expr("call", operator, tuple(arguments))


_SURFACE_FAMILIES: dict[Tier, dict[Split, tuple[str, ...]]] = {
    "C1": {
        "train": ("c1_shop_ledger", "c1_depot_log"),
        "dev": ("c1_school_inventory",),
        "test": ("c1_museum_inventory",),
    },
    "C2": {
        "train": ("c2_shipping_report", "c2_relief_supplies"),
        "dev": ("c2_library_packing",),
        "test": ("c2_expedition_manifest",),
    },
    "C3": {
        "train": ("c3_market_merge", "c3_factory_merge"),
        "dev": ("c3_festival_merge",),
        "test": ("c3_research_supply_merge",),
    },
    "C4": {
        "train": ("c4_engineering_plan", "c4_construction_plan"),
        "dev": ("c4_field_station_plan",),
        "test": ("c4_orbital_habitat_plan",),
    },
}


def _decimal_text(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _stable_seed(seed: int, tier: Tier, split: Split, index: int, retry: int) -> int:
    label = f"{GENERATOR_VERSION}:{seed}:{tier}:{split}:{index}:{retry}"
    return int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:8], "big")


def _fact(
    path: str,
    value: int,
    value_type: str,
    *,
    distractor: bool = False,
) -> Binding:
    return Binding(
        path=path,
        value=Decimal(value),
        value_type=value_type,
        entity=path.split(".", 1)[0],
        disconnected_distractor=distractor,
    )


def _operation(path: str, expression: Expr, value_type: str) -> Operation:
    return Operation(path, expression, value_type, path.split(".", 1)[0])


def _build_c1(rng: random.Random, index: int) -> SemanticGraph:
    operator = ("+", "-", "*", "/")[index % 4]
    if operator == "+":
        left, right = rng.randint(12, 80), rng.randint(3, 35)
        facts = (_fact("stock.initial", left, "count"), _fact("delivery.items", right, "count"))
        schema = "inventory_add"
    elif operator == "-":
        right = rng.randint(3, 25)
        left = rng.randint(right + 10, right + 90)
        facts = (_fact("stock.initial", left, "count"), _fact("order.items", right, "count"))
        schema = "inventory_subtract"
    elif operator == "*":
        left, right = rng.randint(3, 18), rng.randint(4, 24)
        facts = (_fact("shipment.boxes", left, "count"), _fact("box.items", right, "count"))
        schema = "inventory_multiply"
    else:
        right, quotient = rng.randint(2, 12), rng.randint(3, 20)
        left = right * quotient
        facts = (_fact("shipment.items", left, "count"), _fact("team.count", right, "count"))
        schema = "inventory_divide"
    operation = _operation(
        "answer.value",
        binary(operator, reference(facts[0].path), reference(facts[1].path)),
        "count",
    )
    return SemanticGraph(
        "C1",
        schema,
        "single",
        facts,
        (operation,),
        operation.path,
        {"left": int(facts[0].value), "right": int(facts[1].value), "operator": operator},
    )


def _build_c2(rng: random.Random, index: int) -> SemanticGraph:
    if index % 2 == 0:
        boxes, per_box = rng.randint(4, 18), rng.randint(5, 24)
        damaged = rng.randint(2, min(35, boxes * per_box - 1))
        facts = (
            _fact("shipment.boxes", boxes, "count"),
            _fact("box.items", per_box, "count"),
            _fact("damage.items", damaged, "count"),
        )
        operations = (
            _operation(
                "shipment.packed_items",
                binary("*", reference("shipment.boxes"), reference("box.items")),
                "count",
            ),
            _operation(
                "shipment.usable_items",
                binary("-", reference("shipment.packed_items"), reference("damage.items")),
                "count",
            ),
        )
        values = {"boxes": boxes, "per_box": per_box, "damaged": damaged, "schema": 0}
        schema = "pack_then_remove"
    else:
        teams = rng.randint(2, 9)
        each = rng.randint(5, 25)
        first = rng.randint(1, teams * each - 1)
        second = teams * each - first
        facts = (
            _fact("class_a.students", first, "count"),
            _fact("class_b.students", second, "count"),
            _fact("team.count", teams, "count"),
        )
        operations = (
            _operation(
                "workshop.students",
                binary("+", reference("class_a.students"), reference("class_b.students")),
                "count",
            ),
            _operation(
                "team.students",
                binary("/", reference("workshop.students"), reference("team.count")),
                "count",
            ),
        )
        values = {"first": first, "second": second, "teams": teams, "schema": 1}
        schema = "combine_then_share"
    return SemanticGraph("C2", schema, "chain", facts, operations, operations[-1].path, values)


def _build_c3(rng: random.Random, index: int) -> SemanticGraph:
    red_boxes, red_each = rng.randint(3, 12), rng.randint(4, 18)
    blue_boxes, blue_each = rng.randint(3, 12), rng.randint(4, 18)
    display_posters = rng.randint(7, 40)
    use_four = index % 2 == 1
    removed = rng.randint(2, min(30, red_boxes * red_each + blue_boxes * blue_each - 1))
    facts = (
        _fact("red.boxes", red_boxes, "count"),
        _fact("red.items_per_box", red_each, "count"),
        _fact("blue.boxes", blue_boxes, "count"),
        _fact("blue.items_per_box", blue_each, "count"),
        _fact("display.posters", display_posters, "count", distractor=True),
    )
    if use_four:
        facts += (_fact("quality.rejected_items", removed, "count"),)
    operations = (
        _operation(
            "red.items",
            binary("*", reference("red.boxes"), reference("red.items_per_box")),
            "count",
        ),
        _operation(
            "blue.items",
            binary("*", reference("blue.boxes"), reference("blue.items_per_box")),
            "count",
        ),
        _operation(
            "inventory.combined_items",
            binary("+", reference("red.items"), reference("blue.items")),
            "count",
        ),
    )
    if use_four:
        operations += (
            _operation(
                "inventory.accepted_items",
                binary(
                    "-",
                    reference("inventory.combined_items"),
                    reference("quality.rejected_items"),
                ),
                "count",
            ),
        )
    return SemanticGraph(
        "C3",
        "parallel_batches_merge" + ("_then_filter" if use_four else ""),
        "branch_merge",
        facts,
        operations,
        operations[-1].path,
        {
            "red_boxes": red_boxes,
            "red_each": red_each,
            "blue_boxes": blue_boxes,
            "blue_each": blue_each,
            "posters": display_posters,
            "removed": removed,
            "use_four": int(use_four),
        },
    )


def _build_c4(rng: random.Random, index: int) -> SemanticGraph:
    specialized = ("O1", "O5", "O6")[index % 3]
    units = rng.randint(2, 8)
    rejected = rng.randint(2, 25)
    distractors = (
        _fact("weather.temperature", rng.randint(12, 31), "temperature", distractor=True),
        _fact("vehicle.route_km", rng.randint(20, 90), "distance", distractor=True),
    )
    if specialized == "O1":
        side = rng.randint(3, 12)
        face_area = side * side
        capacity = side**3
        rejected = min(rejected, capacity * units - 1)
        facts = (
            _fact("crate.face_area", face_area, "area"),
            _fact("shipment.crates", units, "count"),
            _fact("quality.rejected_blocks", rejected, "count"),
        ) + distractors
        nested = call("cube", call("sqrt", reference("crate.face_area")))
        schema, specialized_name = "cube_from_face_area", "sqrt_then_cube"
        units_path, rejected_path = "shipment.crates", "quality.rejected_blocks"
        values = {"face_area": face_area, "units": units, "rejected": rejected}
    elif specialized == "O5":
        side = rng.randint(4, 18)
        coefficient = rng.randint(2, 8)
        offset = rng.randint(2, 20)
        total = coefficient * side + offset
        capacity = side * side
        rejected = min(rejected, capacity * units - 1)
        facts = (
            _fact("equation.coefficient", coefficient, "scalar"),
            _fact("equation.offset", offset, "scalar"),
            _fact("equation.total", total, "scalar"),
            _fact("building.rooms", units, "count"),
            _fact("quality.broken_tiles", rejected, "count"),
        ) + distractors
        nested = call(
            "square",
            call(
                "solve_linear",
                reference("equation.coefficient"),
                reference("equation.offset"),
                reference("equation.total"),
            ),
        )
        schema, specialized_name = "square_from_linear_side", "solve_linear_then_square"
        units_path, rejected_path = "building.rooms", "quality.broken_tiles"
        values = {
            "coefficient": coefficient,
            "offset": offset,
            "total": total,
            "units": units,
            "rejected": rejected,
        }
    else:
        side = rng.randint(4, 18)
        negative_root = -rng.randint(2, 10)
        linear = -(side + negative_root)
        quadratic_constant = side * negative_root
        capacity = side * side
        rejected = min(rejected, capacity * units - 1)
        facts = (
            _fact("equation.a", 1, "scalar"),
            _fact("equation.b", linear, "scalar"),
            _fact("equation.c", quadratic_constant, "scalar"),
            _fact("garden.plots", units, "count"),
            _fact("garden.empty_spots", rejected, "count"),
        ) + distractors
        nested = call(
            "square",
            call(
                "positive_quadratic_root",
                reference("equation.a"),
                reference("equation.b"),
                reference("equation.c"),
            ),
        )
        schema, specialized_name = "square_from_quadratic_side", "quadratic_root_then_square"
        units_path, rejected_path = "garden.plots", "garden.empty_spots"
        values = {
            "linear": linear,
            "quadratic_constant": quadratic_constant,
            "units": units,
            "rejected": rejected,
        }
    operations = (
        _operation("unit.capacity", nested, "count"),
        _operation(
            "collection.total_capacity",
            binary("*", reference("unit.capacity"), reference(units_path)),
            "count",
        ),
        _operation(
            "collection.after_quality",
            binary("-", reference("collection.total_capacity"), reference(rejected_path)),
            "count",
        ),
    )
    # The nested expression contributes two primitive operations, so the graph
    # has four causal operations and a primitive dependency depth of four.
    return SemanticGraph(
        "C4",
        schema,
        "nested",
        facts,
        operations,
        operations[-1].path,
        {
            **values,
            "specialized": specialized,
            "specialized_name": specialized_name,
            "temperature": int(distractors[0].value),
            "route_km": int(distractors[1].value),
        },
    )


_BUILDERS = {"C1": _build_c1, "C2": _build_c2, "C3": _build_c3, "C4": _build_c4}


def _evaluate_expression(expression: Expr, values: dict[str, Decimal]) -> Decimal:
    if expression.kind == "constant":
        return Decimal(expression.value)
    if expression.kind == "reference":
        return values[str(expression.value)]
    arguments = [_evaluate_expression(item, values) for item in expression.arguments]
    if expression.kind == "call":
        return ARITHMETIC_FUNCTIONS[str(expression.value)](arguments)
    left, right = arguments
    return {
        "+": lambda: left + right,
        "-": lambda: left - right,
        "*": lambda: left * right,
        "/": lambda: left / right,
    }[str(expression.value)]()


def _references(expression: Expr) -> list[str]:
    if expression.kind == "reference":
        return [str(expression.value)]
    return [item for argument in expression.arguments for item in _references(argument)]


def _operator_count(expression: Expr) -> int:
    own = int(expression.kind in ("binary", "call"))
    return own + sum(_operator_count(argument) for argument in expression.arguments)


def _function_nesting(expression: Expr) -> int:
    child = max((_function_nesting(argument) for argument in expression.arguments), default=0)
    return child + int(expression.kind == "call")


def _operator_family(expression: Expr) -> list[str]:
    families = []
    if expression.kind == "binary":
        families.append("O0")
    elif expression.kind == "call":
        families.append(
            {
                "square": "O1",
                "cube": "O1",
                "sqrt": "O1",
                "solve_linear": "O5",
                "positive_quadratic_root": "O6",
            }[str(expression.value)]
        )
    for argument in expression.arguments:
        families.extend(_operator_family(argument))
    return families


def _graph_metrics(graph: SemanticGraph) -> tuple[dict[str, Any], dict[str, Decimal]]:
    values = {fact.path: fact.value for fact in graph.facts}
    depths = {fact.path: 0 for fact in graph.facts}
    seen_paths = set(values)
    operator_count = 0
    reference_count = 0
    families: list[str] = []
    maximum_nesting = 0
    for operation in graph.operations:
        if operation.path in seen_paths:
            raise ValueError(f"duplicate graph binding: {operation.path}")
        references = _references(operation.expression)
        missing = sorted(set(references) - seen_paths)
        if missing:
            raise ValueError(f"forward or missing references for {operation.path}: {missing}")
        values[operation.path] = _evaluate_expression(operation.expression, values)
        child_depth = max((depths[path] for path in references), default=0)
        expression_ops = _operator_count(operation.expression)
        depths[operation.path] = child_depth + max(1, expression_ops)
        operator_count += expression_ops
        reference_count += len(references)
        maximum_nesting = max(maximum_nesting, _function_nesting(operation.expression))
        families.extend(_operator_family(operation.expression))
        seen_paths.add(operation.path)
    if graph.answer_path not in values:
        raise ValueError(f"unknown answer path: {graph.answer_path}")
    distractors = [fact.path for fact in graph.facts if fact.disconnected_distractor]
    causal_references = {
        path for operation in graph.operations for path in _references(operation.expression)
    }
    connected_distractors = sorted(set(distractors) & causal_references)
    if connected_distractors:
        raise ValueError(f"distractors entered causal graph: {connected_distractors}")
    metrics = {
        "operator_count": operator_count,
        "primitive_operation_count": operator_count,
        "dependency_depth": depths[graph.answer_path],
        "function_nesting_depth": maximum_nesting,
        "entity_count": len({item.entity for item in (*graph.facts, *graph.operations)}),
        "binding_count": len(graph.facts) + len(graph.operations),
        "reference_count": reference_count,
        "graph_structure": graph.graph_structure,
        "distractor_count": len(distractors),
        "disconnected_distractor_paths": distractors,
        "operator_families": sorted(set(families)),
        "operator_family_signature": families,
    }
    return metrics, values


def _assert_controls(metrics: dict[str, Any], controls: GenerationControls) -> None:
    bounded = {
        "operator_count": controls.operator_count,
        "dependency_depth": controls.dependency_depth,
        "function_nesting_depth": controls.function_nesting_depth,
    }
    for key, (minimum, maximum) in bounded.items():
        if not minimum <= metrics[key] <= maximum:
            raise AssertionError(f"{controls.tier} {key}={metrics[key]} outside [{minimum}, {maximum}]")
    if metrics["entity_count"] < controls.minimum_entity_count:
        raise AssertionError(f"{controls.tier} has too few entities")
    if metrics["binding_count"] < controls.minimum_binding_count:
        raise AssertionError(f"{controls.tier} has too few bindings")
    if metrics["graph_structure"] != controls.graph_structure:
        raise AssertionError(f"{controls.tier} graph structure mismatch")
    if metrics["distractor_count"] != controls.distractor_count:
        raise AssertionError(f"{controls.tier} distractor count mismatch")
    if not set(metrics["operator_families"]).issubset(controls.operator_families):
        raise AssertionError(f"{controls.tier} used an unsupported operator family")
    if controls.tier == "C4" and not ({"O1", "O5", "O6"} & set(metrics["operator_families"])):
        raise AssertionError("C4 requires a specialized operator")


def _lower_asl(graph: SemanticGraph) -> str:
    lines = [f"{fact.path} = {_decimal_text(fact.value)}" for fact in graph.facts]
    lines.extend(f"{operation.path} = {operation.expression.to_asl()}" for operation in graph.operations)
    lines.append(f"RETURN {graph.answer_path}")
    return "\n".join(lines)


def _render_c1(graph: SemanticGraph, family: str) -> str:
    values = graph.render_values
    setting = {
        "c1_shop_ledger": "A shop clerk's ledger says",
        "c1_depot_log": "A depot log records",
        "c1_school_inventory": "A school inventory report says",
        "c1_museum_inventory": "A museum supply record says",
    }[family]
    left, right, operator = values["left"], values["right"], values["operator"]
    clauses = {
        "+": f"there were {left} items in stock before a delivery of {right} more items",
        "-": f"there were {left} items in stock before an order removed {right} items",
        "*": f"a shipment contained {left} boxes with {right} items in every box",
        "/": f"a shipment of {left} items was divided equally among {right} teams",
    }
    questions = {
        "+": "How many items are now in stock?",
        "-": "How many items remain in stock?",
        "*": "How many items did the shipment contain?",
        "/": "How many items did each team receive?",
    }
    return f"{setting} that {clauses[operator]}. {questions[operator]}"


def _render_c2(graph: SemanticGraph, family: str) -> str:
    values = graph.render_values
    lead = {
        "c2_shipping_report": "A shipping report states",
        "c2_relief_supplies": "A relief-supply coordinator reports",
        "c2_library_packing": "A library packing record states",
        "c2_expedition_manifest": "An expedition manifest states",
    }[family]
    if values["schema"] == 0:
        return (
            f"{lead} that {values['boxes']} boxes arrived, with {values['per_box']} items "
            f"in each box. Inspectors found {values['damaged']} damaged items. "
            "How many usable items remain?"
        )
    return (
        f"{lead} that one class sent {values['first']} students and another sent "
        f"{values['second']} students. They formed {values['teams']} equal teams. "
        "How many students were in each team?"
    )


def _render_c3(graph: SemanticGraph, family: str) -> str:
    values = graph.render_values
    lead = {
        "c3_market_merge": "A market combines two deliveries",
        "c3_factory_merge": "A factory combines two production batches",
        "c3_festival_merge": "A festival combines supplies from two depots",
        "c3_research_supply_merge": "A research station combines two supply drops",
    }[family]
    text = (
        f"{lead}. The red batch has {values['red_boxes']} boxes with {values['red_each']} "
        f"items per box, while the blue batch has {values['blue_boxes']} boxes with "
        f"{values['blue_each']} items per box. A separate display uses {values['posters']} "
        "posters and is not part of either batch. "
    )
    if values["use_four"]:
        text += f"Quality control rejects {values['removed']} items from the combined batches. "
        return text + "How many accepted batch items remain?"
    return text + "How many items are in the two batches altogether?"


def _render_c4(graph: SemanticGraph, family: str) -> str:
    values = graph.render_values
    lead = {
        "c4_engineering_plan": "An engineering plan specifies",
        "c4_construction_plan": "A construction plan specifies",
        "c4_field_station_plan": "A field-station plan specifies",
        "c4_orbital_habitat_plan": "An orbital-habitat plan specifies",
    }[family]
    aside = (
        f" The forecast temperature is {values['temperature']} degrees, and the delivery "
        f"vehicle's route is {values['route_km']} kilometers; neither measurement is used "
        "in the requested calculation."
    )
    if values["specialized"] == "O1":
        core = (
            f"{lead} cube-shaped storage crates whose square face area is "
            f"{values['face_area']} square units. The edge is the positive square root of "
            "that face area, and each cubic unit holds one block. There are "
            f"{values['units']} crates, each filled to capacity, but {values['rejected']} "
            "of the loaded blocks are rejected."
        )
        question = " How many usable loaded blocks remain?"
    elif values["specialized"] == "O5":
        core = (
            f"{lead} square rooms. A room's side length x satisfies "
            f"{values['coefficient']} times x plus {values['offset']} equals "
            f"{values['total']}. The building has {values['units']} such rooms, and one "
            "tile was prepared for every square unit of floor. Of those prepared tiles, "
            f"{values['rejected']} are broken."
        )
        question = " How many usable prepared tiles remain?"
    else:
        linear = int(values["linear"])
        quadratic_constant = int(values["quadratic_constant"])
        linear_term = f"+ {linear}x" if linear >= 0 else f"- {abs(linear)}x"
        constant_term = (
            f"+ {quadratic_constant}"
            if quadratic_constant >= 0
            else f"- {abs(quadratic_constant)}"
        )
        core = (
            f"{lead} square garden plots. A plot's positive side length x is the only "
            f"positive root of x squared {linear_term} {constant_term} equals zero. "
            "Each square unit has one planting "
            f"spot. There are {values['units']} plots, and {values['rejected']} spots must "
            "remain empty."
        )
        question = " How many planting spots can be used?"
    return core + aside + question


_RENDERERS = {"C1": _render_c1, "C2": _render_c2, "C3": _render_c3, "C4": _render_c4}


def _asl_prompt(question: str) -> str:
    return (
        "Compile the quantitative problem into executable ASL-Arith. Available exact "
        "operators include +, -, *, /, square(x), cube(x), sqrt(x), "
        "solve_linear(a, b, c) for a*x+b=c, and "
        "positive_quadratic_root(a, b, c). Preserve all entities, source facts, "
        "dependencies, and the requested RETURN. Select operators from the language "
        "semantics; no operator has been preselected. Return only ASL, one statement "
        "per line; do not explain.\n\n"
        f"Problem: {question}\nASL:"
    )


def _serialize_graph(graph: SemanticGraph) -> dict[str, Any]:
    return {
        "graph_schema_id": graph.graph_schema_id,
        "graph_structure": graph.graph_structure,
        "facts": [
            {
                "path": fact.path,
                "value": _decimal_text(fact.value),
                "value_type": fact.value_type,
                "entity": fact.entity,
                "disconnected_distractor": fact.disconnected_distractor,
            }
            for fact in graph.facts
        ],
        "operations": [
            {
                "path": operation.path,
                "expression": operation.expression.to_dict(),
                "value_type": operation.value_type,
                "entity": operation.entity,
            }
            for operation in graph.operations
        ],
        "answer_path": graph.answer_path,
    }


def generate_compositional_record(
    *, tier: Tier, split: Split, index: int, seed: int, retry: int = 0
) -> dict[str, Any]:
    """Generate, lower, execute, and validate one deterministic example."""

    if tier not in TIER_CONTROLS:
        raise ValueError(f"unsupported tier: {tier}")
    if split not in ("train", "dev", "test"):
        raise ValueError(f"unsupported split: {split}")
    if index < 0:
        raise ValueError("index must be non-negative")
    if retry < 0:
        raise ValueError("retry must be non-negative")
    record_seed = _stable_seed(seed, tier, split, index, retry)
    rng = random.Random(record_seed)
    graph = _BUILDERS[tier](rng, index)
    metrics, values = _graph_metrics(graph)
    controls = TIER_CONTROLS[tier]
    _assert_controls(metrics, controls)
    families = _SURFACE_FAMILIES[tier][split]
    template_family_id = families[index % len(families)]
    question = _RENDERERS[tier](graph, template_family_id)
    gold_asl = _lower_asl(graph)
    example_id = f"paper1-compositional-{tier.lower()}-{split}-{index:05d}"
    scope = {
        "id": example_id,
        "parent": None,
        "kind": "benchmark_case",
        "source": GENERATOR_VERSION,
    }
    validation = validate_asl(gold_asl, effective_scope=scope)
    checks = {
        "syntax_verified": bool(validation["syntax_verified"]),
        "lower_verified": bool(validation["lower_verified"]),
        "type_verified": bool(validation["type_verified"]),
        "scope_verified": bool(validation["scope_verified"]),
        "execution_verified": bool(validation["execution_verified"]),
    }
    checks["valid"] = all(checks.values())
    if not checks["valid"]:
        raise AssertionError(f"generated ASL failed validation for {example_id}: {validation['errors']}")
    expected = values[graph.answer_path]
    returned = Decimal(str(validation["execution"]["workspace"][example_id]["returned"]))
    checks["answer_verified"] = returned == expected
    checks["valid"] = checks["valid"] and checks["answer_verified"]
    if not checks["answer_verified"]:
        raise AssertionError(f"generated answer mismatch for {example_id}: {returned} != {expected}")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset": "cogmath_comp_v1",
        "dataset_version": GENERATOR_VERSION,
        "example_id": example_id,
        "parent_example_id": example_id,
        "source_row": index,
        "split": split,
        "tier": tier,
        "difficulty_steps": metrics["operator_count"],
        "difficulty_stratum": tier.lower(),
        "generation_seed": seed,
        "generation_retry": retry,
        "record_seed": record_seed,
        "lineage": {
            "generator_version": GENERATOR_VERSION,
            "tier_id": tier,
            "graph_schema_id": graph.graph_schema_id,
            "template_family_id": template_family_id,
        },
        "question": question,
        "question_sha256": fingerprint(question),
        "question_hash": fingerprint(question),
        "prompt": _asl_prompt(question),
        "gold_asl": gold_asl,
        "target": gold_asl,
        "reference_return": _decimal_text(expected),
        "reference_answer": _decimal_text(expected),
        "effective_scope": scope,
        "source_fields_visible_to_model": ["question"],
        "complexity": {
            "requested": {
                "operator_count": list(controls.operator_count),
                "dependency_depth": list(controls.dependency_depth),
                "function_nesting_depth": list(controls.function_nesting_depth),
                "minimum_entity_count": controls.minimum_entity_count,
                "minimum_binding_count": controls.minimum_binding_count,
                "graph_structure": controls.graph_structure,
                "distractor_count": controls.distractor_count,
                "operator_families": list(controls.operator_families),
            },
            "observed": metrics,
        },
        "typed_symbolic_graph": _serialize_graph(graph),
        "validation_status": {
            **checks,
            "errors": list(validation["errors"]),
            "returned": _decimal_text(returned),
        },
    }


def _complexity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [row["complexity"]["observed"] for row in rows]
    numeric_keys = (
        "operator_count",
        "dependency_depth",
        "function_nesting_depth",
        "entity_count",
        "binding_count",
        "reference_count",
        "distractor_count",
    )
    summary: dict[str, Any] = {}
    for key in numeric_keys:
        values = [int(item[key]) for item in observed]
        summary[key] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "counts": dict(sorted(Counter(str(value) for value in values).items())),
        }
    summary["graph_structures"] = dict(sorted(Counter(item["graph_structure"] for item in observed).items()))
    summary["operator_families"] = dict(
        sorted(Counter(family for item in observed for family in item["operator_families"]).items())
    )
    return summary


def _normalize_tiers(tiers: list[str] | tuple[str, ...]) -> list[Tier]:
    requested = [item.upper() for item in tiers]
    if not requested or "ALL" in requested:
        return ["C1", "C2", "C3", "C4"]
    invalid = sorted(set(requested) - set(TIER_CONTROLS))
    if invalid:
        raise ValueError(f"unsupported tiers: {invalid}")
    return [tier for tier in ("C1", "C2", "C3", "C4") if tier in requested]


def freeze_compositional_pilots(
    output_dir: str | Path,
    *,
    tiers: list[str] | tuple[str, ...] = ("C1", "C2", "C3", "C4"),
    train_count: int = 250,
    dev_count: int = 30,
    test_count: int = 100,
    seed: int = 124001,
) -> dict[str, Any]:
    """Freeze exact-tier C1-C4 pilot parts and deterministic manifests."""

    counts = {"train": train_count, "dev": dev_count, "test": test_count}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all split counts must be positive")
    selected = _normalize_tiers(tiers)
    output = Path(output_dir)
    tier_manifests: dict[str, dict[str, Any]] = {}
    for tier in selected:
        tier_dir = output / tier
        split_rows: dict[str, list[dict[str, Any]]] = {}
        split_paths: dict[str, Path] = {}
        for split, count in counts.items():
            rows = []
            question_hashes: set[str] = set()
            for index in range(count):
                for retry in range(100):
                    row = generate_compositional_record(
                        tier=tier,
                        split=split,  # type: ignore[arg-type]
                        index=index,
                        seed=seed,
                        retry=retry,
                    )
                    if row["question_sha256"] not in question_hashes:
                        break
                else:
                    raise RuntimeError(
                        f"could not generate a unique {tier}/{split} row at index {index}"
                    )
                rows.append(row)
                question_hashes.add(row["question_sha256"])
            split_rows[split] = rows
            split_paths[split] = write_jsonl(tier_dir / f"{split}.jsonl", rows)
        train_templates = {row["lineage"]["template_family_id"] for row in split_rows["train"]}
        test_templates = {row["lineage"]["template_family_id"] for row in split_rows["test"]}
        if train_templates & test_templates:
            raise AssertionError(f"{tier} train/test template families overlap")
        validations = Counter(
            key
            for rows in split_rows.values()
            for row in rows
            for key, passed in row["validation_status"].items()
            if key.endswith("_verified") and passed
        )
        total_rows = sum(counts.values())
        manifest = {
            "schema_version": "ccpu.paper1.compositional_manifest.v1",
            "dataset_version": GENERATOR_VERSION,
            "tier": tier,
            "seed": seed,
            "counts": counts,
            "tier_controls": split_rows["train"][0]["complexity"]["requested"],
            "template_families": {
                split: list(_SURFACE_FAMILIES[tier][split]) for split in counts
            },
            "train_test_template_families_disjoint": True,
            "observed_complexity": {
                split: _complexity_summary(rows) for split, rows in split_rows.items()
            },
            "validation_counts": {
                key: {"passed": validations[key], "total": total_rows}
                for key in (
                    "syntax_verified",
                    "lower_verified",
                    "type_verified",
                    "scope_verified",
                    "execution_verified",
                    "answer_verified",
                )
            },
            "output_sha256": {
                split: file_sha256(path) for split, path in split_paths.items()
            },
            "model_or_teacher_used": False,
            "prompt_fields": ["question"],
            "hidden_fields": [
                "gold_asl",
                "target",
                "reference_return",
                "reference_answer",
                "typed_symbolic_graph",
            ],
        }
        write_json(tier_dir / "manifest.json", manifest)
        tier_manifests[tier] = manifest
    root_manifest = {
        "schema_version": "ccpu.paper1.compositional_collection_manifest.v1",
        "dataset_version": GENERATOR_VERSION,
        "seed": seed,
        "tiers": selected,
        "counts_per_tier": counts,
        "tier_manifest_sha256": {
            tier: file_sha256(output / tier / "manifest.json") for tier in selected
        },
        "tier_summaries": {
            tier: {
                "controls": tier_manifests[tier]["tier_controls"],
                "validation_counts": tier_manifests[tier]["validation_counts"],
            }
            for tier in selected
        },
        "model_or_teacher_used": False,
    }
    write_json(output / "manifest.json", root_manifest)
    return root_manifest


def build_compositional_curriculum(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    tiers: list[str] | tuple[str, ...] = ("C1", "C2", "C3", "C4"),
    seed: int = 124019,
) -> dict[str, Any]:
    """Combine frozen tier parts into deterministic, leakage-checked SFT splits."""

    source = Path(source_dir)
    output = Path(output_dir)
    selected = _normalize_tiers(tiers)
    split_paths: dict[str, Path] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for split_index, split in enumerate(("train", "dev")):
        rows: list[dict[str, Any]] = []
        source_hashes[split] = {}
        for tier in selected:
            path = source / tier / f"{split}.jsonl"
            tier_rows = read_jsonl(path)
            if not tier_rows:
                raise ValueError(f"empty compositional source: {path}")
            for row in tier_rows:
                if row.get("tier") != tier or row.get("split") != split:
                    raise ValueError(f"tier/split mismatch in {path}")
                if not row.get("validation_status", {}).get("valid"):
                    raise ValueError(f"unvalidated compositional row in {path}")
                if not row.get("prompt") or not row.get("target"):
                    raise ValueError(f"missing SFT prompt/target in {path}")
            rows.extend(tier_rows)
            source_hashes[split][tier] = file_sha256(path)
        random.Random(seed + split_index).shuffle(rows)
        split_rows[split] = rows
        split_paths[split] = write_jsonl(output / f"{split}.jsonl", rows)

    train_ids = {str(row["example_id"]) for row in split_rows["train"]}
    dev_ids = {str(row["example_id"]) for row in split_rows["dev"]}
    train_questions = {str(row["question_sha256"]) for row in split_rows["train"]}
    dev_questions = {str(row["question_sha256"]) for row in split_rows["dev"]}
    if train_ids & dev_ids or train_questions & dev_questions:
        raise AssertionError("compositional train/dev leakage detected")
    manifest = {
        "schema_version": "ccpu.paper1.compositional_curriculum_manifest.v1",
        "dataset_version": GENERATOR_VERSION,
        "source_dir": str(source),
        "tiers": selected,
        "seed": seed,
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "counts_by_tier": {
            split: dict(sorted(Counter(str(row["tier"]) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "source_sha256": source_hashes,
        "output_sha256": {
            split: file_sha256(path) for split, path in split_paths.items()
        },
        "train_dev_id_overlap": 0,
        "train_dev_question_overlap": 0,
        "test_rows_included": False,
        "model_or_teacher_used": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


__all__ = [
    "GENERATOR_VERSION",
    "TIER_CONTROLS",
    "GenerationControls",
    "build_compositional_curriculum",
    "freeze_compositional_pilots",
    "generate_compositional_record",
]
