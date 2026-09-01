"""Deterministic F3 evidence checks and lowering into the existing ASL runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ccpu.dsl import validate_asl
from ccpu.paper1.functor_runtime import Call, lower_f2, solve_f2_to_asl

from .ast import Form, Program, Value
from .normalize import normalize_text, program_record
from .parser import parse_f3_program
from .registry import RUNTIME_MODES

_PATH = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*$")
_RELATIONS = {
    "same",
    "offset",
    "older_than",
    "younger_than",
    "multiple",
    "fraction_of",
    "difference_relation",
    "absolute_difference",
    "quotient_relation",
    "percent_of",
    "percent_more",
    "percent_less",
    "sum_relation",
    "product_relation",
    "rate_relation",
    "mean_relation",
    "minimum_relation",
    "maximum_relation",
}
_EVENTS = {"remove", "add", "consume", "produce", "transfer"}


def _decimal(value: Value) -> int | Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise TypeError(f"expected numeric literal, got {value!r}")
    return value


def _string(value: Value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"expected non-empty {label}, got {value!r}")
    return value


def _path(value: Value) -> str:
    raw = _string(value, "path").casefold()
    path = ".".join(f"y{part}" if part and part[0].isdigit() else part for part in raw.split("."))
    if not _PATH.fullmatch(path):
        raise ValueError(f"invalid F3 path: {value!r}")
    return path


def _time_suffix(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "current"
    return f"y{token}" if token[0].isdigit() else token


def _evidence_forms(value: Value) -> list[Form]:
    found: list[Form] = []
    if isinstance(value, Form):
        if value.name in {"source", "cell"}:
            found.append(value)
        for argument in value.args:
            found.extend(_evidence_forms(argument))
    return found


def _source_document(question: str, source_context: dict[str, Any] | None) -> tuple[str, set[str]]:
    text = [question]
    cells: set[str] = set()
    if source_context:
        for paragraph in source_context.get("paragraphs", []):
            text.append(str(paragraph.get("text", "")))
        for row in source_context.get("table", []):
            for cell in row:
                normalized = normalize_text(str(cell).replace("$", "").replace(",", ""))
                if normalized:
                    cells.add(normalized)
                    text.append(str(cell))
    return normalize_text(" ".join(text)), cells


def validate_evidence(
    program: Program, *, question: str, source_context: dict[str, Any] | None
) -> list[str]:
    """Require every source/cell anchor to match supplied evidence."""

    document, cells = _source_document(question, source_context)
    errors: list[str] = []
    for form in program.forms[:-1]:
        anchors = _evidence_forms(form)
        if not anchors:
            errors.append(f"{form.name} is missing evidence")
            continue
        for anchor in anchors:
            if anchor.name == "source":
                span = normalize_text(_string(anchor.args[0], "source span"))
                if not span or span not in document:
                    errors.append(f"source span is not present: {anchor.args[0]!r}")
            else:
                row = normalize_text(_string(anchor.args[0], "table row"))
                column = normalize_text(_string(anchor.args[1], "table column"))
                if row not in cells:
                    errors.append(f"table row is not present: {anchor.args[0]!r}")
                if column not in cells:
                    errors.append(f"table column is not present: {anchor.args[1]!r}")
    return errors


@dataclass
class Compiler:
    mode: str
    calls: list[Call] = field(default_factory=list)
    current: dict[str, str] = field(default_factory=dict)
    observed: dict[tuple[str, str], str] = field(default_factory=dict)
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    collections: set[str] = field(default_factory=set)
    members: dict[str, set[str]] = field(default_factory=dict)
    query_path: str | None = None
    model_edges: list[tuple[str, str, str]] = field(default_factory=list)
    runtime_edges: list[tuple[str, str, str]] = field(default_factory=list)
    bound: set[str] = field(default_factory=set)
    expression_index: int = 0

    @property
    def level(self) -> int:
        return RUNTIME_MODES[self.mode]

    def reference(self, value: Value) -> str:
        if isinstance(value, str):
            path = _path(value)
            return self.current.get(path, path)
        if isinstance(value, Form) and value.name in {"scale", "fraction"}:
            self.expression_index += 1
            target = f"expressions.e{self.expression_index}.value"
            base = self.quantity_operand(value.args[0])
            if value.name == "scale":
                self.calls.append(Call("multiple", (target, base, _decimal(value.args[1]))))
            else:
                self.calls.append(
                    Call(
                        "fraction_of",
                        (target, base, _decimal(value.args[1]), _decimal(value.args[2])),
                    )
                )
            self.bound.add(target)
            return target
        if isinstance(value, Form) and value.name == "event_field":
            source_id = _string(value.args[0], "event ID").casefold()
            field_name = _string(value.args[1], "event field").casefold()
            if field_name != "quantity" or source_id not in self.events:
                raise ValueError(f"unknown event field: {source_id}.{field_name}")
            return str(self.events[source_id]["quantity"])
        if not isinstance(value, Form) or value.name != "at":
            raise TypeError(f"expected at(path, time) reference, got {value!r}")
        path = _path(value.args[0])
        time = _string(value.args[1], "time").casefold()
        if time in {"current", "now"} and path in self.current:
            return self.current[path]
        key = (path, time)
        if key in self.observed:
            return self.observed[key]
        return f"{path}.{_time_suffix(time)}"

    def operand(self, value: Value) -> int | Decimal | str:
        if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
            return value
        return self.reference(value)

    def quantity(self, value: Value, event_id: str) -> int | Decimal | str:
        if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
            target = f"events.{event_id}.quantity"
            self.calls.append(Call("given", (target, value)))
            return target
        if isinstance(value, str) or (isinstance(value, Form) and value.name == "at"):
            return self.reference(value)
        if not isinstance(value, Form):
            raise TypeError(f"invalid event quantity: {value!r}")
        if value.name == "event_field":
            source_id = _string(value.args[0], "event ID").casefold()
            field_name = _string(value.args[1], "event field").casefold()
            if field_name != "quantity" or source_id not in self.events:
                raise ValueError(f"unknown event field: {source_id}.{field_name}")
            return str(self.events[source_id]["quantity"])
        target = f"events.{event_id}.quantity"
        if value.name == "scale":
            base = self.quantity_operand(value.args[0])
            self.calls.append(Call("multiple", (target, base, _decimal(value.args[1]))))
        elif value.name == "fraction":
            base = self.quantity_operand(value.args[0])
            self.calls.append(
                Call(
                    "fraction_of",
                    (target, base, _decimal(value.args[1]), _decimal(value.args[2])),
                )
            )
        else:
            raise ValueError(f"unsupported quantity expression: {value.name}")
        return target

    def quantity_operand(self, value: Value) -> int | Decimal | str:
        if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
            return value
        if isinstance(value, Form) and value.name == "event_field":
            source_id = _string(value.args[0], "event ID").casefold()
            if source_id not in self.events or value.args[1] != "quantity":
                raise ValueError(f"unknown event quantity: {source_id}")
            return str(self.events[source_id]["quantity"])
        if isinstance(value, Form) and value.name == "at":
            return self.reference(value)
        if isinstance(value, str):
            return self.reference(value)
        raise TypeError(f"invalid quantity operand: {value!r}")

    def add_observation(self, form: Form) -> None:
        reference, value, _unit, _evidence = form.args
        if not isinstance(reference, Form) or reference.name != "at":
            raise TypeError("observe requires at(path, time)")
        path = _path(reference.args[0])
        time = _string(reference.args[1], "time").casefold()
        target = f"{path}.{_time_suffix(time)}"
        self.calls.append(Call("given", (target, _decimal(value))))
        self.bound.add(target)
        self.observed[(path, time)] = target
        if time in {"initial", "current", "now"}:
            previous = self.current.get(path)
            if previous is not None and previous != target and time in {"current", "now"}:
                self.calls.append(Call("same", (target, previous)))
                self.runtime_edges.append((previous, target, "state_observation"))
            self.current[path] = target
        match = re.fullmatch(r"(plus|minus)_(\d+)_year", time)
        if match and path.endswith(".age"):
            now = f"{path}.now"
            delta = int(match.group(2)) * (1 if match.group(1) == "plus" else -1)
            self.calls.append(Call("offset", (target, now, delta)))
            self.runtime_edges.append((now, target, "age_progression"))

    def add_relation(self, form: Form) -> None:
        if self.level < 1:
            raise ValueError(f"{form.name} requires F3-R1 or F3-R2")
        args = form.args[:-1]
        name = form.name
        if name == "same":
            call = Call("same", (self.reference(args[0]), self.reference(args[1])))
        elif name in {"offset", "older_than", "younger_than"}:
            delta_index = 2
            delta = _decimal(args[delta_index])
            if name == "younger_than":
                delta = -delta
            call = Call("offset", (self.reference(args[0]), self.reference(args[1]), delta))
        elif name == "multiple":
            call = Call(
                "multiple",
                (self.reference(args[0]), self.reference(args[1]), _decimal(args[2])),
            )
        elif name == "fraction_of":
            call = Call(
                "fraction_of",
                (
                    self.reference(args[0]),
                    self.reference(args[1]),
                    _decimal(args[2]),
                    _decimal(args[3]),
                ),
            )
        elif name in {"difference_relation", "absolute_difference", "quotient_relation"}:
            lowered = {
                "difference_relation": "difference",
                "absolute_difference": "absolute_difference",
                "quotient_relation": "quotient",
            }[name]
            call = Call(
                lowered,
                (self.reference(args[0]), self.operand(args[1]), self.operand(args[2])),
            )
        elif name in {"percent_of", "percent_more", "percent_less"}:
            lowered = {
                "percent_of": "percent_of",
                "percent_more": "increase_percent",
                "percent_less": "decrease_percent",
            }[name]
            call = Call(
                lowered,
                (self.reference(args[0]), self.reference(args[1]), _decimal(args[2])),
            )
        elif name in {"sum_relation", "product_relation"}:
            lowered = "sum_of" if name == "sum_relation" else "product_of"
            call = Call(lowered, tuple(self.operand(item) for item in args))
        elif name == "rate_relation":
            call = Call("rate_total", tuple(self.operand(item) for item in args))
        elif name in {"mean_relation", "minimum_relation", "maximum_relation"}:
            lowered = {
                "mean_relation": "mean_of",
                "minimum_relation": "minimum_of",
                "maximum_relation": "maximum_of",
            }[name]
            call = Call(lowered, tuple(self.operand(item) for item in args))
        else:
            raise ValueError(f"unsupported relation: {name}")
        self.calls.append(call)
        target = str(call.args[0])
        self.bound.add(target)
        target_form = args[0]
        if isinstance(target_form, Form) and target_form.name == "at":
            path = _path(target_form.args[0])
            time = _string(target_form.args[1], "time").casefold()
            self.observed[(path, time)] = target
            if time in {"initial", "current", "now"}:
                self.current[path] = target
        elif isinstance(target_form, str):
            self.current[_path(target_form)] = target
        for operand in call.args[1:]:
            if isinstance(operand, str):
                self.model_edges.append((operand, target, name))

    def add_event(self, form: Form) -> None:
        if self.level < 1:
            raise ValueError(f"{form.name} requires F3-R1 or F3-R2")
        event_id = _string(form.args[0], "event ID").casefold()
        if event_id in self.events:
            raise ValueError(f"duplicate event ID: {event_id}")
        if form.name == "transfer":
            _actor, source_ref, destination_ref, amount, _evidence = form.args[1:]
            source_path = _path(source_ref.args[0]) if isinstance(source_ref, Form) else _path(source_ref)
            destination_path = (
                _path(destination_ref.args[0])
                if isinstance(destination_ref, Form)
                else _path(destination_ref)
            )
            source_before = self.reference(source_ref)
            destination_before = self.reference(destination_ref)
            quantity = self.quantity(amount, event_id)
            source_after = f"{source_path}.after_{event_id}"
            destination_after = f"{destination_path}.after_{event_id}"
            if source_before in self.bound:
                self.calls.append(Call("remaining", (source_after, source_before, quantity)))
                self.bound.add(source_after)
                self.current[source_path] = source_after
                self.runtime_edges.append((source_before, source_after, "transfer_source"))
            if destination_before in self.bound:
                self.calls.append(Call("sum_of", (destination_after, destination_before, quantity)))
                self.bound.add(destination_after)
                self.current[destination_path] = destination_after
                self.runtime_edges.append(
                    (destination_before, destination_after, "transfer_destination")
                )
        else:
            _actor, state_ref, amount, _evidence = form.args[1:]
            path = _path(state_ref.args[0]) if isinstance(state_ref, Form) else _path(state_ref)
            before = self.reference(state_ref)
            quantity = self.quantity(amount, event_id)
            after = f"{path}.after_{event_id}"
            relation = "remaining" if form.name in {"remove", "consume"} else "sum_of"
            self.calls.append(Call(relation, (after, before, quantity)))
            self.bound.add(after)
            self.current[path] = after
            self.runtime_edges.append((before, after, form.name))
        self.events[event_id] = {"type": form.name, "quantity": quantity}

    def add_declaration(self, form: Form) -> None:
        if form.name == "collection":
            self.collections.add(_path(form.args[0]))
        elif form.name == "member":
            collection = _path(form.args[0])
            self.members.setdefault(collection, set()).add(_path(form.args[1]))
        elif form.name == "partition":
            collection = _path(form.args[0])
            self.collections.add(collection)
            total = self.reference(form.args[1])
            member_paths = sorted(self.members.get(collection, set()))
            if member_paths and self.level >= 2:
                member_references: set[str] = set()
                for member in member_paths:
                    member_references.update(
                        reference
                        for path, reference in self.current.items()
                        if path == member or path.startswith(member + ".")
                    )
                    member_references.update(
                        reference
                        for (path, _time), reference in self.observed.items()
                        if path == member or path.startswith(member + ".")
                    )
                if member_references:
                    self.calls.append(Call("sum_of", (total, *sorted(member_references))))
                    self.bound.add(total)
                    target_form = form.args[1]
                    if isinstance(target_form, Form) and target_form.name == "at":
                        path = _path(target_form.args[0])
                        time = _string(target_form.args[1], "time").casefold()
                        self.observed[(path, time)] = total
                        if time in {"initial", "current", "now"}:
                            self.current[path] = total

    def add_query(self, form: Form) -> None:
        kind = _string(form.args[0], "query kind").casefold()
        target = "queries.result"
        if kind == "value" and len(form.args) == 2:
            source = self.reference(form.args[1])
            self.calls.append(Call("same", (target, source)))
        elif kind == "mean" and len(form.args) >= 3:
            self.calls.append(Call("mean_of", (target, *(self.operand(item) for item in form.args[1:]))))
        elif kind in {"remaining_count", "sum"} and len(form.args) == 3:
            collection = _path(form.args[1])
            query_time = _string(form.args[2], "query time").casefold()
            declared = self.members.get(collection, set())

            def belongs(path: str) -> bool:
                if declared:
                    return any(path == member or path.startswith(member + ".") for member in declared)
                return path.startswith(collection + ".")

            references = {
                reference
                for (path, time), reference in self.observed.items()
                if belongs(path)
                and (
                    bool(declared)
                    or time == query_time
                    or query_time in time
                    or time in query_time
                )
                and (kind == "sum" or path.endswith(".count"))
            }
            if query_time in {"current", "now"}:
                references.update(
                    reference
                    for path, reference in self.current.items()
                    if belongs(path) and (kind == "sum" or path.endswith(".count"))
                )
            if not references:
                raise ValueError(f"query collection has no count state: {collection}")
            self.calls.append(Call("sum_of", (target, *sorted(references))))
        elif kind in {
            "difference",
            "absolute_difference",
            "percentage_change",
            "percentage_ratio",
        } and len(form.args) == 3:
            before = self.reference(form.args[1])
            after = self.reference(form.args[2])
            if kind == "percentage_ratio":
                self.calls.append(Call("percentage_ratio", (target, before, after)))
            else:
                difference = "queries.change"
                relation = "absolute_difference" if kind == "absolute_difference" else "difference"
                self.calls.append(Call(relation, (difference, after, before)))
            if kind in {"difference", "absolute_difference"}:
                self.calls.append(Call("same", (target, difference)))
            elif kind == "percentage_change":
                self.calls.append(Call("percentage_ratio", (target, difference, before)))
        else:
            raise ValueError(f"invalid or unsupported query signature: {kind}")
        self.query_path = target

    def query_connected_calls(self) -> list[Call]:
        """Keep the undirected constraint component that can affect the query."""

        if self.query_path is None:
            return []
        connected = {self.query_path}
        selected: set[int] = set()
        changed = True
        while changed:
            changed = False
            for index, call in enumerate(self.calls):
                variables = {item for item in call.args if isinstance(item, str)}
                if not variables.intersection(connected):
                    continue
                if index not in selected or not variables.issubset(connected):
                    selected.add(index)
                    connected.update(variables)
                    changed = True
        return [call for index, call in enumerate(self.calls) if index in selected]

    def compile(self, program: Program) -> str:
        for form in program.forms:
            if form.name == "observe":
                self.add_observation(form)
            elif form.name in _RELATIONS:
                self.add_relation(form)
            elif form.name in _EVENTS:
                self.add_event(form)
            elif form.name in {"collection", "member", "partition"}:
                self.add_declaration(form)
            elif form.name == "query":
                self.add_query(form)
            else:
                raise ValueError(f"unsupported top-level form: {form.name}")
        if self.query_path is None:
            raise ValueError("F3 program has no query target")
        calls = [*self.query_connected_calls(), Call("query", (self.query_path,))]
        return solve_f2_to_asl(calls) if self.level >= 2 else lower_f2(calls)


def validate_f3_program(
    text: str,
    *,
    question: str,
    source_context: dict[str, Any] | None,
    effective_scope: dict[str, Any],
    mode: str = "r2",
) -> dict[str, Any]:
    """Parse, ground, lower, type-check, and execute one F3 program."""

    if mode not in RUNTIME_MODES:
        raise ValueError(f"unsupported F3 runtime mode: {mode}")
    result: dict[str, Any] = {
        "parse_valid": False,
        "evidence_valid": False,
        "lowerable": False,
        "type_valid": False,
        "executable": False,
        "lowered_asl": "",
        "errors": [],
        "mode": mode,
    }
    try:
        program = parse_f3_program(text)
        result["parse_valid"] = True
        result["ast"] = program_record(program)
        evidence_errors = validate_evidence(
            program, question=question, source_context=source_context
        )
        result["errors"].extend(evidence_errors)
        result["evidence_valid"] = not evidence_errors
        if evidence_errors:
            return result
        compiler = Compiler(mode)
        lowered = compiler.compile(program)
        result["lowered_asl"] = lowered
        result["lowerable"] = True
        result["model_edges"] = compiler.model_edges
        result["runtime_edges"] = compiler.runtime_edges
        validation = validate_asl(lowered, effective_scope=effective_scope)
        result["validation"] = validation
        result["type_valid"] = bool(validation["type_verified"])
        result["executable"] = bool(validation["execution_verified"])
        result["errors"].extend(validation["errors"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        result["errors"].append(str(error))
    return result
