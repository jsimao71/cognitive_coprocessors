"""Finite, bounded Horn-clause forward chaining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ccpu.common.schema import CoprocessorRequest, CoprocessorResult

from .state import StateLimitError, TypedMicroState


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Atom:
        predicate = str(raw["predicate"])
        arguments = tuple(str(value) for value in raw["arguments"])
        if not predicate.isidentifier() or not arguments or len(arguments) > 4:
            raise ValueError("invalid bounded Horn atom")
        if any(not value or (not value.startswith("?") and not value.isidentifier()) for value in arguments):
            raise ValueError("invalid bounded Horn term")
        return cls(predicate, arguments)

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "arguments": list(self.arguments)}

    @property
    def ground(self) -> bool:
        return not any(argument.startswith("?") for argument in self.arguments)


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Atom, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Rule:
        body = tuple(Atom.from_dict(atom) for atom in raw["body"])
        if not body or len(body) > 8:
            raise ValueError("Horn rules require one to eight body atoms")
        return cls(head=Atom.from_dict(raw["head"]), body=body)


@dataclass(frozen=True)
class HornLimits:
    max_facts: int = 256
    max_rules: int = 64
    max_rounds: int = 16


def _unify(pattern: Atom, fact: Atom, substitution: dict[str, str]) -> dict[str, str] | None:
    if pattern.predicate != fact.predicate or len(pattern.arguments) != len(fact.arguments):
        return None
    result = dict(substitution)
    for expected, actual in zip(pattern.arguments, fact.arguments, strict=True):
        if expected.startswith("?"):
            if expected in result and result[expected] != actual:
                return None
            result[expected] = actual
        elif expected != actual:
            return None
    return result


def _matches(
    body: tuple[Atom, ...], facts: set[Atom]
) -> list[tuple[dict[str, str], tuple[Atom, ...]]]:
    substitutions: list[tuple[dict[str, str], tuple[Atom, ...]]] = [({}, ())]
    for atom in body:
        next_substitutions: list[tuple[dict[str, str], tuple[Atom, ...]]] = []
        for substitution, premises in substitutions:
            for fact in facts:
                matched = _unify(atom, fact, substitution)
                if matched is not None:
                    next_substitutions.append((matched, (*premises, fact)))
        substitutions = next_substitutions
    return substitutions


class HornEngine:
    name = "horn"

    def __init__(self, state: TypedMicroState, limits: HornLimits | None = None) -> None:
        self.state = state
        self.limits = limits or HornLimits()

    def execute(self, request: CoprocessorRequest) -> CoprocessorResult:
        if request.operation != "horn.query":
            return self._failure(request, "unsupported_operation")
        try:
            supplied_facts = {Atom.from_dict(row) for row in request.payload.get("facts", [])}
            rules = tuple(Rule.from_dict(row) for row in request.payload.get("rules", []))
            query = Atom.from_dict(dict(request.payload["query"]))
            if not query.ground or any(not fact.ground for fact in supplied_facts):
                raise ValueError("facts and query must be ground")
            if len(rules) > self.limits.max_rules:
                raise ValueError("Horn rule budget exceeded")
            facts = {Atom.from_dict(item.payload) for item in self.state.by_kind("horn_fact")}
            facts.update(supplied_facts)
            for fact in supplied_facts:
                self.state.add("horn_fact", fact.to_dict(), provenance={"request_id": request.request_id})
            fact_items = {
                Atom.from_dict(item.payload): item for item in self.state.by_kind("horn_fact")
            }
            rounds = 0
            derived = 0
            while rounds < self.limits.max_rounds:
                additions: dict[Atom, tuple[Atom, ...]] = {}
                for rule in rules:
                    for substitution, premises in _matches(rule.body, facts):
                        head = Atom(
                            rule.head.predicate,
                            tuple(substitution.get(term, term) for term in rule.head.arguments),
                        )
                        if head.ground and head not in facts:
                            additions.setdefault(head, premises)
                if not additions:
                    break
                if len(facts) + len(additions) > self.limits.max_facts:
                    raise StateLimitError("Horn fact budget exceeded")
                for atom in sorted(additions):
                    item = self.state.add(
                        "horn_fact",
                        atom.to_dict(),
                        dependencies=tuple(
                            fact_items[premise].state_id for premise in additions[atom]
                        ),
                        provenance={"request_id": request.request_id, "derived": True},
                    )
                    fact_items[atom] = item
                facts.update(additions)
                derived += len(additions)
                rounds += 1
            answer = query in facts
            return CoprocessorResult(
                request_id=request.request_id,
                engine=self.name,
                ok=True,
                value=answer,
                display=str(answer).lower(),
                metadata={"rounds": rounds, "derived_facts": derived, "fact_count": len(facts)},
            )
        except (KeyError, TypeError, ValueError, StateLimitError) as error:
            return self._failure(request, "invalid_or_bounded_ir", str(error))

    def _failure(self, request: CoprocessorRequest, code: str, message: str = "") -> CoprocessorResult:
        return CoprocessorResult(
            request_id=request.request_id,
            engine=self.name,
            ok=False,
            error_code=code,
            error_message=message,
        )
