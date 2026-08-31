from __future__ import annotations

import pytest

from ccpu.paper1.functor_data import functor_prompt
from ccpu.paper1.functor_runtime import (
    lower_functor_program,
    parse_functor_program,
    validate_functor_program,
)

SCOPE = {"id": "test:functor:1", "kind": "benchmark_case", "parent": None}


def test_f1_isomorphic_program_lowers_and_executes() -> None:
    program = """set("jessica.age_now", add(ref("claire.age_now"), const(6)))
set("claire.age_in_2y", const(20))
set("claire.age_now", sub(ref("claire.age_in_2y"), const(2)))
query("jessica.age_now")"""
    result = validate_functor_program(program, "f1", effective_scope=SCOPE)
    assert result["executable"]
    assert result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"] == 24


def test_f2_relations_preserve_forward_dependency() -> None:
    program = """offset("jessica.age_now", "claire.age_now", 6)
given("claire.age_in_2y", 20)
offset("claire.age_now", "claire.age_in_2y", -2)
query("jessica.age_now")"""
    result = validate_functor_program(program, "f2", effective_scope=SCOPE)
    assert result["executable"]
    assert result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"] == 24


def test_f2_percentage_and_rate_lower_through_runtime() -> None:
    program = """given("job.hourly_rate", 12)
given("job.hours", 5)
rate_total("job.gross", "job.hourly_rate", "job.hours")
percent_of("job.tax", "job.gross", 10)
remaining("job.net", "job.gross", "job.tax")
query("job.net")"""
    result = validate_functor_program(program, "f2", effective_scope=SCOPE)
    assert result["executable"]
    assert result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"] == 54


def test_f2_percentage_ratio_is_runtime_owned() -> None:
    program = """given("company.financing", 8868)
given("company.purchase", 97219)
percentage_ratio("company.financing_pct", "company.financing", "company.purchase")
query("company.financing_pct")"""
    result = validate_functor_program(program, "f2", effective_scope=SCOPE)
    assert result["executable"]
    returned = result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"]
    assert float(returned) == pytest.approx(9.1217, abs=0.0001)


def test_functor_parser_rejects_non_allowlisted_python() -> None:
    with pytest.raises(ValueError, match="unsupported functor"):
        parse_functor_program('__import__("os")\nquery("safe.path")', "f2")


def test_functor_program_requires_one_terminal_query() -> None:
    with pytest.raises(ValueError, match="end with exactly one"):
        parse_functor_program('given("x.value", 3)', "f2")


def test_student_prompt_is_fixed_and_contains_no_state_or_answer() -> None:
    row = {
        "question": "How many remain?",
        "source_context": {"table": [["items", "8"]], "paragraphs": []},
    }
    prompt = functor_prompt(row, "f2")
    assert "Mira has 12 cards" in prompt
    assert "How many remain?" in prompt
    assert "state_after" not in prompt
    assert "reference_asl" not in prompt
    assert "correct_answer" not in prompt


def test_lowered_f1_and_f2_are_distinct_surfaces() -> None:
    f1 = 'set("team.total", mul(ref("team.count"), ref("team.each")))\nquery("team.total")'
    f2 = 'per_unit_total("team.total", "team.count", "team.each")\nquery("team.total")'
    assert lower_functor_program(f1, "f1") == lower_functor_program(f2, "f2")
