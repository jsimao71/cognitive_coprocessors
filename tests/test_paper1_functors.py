from __future__ import annotations

import json

import pytest

from ccpu.paper1.functor_data import functor_prompt
from ccpu.paper1.functor_metrics import (
    _f2_call_metrics,
    _state_metrics,
    compare_functor_model_sizes,
)
from ccpu.paper1.functor_runtime import (
    functor_registry,
    lower_functor_program,
    parse_functor_program,
    validate_functor_program,
)

SCOPE = {"id": "test:functor:1", "kind": "benchmark_case", "parent": None}


def test_f1_isomorphic_program_lowers_and_executes() -> None:
    program = """add("jessica.age_now", "claire.age_now", 6)
value("claire.age_in_2y", 20)
subtract("claire.age_now", "claire.age_in_2y", 2)
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


def test_f2_blackboard_solves_single_unknown_constraints_bidirectionally() -> None:
    program = """given("trader.initial", 55)
given("trader.restocked", 132)
remaining("trader.after_sale", "trader.initial", "trader.sold")
sum_of("trader.current", "trader.after_sale", "trader.restocked")
given("trader.current", 164)
query("trader.sold")"""
    result = validate_functor_program(program, "f2", effective_scope=SCOPE)
    assert result["executable"]
    returned = result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"]
    assert returned == 23


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


def test_f1_nary_subtraction_is_left_associative() -> None:
    program = """value("wallet.total", 20)
subtract("wallet.remaining", "wallet.total", 3, 4, 5)
query("wallet.remaining")"""
    result = validate_functor_program(program, "f1", effective_scope=SCOPE)
    assert result["executable"]
    returned = result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"]
    assert returned == 8


def test_functor_parser_rejects_non_allowlisted_python() -> None:
    with pytest.raises(ValueError, match="unsupported functor"):
        parse_functor_program('__import__("os")\nquery("safe.path")', "f2")


def test_runtime_symbol_table_canonicalizes_numeric_path_segments() -> None:
    program = """given("revenue.2019.taiwan", 118)
query("revenue.2019.taiwan")"""
    result = validate_functor_program(program, "f2", effective_scope=SCOPE)
    assert result["executable"]
    assert "revenue.y2019.taiwan" in result["lowered_asl"]


def test_exact_rational_literals_are_parsed_without_eval() -> None:
    program = """given("team.total", 30)
fraction_of("team.share", "team.total", 1/6, 1)
query("team.share")"""
    result = validate_functor_program(program, "f2", effective_scope=SCOPE)
    assert result["executable"]
    returned = result["validation"]["execution"]["workspace"][SCOPE["id"]]["returned"]
    assert float(returned) == pytest.approx(5.0)


def test_functor_program_requires_one_terminal_query() -> None:
    with pytest.raises(ValueError, match="end with exactly one"):
        parse_functor_program('given("x.value", 3)', "f2")


def test_student_prompt_is_fixed_and_contains_no_state_or_answer() -> None:
    row = {
        "question": "How many remain?",
        "source_context": {"table": [["items", "8"]], "paragraphs": []},
    }
    prompt = functor_prompt(row, "f2")
    assert "Fixed example" not in prompt
    assert "How many remain?" in prompt
    assert "state_after" not in prompt
    assert "reference_asl" not in prompt
    assert "correct_answer" not in prompt


def test_lowered_f1_and_f2_are_distinct_surfaces() -> None:
    f1 = 'multiply("team.total", "team.count", "team.each")\nquery("team.total")'
    f2 = 'per_unit_total("team.total", "team.count", "team.each")\nquery("team.total")'
    assert lower_functor_program(f1, "f1") == lower_functor_program(f2, "f2")


def test_functor_registry_is_public_and_defensive() -> None:
    registry = functor_registry("f2")
    registry.pop("given")
    assert "given" in functor_registry("f2")


def test_f2_metrics_separate_class_from_argument_binding() -> None:
    gold = '''given("wallet.total", 20)
given("wallet.used", 7)
difference("wallet.remaining", "wallet.total", "wallet.used")
query("wallet.remaining")'''
    swapped = '''given("wallet.total", 20)
given("wallet.used", 7)
difference("wallet.remaining", "wallet.used", "wallet.total")
query("wallet.remaining")'''
    metrics = _f2_call_metrics(gold, swapped, SCOPE)
    assert metrics["class_metrics"]["f1"] == 1.0
    assert metrics["binding_correct"] < metrics["binding_support"]
    assert metrics["direction_correct"] < metrics["direction_support"]


def test_blackboard_metrics_penalize_spurious_state() -> None:
    reference = "wallet.total = 20\nRETURN wallet.total"
    predicted = "wallet.total = 20\nwallet.extra = 7\nRETURN wallet.total"
    metrics = _state_metrics(reference, predicted, SCOPE)
    assert metrics["blackboard_recall"] == 1.0
    assert metrics["blackboard_precision"] < 1.0
    assert metrics["spurious_state_rate"] > 0.0


def test_model_size_comparison_uses_difference_in_differences(tmp_path) -> None:
    def summary(label: str, answers: tuple[float, float, float]) -> dict:
        conditions = {}
        for name, answer in zip(("F0", "F1", "F2"), answers):
            conditions[name] = {
                "rates": {
                    "parse": {"rate": 1.0},
                    "semantic_structure": {"rate": answer},
                    "final_answer": {"rate": answer},
                },
                "mean_semantic": {"dependency": answer},
                "mean_blackboard": {"blackboard_f1": answer},
            }
        conditions["F2"]["f2_semantics"] = {
            "functor_class_micro": {"f1": answers[2]},
            "argument_role_accuracy": {"rate": answers[2]},
            "argument_binding_exact": {"rate": answers[2]},
            "relation_direction_accuracy": {"rate": answers[2]},
        }
        return {"model_label": label, "frozen_identity_count": 25, "conditions": conditions}

    small, large = tmp_path / "small.json", tmp_path / "large.json"
    small.write_text(json.dumps(summary("small", (0.4, 0.2, 0.3))), encoding="utf-8")
    large.write_text(json.dumps(summary("large", (0.5, 0.3, 0.6))), encoding="utf-8")
    result = compare_functor_model_sizes(small, large, tmp_path / "comparison.json")
    assert result["representation_by_capacity_interaction"]["capacity_bottleneck_supported"]
