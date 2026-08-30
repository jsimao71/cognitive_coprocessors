from ccpu.dsl import execute_program, lower_program, parse_asl, validate_asl


def test_asl_arithmetic_parses_lowers_and_executes_statefully():
    source = """
month1.downloads = 60
month2.downloads = month1.downloads * 3
month3.downloads = dec_pct(month2.downloads, 30)
total = month1.downloads + month2.downloads + month3.downloads
RETURN total
"""
    scope = {
        "id": "gsm8k:test:123",
        "parent": None,
        "kind": "benchmark_case",
        "source": "dataset",
    }
    program = parse_asl(source, effective_scope=scope)
    lowered = lower_program(program)
    execution = execute_program(program)
    values = execution["workspace"]["gsm8k:test:123"]["values"]
    assert values == {
        "month1.downloads": 60,
        "month2.downloads": 180,
        "month3.downloads": 126,
        "total": 366,
    }
    assert execution["workspace"]["gsm8k:test:123"]["returned"] == 366
    assert lowered["operations"][-1]["operation"]["op"] == "RETURN"


def test_asl_explicit_scopes_require_qualified_sibling_references():
    source = """
SCOPE exercise1
SCOPE a
x = 12
RETURN x
END
SCOPE b
y = a.x * 3
RETURN y
END
END
"""
    validation = validate_asl(source)
    assert validation["execution_verified"] is True
    scopes = validation["execution"]["workspace"]
    assert scopes["root/exercise1/a"]["returned"] == 12
    assert scopes["root/exercise1/b"]["returned"] == 36


def test_asl_core_parses_future_rule_and_retrieval_syntax_without_lowering_it():
    rule = parse_asl("has_wings(?x) :- bird(?x)")
    statement = rule["records"][0]["statement"]
    assert statement["operator"] == ":-"
    assert statement["left"]["type"] == "call"
    retrieval = parse_asl('evidence <- lookup(query="penguin")')
    assert retrieval["records"][0]["statement"]["operator"] == "<-"


def test_asl_validation_rejects_dangling_references():
    result = validate_asl("total = missing.value + 2\nRETURN total")
    assert result["syntax_verified"] is True
    assert result["execution_verified"] is False
    assert "unresolved reference" in result["errors"][0]
