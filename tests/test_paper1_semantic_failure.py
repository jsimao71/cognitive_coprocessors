from ccpu.paper1.semantic_failure import analyze_program

SCOPE = {"id": "test", "parent": None, "kind": "benchmark_case", "source": "test"}


def analyze(reference: str, predicted: str, **kwargs):
    return analyze_program(reference, predicted, SCOPE, **kwargs)


def error_types(result):
    return {item["type"] for item in result["errors"]}


def test_wrong_entity_right_attribute_is_separated():
    result = analyze("claire.age_now = 18", "jessica.age_now = 18")
    assert result["entity_metrics"]["f1"] == 0
    assert result["attribute_metrics"]["f1"] == 1
    assert "entity_grounding" in error_types(result)


def test_right_entity_wrong_attribute_is_separated():
    result = analyze("claire.age_now = 18", "claire.height_now = 18")
    assert result["entity_metrics"]["f1"] == 1
    assert result["attribute_metrics"]["f1"] == 0
    assert "attribute_grounding" in error_types(result)


def test_qualifier_only_mismatch_is_visible():
    result = analyze("claire.age_now = 18", "claire.age_future = 18")
    assert result["attribute_metrics"]["f1"] == 1
    assert result["qualifier_metrics"]["f1"] == 0
    assert "temporal_qualifier" in error_types(result)


def test_consistent_standalone_rename_can_align():
    result = analyze(
        "claire.age_now = 18\nRETURN claire.age_now",
        "claire.current_age = 18\nRETURN claire.current_age",
    )
    assert result["semantic_name_equivalent"] is True
    assert result["return_target_accuracy"]["accuracy"] == 1


def test_stateful_rename_is_forbidden_after_symbol_exists():
    result = analyze(
        "jessica.age_now = claire.age_now + 6",
        "jessica.age_now = claire.current_age + 6",
        workspace_paths={"claire.age_now"},
        condition="incremental_predicted",
    )
    assert result["workspace_symbol_stable"] is False
    assert result["workspace_path_reuse_rate"]["accuracy"] == 0
    assert "forbidden_stateful_rename" in error_types(result)


def test_relation_reversal_is_not_operator_error():
    result = analyze(
        "jessica.age_now = claire.age_now + 6",
        "claire.age_now = jessica.age_now + 6",
    )
    assert result["operator_metrics"]["f1"] == 1
    assert "relation_direction" in error_types(result)


def test_noncommutative_argument_reversal_is_detected():
    result = analyze(
        "stock.remaining = stock.total - stock.used",
        "stock.remaining = stock.used - stock.total",
    )
    assert result["operator_metrics"]["f1"] == 1
    assert result["argument_order_accuracy"]["accuracy"] == 0
    assert "argument_order" in error_types(result)


def test_wrong_relation_constant_is_detected():
    result = analyze(
        "jessica.age_now = claire.age_now + 6",
        "jessica.age_now = claire.age_now + 7",
    )
    assert result["relation_constant_accuracy"]["accuracy"] == 0
    assert "relation_constant" in error_types(result)


def test_premature_literal_collapse_preserves_computational_distinction():
    result = analyze(
        "claire.age_now = 18\njessica.age_now = claire.age_now + 6",
        "claire.age_now = 18\njessica.age_now = 24",
    )
    assert result["dependency_collapse"]["count"] == 1
    assert result["dependency_collapse"]["correct_value_lost_derivation_count"] == 1
    assert "correct_value_lost_derivation" in error_types(result)


def test_transitive_shortcut_is_separate_from_value_equivalence():
    result = analyze(
        "claire.age_future = 20\nclaire.age_now = claire.age_future - 2\n"
        "jessica.age_now = claire.age_now + 6",
        "claire.age_future = 20\njessica.age_now = claire.age_future + 4",
    )
    assert result["transitive_shortcut"]["count"] == 1
    assert "transitive_shortcut" in error_types(result)


def test_each_vs_total_is_an_aggregation_cardinality_failure():
    result = analyze(
        "boxes.count = 4\nitems.each = 3\nitems.total = boxes.count * items.each",
        "boxes.count = 4\nitems.each = 3\nitems.total = boxes.count / items.each",
    )
    assert result["aggregation_cardinality_accuracy"]["accuracy"] == 0
    assert "aggregation_cardinality" in error_types(result)


def test_temporal_relation_reversal_is_visible():
    result = analyze(
        "claire.age_now = claire.age_future - 2",
        "claire.age_future = claire.age_now - 2",
    )
    assert "relation_direction" in error_types(result)
    assert "temporal_semantics" in error_types(result)


def test_wrong_return_target_is_detected():
    result = analyze(
        "claire.age_now = 18\njessica.age_now = 24\nRETURN jessica.age_now",
        "claire.age_now = 18\njessica.age_now = 24\nRETURN claire.age_now",
    )
    assert result["return_target_accuracy"]["accuracy"] == 0
    assert "return_wrong_target" in error_types(result)


def test_one_prediction_can_receive_multiple_failure_labels():
    result = analyze(
        "jessica.age_now = claire.age_now + 6\nRETURN jessica.age_now",
        "maria.height_future = claire.age_now * 7\nRETURN claire.age_now",
    )
    errors = error_types(result)
    assert {"entity_grounding", "attribute_grounding", "qualifier_grounding"} <= errors
    assert {"operator_mapping", "return_wrong_target"} <= errors
    assert len(errors) >= 5
