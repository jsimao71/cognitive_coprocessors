import json

from ccpu.cli import main
from ccpu.common.artifacts import read_json, read_jsonl, write_json
from ccpu.paper1_5.benchmark_next import (
    NextBenchmarkConfig,
    build_next_candidates,
    select_measured_quadrants,
)
from ccpu.paper1_5.dataset import load_benchmark
from ccpu.paper1_5.evaluate import evaluate
from ccpu.paper1_5.natural_robustness import (
    NaturalRobustnessConfig,
    generate_natural_benchmark,
    lexical_audit,
    run_longform_opportunities,
    semantic_features,
    tokenizer_trigger_comparison,
)
from ccpu.paper1_5.policy_lora import (
    PolicyDataConfig,
    generate_policy_data,
    parse_retrieval_block,
)
from ccpu.paper1_5.source import ControlledFactStore, EvidenceStatus, FactRecord
from ccpu.paper1_5.triggers import decide, fit_confidence_threshold, semantic_risk


def _store() -> ControlledFactStore:
    return ControlledFactStore(
        source_id="test-source",
        version="v2",
        records=(
            FactRecord("old", "item", "value", "old", "2020-01-01", "2025-12-31"),
            FactRecord("new", "item", "value", "new", "2026-01-01"),
            FactRecord("c1", "conflict", "owner", "A", "2026-01-01"),
            FactRecord("c2", "conflict", "owner", "B", "2026-01-01"),
        ),
    )


def test_controlled_source_reports_supported_stale_conflict_and_unverified():
    store = _store()

    def status(entity: str, candidate: str) -> str:
        request = store.request(
            example_id=entity,
            entity=entity,
            attribute="owner" if entity == "conflict" else "value",
            as_of="2026-08-01",
            forecast=candidate,
            candidate_answer=candidate,
        )
        return store.execute(request).value["status"]

    assert status("item", "new") == EvidenceStatus.SUPPORTED
    assert status("item", "old") == EvidenceStatus.STALE
    assert status("conflict", "A") == EvidenceStatus.CONFLICT
    assert status("missing", "guess") == EvidenceStatus.UNVERIFIED


def test_confidence_and_semantic_triggers_are_independent():
    high_risk = decide("According to registry version 2026, give the value", "known", (0.9,), 0.5)
    low_safe = decide("Repeat the explicit code", "QX-7", (0.1,), 0.5)
    assert (high_risk.confidence, high_risk.semantic) == (False, True)
    assert (low_safe.confidence, low_safe.semantic) == (True, False)
    assert semantic_risk("What is the SI unit of electric current?")[0] is False


def test_threshold_is_fit_from_measured_probabilities():
    threshold = fit_confidence_threshold([((0.9,), False), ((0.2,), True), ((0.1,), True)])
    assert 0.2 < threshold <= 0.9


def test_evaluation_reports_quadrants_and_unsupported_claims():
    rows = []
    for index, (low, risk) in enumerate(((False, False), (True, False), (False, True), (True, True))):
        rows.append(
            {
                "example_id": str(index),
                "split": "test",
                "condition": "llm_only",
                "correct": not risk,
                "evidence_required": risk,
                "confidence_low": low,
                "baseline_correct": not risk,
                "retrieved": False,
                "quadrant": f"{'low' if low else 'high'}_confidence__{'high' if risk else 'low'}_risk",
                "minimum_token_probability": 0.2 if low else 0.8,
                "generated_tokens": 1,
                "model_calls": 1,
                "wall_time_ns": 1,
                "confidence_threshold": 0.5,
                "gold_answer": "ok" if not risk else "source",
                "unsupported_commitment": risk,
                "authorized_commitment": False,
                "runtime_enforced": False,
            }
        )
    summary = evaluate(rows)
    condition = summary["by_condition"][0]
    assert summary["all_four_quadrants_observed"] is True
    assert condition["unsupported_claim_rate"] == 1.0
    assert condition["unsupported_commitment_rate"] == 1.0


def test_next_benchmark_covers_subclasses_and_freezes_measured_quadrants():
    config = NextBenchmarkConfig(dev_per_design=2, test_per_design=4, target_per_quadrant=2)
    source, examples, audit = build_next_candidates(config)
    assert audit["unique_record_ids"] is True
    assert len(audit["retrieval_required_subclasses"]) == 8
    assert len(audit["retrieval_not_required_subclasses"]) == 7
    probabilities = {}
    for example in examples:
        low_design = "likely_low" in str(example.design_group)
        probabilities[example.example_id] = 0.1 if low_design else 0.9
    selected, freeze = select_measured_quadrants(
        examples, probabilities, threshold=0.5, target_per_quadrant=2
    )
    assert len(selected) == 16
    assert set(freeze["selected_counts"].values()) == {2}
    assert source["source_id"] == "atlas-controlled-registry"


def test_cli_validates_controlled_benchmark(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "source": {"source_id": "one", "version": "v1", "records": []},
                "examples": [
                    {
                        "split": split,
                        "question": split,
                        "answer": split,
                        "entity": split,
                        "attribute": "value",
                        "as_of": "2026-01-01",
                        "evidence_required": False,
                        "category": "control",
                    }
                    for split in ("dev", "test")
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["paper1.5", "validate", "--config", str(config)]) == 0
    _, examples = load_benchmark(read_json(config))
    assert len(examples) == 2


def test_policy_data_excludes_answers_and_parses_typed_request(tmp_path):
    _, examples, _ = build_next_candidates(
        NextBenchmarkConfig(dev_per_design=1, test_per_design=1, target_per_quadrant=1)
    )
    result = generate_policy_data(
        PolicyDataConfig(
            train_required=2, train_controls=2, dev_required=1, dev_controls=1
        ),
        excluded_examples=examples,
        output_dir=tmp_path,
    )
    assert result["leakage_audit"]["targets_contain_answer_values"] is False
    request = parse_retrieval_block(
        "```retrieve\nentity=Demo\nattribute=code\nas_of=2026-08-28\nsource=atlas\n```"
    )
    assert request == {
        "entity": "Demo",
        "attribute": "code",
        "as_of": "2026-08-28",
        "source": "atlas",
    }
    assert parse_retrieval_block("NO_RETRIEVAL") is None


def test_natural_benchmark_audit_features_and_longform(tmp_path):
    result = generate_natural_benchmark(
        NaturalRobustnessConfig(
            train_per_category=2, dev_per_category=1, test_per_category=1
        ),
        tmp_path,
    )
    assert result["example_count"] == 64
    audit = read_json(tmp_path / "freeze_audit.json")
    assert audit["label_balance"]["test"] == {"not_required": 8, "required": 8}
    assert not any(audit["exact_signature_overlap"].values())
    assert not any(audit["normalized_template_overlap"].values())
    assert not any(audit["within_split_duplicate_questions"].values())
    assert audit["source_key_collisions"] == 0
    assert audit["answer_consistency_errors"] == 0
    source = read_json(tmp_path / "source.json")
    keys = [(row["entity"], row["attribute"]) for row in source["records"]]
    assert len(keys) == len(set(keys))
    controls = [
        row
        for row in read_jsonl(tmp_path / "benchmark.jsonl")
        if not row["evidence_required"]
    ]
    expected_by_category = {
        "quoted_freshness": ("latest custodian", "current registry", "updated owner"),
        "historical_date": ("18th", "yes", "1848"),
        "compute_not_retrieve": ("42", "120", "2026-08-29"),
        "stable_familiar": ("Au", "Lisbon", "H2O"),
    }
    for row in controls:
        if row["category"] in expected_by_category:
            assert row["answer"] == expected_by_category[row["category"]][
                row["template_index"]
            ]

    risky = semantic_features("Complete field 7B for CASE-PA200.")
    supplied = semantic_features(
        "The active brief states that CASE-PA200's custodian is CTX-200."
    )
    assert risky["combined"]
    assert supplied["active_context"] and not supplied["combined"]

    lexical = lexical_audit(tmp_path / "benchmark.jsonl")
    assert len(lexical["results"]) == 3
    write_json(tmp_path / "tokenizers.json", {"models": []})
    tokenizer_result = tokenizer_trigger_comparison(
        tmp_path / "benchmark.jsonl",
        tmp_path / "tokenizers.json",
        tmp_path / "tokenizers",
    )
    assert tokenizer_result["paper1_5_decision"]["selection_is_development_only"]
    assert any(
        row["condition"] == "transparent_semantic_runtime"
        for row in tokenizer_result["results"]
    )
    longform = run_longform_opportunities(tmp_path / "benchmark.jsonl")
    assert longform["opportunity_count"] == 12
    assert longform["runtime_ucr"] <= longform["advisory_ucr"]
