import json

from ccpu.cli import main
from ccpu.common.artifacts import read_json
from ccpu.paper1_5.dataset import load_benchmark
from ccpu.paper1_5.evaluate import evaluate
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
            }
        )
    summary = evaluate(rows)
    condition = summary["by_condition"][0]
    assert summary["all_four_quadrants_observed"] is True
    assert condition["unsupported_claim_rate"] == 1.0


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
