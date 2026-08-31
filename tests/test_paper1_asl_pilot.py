from types import SimpleNamespace

from ccpu.cli import build_parser
from ccpu.common.artifacts import read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl
from ccpu.paper1.asl_incremental_eval import run_incremental_program
from ccpu.paper1.asl_pilot_data import (
    _choose_grouped_splits,
    build_asl_expansion_data,
    build_asl_incremental_data,
    incremental_prompt,
    pattern_id,
    perturb_row,
)
from ccpu.paper1.asl_pilot_eval import extract_asl, score_asl


def _scope(source_id="one"):
    return {
        "id": f"gsm8k:train:{source_id}",
        "parent": None,
        "kind": "benchmark_case",
        "source": "dataset",
    }


def _row(source_id: str, asl: str, question: str):
    validation = validate_asl(asl, effective_scope=_scope(source_id))
    assert validation["execution_verified"]
    return {
        "dataset": "gsm8k",
        "source_id": source_id,
        "record_sha256": source_id.rjust(64, "0"),
        "question": question,
        "asl": asl,
        "ccir": validation["ccir"],
        "part_mappings": [{"asl": asl.splitlines()}],
        "provenance": {"repair_round": 0},
        "effective_scope": _scope(source_id),
    }


def test_pattern_signature_ignores_names_and_literal_values():
    first = _row(
        "1",
        "claire.age_now = 18\njessica.age_now = claire.age_now + 6\nRETURN jessica.age_now",
        "Claire is 18. Jessica is six years older. How old is Jessica?",
    )
    second = _row(
        "2",
        "helen.age_now = 900\nmaria.age_now = helen.age_now + 77\nRETURN maria.age_now",
        "Helen is 900. Maria is 77 years older. How old is Maria?",
    )
    assert pattern_id(first) == pattern_id(second)


def test_grouped_split_keeps_exact_counts_without_group_overlap():
    groups = {
        f"p{index}": [
            {
                "dataset": "tatqa" if index % 3 == 0 else "gsm8k",
                "provenance": {"repair_round": 1 if index % 4 == 0 else 0},
            }
        ]
        for index in range(150)
    }
    splits = _choose_grouped_splits(groups, seed=7)
    assert [len(splits[name]) for name in ("train", "dev", "test")] == [100, 25, 25]
    assert not splits["train"] & splits["dev"]
    assert not splits["train"] & splits["test"]
    assert not splits["dev"] & splits["test"]


def test_numeric_perturbation_updates_nl_and_asl_and_remains_executable():
    row = _row(
        "3",
        "claire.age_now = 18\njessica.age_now = claire.age_now + 6\nRETURN jessica.age_now",
        "Claire is 18 years old. Jessica is 6 years older. How old is Jessica?",
    )
    transformed = perturb_row(row, variant=11, mode="large")
    assert transformed["perturbation"]["numeric_replacements"]
    assert transformed["question"] != row["question"]
    assert transformed["asl"] != row["asl"]
    assert validate_asl(transformed["asl"], effective_scope=row["effective_scope"])[
        "execution_verified"
    ]


def test_large_perturbation_preserves_percentage_parameters():
    row = _row(
        "pct",
        "item.price = 250\nitem.discount_pct = 20\n"
        "item.sale = dec_pct(item.price, item.discount_pct)\nRETURN item.sale",
        "A store discounts a 250 euro item by 20 percent.",
    )
    transformed = perturb_row(row, variant=1, mode="large")
    assert "discount_pct = 20" in transformed["asl"]
    assert "20 percent" in transformed["question"]
    assert transformed["perturbation"]["numeric_replacements"]["250"] != "250"


def test_semantic_scoring_accepts_equivalent_path_names():
    reference = """claire.age_in_2y = 20
claire.age_now = claire.age_in_2y - 2
jessica.age_now = claire.age_now + 6
RETURN jessica.age_now"""
    predicted = """claire.future_age = 20
claire.age_now = claire.future_age - 2
jessica.age_now = claire.age_now + 6
RETURN jessica.age_now"""
    metrics = score_asl(reference, predicted, _scope())
    assert metrics["exact_asl"] is False
    assert metrics["parse_valid"] is True
    assert metrics["executable"] is True
    assert metrics["semantic_return_equivalent"] is True
    assert metrics["semantic_state_equivalent"] is True
    assert metrics["final_answer_correct"] is True


def test_semantic_scoring_handles_typed_but_unlowerable_program():
    reference = "family.children = 3\nRETURN family.children"
    predicted = (
        "family.has_parent = true\n"
        "family.parent_legs = family.has_parent * 2\n"
        "RETURN family.parent_legs"
    )
    metrics = score_asl(reference, predicted, _scope())
    assert metrics["parse_valid"] is True
    assert metrics["lowerable_to_ccir"] is True
    assert metrics["type_valid"] is False
    assert metrics["executable"] is False
    assert metrics["final_answer_correct"] is False


def test_extract_asl_ignores_fenced_explanation():
    output = "Here is the program:\n```asl\na.count = 4\nRETURN a.count\n```\nDone."
    assert extract_asl(output) == "a.count = 4\nRETURN a.count"


def test_expansion_data_preserves_frozen_pattern_boundary(tmp_path):
    def frozen_row(source_id, target):
        row = _row(
            source_id,
            f"{target} = 1\nRETURN {target}",
            f"The grounded value of {target} is 1.",
        )
        return {
            **row,
            "split": "train",
            "semantic_pattern_id": pattern_id(row),
        }

    freeze = tmp_path / "freeze"
    original_train = frozen_row("train", "box.count")
    dev = frozen_row("dev", "person.age")
    test = frozen_row("test", "shop.price")
    expansion = frozen_row("expansion", "trip.distance")
    write_jsonl(freeze / "splits" / "train.jsonl", [original_train])
    write_jsonl(freeze / "splits" / "dev.jsonl", [dev])
    write_jsonl(freeze / "splits" / "test.jsonl", [test])
    write_json(freeze / "freeze_manifest.json", {"frozen": True})
    expansion_path = write_jsonl(tmp_path / "expansion.jsonl", [expansion])
    manifest = build_asl_expansion_data(freeze, expansion_path, tmp_path / "output")
    assert manifest["train_rows"] == 2
    assert manifest["train_pattern_count"] == 2
    assert len(read_jsonl(tmp_path / "output" / "sft" / "train_450.jsonl")) == 2


def test_incremental_data_exposes_only_prior_executed_state(tmp_path):
    def incremental_row(source_id, target):
        row = _row(
            source_id,
            f"{target} = 2\nRETURN {target}",
            f"There are 2 {target}. How many are there?",
        )
        row.update(
            {
                "split": "train",
                "semantic_pattern_id": pattern_id(row),
                "parts": [
                    {"part_id": 0, "text": f"There are 2 {target}."},
                    {"part_id": 1, "text": "How many are there?"},
                ],
                "part_mappings": [
                    {"part_id": 0, "asl": [f"{target} = 2"]},
                    {"part_id": 1, "asl": [f"RETURN {target}"]},
                ],
            }
        )
        return row

    freeze = tmp_path / "freeze"
    train = incremental_row("train-inc", "box.count")
    dev = incremental_row("dev-inc", "person.age")
    expansion = incremental_row("exp-inc", "trip.distance")
    write_jsonl(freeze / "splits" / "train.jsonl", [train])
    write_jsonl(freeze / "splits" / "dev.jsonl", [dev])
    write_json(freeze / "freeze_manifest.json", {"frozen": True})
    expansion_path = write_jsonl(tmp_path / "expansion.jsonl", [expansion])
    manifest = build_asl_incremental_data(freeze, expansion_path, tmp_path / "incremental")
    records = read_jsonl(tmp_path / "incremental" / "sft" / "train_incremental.jsonl")
    assert manifest["train_transitions"] == 4
    assert '"values":{}' in records[0]["prompt"]
    assert "box.count" in records[1]["prompt"] or "trip.distance" in records[1]["prompt"]


def test_incremental_evaluation_feeds_predicted_state_forward():
    row = _row(
        "closed-loop",
        "box.count = 2\nRETURN box.count",
        "There are 2 boxes. How many boxes are there?",
    )
    row.update(
        {
            "parts": [
                {"part_id": 0, "text": "There are 2 boxes."},
                {"part_id": 1, "text": "How many boxes are there?"},
            ],
            "part_mappings": [
                {"part_id": 0, "asl": ["box.count = 2"]},
                {"part_id": 1, "asl": ["RETURN box.count"]},
            ],
        }
    )

    class Backend:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, *, seed):
            del seed
            self.prompts.append(prompt)
            text = "box.count = 2" if len(self.prompts) == 1 else "RETURN box.count"
            return SimpleNamespace(
                generated_text=text,
                prompt_tokens=10,
                generated_tokens=3,
                wall_time_ns=1,
            )

    backend = Backend()
    predicted, traces, stopped = run_incremental_program(row, backend, seed=1)
    assert predicted == row["asl"]
    assert stopped is None
    assert all(trace["accepted"] for trace in traces)
    assert '"box.count":2' in backend.prompts[1]


def test_incremental_full_question_context_preserves_local_target():
    row = _row(
        "full-context",
        "box.count = 2\nRETURN box.count",
        "There are 2 boxes. How many boxes are there?",
    )
    part = {"part_id": 0, "text": "There are 2 boxes."}
    causal = incremental_prompt(row, part, {"values": {}, "unresolved": []})
    full = incremental_prompt(
        row,
        part,
        {"values": {}, "unresolved": []},
        context_mode="full_question",
    )
    assert "Full original question:" not in causal
    assert row["question"] in full


def test_incremental_oracle_mode_feeds_gold_state_after_student_error():
    row = _row(
        "oracle-state",
        "box.count = 2\nRETURN box.count",
        "There are 2 boxes. How many boxes are there?",
    )
    row.update(
        {
            "parts": [
                {"part_id": 0, "text": "There are 2 boxes."},
                {"part_id": 1, "text": "How many boxes are there?"},
            ],
            "part_mappings": [
                {"part_id": 0, "asl": ["box.count = 2"]},
                {"part_id": 1, "asl": ["RETURN box.count"]},
            ],
        }
    )

    class Backend:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, *, seed):
            del seed
            self.prompts.append(prompt)
            text = "box.count = 99" if len(self.prompts) == 1 else "RETURN box.count"
            return SimpleNamespace(
                generated_text=text,
                prompt_tokens=10,
                generated_tokens=3,
                wall_time_ns=1,
            )

    backend = Backend()
    run_incremental_program(row, backend, seed=1, state_mode="oracle")
    assert '"box.count":2' in backend.prompts[1]
    assert '"box.count":99' not in backend.prompts[1]


def test_incremental_cli_forwards_oracle_state_mode(tmp_path, monkeypatch):
    config = write_json(tmp_path / "config.json", {"model": {}})
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"prediction_count": 0, "rates": {"final_answer_correct": 0.0}}

    monkeypatch.setattr("ccpu.paper1.cli.run_asl_incremental", fake_run)
    args = build_parser().parse_args(
        [
            "paper1",
            "run-asl-incremental",
            "--programs",
            "programs.jsonl",
            "--config",
            str(config),
            "--adapter-path",
            "adapter",
            "--output-dir",
            str(tmp_path / "output"),
            "--state-mode",
            "oracle",
        ]
    )
    args.handler(args)
    assert captured["state_mode"] == "oracle"
