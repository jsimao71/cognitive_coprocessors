import json

import pyarrow as pa
import pyarrow.parquet as pq

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_jsonl
from ccpu.dsl_dataset.semantic import prepare_local_annotation_batches
from ccpu.paper1.e3.cross_dataset import (
    build_cross_dataset_sft_data,
    build_cross_dataset_teacher_seed,
    freeze_cross_dataset_benchmark,
)
from ccpu.paper1.e3.gsm8k_confirmatory import _score_prediction


def _parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    return adapter


def test_asdiv_freeze_is_question_only_and_split_disjoint(tmp_path):
    source = _parquet(
        tmp_path / "asdiv.parquet",
        [
            {
                "body": f"There are {index} red balls and 2 blue balls.",
                "question": "How many balls are there?",
                "solution_type": "Addition",
                "answer": f"{index + 2} (balls)",
                "formula": f"{index}+2={index + 2}",
            }
            for index in range(1, 13)
        ],
    )
    gsm_train = write_jsonl(tmp_path / "gsm.jsonl", [{"question": "Unrelated."}])
    output = tmp_path / "frozen"
    manifest = freeze_cross_dataset_benchmark(
        dataset="asdiv",
        source_paths={"all": source},
        expected_sha256={"all": file_sha256(source)},
        gsm_train_paths=[gsm_train],
        gsm_adapter_path=_adapter(tmp_path),
        output_dir=output,
        diagnostic_size=4,
        dev_size=3,
        seed=7,
    )
    assert manifest["counts"]["train_source"] == 5
    diagnostic = read_jsonl(output / "diagnostic.jsonl")
    assert all(row["source_fields_visible_to_model"] == ["question"] for row in diagnostic)
    assert all("supervision" not in row for row in diagnostic)
    parent_sets = [
        {row["parent_example_id"] for row in read_jsonl(output / name)}
        for name in ("diagnostic.jsonl", "dev.jsonl", "train_source.jsonl")
    ]
    assert not parent_sets[0] & parent_sets[1]
    assert not parent_sets[0] & parent_sets[2]
    assert not parent_sets[1] & parent_sets[2]


def test_gsm_plus_groups_seed_families_across_splits(tmp_path):
    source = tmp_path / "gsm-plus.jsonl"
    rows = []
    for parent in range(8):
        for kind in ("numerical substitution", "distraction insertion"):
            rows.append(
                {
                    "question": f"Variant {kind} for source {parent}: what is {parent} plus 1?",
                    "solution": f"{parent}+1={parent + 1}",
                    "answer": str(parent + 1),
                    "perturbation_type": kind,
                    "seed_question": f"Source {parent}: what is {parent} plus 1?",
                    "seed_solution": f"{parent}+1={parent + 1}",
                    "seed_answer": str(parent + 1),
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    gsm_train = write_jsonl(tmp_path / "gsm.jsonl", [{"question": "Unrelated."}])
    output = tmp_path / "frozen"
    freeze_cross_dataset_benchmark(
        dataset="gsm_plus",
        source_paths={"test": source},
        expected_sha256={"test": file_sha256(source)},
        gsm_train_paths=[gsm_train],
        gsm_adapter_path=_adapter(tmp_path),
        output_dir=output,
        diagnostic_size=2,
        dev_size=2,
        seed=11,
    )
    parts = {
        name: {row["parent_example_id"] for row in read_jsonl(output / name)}
        for name in ("diagnostic.jsonl", "dev.jsonl", "train_source.jsonl")
    }
    assert len(parts["diagnostic.jsonl"]) == 2
    assert len({row["semantic_category"] for row in read_jsonl(output / "diagnostic.jsonl")}) == 2
    assert not parts["diagnostic.jsonl"] & parts["dev.jsonl"]
    assert not parts["diagnostic.jsonl"] & parts["train_source.jsonl"]
    assert not parts["dev.jsonl"] & parts["train_source.jsonl"]
    assert read_json(output / "manifest.json")["gsm_adapter_snapshot"]["files"]


def test_fractional_reference_answer_scores_numerically():
    score = _score_prediction(
        "result.value = 56 / 9\nRETURN result.value",
        "56/9",
        {"id": "global"},
    )
    assert score["final_answer_correct"]


def test_teacher_seed_keeps_answers_out_of_primary_requests(tmp_path):
    source = _parquet(
        tmp_path / "asdiv.parquet",
        [
            {
                "body": f"Sam has {index} red balls and 2 blue balls.",
                "question": "How many balls does Sam have?",
                "solution_type": "Addition",
                "answer": f"{index + 2} (balls)",
                "formula": f"{index}+2={index + 2}",
            }
            for index in range(1, 7)
        ],
    )
    gsm_train = write_jsonl(tmp_path / "gsm.jsonl", [{"question": "Unrelated."}])
    frozen = tmp_path / "frozen"
    freeze_cross_dataset_benchmark(
        dataset="asdiv",
        source_paths={"all": source},
        expected_sha256={"all": file_sha256(source)},
        gsm_train_paths=[gsm_train],
        gsm_adapter_path=_adapter(tmp_path),
        output_dir=frozen,
        diagnostic_size=2,
        dev_size=1,
        seed=13,
    )
    seed_path = tmp_path / "teacher_seed.jsonl"
    report = build_cross_dataset_teacher_seed(
        frozen_dir=frozen, output_path=seed_path
    )
    assert report["count"] == 3
    request_dir = tmp_path / "requests"
    prepare_local_annotation_batches([seed_path], request_dir, batch_size=3)
    payload = read_json(request_dir / "requests" / "batch_000.json")
    assert payload["answer_hidden"]
    assert payload["rationale_hidden"]
    assert all("answer" not in item for item in payload["items"])
    assert all("gold_reasoning" not in item for item in payload["items"])


def test_dev_teacher_seed_and_matched_sft_materialization(tmp_path):
    source = _parquet(
        tmp_path / "asdiv.parquet",
        [
            {
                "body": f"Sam has {index} red balls and 2 blue balls.",
                "question": "How many balls does Sam have?",
                "solution_type": "Addition",
                "answer": f"{index + 2} (balls)",
                "formula": f"{index}+2={index + 2}",
            }
            for index in range(1, 9)
        ],
    )
    gsm_train = write_jsonl(tmp_path / "gsm.jsonl", [{"question": "Unrelated."}])
    frozen = tmp_path / "frozen"
    freeze_cross_dataset_benchmark(
        dataset="asdiv",
        source_paths={"all": source},
        expected_sha256={"all": file_sha256(source)},
        gsm_train_paths=[gsm_train],
        gsm_adapter_path=_adapter(tmp_path),
        output_dir=frozen,
        diagnostic_size=2,
        dev_size=2,
        seed=17,
    )
    train_seed = tmp_path / "train_seed.jsonl"
    dev_seed = tmp_path / "dev_seed.jsonl"
    build_cross_dataset_teacher_seed(frozen_dir=frozen, output_path=train_seed)
    build_cross_dataset_teacher_seed(
        frozen_dir=frozen, output_path=dev_seed, source_role="dev", max_records=1
    )
    assert all(row["split"] == "dev" for row in read_jsonl(dev_seed))

    def accepted(seed_path, output_path):
        rows = []
        for seed in read_jsonl(seed_path):
            rows.append(
                {
                    "dataset": seed["dataset"],
                    "split": seed["split"],
                    "source_id": seed["source_id"],
                    "question": seed["question"],
                    "asl": f"answer.total = {seed['answer']}\nRETURN answer.total",
                    "quality_grade": "Q1_LOCAL_CODEX_EXEC_VERIFIED",
                    "validation": {
                        "execution_verified": True,
                        "final_answer_verified": True,
                    },
                }
            )
        return write_jsonl(output_path, rows)

    train_accepted = accepted(train_seed, tmp_path / "accepted_train.jsonl")
    dev_accepted = accepted(dev_seed, tmp_path / "accepted_dev.jsonl")
    report = build_cross_dataset_sft_data(
        frozen_dir=frozen,
        accepted_train_path=train_accepted,
        accepted_dev_path=dev_accepted,
        output_dir=tmp_path / "sft",
        seed=19,
    )
    assert report["counts"] == {"train": 4, "dev": 1}
    assert report["leakage_audit"]["passed"]
    train_rows = read_jsonl(tmp_path / "sft/train.jsonl")
    assert all(row["prompt"].endswith("ASL:") for row in train_rows)
