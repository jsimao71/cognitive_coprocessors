import json

import pyarrow as pa
import pyarrow.parquet as pq

from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_jsonl
from ccpu.paper1.e3.cross_dataset import freeze_cross_dataset_benchmark
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
