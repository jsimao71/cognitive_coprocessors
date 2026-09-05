from ccpu.common.artifacts import file_sha256, read_json, read_jsonl, write_jsonl
from ccpu.paper1.e3.gsm8k_confirmatory import freeze_official_gsm8k
from ccpu.paper1.e3.large_number_suite import freeze_large_number_gsm8k


def _inputs(tmp_path):
    source = write_jsonl(
        tmp_path / "official.jsonl",
        [
            {
                "question": (
                    "A warehouse has 12 crates with 3 bolts in each crate and ships "
                    "6 bolts. How many bolts remain?"
                ),
                "answer": (
                    "There are 12 * 3 = <<12*3=36>>36 bolts. "
                    "After shipping, 36 - 6 = <<36-6=30>>30.\n#### 30"
                ),
            },
            {
                "question": "A price of $20 is reduced by 10%. What is the new price?",
                "answer": "The reduction is <<20*10/100=2>>2. <<20-2=18>>18.\n#### 18",
            },
            {
                "question": "A 20-year-old has 12 coins and receives 3 more. How many now?",
                "answer": "The total is <<12+3=15>>15.\n#### 15",
            },
            {
                "question": "A train runs from 1:00 to 5:00 at 20 miles per hour. How far?",
                "answer": "The trip is <<5-1=4>>4 hours. The distance is <<4*20=80>>80.\n#### 80",
            },
            {
                "question": "Scores were 50, 80, and 60. What is their sum?",
                "answer": "The sum is <<50+80+60=190>>190.\n#### 190",
            },
        ],
    )
    train = write_jsonl(tmp_path / "train.jsonl", [{"question": "Different."}])
    official = tmp_path / "frozen"
    freeze_official_gsm8k(
        source_path=source,
        train_paths=[train],
        output_dir=official,
        expected_sha256=file_sha256(source),
        expected_rows=5,
        confirmatory_size=5,
    )
    return source, official / "full.jsonl"


def test_large_number_freeze_replays_trace_and_excludes_unsafe_rows(tmp_path):
    source, official = _inputs(tmp_path)
    output = tmp_path / "large"
    manifest = freeze_large_number_gsm8k(
        source_path=source,
        official_eval_path=official,
        output_dir=output,
        expected_source_sha256=file_sha256(source),
        factor=1000,
    )
    assert manifest["counts"]["eligible"] == 2
    assert manifest["counts"]["excluded"] == 3
    assert manifest["counts"]["exclusion_reasons"] == {
        "calendar_or_age": 1,
        "clock_or_ratio_notation": 1,
        "percentage": 1,
    }
    assert manifest["counts"]["transformed_source_values"] == {
        "minimum": 3,
        "maximum": 3,
        "total": 6,
    }
    rows = read_jsonl(output / "large.jsonl")
    row = rows[0]
    assert row["parent_example_id"] == "gsm8k-official-test-0000"
    assert "12000 crates" in row["question"]
    assert "3000 bolts" in row["question"]
    assert "6000 bolts" in row["question"]
    assert row["reference_return"] == "35994000"
    assert row["source_fields_visible_to_model"] == ["question"]
    assert not row["transformation"]["trace_visible_to_model"]
    assert not row["transformation"]["answer_visible_to_model"]
    trace = row["transformation"]["hidden_execution_trace"]
    assert trace[0]["transformed_expression"] == "12000*3000"
    assert trace[0]["transformed_result"] == "36000000"
    assert trace[1]["transformed_expression"] == "36000000-6000"
    scores = rows[1]
    assert scores["question"] == "Scores were 50000, 80000, and 60000. What is their sum?"
    assert scores["reference_return"] == "190000"
    assert read_json(output / "manifest.json") == manifest


def test_large_number_freeze_is_deterministic(tmp_path):
    source, official = _inputs(tmp_path)
    first = freeze_large_number_gsm8k(
        source_path=source,
        official_eval_path=official,
        output_dir=tmp_path / "first",
        expected_source_sha256=file_sha256(source),
    )
    second = freeze_large_number_gsm8k(
        source_path=source,
        official_eval_path=official,
        output_dir=tmp_path / "second",
        expected_source_sha256=file_sha256(source),
    )
    assert first["output_sha256"] == second["output_sha256"]


def test_large_number_freeze_rejects_wrong_source_hash(tmp_path):
    source, official = _inputs(tmp_path)
    try:
        freeze_large_number_gsm8k(
            source_path=source,
            official_eval_path=official,
            output_dir=tmp_path / "large",
            expected_source_sha256="0" * 64,
        )
    except ValueError as error:
        assert "source hash differs" in str(error)
    else:
        raise AssertionError("wrong source hash was accepted")
