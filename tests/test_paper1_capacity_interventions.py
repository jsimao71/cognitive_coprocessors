from ccpu.common.artifacts import write_json, write_jsonl
from ccpu.paper1.asl_incremental_analysis import analyze_adapter_capacity_interventions


def _summary(path, *, answer, parse, state, adapter_rank):
    write_json(
        path,
        {
            "rates": {
                "final_answer_correct": answer,
                "parse_valid": parse,
                "executable": 0.6,
                "semantic_state_equivalent": state,
                "dependency_correct": 0.2,
            },
            "incremental": {
                "transition_rates": {
                    "operator_exact": 0.25,
                    "path_exact": 0.2,
                    "state_delta_exact": 0.3,
                },
                "completed_program_count": 2,
            },
            "prediction_count": 2,
            "scored_predictions_sha256": f"scored-{adapter_rank}",
            "run": {
                "programs_sha256": "programs",
                "seed": 44017,
                "model": {"model_id": "model", "revision": "revision"},
                "adapter_rank": adapter_rank,
            },
        },
    )
    write_jsonl(
        path.with_name("scored_predictions.jsonl"),
        [
            {"parent_source_id": "a", "metrics": {"final_answer_correct": True}},
            {"parent_source_id": "b", "metrics": {"final_answer_correct": False}},
        ],
    )


def _training(path, *, rank, train_loss, dev_loss):
    write_json(
        path,
        {
            "adapter_id": f"r{rank}",
            "history": [{"mean_train_loss": train_loss, "mean_dev_loss": dev_loss}],
            "training": {"rank": rank, "target_modules": ["q_proj"], "epochs": 10},
            "trainable_parameters": rank * 100,
            "train_rows": 10,
            "dev_rows": 2,
            "train_sha256": "train",
            "dev_sha256": "dev",
        },
    )


def test_capacity_intervention_analysis_is_matched_and_deterministic(tmp_path):
    paths = {}
    for rank, answer, parse, state in (("r8", 0.25, 0.75, 0.2), ("r16", 0.25, 1.0, 0.1)):
        for mode in ("predicted", "oracle", "full"):
            directory = tmp_path / rank / mode
            directory.mkdir(parents=True)
            path = directory / "summary.json"
            mode_answer = answer + (0.25 if mode == "full" else 0.0)
            _summary(path, answer=mode_answer, parse=parse, state=state, adapter_rank=rank)
            paths[(rank, mode)] = path

    r8_training = tmp_path / "r8_training.json"
    r16_training = tmp_path / "r16_training.json"
    _training(r8_training, rank=8, train_loss=0.03, dev_loss=1.0)
    _training(r16_training, rank=16, train_loss=0.02, dev_loss=1.1)
    pilot = tmp_path / "pilot.json"
    write_json(
        pilot,
        {
            "conditions": [
                {"condition": "lora_100", "rates": {"final_answer_correct": 0.2}},
                {"condition": "lora_100_icl_3", "rates": {"final_answer_correct": 0.4}},
            ]
        },
    )
    semantic = tmp_path / "semantic.json"
    write_json(
        semantic,
        {
            "conditions": {
                "whole_lora500": {
                    "semantic": {"attribute": {"f1": 0.1}, "dependency": {"f1": 0.2}}
                }
            },
            "teacher_consistency": {"canonical_attribute_shapes": 100},
        },
    )
    output = tmp_path / "analysis.json"

    report = analyze_adapter_capacity_interventions(
        baseline_predicted_summary=paths[("r8", "predicted")],
        baseline_oracle_summary=paths[("r8", "oracle")],
        baseline_full_summary=paths[("r8", "full")],
        candidate_predicted_summary=paths[("r16", "predicted")],
        candidate_oracle_summary=paths[("r16", "oracle")],
        candidate_full_summary=paths[("r16", "full")],
        baseline_training_report=r8_training,
        candidate_training_report=r16_training,
        pilot_checkpoint=pilot,
        semantic_summary=semantic,
        output_path=output,
    )

    assert report["matched_protocol"]["programs_sha256"] == "programs"
    assert report["rank_deltas_r16_minus_r8"]["predicted"]["answer"] == 0
    assert report["rank_deltas_r16_minus_r8"]["predicted"]["parse"] == 0.25
    assert report["context_answer_gains"] == {"r8": 0.25, "r16": 0.25}
    assert report["interventions_other_than_more_data"][0]["intervention"] == (
        "context_preserving_incremental_input"
    )
    assert output.exists()
