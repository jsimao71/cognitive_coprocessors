"""Matched Qwen LoRA controls for pretrained ASL grounding patches."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl

from .data import MatrixExample, RegimeBuilder, StaticMixture

QWEN_CONDITIONS = ("q0_t3", "q1_serialized_mixed")

_INSTRUCTION = (
    "Compile the quantitative problem into semantically grounded ASL-Arith. "
    "Preserve entities, measured quantities, source facts, relations, temporal state, "
    "dependencies, and the requested RETURN. Use meaningful lowercase paths. "
    "Return only ASL, one statement per line; do not explain."
)


def _prompt(view: dict[str, Any]) -> str:
    prompt = f"{_INSTRUCTION}\n\nInput:\nProblem: {view['nl_input']}"
    if view["external_asl_input"] is not None:
        prompt += f"\n\nExternal ASL teacher:\n{view['external_asl_input']}"
    return prompt + "\nASL:"


def _examples(path: Path) -> list[MatrixExample]:
    return [MatrixExample(**row) for row in read_jsonl(path)]


def build_qwen_patch_data(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    condition: str,
    epochs: int = 10,
    seed: int = 11,
) -> dict[str, Any]:
    """Unfold runtime regimes into a target-token-matched causal SFT stream."""

    if condition not in QWEN_CONDITIONS:
        raise ValueError(f"unsupported Qwen matrix condition: {condition}")
    if epochs < 1:
        raise ValueError("Qwen matrix epochs must be positive")
    data = Path(data_dir)
    train = _examples(data / "source" / "train.jsonl")
    dev = _examples(data / "source" / "dev.jsonl")
    mixture = (
        StaticMixture(full_teacher=0.0, partial_teacher=0.0, autonomous=1.0)
        if condition == "q0_t3"
        else StaticMixture()
    )
    builder = RegimeBuilder(
        mixture=mixture,
        corruption_policies=("record_dropout", "value_mask", "argument_mask"),
        corruption_severity=0.5,
        seed=seed,
    )
    records = []
    regime_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    for epoch in range(epochs):
        for index, example in enumerate(train):
            regime = builder.sample_regime(example, epoch=epoch)
            noise = train[(index + 1) % len(train)].target_asl
            view = builder.make_view(example, regime=regime, epoch=epoch, noise_asl=noise)
            regime_counts[regime] += 1
            if view["external_asl_corruption"]:
                policy_counts[view["external_asl_corruption"]["policy"]] += 1
            records.append(
                {
                    "schema_version": "ccpu.paper1.asl_matrix.qwen_sft.v1",
                    "example_id": f"{condition}-{epoch}-{fingerprint(example.example_id, 12)}",
                    "parent_example_id": example.example_id,
                    "parent_source_id": example.parent_source_id,
                    "semantic_pattern_id": example.semantic_pattern_id,
                    "dataset": example.dataset,
                    "epoch_view": epoch,
                    "regime": regime,
                    "prompt": _prompt(view),
                    "target": example.target_asl,
                    "has_external_asl": view["has_external_asl"],
                    "external_asl_fraction": view["external_asl_fraction"],
                    "external_asl_corruption": view["external_asl_corruption"],
                    "source_fields_visible_to_model": view["source_fields_visible_to_model"],
                }
            )
    dev_records = []
    for example in dev:
        view = builder.make_view(example, regime="autonomous")
        dev_records.append(
            {
                "schema_version": "ccpu.paper1.asl_matrix.qwen_sft.v1",
                "example_id": f"{condition}-dev-{fingerprint(example.example_id, 12)}",
                "parent_example_id": example.example_id,
                "parent_source_id": example.parent_source_id,
                "semantic_pattern_id": example.semantic_pattern_id,
                "dataset": example.dataset,
                "epoch_view": None,
                "regime": "autonomous",
                "prompt": _prompt(view),
                "target": example.target_asl,
                "has_external_asl": False,
                "external_asl_fraction": 0.0,
                "external_asl_corruption": None,
                "source_fields_visible_to_model": ["nl_input"],
            }
        )
    if condition == "q0_t3" and any(row["has_external_asl"] for row in records):
        raise AssertionError("Q0 must contain only autonomous T3 views")
    output = Path(output_dir)
    train_path = write_jsonl(output / "train.jsonl", records)
    dev_path = write_jsonl(output / "dev.jsonl", dev_records)
    manifest = {
        "schema_version": "ccpu.paper1.asl_matrix.qwen_data.v1",
        "condition": condition,
        "seed": seed,
        "runtime_epochs_unfolded": epochs,
        "source_train_rows": len(train),
        "train_views": len(records),
        "dev_rows": len(dev_records),
        "target_exposures": len(records),
        "regime_counts": dict(sorted(regime_counts.items())),
        "corruption_policy_counts": dict(sorted(policy_counts.items())),
        "fixed_prompt": True,
        "document_dependent_icl": False,
        "target_visible_in_source_fields": False,
        "input_sha256": {
            "source_train": file_sha256(data / "source" / "train.jsonl"),
            "source_dev": file_sha256(data / "source" / "dev.jsonl"),
        },
        "output_sha256": {
            "train": file_sha256(train_path),
            "dev": file_sha256(dev_path),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
