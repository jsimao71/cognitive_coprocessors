"""Build and audit matched semantic-bottleneck data from the frozen F0 corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from ccpu.dsl import validate_asl
from ccpu.paper1.asl_pilot_eval import score_asl

from .bottleneck import asl_to_bottleneck, lower_bottleneck_to_asl, render_bottleneck
from .components import component_labels
from .negatives import generate_hard_negatives

FIXED_PROMPT = (
    "Compile the quantitative problem into the canonical semantic bottleneck JSON. "
    "Ground every symbol in bindings, preserve source facts and dependencies, and emit "
    "exactly one final return. Return only one JSON object; do not explain.\n\n"
    "Input:\nProblem: {problem}\nSemantic bottleneck:"
)


def _build_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = str(row["target_asl"])
    scope = dict(row["effective_scope"])
    program = asl_to_bottleneck(reference, effective_scope=scope)
    target = render_bottleneck(program)
    lowered = lower_bottleneck_to_asl(target)
    validation = validate_asl(lowered, effective_scope=scope)
    score = score_asl(reference, lowered, scope)
    if not validation["execution_verified"]:
        raise AssertionError(f"bottleneck does not execute for {row['example_id']}")
    required = (
        "semantic_return_equivalent",
        "semantic_state_equivalent",
        "final_answer_correct",
        "dependency_correct",
    )
    if not all(score[name] for name in required):
        raise AssertionError(f"bottleneck round-trip changed semantics for {row['example_id']}")
    labels = component_labels(program)
    output = {
        "schema_version": "ccpu.paper1.e3_bottleneck_sft.v1",
        "example_id": f"e3b-{row['split']}-{fingerprint(row['example_id'], 16)}",
        "parent_example_id": row["example_id"],
        "parent_source_id": row["parent_source_id"],
        "dataset": row["dataset"],
        "semantic_pattern_id": row["semantic_pattern_id"],
        "split": row["split"],
        "prompt": FIXED_PROMPT.format(problem=str(row["nl_input"]).strip()),
        "target": target,
        "target_asl": reference,
        "effective_scope": scope,
        "component_labels": labels,
        "source_fields_visible_to_model": ["nl_input"],
        "prompt_policy": "fixed-no-icl-v1",
        "roundtrip": {
            "lowered_asl": lowered,
            "parse_valid": score["parse_valid"],
            "lowerable": score["lowerable_to_ccir"],
            "type_valid": score["type_valid"],
            "executable": score["executable"],
            "semantic_return_equivalent": score["semantic_return_equivalent"],
            "semantic_state_equivalent": score["semantic_state_equivalent"],
            "final_answer_correct": score["final_answer_correct"],
        },
    }
    negatives = generate_hard_negatives(
        program, reference_asl=reference, effective_scope=scope
    )
    return output, negatives


def build_bottleneck_data(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create matched 450/25/25 data and executable train/dev hard negatives."""

    source = Path(data_dir) / "source"
    output = Path(output_dir)
    expected = {"train": 450, "dev": 25, "test": 25}
    files: dict[str, Any] = {}
    negative_counts: Counter[str] = Counter()
    identities: dict[str, set[tuple[str, str]]] = {}
    patterns: dict[str, set[str]] = {}
    for split, expected_count in expected.items():
        rows = read_jsonl(source / f"{split}.jsonl")
        if len(rows) != expected_count:
            raise AssertionError(f"expected {expected_count} {split} rows, found {len(rows)}")
        built = []
        negatives = []
        identities[split] = set()
        patterns[split] = set()
        for row in rows:
            item, item_negatives = _build_row(row)
            built.append(item)
            identity = (str(row["dataset"]), str(row["parent_source_id"]))
            if identity in identities[split]:
                raise AssertionError(f"duplicate source identity in {split}: {identity}")
            identities[split].add(identity)
            patterns[split].add(str(row["semantic_pattern_id"]))
            if split != "test":
                for index, negative in enumerate(item_negatives):
                    negative_counts[negative["negative_type"]] += 1
                    negatives.append(
                        {
                            "schema_version": "ccpu.paper1.e3_hard_negative.v1",
                            "example_id": f"{item['example_id']}-neg-{index}",
                            "parent_example_id": item["example_id"],
                            "split": split,
                            "positive_target": item["target"],
                            **negative,
                        }
                    )
        path = write_jsonl(output / "sft" / f"{split}.jsonl", built)
        files[split] = {"rows": len(built), "sha256": file_sha256(path)}
        if split != "test":
            negative_path = write_jsonl(output / "negatives" / f"{split}.jsonl", negatives)
            files[f"{split}_negatives"] = {
                "rows": len(negatives),
                "sha256": file_sha256(negative_path),
            }
    overlap = {
        "train_dev_sources": sorted(identities["train"] & identities["dev"]),
        "train_test_sources": sorted(identities["train"] & identities["test"]),
        "dev_test_sources": sorted(identities["dev"] & identities["test"]),
        "train_test_patterns": sorted(patterns["train"] & patterns["test"]),
        "dev_test_patterns": sorted(patterns["dev"] & patterns["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"bottleneck split leakage: {overlap}")
    manifest = {
        "schema_version": "ccpu.paper1.e3_bottleneck_manifest.v1",
        "representation": "ASL-isomorphic symbol table plus expression graph",
        "prompt_policy": "one fixed zero-shot prompt; no record-dependent ICL or state",
        "source_data_dir": str(Path(data_dir).resolve()),
        "source_files": {
            split: file_sha256(source / f"{split}.jsonl") for split in expected
        },
        "files": files,
        "split_overlap": overlap,
        "gold_roundtrip_passed": sum(expected.values()),
        "gold_roundtrip_total": sum(expected.values()),
        "hard_negative_counts": dict(sorted(negative_counts.items())),
        "test_negatives_generated": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def build_direct_preference_data(
    qwen_data_dir: str | Path,
    bottleneck_data_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Attach one deterministic within-example ASL negative to each unchanged F0 row."""

    qwen = Path(qwen_data_dir)
    bottleneck = Path(bottleneck_data_dir)
    output = Path(output_dir)
    files: dict[str, Any] = {}
    selected_counts: Counter[str] = Counter()
    accidental_answers = 0
    for split in ("train", "dev"):
        semantic_rows = read_jsonl(bottleneck / "sft" / f"{split}.jsonl")
        by_matrix_id = {str(row["parent_example_id"]): row for row in semantic_rows}
        negatives_by_parent: dict[str, list[dict[str, Any]]] = {}
        for negative in read_jsonl(bottleneck / "negatives" / f"{split}.jsonl"):
            negatives_by_parent.setdefault(str(negative["parent_example_id"]), []).append(negative)
        for negatives in negatives_by_parent.values():
            negatives.sort(key=lambda row: (str(row["negative_type"]), str(row["example_id"])))

        source_rows = read_jsonl(qwen / f"{split}.jsonl")
        built = []
        for row in source_rows:
            matrix_id = str(row["parent_example_id"])
            semantic = by_matrix_id.get(matrix_id)
            if semantic is None:
                raise AssertionError(f"missing bottleneck identity for {matrix_id}")
            candidates = negatives_by_parent.get(str(semantic["example_id"]), [])
            if not candidates:
                raise AssertionError(f"no executable semantic negatives for {matrix_id}")
            if split == "train":
                choice_index = int(row.get("epoch_view", 0)) % len(candidates)
            else:
                choice_index = int(fingerprint(matrix_id, 8), 16) % len(candidates)
            negative = candidates[choice_index]
            selected_counts[str(negative["negative_type"])] += 1
            accidental_answers += int(bool(negative["final_answer_accidentally_correct"]))
            built.append(
                {
                    **row,
                    "schema_version": "ccpu.paper1.e3_direct_preference.v1",
                    "negative_target": str(negative["lowered_asl"]),
                    "negative_type": str(negative["negative_type"]),
                    "negative_example_id": str(negative["example_id"]),
                    "negative_binding_changed": bool(negative["binding_changed"]),
                    "negative_final_answer_accidentally_correct": bool(
                        negative["final_answer_accidentally_correct"]
                    ),
                    "preference_source": "deterministic-executable-within-example-ast",
                }
            )
        if any(
            built[index]["prompt"] != source_rows[index]["prompt"]
            or built[index]["target"] != source_rows[index]["target"]
            for index in range(len(built))
        ):
            raise AssertionError("preference construction changed an F0 prompt or positive target")
        path = write_jsonl(output / f"{split}.jsonl", built)
        files[split] = {"rows": len(built), "sha256": file_sha256(path)}

    total = sum(item["rows"] for item in files.values())
    manifest = {
        "schema_version": "ccpu.paper1.e3_direct_preference_manifest.v1",
        "positive_policy": "byte-equivalent existing F0 prompt and target",
        "negative_policy": "one executable within-example semantic corruption per row",
        "score_policy": "semantic-weighted conditional mean log likelihood",
        "test_rows_generated": 0,
        "files": files,
        "selected_negative_counts": dict(sorted(selected_counts.items())),
        "accidental_answer_match_rows": accidental_answers,
        "accidental_answer_match_rate": accidental_answers / total,
        "source_files": {
            f"qwen_{split}": file_sha256(qwen / f"{split}.jsonl")
            for split in ("train", "dev")
        }
        | {
            f"negative_{split}": file_sha256(
                bottleneck / "negatives" / f"{split}.jsonl"
            )
            for split in ("train", "dev")
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def build_bottleneck_preference_data(
    bottleneck_data_dir: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 10,
) -> dict[str, Any]:
    """Attach rotating native-F4 hard negatives to unchanged F4 positives."""

    if epochs < 1:
        raise ValueError("F4 preference epochs must be positive")
    bottleneck = Path(bottleneck_data_dir)
    output = Path(output_dir)
    files: dict[str, Any] = {}
    selected_counts: Counter[str] = Counter()
    for split in ("train", "dev"):
        positive_rows = read_jsonl(bottleneck / "sft" / f"{split}.jsonl")
        negatives_by_parent: dict[str, list[dict[str, Any]]] = {}
        for negative in read_jsonl(bottleneck / "negatives" / f"{split}.jsonl"):
            negatives_by_parent.setdefault(str(negative["parent_example_id"]), []).append(
                negative
            )
        for negatives in negatives_by_parent.values():
            negatives.sort(key=lambda row: (str(row["negative_type"]), str(row["example_id"])))

        built = []
        epoch_values = range(epochs) if split == "train" else (None,)
        for epoch in epoch_values:
            for row in positive_rows:
                candidates = negatives_by_parent.get(str(row["example_id"]), [])
                if not candidates:
                    raise AssertionError(f"no native-F4 negatives for {row['example_id']}")
                choice_index = (
                    int(epoch) % len(candidates)
                    if epoch is not None
                    else int(fingerprint(str(row["example_id"]), 8), 16) % len(candidates)
                )
                negative = candidates[choice_index]
                selected_counts[str(negative["negative_type"])] += 1
                built.append(
                    {
                        **row,
                        "schema_version": "ccpu.paper1.e3_bottleneck_preference.v1",
                        "example_id": (
                            f"{row['example_id']}-epoch-{epoch}"
                            if epoch is not None
                            else row["example_id"]
                        ),
                        "epoch_view": epoch,
                        "negative_target": render_bottleneck(negative["bottleneck"]),
                        "negative_type": str(negative["negative_type"]),
                        "negative_example_id": str(negative["example_id"]),
                        "preference_source": (
                            "deterministic-executable-within-example-native-f4"
                        ),
                    }
                )
        path = write_jsonl(output / f"{split}.jsonl", built)
        files[split] = {"rows": len(built), "sha256": file_sha256(path)}

    manifest = {
        "schema_version": "ccpu.paper1.e3_bottleneck_preference_manifest.v1",
        "representation_id": "F4",
        "objective_id": "L2",
        "positive_policy": "byte-equivalent existing F4 prompt and target",
        "negative_policy": "rotating executable native-F4 semantic corruption",
        "logical_epochs": epochs,
        "test_rows_generated": 0,
        "files": files,
        "selected_negative_counts": dict(sorted(selected_counts.items())),
        "source_files": {
            f"sft_{split}": file_sha256(bottleneck / "sft" / f"{split}.jsonl")
            for split in ("train", "dev")
        }
        | {
            f"negative_{split}": file_sha256(
                bottleneck / "negatives" / f"{split}.jsonl"
            )
            for split in ("train", "dev")
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
