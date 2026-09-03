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
