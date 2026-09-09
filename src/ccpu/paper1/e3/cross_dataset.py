"""Freeze leakage-safe arithmetic transfer benchmarks for Paper 1."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ccpu.common.artifacts import (
    file_sha256,
    fingerprint,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl_dataset.chop import chop_example


DATASET_SOURCES = {
    "asdiv": {
        "hub_id": "EleutherAI/asdiv",
        "revision": "8f95807222d87b4c688c3c22a6ba2801e1fa03e2",
        "license": "CC-BY-NC-4.0",
    },
    "svamp": {
        "hub_id": "ChilleD/SVAMP",
        "revision": "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
        "license": "MIT",
    },
    "mawps": {
        "hub_id": "MU-NLPC/Calc-mawps",
        "revision": "38c10053efeafd20ab6ff4e08c3ec17de26c19b7",
        "license": "MIT",
        "configuration": "original-splits",
    },
    "gsm_plus": {
        "hub_id": "qintongli/GSM-Plus",
        "revision": "3b708db57b96a16e8e3368ed2956990c0809440e",
        "license": "CC-BY-SA-4.0",
    },
}

_NUMBER = re.compile(r"[-+]?\d[\d,_]*(?:\.\d+)?(?:\s*/\s*\d[\d,_]*(?:\.\d+)?)?")
_OPERATORS = re.compile(r"(?<![eE])(?:\+|-|\*|/|\^)")


def _normalize_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _question_hash(value: Any) -> str:
    return fingerprint(_normalize_question(value))


def _answer(value: Any) -> str:
    text = str(value).strip().replace("$", "")
    match = _NUMBER.search(text)
    if not match:
        raise ValueError(f"answer has no numeric value: {value!r}")
    return re.sub(r"\s+", "", match.group(0)).replace(",", "").replace("_", "")


def _difficulty(expression: Any) -> tuple[int, str]:
    steps = max(1, len(_OPERATORS.findall(str(expression))))
    if steps <= 1:
        return steps, "low"
    if steps <= 3:
        return steps, "medium"
    return steps, "high"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix.casefold() == ".jsonl":
        return read_jsonl(path)
    raise ValueError(f"unsupported source format: {path}")


def _normalized_row(dataset: str, role: str, index: int, raw: dict[str, Any]) -> dict[str, Any]:
    if dataset == "asdiv":
        question = f"{raw['body'].strip()} {raw['question'].strip()}"
        answer = raw["answer"]
        expression = raw["formula"]
        category = raw.get("solution_type", "unknown")
        source_id = f"asdiv-{index:04d}"
        parent_id = source_id
    elif dataset == "svamp":
        question = raw.get("question_concat") or f"{raw['Body'].strip()} {raw['Question'].strip()}"
        answer = raw["Answer"]
        expression = raw["Equation"]
        category = raw.get("Type", "unknown")
        source_id = f"svamp-{raw.get('ID', index)}"
        parent_id = source_id
    elif dataset == "mawps":
        question = raw["question"]
        answer = raw["result"]
        expression = raw.get("expression") or raw.get("equation", "")
        category = "arithmetic"
        source_id = str(raw.get("id", f"mawps-{index:04d}"))
        parent_id = source_id
    elif dataset == "gsm_plus":
        question = raw["question"]
        answer = raw["answer"]
        expression = raw.get("solution", "")
        category = raw["perturbation_type"]
        parent_hash = _question_hash(raw["seed_question"])
        source_id = f"gsm-plus-{index:05d}"
        parent_id = f"gsm8k-seed-{parent_hash[:16]}"
    else:
        raise ValueError(f"unsupported cross-dataset benchmark: {dataset}")

    steps, stratum = _difficulty(expression)
    question_hash = _question_hash(question)
    return {
        "schema_version": "ccpu.paper1.arithmetic_transfer_source.v1",
        "example_id": source_id,
        "parent_example_id": parent_id,
        "dataset": dataset,
        "source_split": role,
        "source_row": index,
        "question": question,
        "question_sha256": question_hash,
        "reference_return": _answer(answer),
        "difficulty_steps": steps,
        "difficulty_stratum": stratum,
        "semantic_category": str(category),
        "effective_scope": {"id": "global"},
        "source_fields_visible_to_model": ["question"],
        "supervision": {
            "expression": str(expression),
            "source_answer": str(answer),
        },
    }


def _eval_view(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in (dict(row, split=split)).items()
        if key != "supervision"
    }


def _training_question(row: dict[str, Any]) -> str:
    if "question" in row:
        return str(row["question"])
    prompt = str(row.get("prompt", ""))
    marker = "\nProblem: "
    if marker not in prompt or not prompt.endswith("\nASL:"):
        raise ValueError("training row has neither question nor registered autonomous prompt")
    return prompt.split(marker, 1)[1][: -len("\nASL:")]


def _rank(row: dict[str, Any], seed: int, label: str) -> str:
    return fingerprint(f"{seed}:{label}:{row['example_id']}")


def _balanced_select(
    rows: list[dict[str, Any]], count: int, seed: int, label: str
) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError(f"{label} size {count} is invalid for {len(rows)} available rows")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["difficulty_stratum"]), str(row["semantic_category"]))
        buckets.setdefault(key, []).append(row)
    for key, members in buckets.items():
        buckets[key] = sorted(members, key=lambda row: _rank(row, seed, label))
    selected: list[dict[str, Any]] = []
    offsets = Counter()
    keys = sorted(buckets, key=lambda value: fingerprint(f"{seed}:{label}:{value}"))
    while len(selected) < count:
        progressed = False
        for key in keys:
            if offsets[key] < len(buckets[key]) and len(selected) < count:
                selected.append(buckets[key][offsets[key]])
                offsets[key] += 1
                progressed = True
        if not progressed:
            raise AssertionError("balanced selection exhausted before target")
    return sorted(selected, key=lambda row: (str(row["source_split"]), int(row["source_row"])))


def _select_distinct_parents(
    rows: list[dict[str, Any]], count: int, seed: int, label: str
) -> list[dict[str, Any]]:
    selected = []
    parents = set()
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["semantic_category"]), []).append(row)
    for category, members in buckets.items():
        buckets[category] = sorted(members, key=lambda row: _rank(row, seed, label))
    offsets = Counter()
    categories = sorted(buckets, key=lambda value: fingerprint(f"{seed}:{label}:{value}"))
    while len(selected) < count:
        progressed = False
        for category in categories:
            while offsets[category] < len(buckets[category]):
                row = buckets[category][offsets[category]]
                offsets[category] += 1
                if row["parent_example_id"] in parents:
                    continue
                selected.append(row)
                parents.add(row["parent_example_id"])
                progressed = True
                break
            if len(selected) == count:
                return sorted(selected, key=lambda value: int(value["source_row"]))
        if not progressed:
            break
    raise ValueError(f"only {len(selected)} distinct parents available for {label}")


def _adapter_files(adapter_path: str | Path) -> dict[str, str]:
    root = Path(adapter_path)
    if not root.is_dir():
        raise ValueError(f"GSM adapter directory not found: {root}")
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError(f"GSM adapter directory is empty: {root}")
    return files


def freeze_cross_dataset_benchmark(
    *,
    dataset: str,
    source_paths: dict[str, str | Path],
    expected_sha256: dict[str, str],
    gsm_train_paths: list[str | Path],
    gsm_adapter_path: str | Path,
    output_dir: str | Path,
    diagnostic_size: int = 250,
    dev_size: int = 100,
    seed: int = 93001,
) -> dict[str, Any]:
    """Freeze E0 evaluation and disjoint E1/E2 source pools."""

    if dataset not in DATASET_SOURCES:
        raise ValueError(f"unknown dataset: {dataset}")
    paths = {role: Path(path) for role, path in source_paths.items()}
    if set(paths) != set(expected_sha256):
        raise ValueError("source roles and checksum roles differ")
    observed_sha256 = {role: file_sha256(path) for role, path in paths.items()}
    mismatches = {
        role: digest
        for role, digest in observed_sha256.items()
        if digest.casefold() != expected_sha256[role].casefold()
    }
    if mismatches:
        raise ValueError(f"source checksum mismatch: {mismatches}")

    normalized: dict[str, list[dict[str, Any]]] = {}
    unsupported_exclusions = []
    for role, path in paths.items():
        normalized[role] = []
        for index, row in enumerate(_read_rows(path)):
            try:
                normalized[role].append(_normalized_row(dataset, role, index, row))
            except ValueError as error:
                unsupported_exclusions.append(
                    {"role": role, "source_row": index, "reason": str(error)}
                )

    gsm_hashes = {
        _question_hash(_training_question(row))
        for path in gsm_train_paths
        for row in read_jsonl(path)
    }
    excluded_gsm = []
    for role, rows in normalized.items():
        kept = []
        for row in rows:
            if row["question_sha256"] in gsm_hashes:
                excluded_gsm.append({"role": role, "example_id": row["example_id"]})
            else:
                kept.append(row)
        normalized[role] = kept

    duplicate_exclusions = []
    seen = set()
    role_priority = [
        role
        for role in ("test", "validation", "dev", "train", "all")
        if role in normalized
    ]
    for role in role_priority:
        kept = []
        for row in normalized[role]:
            digest = row["question_sha256"]
            if digest in seen:
                duplicate_exclusions.append({"role": role, "example_id": row["example_id"]})
            else:
                seen.add(digest)
                kept.append(row)
        normalized[role] = kept

    if dataset == "asdiv":
        pool = normalized["all"]
        diagnostic = _balanced_select(pool, diagnostic_size, seed, "diagnostic")
        diagnostic_ids = {row["example_id"] for row in diagnostic}
        remainder = [row for row in pool if row["example_id"] not in diagnostic_ids]
        dev = _balanced_select(remainder, dev_size, seed, "dev")
        dev_ids = {row["example_id"] for row in dev}
        train = [row for row in remainder if row["example_id"] not in dev_ids]
        full_test = diagnostic
    elif dataset in {"svamp", "mawps"}:
        full_test = normalized["test"]
        diagnostic = _balanced_select(full_test, diagnostic_size, seed, "diagnostic")
        dev_pool = normalized.get("validation") or normalized.get("dev") or normalized["train"]
        dev = _balanced_select(dev_pool, min(dev_size, len(dev_pool)), seed, "dev")
        dev_hashes = {row["question_sha256"] for row in dev}
        train = [row for row in normalized["train"] if row["question_sha256"] not in dev_hashes]
    else:
        pool = normalized["test"]
        diagnostic = _select_distinct_parents(pool, diagnostic_size, seed, "diagnostic")
        diagnostic_parents = {row["parent_example_id"] for row in diagnostic}
        remaining_parents = sorted(
            {row["parent_example_id"] for row in pool} - diagnostic_parents,
            key=lambda value: fingerprint(f"{seed}:dev-parent:{value}"),
        )
        selected_dev_parents = set(remaining_parents[:dev_size])
        dev = [row for row in pool if row["parent_example_id"] in selected_dev_parents]
        train = [
            row
            for row in pool
            if row["parent_example_id"] not in diagnostic_parents | selected_dev_parents
        ]
        full_test = pool

    split_parent_sets = {
        "diagnostic": {row["parent_example_id"] for row in diagnostic},
        "dev": {row["parent_example_id"] for row in dev},
        "train": {row["parent_example_id"] for row in train},
    }
    parent_overlaps = {
        "diagnostic_dev": sorted(split_parent_sets["diagnostic"] & split_parent_sets["dev"]),
        "diagnostic_train": sorted(split_parent_sets["diagnostic"] & split_parent_sets["train"]),
        "dev_train": sorted(split_parent_sets["dev"] & split_parent_sets["train"]),
    }
    if any(parent_overlaps.values()):
        raise ValueError(f"parent leakage across frozen splits: {parent_overlaps}")

    output = Path(output_dir)
    output_paths = {
        "e0_full": write_jsonl(
            output / "e0_full.jsonl", (_eval_view(row, "e0_full") for row in full_test)
        ),
        "diagnostic": write_jsonl(
            output / "diagnostic.jsonl",
            (_eval_view(row, "diagnostic") for row in diagnostic),
        ),
        "dev": write_jsonl(output / "dev.jsonl", (_eval_view(row, "dev") for row in dev)),
        "train_source": write_jsonl(
            output / "train_source.jsonl", (dict(row, split="train") for row in train)
        ),
    }
    manifest = {
        "schema_version": "ccpu.paper1.arithmetic_transfer_manifest.v1",
        "dataset": dataset,
        "source_registry": DATASET_SOURCES[dataset],
        "selection_seed": seed,
        "protocol": {
            "E0": "existing GSM LoRA, no target training",
            "E1": "copy GSM LoRA, then continue on target train_source",
            "E2": "fresh target LoRA from the same base model and train_source",
            "comparison": "E1 and E2 must match target rows and optimizer budget",
        },
        "roles": {
            "diagnostic": "immutable 250-item E0/E1/E2 comparison; never train",
            "dev": "selection only; never report as test",
            "train_source": "target annotation pool; no ASL target yet",
            "e0_full": "complete upstream test view for E0 before target training",
        },
        "counts": {
            "source": {role: len(_read_rows(path)) for role, path in paths.items()},
            "after_audit": {role: len(rows) for role, rows in normalized.items()},
            "e0_full": len(full_test),
            "diagnostic": len(diagnostic),
            "dev": len(dev),
            "train_source": len(train),
            "diagnostic_by_category": dict(Counter(row["semantic_category"] for row in diagnostic)),
            "diagnostic_by_difficulty": dict(
                Counter(row["difficulty_stratum"] for row in diagnostic)
            ),
        },
        "leakage_audit": {
            "gsm_train_files": {str(path): file_sha256(path) for path in gsm_train_paths},
            "excluded_exact_gsm_train_questions": excluded_gsm,
            "excluded_cross_split_duplicates": duplicate_exclusions,
            "excluded_unsupported_rows": unsupported_exclusions,
            "parent_overlaps": parent_overlaps,
            "passed": not any(parent_overlaps.values()),
        },
        "gsm_adapter_snapshot": {
            "path": str(gsm_adapter_path),
            "files": _adapter_files(gsm_adapter_path),
            "immutable_for_E0_E1_E2": True,
        },
        "sources": {
            role: {"path": str(path), "sha256": observed_sha256[role]}
            for role, path in paths.items()
        },
        "outputs": {
            role: {"path": str(path), "sha256": file_sha256(path)}
            for role, path in output_paths.items()
        },
        "answers_visible_to_model": False,
        "equations_visible_to_model": False,
        "rationales_visible_to_model": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def build_cross_dataset_teacher_seed(
    *, frozen_dir: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Build scorer-rich seeds whose downstream teacher view remains question-only."""

    frozen = Path(frozen_dir)
    manifest_path = frozen / "manifest.json"
    train_path = frozen / "train_source.jsonl"
    manifest = read_json(manifest_path)
    if file_sha256(train_path) != manifest["outputs"]["train_source"]["sha256"]:
        raise ValueError("frozen train_source checksum mismatch")
    rows = []
    for source in read_jsonl(train_path):
        seed = {
            "schema_version": "ccpu.dsl_dataset.raw_record.v1",
            "dataset": source["dataset"],
            "split": "train",
            "source_id": source["example_id"],
            "question": source["question"],
            "answer": source["reference_return"],
            "gold_reasoning": source["supervision"]["expression"],
            "effective_scope": source["effective_scope"],
            "source_context": None,
            "metadata": {
                "arithmetic_compatible": True,
                "semantic_category": source["semantic_category"],
                "parent_example_id": source["parent_example_id"],
                "teacher_visible_fields": ["question"],
            },
        }
        seed["parts"] = chop_example(seed)
        seed["record_sha256"] = fingerprint(
            {
                "dataset": seed["dataset"],
                "split": seed["split"],
                "source_id": seed["source_id"],
                "question": seed["question"],
            }
        )
        rows.append(seed)
    path = write_jsonl(output_path, rows)
    report = {
        "schema_version": "ccpu.paper1.cross_dataset_teacher_seed_manifest.v1",
        "dataset": manifest["dataset"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_train_sha256": file_sha256(train_path),
        "output": str(path),
        "output_sha256": file_sha256(path),
        "count": len(rows),
        "teacher_visible_fields": ["question"],
        "answers_hidden_during_primary_annotation": True,
        "rationales_hidden_during_primary_annotation": True,
    }
    write_json(Path(output_path).with_suffix(".manifest.json"), report)
    return report
