"""Condition-blind integrity and semantic-consistency audit for interventions."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl import validate_asl

_OPERATOR_PREAMBLE = re.compile(
    r"^First determine a quantity defined as .*? In the problem below, "
    r"'this derived quantity' refers to that value\.\s*",
    re.IGNORECASE | re.DOTALL,
)
_HOW_MANY = re.compile(r"\bhow many\s+(?:more\s+)?(?P<noun>[a-z-]+)", re.IGNORECASE)
_PARTITION = re.compile(
    r"\b(?P<label>first|last|all)\s+(?P<count>[\d,]+)\s+"
    r"(?P<unit>days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)
_NONNEGATIVE_PATH = re.compile(
    r"(?:^|\.)(?:remaining|additional|needed|left|count|quantity|inventory|"
    r"distance|duration|pieces|items|people|students|children)(?:_|\.|$)",
    re.IGNORECASE,
)
_DISCRETE_NOUNS = {
    "apple",
    "apples",
    "bag",
    "bags",
    "book",
    "books",
    "box",
    "boxes",
    "card",
    "cards",
    "child",
    "children",
    "cookie",
    "cookies",
    "day",
    "days",
    "dog",
    "dogs",
    "game",
    "games",
    "hat",
    "hats",
    "item",
    "items",
    "kid",
    "kids",
    "person",
    "people",
    "piece",
    "pieces",
    "student",
    "students",
    "ticket",
    "tickets",
}
_ACTION_TOKENS = {
    "donated",
    "eaten",
    "gave",
    "lost",
    "removed",
    "repacked",
    "sold",
    "spent",
    "used",
}
_LIMIT_TOKENS = {"available", "bought", "capacity", "count", "had", "initial", "total"}
_GENERIC_PATH_TOKENS = {
    "amount",
    "count",
    "initial",
    "number",
    "quantity",
    "total",
    "value",
}
_BOUNDED_PATH_LIMITS = {
    "hours_per_day": Decimal(24),
    "days_per_week": Decimal(7),
    "months_per_year": Decimal(12),
    "minutes_per_hour": Decimal(60),
    "seconds_per_minute": Decimal(60),
}


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _flag(code: str, category: str, evidence: str) -> dict[str, str]:
    return {"code": code, "category": category, "evidence": evidence}


def _materialized_question(row: dict[str, Any]) -> str:
    question = _OPERATOR_PREAMBLE.sub("", str(row.get("question", "")))
    transformed = row.get("transformation", {}).get("transformed_value")
    if transformed is not None:
        question = re.sub(
            r"\bthis derived quantity\b",
            f"{transformed:,}" if isinstance(transformed, int) else str(transformed),
            question,
            flags=re.IGNORECASE,
        )
    return question


def _execution_audit(row: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    scope_id = str(row.get("example_id", "audit"))
    validation = validate_asl(
        str(row.get("gold_asl", "")),
        effective_scope={
            "id": scope_id,
            "parent": None,
            "kind": "benchmark_case",
            "source": "paper1_intervention_audit",
        },
    )
    checks = {
        "question_hash_matches": row.get("question_sha256")
        == fingerprint(str(row.get("question", ""))),
        "syntax_verified": bool(validation["syntax_verified"]),
        "lower_verified": bool(validation["lower_verified"]),
        "type_verified": bool(validation["type_verified"]),
        "execution_verified": bool(validation["execution_verified"]),
        "reference_return_matches": False,
    }
    workspace: dict[str, Any] = {}
    if validation["execution_verified"]:
        workspace = validation["execution"]["workspace"].get(scope_id, {})
        observed = _decimal(workspace.get("returned"))
        expected = _decimal(row.get("reference_return"))
        checks["reference_return_matches"] = observed is not None and observed == expected
    return checks, workspace


def _partition_flags(question: str) -> list[dict[str, str]]:
    groups: dict[str, dict[str, int]] = defaultdict(dict)
    for match in _PARTITION.finditer(question):
        groups[match.group("unit").lower().rstrip("s")][match.group("label").lower()] = int(
            match.group("count").replace(",", "")
        )
    flags = []
    for unit, values in groups.items():
        if {"first", "last", "all"} <= values.keys() and (
            values["first"] + values["last"] != values["all"]
        ):
            flags.append(
                _flag(
                    "partition_total_mismatch",
                    "semantic_consistency",
                    f"first {values['first']} + last {values['last']} != all {values['all']} {unit}s",
                )
            )
    return flags


def _component_limit_flags(
    row: dict[str, Any], values: dict[str, Any]
) -> list[dict[str, str]]:
    transformation = row.get("transformation", {})
    binding_path = str(transformation.get("binding_path", ""))
    transformed = _decimal(transformation.get("transformed_value"))
    binding_tokens = set(binding_path.lower().split("."))
    if transformed is None or not (binding_tokens & _ACTION_TOKENS):
        return []
    units = binding_tokens - _ACTION_TOKENS - _GENERIC_PATH_TOKENS
    flags = []
    for path, raw_value in values.items():
        if path.startswith("cogcop_probe.") or path == binding_path:
            continue
        path_tokens = set(path.lower().split("."))
        limit = _decimal(raw_value)
        if (
            limit is not None
            and limit >= 0
            and transformed > limit
            and path_tokens & _LIMIT_TOKENS
            and units & path_tokens
        ):
            flags.append(
                _flag(
                    "component_exceeds_available_total",
                    "semantic_consistency",
                    f"{binding_path}={transformed} exceeds {path}={limit}",
                )
            )
            break
    return flags


def _semantic_flags(
    row: dict[str, Any], workspace: dict[str, Any]
) -> list[dict[str, str]]:
    question = _materialized_question(row)
    answer = _decimal(row.get("reference_return"))
    values = workspace.get("values", {}) if isinstance(workspace, dict) else {}
    flags: list[dict[str, str]] = []

    if answer is not None and answer < 0 and re.search(
        r"\b(?:how (?:many|much) more|remain(?:ing)?|left|needed?)\b",
        question,
        re.IGNORECASE,
    ):
        flags.append(
            _flag(
                "negative_requested_quantity",
                "semantic_consistency",
                f"requested nonnegative quantity has reference {answer}",
            )
        )

    negative_paths = [
        f"{path}={value}"
        for path, value in values.items()
        if not path.startswith("cogcop_probe.")
        and (_decimal(value) is not None and _decimal(value) < 0)
        and _NONNEGATIVE_PATH.search(path)
    ]
    if negative_paths:
        flags.append(
            _flag(
                "negative_world_quantity",
                "semantic_consistency",
                "; ".join(negative_paths[:3]),
            )
        )

    target = list(_HOW_MANY.finditer(question))
    if answer is not None and answer != answer.to_integral_value() and target:
        noun = target[-1].group("noun").lower()
        if noun in _DISCRETE_NOUNS:
            flags.append(
                _flag(
                    "fractional_discrete_answer",
                    "semantic_consistency",
                    f"how many {noun} has non-integral reference {answer}",
                )
            )

    flags.extend(_partition_flags(question))
    flags.extend(_component_limit_flags(row, values))

    transformation = row.get("transformation", {})
    binding_path = str(transformation.get("binding_path", "")).lower()
    transformed = _decimal(transformation.get("transformed_value"))
    if transformed is not None:
        for suffix, limit in _BOUNDED_PATH_LIMITS.items():
            if suffix in binding_path and transformed > limit:
                flags.append(
                    _flag(
                        "bounded_rate_or_duration_exceeded",
                        "world_plausibility",
                        f"{binding_path}={transformed} exceeds natural bound {limit}",
                    )
                )
                break
        if transformed >= 1000:
            flags.append(
                _flag(
                    "extreme_world_magnitude",
                    "intervention_stratum",
                    f"{binding_path}={transformed}",
                )
            )

    raw_question = str(row.get("question", ""))
    if re.search(r"\bthis derived quantity-[a-z]+", raw_question, re.IGNORECASE):
        flags.append(
            _flag(
                "degraded_hyphenated_rewrite",
                "surface_quality",
                "operator placeholder replaced a numeric compound modifier",
            )
        )
    return flags


def audit_intervention_panel(
    *, input_dirs: list[str | Path], output_dir: str | Path, split: str = "test"
) -> dict[str, Any]:
    """Audit all matching split files without modifying or excluding source rows."""

    output = Path(output_dir)
    roots = [Path(path) for path in input_dirs]
    files: list[tuple[Path, Path]] = []
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"intervention audit input is not a directory: {root}")
        files.extend((root, path) for path in sorted(root.rglob(f"{split}.jsonl")))
    if not files:
        raise ValueError(f"no {split}.jsonl files found under the supplied input directories")

    audited: list[dict[str, Any]] = []
    file_parent_sets: dict[str, set[str]] = {}
    input_hashes: dict[str, str] = {}
    for root, path in files:
        key = f"{root.name}/{path.relative_to(root).as_posix()}"
        if key in input_hashes:
            key = f"{root.parent.name}/{key}"
        input_hashes[key] = file_sha256(path)
        rows = read_jsonl(path)
        example_ids = [str(row.get("example_id", "")) for row in rows]
        duplicate_ids = {item for item, count in Counter(example_ids).items() if count > 1}
        file_parent_sets[key] = {str(row.get("parent_example_id", "")) for row in rows}
        for index, row in enumerate(rows):
            checks, workspace = _execution_audit(row)
            checks["example_id_unique_within_file"] = example_ids[index] not in duplicate_ids
            flags = _semantic_flags(row, workspace)
            integrity_pass = all(checks.values())
            review_required = any(
                flag["category"]
                in {"semantic_consistency", "world_plausibility", "surface_quality"}
                for flag in flags
            )
            audited.append(
                {
                    "schema_version": "ccpu.paper1.intervention_audit.v1",
                    "source_file": key,
                    "source_index": index,
                    "example_id": row.get("example_id"),
                    "parent_example_id": row.get("parent_example_id"),
                    "condition": row.get("condition"),
                    "operator_level": row.get("operator_level"),
                    "factor": row.get("transformation", {}).get("factor"),
                    "jitter_seed": row.get("transformation", {}).get("jitter_seed"),
                    "question": row.get("question"),
                    "materialized_question": _materialized_question(row),
                    "reference_return": row.get("reference_return"),
                    "integrity_checks": checks,
                    "integrity_pass": integrity_pass,
                    "semantic_flags": flags,
                    "semantic_review_status": (
                        "manual_review_required" if review_required else "no_review_flag"
                    ),
                    "analysis_disposition": (
                        "integrity_failure"
                        if not integrity_pass
                        else "manual_review_required"
                        if review_required
                        else "eligible_pending_manual_review"
                    ),
                }
            )

    parent_sets = list(file_parent_sets.values())
    paired_parent_sets = all(values == parent_sets[0] for values in parent_sets[1:])
    review_parent_ids = {
        str(record["parent_example_id"])
        for record in audited
        if record["analysis_disposition"] != "eligible_pending_manual_review"
    }
    common_parent_ids = sorted(parent_sets[0] - review_parent_ids)
    for record in audited:
        record["strict_common_support"] = record["parent_example_id"] in common_parent_ids
    flag_counts = Counter(
        flag["code"] for record in audited for flag in record["semantic_flags"]
    )
    flagged_parents: dict[str, set[str]] = defaultdict(set)
    for record in audited:
        for flag in record["semantic_flags"]:
            flagged_parents[flag["code"]].add(str(record["parent_example_id"]))
    by_file = {}
    for key in input_hashes:
        records = [record for record in audited if record["source_file"] == key]
        by_file[key] = {
            "rows": len(records),
            "integrity_failures": sum(not record["integrity_pass"] for record in records),
            "flagged_rows": sum(bool(record["semantic_flags"]) for record in records),
            "flags": dict(
                sorted(
                    Counter(
                        flag["code"]
                        for record in records
                        for flag in record["semantic_flags"]
                    ).items()
                )
            ),
        }

    records_path = write_jsonl(output / "audit_records.jsonl", audited)
    review_path = write_jsonl(
        output / "manual_review.jsonl",
        (record for record in audited if record["analysis_disposition"] != "eligible_pending_manual_review"),
    )
    common_support_path = write_json(
        output / "strict_common_support.json",
        {
            "schema_version": "ccpu.paper1.intervention_common_support.v1",
            "policy": (
                "Condition-independent parent IDs with no integrity, semantic-consistency, "
                "bounded-plausibility, or surface-quality review flag in any audited cell."
            ),
            "parent_example_ids": common_parent_ids,
        },
    )
    report = {
        "schema_version": "ccpu.paper1.intervention_audit_summary.v1",
        "policy": {
            "condition_blind": True,
            "automatic_exclusion": False,
            "interpretation": (
                "Integrity failures invalidate a record. Semantic and plausibility flags "
                "define a documented manual-review queue and sensitivity strata; they do "
                "not silently remove records."
            ),
        },
        "split": split,
        "input_roots": [str(root) for root in roots],
        "input_sha256": input_hashes,
        "counts": {
            "files": len(files),
            "rows": len(audited),
            "unique_parent_examples": len(
                {str(record["parent_example_id"]) for record in audited}
            ),
            "integrity_failures": sum(not record["integrity_pass"] for record in audited),
            "labeled_rows": sum(bool(record["semantic_flags"]) for record in audited),
            "review_required_rows": sum(
                record["analysis_disposition"] == "manual_review_required"
                for record in audited
            ),
            "review_required_unique_parents": len(review_parent_ids),
            "strict_common_support_parents": len(common_parent_ids),
            "labeled_unique_parents": len(
                {
                    str(record["parent_example_id"])
                    for record in audited
                    if record["semantic_flags"]
                }
            ),
        },
        "pairing_audit": {
            "identical_parent_sets_across_files": paired_parent_sets,
            "parent_count_by_file": {
                key: len(values) for key, values in file_parent_sets.items()
            },
        },
        "flag_counts": dict(sorted(flag_counts.items())),
        "flagged_unique_parent_counts": {
            code: len(parents) for code, parents in sorted(flagged_parents.items())
        },
        "by_file": by_file,
        "outputs": {
            "audit_records": {"path": str(records_path), "sha256": file_sha256(records_path)},
            "manual_review": {"path": str(review_path), "sha256": file_sha256(review_path)},
            "strict_common_support": {
                "path": str(common_support_path),
                "sha256": file_sha256(common_support_path),
            },
        },
    }
    summary_path = output / "summary.json"
    report["outputs"]["summary"] = {"path": str(summary_path)}
    write_json(summary_path, report)
    return report
