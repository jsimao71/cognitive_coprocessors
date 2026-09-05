"""Paired direct-versus-ASL and large-number contribution analysis."""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json


def _parse_paths(values: list[tuple[str, str | Path]]) -> dict[str, Path]:
    paths = {}
    for label, value in values:
        if label in paths:
            raise ValueError(f"duplicate contribution label: {label}")
        paths[label] = Path(value)
    if not paths:
        raise ValueError("at least one contribution path is required")
    return paths


def _load_predictions(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {str(row["example_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate prediction identity in {path}")
    if set(indexed) != expected_ids:
        missing = sorted(expected_ids - set(indexed))
        unexpected = sorted(set(indexed) - expected_ids)
        raise ValueError(
            f"prediction identities differ in {path}: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    return indexed


def _correct(row: dict[str, Any]) -> bool:
    return bool(row["metrics"]["final_answer_correct"])


def _resources(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = [int(row["prompt_tokens"]) for row in rows.values()]
    generated_tokens = [int(row["generated_tokens"]) for row in rows.values()]
    wall_times = [int(row["wall_time_ns"]) for row in rows.values()]
    observed_maximum = max(generated_tokens)
    maximum_rows = [
        row
        for row in rows.values()
        if int(row["generated_tokens"]) == observed_maximum
    ]
    return {
        "count": len(rows),
        "mean_prompt_tokens": mean(prompt_tokens),
        "mean_generated_tokens": mean(generated_tokens),
        "total_generated_tokens": sum(generated_tokens),
        "observed_maximum_generated_tokens": observed_maximum,
        "observed_maximum_count": len(maximum_rows),
        "observed_maximum_rate": len(maximum_rows) / len(rows),
        "observed_maximum_scorable_count": sum(
            bool(row.get("metrics", {}).get("answer_scorable"))
            for row in maximum_rows
        ),
        "observed_maximum_correct_count": sum(_correct(row) for row in maximum_rows),
        "mean_generation_wall_time_ms": mean(wall_times) / 1e6,
        "total_generation_wall_time_seconds": sum(wall_times) / 1e9,
    }


def _mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired contribution identities differ")
    both_correct = sum(left[key] and right[key] for key in left)
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(not left[key] and right[key] for key in left)
    both_wrong = len(left) - both_correct - left_only - right_only
    return {
        "count": len(left),
        "both_correct": both_correct,
        "left_only": left_only,
        "right_only": right_only,
        "both_wrong": both_wrong,
        "left_rate": (both_correct + left_only) / len(left),
        "right_rate": (both_correct + right_only) / len(left),
        "delta": (left_only - right_only) / len(left),
        "two_sided_exact_mcnemar_p": _mcnemar(left_only, right_only),
    }


def _robustness(
    original: dict[str, dict[str, Any]],
    large: dict[str, dict[str, Any]],
    large_to_parent: dict[str, str],
) -> tuple[dict[str, Any], dict[str, int]]:
    original_vector = {
        large_id: int(_correct(original[parent_id]))
        for large_id, parent_id in large_to_parent.items()
    }
    large_vector = {large_id: int(_correct(large[large_id])) for large_id in large_to_parent}
    paired = _paired(
        {key: bool(value) for key, value in large_vector.items()},
        {key: bool(value) for key, value in original_vector.items()},
    )
    paired.update(
        {
            "original_eligible_rate": paired.pop("right_rate"),
            "large_rate": paired.pop("left_rate"),
            "large_minus_original": paired.pop("delta"),
            "retained_correct": paired.pop("both_correct"),
            "gained_on_large": paired.pop("left_only"),
            "lost_on_large": paired.pop("right_only"),
            "stable_wrong": paired.pop("both_wrong"),
        }
    )
    changes = {
        large_id: large_vector[large_id] - original_vector[large_id]
        for large_id in large_to_parent
    }
    return paired, changes


def _bootstrap_difference(
    left: dict[str, int], right: dict[str, int], *, seed: int, samples: int
) -> list[float]:
    if set(left) != set(right):
        raise ValueError("bootstrap identities differ")
    identities = sorted(left)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draws = [identities[rng.randrange(len(identities))] for _ in identities]
        estimates.append(mean(left[key] - right[key] for key in draws))
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return [low, high]


def _magnitude_band(value: Any) -> str:
    number = Fraction(str(value).replace(",", "").replace("$", ""))
    if number.denominator != 1:
        raise ValueError(f"large-number answer is not integral: {value}")
    digits = len(str(abs(number.numerator)))
    if digits <= 6:
        return "4_to_6_digits"
    if digits <= 9:
        return "7_to_9_digits"
    return "10plus_digits"


def _subgroup_robustness(
    *,
    groups: dict[str, set[str]],
    predictions: dict[str, dict[str, dict[str, dict[str, Any]]]],
    direct_labels: set[str],
    asl_labels: set[str],
    large_to_parent: dict[str, str],
) -> dict[str, Any]:
    report = {}
    for group, identities in sorted(groups.items()):
        mapping = {
            identity: large_to_parent[identity] for identity in sorted(identities)
        }
        robustness = {}
        changes = {}
        for family, labels in (("direct", direct_labels), ("asl", asl_labels)):
            for label in sorted(labels):
                original = predictions[f"original_{family}"][label]
                large = predictions[f"large_{family}"][label]
                result, change = _robustness(original, large, mapping)
                robustness[label] = result
                changes[label] = change
        differential = {
            f"{asl_label}__vs__{direct_label}": (
                robustness[asl_label]["large_minus_original"]
                - robustness[direct_label]["large_minus_original"]
            )
            for asl_label in sorted(asl_labels)
            for direct_label in sorted(direct_labels)
        }
        report[group] = {
            "count": len(identities),
            "robustness": robustness,
            "differential_degradation": differential,
        }
    return report


def analyze_gsm8k_contribution(
    *,
    original_eval_path: str | Path,
    large_eval_path: str | Path,
    original_direct_paths: list[tuple[str, str | Path]],
    original_asl_paths: list[tuple[str, str | Path]],
    large_direct_paths: list[tuple[str, str | Path]],
    large_asl_paths: list[tuple[str, str | Path]],
    output_path: str | Path,
    bootstrap_seed: int = 22903,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Analyze matched contribution without pooling ASL training seeds."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    original_eval = read_jsonl(original_eval_path)
    large_eval = read_jsonl(large_eval_path)
    original_ids = {str(row["example_id"]) for row in original_eval}
    large_ids = {str(row["example_id"]) for row in large_eval}
    if len(original_ids) != len(original_eval) or len(large_ids) != len(large_eval):
        raise ValueError("evaluation contains duplicate identities")
    large_to_parent = {
        str(row["example_id"]): str(row["parent_example_id"]) for row in large_eval
    }
    if len(set(large_to_parent.values())) != len(large_to_parent):
        raise ValueError("large-number parents are not one-to-one")
    if not set(large_to_parent.values()) <= original_ids:
        raise ValueError("large-number parent is outside original evaluation")

    path_groups = {
        "original_direct": _parse_paths(original_direct_paths),
        "original_asl": _parse_paths(original_asl_paths),
        "large_direct": _parse_paths(large_direct_paths),
        "large_asl": _parse_paths(large_asl_paths),
    }
    if set(path_groups["original_direct"]) != set(path_groups["large_direct"]):
        raise ValueError("direct labels differ between original and large evaluations")
    if set(path_groups["original_asl"]) != set(path_groups["large_asl"]):
        raise ValueError("ASL labels differ between original and large evaluations")
    if set(path_groups["original_direct"]) & set(path_groups["original_asl"]):
        raise ValueError("direct and ASL labels must be distinct")

    predictions = {
        group: {
            label: _load_predictions(path, large_ids if group.startswith("large") else original_ids)
            for label, path in paths.items()
        }
        for group, paths in path_groups.items()
    }
    original_vectors = {
        group: {
            label: {identity: _correct(row) for identity, row in rows.items()}
            for label, rows in predictions[group].items()
        }
        for group in ("original_direct", "original_asl")
    }

    original_pairwise = {}
    for asl_label, asl_vector in original_vectors["original_asl"].items():
        for direct_label, direct_vector in original_vectors["original_direct"].items():
            key = f"{asl_label}__vs__{direct_label}"
            result = _paired(asl_vector, direct_vector)
            result["delta_bootstrap_identity_95"] = _bootstrap_difference(
                {identity: int(value) for identity, value in asl_vector.items()},
                {identity: int(value) for identity, value in direct_vector.items()},
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            )
            original_pairwise[key] = result
    robustness = {}
    changes = {}
    for family in ("direct", "asl"):
        for label in path_groups[f"original_{family}"]:
            result, change = _robustness(
                predictions[f"original_{family}"][label],
                predictions[f"large_{family}"][label],
                large_to_parent,
            )
            result["large_minus_original_bootstrap_identity_95"] = (
                _bootstrap_difference(
                    change,
                    {identity: 0 for identity in change},
                    seed=bootstrap_seed,
                    samples=bootstrap_samples,
                )
            )
            robustness[label] = result
            changes[label] = change

    differential = {}
    for asl_label in path_groups["original_asl"]:
        for direct_label in path_groups["original_direct"]:
            key = f"{asl_label}__vs__{direct_label}"
            estimate = robustness[asl_label]["large_minus_original"] - robustness[
                direct_label
            ]["large_minus_original"]
            differential[key] = {
                "estimate": estimate,
                "bootstrap_identity_95": _bootstrap_difference(
                    changes[asl_label],
                    changes[direct_label],
                    seed=bootstrap_seed,
                    samples=bootstrap_samples,
                ),
            }

    large_by_id = {str(row["example_id"]): row for row in large_eval}
    subgroup_definitions = {
        "difficulty": {
            stratum: {
                identity
                for identity, row in large_by_id.items()
                if str(row["difficulty_stratum"]) == stratum
            }
            for stratum in sorted(
                {str(row["difficulty_stratum"]) for row in large_by_id.values()}
            )
        },
        "transformed_answer_magnitude": {
            band: {
                identity
                for identity, row in large_by_id.items()
                if _magnitude_band(row["reference_return"]) == band
            }
            for band in sorted(
                {_magnitude_band(row["reference_return"]) for row in large_by_id.values()}
            )
        },
    }
    subgroups = {
        dimension: _subgroup_robustness(
            groups=groups,
            predictions=predictions,
            direct_labels=set(path_groups["original_direct"]),
            asl_labels=set(path_groups["original_asl"]),
            large_to_parent=large_to_parent,
        )
        for dimension, groups in subgroup_definitions.items()
    }

    asl_original_rates = [
        sum(vector.values()) / len(vector)
        for vector in original_vectors["original_asl"].values()
    ]
    report = {
        "schema_version": "ccpu.paper1.gsm8k_contribution.v1",
        "identity_counts": {"original": len(original_ids), "large_number": len(large_ids)},
        "original_rates": {
            **{
                label: sum(vector.values()) / len(vector)
                for label, vector in original_vectors["original_direct"].items()
            },
            **{
                label: sum(vector.values()) / len(vector)
                for label, vector in original_vectors["original_asl"].items()
            },
        },
        "asl_seed_original_summary": {
            "mean": mean(asl_original_rates),
            "minimum": min(asl_original_rates),
            "maximum": max(asl_original_rates),
            "seed_count": len(asl_original_rates),
        },
        "original_asl_vs_direct": original_pairwise,
        "large_number_robustness": robustness,
        "large_number_differential_degradation": differential,
        "large_number_subgroups": subgroups,
        "resource_use": {
            scope: {
                label: _resources(rows)
                for family in ("direct", "asl")
                for label, rows in predictions[f"{scope}_{family}"].items()
            }
            for scope in ("original", "large")
        },
        "bootstrap": {
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "resampling_unit": "large-number parent identity",
        },
        "sources": {
            "original_eval": {
                "path": str(original_eval_path),
                "sha256": file_sha256(original_eval_path),
            },
            "large_eval": {
                "path": str(large_eval_path),
                "sha256": file_sha256(large_eval_path),
            },
            **{
                group: {
                    label: {"path": str(path), "sha256": file_sha256(path)}
                    for label, path in paths.items()
                }
                for group, paths in path_groups.items()
            },
        },
        "statistical_boundary": (
            "ASL training seeds are reported separately and are not pooled as independent "
            "question observations; large-number intervals resample paired parent identities."
        ),
    }
    write_json(output_path, report)
    return report
