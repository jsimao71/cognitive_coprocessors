"""Deterministic matched-seed analysis for ASL evaluation records."""

from __future__ import annotations

from collections.abc import Iterable
from math import comb
from pathlib import Path
from statistics import mean
from typing import Any

from ccpu.common.artifacts import file_sha256, read_jsonl, write_json

ENDPOINTS = (
    "parse_valid",
    "lowerable_to_ccir",
    "type_valid",
    "executable",
    "semantic_state_equivalent",
    "semantic_return_equivalent",
    "alpha_state_equivalent",
    "alpha_return_equivalent",
    "final_answer_correct",
)


def _index_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed = {str(row["example_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate example_id in {path}")
    return indexed


def _exact_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _endpoint(row: dict[str, Any], name: str) -> bool:
    return bool(row["metrics"].get(name, False))


def _alpha_state_f1(row: dict[str, Any]) -> float:
    return float(row["metrics"].get("alpha_state_metrics", {}).get("f1", 0.0))


def analyze_asl_replications(
    *,
    baseline_path: str | Path,
    candidate_paths: Iterable[tuple[str, str | Path]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare matched candidate seeds without pooling repeated test identities."""
    baseline = _index_rows(baseline_path)
    candidates = [(label, Path(path), _index_rows(path)) for label, path in candidate_paths]
    if not candidates:
        raise ValueError("at least one candidate is required")

    identities = set(baseline)
    for label, _, rows in candidates:
        if set(rows) != identities:
            raise ValueError(f"candidate {label!r} does not match baseline identities")

    per_seed: dict[str, Any] = {}
    for label, path, rows in candidates:
        endpoints: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            baseline_values = [_endpoint(baseline[key], endpoint) for key in sorted(identities)]
            candidate_values = [_endpoint(rows[key], endpoint) for key in sorted(identities)]
            gains = sum(not left and right for left, right in zip(baseline_values, candidate_values))
            losses = sum(left and not right for left, right in zip(baseline_values, candidate_values))
            baseline_count = sum(baseline_values)
            candidate_count = sum(candidate_values)
            endpoints[endpoint] = {
                "baseline_count": baseline_count,
                "candidate_count": candidate_count,
                "delta_count": candidate_count - baseline_count,
                "gains": gains,
                "losses": losses,
                "ties": len(identities) - gains - losses,
                "two_sided_exact_mcnemar_p": _exact_p(gains, losses),
            }
        per_seed[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "endpoints": endpoints,
            "mean_alpha_state_f1": mean(_alpha_state_f1(rows[key]) for key in identities),
        }

    aggregate: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        baseline_count = sum(_endpoint(row, endpoint) for row in baseline.values())
        counts = [
            per_seed[label]["endpoints"][endpoint]["candidate_count"]
            for label, _, _ in candidates
        ]
        aggregate[endpoint] = {
            "baseline_count": baseline_count,
            "candidate_counts": counts,
            "candidate_mean_count": mean(counts),
            "candidate_min_count": min(counts),
            "candidate_max_count": max(counts),
            "all_seeds_above_baseline": all(value > baseline_count for value in counts),
            "all_seeds_at_or_above_baseline": all(value >= baseline_count for value in counts),
        }

    alpha_f1_values = [per_seed[label]["mean_alpha_state_f1"] for label, _, _ in candidates]
    report = {
        "schema_version": "ccpu.paper1.asl_replication_analysis.v1",
        "identity_count": len(identities),
        "seed_count": len(candidates),
        "baseline": {
            "path": str(baseline_path),
            "sha256": file_sha256(baseline_path),
            "mean_alpha_state_f1": mean(_alpha_state_f1(row) for row in baseline.values()),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "alpha_state_f1": {
            "candidate_values": alpha_f1_values,
            "candidate_mean": mean(alpha_f1_values),
            "candidate_min": min(alpha_f1_values),
            "candidate_max": max(alpha_f1_values),
        },
        "statistical_boundary": (
            "Seeds reuse the same frozen identities; seed counts and ranges are reported "
            "without pooling them as independent test observations."
        ),
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
