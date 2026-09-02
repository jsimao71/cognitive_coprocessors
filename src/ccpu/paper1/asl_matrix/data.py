"""Leakage-audited F0 views for the ASL grounding architecture matrix."""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    file_sha256,
    fingerprint,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.dsl import validate_asl
from ccpu.paper1.asl_pilot_eval import score_asl

REGIMES = ("full", "partial", "autonomous")
CORRUPTION_POLICIES = (
    "record_dropout",
    "entity_mask",
    "relation_mask",
    "argument_mask",
    "value_mask",
    "dependency_mask",
    "attribute_mask",
    "span_mask",
    "random_token_mask",
    "record_permutation",
    "noise_record_insertion",
)

_PATH = re.compile(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+\b", re.IGNORECASE)
_VALUE = re.compile(r'(?<![A-Za-z0-9_])(?:-?\d+(?:\.\d+)?|"[^"]*")')
_TOKEN = re.compile(r'"[^"]*"|[A-Za-z_][A-Za-z0-9_.]*|-?\d+(?:\.\d+)?|\S')


@dataclass(frozen=True)
class MatrixExample:
    """One identity-preserving F0 example used to create runtime views."""

    example_id: str
    dataset: str
    parent_source_id: str
    semantic_pattern_id: str
    nl_input: str
    target_asl: str
    effective_scope: dict[str, Any]
    split: str


@dataclass(frozen=True)
class StaticMixture:
    """Static T1/T2/T3 proportions."""

    full_teacher: float = 0.25
    partial_teacher: float = 0.35
    autonomous: float = 0.40

    def __post_init__(self) -> None:
        values = (self.full_teacher, self.partial_teacher, self.autonomous)
        if any(value < 0 for value in values):
            raise ValueError("mixture fractions must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("mixture fractions must sum to one")

    def weights(self) -> tuple[float, float, float]:
        return (self.full_teacher, self.partial_teacher, self.autonomous)


def _scope(dataset: str, source_id: str) -> dict[str, Any]:
    return {
        "id": f"{dataset}:matrix:{source_id}",
        "parent": None,
        "kind": "benchmark_case",
        "source": "dataset",
    }


def _problem_from_prompt(prompt: str) -> str:
    marker = "Problem:"
    if marker not in prompt:
        raise ValueError("F0 prompt is missing Problem marker")
    value = prompt.split(marker, 1)[1]
    return value.rsplit("\nASL:", 1)[0].strip()


def _sft_example(row: dict[str, Any], split: str) -> MatrixExample:
    dataset = str(row["dataset"])
    source_id = str(row["parent_source_id"])
    return MatrixExample(
        example_id=f"matrix-{split}-{fingerprint((dataset, source_id), 16)}",
        dataset=dataset,
        parent_source_id=source_id,
        semantic_pattern_id=str(row["semantic_pattern_id"]),
        nl_input=_problem_from_prompt(str(row["prompt"])),
        target_asl=str(row["target"]).strip(),
        effective_scope=_scope(dataset, source_id),
        split=split,
    )


def _test_example(row: dict[str, Any]) -> MatrixExample:
    dataset = str(row["dataset"])
    source_id = str(row["source_id"])
    return MatrixExample(
        example_id=f"matrix-test-{fingerprint((dataset, source_id), 16)}",
        dataset=dataset,
        parent_source_id=source_id,
        semantic_pattern_id=str(row["semantic_pattern_id"]),
        nl_input=str(row["question"]).strip(),
        target_asl=str(row["asl"]).strip(),
        effective_scope=dict(row["effective_scope"]),
        split="test",
    )


def canonicalize_asl(asl: str, scope: dict[str, Any]) -> dict[str, Any]:
    """Return stable text and CCIR forms for an executable ASL program."""

    validation = validate_asl(asl, effective_scope=scope)
    if not validation["execution_verified"]:
        raise ValueError("matrix reference ASL must parse, lower, type-check, and execute")
    lines = [" ".join(line.strip().rstrip(";").split()) for line in asl.splitlines() if line.strip()]
    operations = [item["operation"] for item in validation["ccir"]["operations"]]
    return {
        "text": "\n".join(lines),
        "ccir": canonical_json(operations),
        "record_count": len(operations),
    }


def _select(values: Iterable[str], severity: float, rng: random.Random) -> set[str]:
    unique = sorted(set(values))
    if not unique:
        return set()
    count = max(1, min(len(unique), round(len(unique) * severity)))
    return set(rng.sample(unique, count))


def _replace_paths(text: str, selected: set[str], mode: str) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(0)
        if path not in selected:
            return path
        parts = path.split(".")
        if mode == "entity":
            parts[0] = "masked_entity"
        elif mode == "relation":
            parts[-1] = "masked_relation"
        elif mode == "attribute":
            index = 1 if len(parts) > 2 else len(parts) - 1
            parts[index] = "masked_attribute"
        return ".".join(parts)

    return _PATH.sub(replace, text)


def corrupt_asl(
    asl: str,
    *,
    policy: str,
    severity: float,
    seed: int,
    noise_asl: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create one deterministic partial-teacher view and auditable metadata."""

    if policy not in CORRUPTION_POLICIES:
        raise ValueError(f"unsupported ASL corruption policy: {policy}")
    if not 0 < severity <= 1:
        raise ValueError("corruption severity must be in (0, 1]")
    rng = random.Random(seed)
    lines = [line.strip() for line in asl.splitlines() if line.strip()]
    before_tokens = _TOKEN.findall(asl)
    selected: list[str] = []

    if policy == "record_dropout":
        indexes = _select((str(index) for index in range(len(lines))), severity, rng)
        selected = sorted(indexes)
        output = "\n".join(line for index, line in enumerate(lines) if str(index) not in indexes)
    elif policy in {"entity_mask", "relation_mask", "attribute_mask"}:
        paths = _PATH.findall(asl)
        chosen = _select(paths, severity, rng)
        selected = sorted(chosen)
        output = _replace_paths(asl, chosen, policy.split("_", 1)[0])
    elif policy == "value_mask":
        values = _VALUE.findall(asl)
        chosen = _select(values, severity, rng)
        selected = sorted(chosen)
        output = _VALUE.sub(lambda match: "masked_value" if match.group(0) in chosen else match.group(0), asl)
    elif policy in {"argument_mask", "dependency_mask"}:
        output_lines = []
        eligible = [str(index) for index, line in enumerate(lines) if "=" in line]
        chosen_indexes = _select(eligible, severity, rng)
        selected = sorted(chosen_indexes)
        for index, line in enumerate(lines):
            if str(index) not in chosen_indexes:
                output_lines.append(line)
                continue
            left, right = line.split("=", 1)
            if policy == "argument_mask":
                right = " masked_argument"
            else:
                refs = set(_PATH.findall(right))
                right = _PATH.sub(
                    lambda match, references=refs: "masked_dependency"
                    if match.group(0) in references
                    else match.group(0),
                    right,
                )
            output_lines.append(f"{left.strip()} = {right.strip()}")
        output = "\n".join(output_lines)
    elif policy == "span_mask":
        count = max(1, round(len(lines) * severity))
        start = rng.randrange(max(1, len(lines) - count + 1))
        selected = [f"lines:{start}:{start + count}"]
        output = "\n".join([*lines[:start], "masked_span", *lines[start + count :]])
    elif policy == "random_token_mask":
        tokens = _TOKEN.findall(asl)
        indexes = _select((str(index) for index in range(len(tokens))), severity, rng)
        selected = sorted(indexes)
        output = " ".join(
            "masked_token" if str(index) in indexes else token
            for index, token in enumerate(tokens)
        )
    elif policy == "record_permutation":
        output_lines = list(lines)
        rng.shuffle(output_lines)
        if output_lines == lines and len(output_lines) > 1:
            output_lines = output_lines[1:] + output_lines[:1]
        selected = [str(index) for index in range(len(lines))]
        output = "\n".join(output_lines)
    else:
        noise_lines = [line.strip() for line in (noise_asl or "").splitlines() if line.strip()]
        noise_candidates = [line for line in noise_lines if not line.upper().startswith("RETURN ")]
        injected = rng.choice(noise_candidates) if noise_candidates else "noise.unrelated = 0"
        index = rng.randrange(len(lines) + 1)
        selected = [injected]
        output_lines = list(lines)
        output_lines.insert(index, injected)
        output = "\n".join(output_lines)

    after_tokens = _TOKEN.findall(output)
    retained = min(1.0, len(after_tokens) / len(before_tokens)) if before_tokens else 0.0
    return output, {
        "policy": policy,
        "severity": severity,
        "selected": selected,
        "input_token_count": len(before_tokens),
        "output_token_count": len(after_tokens),
        "external_asl_fraction": retained,
        "seed": seed,
    }


class RegimeBuilder:
    """Construct T1/T2/T3 views from one example without copying datasets."""

    def __init__(
        self,
        *,
        mixture: StaticMixture | None = None,
        corruption_policies: tuple[str, ...] = CORRUPTION_POLICIES,
        corruption_severity: float = 0.5,
        seed: int = 912736,
    ) -> None:
        self.mixture = mixture or StaticMixture()
        self.corruption_policies = corruption_policies
        self.corruption_severity = corruption_severity
        self.seed = seed
        if not corruption_policies or any(p not in CORRUPTION_POLICIES for p in corruption_policies):
            raise ValueError("corruption_policies must contain supported policies")

    def sample_regime(self, example: MatrixExample, *, epoch: int) -> str:
        rng = random.Random(f"{self.seed}:{epoch}:{example.example_id}:regime")
        return rng.choices(REGIMES, weights=self.mixture.weights(), k=1)[0]

    def make_view(
        self,
        example: MatrixExample,
        *,
        regime: str,
        epoch: int = 0,
        noise_asl: str | None = None,
        corruption_policy: str | None = None,
        corruption_severity: float | None = None,
    ) -> dict[str, Any]:
        if regime not in REGIMES:
            raise ValueError(f"unsupported training regime: {regime}")
        external = None
        metadata = None
        if regime == "full":
            external = example.target_asl
        elif regime == "partial":
            rng = random.Random(f"{self.seed}:{epoch}:{example.example_id}:corruption")
            policy = corruption_policy or rng.choice(self.corruption_policies)
            corruption_seed = rng.randrange(2**31)
            external, metadata = corrupt_asl(
                example.target_asl,
                policy=policy,
                severity=corruption_severity or self.corruption_severity,
                seed=corruption_seed,
                noise_asl=noise_asl,
            )
        visible = ["nl_input"] if regime == "autonomous" else ["nl_input", "external_asl_input"]
        view = {
            "schema_version": "ccpu.paper1.asl_matrix.view.v1",
            "example_id": example.example_id,
            "dataset": example.dataset,
            "parent_source_id": example.parent_source_id,
            "semantic_pattern_id": example.semantic_pattern_id,
            "split": example.split,
            "nl_input": example.nl_input,
            "external_asl_input": external,
            "target_asl": example.target_asl,
            "effective_scope": example.effective_scope,
            "regime": regime,
            "has_external_asl": external is not None,
            "external_asl_fraction": (
                0.0
                if external is None
                else metadata["external_asl_fraction"]
                if metadata
                else 1.0
            ),
            "external_asl_corruption": metadata,
            "source_fields_visible_to_model": visible,
            "epoch": epoch,
        }
        assert_view_is_leakage_safe(view)
        return view


def assert_view_is_leakage_safe(view: dict[str, Any]) -> None:
    """Enforce the model-visible boundary for one generated view."""

    visible = set(view["source_fields_visible_to_model"])
    if "target_asl" in visible:
        raise AssertionError("target ASL cannot be a model-visible source field")
    if view["regime"] == "autonomous":
        if view["external_asl_input"] is not None or view["has_external_asl"]:
            raise AssertionError("T3 autonomous views cannot contain external ASL")
        if visible != {"nl_input"}:
            raise AssertionError("T3 autonomous views may expose only natural language")
    elif view["external_asl_input"] is None or not view["has_external_asl"]:
        raise AssertionError("teacher-assisted views require external ASL")


def _load_examples(train_path: Path, dev_path: Path, test_path: Path) -> dict[str, list[MatrixExample]]:
    return {
        "train": [_sft_example(row, "train") for row in read_jsonl(train_path)],
        "dev": [_sft_example(row, "dev") for row in read_jsonl(dev_path)],
        "test": [_test_example(row) for row in read_jsonl(test_path)],
    }


def _identity_audit(splits: dict[str, list[MatrixExample]]) -> dict[str, Any]:
    ids = {
        name: {(row.dataset, row.parent_source_id) for row in rows}
        for name, rows in splits.items()
    }
    patterns = {
        name: {row.semantic_pattern_id for row in rows}
        for name, rows in splits.items()
    }
    overlaps = {
        "train_dev_parent_overlap": sorted(ids["train"] & ids["dev"]),
        "train_test_parent_overlap": sorted(ids["train"] & ids["test"]),
        "dev_test_parent_overlap": sorted(ids["dev"] & ids["test"]),
        "train_test_pattern_overlap": sorted(patterns["train"] & patterns["test"]),
        "dev_test_pattern_overlap": sorted(patterns["dev"] & patterns["test"]),
    }
    counts_valid = {name: len(rows) for name, rows in splits.items()} == {
        "train": 450,
        "dev": 25,
        "test": 25,
    }
    passed = counts_valid and not any(overlaps.values())
    return {
        "schema_version": "ccpu.paper1.asl_matrix.leakage_audit.v1",
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_unique_patterns": {
            name: len(patterns[name]) for name in ("train", "dev", "test")
        },
        **overlaps,
        "expected_counts_valid": counts_valid,
        "passed": passed,
    }


def build_matrix_data(
    train_path: str | Path,
    dev_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 912736,
) -> dict[str, Any]:
    """Freeze matched F0 identities and deterministic P0 audit views."""

    paths = tuple(Path(path) for path in (train_path, dev_path, test_path))
    splits = _load_examples(*paths)
    audit = _identity_audit(splits)
    if not audit["passed"]:
        raise AssertionError(f"ASL matrix identity audit failed: {audit}")
    for rows in splits.values():
        for row in rows:
            canonicalize_asl(row.target_asl, row.effective_scope)

    output = Path(output_dir)
    source_files = {}
    for name, rows in splits.items():
        path = write_jsonl(output / "source" / f"{name}.jsonl", (asdict(row) for row in rows))
        source_files[name] = {"rows": len(rows), "sha256": file_sha256(path)}

    builder = RegimeBuilder(seed=seed)
    train_views = []
    regime_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    for index, example in enumerate(splits["train"]):
        regime = builder.sample_regime(example, epoch=0)
        noise = splits["train"][(index + 1) % len(splits["train"])].target_asl
        view = builder.make_view(example, regime=regime, noise_asl=noise)
        train_views.append(view)
        regime_counts[regime] += 1
        if view["external_asl_corruption"]:
            policy_counts[view["external_asl_corruption"]["policy"]] += 1
    views_path = write_jsonl(output / "audit" / "train_epoch0_views.jsonl", train_views)

    corruption_smoke = []
    smoke_example = splits["train"][0]
    noise = splits["train"][1].target_asl
    for index, policy in enumerate(CORRUPTION_POLICIES):
        view = builder.make_view(
            smoke_example,
            regime="partial",
            corruption_policy=policy,
            corruption_severity=0.5,
            noise_asl=noise,
            epoch=index,
        )
        corruption_smoke.append(
            {
                "policy": policy,
                "changed": view["external_asl_input"] != smoke_example.target_asl,
                "metadata": view["external_asl_corruption"],
            }
        )
    if not all(row["changed"] and row["metadata"] for row in corruption_smoke):
        raise AssertionError("every corruption operator must change ASL and expose metadata")

    semantic_self_scores = [
        score_asl(row.target_asl, row.target_asl, row.effective_scope)
        for row in splits["test"]
    ]
    evaluator_gate = {
        "test_rows": len(semantic_self_scores),
        "all_parse_valid": all(row["parse_valid"] for row in semantic_self_scores),
        "all_executable": all(row["executable"] for row in semantic_self_scores),
        "all_semantic_return_equivalent": all(
            row["semantic_return_equivalent"] for row in semantic_self_scores
        ),
        "all_final_answer_correct": all(row["final_answer_correct"] for row in semantic_self_scores),
    }
    evaluator_gate["passed"] = all(
        value for key, value in evaluator_gate.items() if key.startswith("all_")
    )
    audit_path = write_json(output / "leakage_audit.json", audit)
    smoke_path = write_json(output / "corruption_smoke.json", corruption_smoke)
    evaluator_path = write_json(output / "evaluator_gate.json", evaluator_gate)
    manifest = {
        "schema_version": "ccpu.paper1.asl_matrix.data.v1",
        "seed": seed,
        "input_sha256": {
            "train": file_sha256(paths[0]),
            "dev": file_sha256(paths[1]),
            "test": file_sha256(paths[2]),
        },
        "source_files": source_files,
        "epoch0_regime_counts": dict(sorted(regime_counts.items())),
        "epoch0_corruption_policy_counts": dict(sorted(policy_counts.items())),
        "corruption_policies": list(CORRUPTION_POLICIES),
        "runtime_view_generation": True,
        "fixed_test_denominator": 25,
        "p0_gate_passed": audit["passed"] and evaluator_gate["passed"],
        "output_sha256": {
            "leakage_audit": file_sha256(audit_path),
            "corruption_smoke": file_sha256(smoke_path),
            "evaluator_gate": file_sha256(evaluator_path),
            "train_epoch0_views": file_sha256(views_path),
        },
    }
    write_json(output / "data_manifest.json", manifest)
    return manifest
