"""Protocol-only LoRA data and evaluation for one-source retrieval policy."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import read_jsonl, write_json, write_jsonl
from ccpu.paper1.generation import HuggingFaceBackend

from .dataset import RetrievalExample

_BLOCK = re.compile(r"```retrieve\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class PolicyDataConfig:
    seed: int = 15191
    train_required: int = 80
    train_controls: int = 80
    dev_required: int = 20
    dev_controls: int = 20

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyDataConfig:
        value = raw.get("policy_data", raw)
        return cls(**{key: int(value.get(key, default)) for key, default in asdict(cls()).items()})


def _request_target(entity: str, attribute: str, as_of: str) -> str:
    return (
        "```retrieve\n"
        f"entity={entity}\nattribute={attribute}\nas_of={as_of}\nsource=atlas\n"
        "```"
    )


def generate_policy_data(
    config: PolicyDataConfig,
    *,
    excluded_examples: list[RetrievalExample],
    output_dir: str | Path,
) -> dict[str, Any]:
    rng = random.Random(config.seed)
    output_dir = Path(output_dir)
    attributes = ("assigned code", "current owner", "release channel", "source value")

    def rows(split: str, required_count: int, control_count: int) -> list[dict[str, Any]]:
        result = []
        offset = 0 if split == "train" else 10000
        for index in range(required_count):
            entity = f"Policy {split.title()} Entity {offset + index:05d}"
            attribute = attributes[index % len(attributes)]
            as_of = "2026-08-28"
            value = f"POLICYVAL-{rng.randrange(10**9):09d}"
            result.append(
                {
                    "example_id": f"p15-policy-{split}-required-{index:04d}",
                    "split": split,
                    "kind": "retrieval_required",
                    "entity": entity,
                    "attribute": attribute,
                    "answer_value": value,
                    "prompt": (
                        f"Use the controlled Atlas source as of {as_of} to find the latest "
                        f"{attribute} for {entity}."
                    ),
                    "target": _request_target(entity, attribute, as_of),
                }
            )
        for index in range(control_count):
            supplied = f"CONTEXTVAL-{rng.randrange(10**9):09d}"
            result.append(
                {
                    "example_id": f"p15-policy-{split}-control-{index:04d}",
                    "split": split,
                    "kind": "retrieval_not_required",
                    "entity": f"Context {split.title()} {offset + index:05d}",
                    "attribute": "supplied value",
                    "answer_value": supplied,
                    "prompt": f"The active context already gives {supplied}. Repeat it without retrieval.",
                    "target": "NO_RETRIEVAL",
                }
            )
        random.Random(config.seed + offset).shuffle(result)
        return result

    train = rows("train", config.train_required, config.train_controls)
    dev = rows("dev", config.dev_required, config.dev_controls)
    excluded_entities = {example.entity.casefold() for example in excluded_examples}
    excluded_values = {example.answer.casefold() for example in excluded_examples}
    generated_entities = {row["entity"].casefold() for row in [*train, *dev]}
    generated_values = {row["answer_value"].casefold() for row in [*train, *dev]}
    train_entities = {row["entity"].casefold() for row in train}
    dev_entities = {row["entity"].casefold() for row in dev}
    train_values = {row["answer_value"].casefold() for row in train}
    dev_values = {row["answer_value"].casefold() for row in dev}
    audit = {
        "schema_version": "ccpu.paper1_5.policy_leakage_audit.v1",
        "train_dev_entity_overlap": sorted(train_entities & dev_entities),
        "train_dev_value_overlap": sorted(train_values & dev_values),
        "excluded_entity_overlap": sorted(generated_entities & excluded_entities),
        "excluded_answer_value_overlap": sorted(generated_values & excluded_values),
        "targets_contain_answer_values": any(
            str(row["answer_value"]) in str(row["target"]) for row in [*train, *dev]
        ),
    }
    if any(audit[key] for key in audit if key != "schema_version"):
        raise ValueError(f"policy data leakage audit failed: {audit}")
    train_path = write_jsonl(output_dir / "train.jsonl", train)
    dev_path = write_jsonl(output_dir / "dev.jsonl", dev)
    write_json(output_dir / "leakage_audit.json", audit)
    result = {
        "schema_version": "ccpu.paper1_5.policy_data.v1",
        "config": asdict(config),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "leakage_audit": audit,
    }
    write_json(output_dir / "manifest.json", result)
    return result


def parse_retrieval_block(text: str) -> dict[str, str] | None:
    matches = _BLOCK.findall(text)
    if len(matches) != 1:
        return None
    fields: dict[str, str] = {}
    for line in matches[0].splitlines():
        if "=" not in line:
            return None
        key, value = line.split("=", 1)
        fields[key.strip().casefold()] = value.strip()
    required = {"entity", "attribute", "as_of", "source"}
    return fields if fields.keys() >= required else None


def policy_prompt(example: RetrievalExample, mode: str) -> str:
    contract = (
        "Emit exactly one fenced retrieve block only when a reliable answer requires the "
        "controlled Atlas source. Otherwise output NO_RETRIEVAL. Do not answer the question."
    )
    if mode == "context":
        demonstrations = (
            "\n\nExample request: Atlas currently assigns a code to Demo Station.\nOutput:\n"
            "```retrieve\nentity=Demo Station\nattribute=assigned code\n"
            "as_of=2026-08-28\nsource=atlas\n```"
            "\n\nExample request: The context explicitly says code DEMO-7.\nOutput:\nNO_RETRIEVAL"
        )
    elif mode == "minimal":
        demonstrations = ""
    else:
        raise ValueError(f"unknown policy prompt mode: {mode}")
    return f"{contract}{demonstrations}\n\nRequest: {example.question}\nOutput:"


def run_policy_hf(
    examples: list[RetrievalExample],
    backend: HuggingFaceBackend,
    *,
    mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        if example.split != "test":
            continue
        generation = backend.generate(policy_prompt(example, mode), seed=seed)
        request = parse_retrieval_block(generation.generated_text)
        selected = request is not None
        request_valid = bool(
            request
            and request["entity"].casefold() == example.entity.casefold()
            and request["attribute"].casefold() == example.attribute.casefold()
            and request["as_of"] == example.as_of
            and request["source"].casefold() == "atlas"
        )
        rows.append(
            {
                "schema_version": "ccpu.paper1_5.policy_prediction.v1",
                "example_id": example.example_id,
                "model_id": backend.model_id,
                "mode": mode,
                "evidence_required": example.evidence_required,
                "selected": selected,
                "request_valid": request_valid,
                "selection_correct": selected == example.evidence_required,
                "interface_success": (
                    request_valid if example.evidence_required else not selected
                ),
                "false_activation": selected and not example.evidence_required,
                "missed_retrieval": not selected and example.evidence_required,
                "generated_text": generation.generated_text,
                "prompt_tokens": generation.prompt_tokens,
                "generated_tokens": generation.generated_tokens,
                "wall_time_ns": generation.wall_time_ns,
                "backend_metadata": generation.metadata,
            }
        )
    return rows


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy evaluation requires predictions")
    count = len(rows)
    return {
        "schema_version": "ccpu.paper1_5.policy_evaluation.v1",
        "model_id": rows[0]["model_id"],
        "mode": rows[0]["mode"],
        "count": count,
        "selection_accuracy": sum(row["selection_correct"] for row in rows) / count,
        "interface_success_rate": sum(row["interface_success"] for row in rows) / count,
        "false_activation_rate": sum(row["false_activation"] for row in rows)
        / sum(not row["evidence_required"] for row in rows),
        "retrieval_recall": 1
        - sum(row["missed_retrieval"] for row in rows)
        / sum(row["evidence_required"] for row in rows),
        "mean_prompt_tokens": sum(row["prompt_tokens"] for row in rows) / count,
        "mean_generated_tokens": sum(row["generated_tokens"] for row in rows) / count,
        "mean_wall_time_ms": sum(row["wall_time_ns"] for row in rows) / count / 1e6,
    }


def read_policy_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
