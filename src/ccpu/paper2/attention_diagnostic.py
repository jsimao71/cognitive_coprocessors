"""Bounded attention and causal-content diagnostics for Paper 2 result use."""

from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    environment_manifest,
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.metrics import safe_mean


def select_attention_rows(
    dataset_path: str | Path,
    baseline_predictions_path: str | Path,
    *,
    per_task: int = 12,
) -> list[dict[str, Any]]:
    predictions = {row["example_id"]: row for row in read_jsonl(baseline_predictions_path)}
    rows = [
        row
        for row in read_jsonl(dataset_path)
        if row["format"] == "authority" and row["task"] in {"INTERPRET", "CONTINUE"}
    ]
    selected = []
    for task in ("INTERPRET", "CONTINUE"):
        members = [row for row in rows if row["task"] == task]
        correct = [row for row in members if predictions[row["example_id"]]["exact"]]
        incorrect = [row for row in members if not predictions[row["example_id"]]["exact"]]
        target_correct = min(len(correct), per_task // 2)
        chosen = correct[:target_correct] + incorrect[: per_task - target_correct]
        if len(chosen) != per_task:
            remainder = [row for row in members if row not in chosen]
            chosen.extend(remainder[: per_task - len(chosen)])
        for row in chosen:
            selected.append(
                {
                    **row,
                    "baseline_exact": predictions[row["example_id"]]["exact"],
                }
            )
    return selected


class AttentionDiagnosticBackend:
    def __init__(self, model: dict[str, Any], *, device: str, max_new_tokens: int) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("attention diagnostics require torch and transformers") from error
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model["model_id"], revision=model["revision"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model["model_id"],
            revision=model["revision"],
            dtype=getattr(torch, str(model.get("dtype", "float16"))),
            attn_implementation="eager",
        ).to(device)
        self.model.eval()
        self.eos = {
            value
            for candidate in (
                self.tokenizer.eos_token_id,
                getattr(self.model.generation_config, "eos_token_id", None),
            )
            for value in (candidate if isinstance(candidate, list) else [candidate])
            if value is not None
        }

    def _chat(self, prompt: str) -> str:
        arguments = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], **arguments
                )
            )
        except TypeError:
            arguments.pop("enable_thinking")
            return str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], **arguments
                )
            )

    def generate(self, prompt: str, *, collect_attention: bool) -> dict[str, Any]:
        torch = self.torch
        rendered = self._chat(prompt)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_length = int(input_ids.shape[-1])
        spans = _token_spans(rendered, prompt, offsets)
        generated_ids = []
        layer_question = None
        layer_result = None
        layer_head_question = None
        layer_head_result = None
        token_attention = []
        token_count = 0
        if self.device == "xpu":
            torch.xpu.reset_peak_memory_stats()
            torch.xpu.synchronize()
        started = time.perf_counter_ns()
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=collect_attention,
                    use_cache=False,
                )
                if collect_attention:
                    question_values = []
                    result_values = []
                    question_head_values = []
                    result_head_values = []
                    for attention in output.attentions:
                        last = attention[0, :, -1, :prompt_length].float()
                        question_heads = last[:, spans["question"]].sum(dim=-1)
                        result_heads = last[:, spans["result"]].sum(dim=-1)
                        question_values.append(float(question_heads.mean().cpu()))
                        result_values.append(float(result_heads.mean().cpu()))
                        question_head_values.append(question_heads.cpu().tolist())
                        result_head_values.append(result_heads.cpu().tolist())
                    if layer_question is None:
                        layer_question = [0.0] * len(question_values)
                        layer_result = [0.0] * len(result_values)
                        layer_head_question = [[0.0] * len(heads) for heads in question_head_values]
                        layer_head_result = [[0.0] * len(heads) for heads in result_head_values]
                    layer_question = [
                        total + value
                        for total, value in zip(layer_question, question_values, strict=True)
                    ]
                    layer_result = [
                        total + value
                        for total, value in zip(layer_result, result_values, strict=True)
                    ]
                    layer_head_question = [
                        [total + value for total, value in zip(totals, values, strict=True)]
                        for totals, values in zip(
                            layer_head_question, question_head_values, strict=True
                        )
                    ]
                    layer_head_result = [
                        [total + value for total, value in zip(totals, values, strict=True)]
                        for totals, values in zip(
                            layer_head_result, result_head_values, strict=True
                        )
                    ]
                    token_attention.append(
                        {
                            "generated_token_index": token_count,
                            "question_mass": safe_mean(question_values),
                            "result_mass": safe_mean(result_values),
                        }
                    )
                    token_count += 1
                token_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
                generated_ids.append(token_id)
                token = torch.tensor([[token_id]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat((input_ids, token), dim=-1)
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device),
                    ),
                    dim=-1,
                )
                if token_id in self.eos:
                    break
        if self.device == "xpu":
            torch.xpu.synchronize()
        wall_time_ns = time.perf_counter_ns() - started
        result = {
            "generated_text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "generated_tokens": len(generated_ids),
            "prompt_tokens": prompt_length,
            "wall_time_ns": wall_time_ns,
        }
        if collect_attention and token_count:
            result["attention"] = {
                "question_by_layer": [value / token_count for value in layer_question],
                "result_by_layer": [value / token_count for value in layer_result],
                "question_by_layer_head": [
                    [value / token_count for value in layer] for layer in layer_head_question
                ],
                "result_by_layer_head": [
                    [value / token_count for value in layer] for layer in layer_head_result
                ],
                "question_mass": safe_mean(value / token_count for value in layer_question),
                "result_mass": safe_mean(value / token_count for value in layer_result),
                "generated_token_count": token_count,
                "by_generated_token": token_attention,
            }
        return result


def _token_spans(rendered: str, prompt: str, offsets: list[list[int]]) -> dict[str, list[int]]:
    prompt_start = rendered.find(prompt)
    if prompt_start < 0:
        raise ValueError("rendered chat prompt does not contain source prompt")
    question_end = prompt.find("\n\n")
    result_text = _result_text(prompt)
    result_start = prompt.rfind(result_text)
    character_spans = {
        "question": (prompt_start, prompt_start + question_end),
        "result": (
            prompt_start + result_start,
            prompt_start + result_start + len(result_text),
        ),
    }
    token_spans = {}
    for name, (start, end) in character_spans.items():
        members = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        if not members:
            raise ValueError(f"no tokens found for {name} span")
        token_spans[name] = members
    return token_spans


def _result_text(prompt: str) -> str:
    marker = "AUTHORITATIVE EXACT RESULT ("
    start = prompt.find(marker)
    if start < 0:
        raise ValueError("attention subset requires authority format")
    value_start = prompt.find(": ", start) + 2
    return prompt[value_start : prompt.find("\n", value_start)]


def _variant_prompt(row: dict[str, Any], condition: str) -> str:
    prompt = str(row["prompt"])
    if condition == "full":
        return prompt
    if condition == "result_masked":
        result = str(row["result"])
        head, separator, tail = prompt.rpartition(result)
        if not separator:
            raise ValueError("result not found in authority prompt")
        return f"{head}[MASKED RESULT]{tail}"
    if condition == "distractor_removed":
        return prompt.replace(str(row["distractor"]), "[REMOVED DRAFT]", 1)
    raise ValueError(f"unknown attention condition: {condition}")


def run_attention_diagnostic(
    rows: list[dict[str, Any]], backend: AttentionDiagnosticBackend
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    for row in rows:
        for condition in ("full", "result_masked", "distractor_removed"):
            generated = backend.generate(
                _variant_prompt(row, condition), collect_attention=condition == "full"
            )
            text = generated["generated_text"].strip()
            predictions.append(
                {
                    "schema_version": "ccpu.paper2.attention_prediction.v1",
                    "condition": condition,
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "baseline_exact": row["baseline_exact"],
                    "expected": row["expected"],
                    "generated_text": generated["generated_text"],
                    "exact": text == str(row["expected"]),
                    "prompt_tokens": generated["prompt_tokens"],
                    "generated_tokens": generated["generated_tokens"],
                    "wall_time_ns": generated["wall_time_ns"],
                    "attention": generated.get("attention"),
                }
            )
    return predictions, summarize_attention_diagnostic(predictions)


def summarize_attention_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = []
    for condition in ("full", "result_masked", "distractor_removed"):
        members = [row for row in rows if row["condition"] == condition]
        by_condition.append(
            {
                "condition": condition,
                "count": len(members),
                "exact_rate": safe_mean(row["exact"] for row in members),
                "mean_wall_time_ms": safe_mean(row["wall_time_ns"] for row in members) / 1e6,
            }
        )
    full = [row for row in rows if row["condition"] == "full"]
    correct = [row for row in full if row["exact"]]
    incorrect = [row for row in full if not row["exact"]]
    attention = {
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "correct_question_mass": safe_mean(row["attention"]["question_mass"] for row in correct),
        "correct_result_mass": safe_mean(row["attention"]["result_mass"] for row in correct),
        "incorrect_question_mass": safe_mean(
            row["attention"]["question_mass"] for row in incorrect
        ),
        "incorrect_result_mass": safe_mean(row["attention"]["result_mass"] for row in incorrect),
    }
    rates = {row["condition"]: row["exact_rate"] for row in by_condition}
    result_dependence = rates["full"] - rates["result_masked"]
    distractor_effect = rates["distractor_removed"] - rates["full"]
    competition_supported = result_dependence >= 0.1 and distractor_effect >= 0.1
    return {
        "schema_version": "ccpu.paper2.attention_diagnostic_summary.v1",
        "count": len(rows),
        "by_condition": by_condition,
        "attention": attention,
        "causal_effects": {
            "result_mask_drop": result_dependence,
            "distractor_removal_gain": distractor_effect,
        },
        "decision": {
            "question_result_competition_supported": competition_supported,
            "fixed_bias_gate": "run_beta_sweep" if competition_supported else "defer_beta_sweep",
            "attention_is_causal_proof": False,
        },
    }


def run_and_write_attention_diagnostic(
    *,
    dataset_path: str | Path,
    baseline_predictions_path: str | Path,
    model: dict[str, Any],
    device: str,
    max_new_tokens: int,
    per_task: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    rows = select_attention_rows(dataset_path, baseline_predictions_path, per_task=per_task)
    backend = AttentionDiagnosticBackend(model, device=device, max_new_tokens=max_new_tokens)
    predictions, summary = run_attention_diagnostic(rows, backend)
    output_dir = Path(output_dir)
    subset_path = write_jsonl(output_dir / "subset.jsonl", rows)
    predictions_path = write_jsonl(output_dir / "predictions.jsonl", predictions)
    summary_path = write_json(output_dir / "summary.json", summary)
    plot_path = None
    try:
        plot_path = _plot_attention_diagnostic(
            summary, output_dir / "attention_causal_diagnostic.png"
        )
    except RuntimeError:
        pass
    manifest = {
        "schema_version": "ccpu.paper2.attention_diagnostic_manifest.v1",
        "model": model,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "dataset_sha256": file_sha256(dataset_path),
        "baseline_predictions_sha256": file_sha256(baseline_predictions_path),
        "subset_sha256": file_sha256(subset_path),
        "predictions_sha256": file_sha256(predictions_path),
        "summary_sha256": file_sha256(summary_path),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    if plot_path:
        manifest["plot_sha256"] = file_sha256(plot_path)
    write_json(
        output_dir / "manifest.json",
        manifest,
    )
    return summary


def plot_attention_diagnostic(summary_path: str | Path, output: str | Path) -> Path:
    output = _plot_attention_diagnostic(read_json(summary_path), Path(output))
    manifest_path = Path(summary_path).parent / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest["plot_sha256"] = file_sha256(output)
        write_json(manifest_path, manifest)
    return output


def _plot_attention_diagnostic(summary: dict[str, Any], output: Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Paper 2 attention plots require matplotlib") from error
    conditions = summary["by_condition"]
    attention = summary["attention"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].bar(
        range(len(conditions)),
        [row["exact_rate"] for row in conditions],
        color=("#176b87", "#b33f40", "#d99b2b"),
    )
    axes[0].set_xticks(
        range(len(conditions)),
        [row["condition"].replace("_", " ") for row in conditions],
        rotation=18,
    )
    axes[0].set(ylabel="strict exact rate", ylim=(0, 1.0))
    axes[0].grid(axis="y", alpha=0.2)
    x = [0, 1]
    width = 0.34
    axes[1].bar(
        [value - width / 2 for value in x],
        [attention["correct_question_mass"], attention["incorrect_question_mass"]],
        width,
        label="question span",
        color="#176b87",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        [attention["correct_result_mass"], attention["incorrect_result_mass"]],
        width,
        label="result span",
        color="#d99b2b",
    )
    axes[1].set_xticks(x, ["correct", "incorrect"])
    axes[1].set_ylabel("mean generated-token attention mass")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output
