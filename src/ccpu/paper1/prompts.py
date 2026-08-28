"""Matched condition prompts for Paper 1."""

from __future__ import annotations

from .dataset import ArithmeticExample

PROMPT_VERSION = "paper1_hard_interfaces_v2_concise"

CONDITIONS = (
    "llm_only",
    "matched_prompt",
    "explicit_tool",
    "reflex",
    "normalized_reflex",
    "calculator_block",
    "oracle",
)


def condition_prompt(example: ArithmeticExample, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if example.task_kind == "control":
        return f"{example.prompt}\nSentence: {example.reference_completion}\nResponse:"
    if condition == "llm_only":
        instruction = "Return only the exact answer as an integer or reduced fraction. No explanation."
    elif condition == "matched_prompt":
        instruction = (
            "Work carefully and check each operation internally. Return only the exact answer "
            "as an integer or reduced fraction. No explanation."
        )
    elif condition == "explicit_tool":
        instruction = (
            "Call the calculator by writing <tool:calculator>EXPRESSION</tool>. "
            "After the tool result appears, return only that exact answer."
        )
    elif condition in {"reflex", "oracle"}:
        instruction = (
            "Begin the calculation by copying the expression followed by one equals sign. "
            "Continue from any result inserted after that sign, then give only the exact answer."
        )
    elif condition == "normalized_reflex":
        instruction = (
            "Begin by copying the complete expression in ordinary, LaTeX, or Unicode math "
            "notation followed by one equals sign. Continue from the inserted result, then give "
            "only the exact answer."
        )
    else:
        instruction = (
            "Write exactly one fenced calculator block containing the complete expression, "
            "then continue from the inserted calculator result and give only the exact answer. "
            "Use this form:\n"
            "```calculator\nEXPRESSION\n```"
        )
    return f"{instruction}\n{example.prompt}\nResponse:"
