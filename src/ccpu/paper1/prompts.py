"""Matched condition prompts for Paper 1."""

from __future__ import annotations

from .dataset import ArithmeticExample

CONDITIONS = ("llm_only", "matched_prompt", "explicit_tool", "reflex", "oracle")


def condition_prompt(example: ArithmeticExample, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if example.task_kind == "control":
        return f"{example.prompt}\nSentence: {example.reference_completion}\nResponse:"
    if condition == "llm_only":
        instruction = "Return the exact answer."
    elif condition == "matched_prompt":
        instruction = "Work carefully, check each operation, and return the exact answer."
    elif condition == "explicit_tool":
        instruction = (
            "Call the calculator by writing <tool:calculator>EXPRESSION</tool>. "
            "After the tool result appears, return the exact answer."
        )
    else:
        instruction = (
            "Begin the calculation by copying the expression followed by one equals sign. "
            "Continue from any result inserted after that sign."
        )
    return f"{instruction}\n{example.prompt}\nResponse:"
