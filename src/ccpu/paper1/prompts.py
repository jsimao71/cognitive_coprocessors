"""Matched condition prompts for Paper 1."""

from __future__ import annotations

from .dataset import ArithmeticExample

PROMPT_VERSION = "paper1_hard_interfaces_v2_concise"
ICL_PROMPT_VERSION = "paper1_calculator_block_icl_v1_developmental"
ICL_ORDER_CONTROL_PROMPT_VERSION = "paper1_calculator_block_icl_v2_order_control"

CORE_CONDITIONS = (
    "llm_only",
    "matched_prompt",
    "explicit_tool",
    "reflex",
    "normalized_reflex",
    "calculator_block",
    "oracle",
)
ICL_BLOCK_CONDITIONS = (
    "calculator_block_icl_a",
    "calculator_block_icl_b",
    "calculator_block_icl_c",
    "calculator_block_icl_d",
    "calculator_block_icl_e",
    "calculator_block_icl_f",
    "calculator_block_icl_g",
)
BLOCK_CONDITIONS = ("calculator_block", *ICL_BLOCK_CONDITIONS)
SEEDED_OPEN_BLOCK_CONDITIONS = ("calculator_block_icl_e",)
SEEDED_WRAPPER_BLOCK_CONDITIONS = ("calculator_block_icl_f",)
CONDITIONS = (*CORE_CONDITIONS, *ICL_BLOCK_CONDITIONS)

_DEMO_SIMPLE = """Expression: 17 * 23

Output:
```calculator
17 * 23
```"""

_DEMO_NESTED = """Expression: (19 + 7) * 43

Output:
```calculator
(19 + 7) * 43
```"""

_DEMO_NEGATIVE = """Sentence: Version 2.5 is stable.

Output:
Version 2.5 is stable."""


def _icl_target(example: ArithmeticExample) -> str:
    if example.task_kind == "arithmetic":
        return f"Now solve:\nExpression: {example.expression}\n\nOutput:"
    return (
        "Now respond to this non-arithmetic input without a calculator block:\n"
        f"Sentence: {example.reference_completion}\n\nOutput:"
    )


def _icl_block_prompt(example: ArithmeticExample, condition: str) -> str:
    target = _icl_target(example)
    if condition == "calculator_block_icl_a":
        return (
            "When arithmetic should be executed by the calculator, copy the complete arithmetic "
            "expression inside exactly one calculator block.\n\n"
            f"Example:\n{_DEMO_SIMPLE}\n\n{target}"
        )
    if condition == "calculator_block_icl_b":
        return (
            "For arithmetic, copy the complete expression inside exactly one calculator block. "
            "Do not calculate it yourself.\n\n"
            f"Example 1:\n{_DEMO_SIMPLE}\n\nExample 2:\n{_DEMO_NESTED}\n\n{target}"
        )
    if condition == "calculator_block_icl_c":
        return (
            "Use one calculator block only for a requested arithmetic operation. For prose that "
            "merely contains numbers, return the prose and do not open a calculator block.\n\n"
            f"Positive example:\n{_DEMO_SIMPLE}\n\nNegative example:\n{_DEMO_NEGATIVE}\n\n"
            f"{target}"
        )
    if condition == "calculator_block_icl_d":
        return (
            "Copy the arithmetic expression shown after `Expression:` verbatim between the "
            "calculator fences. Do not calculate it yourself. If no expression is requested, "
            "do not use calculator fences.\n\n"
            f"Example:\n{_DEMO_SIMPLE}\n\n{target}"
        )
    if condition == "calculator_block_icl_e":
        return (
            "The opening calculator fence will already be supplied. For arithmetic, copy only "
            "the complete expression shown after `Expression:`, then write a newline and the "
            "closing ``` fence. Do not calculate it yourself. For non-arithmetic prose, do not "
            "write a closing fence.\n\n"
            f"Example of the intended complete block:\n{_DEMO_SIMPLE}\n\n{target}"
        )
    if condition == "calculator_block_icl_f":
        return (
            "The harness supplies both calculator fences. Output only the complete arithmetic "
            "expression shown after `Expression:` with no fences, answer, or explanation. For "
            "non-arithmetic prose, copy the sentence.\n\n"
            "Example:\nExpression: 17 * 23\n\nOutput:\n17 * 23\n\n"
            f"{target}"
        )
    if condition == "calculator_block_icl_g":
        return (
            "Use one calculator block only for a requested arithmetic operation. For prose that "
            "merely contains numbers, return the prose and do not open a calculator block.\n\n"
            f"Negative example:\n{_DEMO_NEGATIVE}\n\nPositive example:\n{_DEMO_SIMPLE}\n\n"
            f"{target}"
        )
    raise ValueError(f"unknown ICL block condition: {condition}")


def condition_prompt(example: ArithmeticExample, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if condition in ICL_BLOCK_CONDITIONS:
        return _icl_block_prompt(example, condition)
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
