"""Offset-preserving, dataset-aware clause chopping."""

from __future__ import annotations

from typing import Any

_ABBREVIATIONS = {
    "a.m",
    "cm",
    "dr",
    "ft",
    "in",
    "jr",
    "kg",
    "km",
    "lb",
    "lbs",
    "mi",
    "mm",
    "mr",
    "mrs",
    "ms",
    "no",
    "oz",
    "p.m",
    "sr",
    "st",
    "vs",
    "yd",
}


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\n":
            if text[start:index].strip():
                spans.append((start, index))
            start = index + 1
        elif character in ".!?":
            end = index + 1
            while end < len(text) and text[end] in ".!?":
                end += 1
            previous = text[start:index].rstrip().rsplit(maxsplit=1)[-1].casefold().rstrip(".")
            boundary = end == len(text) or text[end].isspace()
            if boundary and previous not in _ABBREVIATIONS:
                spans.append((start, end))
                while end < len(text) and text[end].isspace() and text[end] != "\n":
                    end += 1
                start = end
                index = end - 1
        index += 1
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _parts(text: str, *, kind: str, heuristic: str, teacher_default: bool) -> list[dict[str, Any]]:
    parts = []
    for span_start, span_end in _sentence_spans(text):
        raw = text[span_start:span_end]
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        start = span_start + leading
        end = start + len(stripped)
        parts.append(
            {
                "part_id": len(parts),
                "text": stripped,
                "kind": kind,
                "start": start,
                "end": end,
                "heuristic": heuristic,
                "confidence": 0.8,
                "teacher_input_default": teacher_default,
                "warnings": [],
            }
        )
    if not parts and text.strip():
        start = len(text) - len(text.lstrip())
        parts.append(
            {
                "part_id": 0,
                "text": text.strip(),
                "kind": kind,
                "start": start,
                "end": start + len(text.strip()),
                "heuristic": "whole_text_fallback",
                "confidence": 0.5,
                "teacher_input_default": teacher_default,
                "warnings": ["sentence_split_empty"],
            }
        )
    return parts


def chop_example(record: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = str(record["dataset"])
    question = str(record["question"])
    heuristic = "sentence_punctuation"
    if dataset == "tatqa":
        heuristic = "tatqa_question_sentence"
    elif dataset in {"clutrr", "ruletaker"}:
        heuristic = f"{dataset}_story_sentence"
    parts = _parts(
        question,
        kind="question_clause",
        heuristic=heuristic,
        teacher_default=True,
    )
    reasoning = str(record.get("gold_reasoning", "")).strip()
    if reasoning:
        rationale_parts = _parts(
            reasoning,
            kind="gold_reasoning",
            heuristic="dataset_rationale_step",
            teacher_default=False,
        )
        offset = len(parts)
        for part in rationale_parts:
            part["part_id"] += offset
            part["start"] = None
            part["end"] = None
        parts.extend(rationale_parts)
    return parts
