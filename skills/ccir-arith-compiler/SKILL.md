---
name: ccir-arith-compiler
description: Compile quantitative dataset clauses into semantically grounded ASL-Arith state deltas. Use for local Codex or remote-teacher annotation and review of Paper 1 NL-to-ASL rows; do not use for answer-only arithmetic traces or non-arithmetic logic compilation.
---

# CCIR Arithmetic Compiler

Compile only `current_part` in the supplied request. Read
[references/asl-arith-v0.md](references/asl-arith-v0.md) for the surface
language and [references/semantic-annotation.md](references/semantic-annotation.md)
for the annotation standard before producing a mapping.

## Contract

- Treat `state_before`, `effective_scope`, source evidence, and `operator_registry` as authoritative.
- Emit the minimal ASL statement list that preserves the clause's semantic delta; one clause may require several statements.
- Emit compact ASL, not JSON AST and not a calculator/tool name.
- Do not repeat `SCOPE` when `effective_scope` already identifies the record.
- Use stable lowercase semantic paths, normally `entity.context.quantity`. Do not use `step_N`, `tmp`, or `result` when the text supports a meaningful name.
- Every path segment must start with a letter or underscore. Prefix numeric years and periods, for example `revenue.y2019` and `quarter.q4`, never `revenue.2019`.
- Preserve entities, measured quantities, relations, units, and temporal/state distinctions. A nearby noun is not automatically the measured quantity.
- Do not invent source facts, values, entities, or fields. A relation may reference a not-yet-grounded semantic path when the full question establishes that entity or quantity; the runtime stores it symbolically until grounded.
- Distinguish source facts from derived values. Assign stated literals directly and preserve source relations instead of collapsing them to solved constants.
- Use qualified references only for a permitted sibling or nested scope. Never reference another dataset record.
- Return `ambiguous` when reference resolution or intended arithmetic is underdetermined.
- Return `unsupported` when the clause requires semantics outside the active arithmetic profile.
- Use `RETURN semantic.path` only when the current clause asks for the externally meaningful result. Name a requested aggregate or derived quantity before returning it.
- Do not use `correct_answer`, gold rationale, or annotated equations during primary generation. Those fields are validator-only; if exposed during repair, label the mapping as rationale-assisted.

## Required Output

Return one JSON object and no surrounding prose or Markdown:

```json
{
  "status": "ok",
  "part_id": 1,
  "asl": [
    "month2.downloads = month1.downloads * 3"
  ],
  "semantic_notes": [],
  "assumptions": [],
  "confidence": 0.97
}
```

For `ambiguous` or `unsupported`, set `asl` to an empty list and explain the
specific issue in `assumptions`. `semantic_notes` are review metadata, not a
training target. Confidence must be between 0 and 1.

Before returning `ok`, check that the ASL parses deterministically, references
are visible in the effective scope or are legitimate forward dependencies, and
every operator is in the supplied registry. Then check semantic grounding
separately: the state should retain enough meaning to support a plausible
follow-up question, not merely reproduce the benchmark answer.
