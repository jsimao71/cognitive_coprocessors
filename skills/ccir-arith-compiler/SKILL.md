---
name: ccir-arith-compiler
description: Compile one quantitative natural-language clause into compact ASL-Arith for a supplied teacher-request envelope. Use for local Codex bootstrap labeling or review of Paper 1 NL-to-ASL dataset rows; do not use for direct prose answers or non-arithmetic logic compilation.
---

# CCIR Arithmetic Compiler

Compile only `current_part` in the supplied request. Read
[references/asl-arith-v0.md](references/asl-arith-v0.md) before producing a
mapping.

## Contract

- Treat `state_before`, `effective_scope`, and `operator_registry` as authoritative.
- Emit the shortest ASL statement or statements that preserve the clause's semantics.
- Emit compact ASL, not JSON AST and not a calculator/tool name.
- Do not repeat `SCOPE` when `effective_scope` already identifies the record.
- Do not invent source facts, values, entities, fields, or unresolved references.
- Distinguish source facts from derived values. Assign facts directly; express derivations with operators.
- Use qualified references only for a permitted sibling or nested scope. Never reference another dataset record.
- Return `ambiguous` when reference resolution or intended arithmetic is underdetermined.
- Return `unsupported` when the clause requires semantics outside the active arithmetic profile.
- Use `RETURN expression` only when the current clause asks for the externally meaningful result.

## Required Output

Return one JSON object and no surrounding prose or Markdown:

```json
{
  "status": "ok",
  "part_id": 1,
  "asl": "month2.downloads = month1.downloads * 3",
  "assumptions": [],
  "confidence": 0.97
}
```

For `ambiguous` or `unsupported`, set `asl` to an empty string and explain the
specific issue in `assumptions`. Confidence must be between 0 and 1.

Before returning `ok`, check that the ASL parses deterministically, every read
reference exists in `state_before` or is established by the emitted statements,
and the statement uses only the supplied operator registry.
