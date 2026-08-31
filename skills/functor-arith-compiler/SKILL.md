---
name: functor-arith-compiler
description: Annotate raw quantitative documents with paired low-level F1 and semantic F2 functor programs for the Paper 1 matched representation experiment. Use only for answer-blind functor annotation, not ASL translation or answer solving.
---

# Functor Arithmetic Compiler

Read [references/functor-v1.md](references/functor-v1.md) before annotating.

Infer both programs independently from the supplied raw problem and raw evidence. No
prior ASL, answer, rationale, blackboard record, or intermediate runtime value is an
allowed input. Do not calculate and substitute hidden intermediate values.

## Contract

- Preserve stated facts, entities, measured quantities, qualifiers, temporal states,
  dependencies, and the requested query.
- Use stable lowercase semantic paths. Never use anonymous `step_N`, `tmp`, `result`,
  or `value_N` paths when the text supports a meaningful name.
- F1 is the low-level control: use flat assignment functors such as `value`, `add`,
  `subtract`, `multiply`, and `divide`, each with an explicit target. It must retain
  the same dependencies as the text and must not use nested calls.
- F2 is the semantic condition: use relation functors such as `offset`, `multiple`,
  `percentage_ratio`, `percent_of`, `remaining`, and `per_unit_total`; the runtime owns arithmetic
  lowering and dependency resolution.
- Keep source relations symbolic even when mental arithmetic is easy. For example,
  use `offset("jon.cards", "mira.cards", 5)`, not `given("jon.cards", 17)`.
- Prefer the most specific F2 relation licensed by the text. Do not encode a named
  percentage, remainder, rate, or per-unit relation as a chain of generic arithmetic
  functors merely because the chain would execute.
- A referenced path may be grounded by a later source fact. Preserve that forward
  dependency rather than reordering semantics to make manual execution easy.
- End each program with exactly one `query("semantic.path")`.
- Mark a condition `ambiguous` or `unsupported` rather than inventing a relation.

The fixed examples in the reference are the only ICL examples. Do not inspect other
artifact, seed, accepted, rejected, answer, rationale, or ASL files.
