---
name: f3-grounded-compiler
description: Annotate raw quantitative documents with canonical F3 grounded observations, events, relations, evidence, and query intent for the Paper 1 matched representation experiment. Use only for answer-blind F3 annotation, not solution-program translation or answer solving.
---

# F3 Grounded Compiler

Read [references/f3-v1.md](references/f3-v1.md) before annotating.

Represent what the supplied source asserts, not a hidden calculation plan. Use
only the raw question and permitted source context. Do not inspect or infer from
answers, rationales, prior ASL/functor programs, accepted/rejected artifacts,
runtime state, or validator output.

## Contract

- Preserve source entities, attributes, quantities, units, time, event roles,
  relations, coreference, collections, and query intent.
- Attach exact `source(...)` spans or exact `cell(row,column)` labels to every
  non-query form.
- Use stable lowercase semantic paths. Keep a path consistent within a program;
  do not use `step_N`, `tmp`, or anonymous solution variables.
- Use `observe` only for explicit numeric source facts.
- Use events only for state-changing assertions and preserve actor, affected
  state, quantity, and event identity.
- Preserve symbolic references such as "that many" with `event_field`; never
  replace them with mentally calculated constants.
- Use declarative relations for source-stated comparisons, ratios, rates,
  totals, products, differences, and averages.
- Express the question with an intent-level `query`; do not insert unstated
  aggregation or arithmetic steps.
- Emit one canonical call per line and exactly one terminal `query(...)`.
- Mark the record `ambiguous` or `unsupported` instead of inventing a predicate
  or unsupported source fact.

The examples in the F3 reference are the only teacher ICL examples. They are
synthetic and are not frozen benchmark records.
