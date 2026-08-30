# Semantic NL-to-ASL Annotation

## Objective

Map each natural-language clause to the smallest state delta that preserves its
meaning. The target is a semantic program and dependency graph, not an anonymous
operation trace.

For each part, use the full question and source evidence for context, previous
parts, `state_before`, the effective scope, and the active ASL profile. Emit only
facts, relations, derived values, or query intent introduced by the current part.
Do not force one statement per part.

## Grounding Rules

- Represent important entities explicitly and attach values to what they measure.
  In "sold clips to 48 friends", `48` measures clips sold in the intended task,
  not a generic `friends` field.
- Prefer stable lowercase paths such as `natalia.april.clips`,
  `claire.age_now`, or `store.week1.revenue`.
- Path segments cannot begin with a digit. Render years and periods as `y2019`,
  `q4`, or another semantic identifier, not `.2019` or `.4`.
- Preserve source literals and derivations separately. For "will be 20 in two
  years", use `claire.age_in_2y = 20` and
  `claire.age_now = claire.age_in_2y - 2`, not only `claire.age_now = 18`.
- Preserve temporal and state distinctions instead of overwriting one field.
- Encode declarative relations even when a dependency is not grounded yet. The
  runtime can defer `jessica.age_now = claire.age_now + 6` until Claire's age is
  known.
- Prefer semantic calls such as `dec_pct(x, 30)`, `inc_pct(x, 10)`,
  `percent_of(x, 20)`, and `mean(...)` when they preserve the source wording.
- Name requested aggregates and return that semantic path. Do not emit only a
  literal final answer.
- Use explicit `SCOPE/END` only for genuine nested subproblems. The dataset record
  already supplies the root scope.

## Examples

For "Jessica is six years older than Claire":

```text
jessica.age_now = claire.age_now + 6
```

For "In two years, Claire will be 20 years old":

```text
claire.age_in_2y = 20
claire.age_now = claire.age_in_2y - 2
```

For "How old is Jessica now?":

```text
RETURN jessica.age_now
```

For "Natalia sold 48 clips in April and half as many in May":

```text
natalia.april.clips = 48
natalia.may.clips = natalia.april.clips / 2
```

For "How many clips altogether in April and May?":

```text
natalia.total.clips = natalia.april.clips + natalia.may.clips
RETURN natalia.total.clips
```

## Failure Modes

Reject or repair mappings that use anonymous arithmetic (`step_1`, `tmp`, or
`result`), return only the final answer, collapse a stated relation to a literal,
attach a value to the wrong quantity, overwrite distinct times/states, or invent
facts. Return `ambiguous` for unresolved pronouns, comparison targets, or units;
return `unsupported` when the arithmetic profile cannot express the semantics.

## Validation And Provenance

Validate syntax, scope, types, execution after grounding, final answer,
trustworthy intermediate operations, and semantic grounding separately.
Execution correctness never substitutes for semantic review.

Use these grades:

- `Q0_OPERATION_LEDGER`: annotation-derived arithmetic trace only.
- `Q1_TEACHER_EXEC_VERIFIED`: semantic teacher output that parses and reaches the answer.
- `Q2_TEACHER_TRACE_VERIFIED`: Q1 plus agreement with trustworthy operations.
- `Q3_MULTI_TEACHER_SEMANTIC_AGREEMENT`: independent teachers agree semantically.
- `Q4_MANUAL_SEMANTIC_GOLD`: Codex/human review confirms grounding.

Primary generation must not expose the final answer, rationale, equations, or an
existing program. A repair/adjudication pass may use them, but the row must say
that it is rationale-assisted. In repair, correct only the diagnosed issue while
retaining or improving semantic grounding; do not regress to an operation ledger.
Compare normalized AST/CCIR and dependency graphs,
not exact strings. Retain the Q0 operation ledger as an execution constraint.

## Review Checklist

Confirm that entities and measured quantities are correct, time/state is
preserved, source relations remain explicit, unresolved dependencies are legal,
no fact is invented, each clause contributes the right delta, grounded execution
matches the benchmark, and the state would help answer a plausible follow-up.
