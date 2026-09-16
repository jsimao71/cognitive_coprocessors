# AGENTS — ASL Example Generation Guidelines
## Semantic NL → ASL Annotation for Paper 1

## Purpose

This document instructs Codex, remote teacher models, and human reviewers how to generate high-quality ASL supervision examples from natural-language quantitative reasoning tasks.

The goal is NOT merely to reconstruct the arithmetic operations needed to get the final answer.

The goal is to produce a semantically grounded, stateful intermediate representation that preserves entities, quantities, relations, temporal/state distinctions, dependencies, unresolved references, derived values, query intent, and scope.

The generated ASL must be compact, executable when grounded, and useful for downstream continuation.

## 1. Core distinction: operation trace vs semantic program

A benchmark rationale may imply:

```text
20 - 2
6 + 18
```

A weak mapping is:

```text
step_1 = 20 - 2
step_2 = 6 + 18
RETURN step_2
```

This is acceptable only as an operation ledger.

It is NOT preferred semantic ASL because it discards who the values belong to, what they represent, why the operations are valid, and how later clauses can reuse the state.

Preferred:

```text
claire.age_in_2y = 20
claire.age_now = claire.age_in_2y - 2
jessica.age_now = claire.age_now + 6
RETURN jessica.age_now
```

The teacher should generate semantic ASL whenever the natural language supports it.

## 2. Primary annotation objective

For each NL clause or chopped part:

```text
state_before + current_part + local problem context
    ->
semantic ASL delta
```

The target is the minimal set of ASL statements that faithfully represents the meaning introduced by that part.

Do NOT force one NL part to one ASL statement.

A single sentence may introduce multiple facts, one or more relations, derived values, a query, or a return.

## 3. Preserve semantic names

Prefer names derived from the meaning of the text.

Good:

```text
natalia.april.clips = 48
natalia.may.clips = natalia.april.clips / 2
```

Bad:

```text
step_1 = 48
step_2 = step_1 / 2
```

Use anonymous `step_N` names only when no meaningful semantic label can be inferred.

## 4. Preserve entities explicitly

If multiple entities exist, encode them.

Example:

```text
Jessica is six years older than Claire.
```

Preferred:

```text
jessica.age_now = claire.age_now + 6
```

Not:

```text
x = y + 6
```

unless the source truly uses abstract variables.

## 5. Preserve quantity type/field

Map numbers to what they measure, not merely to the nearest noun.

Example:

```text
Natalia sold clips to 48 of her friends in April.
```

Preferred:

```text
natalia.april.clips = 48
```

Potentially wrong:

```text
natalia.april.friends = 48
```

The teacher must reason about the semantic role of the quantity.

## 6. Represent unresolved dependencies

Do NOT refuse to emit a relation merely because the referenced value is not known yet.

Example:

```text
Jessica is six years older than Claire.
```

Valid:

```text
jessica.age_now = claire.age_now + 6
```

The runtime may keep this symbolic until `claire.age_now` is grounded.

This is a core ASL behavior.

## 7. Prefer declarative relations over premature calculation

If a clause introduces a relation:

```text
Month 2 has three times as many.
```

Preferred:

```text
month2.downloads = month1.downloads * 3
```

Do not collapse immediately to a literal unless the runtime separately preserves the derivation.

## 8. Preserve source-vs-derived distinction

A literal stated in the question is a source fact.

Example:

```text
Claire will be 20 in two years.
```

Preferred:

```text
claire.age_in_2y = 20
claire.age_now = claire.age_in_2y - 2
```

Do not emit only:

```text
claire.age_now = 18
```

because that loses the source relation.

## 9. Temporal/state distinctions matter

Different times/states should not overwrite one another.

Good:

```text
claire.age_in_2y = 20
claire.age_now = claire.age_in_2y - 2
```

Bad:

```text
claire.age = 20
claire.age = 18
```

Use explicit state/time-qualified fields where needed.

## 10. Percentages and rates

Prefer semantic operators when they reduce ambiguity.

Example:

```text
Month 3 is 30% lower than month 2.
```

Preferred:

```text
month3.downloads = dec_pct(month2.downloads, 30)
```

Equivalent arithmetic:

```text
month3.downloads = month2.downloads * 0.70
```

The semantic form is preferred for supervision because it preserves the NL meaning.

## 11. Aggregation/query semantics

Example:

```text
How many clips did Natalia sell altogether in April and May?
```

Preferred:

```text
natalia.total.clips = natalia.april.clips + natalia.may.clips
RETURN natalia.total.clips
```

Do not simply emit:

```text
RETURN 72
```

The model must represent the requested computation.

## 12. RETURN semantics

`RETURN` marks the externally requested result.

Prefer returning a named semantic state value:

```text
RETURN natalia.total.clips
```

rather than repeating a long expression.

## 13. Scope rules

Every example has an effective scope, even if no explicit `SCOPE` syntax is generated.

For public benchmark cases, the external dataset record normally supplies the root scope.

Do NOT emit explicit scope syntax unless:
- the source itself contains meaningful nested subproblems;
- the task explicitly asks to reconstruct scope;
- sibling/local state separation is needed within one external root scope.

When explicit scope is needed:

```text
SCOPE exercise1
  SCOPE a
    ...
  END
  SCOPE b
    y = a.x * 3
  END
END
```

Do not reference unrelated scopes.

Use qualified names for sibling/subscope references.

## 14. Multiple statements per part

Teacher output must allow arrays/lists of ASL statements.

Example:

NL:

```text
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
```

Preferred:

```text
natalia.april.clips = 48
natalia.may.clips = natalia.april.clips / 2
```

One sentence, two ASL statements.

## 15. Clause-by-clause causal annotation

Prefer incremental generation.

For part `i`, teacher receives:
- full question for context;
- current part;
- previous parts;
- current semantic state;
- effective scope;
- allowed ASL profile.

Teacher should emit only semantic delta introduced or requested by the current part.

Avoid re-emitting all previous state unless needed.

## 16. Good example: Natalia

NL part 0:

```text
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
```

Good:

```text
natalia.april.clips = 48
natalia.may.clips = natalia.april.clips / 2
```

NL part 1:

```text
How many clips did Natalia sell altogether in April and May?
```

Good:

```text
natalia.total.clips = natalia.april.clips + natalia.may.clips
RETURN natalia.total.clips
```

## 17. Good example: Jessica/Claire

NL part 0:

```text
Jessica is six years older than Claire.
```

Good:

```text
jessica.age_now = claire.age_now + 6
```

Do NOT wait for Claire's age.

NL part 1:

```text
In two years, Claire will be 20 years old.
```

Good:

```text
claire.age_in_2y = 20
claire.age_now = claire.age_in_2y - 2
```

NL part 2:

```text
How old is Jessica now?
```

Good:

```text
RETURN jessica.age_now
```

The runtime should resolve Jessica's age after Claire's age becomes grounded.

## 18. Bad example patterns

### Anonymous arithmetic only

Bad:

```text
step_1 = 20 - 2
step_2 = 18 + 6
RETURN step_2
```

Reason:
- loses semantic grounding;
- poor continuation value;
- difficult to reuse.

### Final answer only

Bad:

```text
RETURN 24
```

Reason:
- no semantic compilation.

### Premature numeric collapse

Bad:

```text
jessica.age_now = 24
```

when NL supplies relational structure.

### Invented facts

Bad:

```text
claire.age_now = 18
```

before the relevant clause is seen.

### Wrong quantity grounding

Bad:

```text
natalia.april.friends = 48
```

when 48 represents clips sold.

## 19. Ambiguity

If the clause cannot be mapped faithfully, return:

```json
{
  "status": "ambiguous",
  "reason": "...",
  "asl": []
}
```

Do not invent missing semantics.

Examples:
- unclear pronoun;
- underspecified comparison target;
- ambiguous unit;
- inconsistent source text.

## 20. Unsupported semantics

If the current ASL profile cannot express a valid relation, use:

```json
{
  "status": "unsupported",
  "reason": "requires operator family not available in asl-arith-v0",
  "asl": []
}
```

Do not force logic/graph/data semantics into arithmetic operators.

## 21. Teacher output shape

Recommended API response:

```json
{
  "status": "ok",
  "part_id": 0,
  "asl": [
    "natalia.april.clips = 48",
    "natalia.may.clips = natalia.april.clips / 2"
  ],
  "semantic_notes": [
    "48 refers to clips sold, not number of friends"
  ],
  "assumptions": [],
  "confidence": 0.97
}
```

`semantic_notes` are optional and are NOT part of the training target.

## 22. Validation levels

Validate separately for:

### Syntax
Does it parse?

### Scope
Are references legal and visible?

### Types
Are operators applied to compatible values?

### Execution
Can grounded expressions execute?

### Final answer
Does the resulting program reproduce the benchmark answer?

### Intermediate trace
Where trustworthy, does it match annotated operations?

### Semantic grounding
Does the ASL preserve entities, quantities, relations, and temporal/state meaning from NL?

Execution correctness does NOT replace semantic-grounding review.

## 23. Quality grades

Suggested grades:

### Q0_OPERATION_LEDGER
Mechanically derived arithmetic trace only.

### Q1_TEACHER_EXEC_VERIFIED
Teacher-generated semantic ASL that parses and produces correct final answer.

### Q2_TEACHER_TRACE_VERIFIED
Also matches trustworthy intermediate operations.

### Q3_MULTI_TEACHER_SEMANTIC_AGREEMENT
Multiple independent teachers produce semantically equivalent ASL.

### Q4_MANUAL_SEMANTIC_GOLD
Human/Codex reviewed for semantic grounding.

Keep provenance for every grade.

## 24. Use existing operation ledgers as constraints

Do NOT discard annotation-derived `step_N` programs.

Use them as lower-level constraints.

Pipeline:

```text
NL
 -> semantic teacher ASL
 -> execute/lower
 -> operation trace
 -> compare to annotation-derived operation ledger
```

Agreement strengthens confidence.

Disagreement must be inspected:
- teacher may be wrong;
- dataset rationale may be incomplete;
- multiple equivalent programs may exist.

Do not require exact string equality.

## 25. Semantic equivalence over exact match

Two ASL programs may be executable-equivalent but differ in semantic quality.

Example:

```text
claire.age_now = claire.age_in_2y - 2
jessica.age_now = claire.age_now + 6
```

versus:

```text
jessica.age_now = claire.age_in_2y - 2 + 6
```

The first is preferred because it preserves intermediate semantic state.

Evaluation should distinguish:
- executable equivalence;
- final-answer equivalence;
- semantic-state quality.

## 26. Naming conventions

Use normalized lowercase identifiers.

Prefer:

```text
jessica
claire
month1
age_now
downloads
total_clips
```

Avoid arbitrary names such as `tmp1`, `foo`, `step_2` unless unavoidable.

Use stable names across clauses.

## 27. Path conventions

Prefer:

```text
entity.context.quantity
```

Examples:

```text
natalia.april.clips
claire.age_now
month2.downloads
store.week1.revenue
```

Do not create excessive hierarchy when unnecessary.

## 28. Canonical rendering

Teacher surface ASL should be compact and consistent.

Preferred:

```text
x = a + b
```

One statement per line.

No semicolons in the Paper 1 canonical renderer.

## 29. Numeric literals

Preserve exact source values.

Avoid converting:

```text
30%
```

to `0.3` if a semantic operator such as `dec_pct(..., 30)` is available.

Use integer/rational/decimal-safe representations where possible.

## 30. Dependency graph

Every semantic mapping should induce a dependency graph.

Example:

```text
claire.age_in_2y
    ↓
claire.age_now
    ↓
jessica.age_now
    ↓
RETURN
```

The validator should record:
- unresolved dependencies;
- newly grounded values;
- cycles;
- illegal references.

Paper 1 arithmetic should normally be acyclic.

## 31. Deferred evaluation

ASL runtime must allow symbolic expressions whose dependencies are unresolved.

Example:

```text
jessica.age_now = claire.age_now + 6
```

is valid even if `claire.age_now` is unknown.

Store the expression.

When later state grounds `claire.age_now`, reevaluate dependents.

This is required for causal clause-by-clause annotation.

## 32. Do not leak gold rationale to teacher by default

Teacher input should normally include:
- question;
- chopped question clauses;
- current part;
- state_before.

Do NOT include:
- gold rationale;
- annotated equations;
- final program.

Gold rationale is validator data.

A separate repair/adjudication mode may expose it, but such rows must be labeled.

## 33. Final-answer visibility

For primary semantic teacher generation, prefer NOT to show the final answer.

Separate:
- generation input;
- validator input.

Teacher should infer semantics from NL, not backward-fit a known answer.

## 34. Teacher consensus

For selected examples, query multiple teachers independently.

Compare:
- entity structure;
- quantity fields;
- operators;
- references;
- execution;
- final state.

Use normalized AST/CCIR equivalence rather than raw ASL string match.

Disagreements go to review.

## 35. Perturbation

Once semantic ASL is verified, controlled perturbations may generate new NL/ASL pairs.

Permitted:
- numeric changes;
- large-number scaling;
- operator-aware paraphrase;
- entity renaming;
- quantity-field renaming;
- safe clause reordering;
- distractors.

Every perturbation must transform ASL consistently.

## 36. Reviewer checklist

For every manually reviewed example ask:

1. Are all important entities represented?
2. Are quantities attached to the correct entity/field?
3. Are temporal/state distinctions preserved?
4. Are relations preserved rather than prematurely collapsed?
5. Are unresolved references allowed where appropriate?
6. Is any fact invented?
7. Does each NL clause map to the correct semantic delta?
8. Does ASL execute when dependencies are grounded?
9. Does it produce the benchmark answer?
10. Would the resulting state help answer a plausible follow-up?

If answer 10 is no, the mapping may be too operation-centric.

## 37. Core Paper 1 hypothesis

The desired compiler learns:

```text
natural language relation
    ->
semantic state / dependency expression
```

not merely:

```text
word problem
    ->
arithmetic trace
```

The runtime/coprocessor then performs exact arithmetic.

A successful ASL representation should remain semantically stable under:
- large-number perturbation;
- numeric remapping;
- paraphrase;
- surface wording changes.

This distinction must be explicit in teacher prompts, skill documentation, dataset quality grades, and Paper 1 evaluation.
