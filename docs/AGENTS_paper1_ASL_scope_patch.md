# AGENTS PATCH — Paper 1: ASL Surface DSL + Optional Hierarchical Scopes

## Purpose
Patch `AGENTS_paper1_dsl_bootstrap.md`.

The model-facing language must be a minimal, permissive ASL rather than JSON AST. JSON/Python-dict remains the canonical parsed representation.

Target pipeline:

`NL -> ASL -> fast parser -> generic AST -> semantic/type/scope validation -> CCIR -> operator registry -> CogCop`

The grammar should remain stable while operator/predicate vocabularies and coprocessors grow.

## ASL-Core grammar

Treat this as a semantic sketch; implement with a small Pratt/precedence parser rather than a large rigid CFG.

```text
atom :=
    identifier | variable | number | string | boolean | null
  | path | call | list | record | "(" expression ")"

term :=
    unary-op* atom

expression :=
    term (binary-op term)*

statement :=
    expression
  | expression statement-op expression
  | query-op expression
  | return-op expression
  | scope-start
  | scope-end
```

Function/predicate names are open-ended. The grammar MUST NOT enumerate registered functions.

Examples:

```text
month2.downloads = month1.downloads * 3
bird(penguin)
has_wings(?x) :- bird(?x)
sales <- lookup(dataset="sales", year=2026)
? has_wings(penguin)
RETURN total
```

## Operator registry

The parser knows token/fixity/precedence/associativity. Semantic lowering knows meaning, types, and executor.

Prepare:

```text
statement-op ::= = | <- | :- | ...
binary-op    ::= + | - | * | / | comparisons | AND | OR | , | ; | ...
unary-op     ::= - | ! | NOT | ...
```

Do not conceptualize `=`, `<-`, and `:-` all as assignment. They are generic relation/statement operators.

Suggested precedence, high to low:
1. call/path/parentheses
2. unary
3. `* /`
4. `+ -`
5. comparisons
6. `AND` / expression-level comma
7. `OR` / expression-level semicolon
8. `=`, `<-`, `:-`

Use parser context to distinguish comma as call-argument separator from conjunction.

## Paper 1 ASL-Arith profile

Prefer compact expressions familiar to tiny models:

```text
month1.downloads = 60
month2.downloads = month1.downloads * 3
month3.downloads = dec_pct(month2.downloads, 30)
total = month1.downloads + month2.downloads + month3.downloads
RETURN total
```

The same canonical AST may also render function arithmetic:

```text
month2.downloads = mul(month1.downloads, 3)
```

Run a small syntax ablation before large training to compare infix vs function-call arithmetic. JSON AST is a storage/control baseline, not the primary model target.

## Paper 2 grammar compatibility

ASL-Core must already parse:

```text
bird(penguin)

has_wings(?x) :- bird(?x)

grandparent(?x, ?z) :-
    parent(?x, ?y),
    parent(?y, ?z)

? grandparent(alice, carol)

evidence <- lookup(query="...")
ok <- verify(claim, evidence)
```

Paper 1 does not train/evaluate these semantics. Add parser/extensibility tests only.

## Generic AST mapping

Surface:

```text
month2.downloads = month1.downloads * 3
```

maps mechanically to:

```json
{
  "type": "statement",
  "operator": "=",
  "left": {"type": "path", "parts": ["month2", "downloads"]},
  "right": {
    "type": "binary",
    "operator": "*",
    "left": {"type": "path", "parts": ["month1", "downloads"]},
    "right": {"type": "number", "value": 3}
  }
}
```

Surface:

```text
has_wings(?x) :- bird(?x)
```

maps to a statement AST preserving `:-`, calls, and variable nodes.

A second semantic-lowering pass converts generic syntax AST to typed CCIR based on registry/type/state context.

## Optional scopes: core invariant

ASL-Core MUST support hierarchical scopes, but explicit scope syntax is optional.

Invariant:

> Every ASL statement/state item belongs to exactly one effective scope, even when no scope tokens appear in ASL.

Scopes may be supplied by:
1. explicit ASL syntax;
2. dataset/application metadata;
3. PRA document/record boundaries;
4. configured syntactic hints;
5. runtime default root scope.

This prevents long-running state from becoming one flat namespace.

## Explicit scope syntax

Minimal syntax:

```text
SCOPE exercise1
  ...
END
```

Nested example:

```text
SCOPE exercise1

  SCOPE a
    x = 12
    RETURN x
  END

  SCOPE b
    y = a.x * 3
    RETURN y
  END

END

SCOPE exercise2
  x = ...
  RETURN x
END
```

Indentation is readability only. `SCOPE` pushes; `END` closes.

Do not add braces or mandatory `END SCOPE` unless experiments show a reliability benefit.

## Scope semantics

Each scope records:
- `scope_id`
- `parent_scope_id`
- optional `kind`
- state/facts/rules/results
- provenance
- lifecycle metadata

Unqualified lookup:
1. current scope;
2. permitted ancestors;
3. workspace/global scope if policy allows.

Sibling locals are NOT implicitly visible. Use qualified references:

```text
a.x
exercise1.a.x
```

This makes subexercise dependencies explicit and prevents accidental state leakage.

## Default/external scope

For public datasets such as GSM8K, each example already defines a natural root scope.

Example runtime metadata:

```text
scope_id = gsm8k:test:123
scope_kind = benchmark_case
scope_source = dataset
```

Therefore the Paper 1 model should normally NOT generate:

```text
SCOPE gsm8k_test_123
```

Scope syntax exists for general real-world use but is omitted when the application already provides scope.

## PRA-defined scope

A new PRA document/reference may automatically create:

```text
scope_kind = document
scope_id = <PRA document/reference identity>
```

Sections/subsections may create nested scopes according to application policy.

Closing a scope may later trigger PRA/runtime policies:
- freeze local state;
- index/summarize;
- stop materializing locals by default;
- persist provenance;
- move state to cold storage;
- retain qualified references.

ASL scope semantics must also work without PRA.

## Configurable scope hints

Applications/users may provide prompt/document-specific hints.

Examples:

```text
---
```

Markdown:

```text
# Exercise 1
## a)
## b)
# Exercise 2
```

Dataset fields:

```text
exercise_id
subquestion_id
document_id
section_id
```

Custom markers:

```text
BEGIN TASK
END TASK
```

Implement configurable policies, e.g.:

```yaml
scope_policy:
  root: per_record
  hints:
    - type: separator
      value: "---"
      action: new_sibling_scope
    - type: markdown_heading
      levels:
        1: root_child
        2: nested_child
```

Do not hard-code prompt-specific boundaries into ASL grammar.

## Scope precedence

Effective scope is composed from external ownership plus optional explicit nesting.

Example:

```text
external: document:math_book
ASL: SCOPE exercise1
```

effective scope path:

```text
document:math_book / exercise1
```

Explicit ASL scope augments the external scope; it does not erase provenance/ownership.

## Canonical scope metadata

Even when ASL contains no `SCOPE`, canonical records include scope:

```json
{
  "scope": {
    "id": "gsm8k:test:123",
    "parent": null,
    "kind": "benchmark_case",
    "source": "dataset"
  },
  "statement": {
    "type": "statement",
    "operator": "=",
    "left": {"type": "path", "parts": ["month1", "downloads"]},
    "right": {"type": "number", "value": 60}
  }
}
```

Record scope provenance as:
- model/ASL;
- dataset;
- PRA;
- application;
- heuristic/hint;
- runtime default.

## RETURN and visibility

`RETURN` marks the externally meaningful result of the current effective scope.

Do not automatically export every local variable to sibling scopes.

A dependent sibling should explicitly reference:

```text
a.x
```

Explicit import/export can be added later if needed; do not complicate Paper 1.

## Skill patch

Update `skills/ccir_arith_compiler/SKILL.md`:

- Generate compact ASL, not raw AST JSON.
- Use the shortest semantically sufficient representation.
- The teacher receives `effective_scope`.
- If scope is already supplied by dataset/app/PRA, DO NOT redundantly emit `SCOPE`.
- Emit `SCOPE/END` only when reconstructing meaningful nested subproblems/sections is part of the requested task.
- Never reference unrelated scopes.
- Use qualified references for sibling/subscope state.
- Return `ambiguous` rather than inventing semantics.
- Every ASL output must parse deterministically to canonical AST.

## Remote teacher protocol

The remote API envelope may remain JSON for reliable structured output:

```json
{
  "asl_version": "asl-core-v0",
  "profile": "asl-arith-v0",
  "effective_scope": {
    "id": "gsm8k:test:123",
    "kind": "benchmark_case",
    "source": "dataset"
  },
  "question": "...",
  "parts": [...],
  "current_part": 1,
  "state_before": {},
  "operator_registry": []
}
```

Teacher response:

```json
{
  "status": "ok",
  "part_id": 1,
  "asl": "month2.downloads = month1.downloads * 3",
  "assumptions": [],
  "confidence": 0.97
}
```

Important distinction:

`JSON API envelope -> ASL field -> local parser -> canonical AST`

The model is learning ASL, not JSON AST serialization.

## Validation patch

Reject/retry if:
- ASL fails parsing;
- operator/fixity is unknown for active profile;
- precedence produces invalid structure;
- references do not resolve;
- scope visibility is violated;
- type/signature validation fails;
- deterministic execution fails;
- verified final answer disagrees.

Harmless whitespace/canonical formatting differences should normalize automatically.

## Dataset record patch

Accepted compiler records store both representations:

```json
{
  "source_id": "...",
  "effective_scope": {...},
  "part": "...",
  "asl": "month2.downloads = month1.downloads * 3",
  "ast": {...},
  "state_before": {...},
  "state_after": {...},
  "validation": {...}
}
```

This enables:
- compact LoRA targets;
- deterministic semantic evaluation;
- later alternative surface renderings from one AST.

## Chopping + scope inference

Extend the dataset miner so chopping and scope are distinct operations.

Pipeline:

```text
source record
  -> infer external/default scope
  -> chop into clauses/parts
  -> optional infer nested scope hints
  -> teacher NL->ASL
```

For current public datasets:
- one QA item = one root scope;
- chopped clauses stay in that scope unless dataset structure explicitly has subquestions.

Do not invent subscopes merely because a rationale has multiple steps.

## Scope audit

Add statistics:
- records per root scope;
- nested scopes inferred;
- unresolved qualified refs;
- cross-scope violations;
- accidental symbol collisions;
- scope source distribution.

For public Paper 1 datasets, unexpected cross-record references are a hard error.

## Training

Primary Paper 1 LoRA target is compact ASL.

The model input should include the effective current state and, when relevant, scope identity/accessible symbols.

Do not force scope tokens into every training target.

Include a small auxiliary set with explicit `SCOPE/END` to ensure grammar learnability, but do not let it dominate arithmetic training.

## Syntax ablation before scale

Before generating a very large teacher corpus, take a verified seed AST set and deterministically render:

1. compact infix ASL;
2. function-call ASL;
3. JSON AST baseline.

Train/evaluate small ICL/LoRA pilots.

Metrics:
- output tokens;
- parse validity;
- semantic-equivalent AST;
- reference accuracy;
- sample efficiency;
- numeric-remapping robustness.

Select the Paper 1 primary surface empirically.

## Paper 2 handoff

Freeze ASL-Core grammar after Paper 1 if possible.

Paper 2 should expand:
- operator registry;
- predicate/function signatures;
- semantic lowering;
- coprocessor executors;
- training data for new semantic families;

rather than redesigning syntax.

The desired property is:

> New coprocessors usually add semantic vocabulary and runtime handlers, not new grammar.

Logic/Datalog support should therefore be possible using already-supported `:-`, variables, calls, conjunction, queries, and scopes.

## Immediate patch order

1. Patch the existing AGENTS plan with this ASL-Core design.
2. Implement generic parser + canonical AST mapping.
3. Implement external/default scope objects.
4. Add explicit `SCOPE/END` parser tests.
5. Make GSM8K loader assign one root scope per example.
6. Update bootstrap skill to generate ASL.
7. Update teacher JSON envelope to carry ASL output.
8. Add scope/reference validation.
9. Run the small surface-syntax ablation.
10. Only then scale teacher generation and LoRA training.

## Claim boundary

Paper 1 does not claim learned general scope discovery.

It establishes:
- a scope-capable ASL/runtime;
- externally supplied natural scopes for public datasets;
- compact NL->ASL arithmetic compilation;
- deterministic mapping to canonical AST;
- extensibility tests showing future rule/predicate syntax parses without redesign.

Learned/document-level scope discovery is future work unless unexpectedly cheap and compelling.
