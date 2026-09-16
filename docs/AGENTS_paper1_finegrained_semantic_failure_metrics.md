# AGENTS — Paper 1 Fine-Grained Semantic Failure Metrics
## Diagnosing NL → ASL / CCIR Grounding Errors

## Purpose

Extend Paper 1 evaluation so semantic failures are decomposed rather than hidden inside aggregate `paths`, `edges`, `semantic_state`, execution, and answer metrics.

Use frozen saved predictions wherever possible. Do not regenerate model outputs just to add metrics.

Primary conditions:
- whole-program LoRA-500;
- incremental predicted-state;
- incremental oracle-state;
- optionally LoRA-100, LoRA-100+3-shot, and LoRA-B for historical comparison.

Current evidence says syntax/runtime validity is relatively high while semantic grounding is weak. The goal is to determine *which* semantics fail.

## 1. Preserve distinct correctness levels

Report separately:

1. **Surface correctness** — parse/lower/type validity.
2. **Computational correctness** — execution and final answer.
3. **Semantic-workspace correctness** — entities, attributes, qualifiers, relations, dependencies, references, scope, and reusable state.

A correct answer with anonymous `step_N` arithmetic is not equivalent to a correct semantic workspace.

## 2. Multi-label semantic taxonomy

A prediction may have several errors. Do not force one exclusive label.

### Entity grounding
Correct problem object/person/group?

Example:
`jessica.age_now` vs wrong `claire.age_now`.

Metrics:
- entity precision/recall/F1;
- missing/spurious/substituted entities.

### Attribute / quantity grounding
Correct property attached to entity?

Examples: `age`, `clips`, `downloads`, `revenue`, `cartons`.

Track wrong-attribute substitutions separately from wrong entity.

### Qualifier grounding
Separate base attribute from qualifiers:
- temporal: now, future, April, May, before/after;
- cardinality: each, total, remaining;
- state/status: returned, damaged, accepted;
- grouping: month1, week2, group_a.

Metrics:
- qualifier F1;
- temporal qualifier accuracy;
- cardinality qualifier accuracy;
- status qualifier accuracy.

### Coreference / entity continuity
Resolve `she`, `her`, `each customer`, `the remaining boxes`, etc. to prior entities.

Metrics:
- coreference target accuracy;
- unnecessary-new-entity rate;
- reuse-existing-entity rate.

### Path consistency / symbol stability
If workspace already has `claire.age_now`, later `claire.current_age` is a stateful reference failure unless aliasing was established.

Report:
- standalone semantic-name equivalence;
- workspace exact-path reuse;
- accidental rename rate;
- duplicate semantic slot rate;
- unresolved due to rename.

## 3. Relation structure

Normalize each semantic relation into target, sources, operator, constants, and directed dependency edges.

For:
`jessica.age_now = claire.age_now + 6`

extract:
- target = `jessica.age_now`;
- source = `claire.age_now`;
- operator = `+`;
- constant = 6;
- edge = `claire.age_now -> jessica.age_now`.

Measure separately:

### Relation participants
Are correct entities/attributes involved?

### Argument order
Especially for `-`, `/`, comparisons.

### Relation direction
Example:
`Jessica is six years older than Claire`

Correct:
`jessica.age_now = claire.age_now + 6`

Wrong direction:
`claire.age_now = jessica.age_now + 6`

### Relation constant/magnitude
Correct structure/operator but wrong literal.

## 4. Inter-object vs intra-object relations

Separate:

Inter-object:
`jessica.age_now = claire.age_now + 6`

Intra-object/state:
`claire.age_now = claire.age_in_2y - 2`

Report accuracy for each and cross-category confusion.

## 5. Arithmetic/operator mapping

Break existing operator F1 into:
- ADD;
- SUB;
- MUL;
- DIV;
- percentage increase;
- percentage decrease;
- ratio/fraction;
- SUM/aggregation;
- MIN/MAX where supported;
- rate × duration where supported.

Produce confusion matrix.

Report:
- correct operator + wrong arguments;
- correct arguments + wrong operator;
- operator correct + relation direction wrong.

This is crucial because current evidence suggests operator mapping is stronger than semantic grounding.

## 6. Numeric/source-fact grounding

Separate:
- source literal extraction;
- correct number attached to wrong path;
- hallucinated number;
- omitted number;
- mutated number;
- derived literal presented as source fact.

Example:
`Claire will be 20 in two years`

Correct source fact:
`claire.age_in_2y = 20`

Do not give full credit to `claire.age_now = 20`.

## 7. Premature literal collapse

Detect loss of semantic dependency.

Reference:
`jessica.age_now = claire.age_now + 6`

Prediction:
`jessica.age_now = 24`

Even if correct numerically, classify:
- computation may be correct;
- dependency preservation wrong;
- workspace quality degraded.

Metrics:
- dependency-collapse rate;
- correct-value-but-lost-derivation rate.

## 8. Dependency completeness

Compare dependency graphs:
- edge precision/recall/F1;
- missing edge;
- spurious edge;
- wrong source;
- wrong target;
- wrong operator label;
- transitive shortcut.

Example reference:
`claire.age_now = claire.age_in_2y - 2`
`jessica.age_now = claire.age_now + 6`

Prediction:
`jessica.age_now = claire.age_in_2y + 4`

May be computationally equivalent but loses intermediate semantic state. Record both computational equivalence and semantic decomposition mismatch.

## 9. Aggregation/cardinality

Diagnose:
- each vs total;
- group size;
- sum over periods;
- multiplication by count;
- average vs total;
- remaining after removal/returns;
- aggregation member completeness.

Metrics:
- cardinality interpretation;
- aggregation operator;
- member selection;
- group multiplier.

## 10. Temporal semantics

Where supported, measure:
- now vs future/past;
- duration value;
- temporal shift direction;
- period identity;
- before/after;
- month/week/day qualifier.

Do not bury temporal errors inside generic path errors.

## 11. Units/dimensions

Report support counts.

Where measurable:
- unit extraction;
- unit attachment;
- dimension compatibility;
- conversion requirement;
- count/money/time confusion;
- date/quantity confusion.

If ASL-Arith typing cannot adjudicate dimensions, mark `not_measurable_with_current_type_system`. Do not infer unit competence from generic type validity.

## 12. Query / RETURN grounding

Measure:
- correct return path;
- wrong existing path;
- intermediate returned instead of final;
- literal return instead of semantic slot;
- missing return;
- premature return.

Report `return_target_accuracy`.

## 13. Scope correctness

For stateful cases:
- correct root scope;
- legal ancestor lookup;
- illegal sibling/local lookup;
- cross-example leakage;
- accidental scope creation;
- qualified-reference correctness.

Cross-record references are hard failures.

## 14. Semantic-name equivalence vs workspace stability

For standalone whole programs, allow bounded one-to-one semantic symbol mapping where usage and graph roles are consistent, e.g. `age_now` vs `current_age`.

For incremental evaluation, once a workspace symbol exists it is sticky. Free remapping is NOT allowed.

Produce:
- `semantic_name_equivalent`;
- `workspace_symbol_stable`.

## 15. Layered judge

### Stage A — deterministic AST/CCIR
Extract:
- entities;
- path segments;
- constants;
- operators;
- targets;
- references;
- dependency graph;
- RETURN;
- scope.

Most metrics must be deterministic.

### Stage B — deterministic symbol alignment
Whole-program only. Attempt one-to-one alignment using:
- lexical normalization;
- source entity names;
- attribute token overlap;
- graph role;
- value/relation structure.

Never silently merge ambiguous symbols.

### Stage C — optional strong semantic judge
Only adjudicate unresolved cases:
- `age_now` vs `current_age`;
- alternative valid decompositions;
- equivalent local ontology.

Input:
- minimal source question/parts;
- reference ASL;
- predicted ASL;
- structural comparison.

Do not expose gold rationale unless explicitly labeled adjudication mode.

Store judge provenance/confidence. Never let model judgment override deterministic syntax/type/execution facts.

## 16. Error record schema

Create per-program and per-transition records, e.g.:

```json
{
  "example_id": "...",
  "dataset": "gsm8k",
  "condition": "lora_500",
  "metrics": {
    "entity_f1": 1.0,
    "attribute_f1": 0.5,
    "qualifier_f1": 0.5,
    "operator_accuracy": 1.0,
    "relation_direction_accuracy": 0.0,
    "dependency_f1": 0.5,
    "return_target_correct": true
  },
  "errors": [
    {
      "type": "relation_direction",
      "source": "deterministic",
      "confidence": 1.0
    }
  ]
}
```

Follow repository policy for raw benchmark text. Tracked summaries should prefer IDs, hashes, counts, and safe derived data.

## 17. Conditional metrics

Compute:
- P(operator correct | parse/type valid)
- P(path correct | parse/type valid)
- P(direction correct | operator correct)
- P(answer correct | semantic state correct)
- P(answer correct | semantic state incorrect)
- P(answer correct | operator correct, path wrong)

These quantify how often correct answers arise from poor semantic state.

## 18. Whole-program vs incremental

On identical frozen parents, report category deltas:

`incremental - whole_program`

for:
- entities;
- attributes;
- qualifiers;
- paths;
- operators;
- relation direction;
- dependencies;
- returns.

Do predicted-state and oracle-state separately.

Question: did incremental compilation simplify any local semantic subtask even though final accuracy fell?

## 19. GSM8K vs TAT-QA

Report taxonomy separately because LoRA-500 is 4/17 GSM8K vs 7/8 TAT-QA.

For every category include support count, not only percentage.

Compare:
- entity/path errors;
- qualifiers;
- relation direction;
- aggregation/cardinality;
- temporal;
- arithmetic;
- dependency;
- return.

## 20. Semantic-pattern families

Aggregate by existing leakage-safe tags:
- relative quantity;
- ratio/fraction;
- percentage;
- aggregation;
- rate;
- equal allocation;
- temporal shift;
- remaining/difference;
- multi-entity;
- chain;
- nested dependency.

Use this to target future teacher-data expansion.

## 21. Complexity analysis

Measure errors versus:
- NL part count;
- ASL statement count;
- dependency depth;
- entity count;
- semantic path count;
- operator count;
- unresolved dependency count.

Question: does failure come from individual grounding or compositional depth?

## 22. Stateful naming metric

Add:

```text
workspace_path_reuse_rate =
correct references to existing semantics using existing path
/
all references to existing semantics
```

Also report:
- accidental synonym creation;
- duplicate-slot creation;
- unresolved due to rename.

This is a primary metric for future persistent workspaces.

## 23. Teacher-data consistency

Apply naming analysis to accepted teacher programs.

Look for inconsistent conventions:
- `age_now` vs `current_age`;
- `total.clips` vs `clips_total`;
- `month1` vs `april`;
- different encodings of `each`.

Report:
- canonical-name diversity;
- likely synonym clusters;
- path-shape entropy for similar semantic relations.

Do NOT automatically rewrite training data without reviewed policy.

## 24. Outputs

Create local full artifacts:

```text
artifacts/paper1/semantic_failure_analysis_v1/
  whole_lora500.jsonl
  incremental_predicted.jsonl
  incremental_oracle.jsonl
  symbol_alignment.jsonl
  judge_queue.jsonl
  summary.json
  by_dataset.json
  by_pattern.json
  by_complexity.json
  error_examples.json
```

Track safe manifests/summaries/hashes according to existing repo policy.

## 25. Summary tables for Paper 1

Add a compact table with support and accuracy/F1 for:
- entity;
- attribute;
- qualifier;
- path reuse;
- operator;
- argument order;
- relation direction;
- source fact;
- dependency;
- aggregation/cardinality;
- temporal;
- return target.

Add a second table:
- whole LoRA-500;
- incremental predicted;
- incremental oracle.

Do not clutter the main paper with every submetric; put full taxonomy in appendix/artifacts.

## 26. Data-expansion decisions

Use diagnostics before selecting the next 500/1,000 originals.

Examples:
- low relation-direction accuracy -> oversample independently sourced comparative relations;
- low temporal qualifier accuracy -> oversample temporal-state problems;
- low path reuse -> add stateful multi-part examples;
- low aggregation/cardinality -> add each/total/group-size examples;
- arithmetic already strong -> do not waste teacher budget on trivial operator-only variants.

Preserve frozen test semantic-pattern exclusions.

## 27. Adapter-capacity interpretation

Run the planned QKVO-r16 and QKVO+MLP-r8 conditions, but interpret them with these metrics.

Examples:
- operator improves, paths do not -> more capacity is helping operation mapping but not grounding;
- entity/attribute/direction improve with MLP -> evidence semantic transformation benefits from MLP adaptation;
- all semantic metrics flat -> data/base-model understanding likely dominates.

Do not judge adapter changes only by final-answer accuracy.

## 28. Tests

Add deterministic regression tests for:
- wrong entity, right attribute;
- right entity, wrong attribute;
- qualifier-only mismatch;
- consistent standalone rename;
- forbidden stateful rename;
- relation reversal;
- argument reversal;
- wrong constant;
- premature literal collapse;
- transitive shortcut;
- each vs total;
- temporal shift;
- wrong return target;
- multi-label failure.

Tests must not require remote model calls.

## 29. Claim boundary

Current evidence supports:

> The dominant remaining Paper 1 failure is semantic grounding rather than syntax or exact arithmetic execution.

This analysis should determine *which semantic grounding components* dominate.

Do not claim object naming, attribute naming, relation direction, temporal reasoning, or any other subcategory is the primary cause until measured.

## 30. Immediate execution order

P0. Freeze current whole/incremental predictions.

P1. Implement deterministic AST/CCIR semantic decomposition.

P2. Add regression tests for every failure class.

P3. Score whole-program LoRA-500.

P4. Replay-score incremental predicted and oracle saved deltas.

P5. Produce GSM8K/TAT-QA and semantic-pattern breakdowns.

P6. Produce complexity analysis.

P7. Build ambiguous symbol/equivalence judge queue.

P8. Use strong judge only for unresolved semantic-equivalence cases.

P9. Freeze summary artifacts/hashes.

P10. Patch Paper 1 main text with compact results and appendix with taxonomy.

P11. Use measured weak categories to guide next data and LoRA-capacity runs.
