# AGENTS ADDON — Paper 1 Functor vs ASL Evaluation Metrics
## Fine-Grained Comparison of F0 Simple ASL, F1 Low-Level Functors, and F2 Semantic Functors

## Purpose

Add a dedicated evaluation layer for the Paper 1 representation comparison:

- **F0 — Simple semantic ASL**
- **F1 — Low-level functors isomorphic to ASL**
- **F2 — Semantic functors with runtime lowering / blackboard integration**

The goal is to determine why one representation works better than another, not only which one returns more correct answers.

Current frozen 0.6B checkpoint:

```text
F0 ASL:
  parse      22/25 = 88%
  executable 19/25 = 76%
  final      11/25 = 44%

F1 low-level functor:
  parse      17/25 = 68%
  executable 16/25 = 64%
  final       6/25 = 24%

F2 semantic functor:
  parse      15/25 = 60%
  executable 15/25 = 60%
  final       9/25 = 36%
```

Dataset split:
- GSM8K: F0 4/17, F1 2/17, F2 5/17
- TAT-QA: F0 7/8, F1 4/8, F2 4/8

F1/F2 corpus:
- 483/500 paired annotations accepted
- 434 train
- 24 dev
- all 25 frozen test records
- 17 train/dev records quarantined
- F0 has 450 train records, so F1/F2 have a 16-record training disadvantage

Do not reinterpret existing results. Add diagnostics over frozen predictions wherever possible.

## 1. Central questions

The evaluation must answer separately:

1. Is F1 worse because functor syntax is harder than ASL?
2. Is F2 better than F1 because semantic decomposition helps?
3. Is F2 still worse than F0 because semantic functor classification is difficult, argument binding is difficult, syntax reliability is lower, runtime lowering loses information, the vocabulary is too restrictive, the 0.6B model is too small, the training corpus is smaller, or TAT-QA simply fits ASL better?
4. Does a larger model benefit disproportionately from F2?
5. Does F2 construct a better reusable blackboard even when final-answer accuracy is similar?

## 2. Preserve matched evidence

Use identical frozen test IDs for F0, F1, and F2.

For larger-model replication use:
- same 25 test IDs
- same F1/F2 teacher targets
- same decoding where possible
- same evaluator
- same prompt contract
- same fixed ICL policy if ICL is used

Do not compare models using different test subsets.

Where F0 has 450 train and F1/F2 have 434, report this explicitly.

If feasible, add a matched F0-434 control later by dropping the corresponding 16 F0 training rows, but do not block the main analysis.

## 3. Evaluation layers

Score each condition at five levels.

### L0 — Surface language validity

For F0:
- ASL parse validity
- AST construction
- canonical statement count
- unknown operator/token rate

For F1/F2:
- functor parse validity
- balanced delimiters
- registered-functor syntax
- arity validity
- named/positional argument syntax

Report:
- parse rate
- malformed constructs/example
- invalid token/functor rate

### L1 — Structural semantic validity

Normalize all representations into one common semantic structure:
- entities
- attributes
- qualifiers
- literals
- relation/functor type
- argument roles
- direction
- dependency edges
- query/return target

F0 derives this from ASL/CCIR.
F1 derives it from low-level functors.
F2 derives it directly from semantic functors before runtime lowering.

All three must be scored against the same semantic reference.

### L2 — Runtime lowerability

Measure:
- lowerable rate
- type-valid rate
- registered-functor rate
- unresolved-functor rate
- unsupported-relation rate
- ambiguous-lowering rate

For F2 also report:
- deterministic lowering coverage
- runtime operations per functor
- runtime-created canonical slots
- unresolved constraints

### L3 — Executed semantic state

After deterministic execution/integration compare:
- semantic-state equivalence
- entity-state F1
- attribute-state F1
- value-attachment F1
- dependency F1
- canonical-slot correctness
- query-target correctness
- blackboard completeness
- spurious-state rate

A correct answer with poor state is not semantic-state success.

### L4 — Final result

Report:
- executable
- final-answer accuracy
- exact answer
- fail-closed/abstain rate

Also:
- P(answer correct | semantic structure correct)
- P(answer correct | semantic structure incorrect)
- P(answer correct | runtime state correct)
- P(answer correct | runtime state incorrect)

## 4. F0-specific metrics

Reuse current Paper 1 analyzer:
- path exact F1
- entity F1
- attribute F1
- qualifier F1
- operator F1
- relation participant accuracy
- relation direction
- argument order
- source literal F1
- source-fact attachment
- dependency F1
- premature literal collapse
- return target
- semantic-state equivalence

## 5. F1-specific metrics

F1 is intended to be isomorphic to ASL.

Measure:
- low-level functor parse
- vocabulary accuracy
- arity
- nesting depth
- malformed nesting

Deterministically map F1 to F0-equivalent CCIR and compare:
- target path
- operator
- source references
- dependency graph
- return target

Define:
`f1_asl_isomorphic_accuracy`

If F1 is worse than F0 but normalized semantic metrics are similar, the loss is mostly surface syntax.
If semantic metrics are also worse, functorization itself hurts generation.

## 6. F2-specific metrics

### Functor class

Examples:
- older_than
- times_as_many
- percent_less
- partition
- each
- count
- query_count
- date_add
- convert

Metrics:
- micro F1
- macro F1
- confusion matrix
- unsupported-functor rate

### Argument roles

Measure independently:
- subject
- object/source
- target
- attribute
- qualifier
- amount/constant
- group/member

Report:
- `argument_role_accuracy`
- `argument_binding_exact`

A correct functor with swapped arguments is wrong.

### Direction

For asymmetric relations:
- older/younger
- more/less
- before/after
- increase/decrease
- source/target conversion

Report relation-direction accuracy independently.

### Attribute grounding

Report attribute F1 before runtime lowering.

### Qualifiers

Report qualifier F1 for:
- now/future
- each/total
- remaining
- period/month
- status such as returned/damaged

### Query functor

Measure:
- query functor class
- query entity
- query attribute
- query qualifier
- final query target

## 7. Runtime-created state metrics for F2

Per test example report:
- predicted semantic functors
- canonical slots created by runtime
- arithmetic/logic operations generated by lowering
- dependency edges generated by runtime
- derived values
- unresolved constraints
- runtime-only state records

Define:
- `model_semantic_decisions`
- `runtime_lowered_decisions`

Estimate the fraction of final dependency graph supplied by model vs deterministic registry/lowering.

## 8. Blackboard quality

For F2 add:
- canonical entity count
- canonical slot count
- duplicate entity rate
- duplicate slot rate
- unresolved symbol rate
- conflicting fact rate
- dependency completeness
- derived-fact correctness
- spurious derived-fact rate
- query resolution success

Define:
- `blackboard_precision`
- `blackboard_recall`
- `blackboard_f1`

over normalized semantic facts/relations.

## 9. Representation cost

Compare:
- prompt tokens
- generated tokens
- semantic statements/functors
- tokens per semantic relation
- tokens per executable operation
- tokens per correct answer
- tokens per correct semantic-state record

For F2 also:
- runtime-generated CCIR size
- blackboard serialized size if materialized

## 10. Vocabulary burden

For each condition report:
- distinct syntax/operator tokens
- distinct path segments
- distinct functor names
- distinct attribute names
- distinct qualifier names

For F2:
- registered vocabulary size
- train functors
- dev-only functors
- test-only functors
- unseen-functor rate

## 11. Functor frequency and long tail

Bucket F2 functors:
- frequent
- medium
- rare
- unseen

Report by bucket:
- functor-class accuracy
- argument binding
- final answer

This tests whether F2 works only for commonly seen semantic relations.

## 12. Common semantic normalization

Build one common canonical object.

Example:

```json
{
  "relation": "relative_difference",
  "target_entity": "jessica",
  "target_attribute": "age",
  "target_qualifier": "now",
  "source_entity": "claire",
  "source_attribute": "age",
  "source_qualifier": "now",
  "direction": "target_greater",
  "constant": 6
}
```

Map:

F0:
`jessica.age_now = claire.age_now + 6`

F1:
`set(jessica.age_now, add(claire.age_now,6))`

F2:
`older_than(jessica,claire,age,now,6)`

into the same comparison object.

Use deterministic normalization first. Use a strong semantic judge only for unresolved naming equivalence.

## 13. Paired F0/F1/F2 analysis

For each identical test ID classify:
- F0 correct/F1 correct/F2 correct

Count:
- F2 rescues F0 failures
- F0 rescues F2 failures
- F2 rescues F1 failures
- F1 rescues F2 failures
- all correct
- all wrong

For each flip, record semantic error categories.

Inspect GSM8K cases rescued by F2 and TAT-QA regressions explicitly.

## 14. GSM8K vs TAT-QA

Report all metrics separately.

For each dataset add:
- average relation count
- average functor count
- average path count
- average dependency depth
- average target tokens
- functor-family distribution

Test whether F2's benefit is specifically relational while TAT-QA favors direct value/path representation.

## 15. Syntax-vs-semantics decomposition

Calculate:
- semantic accuracy given parse
- semantic accuracy given lowerable
- final accuracy given semantic correct

If F2's raw result is low mainly because parse is 60% but parsed outputs have strong semantics, syntax is the main issue.
If parsed outputs still have poor roles/binding, semantic representation is the issue.

## 16. Model-size scaling

For 0.6B and larger model compute:

- ΔF0
- ΔF1
- ΔF2

for:
- parse
- functor/operator class
- argument binding
- attribute
- dependency
- blackboard F1
- final answer

Key question:
`Does F2 benefit more from model capacity than F1/F0?`

## 17. Larger-model F0 control

Do not compare only larger-model F1/F2.

Required matrix where feasible:

```text
             F0     F1     F2
0.6B         yes    yes    yes
larger       yes    yes    yes
```

This allows exploratory representation × model-size interaction analysis.

## 18. Overfitting metrics

Current F1/F2 train loss near 0.02 and dev around 0.46–0.49 requires epoch-level evaluation in future runs.

Record each epoch:
- train loss
- dev token loss
- dev parse
- dev semantic structure
- dev dependency
- dev final accuracy where affordable

Select checkpoint using a prespecified semantic dev score, not minimum train loss.

Do not tune selection formula on test.

## 19. Training-size control

Optional:
train `F0-434` using the exact source IDs available to F1/F2.

Keep F0-450 as the main existing baseline.

This estimates whether 16 extra F0 examples explain part of the gap.

## 20. Target complexity

Before training compare F0/F1/F2:
- target tokens
- AST nodes
- functor count
- mean arity
- max nesting
- distinct symbols
- distinct paths
- relation count

If F2 targets are longer or more complex, report that as a possible syntax confound.

## 21. Runtime lowering ceiling

For F2 run:
`gold F2 functors -> runtime -> answer`

Report:
- lowering success
- execution
- final answer

Likewise:
- gold F0 ASL -> runtime
- gold F1 functors -> runtime

The three gold representations should ideally have the same deterministic answer ceiling on paired test records.

Any failure on gold functors is runtime/coverage error, not student semantic failure.

## 22. Unsupported/open relations

For F2 track:
- registered relation
- unknown relation
- generic fallback
- unsupported relation
- ambiguous functor

Report coverage:
`supported_test_relations / all_test_relations`

Do not count unsupported gold semantics as student errors without separating coverage.

## 23. Error taxonomy

### Surface
- malformed syntax
- unknown functor
- wrong arity

### Semantic
- wrong functor class
- right functor/wrong roles
- wrong direction
- wrong constant
- wrong attribute
- wrong qualifier
- missing functor
- extra functor

### Blackboard
- duplicate symbol
- unresolved entity
- wrong canonical slot
- missing dependency
- spurious dependency
- conflict

### Query
- wrong query functor
- wrong target
- missing query

Use multi-label accounting.

## 24. Functor confusion matrix

Produce examples like:
- older_than -> less_than
- times_as_many -> difference
- percent_less -> times_as_many
- each -> total
- partition -> count
- query_count -> query_value

## 25. Argument-role confusion

For asymmetric functors report:
- subject <-> object
- target <-> source
- group <-> member
- base <-> derived
- from_unit <-> to_unit
- before <-> after

This is a primary F2 diagnostic.

## 26. Semantic burden accounting

For each representation list whether the model predicts:
- entity identity
- attribute identity
- qualifier
- relation class
- argument roles
- storage path
- arithmetic lowering
- dependency wiring
- query target
- engine identity

Expected:

F0/F1 model predicts most of these.

F2 should leave storage layout, much arithmetic lowering, and much dependency wiring to runtime.

Use this as interpretation, not a score.

## 27. Efficiency metrics

Report:
- semantic success per generated token
- final success per generated token

Optionally:
- semantic success per trainable parameter

for matched models.

## 28. Statistical treatment

With 25 test examples:
- report exact counts first
- Wilson intervals
- paired exact/binomial or McNemar only as exploratory

For relation-level metrics report both micro and per-program macro values.

Do not treat all relation observations as independent for significance claims.

## 29. Artifacts

Create:

```text
artifacts/paper1/functor_v1/analysis_v2/
  f0_normalized.jsonl
  f1_normalized.jsonl
  f2_normalized.jsonl
  cross_representation.jsonl
  paired_flips.json
  by_dataset.json
  by_functor.json
  argument_roles.json
  confusion_matrix.json
  blackboard_metrics.json
  runtime_lowering.json
  representation_cost.json
  model_size_comparison.json
  summary.json
```

Track safe summaries/hashes according to repository policy.

## 30. Paper tables

Main table:

```text
Condition | Parse | Semantic Struct. | Dependency | Blackboard | Exec | Final
```

Appendix table:

```text
Condition | Entity | Attribute | Qualifier | Functor/Operator | Roles | Direction | Query
```

Add model-size table when larger-model results complete.

## 31. Interpretation gates

### Syntax bottleneck
F2 semantic metrics conditional on parse are strong, but raw performance is dragged down by parse failures.

### Semantic-functor bottleneck
F2 parses but functor/argument metrics remain weak.

### Runtime bottleneck
Gold F2 functors fail deterministic lowering/execution.

### Model-capacity bottleneck
Larger model improves F2 much more than F0/F1.

### Representation success
F2 improves semantic structure/dependency/blackboard quality and at least matches final accuracy.

### Partial success
F2 improves semantic structure but not final answer.

### Representation failure
F2 remains worse in semantic structure and final answer after syntax/runtime controls.

## 32. Immediate execution order

P0. Freeze current 0.6B F0/F1/F2 predictions.

P1. Implement common semantic normalization.

P2. Add F1 isomorphism metrics.

P3. Add F2 functor-class and argument-role metrics.

P4. Add gold-runtime ceiling audits.

P5. Add blackboard precision/recall/F1.

P6. Add representation token/complexity metrics.

P7. Produce GSM8K/TAT-QA breakdown.

P8. Produce paired flip analysis.

P9. Integrate larger-model F1/F2 results.

P10. Run larger-model F0 matched control.

P11. Add model-size × representation comparison.

P12. Optionally run F0-434 training-size control.

P13. Update Paper 1 manuscript, appendix, artifacts, and claims.

## 33. Claim boundary

Current 0.6B evidence supports only:

> Semantic functors improve over low-level functors on final-answer accuracy, especially on the small GSM8K subset, but do not beat simple ASL overall and do not yet produce reliable reusable semantic state.

This addon should determine whether that deficit is caused by:
- syntax
- semantic functor selection
- argument binding
- blackboard integration
- representation coverage
- data imbalance
- or model capacity.

Do not claim the blackboard/functor representation is superior until these are measured.
