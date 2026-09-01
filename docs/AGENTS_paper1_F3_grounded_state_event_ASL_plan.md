# AGENTS — Paper 1 F3 Grounded State/Event ASL
## Design + 500-Example Experiment Plan
## Compare Better Representation vs More Data

## Purpose

Introduce a new Paper 1 representation condition, **F3**, motivated by the failure analysis of F0/F1/F2.

F3 must NOT be another cosmetic syntax around the same executable solution program.

F3 should test:

> The model may be better at translating natural language into grounded entities, state assertions, events, relations, and queries than at generating a complete solution program with storage paths, arithmetic lowering, dependency wiring, and explicit intermediates.

F3 therefore represents **what the source text asserts** while the runtime constructs canonical state, temporal history, dependency structure, deterministic lowering, derived facts, and query resolution.

Run this independently from the ongoing **F0 large-data scaling** effort generated through OpenRouter.

Central comparison:

```text
BETTER REPRESENTATION
F3 on ~500 audited examples

vs

MORE DATA
F0 on the large OpenRouter-generated corpus
```

Do not confound these axes.

## 1. Current evidence motivating F3

Frozen 0.6B checkpoint:

```text
F0 ASL: parse 88%, executable 76%, final 44%
F1 low-level functors: parse 68%, executable 64%, final 24%
F2 semantic functors: parse 60%, executable 60%, final 36%
```

Fine-grained F2:
- gold F0/F1/F2 runtime ceiling = 25/25;
- no unseen F2 test functors;
- class micro-F1 ≈ 40.5%;
- role accuracy ≈ 38.8%;
- exact binding ≈ 30.7%;
- direction ≈ 55.6%;
- query target 7/15 among parse-valid outputs.

F0 scaling from Qwen3-0.6B to Qwen3-1.7B improves final answers but does not clearly improve semantic-state/dependency quality.

Interpretation:

> The bottleneck is not primarily parser/lowering/runtime execution. Semantic selection, role binding, path/state construction, and dependency representation remain weak.

F3 should remove unnecessary solution-construction burden while preserving the genuinely neural semantic problem.

## 2. Design principle

Primary rule:

> **Represent what the text says, not the computation required to answer the question.**

Example:

NL:
```text
John removes 6 pink hats and twice that many green hats.
```

Do NOT require:
```text
john.green_removed = john.pink_removed * 2
green_remaining = green_initial - john.green_removed
```

Prefer:
```text
e1 := remove(john, hats.pink, 6)
remove(john, hats.green, 2 * e1.quantity)
```

Runtime owns the state transition implied by `remove`.

Likewise:

```text
Jessica is six years older than Claire.
```

Prefer:
```text
older_than(jessica.age@now, claire.age@now, 6 year)
```

rather than forcing:
```text
jessica.age_now = claire.age_now + 6
```

## 3. F3 semantic primitives

Keep a small domain-open upper layer:

```text
ENTITY
PATH
STATE
VALUE
RELATION
EVENT
COLLECTION
MEMBER
TIME
QUERY
```

Do not enumerate domain-specific nouns like `hat`, `invoice`, `gene`, etc.

Domain vocabulary remains open-world.

## 4. F3 surface language

Use a compact hybrid of paths, state assertions, functors, events, temporal qualifiers, and queries.

Initial grammar concept:

```text
statement :=
    path := expression
  | relation(args...)
  | event_id := event(args...)
  | event(args...)
  | entity_decl
  | collection_decl
  | ? expression
  | ? query_functor(args...)

expression :=
    literal
  | path
  | event_field
  | expression binop expression
  | functor(args...)
```

Do not over-engineer CFG details before auditing examples.

## 5. Paths remain first-class

Use hierarchical paths for:
- readable grounding;
- persistent identity;
- hierarchical structure;
- deterministic lookup;
- future typed-record/PRA compatibility.

Examples:

```text
hats.pink.count
hats.green.count
company.revenue@2019
claire.age@now
necklace.beads.amethyst.count
```

Prefer paths for persistent state, not anonymous `step_N` intermediates.

## 6. Events are first-class

State-changing language maps to events.

NL:
```text
Carl removes 4 pink hats.
```

F3:
```text
e1 := remove(carl, hats.pink, 4)
```

Runtime semantics:
```text
remove(actor, collection, q)
  requires collection.count
  effect collection.count := collection.count - q
```

Runtime stores before state, event, after state, provenance, and dependency.

The model should not generate `pink_remaining`.

## 7. Temporal history

Prefer event-sourced/versioned state.

Conceptually:

```text
hats.pink.count@t0 = 26
e1 = remove(...)
hats.pink.count@t1 = 22
```

The model need not emit runtime timestamps unless the source requires them.

Surface qualifiers may include:

```text
@now
@2019
@before(e1)
@after(e1)
```

only when semantically necessary.

## 8. Relations are declarative

Examples:

```text
older_than(jessica.age@now, claire.age@now, 6 year)
times_as_many(hats.green.removed_by(john),
              hats.pink.removed_by(john),
              2)
percent_less(month3.downloads, month2.downloads, 30)
member(necklace.beads, amethyst)
partition(necklace.beads, total=40)
```

A relation may remain a constraint, generate dependency edges, lower to arithmetic, invoke a CogCop, or become resolvable later.

The LLM does not choose the execution engine.

## 9. Collections and partitions

F3 should directly support cardinality/aggregation.

NL:
```text
A 40-bead necklace is made up of three kinds of beads.
```

F3:
```text
necklace.beads : partition
necklace.beads.total_count := 40
necklace.beads.category_count := 3
```

Then:
```text
member(necklace.beads, amethyst)
amethyst.count := 7

member(necklace.beads, amber)
times_as_many(amber.count, amethyst.count, 2)

member(necklace.beads, turquoise)

? turquoise.count
```

Runtime may infer:
```text
sum(member.count) = partition.total_count
```

and derive the missing count.

The model does not generate the subtraction program.

## 10. Query semantics

Keep queries close to language intent.

Examples:

```text
? hats.*.count@now
? sum(hats.*.count@now)
? jessica.age@now
? percentage_change(company.acquisition_cost, 2018, 2019)
```

For simple values, prefer `? path`.

Do not force all queries into `RETURN temp_var`.

## 11. Example: hats

NL:

```text
26 pink, 15 green, and 24 yellow hats.
Carl removes 4 pink hats.
John removes 6 pink hats and twice that many green hats.
How many remain?
```

F3:

```text
hats.pink.count := 26
hats.green.count := 15
hats.yellow.count := 24

e1 := remove(carl, hats.pink, 4)
e2 := remove(john, hats.pink, 6)
remove(john, hats.green, 2 * e2.quantity)

? sum(hats.*.count@now)
```

Runtime derives 16 + 3 + 24 = 43.

## 12. Example: age relation

NL:

```text
Jessica is six years older than Claire.
In two years, Claire will be 20 years old.
How old is Jessica now?
```

F3:

```text
older_than(jessica.age@now, claire.age@now, 6 year)
claire.age@(+2 year) := 20 year
? jessica.age@now
```

Runtime handles temporal age and difference constraints.

## 13. Example: TAT-QA

Question:
```text
What was the percentage change in Acquisition and integration costs in 2019 from 2018?
```

F3:

```text
company.acquisition_integration_cost@2019 := 17 million_usd
company.acquisition_integration_cost@2018 := 8 million_usd

? percentage_change(
    company.acquisition_integration_cost@2018,
    company.acquisition_integration_cost@2019
  )
```

This still requires correct evidence binding. F3 does not solve source grounding magically.

## 14. What F3 moves into runtime

Runtime owns more of:

```text
canonical symbol IDs
path normalization
event effects
temporal history
dependency edges
arithmetic lowering
aggregation lowering
query execution
engine selection
exact computation
```

Model still owns:

```text
entity grounding
attribute grounding
qualifier grounding
relation/event selection
argument roles
coreference
source fact attachment
query target
```

This is the intended scientific boundary.

## 15. F3 runtime

Implement:

```text
F3 parser
  ↓
semantic AST
  ↓
blackboard/state normalizer
  ↓
registered relation/event semantics
  ↓
dependency graph
  ↓
CogCop/solver
  ↓
derived state
  ↓
query
```

Gold F3 runtime ceiling should be near 100% before evaluating students.

## 16. Initial relation/event registry

Start bounded:

```text
remove
add
transfer
older_than
younger_than
times_as_many
fraction_of
percent_more
percent_less
member
partition
each
total
before
after
change
```

Do not add one predicate per benchmark pattern.

Prefer reusable primitives.

Track functor frequency, unseen test relations, and unsupported cases.

## 17. Generic fallback

Support a non-executable fallback such as:

```text
relation(name, args...)
event(name, args...)
```

or explicit:
```text
unsupported_relation(...)
```

Do not execute unknown relations.

Track fallback use separately.

## 18. F3 500-example corpus

Create a parallel F3 corpus from the existing frozen ~500 Paper 1 source records.

Target:
- preserve 450/25/25 source split if possible;
- quarantine unsafe examples;
- preserve all 25 test IDs whenever possible;
- never replace test records with easier alternatives.

Store:
- source ID;
- split;
- source hash;
- source parts/evidence references;
- F0 target;
- F3 target;
- F3 AST;
- runtime result;
- quality grade;
- annotation provenance;
- repair history;
- semantic-pattern signature.

## 19. Annotation must be answer-blind

Primary teacher input:
- raw NL problem/question;
- permitted table/source evidence;
- fixed F3 specification;
- fixed examples.

Do NOT provide:
- benchmark rationale;
- gold solution program;
- gold answer as semantic hint;
- F0 program as teacher input.

After generation:
- parse;
- semantic/type validate;
- execute;
- compare final result to benchmark answer only as validation.

If wrong:
- quarantine;
- or use explicitly labeled repair mode.

Preserve provenance.

## 20. Do not mechanically translate F0 to F3

F3 must be generated from source semantics.

Mechanical:
```text
F0 solution -> prettier F3
```

invalidates the experiment.

F0 is post-hoc comparison/audit only.

## 21. Teacher skill

Create a dedicated F3 annotation skill:

1. Represent source assertions, not solution steps.
2. Preserve entities/attributes.
3. Use state assertions for explicit facts.
4. Use events for state-changing verbs.
5. Use declarative relations for comparisons/ratios/percent relations.
6. Use collections/members/partitions when source asserts grouping.
7. Use event IDs for references like `that many`.
8. Do not derive unstated arithmetic solution steps.
9. Express question as query.
10. Prefer persistent semantic paths over anonymous steps.
11. Do not invent facts.
12. Emit `unsupported` rather than forcing a misleading encoding.

## 22. Canonical surface

Normalize surface variants into one semantic AST, but train the student on ONE canonical syntax initially.

Do not mix equivalent syntaxes in F3-v1 training.

## 23. Student training

First:

```text
Qwen3-0.6B
QKVO-r8
```

Use matched optimization to F0/F1/F2 where practical.

Input:
- fixed instruction;
- fixed ICL policy;
- no per-example retrieval;
- no runtime state as privileged input;
- raw problem/evidence only.

Then repeat with Qwen3-1.7B if warranted or in parallel if compute permits.

## 24. Whole vs sentence-local

Initial headline:
```text
F3-WHOLE:
full raw problem -> full F3 representation
```

Later:
```text
F3-SENTENCE:
each source sentence/part -> local F3 records
runtime integrates
```

Do not feed executed intermediate values to F3-SENTENCE.

## 25. F3 metrics

Add:

```text
event_class_accuracy
event_argument_role_accuracy
event_coreference_accuracy
state_path_accuracy
relation_class_accuracy
relation_argument_binding
temporal_qualifier_accuracy
collection_member_accuracy
partition_accuracy
query_target_accuracy
runtime_derived_dependency_f1
blackboard_f1
final_answer
```

Separate model-created semantic edges from runtime-derived dependency edges.

## 26. Cross-representation normalization

Normalize F0 and F3 into a common semantic graph where possible.

Do not require exact graph identity if F3 legitimately retains richer event semantics.

Report:
- query-relevant computational equivalence;
- semantic-state richness.

## 27. Runtime burden metric

For F0 and F3 estimate:

```text
model_semantic_decisions
runtime_derived_operations
runtime_derived_dependencies
```

Use as descriptive accounting.

## 28. F3 success criteria

Useful first positive checkpoint:
- gold runtime ceiling ~25/25;
- parse >= F2;
- event/relation class accuracy meaningfully above F2 class accuracy;
- argument binding improves over F2;
- dependency/blackboard F1 improves over F0/F2;
- final answer matches or beats F0;
- ideally GSM8K improves without severe TAT-QA regression.

## 29. Interpretation

### A — F3 > F0 with 500
Strong evidence representation matters.

### B — F3 semantic state > F0 but final <= F0
Partial success; inspect query/runtime integration.

### C — F3 ≈ F0
Representation roughly equivalent; compare data scaling.

### D — F3 < F0 and binding remains weak
Bottleneck likely earlier NL→semantic-role binding.

## 30. Parallel F0 large-data track

Keep OpenRouter F0-large generation running independently.

Track A — representation:
```text
F0-500
F1-500
F2-500
F3-500
```

Track B — data:
```text
F0-500
F0-large
```

Only create F3-large later if F3-500 warrants it.

## 31. Critical matrix

Eventually:

```text
                    ~500 data      large data
F0 simple ASL          ✓              ✓
F3 grounded ASL        ✓              later
```

Questions:
- `F3-500 vs F0-500` → representation
- `F0-large vs F0-500` → data scaling
- `F3-500 vs F0-large` → can better representation compensate for much more supervision?

## 32. Learning curves

Where feasible compare:

```text
50
100
250
500
```

Use unique semantic signatures as x-axis as well as row count.

Do not regenerate test set.

## 33. Overfitting controls

For F3:
- evaluate each epoch;
- checkpoint each epoch;
- use dev semantic metrics;
- early stop;
- consider lower LR if needed;
- do not select by test.

Do not add auxiliary losses in the first F3-vs-F0 comparison.

## 34. Later training extensions

Only after baseline:
- event/relation class auxiliary loss;
- argument-role loss;
- path/entity loss;
- query-target loss;
- controlled hard negatives.

First isolate representation.

## 35. Hard negatives

Later examples:
- swapped actor/object;
- reversed relation;
- wrong attribute;
- wrong temporal qualifier;
- wrong event coreference;
- each vs total;
- wrong table row;
- wrong query target.

## 36. TAT-QA care

Do not force events for static table data.

Prefer:
```text
company.metric@period := value
? percentage_change(...)
```

This is intended to avoid F2's TAT-QA regression.

## 37. GSM8K care

Use events for:
- give;
- remove;
- buy;
- sell;
- lose;
- gain;
- transfer;
- consume;
- produce.

Use relations for:
- older/younger;
- more/fewer;
- times as many;
- percentages;
- rates;
- partitions.

Do not force all GSM8K into event form.

## 38. Generality audit

Audit a small held-out sample from:
- CLUTRR;
- ProofWriter;
- dates/units;
- simple graph statements.

Do not train on this for the first Paper 1 headline.

Goal: ensure F3 has not become a GSM8K-specific ontology.

## 39. Paper 2 connection

If F3 works, Paper 2 can extend the same state/event/relation substrate to:
- calculator;
- Datalog;
- graph;
- dates;
- units;
- retrieval;
- verification.

Avoid arithmetic-only assumptions.

## 40. Long-context connection

F3 is a candidate payload for later sparse semantic blocks:

```text
detect block
→ activate semantic adapter
→ generate F3
→ update blackboard
→ continue NL
```

Do not mix block-detection experiments into F3-v1.

## 41. Required artifacts

Create:

```text
docs/papers/paper1/AGENTS_F3_grounded_asl.md

src/ccpu/paper1/f3/
  grammar.py
  parser.py
  ast.py
  registry.py
  runtime.py
  normalize.py
  dataset.py
  evaluator.py

configs/paper1/
  f3_annotation.schema.json
  f3_qwen_0_6b_lora_500_xpu.json
  f3_qwen_1_7b_lora_500_xpu.json

artifacts/paper1/f3_v1/
  data/
  manifests/
  audits/
  runs/
  analysis/
```

Follow existing ignore/provenance policy.

## 42. Tests

Add deterministic tests for:
- state assertion;
- event parsing;
- event field reference;
- event state effect;
- sequential events;
- relation lowering;
- temporal state;
- partition inference;
- query;
- unresolved relation;
- unsupported relation;
- scope isolation;
- TAT-QA static metric;
- gold runtime ceiling.

No remote calls in unit tests.

## 43. Milestone 1 — representation

Before LoRA:
1. Freeze F3 grammar v1.
2. Implement parser/AST/runtime.
3. Manually author 20–30 representative examples.
4. Ensure runtime solves them.
5. Define teacher skill.
6. Generate/audit first 100 examples.
7. Verify targets represent source assertions, not solution plans.
8. Scale to ~500.
9. Verify gold runtime ceiling on all accepted test rows.

Do not train if this audit fails.

## 44. Milestone 2 — F3-500 0.6B

Train Qwen3-0.6B on F3-500.

Run identical frozen 25 test IDs.

Produce:

```text
F0-500
F1-500
F2-500
F3-500
```

with parse, class, roles, binding, dependency, blackboard, final.

## 45. Milestone 3 — 1.7B

Repeat F3-500 on Qwen3-1.7B.

Compare representation × model-size interaction.

Do not move directly to 4B.

## 46. Milestone 4 — data vs representation

When F0-large is complete, train/evaluate F0-large.

Compare:

```text
F0-500
F3-500
F0-large
```

This is the central **more data vs better representation** experiment.

## 47. F3-large gate

Only build a large F3 corpus if one or more hold:
- F3-500 beats F0-500 final;
- F3 materially improves dependency/blackboard semantics;
- F3 shows better sample efficiency;
- F3 scales more strongly with 1.7B;
- F3 fixes GSM8K relational failures without unacceptable TAT-QA loss.

Otherwise do not spend large teacher budget.

## 48. Paper framing

Positive framing:

> F0 asks the model to translate natural language into an executable solution program. F3 instead asks it to represent grounded state, events, relations, and queries while a deterministic cognitive runtime constructs state transitions and executable dependencies. The comparison tests whether semantic representation design can substitute for additional supervision/model capacity.

Negative framing:

> Richer grounded/event structure does not improve semantic compilation at this model/data scale; the dominant bottleneck remains language-to-role binding.

Both outcomes are useful.

## 49. Claim boundary

Do NOT claim:
- physical grounding;
- general ontology;
- universal cognitive language;
- solved symbolic reasoning;
- F3 superiority from gold runtime alone.

The first claim is only whether F3 improves learnability, semantic-state quality, and answer accuracy on frozen Paper 1 tasks.

## 50. Immediate execution order

P0. Keep OpenRouter F0-large generation running independently.

P1. Create F3 design/grammar docs and runtime skeleton.

P2. Manually validate 20–30 examples.

P3. Create teacher annotation skill.

P4. Generate answer-blind F3 targets for first 100 existing source records.

P5. Audit representation quality; revise grammar once if necessary and version it.

P6. Generate full ~500 F3 corpus preserving frozen test IDs.

P7. Gold runtime ceiling audit.

P8. Train Qwen3-0.6B QKVO-r8 F3-500.

P9. Run common semantic metrics.

P10. Compare F0/F1/F2/F3 on 0.6B.

P11. Train/evaluate F3-500 on Qwen3-1.7B.

P12. Compare model-size interaction.

P13. When F0-large is ready, train/evaluate F0-large.

P14. Produce explicit representation-vs-data scaling comparison.

P15. Decide whether F3-large is justified.

P16. Patch Paper 1 manuscript, appendix, artifacts, and conclusions.
