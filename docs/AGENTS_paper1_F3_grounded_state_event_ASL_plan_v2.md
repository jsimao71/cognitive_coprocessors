# Paper 1 F3 Grounded State/Event ASL v2

## Status

Implementation plan for a matched 500-record Paper 1 representation experiment.
This document supersedes the v1 proposal for execution, while preserving v1 as
the design history.

## Research question

Can a small model map natural language more reliably into grounded observations,
events, relations, and query intent than into a complete executable solution
program?

F3 represents source assertions. A bounded deterministic runtime supplies event
effects and, in a separately reported condition, constraint closure. F3 does not
receive answers, rationales, executed state, per-example demonstrations, or
future information.

## Primary controls

Use the exact frozen Paper 1 source identities and split:

```text
450 train / 25 dev / 25 test
```

No test record may be replaced or dropped. An unsupported test record is a
failed F3 record. If a training record is quarantined, train a matched F0 subset
control and report both nominal and retained counts.

Run:

```text
F0-500          existing ASL
F1-500          low-level functors
F2-500          semantic solution functors
F3-R0-500       observations and direct query lowering only
F3-R1-500       R0 plus registered local event effects
F3-R2-500       R1 plus registered constraint closure
```

F3-R1 is the primary representation condition. F3-R2 is a system condition and
must not be described as a pure representation gain.

## Canonical F3-v1 surface

F3-v1 is one allowlisted call per line. It permits string and numeric literals
and allowlisted nested calls. It forbids arbitrary Python, keyword arguments,
assignments, binary operators, comprehensions, and free-form predicates.

The final line is exactly one `query(...)`.

### References and evidence

```text
at("path", "time")
event_field("event_id", "quantity")
scale(reference, factor)
fraction(reference, numerator, denominator)
source("exact short source span")
cell("exact row label", "exact column label")
```

Evidence text must occur in the supplied source. Table row and column labels
must match supplied table labels after deterministic whitespace normalization.

### Source assertions

```text
collection("path", "member_kind", "location", evidence)
observe(at("path", "time"), value, "unit", evidence)
same(subject, reference, evidence)
offset(subject, reference, delta, "unit", evidence)
multiple(subject, reference, factor, evidence)
fraction_of(subject, reference, numerator, denominator, evidence)
difference_relation(subject, left, right, evidence)
absolute_difference(subject, left, right, evidence)
quotient_relation(subject, numerator, denominator, evidence)
percent_of(subject, reference, percentage, evidence)
percent_more(subject, reference, percentage, evidence)
percent_less(subject, reference, percentage, evidence)
sum_relation(total, part1, part2, ..., evidence)
product_relation(total, factor1, factor2, ..., evidence)
rate_relation(total, rate, duration, evidence)
mean_relation(mean, value1, value2, ..., evidence)
minimum_relation(minimum, value1, value2, ..., evidence)
maximum_relation(maximum, value1, value2, ..., evidence)
member("collection_path", "member_path", evidence)
partition("collection_path", total_reference, evidence)
```

The first argument of a relation is the semantic subject, not an anonymous
calculation step. Relations may be solved in either direction only in F3-R2.

### Events

```text
remove("event_id", "actor", state_reference, quantity, evidence)
add("event_id", "actor", state_reference, quantity, evidence)
consume("event_id", "actor", state_reference, quantity, evidence)
produce("event_id", "actor", state_reference, quantity, evidence)
transfer("event_id", "actor", source_reference, destination_reference,
         quantity, evidence)
```

The runtime records before state, event, after state, quantity provenance, and
dependency edges. Source order defines event order unless the source states an
explicit temporal relation.

### Intent-level queries

```text
query("value", reference)
query("remaining_count", "collection_path", "time")
query("sum", "collection_path", "time")
query("difference", reference_from, reference_to)
query("absolute_difference", reference_a, reference_b)
query("percentage_ratio", part_reference, whole_reference)
query("percentage_change", reference_from, reference_to)
```

The student must represent query intent, not insert an unstated aggregation
program.

## Canonical examples

### Hats

```text
collection("truck.inventory.hats", "hat", "truck", source("26 pink, 15 green, and 24 yellow hats"))
observe(at("truck.inventory.hats.pink.count", "initial"), 26, "count", source("26 pink"))
observe(at("truck.inventory.hats.green.count", "initial"), 15, "count", source("15 green"))
observe(at("truck.inventory.hats.yellow.count", "initial"), 24, "count", source("24 yellow hats"))
remove("e1", "carl", at("truck.inventory.hats.pink.count", "current"), 4, source("Carl removes 4 pink hats"))
remove("e2", "john", at("truck.inventory.hats.pink.count", "current"), 6, source("John removes 6 pink hats"))
remove("e3", "john", at("truck.inventory.hats.green.count", "current"), scale(event_field("e2", "quantity"), 2), source("twice that many green hats"))
query("remaining_count", "truck.inventory.hats", "current")
```

The query does not say `sum(...)`; aggregation is part of the registered
`remaining_count` query semantics.

### Age

```text
older_than(at("jessica.age", "now"), at("claire.age", "now"), 6, "year", source("Jessica is six years older than Claire"))
observe(at("claire.age", "plus_2_year"), 20, "year", source("In two years, Claire will be 20 years old"))
query("value", at("jessica.age", "now"))
```

Age progression is an explicit registered schema. It is runtime knowledge and
must be included in the registry manifest and runtime-ablation accounting.

### TAT-QA

```text
observe(at("company.acquisition_integration_cost", "2019"), 17, "million_usd", cell("Acquisition and integration costs", "2019"))
observe(at("company.acquisition_integration_cost", "2018"), 8, "million_usd", cell("Acquisition and integration costs", "2018"))
query("percentage_change", at("company.acquisition_integration_cost", "2018"), at("company.acquisition_integration_cost", "2019"))
```

Selecting the `Total` row is an evidence-binding error even if the arithmetic is
otherwise correct.

## Runtime boundary

The model owns:

```text
entity and attribute selection
local symbol/path consistency
evidence selection
event/relation class
argument roles and direction
coreference
temporal qualifier
collection membership
query intent and target
```

The deterministic runtime owns:

```text
syntax and type validation
alpha-renamable symbol IDs
event history and local effects
registered relation lowering
registered query lowering
exact arithmetic
scope isolation
```

Only F3-R2 owns inverse relation solving and constraint closure. Every runtime
schema is frozen before test annotation and listed with a hash. Unknown
relations fail closed.

## Annotation protocol

Teacher input contains only:

```text
raw question/problem
permitted source/table evidence
fixed F3 specification
one fixed ICL set
```

Teacher input never contains the benchmark answer, rationale, F0/F1/F2 target,
executed values, blackboard state, or validator-derived numeric hints.

Validation order:

```text
schema -> parse -> evidence -> type -> lower -> execute -> answer check
```

Syntax, evidence-shape, and type failures may be retried with categorical error
codes. An answer mismatch is quarantined and must not return the expected answer,
numeric delta, failed intermediate state, or executed values to the teacher.

Use two separately reported retry tiers. Tier 1 regenerates from raw source with
all prior programs and validator feedback hidden. If that saturates, Tier 2 may
show the teacher its own latest F3 draft plus one coarse failure class
(`parse_invalid`, `evidence_invalid`, `not_lowerable`, `type_invalid`, or
`answer_mismatch`). Tier 2 still hides answers, rationales, expected/actual
values, numeric deltas, ASL/F0/F1/F2 targets, blackboard state, detailed
validator errors, and execution values. Record `prior_programs_hidden=false`
for Tier 2 and report how many retained labels require it.

Annotations are generated from raw source semantics, never by translating F0 or
F2. F0/F2 are available only to a post-hoc auditor after the F3 target is frozen.

Store source ID/hash, split, F3 text/AST, evidence anchors, runtime modes,
runtime result, answer-check status, quality grade, teacher/model/prompt hashes,
repair history, and normalized semantic signature.

## Dataset gates

Before scaling beyond the pilot:

1. Implement parser, AST, registry, runtime modes, and evaluator.
2. Audit 25 train-only records including both datasets before touching frozen
   dev/test labels.
3. Require deterministic tests for every primitive and failure mode.
4. Require at least 90% execution-and-answer acceptance on the answer-blind
   pilot after raw-only retries; freeze and report unsupported ontology classes.
5. Generate 100 answer-blind targets.
6. Audit source-assertion fidelity and registry pressure.
7. Freeze grammar/registry v1 once; subsequent changes create F3-v2.
8. Generate all 500 exact identities.
9. Require every frozen test identity to have either a validated F3 target or a
   frozen unsupported status. Unsupported rows remain in the denominator and
   count as final-answer failures; they are never silently dropped.

Do not add one predicate per failed benchmark pattern. Report registry coverage,
fallback pressure, unsupported cases, and unseen test primitives.

### Implemented pilot audit

The train-only 25-record pilot used the fixed local teacher prompt and synthetic
ICL, with answers, rationales, prior targets, runtime state, and validator values
hidden. Primary generation accepted 12/25. Representation-preserving runtime
fixes raised the unchanged labels to 16/25, and two raw-input-only retries raised
the union to 23/25 (92%). The accepted set contains 22 unique normalized F3
signatures.

The pilot froze two unsupported ontology classes rather than adding benchmark
facts to the runtime:

```text
US coin-denomination plus dollar/cent conversion
minute/hour conversion
```

F3-v1 therefore includes symbolic unknown event quantities, open-world transfer
projection, time-aware collections/partitions, numeric path canonicalization,
direct mean query intent, and deterministic query-connected graph slicing so
unrelated document facts do not require artificial numerical closure. The full
asserted graph remains available for semantic scoring. F3-v1 does not include a
physical-unit or currency ontology. A later F3-O condition may add a fixed,
hashed ontology as a separate runtime ablation; it must not be folded into the
primary F3 representation row.

## Training protocol

Primary training uses Qwen3-0.6B and Qwen3-1.7B QKVO-r8 with the same revisions,
seed, batch size, accumulation, learning rate, decoding, and fixed ten epochs as
the matched F0 controls.

Do not give only F3 early stopping. A secondary dev-selected comparison may be
run only by applying the same checkpoint policy to all compared conditions.

Report rows, unique semantic signatures, target tokens, optimizer steps, peak
memory, and wall time. Add a token/step-matched control if F3 target volume
differs materially from F0.

Learning curves use nested semantic-family subsets at 50/100/250/all retained
training rows. The 25 test identities never change.

## Evaluation

Primary paired metrics on the identical test identities:

```text
parse / evidence-valid / type-valid / lowerable / executable
common source-fact precision, recall, F1
common entity/attribute/qualifier/role/dependency F1
common query-target accuracy
common semantic-state equivalence
final-answer execution
paired outcome counts and exact exploratory test
```

F3-specific diagnostics:

```text
event class and role accuracy
event coreference
relation class and direction
evidence anchor accuracy
temporal qualifier accuracy
collection/member accuracy
model-created graph F1
runtime-derived graph F1 conditional on correct model assertions
```

Do not compare raw F3 event-class accuracy directly with F2 functor-class
accuracy. The primary semantic comparison is the shared normalized graph.

Because 25 tests move in four-point increments, report exact counts and Wilson
intervals and label significance exploratory. If F3 passes its first gate,
freeze a larger confirmatory matched evaluation before making a strong claim.

## Interpretation gates

```text
F3-R1 > F0 semantic graph and final answer
  evidence that event/state representation improves learnability

F3-R2 > F3-R1 only
  evidence that solver closure helps the system, not representation alone

F3 semantic graph > F0 but final answer <= F0
  representation gain with query/runtime integration weakness

F3 <= F0 with weak role binding
  language-to-role binding remains the dominant bottleneck
```

Do not claim physical grounding, a general ontology, or universal reasoning.

## Execution order

```text
P0  keep OpenRouter F0-large generation separate
P1  implement F3 parser/AST/registry/runtime/evaluator
P2  add deterministic tests and 25-record authored audit
P3  create fixed teacher skill and request schema
P4  generate and audit 100 raw-source targets
P5  freeze grammar/registry and generate exact F3-500
P6  build matched SFT/eval data and runtime manifests
P7  train/evaluate 0.6B F3
P8  train/evaluate 1.7B F3
P9  compare common F0/F1/F2/F3 metrics and runtime ablations
P10 patch Paper 1, rebuild PDF, commit, and push each major milestone
```
