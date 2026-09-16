# AGENTS ADDON — Paper 2 Blackboard/Functor Architecture

## Purpose

Patch `docs/papers/paper2/AGENTS_paper2.md` without changing or reinterpreting existing Paper 2 evidence.

Paper 2 currently tests explicit multi-engine execution: the model selects/serializes calculator, Datalog, graph, date, and unit requests while parsing, validation, execution, state, provenance, reinjection, and enforcement remain in the runtime.

Add a second architectural line motivated by Paper 1:

1. First reproduce a **Paper-1-style semantic ASL baseline** across Paper 2 engine families.
2. Then test a **blackboard + functor architecture** where the model posts compact semantic relations to a shared typed workspace and the runtime resolves symbols, chooses capabilities, executes registered functors, and posts derived state back to the blackboard.

Do not mutate frozen Paper 2 artifacts. Treat this as a new versioned experiment.

---

## 1. Motivation from Paper 1

Paper 1 now suggests:
- syntax and numeric/operator recognition are much easier than semantic binding;
- entity/literal recognition can be reasonable while attribute, participant, and dependency binding remain weak;
- persistent path naming/reuse is itself a burden;
- exact execution is not the main bottleneck;
- external state helps path reuse but does not by itself solve semantic compilation.

Paper 2 should therefore test whether asking the model to emit complete engine-specific programs is unnecessarily difficult.

Primary hypothesis:

> A small model may be better at expressing local semantic relations than at simultaneously choosing an engine, inventing stable variable names, serializing engine-specific syntax, lowering semantics to arithmetic/logic, and maintaining execution state.

---

## 2. Architecture

### Existing explicit approach

```text
NL
 ↓
LLM
 ↓
engine-specific block / ASL / tool call
 ↓
runtime
 ↓
engine
 ↓
result
 ↓
LLM
```

### Blackboard/functor approach

```text
NL / observation
      ↓
semantic LLM
      ↓
compact semantic functor(s)
      ↓
┌────────────────────────────┐
│ typed cognitive blackboard │
└────────────────────────────┘
      ↕       ↕       ↕
 calculator logic graph/date/units/...
      ↓       ↓       ↓
 derived facts / values / constraints
              ↓
          blackboard
              ↓
       next semantic step/query
```

The LLM expresses semantics. The runtime owns the machine.

---

## 3. Blackboard

The blackboard is a typed, scoped, provenance-linked shared workspace containing:

- entities;
- canonical slots/attributes;
- values;
- semantic relations;
- constraints;
- goals/queries;
- derived facts;
- pending computations;
- engine outputs;
- unresolved references;
- execution/provenance records.

The blackboard is runtime state, not merely prompt text.

A compact materialized view may be shown to the model.

Each record should carry at least:
- stable runtime ID;
- scope;
- type/kind;
- provenance/source;
- dependencies;
- status;
- creator (`model`, `runtime`, `engine`, `dataset`, `oracle`);
- optional confidence.

---

## 4. Functor

A functor is a compact semantic relation/operator with typed arguments.

Examples:

```text
value(claire, age, now, 18)
older_than(jessica, claire, age, now, 6)
percent_less(month3.downloads, month2.downloads, 30)
each(customers, returned_cartons, 60)
reachable(a, c)
isa(penguin, bird)
convert(result, 7.3, mile, kilometer)
date_add(result, "2026-08-28", "P90D")
```

A functor is **not necessarily an engine call**.

The registry maps a semantic functor to one or more of:
- a blackboard fact;
- a constraint;
- deterministic lowering;
- a candidate CogCop;
- derived facts;
- an executable operation.

The model should normally not emit the concrete engine identity.

---

## 5. Keep the surface syntax minimal

Do not turn the model target into a verbose ontology.

Compare at most two compact forms initially.

### Path-argument

```text
older_than(jessica.age_now, claire.age_now, 6)
```

### Compact positional

```text
older_than(jessica, claire, age, now, 6)
```

Prefer whichever gives better semantic binding and lower token cost.

Do not require JSON ontology output from the student.

---

## 6. Stable runtime identity

Raw model strings must not be authoritative identity.

Example internal state:

```text
entity:e17  display_name="claire"
slot:s42    entity=e17 attribute="age" qualifier="now"
```

The model may emit:

```text
claire.age_now
```

but the resolver lowers this to stable IDs.

Rules:
- first semantic introduction may create a symbol;
- later references should resolve to an existing ID;
- exact existing names are sticky;
- do not silently create duplicate slots;
- do not silently merge ambiguous synonyms;
- aliases must be explicit or registry-backed.

This is a core difference from free ASL path generation.

---

## 7. Functors reduce storage-layout burden

Example:

NL:
```text
Jessica is six years older than Claire.
```

Paper-1-style ASL:
```text
jessica.age_now = claire.age_now + 6
```

Functor:
```text
older_than(jessica, claire, age, now, 6)
```

Runtime lowering:

```text
ensure entity(jessica)
ensure entity(claire)
ensure slot(jessica, age, now)
ensure slot(claire, age, now)
post constraint(
  slot(jessica,age,now) =
  slot(claire,age,now) + 6
)
```

The model still must get:
- entities;
- attribute;
- qualifier;
- relation;
- direction;
- magnitude.

But it does not invent final state layout or arithmetic lowering.

Do not claim this removes the semantic problem.

---

## 8. Experimental ladder

### B0 — Preserve existing Paper 2
Existing typed blocks, generic tools, CPU routing, LoRA routing, TwIL diagnostics, public benchmark checkpoints, and current Paper 3 `no_go`.

### B1 — Common ASL baseline
Create a Paper-1-style common ASL/CCIR representation across calculator, units, dates, logic, and graph where practical.

Examples:

```text
result = 20 - 2
RETURN result
```

```text
reachable(a,c) :- link(a,b), link(b,c)
? reachable(a,c)
```

```text
result = date_add("2026-08-28","P90D")
RETURN result
```

```text
result = convert(7.3,mile,kilometer)
RETURN result
```

This determines how far a shared programming/semantic DSL gets before introducing blackboard functors.

### B2 — Functor generation, runtime capability resolution
Student emits semantic functors only.

Runtime chooses calculator/date/unit/logic/graph implementation.

The model must not emit concrete engine IDs.

### B3 — Persistent blackboard
Functor execution posts canonical derived facts/values back to blackboard.

Subsequent model turns receive a compact relevant state view.

### B4 — Bounded dependency/event propagation
Support deterministic dependency-triggered or query-triggered processors with strict limits:
- max derived records;
- max firings;
- max depth;
- no unrestricted planning;
- no unbounded recursion.

### B5 — Registry/backend changes
Swap or add concrete engine implementations while keeping semantic functor contracts stable.

Measure whether retraining is avoided.

---

## 9. Initial registered functors

Do not attempt a universal ontology.

### Quantitative

```text
value(path, literal)
equal(a,b)
greater_than(a,b)
less_than(a,b)
difference(a,b,d)
times(a,b,k)
fraction_of(a,b,f)
percent_more(a,b,p)
percent_less(a,b,p)
sum(result,members...)
count(group,n)
each(group,attribute,value)
total(result,group,attribute)
```

### Logic

```text
fact(predicate,args...)
rule(head,body...)
query(predicate,args...)
entails(statement)
```

Allow Prolog/Datalog-compatible surface sugar where useful.

### Graph

```text
isa(a,b)
edge(type,a,b)
reachable(a,b)
```

### Date/time

```text
date_add(result,date,duration)
date_diff(result,a,b)
before(a,b)
after(a,b)
```

### Units

```text
quantity(entity,value,unit)
convert(result,value,from_unit,to_unit)
same_dimension(a,b)
```

Keep signatures typed and versioned.

---

## 10. Functor registry

Implement a versioned `FunctorSpec` containing:

```text
name
version
argument schema
arity
types
aliases
purity
state effects
lowering handler
candidate engines
bounds
provenance policy
```

Example:

```yaml
name: percent_less
version: 1
args:
  - target: quantity_ref
  - base: quantity_ref
  - percent: decimal
lowering:
  op: DEC_PCT
candidate_engines:
  - calculator
pure: true
```

The semantic functor vocabulary should be smaller and more stable than the engine registry.

---

## 11. Processor/event modes

Support three bounded modes.

### Immediate
Execute on posting.

Example:
`convert(...)`

### Dependency-triggered
Keep unresolved until required arguments become grounded.

Example:
`older_than(jessica,claire,age,now,6)` before Claire's age is known.

### Query-triggered
Store relations/facts and resolve only when queried.

Example:
`reachable(a,c)`

The runtime owns scheduling, not the LLM.

---

## 12. Blackboard materialization

Do not dump the full blackboard to the model.

Render compact relevant state:

```text
KNOWN
claire.age_now = 18
jessica.age_now = 24

RELATIONS
older_than(jessica,claire,age,now,6)

PENDING
...

QUERY
...
```

Measure:
- total blackboard records;
- materialized records;
- materialized tokens;
- unresolved records;
- derived records;
- state reuse.

Keep representation PRA-compatible for future integration, but PRA is not required for the first experiment.

---

## 13. Dataset generation

Reuse Paper 1 semantic-generation infrastructure where practical.

For each audited example, produce parallel representations:

```text
NL
→ semantic ASL
→ semantic functor(s)
→ canonical blackboard/CCIR records
```

Paper 2 sources can include:
- GSM8K bounded arithmetic;
- TAT-QA arithmetic/structured examples;
- BIG-bench units;
- BIG-bench dates;
- ProofWriter;
- CLUTRR;
- existing synthetic graph/Datalog suite.

Preserve:
- source IDs;
- semantic-pattern IDs;
- leakage controls;
- teacher provenance;
- repair labels;
- quality grade.

Do not use gold rationale/answer as primary teacher input except in explicitly labeled repair mode.

---

## 14. Teacher requirements

For each selected example, teacher should generate both:

### ASL target
```text
jessica.age_now = claire.age_now + 6
```

### Functor target
```text
older_than(jessica,claire,age,now,6)
```

The functor target must reflect semantic meaning, not anonymous arithmetic steps.

Quarantine ambiguous or unsupported cases.

---

## 15. Student conditions

At minimum for Qwen3-0.6B:

1. base;
2. ASL ICL;
3. ASL LoRA;
4. ASL LoRA + small ICL;
5. functor ICL;
6. functor LoRA;
7. functor LoRA + small ICL.

Replicate promising conditions with SmolLM2-1.7B.

Use Paper 1 adapter findings to choose:
- QKVO r8 baseline;
- QKVO r16;
- QKVO+MLP r8 if supported.

Do not assume the same adapter placement is optimal for functors.

---

## 16. Fine-grained metrics

Reuse Paper 1 semantic diagnostics.

Measure:
- entity F1;
- attribute F1;
- qualifier F1;
- source literal;
- source-fact attachment;
- relation participant;
- relation direction;
- relation constant;
- argument order;
- dependency F1;
- aggregation/cardinality;
- temporal;
- units;
- query/return;
- workspace path reuse;
- symbol stability.

Functor-specific:
- functor class accuracy;
- arity validity;
- argument-role accuracy;
- argument binding;
- registered/unregistered rate;
- lowerability;
- capability resolution;
- derived-record correctness.

Factor the pipeline:

```text
NL
→ functor type
→ argument roles
→ argument values/entities
→ lowering
→ engine resolution
→ execution
→ blackboard update
→ final
```

A correct functor name with wrong arguments is not semantic success.

---

## 17. Main ASL vs functor question

Test:

> Does reducing representational freedom improve semantic binding?

Potential functor advantages:
- fewer arbitrary variable/path names;
- stable runtime-owned state identity;
- deterministic storage layout;
- deterministic arithmetic/engine lowering;
- engine independence;
- explicit relation classes.

Potential disadvantages:
- model must learn functor vocabulary;
- fixed ontology may be restrictive;
- positional arguments can increase direction errors;
- ASL/code may be more natural to pretrained models;
- unseen relations may fail.

Measure rather than assume.

---

## 18. Minimal first experiment: arithmetic + units + dates

Do not begin with all five engines.

Use three deterministic families.

### Arithmetic

NL:
```text
Month 3 has 30% fewer downloads than month 2.
```

ASL:
```text
month3.downloads = dec_pct(month2.downloads,30)
```

Functor:
```text
percent_less(month3.downloads,month2.downloads,30)
```

### Units

NL:
```text
Convert 7.3 miles to kilometers.
```

ASL:
```text
result = convert(7.3,mile,kilometer)
RETURN result
```

Functor:
```text
convert(result,7.3,mile,kilometer)
```

### Date

NL:
```text
What date is 90 days after 2026-08-28?
```

ASL:
```text
result = date_add("2026-08-28","P90D")
RETURN result
```

Functor:
```text
date_add(result,"2026-08-28","P90D")
```

Only add logic/graph after understanding this comparison.

---

## 19. Critical Paper 1 transfer experiment: relational binding

Use GSM8K-style relations.

NL:
```text
Jessica is six years older than Claire.
```

ASL:
```text
jessica.age_now = claire.age_now + 6
```

Functor:
```text
older_than(jessica,claire,age,now,6)
```

Measure:
- entity;
- attribute;
- relation participant;
- direction;
- dependency;
- canonical slot construction.

This directly tests whether functors help the Paper 1 semantic-binding bottleneck.

---

## 20. Critical cardinality experiment

Paper 1 shows aggregation/cardinality is a major failure family.

NL:
```text
Four customers each received 100 cartons and each returned 60.
```

ASL:
```text
customers.count = 4
customers.each.received = 100
customers.each.returned = 60
customers.total.accepted =
  customers.count * (customers.each.received-customers.each.returned)
```

Functor:
```text
count(customers,4)
each(customers,received_cartons,100)
each(customers,returned_cartons,60)
```

Runtime derives canonical totals.

This tests whether moving `each → total` lowering into the runtime helps once the semantic cardinality relation is correctly identified.

---

## 21. Generic-intent compatibility

Retain Paper 0 coarse intents:

```text
COMPUTE
RETRIEVE
VERIFY
HELP
```

Possible hierarchy:

```text
R1: COMPUTE
R2: percent_less(...)
R3: runtime chooses calculator implementation
```

The model may emit:
- R1 only when uncertain;
- R1 + functor when confident;
- HELP when formalization is insufficient.

Do not make the LoRA memorize the engine registry.

---

## 22. Tool comparison

Use matched tasks/backends.

Compare:
- four generic tools;
- explicit concrete tools;
- common ASL;
- blackboard functors;
- blackboard + runtime capability resolution;
- oracle functors.

Track:
- final accuracy;
- semantic metrics;
- prompt/schema tokens;
- generated tokens;
- model calls;
- engine calls;
- CPU/XPU latency;
- blackboard state size/reuse.

The blackboard claim must be more than token savings.

---

## 23. Stateful multi-query episodes

Add episodes where state is reused.

Example:
1. establish graph/Datalog facts;
2. query A;
3. query B;
4. add one fact;
5. query C.

Measure:
- repeated serialization avoided;
- derived-fact reuse;
- model tokens per query;
- engine amortization;
- stale-state errors;
- symbol stability.

This is a stronger test than isolated tool calls.

---

## 24. Bounded composition

Compare:
- one whole ASL program;
- sequential functors;
- blackboard event propagation.

Initially allow only two or three functor families in one episode.

No general planner.

Record dependency graph and every firing.

---

## 25. Open-vocabulary control

Only after registered-functor baseline.

Test a generic relation form:

```text
rel(older_than,jessica.age_now,claire.age_now,6)
```

or:

```text
relation("older_than",jessica,claire,attribute=age,amount=6)
```

Question:

> Is any gain due to semantic structure itself, or only due to forcing examples into a small fixed ontology?

Do not make this the primary condition.

---

## 26. Falsification

The blackboard/functor approach is weakened if:
- semantic binding is no better than ASL;
- argument errors dominate;
- functor vocabulary causes many unsupported cases;
- runtime canonicalization does not improve state reuse;
- composition is worse than ASL;
- blackboard state costs exceed context savings;
- generic tools match quality/cost;
- backend independence provides no practical gain.

Preserve negative results.

---

## 27. Hypotheses

H1 — **Representation reduction:** functors improve attribute/participant/dependency accuracy vs free semantic ASL.

H2 — **Canonical state:** runtime-owned IDs improve symbol stability and reuse.

H3 — **Deterministic lowering:** moving arithmetic/engine lowering out of model generation reduces errors.

H4 — **Engine independence:** semantic functor adapters survive backend changes better than engine-specific adapters.

H5 — **Compositionality:** multiple functors over shared state compose better than monolithic programs.

H6 — **Stateful amortization:** persistent blackboard state reduces repeated serialization and computation.

H7 — **Ontology cost:** fixed functor vocabularies may hurt unseen semantics; quantify this explicitly.

---

## 28. Execution order

P0. Preserve all current Paper 2 checkpoints.

P1. Define blackboard records and stable IDs.

P2. Define versioned `FunctorSpec`.

P3. Implement compact functor parser.

P4. Implement arithmetic/date/unit lowering.

P5. Build a small audited ASL-vs-functor dataset.

P6. Run teacher semantic audit.

P7. Train matched Qwen3-0.6B ASL and functor conditions.

P8. Compare fine-grained semantic metrics.

P9. If promising, add graph/Datalog functors.

P10. Replicate best condition on SmolLM2-1.7B.

P11. Add stateful multi-query episodes.

P12. Add bounded functor propagation/composition.

P13. Compare with generic/direct tools.

P14. Test backend/registry swaps.

P15. Update Paper 2 manuscript and machine-readable result gate.

Keep Paper 3 `no_go` unless separately justified.

---

## 29. Success criteria for first checkpoint

A positive result does not require immediate large final-answer gains.

Useful evidence would be:
- equal/better relation/functor-class accuracy;
- materially better attribute/participant binding;
- materially better dependency F1;
- better symbol stability/path reuse;
- lower path-name entropy;
- equal/better final-answer accuracy;
- fewer model tokens per semantic operation.

If semantic state improves but final answer does not, diagnose result integration separately.

---

## 30. Manuscript framing

If positive:

> Paper 1 indicates that small-model semantic compilation is limited more by binding and workspace construction than by operator recognition. Paper 2 therefore compares free semantic ASL with a blackboard architecture where the model emits compact semantic functors and the runtime canonicalizes state, resolves capabilities, lowers operations, executes specialized engines, and posts derived facts.

Do not claim:
- solved symbol grounding;
- solved general reasoning;
- general blackboard AI;
- replacement of LLMs.

This is a measured architectural alternative to engine-specific tools/blocks and free-form semantic ASL.
