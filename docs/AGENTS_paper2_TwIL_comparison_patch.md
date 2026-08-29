# AGENTS PATCH — Paper 2: TwIL-LM3 vs Cognitive Coprocessors

## Purpose
Add a major Paper 2 experimental thread comparing two ways of improving reasoning:

1. **Internalize reasoning in Transformer weights** — TwIL-LM3 / SmolLM3 lineage specialized for formal/logical inference.
2. **Internalize delegation and execute in specialized machines** — Cognitive Coprocessors.

This is a headline comparison, not a related-work footnote.

Core question:

> When a computation has a mature algorithmic substrate, should post-training teach the Transformer to approximate the computation itself, or teach it to recognize, formalize, delegate, and integrate it?

Do not argue that Transformers cannot reason. TwIL is useful precisely because it demonstrates that targeted post-training improves formal reasoning. The narrower hypothesis is that ability to approximate a computation does not imply that the Transformer is the most reliable, efficient, scalable, or compositional execution substrate.

## Architectures

### A — reasoning internalized
Natural language -> reasoning-specialized Transformer -> CoT/recurrent token computation -> answer.

### B — delegation internalized
Natural language -> Transformer + interface policy -> typed block/micro-IR -> specialized engine -> typed result/derivation -> Transformer.

### C — specialized neural formalizer + exact engine
Natural language -> TwIL-LM3 -> formal representation -> Datalog/graph/SMT/algebra engine -> exact result.

Architecture C is mandatory: TwIL and coprocessors may be complementary rather than competitors.

## Compositionality hypothesis
A reasoning-specialized Transformer internalizes a class of reasoning into one neural substrate. A cognitive runtime can compose qualitatively different machines:

- calculator
- date/time
- units
- symbolic algebra
- Datalog/Horn
- graph/frame
- SMT/constraints
- structured-data operations where relevant

Add hypothesis:

> **H-composition:** adding heterogeneous coprocessors expands the computational envelope modularly, while preserving exact execution in supported domains, without requiring the Transformer to learn every computation or combination internally.

Measure both within-domain reasoning and cross-domain composition.

## Model conditions
Where feasible include:

1. SmolLM3-3B base/instruct — untreated lineage baseline.
2. TwIL-LM3 — reasoning internalized in weights.
3. SmolLM3 + Cognitive Coprocessor interface — delegation specialization.
4. TwIL-LM3 + Cognitive Coprocessors — specialized formalizer + exact executor.
5. Existing small CogCoproc models such as Qwen3-0.6B and SmolLM2-1.7B for the small-model leverage curve.
6. Explicit tools — conventional invocation.
7. Oracle formalization + engine.
8. Oracle engine selection + model-authored payload.

Pin model/tokenizer revisions and use matched decoding/reasoning budgets.

## Experimental ladder

### L0 — Neural reasoning only
Compare SmolLM3, TwIL-LM3, and selected baselines without engines.
Establish the value of reasoning fine-tuning itself.

### L1 — Horn/Datalog
Compare internal neural deduction, model->Datalog block->engine, and oracle formalization->engine.
Separate semantic formalization from deductive execution.

### L2 — Graph/frame
Repeat on ISA/frame/transitive graph reasoning.

### L3 — Constraint engine
If implementation gate passes, add bounded Z3/cvc5-class SMT.

### L4 — Non-logic computation
Add calculator plus date/time or units; optionally symbolic algebra.
This tests breadth: logic specialization should not automatically provide exact competence across unrelated computational families.

### L5 — Heterogeneous single-engine selection
Mix task families without telling the model which engine is needed.
Measure engine selection, payload correctness, final answer, and false activation.

### L6 — Two-engine composition
Bounded examples:
- Datalog -> calculator
- graph -> Datalog
- units -> calculator
- algebra -> numeric evaluation
- date -> arithmetic/comparison

No general planner.

### L7 — Repeated-query/persistent-state workload
Provide one fact/rule/world state and ask 1, 5, 20, and 100 queries.
Coprocessor path may parse/compute closure once and reuse typed state.
Measure amortized neural and CPU cost per query.

### L8 — Scaling stress
Increase:
- derivation depth
- fact/rule count
- distractors
- graph depth
- constraint count
- arithmetic difficulty
- repeated-query count

Compare quality and accelerator cost as problem size grows.

## Core hypotheses

### H1 — Conditional execution reliability
Conditional on correct formalization, bounded specialized engines have substantially lower execution error than neural CoT.

Report `P(final correct | formalization correct)`.

### H2 — Closure
Specialized finite closure retains required consequences more reliably as depth/state grows.

Add **closure recall**:
`required valid consequences recovered / required valid consequences`.

Track premature stopping, skipped inference, contradiction, and truncation.

### H3 — Accelerator-cost scaling
As reasoning depth/problem size grows, coprocessor execution requires less accelerator work than internal neural reasoning.

Report raw prompt/generated tokens, XPU/GPU time, CPU-engine time, and total wall time.

### H4 — Reuse/amortization
Persistent typed state increasingly benefits repeated queries over the same world/rule set.

### H5 — Compositionality
A common runtime combines multiple exact computational domains without requiring a separate full reasoning fine-tune for every combination.

### H6 — Specialization efficiency
Compare learned specialization required for reasoning internalization vs delegation/interface policy:
- trainable parameters
- training examples/tokens
- training time
- adapter bytes
- recurring prompt tokens
- inference model size

Parameter count alone is not an efficiency proof.

### H7 — Neural semantic advantage
TwIL-style specialization may remain superior when formalization is ambiguous, inductive, fuzzy, or outside the symbolic fragment.

This counter-hypothesis is mandatory.

### H8 — Hybrid advantage
TwIL + exact coprocessors may outperform both TwIL-only and generic-model + coprocessors if reasoning specialization improves semantic parsing/formalization.

Treat this as a strong positive outcome.

## Formalization vs execution decomposition
For formal tasks separately score:

1. semantic intent
2. engine family
3. entities/facts
4. predicates/relations
5. rules/constraints
6. query
7. engine execution
8. result integration

Report:
- engine-selection accuracy
- formalization accuracy
- execution correctness conditional on valid IR
- final answer
- result-use/override

A perfect engine solving a wrong formalization is not evidence of system correctness.

## Compositionality benchmark
Build orthogonal combinations:

### Logic + arithmetic
Rules determine a quantity/entity; calculator performs hard numeric work.

### Graph + logic
Graph inheritance establishes membership; Datalog derives consequence.

### Units + arithmetic
Normalize quantities then calculate.

### Constraint + arithmetic
SMT finds a valid assignment; calculator evaluates a derived quantity.

### Algebra + numeric
Symbolic engine solves; calculator evaluates instantiated result.

Keep chains short and interpretable. The goal is modular capability composition, not artificial planning depth.

## Incremental capability experiment
Build systems progressively:

- neural only
- + calculator
- + Datalog
- + graph
- + date/units
- + SMT/algebra

At each step measure:
- new task families solved
- regressions
- adapter/context changes
- prompt burden
- runtime additions
- new training required
- mixed-suite performance

This directly tests capability-by-adding-machines versus capability-by-learning-approximation.

## Fairness to TwIL
Do not create only exact solver-friendly tasks.

Include TwIL-aligned tasks where feasible:
- NL -> FOL
- entailment
- semantic parsing
- rule induction
- formalization/critique

Separate semantic/formalization quality from exact deductive execution.

Where TwIL produces useful formal representations, feed them to the coprocessor.

## TwIL + coprocessor integration
Treat TwIL as a candidate neural formalization/interface model.

Test:
1. TwIL emits Paper 2 typed blocks via ICL.
2. Small LoRA teaches the common block protocol if necessary.
3. TwIL emits FOL/structured logic compiled to Datalog/SMT where valid.

Measure whether it needs less adaptation than a generic model.

## Training-cost accounting
For TwIL distinguish published/documented training recipe from costs reproduced locally.

For CogCoproc adapters report actual:
- trainable params
- adapter bytes
- target tokens
- XPU training time
- dataset size

Do not invent or infer TwIL training cost.

## Required efficiency plots
- accuracy vs derivation depth
- accuracy vs facts/rules/constraints
- reasoning tokens vs depth
- accelerator time vs problem size
- CPU-engine time vs problem size
- amortized cost/query vs repeated queries
- mixed capability coverage vs number of coprocessors
- accuracy/cost Pareto

## Failure modes

### Neural internal reasoning
- skipped inference
- invalid inference
- contradiction
- truncation
- unstable recurrence
- correct reasoning but wrong final answer

### Coprocessor
- missed delegation
- wrong engine
- malformed formalization
- runtime rejection
- result ignored/overridden

### Hybrid
- TwIL representation incompatible with engine fragment
- overformalization
- compilation information loss
- engine result misinterpreted

## Interpretation outcomes

### A — coprocessor dominates exact domains
Supports learning semantics/delegation while executing formal computation algorithmically.

### B — TwIL dominates
Investigate whether formalization overhead, engine mismatch, or interface cost erases exact-execution benefits.

### C — hybrid dominates
Potentially strongest result: specialized neural reasoning is most useful as a semantic/formalization front-end to exact engines.

### D — domain-dependent frontier
Likely valuable result:
- neural for fuzzy/inductive semantics
- coprocessor for exact closure/computation
- hybrid for formal NL reasoning

Do not force a universal winner.

## Manuscript requirement
Add a major Paper 2 section:

# Reasoning in Weights versus Reasoning in Coprocessors

Cover:
1. TwIL as alternative hypothesis
2. internalized vs delegated reasoning
3. reliability and closure
4. accelerator scaling
5. persistent-state reuse
6. compositionality
7. specialization/training efficiency
8. hybrid neural-formalizer + exact-executor
9. experimental ladder/results

If this becomes a headline experiment, mention TwIL in abstract and conclusion.

## Broader thesis under test
> Transformer generality does not imply substrate optimality.

The Transformer may be best used for:
- semantics
- ambiguity resolution
- formalization
- delegation
- result integration

while specialized machines perform:
- exact arithmetic
- closure
- graph traversal
- constraints
- algebra
- other well-defined computations.

Paper 2 must test this rather than assume it.

## Immediate order
1. Pin TwIL-LM3 and base SmolLM3 revisions.
2. Reproduce a small TwIL-aligned reasoning subset.
3. Build shared Horn/graph scaling benchmark.
4. Add formalization/execution decomposition.
5. Run SmolLM3 vs TwIL internal reasoning.
6. Add SmolLM3 + coprocessor.
7. Add TwIL + coprocessor.
8. Add calculator/non-logic capability.
9. Add repeated-query state reuse.
10. Add bounded two-engine compositions.
11. Add SMT/algebra only if justified.
12. Produce substrate/division-of-labor conclusion.

## Deliverables
- updated full Paper 2 AGENTS context
- TwIL comparison configs/manifests
- pinned SmolLM3/TwIL checkpoints
- formalization-vs-execution benchmark
- depth/state/repeated-query scaling artifacts
- compositionality benchmark
- internalized-vs-delegated specialization table
- TwIL + coprocessor hybrid
- accelerator/CPU cost plots
- major manuscript section
- evidence-based recommendation for Papers 3, 6, and 7
