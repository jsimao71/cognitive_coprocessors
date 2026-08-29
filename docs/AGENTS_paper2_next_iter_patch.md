# AGENTS PATCH — Paper 2 Next Iteration

## Why this patch is needed
Paper 2 is stale relative to Paper 1. It still assumes the compute gate is closed and that strict syntax is the only admissible predecessor.

Paper 1 now shows across Qwen3-0.6B and SmolLM2-1.7B that:
- LoRA + minimal contract can yield 16/16 block execution, 16/16 answers, and 0/12 false blocks;
- context/ICL is useful for development but adds recurring tokens and false interventions;
- runtime-only normalized reflex is safe but has weaker exposure/use;
- exact computation remains entirely in the deterministic calculator runtime.

The new placement rule is:
> stable semantic selection and serialization policy in adapter weights; exact parsing, bounds, execution, provenance, reinjection, and enforcement in runtime; context for cold-start/development/changing interfaces.

## Revised central question
Can one small model, with a compact learned execution-interface policy, reliably discriminate among multiple local computational engines, emit bounded typed execution regions, and use exact runtime results without context cost growing proportionally with engine count?

## Gate status
Developmental compute gate: OPEN.

Paper 2 is now justified for heterogeneous-interface experiments, while confirmatory claims remain gated on larger untouched sets and broader model-family replication.

## Revised hypotheses
- H1 heterogeneous interface reliability: a small adapter selects the right engine family and payload with low false activation.
- H2 context burden scaling: explicit tool schemas and ICL grow with engine count; adapter+minimal prompt stays approximately constant.
- H3 exact-computation leverage: assisted accuracy degrades more slowly with task difficulty.
- H4 engine-count robustness: irrelevant added engines do not materially degrade selection.
- H5 typed-state reuse: persistent derived state reduces repeated neural/external work.
- H6 small-model leverage: small augmented models may close the gap to larger unassisted models on offloadable tasks.

## Engine suite
Required:
1. Calculator
2. Horn/Datalog
3. ISA/frame graph
4. Date/time
5. Units/dimensions

Optional sixth engine after the core works:
- SymPy-like symbolic algebra, or
- Z3/cvc5 bounded constraints

Do not add both initially.

## Typed execution blocks
Use a shared bounded family syntax derived from Paper 1:

```calculator
15246377 * 746647383
```

```datalog
fact link(a,b)
fact link(b,c)
query reachable(a,c)
```

```graph
isa penguin bird
isa bird animal
query isa penguin animal
```

```date
add 2026-08-28 P90D
```

```units
convert 7.3 mile -> kilometer
```

All blocks must close before execution and normalize to the common typed request envelope.

## Interface placement conditions
A. Context / ICL: few-shot examples for all engine families.
B. Weights / LoRA: one compact multi-engine adapter learns when to offload, which engine, and how to serialize.
C. Runtime: deterministic reflexes and block parsing.
D. Hybrid proposed architecture: learned policy + deterministic runtime validation/execution.

Do not assume LoRA+ICL is better; Paper 1 showed interference can increase false activations.

## Adapter strategy
Start from the Paper 1 protocol and train one adapter per base model across all included engine families.

Priority models:
1. Qwen3-0.6B
2. SmolLM2-1.7B
3. Gemma if the Paper 1 Gemma result succeeds and XPU training is stable

The key target is a common meta-protocol:
semantic need -> engine family -> typed execution region -> runtime -> exact result.

## Capability-count scaling
This becomes a primary experiment.

Use nested engine catalogs:
- 1
- 2
- 3
- 5
- optionally 6

Compare:
- explicit tools with all schemas/descriptions;
- ICL with all engine demonstrations/contracts;
- LoRA + minimal stable contract;
- runtime-only reflex;
- oracle.

Measure:
- prompt tokens vs engine count;
- selection accuracy;
- false activation;
- payload correctness;
- final answer;
- latency;
- generated/reinjected tokens.

## Explicit tool baseline
Use native function/tool calling where reliably supported; otherwise clearly document the textual schema.

Track descriptor/schema tokens, malformed calls, wrong-engine calls, payload correctness, and result use.

## Mixed tasks
Class A: single-engine selection across heterogeneous tasks.
Class B: bounded two-stage compositions such as date->calculator or graph->Datalog.

Do not implement a general planner.

## Persistent typed state
Retain append-only typed state and provenance.
Add only required new record types.
Rollback/retraction remain Paper 4.

## Factorized metrics
Score separately:
1. DETECT
2. ENGINE SELECT
3. PAYLOAD NORMALIZE
4. EXECUTE
5. REINJECT
6. USE
7. PRESERVE
8. FINAL ANSWER

Add ignored-result, overridden-result, and wrong-reinterpretation rates.

## CPU/XPU economics
Track per engine:
- CPU execution time
- engine calls
- state bytes
- XPU prompt tokens
- generated tokens
- reinjection tokens
- total wall time

Do not claim production latency savings from the correctness-first backend; separate prototype latency from token/operation accounting.

## Exploratory small-vs-large comparison
If budget permits, compare:
- small unassisted
- small + coprocessors
- larger unassisted
- larger + same coprocessors

The question is whether augmentation disproportionately closes the size gap on offloadable tasks.

## Falsification
Paper 2 is weakened if:
- explicit tools dominate quality/cost as engine count grows;
- learned selection collapses with multiple families;
- context savings are negligible;
- false activations rise strongly with engine count;
- heterogeneous tasks gain nothing over single-engine controls;
- typed state is not reused;
- a generic code interpreter matches the specialized suite on safety, cost, and reliability.

## Relationship to Paper 3
Paper 2 may use learned adapter weights for an explicit bounded block protocol because Paper 1 established that placement.

Paper 3 remains distinct:
- Paper 2: learned explicit family/block protocol.
- Paper 3: implicit semantic interrupts from unconstrained language/hidden state without requiring explicit execution-region emission.

## Immediate order
1. Patch manuscript/AGENTS; preserve current `empirical:false` smoke.
2. Add date and units engines with deterministic tests.
3. Freeze common typed block grammar.
4. Build multi-engine positives + controls.
5. Train Qwen3-0.6B multi-engine adapter.
6. Replicate on SmolLM2-1.7B.
7. Run capability-count scaling.
8. Add one optional SymPy/SMT engine only if core works.
9. Run bounded two-engine compositions.
10. Decide confirmatory gate.

## Deliverables
- patched `paper2.tex`
- full updated `AGENTS_paper2.md`
- generic coprocessor registry
- calculator/Horn/graph/date/units adapters
- common block/request/result IR
- multi-engine LoRA dataset + leakage audit
- context vs weights vs runtime comparison
- capability-count plots
- factorized traces
- CPU/XPU cost accounting
- explicit Paper 3 gate decision
