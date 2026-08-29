# AGENTS - Paper 2

## Mission
Test whether one small model can learn a compact execution-interface policy for
multiple deterministic engines while exact parsing, execution, state,
provenance, reinjection, and enforcement remain in the runtime.

## Central question
Can a small model discriminate among heterogeneous local computational engines,
emit bounded typed execution regions, and use exact results without context cost
growing proportionally with engine count?

## Frozen next-iteration outcome
- The 200/50/100 benchmark and leakage audit are frozen under
  `artifacts/paper2/next_iter/data_v1`.
- The anchored prompt-derived runtime reflex is exact at 1, 2, 3, and 5 engines;
  the gold-envelope oracle is a separate condition. Neither is model evidence.
- At five engines, Qwen3-0.6B, SmolLM2-1.7B, and Gemma3-1B learned adapters reach
  selection rates 0.1875, 0.35, and 0.0 and runtime-exact rates 0.1875, 0.3125,
  and 0.0, all with false-activation rate 0.0.
- Qwen and SmolLM2 use accuracy is 0.0 on assessed runtime results; every assessed
  result is overridden.
- The machine-readable Paper 3 gate is `no_go`, with no passing model family.

Do not describe deterministic success as learned semantic routing or use low
development loss as held-out interface reliability. Further adapter tuning or an
optional sixth engine requires a new versioned experiment rather than mutating
these artifacts.

## TwIL comparison pilot (diagnostic only)

The first TwIL/SmolLM3 comparison is frozen under `artifacts/paper2/twil` and
documented in `TWIL_DIAGNOSTIC.md`. Do not use it to rank internal reasoning.
The all-positive Horn/graph labels admit a constant-true solution, and the
`/no_think`, 160-token protocol is not aligned with TwIL's documented
thinking-mode 2048-token evaluation. It remains valid evidence that the tested
ICL typed interface is unreliable: strict end-to-end exactness is 6/22 for
SmolLM3 hybrid and 7/22 for TwIL hybrid, with 100% execution conditional on an
exact IR. Paper 3 and Paper 3.5 remain paused until a corrected balanced and
TwIL-aligned Paper 2 comparison is diagnosed.

## Placement rule
- **Weights:** stable semantic selection and typed serialization policy.
- **Runtime:** parsing, validation, bounds, execution, provenance, reinjection,
  state, and enforcement.
- **Context:** cold-start development, demonstrations, and changing interfaces.

Do not assume LoRA plus ICL is superior; measure interference and false
activation.

## Engine suite
1. Bounded exact calculator.
2. Finite Horn/Datalog forward chaining.
3. ISA/frame graph closure.
4. ISO date arithmetic.
5. Dimension-checked unit conversion.

An optional symbolic algebra or bounded SMT engine may be added only after the
five-engine core works; do not add both initially.

## Typed blocks
All blocks must close before execution and normalize to the shared
`CoprocessorRequest` envelope.

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

Malformed, open, over-budget, dimensionally invalid, and unavailable-engine
requests fail closed.

## Interface conditions
1. Base model without engine assistance.
2. Context/ICL with examples for all enabled families.
3. Explicit textual or native tool schemas where supported.
4. One multi-engine LoRA adapter with a minimal stable contract.
5. Runtime-only deterministic reflex.
6. Hybrid learned policy plus deterministic runtime.
7. Oracle selection and serialization.

The adapter learns when to offload, which engine to choose, and how to serialize
inputs. It never learns exact output values.

## Models
Prioritize Qwen3-0.6B and SmolLM2-1.7B. Include Gemma3-1B when Paper 1 placement
and XPU training remain stable. Pin all checkpoint revisions and adapter IDs.

## Capability-count scaling
Use nested engine catalogs of 1, 2, 3, and 5 engines. For each condition report:

- prompt/schema tokens;
- detect accuracy and false activation;
- engine selection;
- payload normalization;
- execution and exact runtime result;
- reinjection and result use;
- final answer;
- engine/model calls and token counts;
- CPU engine time, XPU generation time, state items/bytes, and total wall time.

## Benchmark and leakage
Use disjoint train, development, and untouched test namespaces for operands,
entities, dates, values, and controls. Targets contain typed requests, not
answers. Freeze a machine-readable overlap and target-answer leakage audit before
training.

## Persistent typed state
State is append-only and provenance-linked. Measure reuse of derived Horn and
graph state. Rollback and retraction remain Paper 4.

## Bounded compositions
Allow only deterministic two-stage compositions, initially date -> calculator
and graph -> Datalog. Log dependencies and each engine operation. Do not build a
general planner.

## Factorized metrics
Score DETECT, ENGINE SELECT, PAYLOAD NORMALIZE, EXECUTE, REINJECT, USE, PRESERVE,
and FINAL ANSWER independently. Add ignored-result, overridden-result, and
wrong-reinterpretation rates when model result-use generation is enabled.

## Economics
Track engine CPU time and calls, state growth, prompt/generated/reinjected
tokens, model calls, XPU memory, and wall time. Correctness-first full-prefix
generation is prototype latency; do not claim production speedups from it.

## Falsification
Paper 2 is weakened if explicit tools dominate quality/cost as catalogs grow,
learned selection collapses across families, context savings are negligible,
false activation rises strongly with engine count, specialized engines add no
value over one generic interpreter, compositions fail, or state is not reused.

## Scope
Paper 2 learns an explicit bounded family/block policy. It does not infer hidden
semantic interrupts, perform unrestricted theorem proving, execute generated
code, plan generally, or roll back state. Paper 3 remains the learned implicit
interrupt paper.

## Paper 3 gate
Proceed only if at least two model families at five engines reach 0.8 engine
selection and runtime exact rates with at most 0.1 false activation. Otherwise
record `no_go` and preserve the failure decomposition. The frozen next iteration
records `no_go`; do not treat the gate as open.

## Deliverables
The repository must contain the generic coprocessor registry, five adapters,
shared typed IR, leakage-audited multi-engine LoRA data, context/weights/runtime
comparison, nested capability sweeps, factorized traces and economics, bounded
compositions, state-reuse evidence, plots, rebuilt paper, and explicit Paper 3
gate artifact.
