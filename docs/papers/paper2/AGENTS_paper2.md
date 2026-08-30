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

## Four generic tools checkpoint

The skeptical model-facing baseline is exactly `__compute`, `__retrieve`,
`__verify`, and `__help`. These gateways use the same R2 parser, engine registry,
policy, typed state, provenance, and result integration as CogCop. Concrete
engine identities remain outside the four schemas.

The retained oracle-timed transport audit covers the 100-row five-engine test
freeze: 80 assistance rows and 20 controls. Generic `__compute` and direct
CogCop blocks have 1.000 backend-result agreement and 1.000 deterministic
accuracy. The four schemas cost 72 lexical tokens and remain constant from one
through five registered engines.

This validates transport and schema invariance only. The model does not choose
the call or arguments, so voluntary assistance recall, malformed arguments,
timing, CONTINUE, extra turns, and Automatic Rescue Rate are unmeasured. Do not
use this result to claim either tools or CogCop win. A deeper CogCop claim
requires matched model-facing rescue, timing, integration, scaling, or cost
evidence.

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

## Tokenizer-aware trigger outcome
The matched add-on uses the existing 2,250/450/900 diagnostic split and exact
pinned Qwen/SmolLM2/Gemma tokenizers. It compares word, character, shared
word/punctuation, raw native, and normalized native representations under
unigram/token-ngram TF-IDF, BM25 exemplars, and class prototypes. BM25 `k` and
all thresholds are selected on development only. Token pieces, predictions,
confusions, per-engine metrics, index sizes, and latency are retained under
`artifacts/paper2/tokenizer_triggers/diagnostic_v1`.

The development-selected six-way condition is normalized Gemma unigram TF-IDF;
test accuracy is 0.809, positive engine selection 0.656, FAR 0, and runtime
exactness 0.656. Qwen normalized BM25 reaches 0.870 selection and 0.970 runtime
exactness but FAR 0.200. Neither replaces T1 lexical/regex (0.900 selection,
0.900 runtime exact, FAR 0).

The explicit hierarchy uses CPU graph/Datalog cues, token routing for
NONE/calculator/date/units, and the row-matched Qwen-L2 router for deferred
cases. It avoids 77.44% of model calls and has 0 FAR, but selection is only
0.804, with graph recall 0.500. Runtime exactness of 0.904 includes
answer-equivalent engine substitutions and does not clear the identity gate.
TwIL is not merged because its benchmark is not row-aligned. Paper 3 remains
`no_go`.

## Public benchmark registration

The first external-validity checkpoint is frozen under
`artifacts/paper2/public_benchmarks/data_v1`. It contains 2,253 selected IDs:
500 GSM8K, 600 BIG-bench unit conversion, all 73 BIG-bench date understanding,
all 360 balanced ProofWriter, and 720 CLUTRR examples. Source repositories,
converted-Parquet revisions and checksums, source rows, labels, content hashes,
and difficulty strata are pinned. Benchmark text is loaded locally and is not
redistributed in the artifacts.

This is a registered suite, not a completed comparison. Do not report public
accuracy until task adapters account separately for formalization and backend
coverage. The current typed engines are not silently treated as compatible with
compound BIG-bench units, natural-language dates, ProofWriter English, or CLUTRR
kinship algebra. Paper 3 remains `no_go` while this diagnostic proceeds.

The first factorized coverage audit is frozen under
`artifacts/paper2/public_benchmarks/coverage_v1`. Untuned T1 and T2 transfer at
only 0.139 and 0.016 pooled engine recall. Machine-readable oracle
formalization is present for 0.538 of rows, but the current exact backend
contracts cover only 0.163. GSM8K is the sole executable bridge: 493/500 rows
have annotated traces, 368 fit the bounded integer contract, and all 368 execute
exactly. Decimal literals account for 100 incompatibilities, non-binary
annotation steps for 25, and absent traces for 7.

Do not train or score a public six-way router as if this were end-to-end task
coverage. Implement and freeze matched units/date/ProofWriter/CLUTRR adapters
first, then run generic tools, CPU routing, TwIL, hybrid, and oracle conditions
on identical selected IDs. The measured status is `backend_gap`; Paper 3 remains
`no_go`.

## Executable public checkpoint

The adapter prerequisite is now partially satisfied by a frozen 60-row
developmental slice under `artifacts/paper2/public_compute_v1`: 12 rows each
from GSM8K, BIG-bench units, BIG-bench dates, balanced ProofWriter, and CLUTRR.
Eligibility is target-validated before selection. GSM8K uses calculator-executed
gold traces; units/date/ProofWriter use bounded prompt parsers and exact local
execution; CLUTRR uses annotated proof replay and must not be described as the
ISA/frame engine or end-to-end graph parsing.

The pinned Qwen3-0.6B matched matrix has seven conditions. LLM-only is 0.183,
four generic tools 0.150 with 0/60 valid calls, generic CogCop 0.150 with 1/60
valid controls, transferred T1 0.200 with 3/60 exact routes, disjoint shared-NLP
BM25 plus exact result 0.683, oracle route plus the same exact result 0.683, and
zero-call runtime-copy 1.000. BM25 and oracle use identical prompts and produce
identical generations. Their 19 failures are result-integration overrides, not
routing or backend failures.

BM25 is fitted on 2,184 other registered rows after excluding every source row
sharing an evaluation ID, including nine duplicate-ID collisions. It routes
60/60 correctly, but this is within-suite benchmark-family separation with no
negative controls. Do not generalize it to open-domain semantic routing or use
it to reopen the Paper 3 gate. Automatic Rescue Rate is 2/60 for T1 and 41/60
for BM25 relative to voluntary four-tool misses. Runtime-copy remains the
production upper bound for exact COPY tasks.
