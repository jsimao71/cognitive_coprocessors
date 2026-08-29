# AGENTS — Paper 2 Next Iteration: Heterogeneous Compute Diagnosis, Trigger Ladder, Richer LoRA, and Provenance-Aware Result Use

## Mission
Diagnose the current five-engine failure before escalating to Paper 3. Established facts: deterministic dispatch executes calculator/Datalog/graph/date/units exactly; deterministic two-stage composition is exact; Paper 1 single-engine calculator LoRA works; the current five-engine generative LoRA collapses across families; ICL suffers catalog interference; exact results are often overridden; CPU engine cost is negligible versus neural generation.

The next iteration must factor the failure into dataset richness, native model discrimination, routing, serialization, multi-protocol interference, LoRA capacity/targets, model scale, and result assimilation.

## Part I — Richer dataset
Scale to at least 250 examples/engine first, then 500–1,000 if learning curves justify it, plus >=1,000 controls. Train/dev/test must use disjoint semantic templates and value/entity namespaces.

For each engine create broad paraphrase families and cross-family hard negatives. Examples should include arithmetic word problems and notation; date offset/difference/relative forms; unit conversion/comparison/mismatch; graph ISA/inheritance/path; Datalog reachability/implication/multi-hop Horn forms.

Add lexical-triviality baselines: TF-IDF linear, word n-gram, char n-gram. Plot model performance versus dataset size.

## Part II — Separate classification from serialization
### A — six-way classification only
Output exactly one of NONE/CALCULATOR/DATE/UNITS/GRAPH/DATALOG. No blocks. Test base model, LoRA classifier, larger models, SmolLM3/TwIL where relevant.

### B — oracle engine + generated payload
Tell the model the correct engine and score only payload serialization.

### C — oracle engine + deterministic parser
Runtime knows engine and extracts arguments directly from the original prompt. This bounds the value of eliminating block syntax.

### D — classifier -> per-engine serializer
Shared router plus specialized serializer/adapter per engine. This tests whether one multi-protocol adapter suffers destructive interference.

## Part III — CPU trigger ladder without blocks
T0 current anchored template parser.
T1 lexical/regex trigger.
T2 semantic CPU feature rules (unit lexicon, temporal grammar, arithmetic parser, graph cues, Horn cues).
T3 classical ML CPU classifier.
T4 tiny CPU embedding/classifier.
T5 main-model classifier/LoRA.
T6 generative typed block.

For each report routing accuracy, false activation, CPU/XPU cost, latency, portability, and engineering burden. Default principle: use the cheapest reliable trigger.

## Part IV — Model capability scaling
Add where XPU permits: Qwen3-4B, SmolLM3-3B, TwIL-LM3, optionally another 3–4B model. Test six-way classification, oracle-engine payload generation, and full block generation. Determine whether failure is primarily insufficient native semantic discrimination.

## Part V — LoRA architecture ablations
Current Paper 2 LoRA already targets q/k/v/o projections across layers; it is not end-only. Test:

### L1 current attention-only LoRA
q/k/v/o.

### L2 attention + MLP LoRA
gate_proj/up_proj/down_proj in addition.

### L3 layer-region LoRA
early third, middle third, late third, early+middle, middle+late, all layers. Diagnose semantic classification vs late serialization roles.

### L4 explicit router head + generator
Small six-way classifier plus existing generator/runtime.

### L5 per-engine adapters
One adapter per engine, selected by CPU/neural router. Paper 1 makes this a mandatory comparison.

### L6 dynamic adapter selection/composition
If PEFT/runtime permits, load/select adapters per engine without exposing all protocols in context.

## Part VI — Result-use and provenance
Treat result assimilation as a separate experimental thread. Effective stream is QUESTION -> COPROCESSOR RESULT -> MODEL FINAL ANSWER.

Create three task families:
- COPY: final answer is exact result;
- INTERPRET: explain/classify the exact result;
- CONTINUE: exact result is an intermediate fact for downstream reasoning.

### Fix A — formatting/authority contract
Compare plain result, XML/typed tag, fenced result, engine-specific tag, generic trusted-record tag, and an explicit `authoritative exact result; do not recompute` contract. Keep semantics fixed.

### Fix B — runtime-copy / enforcement
For COPY tasks bypass neural regeneration and render/slot-fill the exact typed result directly. Also test constrained output. This is the strongest production baseline and should remain even if neural fixes work.

### Fix C — provenance-aware self-attention
For INTERPRET/CONTINUE tasks where the model must actually use the result.

#### C1 Inspect attention
Instrument SA during final generation. Partition context into QUESTION, COPROCESSOR RESULT, INSTRUCTION, OTHER. For each layer/head/generated token measure summed attention mass to question and result spans. Compare correct-use vs override cases. Attention is diagnostic, not causal proof.

#### C2 Causal masking
Run full context; reduce/mask question attention; mask result attention; and minimal-result-only variants. If reducing question influence improves preservation and masking result hurts it, question/result competition is supported.

#### C3 Position ablation
Place result before question, after question, immediately before answer, and optionally repeat it near the final position. Hold content constant.

#### C4 Formatting/authority-token ablation
No type vs result tag vs exact/provenance tag vs engine-specific vs generic trusted-record type.

#### C5 Fixed provenance attention bias
For result-span tokens j use attention logit `s'_ij = s_ij + beta`. Sweep beta from zero through small/moderate/large positive values. Test all layers, late-only, and mid+late. Measure result use, control degradation, downstream reasoning, and attention-mass change.

#### C6 Layer-specific bias
Only if C5 helps. Tune/learn one scalar beta_l per layer.

#### C7 Learned provenance/KV embeddings
Only after a fixed/layer-specific bias demonstrates causal value. Candidate mechanisms: provenance embedding added to K/V, authority embedding, or tiny record-type-conditioned adapter.

## Provenance taxonomy
At minimum: EXACT_COMPUTE, LOGIC_DERIVED, DB_AUTHORITATIVE, RETRIEVED_EVIDENCE, MODEL_HYPOTHESIS, STALE, CONTRADICTED. Do not assume all deserve the same bias. Exact/derived authoritative records may receive strong positive prior; stale/contradicted should be suppressed or excluded.

## PRA/task-aware connection
Keep two concepts separate:
1. task-aware PRA decides whether a historical record is materialized at all;
2. provenance-aware SA decides how strongly a materialized authoritative record should influence generation.

Paper 2 may document this bridge but should not require PRA integration yet.

## TwIL comparison
Run the separate TwIL roadmap after dataset/routing factorization. Especially test TwIL on six-way classification, Datalog/graph formalization, oracle-engine payload generation, and TwIL + exact executor. Do not use TwIL to rescue an inadequate dataset.

## Decision rules
- CPU semantic trigger succeeds -> prefer it for those families.
- classifier succeeds, block fails -> classifier + deterministic/per-engine serializer.
- per-engine adapters succeed, joint adapter fails -> multi-protocol interference.
- larger/TwIL model succeeds -> native semantic capacity matters.
- runtime-copy solves COPY -> keep it as production default.
- provenance SA helps INTERPRET/CONTINUE -> carry to Paper 5/7.

## Immediate order
1. Expand dataset/paraphrases and run triviality audit.
2. Six-way classification-only experiment.
3. Oracle-engine payload experiment.
4. CPU trigger ladder.
5. Larger/TwIL diagnostic.
6. Attention+MLP and per-engine LoRA.
7. Fix A/B result-use baselines.
8. Attention instrumentation and causal masking.
9. Position/format ablations.
10. If supported, beta sweep.
11. If supported, layer-specific beta.
12. Only then learned provenance/KV embeddings.
13. Re-evaluate Paper 3 gate.

## Deliverables
Richer benchmark, semantic-diversity/triviality audits, trigger ladder, classification-vs-serialization decomposition, per-engine adapter comparison, LoRA target/layer ablations, larger/TwIL diagnostics, copy/interpret/continue result-use benchmark, SA instrumentation, causal masking, position/format ablations, runtime-copy baseline, provenance-beta results if gated, and updated Paper 3 go/no-go.
