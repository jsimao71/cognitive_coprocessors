# AGENTS — Paper 1 Next Iteration: ICL → Multi-Model → LoRA

## Mission
Continue Paper 1 from the current held-out XPU result without jumping directly to LoRA.

Current evidence:
- Qwen3-0.6B hard held-out:
  - normalized reflex 75.0%
  - explicit tool 62.5%
  - matched prompt 50.0%
  - LLM-only 43.8%
  - strict reflex 37.5%
  - calculator block 12.5%
  - oracle 87.5%
- normalized reflex had 5 paired gains, 0 losses, exact McNemar p=0.0625 over 16 pairs;
- calculator execution was exact whenever reached;
- calculator-block failure was mostly protocol/payload formation: the model often copied the literal placeholder `EXPRESSION`, so normalization failed before execution;
- all 12 controls had zero false interventions.

This iteration must answer, in order:
1. Can stronger ICL teach the block protocol without parameter updates?
2. Does block following improve naturally with model size/capability?
3. If a clear interface gap remains, can a small LoRA/SFT teach only the protocol cheaply?

Primary scientific question:
> Is semantic execution-block competence an emergent/few-shot interface skill of sufficiently capable pretrained LMs, or does it require lightweight adaptation?

## Stage 0 — Preserve current evidence
Do not overwrite `hard_heldout_xpu` artifacts or the current prompt/version.

Any prompt change below defines a new developmental protocol and requires a new held-out set after freezing.

Also rescore existing generations with a condition-independent answer extractor that recognizes forms such as `Response: N`, while preserving the originally reported metrics. Do not use rescoring to reinterpret block protocol success; block execution rate remains the key interface metric.

## Stage 1 — Better ICL for calculator blocks

### Goal
Determine whether Qwen3-0.6B failed because of a weak abstract placeholder prompt rather than lack of protocol-following capability.

The current placeholder:
```text
```calculator
EXPRESSION
```
```
is an attractive literal-copy target for a small model.

### ICL-A — one concrete demonstration
Example:
```text
When arithmetic should be executed by the calculator, copy the complete arithmetic expression inside exactly one calculator block.

Example:
Expression: 17 * 23

Output:
```calculator
17 * 23
```

Now solve:
Expression: 15246377 * 746647383

Output:
```

### ICL-B — two diverse demonstrations
Use one simple multiplication and one nested/mixed expression:
```text
Expression: (19 + 7) * 43

Output:
```calculator
(19 + 7) * 43
```
```

### ICL-C — positive + negative demonstrations
Teach both:
- arithmetic request -> calculator block;
- prose containing numbers but no arithmetic operation -> no calculator block.

Use existing controls where practical.

### ICL-D — verbatim-copy instruction
Minimize semantic burden:
> Copy the arithmetic expression shown after `Expression:` verbatim between the calculator fences. Do not calculate it yourself.

Then give a concrete demonstration.

### ICL-E — seeded opening block
The harness provides:
```text
```calculator
```
The model emits:
1. the expression;
2. the closing fence.

This isolates entering calculator mode from formatting the payload. Treat it as a distinct condition.

### ICL-F — fully seeded wrapper diagnostic
For diagnosis only, harness supplies both wrapper boundaries and the model writes only the payload. This estimates whether payload copying, rather than block syntax, is the remaining problem. Do not treat it as the target architecture.

## Block protocol metrics
Add:
- block_open_rate
- block_close_rate
- block_payload_present_rate
- block_payload_exact_rate
- block_payload_semantically_equivalent_rate
- block_execution_rate
- full-expression selection
- normalization correctness
- calculator correctness
- result-use rate
- final-answer accuracy
- false-block rate on controls
- generated tokens
- wall time

Do not reduce block quality to final exact match.

## Stage 1 gate
Freeze the best generic ICL prompt only if it:
- materially improves valid block execution over 12.5%;
- preserves zero/near-zero false interventions;
- does not leak the target expression through demonstrations;
- works across arithmetic structures.

A useful developmental target is >=75% valid block execution and 0 false interventions on the current controls. This is not a publication threshold.

## Stage 2 — Multi-model ICL evaluation on XPU

### Goal
Test whether execution-block competence improves with model capability before introducing LoRA.

Use the same frozen ICL prompt and identical runtime across models. Do not hand-tune the primary prompt separately per model.

### Recommended run order
Start with the 196-row held-out-scale diagnostic, not the 37,800-generation full matrix:

1. Qwen3-1.7B
2. Llama 3.2-1B
3. Gemma 3-1B
4. Llama 3.2-3B
5. Gemma 2-2B
6. Qwen3-4B

Authorization-gated models should be skipped cleanly until available.

### Why this order
- Qwen3-1.7B: clean matched-family capability comparison.
- Llama/Gemma ~1B: family effect at comparable small scale.
- Llama 3.2-3B / Gemma 2-2B: stronger small-model regime.
- Qwen3-4B: matched-family higher-capability endpoint.

### XPU smoke gate before each 196-row run
Run:
- 2–4 arithmetic examples;
- 2 controls;
and verify:
- model loads on XPU;
- no OOM;
- correct chat template;
- greedy deterministic decoding;
- block recognition;
- runtime reinjection;
- peak memory if available.

Record device, dtype, model/tokenizer revision, chat-template settings, thinking on/off, wall time, and peak memory where possible.

## Multi-model conditions
At minimum:
1. LLM-only
2. matched prompt
3. explicit calculator tool
4. normalized reflex
5. frozen best calculator-block ICL
6. oracle

Strict reflex may remain for continuity but can be secondary if runtime cost is material.

## Primary multi-model questions

### Q1 — Does block protocol compliance scale with capability?
Plot valid block execution and payload correctness by model.

### Q2 — Does normalized reflex remain the best arithmetic interface?
Compare final accuracy, execution rate, tokens, latency.

### Q3 — Does explicit tool calling improve faster with capability than blocks?
Larger models may simply become excellent tool users. This is a valid outcome.

### Q4 — Are block skills family-specific?
Compare Qwen/Llama/Gemma at similar sizes.

### Q5 — Is there a capability threshold?
Estimate the smallest model/regime where:
- payload correctness >=90%;
- false-block rate remains near zero;
- result use remains reliable.

## Runtime discipline
The correctness-first backend recomputes full prefix per token. For now:
- Phase A: run only the 196-row diagnostic per model.
- Phase B: selectively expand promising model/interface combinations.
- Phase C: optimize KV-cache interception only after semantic/interface evidence exists.

Do not conflate runtime optimization with Paper 1’s interface result.

## Stage 3 — Decide whether LoRA is justified

Do not train LoRA just because Qwen3-0.6B was poor.

### Regime A — larger models solve blocks with ICL
Then LoRA asks:
> Can a small model be taught the protocol cheaply enough to match larger-model interface reliability?

This is the strongest LoRA motivation.

### Regime B — all models struggle
Reconsider the block language/prompt first. Do not use LoRA to rescue a poor protocol.

### Regime C — family-dependent performance
Diagnose tokenizer/template/instruction effects before adaptation.

### Regime D — blocks work but explicit tools remain strictly better
LoRA is low priority unless blocks show measured token/routing/system advantages.

## Stage 4 — Minimal LoRA/SFT experiment

### Goal
Teach interface behavior, not arithmetic.

Example:
Input:
`Compute the exact value of 927364 * 82719`

Target:
```calculator
927364 * 82719
```

The calculator remains responsible for the answer.

### Training data
Include:
- different digit widths;
- nested parentheses;
- mixed operations;
- exact division;
- long intermediates;
- multiple-operation temptations;
- LaTeX/Unicode source forms;
- negative prose/versions/dates/ranges/quoted arithmetic/code;
- cases where only the requested full expression should be inside the block.

### Leakage controls
- disjoint expression seeds;
- disjoint exact operands;
- held-out structures where feasible;
- no held-out benchmark item in training;
- overlap audit.

### LoRA model order
Start with the model most clearly motivated by Stage 2.
Likely first:
1. Qwen3-0.6B, if larger Qwen models show ICL block competence.
2. One second small family only after the first result.

Do not train all families initially.

## LoRA evaluation
Compare same base checkpoint:
1. base + best ICL
2. LoRA + same ICL
3. LoRA + minimal instruction
4. normalized reflex
5. explicit tool
6. oracle

Measure:
- block protocol decomposition;
- end accuracy;
- false blocks;
- generated tokens;
- LoRA parameter count;
- training tokens/time;
- inference overhead;
- OOD protocol generalization.

Key result:
> Small parameter adaptation improves machine-protocol reliability while computational competence remains external and exact.

Do not claim LoRA learned arithmetic unless separately tested.

## ICL vs LoRA equivalence test
Compare:
- base model + 2-shot ICL
- LoRA model + 0-shot/minimal instruction

This tests whether interface knowledge is better stored transiently in context or persistently in model weights.

## Relation to Self-RAG/control tokens
Treat Self-RAG-like control tokens as prior evidence that models can learn special inference actions.

Calculator blocks generalize the idea toward typed local computational regions. Do not claim special control tokens or trained tool syntax as novel.

## Relation to future multi-engine syntax
If calculator blocks work, Paper 2 may test:
```calculator
...
```
```datalog
...
```
```graph
...
```

Do not implement a broad `lll-lang` yet. Design blocks so they can later evolve toward typed engine families, bounded payload grammars, budgets, deterministic local execution, and typed result records.

## Revised Paper 2 evidence gates
Paper 2 remains gated until the compute-side interface question is sufficiently resolved.

Any of these can resolve it:
1. replicated normalized-reflex gains;
2. reliable frozen ICL block protocol on one or more models;
3. small LoRA block adaptation with a plausible path to multiple engines.

Paper 2 does not require blocks specifically.

General rule:
> Do not generalize an unresolved interface.

## Statistical plan
The current 16 arithmetic examples are too small for strong paired conclusions.

After ICL/model development is frozen:
- create a new untouched confirmatory arithmetic set;
- use enough paired examples to resolve 20–30 point gains with reasonable power;
- use exact paired tests;
- report effect sizes and confidence intervals;
- for greedy decoding, use multiple dataset seeds/examples rather than treating decode seeds as independent replication.

The 196-row multi-model phase is developmental/diagnostic.

## Failure taxonomy

### ICL/protocol
- copies placeholder;
- copies demonstration expression instead of target;
- omits opening fence;
- omits closing fence;
- writes prose inside block;
- writes the answer instead of expression;
- emits multiple blocks;
- emits an intermediate/partial expression;
- payload semantically wrong.

### Runtime
- recognition failure;
- normalization failure;
- engine failure;
- reinjection failure.

### Model integration
- result ignored;
- result overridden;
- redundant post-result arithmetic;
- final extraction error.

### Controls
- false calculator block;
- accidental arithmetic trigger;
- quoted-example trigger.

## Required artifacts
Per model:
- manifest
- prompt version
- predictions JSONL
- traces JSONL
- summary
- paired analysis
- block protocol failure table
- XPU smoke report
- wall-time/memory report

Across models:
- model comparison table
- block execution vs model/capability plot
- accuracy vs interface plot
- token/latency Pareto plot
- failure-mode heatmap

## Immediate execution order
1. Fix/extend endpoint answer extraction if needed; preserve old metrics and rescore.
2. Test ICL-A through ICL-E on a small developmental subset with Qwen3-0.6B.
3. Freeze the best generic ICL prompt.
4. Smoke and run 196-row experiments:
   - Qwen3-1.7B
   - Llama 3.2-1B
   - Gemma 3-1B
   - Llama 3.2-3B
   - Gemma 2-2B
   - Qwen3-4B
5. Compare block-protocol scaling and normalized-reflex scaling.
6. Decide whether LoRA is justified.
7. Only then run a minimal LoRA/SFT interface experiment.

## Hard scope boundaries
- calculator only;
- no learned general router;
- no hidden-state interrupt;
- no Paper 2 Datalog/graph empirical sweep yet;
- no PRA/KV integration;
- no arbitrary generated Python;
- no 37,800-generation/model matrix before the diagnostic phase establishes value;
- no separate hand-tuned primary prompt per model;
- do not train arithmetic answers into LoRA.

## Deliverables
- updated Paper 1 AGENTS/context;
- improved ICL prompt suite;
- block protocol decomposition metrics;
- XPU smoke configs per model;
- 196-row multi-model diagnostic results;
- model-capability/block-reliability analysis;
- explicit ICL-vs-LoRA decision gate;
- optional minimal LoRA/SFT dataset/training only after the gate;
- updated `paper1.tex` concluding on:
  1. normalized reflex;
  2. zero/few-shot semantic blocks;
  3. model capability effects;
  4. whether lightweight adaptation is necessary.
