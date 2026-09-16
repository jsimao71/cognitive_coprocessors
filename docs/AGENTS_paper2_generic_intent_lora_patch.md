# PATCH — Paper 2: Generic Cognitive-Intent LoRA

## Motivation
The five-engine learned router couples model weights to concrete engine identity. Test a simpler stable interface:
- NONE
- COMPUTE
- RETRIEVE
- VERIFY
- HELP

Paper 2 primarily evaluates NONE/COMPUTE/VERIFY/HELP; RETRIEVE is included as a cross-domain negative/control. The runtime selects the concrete computational capability.

## Hypothesis
Coarse intent should reduce class interference, remain stable under registry changes, and avoid retraining when coprocessors are added.

## Syntax conditions

### A — paired tags
`<COMPUTE>` natural-language need `</COMPUTE>`

and corresponding RETRIEVE/VERIFY/HELP tags.

### B — fenced blocks
Conceptually:
three-backtick COMPUTE
natural-language need
three-backtick

Measure rather than assume token economy.

### C — label only
Emit only COMPUTE/RETRIEVE/VERIFY/HELP/NONE. Runtime uses the original active task as payload.

This is likely the lowest-token condition and must be included.

## Dataset
Create a NEW freeze.

### COMPUTE
Mix calculator/date/units/graph/Datalog examples but targets contain only COMPUTE, never engine identity.

### VERIFY
Candidate arithmetic/logical/derived results that require checking.

### HELP
Tasks where assistance is needed but the appropriate mechanism is deliberately ambiguous or unfamiliar.

### RETRIEVE
Current/source-specific factual needs, primarily as cross-category controls.

### NONE
Strong ordinary-language and quoted/fence controls.

## Registry-change experiment
Train R1 while the runtime registry contains only:
- calculator
- date
- units

Test without retraining after adding:
- graph
- Datalog
- optionally later SMT/algebra.

If R1 remains stable, this directly supports registry-independent model interfaces.

## Conditions
1. runtime-only trigger;
2. generic ICL;
3. generic label-only LoRA;
4. paired-tag LoRA;
5. fenced-block LoRA;
6. original six-way engine LoRA;
7. oracle generic intent.

Start with rank-8 attention LoRA. Add MLP targets only if positive recall is inadequate without FAR inflation.

## R2 runtime routing
After COMPUTE compare:
- T1 lexical/regex;
- semantic rules;
- model-token/NLP-token BM25;
- capability-level heuristic;
- oracle capability.

Do not ask the model for engine identity in the primary generic condition.

## Capability-level scoring
Score requested semantics, not only gold engine identity. Graph/Datalog substitution is valid when semantics are preserved. Keep engine-identity metrics diagnostically.

## Metrics
R1:
- intent accuracy;
- COMPUTE recall;
- RETRIEVE/COMPUTE confusion;
- VERIFY/HELP recall;
- NONE FAR;
- malformed boundary rate.

Syntax:
- exact tokenizer token cost per model;
- generated tokens;
- parse success;
- accidental fence/tag activation;
- latency.

R2:
- capability selection;
- runtime exactness;
- route latency.

End-to-end:
- false intervention;
- model/engine calls;
- CPU/XPU time.

## Token-economy experiment
For Qwen/SmolLM2/Gemma tokenizers measure exact cost of:
- label only;
- fenced syntax;
- paired tags.

Expected but unproven: label-only < fence < paired tags.

## Iterative JIT reasoning
Do not require full multi-engine planning.

Preferred loop:
model -> COMPUTE/HELP -> runtime -> typed result -> CONTINUE -> optional next COMPUTE/VERIFY/HELP.

This is the JIT-COT/JIT-reasoning-assistance path.

## Result-use
Generic intent does not solve CONTINUE. Keep runtime-copy for final exact outputs and existing INTERPRET/CONTINUE diagnostics.

## PRA compatibility
Paper 2 need not implement PRA. Record the future hook:
COMPUTE/HELP -> runtime shortlist -> optional PRA materializes relevant capability skills -> resolve/execute.

Generic-intent LoRA must work without PRA.

## Gate
Predeclare thresholds. Suggested:
- COMPUTE recall >= .95;
- NONE FAR <= .05 preferred, <= .10 maximum;
- low RETRIEVE/COMPUTE confusion;
- no material R1 degradation after registry expansion;
- R1+R2 at least matches engine-specific learned routing with less registry coupling.

## Immediate order
1. freeze generic-intent dataset;
2. measure syntax tokenization;
3. ICL syntax comparison;
4. Qwen label-only LoRA;
5. Qwen paired-tag and fenced LoRA;
6. replicate best condition on another family;
7. registry-expansion test without retraining;
8. connect to T1/token-BM25/capability R2;
9. re-evaluate Paper 3 gate.

## Deliverables
- generic-intent specification;
- syntax/token report;
- dataset/audit;
- ICL + LoRA comparisons;
- registry-change experiment;
- R1/R2 factorized metrics;
- updated Paper 2 conclusion.
