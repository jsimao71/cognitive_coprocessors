# PATCH — Paper 0: Generic Cognitive Intent, Deployment Modes, Optional PRA

## Architectural change
Do not define Cognitive Coprocessors as requiring engine-specific tokens, engine-specific LoRA/SFT, a fixed registry, or PRA.

Introduce a stable coarse model-facing protocol:
- COMPUTE
- RETRIEVE
- VERIFY
- HELP

The LLM may signal only the kind of cognitive assistance required. The runtime resolves the concrete capability from a deployment-specific registry.

### COMPUTE
Deterministic/structured computation: arithmetic, dates, units, logic, graph closure, SMT, algebra, analytics.

### RETRIEVE
External evidence/data: DB, lexical, vector, web, Iceberg, documents.

### VERIFY
Validate an existing candidate claim/result using evidence, recomputation, logic, provenance, or already available state.

### HELP
Catch-all cognitive page fault: the model recognizes that ordinary generation is insufficient but does not know which assistance family is best.

## Two candidate syntaxes

### A — paired tags
`<COMPUTE>` payload `</COMPUTE>`

Likewise RETRIEVE/VERIFY/HELP.

Pros: explicit boundaries, incremental parsing.
Cons: extra tokens and closing-tag overhead.

### B — fenced cognitive block
Conceptually:
three-backtick COMPUTE
payload
three-backtick

Likewise RETRIEVE/VERIFY/HELP.

Pros: likely economical, familiar to instruct models, easy boundary detection.
Cons: ordinary markdown may contain fences; tokenizer cost varies.

Do not declare a winner in Paper 0. Papers 2/2.5 measure both.

Also permit a label-only variant where runtime receives the original active prompt as payload.

## Two-stage routing
R1:
`generation -> NONE | COMPUTE | RETRIEVE | VERIFY | HELP`

R2:
`intent + active task + registry + policy + state -> capability/capability shortlist`

R1 is stable and small. R2 can use deterministic rules, LLM-token/NLP-token BM25, CPU classifiers, PRA capability retrieval, optional learned routing, or conventional tools.

## Capability, not engine identity
Prefer semantic capability names such as arithmetic, transitive closure, Horn entailment, structured aggregate, current factual lookup. Multiple implementations may satisfy one capability.

The runtime chooses based on available state, exactness, latency, cost, policy, credentials, and deployment preference.

## Deployment modes
1. Runtime-trigger CogCoproc — no model modification.
2. ICL CogCoproc — short generic-intent contract.
3. CogCoproc adapter — optional LoRA/SFT for generic intent/result consumption.
4. PRA + CogCoproc ICL — PRA progressively materializes relevant capability/skill interfaces.
5. PRA + CogCoproc adapter — learned coarse intent + PRA dynamic discovery/state.
6. PRA only.
7. CogCoproc only.
8. Conventional tools fallback.

These are interoperable deployment modes, not mutually exclusive editions.

## PRA is optional and orthogonal
Without PRA, registry and routing stay outside the prompt and selected capability metadata can be inserted conventionally.

With PRA:
- capability/tool/skill descriptions are persistent resources;
- only relevant candidates are materialized;
- old cognitive records may remain persistent but unmaterialized;
- task-aware PRA suppresses unrelated historical state.

Thus large registry size need not imply large active context.

## Three knowledge layers
1. Stable cognitive intent — prompt/LoRA/native model.
2. Dynamic capability knowledge — runtime/PRA/skills.
3. Cognitive state — typed results/evidence/provenance/dependencies/task scope.

Do not conflate them.

## Adapter role
LoRA is an optional accelerator, not the capability registry. Adding a new coprocessor should not require retraining the generic adapter.

Engine-specific adapters remain optional optimizations.

## Series implications
- Paper 1: specific learned blocks can work.
- Paper 1.5: runtime epistemic policy can outperform learned retrieval policy.
- Paper 2: test generic COMPUTE/VERIFY/HELP vs engine-specific routing.
- Paper 2.5: test generic RETRIEVE/VERIFY/HELP vs source-specific routing.
- Papers 3/3.5 proceed only if coarse intent + runtime routing leaves meaningful headroom.
- Papers 5/7 may add PRA progressive capability/state materialization.

## Position-paper claim
Preferred scalable architecture:

neural semantic core
-> coarse cognitive interrupt
-> runtime/PRA capability resolution
-> specialized machine
-> typed cognitive state
-> selective materialization/continuation

rather than baking every engine interface into model weights.

## Manuscript patch
Add:
- generic intent protocol;
- syntax alternatives;
- R1/R2 factorization;
- deployment-mode table;
- optional PRA integration;
- dynamic registry/product extensibility;
- LoRA as optional accelerator;
- conventional tools as universal fallback.
