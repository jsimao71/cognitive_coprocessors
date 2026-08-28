# Paper 3.5 AGENTS Patch — Self-RAG-Aware Learned Epistemic Interrupts

## Revised focus
Self-RAG already establishes learned adaptive retrieval/reflection. Do not claim “learning when to retrieve” as novel.

Central architectural question:
**Where should epistemic-control competence live: main generative LM, external semantic sidecar, or lightweight hidden-state probe?**

Systems question:
**Can learned control route among heterogeneous evidence substrates?**

## Mandatory comparisons
A. FLARE-like confidence controller
B. Self-RAG-style main-model learned retrieval/reflection controller
C. sidecar text controller: ordinary LM prefix -> specialist DETECT/ROUTE/NORMALIZE
D. frozen-LM hidden-state sidecar
E. oracle trigger/source/query

If exact Self-RAG training is infeasible, label the approximation clearly; do not call it Self-RAG proper.

## Heterogeneous action space
At minimum: NONE, DB, LEXICAL, VECTOR, WEB; optional KG.
Do not reduce to binary retrieve/no-retrieve.

## Hypotheses for sidecar control
Possible advantages: model independence, upgradeability, heterogeneous routing specialization, benefit for small LMs, lower retraining cost. These are hypotheses, not assumptions.

## Training labels
For each generation prefix: evidence need, epistemic role, preferred source, canonical query/IR, freshness, availability, miss-risk, and whether the need is computational rather than epistemic.

## Confidence feature ablation
Compare semantic-only, confidence-only, and semantic+confidence sidecars.

## Critical analyses
- high-confidence/high-risk recall
- low-confidence/low-risk false retrieval
- routing confusion matrix
- calibration by source
- model-size effects
- OOD paraphrase/domain transfer
- cost vs unsupported-claim Pareto frontier

## Cross-model transfer
Train text sidecar on one/several model traces and test on another frozen model family. Success would support epistemic control as a reusable runtime capability.

## Hidden-state probe
Frozen main model; lightweight probe; compare matched complexity/cost to text sidecar. No native-interface claim unless it wins.

## Paper 4 relation
Do not claim relevance/support labels as novel. Paper 4 should focus on persistent cross-source provenance, dependency propagation, transactional scopes, retraction and backtracking.

## Falsification
Sidecar architecture is weakened if main-model Self-RAG-style control dominates quality/cost, sidecars do not transfer, heterogeneous routing adds no value, confidence explains semantic gains, or hidden-state dependence makes the controller too model-specific.

## Result discipline
Sidecar, main-model, hybrid, or task/model-size-dependent winners are all valid outcomes.
