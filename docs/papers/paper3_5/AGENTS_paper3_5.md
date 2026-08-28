# AGENTS — Paper 3.5

## Mission
Determine where epistemic-control competence should live: in the main
generative LM, a model-independent semantic sidecar, or a lightweight frozen-LM
hidden-state probe. Also test whether learned control can route among genuinely
heterogeneous evidence substrates.

This is the retrieval-track counterpart to Paper 3's learned computational semantic interrupts.

## Core decomposition
### DETECT
Does this generation state need external evidence?

### ROUTE
Which source is best: DB, lexical, vector, KG, web, or none?

### NORMALIZE
What canonical information need/query should be issued?

Evaluate all three separately.

## Candidate models
- FLARE-like confidence controller
- Self-RAG-style main-model learned retrieval/reflection controller
- sidecar text controller over ordinary LM prefixes
- frozen-main-model hidden-state sidecar
- oracle trigger/source/query

If exact Self-RAG training is infeasible, label the condition as a
Self-RAG-inspired approximation, not Self-RAG proper. Main-model training is
allowed only for this mandatory comparison; the sidecar and probe conditions
keep their main LM frozen.

## Dataset requirements
Examples must be generation prefixes, not just original questions.

Labels:
- retrieve/no-retrieve
- need category
- preferred source
- canonical query/IR
- freshness requirement
- evidence availability
- epistemic role
- miss-risk
- whether the need is computational rather than epistemic

Hard negatives:
- quoted claims
- hypotheticals
- counterfactuals
- user-given facts
- facts already in context
- generic non-factual prose
- computational needs better served by calculator/logic
- misleading lexical cues such as “latest” in non-factual contexts

## Epistemic roles
Support at least:
ASSERTION_PENDING, VERIFY_CLAIM, UNCERTAIN, QUOTED, HYPOTHETICAL, COUNTERFACTUAL, USER_GIVEN, COMPUTED_OR_DERIVED, NO_RETRIEVAL.

## Routing classes
Minimum:
DB, LEXICAL, VECTOR, WEB, NONE.
Optional: KG.

Support abstention and top-k suggestions.

## Cost-sensitive evaluation
False positives cost retrieval/latency.
False negatives cost unsupported claims.
Report threshold sweeps and cost curves, not only F1.

## Baselines
1. No retrieval
2. Upfront RAG
3. Explicit search/tool calls
4. Paper 2.5 heuristic reflex
5. FLARE-like confidence controller
6. Self-RAG-style main-model learned controller
7. Text sidecar: learned DETECT/ROUTE/NORMALIZE
8. Frozen-LM hidden-state sidecar
9. Oracle trigger/source/query

## Sidecar hypotheses and ablations
Model independence, upgradeability, heterogeneous-routing specialization,
small-LM leverage, and lower retraining cost are hypotheses, not assumptions.
Compare confidence-only, semantic-only, and semantic+confidence sidecars at
matched controller complexity and runtime cost.

Train the text sidecar on traces from one or several model families and test on
another frozen family. Cross-model transfer is evidence for reusable runtime
control; failure is a valid architectural result.

## Metrics
DETECT P/R, AUROC, AUPRC; calibration; selective risk/coverage; routing top-1/top-k; normalization correctness; unsupported claims; final accuracy; retrieval count/fanout; latency/cost; OOD degradation; high-risk missed claims.

Mandatory analyses are high-confidence/high-risk recall,
low-confidence/low-risk false retrieval, source-routing confusion matrices,
per-source calibration, model-size interactions, OOD paraphrase/domain transfer,
and the cost versus unsupported-claim Pareto frontier.

## Generalization tests
Unseen paraphrases, entity names, relation wording, domains, freshness windows, longer generation trajectories, and multiple base-model sizes.

## Failure taxonomy
Missed semantic need, false interrupt, wrong epistemic role, wrong source, malformed query, stale/conflicting evidence, correct evidence ignored, lexical overfit.

## Hard scope
- No joint training of the main LLM except the declared Self-RAG-style comparison.
- No transactional/backtracking contribution.
- No PRA/native KV dependency.
- No silent promotion of low-confidence interpretation into hard fact.
- Keep explicit tool calling as fallback.

## Evidence gate
Paper 4 should introduce unified transactional state only after Papers 3 and 3.5 show enough persistent compute/retrieval state that provenance and retraction are concrete problems.

Do not claim relevance or support labels as novel. Paper 4's target is persistent
cross-source provenance, dependency propagation, transactional scopes,
retraction, and backtracking.

## Falsification
The sidecar architecture is weakened if main-model Self-RAG-style control wins
the quality/cost frontier, sidecars do not transfer, heterogeneous routing adds
no value, confidence explains semantic gains, or hidden-state dependence makes
the controller too model-specific. Sidecar, main-model, hybrid, and
task/model-size-dependent winners are all valid outcomes.

## Deliverables
`paper3_5.tex`, labeled generation-prefix dataset, FLARE-like,
Self-RAG-style/text-sidecar/hidden-state comparisons, cross-model transfer and
OOD splits, per-source calibration, component traces, and final Pareto tables and
plots.
