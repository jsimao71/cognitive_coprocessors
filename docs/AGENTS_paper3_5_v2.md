# AGENTS --- Paper 3.5 v2

## Latent Epistemic Blocks: Claim-Level Parallel Control for Retrieval and Verification

## Reframe

The old plan learned DETECT/ROUTE/NORMALIZE including
DB/LEXICAL/VECTOR/WEB. Paper 2.5 currently gives no reason to learn
source identity because heuristic routing closes its oracle gap. Paper
1.5 establishes a different unresolved problem: epistemic requirement
differs from confidence, enforcement matters, and long-form generation
needs intervention before unsupported factual commitment.

Paper 3.5 therefore applies Paper 3's latent block controller to
RETRIEVE/VERIFY/HELP at claim level.

## Central question

Can a parallel hidden-state controller mark START/type/END of an
epistemic-risk span early enough to retrieve or verify before factual
commitment, without emitting retrieval-control tokens?

## Control

Primary: NONE, START_RETRIEVE, START_VERIFY, START_HELP, END. COMPUTE is
a hard-negative/shared-control type. Never predict DB/vector/web in R1.

## Evidence action

After a block: 1. infer required evidence type; 2. check active
context/task/PRA state; 3. reuse/materialize sufficient authoritative
evidence if present; 4. otherwise invoke runtime R2 source routing.

Thus: evidence requirement minus available evidence equals retrieval
action.

## Dataset

Use generation trajectories, not only questions. Include
current/source-specific/private/changed facts, attribution,
unavailable/conflicting evidence, context/PRA-sufficient cases,
quotations, hypotheticals, user-given/computed facts, COMPUTE needs, and
non-factual prose.

Annotate earliest safe START, latest acceptable START before unsupported
commitment, canonical END, first unsupported-value token,
evidence-available flag, and epistemic role.

## Architectures

-   Paper 1.5 semantic runtime rule;
-   paired token block;
-   fenced block;
-   label-only voluntary intent;
-   text-prefix sidecar;
-   final-layer hidden-state controller;
-   multi-layer hidden-state controller;
-   voluntary + latent watchdog hybrid;
-   FLARE-like confidence;
-   oracle timing/type.

Self-RAG-style control remains a prior-art comparator if faithfully
implemented, not the center.

## Multi-layer controller

Reuse Paper 3. Probe RETRIEVE/VERIFY/HELP START and END across depth.
Compare final layer, best single layer, fixed averaging, voting, learned
weights, tiny MLP. Allow separate START/END mixtures.

## Independent watchdog

Compare voluntary self-control, independent latent control, and hybrid.
Measure error correlation and Watchdog Rescue Rate. Independence matters
only if it catches voluntary misses.

## R2

Runtime owns source/capability selection using Paper 2.5 heuristic,
tokenizer/BM25, source-native rules, broadcast/oracle. Learned source
routing remains gated on a real heuristic-to-oracle gap.

## Runtime enforcement

If evidence is required but UNVERIFIED/CONFLICT/STALE/unavailable,
runtime may force abstention, qualification, conflict report, or more
evidence. The controller does not claim truth.

## Metrics

Span/type/FAR; Early Catch Rate; Late Catch Rate;
missed-before-commitment; lead tokens; Watchdog Rescue Rate; UCR;
Authorized Commitment Coverage; unnecessary retrieval; evidence
override; abstention P/R; controller/retrieval latency/cost;
cross-model/OOD transfer.

## Layer analysis

Plot epistemic decodability vs depth. Test whether RETRIEVE appears
earlier than VERIFY and whether multi-layer agreement predicts safe
intervention. Probe accuracy alone is descriptive.

## PRA/task awareness

Optional: before remote retrieval search active task/PRA state for
existing evidence; unrelated old task records remain unmaterialized.

## Integration adapter

After trigger quality is established, optionally train a separate
evidence-integration LoRA for INTERPRET/CONTINUE. Keep trigger and
assimilation factorized.

## Falsification

Not justified if semantic runtime rules match the UCR/cost frontier,
voluntary generic retrieval catches nearly all needs, watchdog errors
correlate with generator misses, FAR/abstention cost dominates rescues,
or hidden-state dependence is too model-specific.

## Gate

Source-specific learned router: NO-GO while Paper 2.5 heuristic closes
oracle gap. Generic latent epistemic block/watchdog: exploratory GO
because it tests timing/independence, not source identity.

## Immediate order

1.  Build claim-level span/timing benchmark from Paper 1.5.
2.  Reuse Paper 3 control implementation.
3.  Semantic-rule and FLARE baselines.
4.  Final-layer probe.
5.  Layer sweep + aggregation.
6.  Voluntary vs latent vs hybrid.
7.  Active-evidence sufficiency check.
8.  Runtime enforcement.
9.  Cross-model/OOD tests.
10. Integration LoRA only after trigger gate.
11. Re-evaluate Paper 4/7 integration.

## Deliverables

New `AGENTS_paper3_5.md`, `paper3_5.tex`, claim-level dataset, timing
annotations, multi-layer controller, voluntary/watchdog/hybrid
comparison, UCR-cost curves, cross-model analysis, enforcement results,
Paper 7 recommendation.
