# AGENTS PATCH — Paper 3.5: Independent Epistemic Interrupts and Learned Retrieval Routing

## Purpose
Reframe Paper 3.5 after the strong learned-block result from Paper 1.

Unlike compute-side Paper 3, Paper 3.5 retains a strong independent motivation even if LoRA retrieval blocks work well:

> The same generator that is capable of confidently hallucinating cannot always be relied upon to voluntarily request the evidence needed to constrain its own factual commitment.

Paper 3.5 therefore studies an independent epistemic monitor/watchdog, not merely a more convenient way to emit retrieval blocks.

## Central hypothesis
An independent semantic epistemic controller can detect retrieval-required factual commitments that the generator fails to externalize, particularly high-confidence/current/source-specific claims, and can reduce unsupported commitments at acceptable retrieval and compute cost.

## Primary architecture
Use learned retrieval blocks as the normal voluntary path where Paper 1.5/2.5 support them:

generator + interface policy
    -> optional typed retrieval block
    -> runtime source router/retriever
    -> evidence state

Add an independent watchdog:

ordinary generation without retrieval block
    -> epistemic monitor
    -> retrieval-required?
    -> source/query path
    -> evidence gate
    -> continue / qualify / abstain

The watchdog should primarily recover missed epistemic actions rather than duplicate every voluntary retrieval.

## Why Paper 3.5 remains distinct
Compute:
- if model forgets calculator, answer may be wrong;
- learned block may already solve most cases.

Epistemic:
- model can be highly confident and still rely on stale/non-authoritative weights;
- confidence is not evidence sufficiency;
- failure to retrieve can create unsupported factual assertions;
- independent verification provides architectural diversity.

Paper 1.5's high-confidence/retrieval-required cases are the key precursor.

## Entry gates
Paper 3.5 is justified if Paper 1.5/2.5 establish at least one:
- confirmed high-confidence retrieval-required failure;
- voluntary retrieval-block miss;
- heuristic semantic-risk gap;
- heuristic source-routing gap to oracle;
- excessive source/tool context burden;
- unsupported-commitment reduction headroom.

## Research questions
1. Does a learned independent monitor catch factual commitments missed by voluntary retrieval blocks?
2. Does it outperform confidence-only triggering?
3. Can it distinguish retrieval-required from merely low-confidence text?
4. Can it select source family when Paper 2.5 establishes heterogeneous-source value?
5. Does watchdog+voluntary retrieval dominate either alone?
6. Can runtime enforcement drive unsupported commitment close to zero without excessive abstention/retrieval?

## Conditions
1. LLM only.
2. Anti-hallucination prompt.
3. Upfront RAG.
4. Explicit retrieval tool.
5. FLARE-like confidence trigger.
6. Voluntary learned retrieval block.
7. Heuristic semantic-risk trigger.
8. Learned epistemic watchdog.
9. Voluntary block + watchdog.
10. Watchdog + confidence.
11. Oracle retrieval-required detector.
12. Oracle source/query.
13. Broadcast/upfront evidence where relevant.

## Detection target
Prefer a simple typed decision:

- NO_RETRIEVAL
- RETRIEVAL_REQUIRED
- VERIFY_CLAIM

Optionally include epistemic reason:
- CURRENT
- SOURCE_SPECIFIC
- PRIVATE
- CHANGED
- ATTRIBUTION
- UNVERIFIED
- CONFLICT_SENSITIVE

Do not require the monitor to generate the factual answer.

## Source routing
Only add learned source-family routing if Paper 2.5 demonstrates:
- heterogeneous source-native value;
- oracle routing headroom;
- heuristic routing gap;
- broadcast is not already sufficient.

Possible source labels:
- DB
- LEXICAL
- VECTOR
- KG
- WEB

Keep credentials, source implementations, query validation, and evidence state in runtime.

## Monitor placement ladder
Compare progressively:

1. transparent heuristic semantic rules;
2. tiny CPU text classifier;
3. small neural sidecar;
4. LoRA/head on main model text representations;
5. hidden-state probe/router.

Do not jump directly to hidden-state routing.

## Independence experiment
A key Paper 3.5 experiment should compare:

### Self-monitoring
The generator itself emits retrieval block/control.

### Independent monitoring
A separate sidecar/model/head judges the emerging claim.

### Hybrid
Generator requests evidence when it knows it needs it; independent monitor catches omissions.

Measure error correlation.

The strongest case for independent monitoring is when its misses are not strongly correlated with generator misses.

## Hallucination-oriented metrics

### Unsupported Commitment Rate
unsupported factual commitments / retrieval-required opportunities

### Authorized Commitment Coverage
supported answers emitted / answerable retrieval-required opportunities

### Missed Evidence Need Rate
retrieval-required opportunities with no retrieval before commitment

### Unnecessary Retrieval Rate
retrieval-not-required opportunities that trigger retrieval

### Watchdog Rescue Rate
voluntary retrieval misses correctly recovered by watchdog

### Evidence Override Rate
correct/contradictory evidence subsequently ignored or overridden

### Abstention precision/recall
for unavailable/conflicting evidence.

Plot UCR vs retrieval rate/cost.

## Runtime epistemic enforcement
Paper 3.5 should treat enforcement as distinct from detection.

If:
- evidence is required; and
- configured evidence is unavailable/conflicting/insufficient;

runtime may constrain the allowed final state to:
- abstain;
- qualify;
- report conflict;
- request more evidence.

Do not claim runtime knows objective truth. It enforces a support policy relative to authoritative/configured sources.

## Training data
Use leakage-audited semantic-policy data.

Train decisions/requests, not answer values.

Include:
- current/source-specific facts;
- changed familiar facts;
- private DB facts;
- prompt-contained answers;
- stable familiar facts;
- lexical freshness distractors;
- hypotheticals;
- quotations;
- computational-but-not-epistemic needs;
- unavailable/conflicting evidence.

Use disjoint entities/records/answer values where possible.

## Multi-model evaluation
At minimum:
- Qwen small model;
- SmolLM2 or another second family;
- Gemma if validated.

Because confidence distributions differ, calibrate confidence baselines per model.

The semantic monitor should ideally generalize better than raw confidence thresholds across families.

## CPU economics
A major Paper 3.5 question is whether epistemic monitoring can run cheaply off accelerator.

Measure:
- monitor CPU time;
- monitor model parameters;
- GPU/XPU hooks if any;
- retrieval calls;
- network latency;
- neural tokens saved/added;
- unsupported claims prevented.

A cheap CPU-side classifier with strong rescue value may be preferable to a hidden-state GPU router.

## Observability and audit
Every watchdog intervention must log:
- triggering span/state;
- reason/category;
- confidence;
- chosen source;
- query;
- evidence status;
- enforcement action;
- final claim.

Production epistemic control must be inspectable.

## Falsification
Independent semantic monitoring is not justified if:
- voluntary LoRA retrieval blocks already catch nearly all evidence needs;
- monitor errors are highly correlated with generator misses;
- confidence-only triggering matches the UCR/cost frontier;
- false retrieval/abstention overwhelms rescued hallucinations;
- source routing adds no value;
- hidden-state integration adds complexity without substantial benefit.

If voluntary blocks are sufficient, Paper 3.5 should say so.

## Relationship to Paper 1.5
Paper 1.5 establishes confidence != epistemic risk and tests one-source detection.

Paper 3.5 learns/generalizes the epistemic policy and tests independence/watchdog behavior.

## Relationship to Paper 2.5
Paper 2.5 establishes whether heterogeneous sources matter and quantifies source-routing headroom.

Paper 3.5 only learns source routing if that gate passes.

## Relationship to Paper 7
Paper 7 should use the simplest epistemic architecture supported here:
- voluntary blocks only;
- heuristic watchdog;
- learned CPU sidecar;
- hidden-state monitor;
or hybrid.

Do not force hidden coupling into the integrated runtime.

## Immediate execution order
1. Complete confirmatory Paper 1.5 retrieval-required experiment.
2. Complete Paper 2.5 oracle source-routing gate.
3. Measure voluntary retrieval-block miss rate.
4. Build oracle watchdog headroom.
5. Implement heuristic/tiny classifier watchdog.
6. Compare voluntary vs independent vs hybrid.
7. Add learned source routing only if Paper 2.5 gate passes.
8. Escalate to hidden-state monitoring only if cheaper methods leave substantial headroom.
9. Produce explicit Paper 7 recommendation.

## Deliverables
- revised Paper 3.5 manuscript/AGENTS;
- independent-monitor benchmark;
- voluntary-vs-watchdog-vs-hybrid comparison;
- UCR/coverage/retrieval-cost Pareto curves;
- error-correlation analysis;
- runtime enforcement ablation;
- optional learned heterogeneous source router;
- CPU/GPU cost accounting;
- engineering-complexity comparison;
- explicit Paper 7 epistemic-control recommendation.
