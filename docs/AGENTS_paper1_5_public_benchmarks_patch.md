# AGENTS PATCH --- Paper 1.5: Public Epistemic/Retrieval Benchmarks

## Purpose

Strengthen Paper 1.5's controlled retrieval-risk experiments with public
benchmarks while preserving the paper's central question:

> retrieval requirement is not the same as model confidence.

Do not turn Paper 1.5 into a general multi-source routing paper.

## Primary benchmark candidates

### CRAG subset

Use a carefully selected subset emphasizing: - dynamic/current facts; -
source-dependent factual questions; - questions where parametric
knowledge may be stale; - answerable vs unavailable cases.

Primary use: compare confidence-triggered retrieval, semantic runtime
retrieval, generic `__retrieve(...)`, and oracle retrieval need.

Do not require full web infrastructure initially if CRAG's provided
mock/search interfaces or frozen evidence setup can support
reproducibility.

### KILT subset

Optional provenance-focused subset: - fact verification; - slot
filling; - open-domain QA.

Use only if operational cost is manageable.

## Public-benchmark conditions

1.  LLM only.
2.  Upfront retrieval.
3.  Generic `__retrieve(...)` tool.
4.  Paper 1.5 semantic runtime trigger.
5.  Confidence/FLARE-like trigger.
6.  Generic RETRIEVE block/intent when available.
7.  Oracle retrieval need.

Source routing should be fixed/oracle/runtime-owned; source-choice
learning belongs to Paper 2.5/3.5.

## Core measurements

-   retrieval-needed recall;
-   false retrieval rate;
-   UCR;
-   Authorized Commitment Coverage;
-   confidence-vs-epistemic quadrants;
-   answer accuracy;
-   abstention quality;
-   evidence override;
-   Automatic Rescue Rate relative to voluntary `__retrieve`.

## Context-sufficient controls

Construct matched public-derived controls where evidence is already
supplied in active context.

This is necessary to test:
`evidence requirement - available evidence = retrieval action`.

## Current/stale knowledge

Where benchmark metadata permits, identify: - temporally dynamic
questions; - likely stable questions.

The key Paper 1.5 question is whether model confidence remains a poor
proxy for external-evidence requirement.

## Long-form pilot

Add only a small developmental long-form subset: - model generates a
sentence/short paragraph; - retrieval need appears before a factual
commitment.

Full claim-level watchdog work belongs to Paper 3.5.

## Claim boundary

Public results should support robustness/generalization of
epistemic-trigger findings, not universal hallucination elimination.

## Deliverables

-   CRAG/KILT subset selection protocol;
-   frozen public benchmark artifacts;
-   confidence/semantic/tool/oracle comparison;
-   UCR and retrieval cost curves;
-   public-benchmark discussion in Paper 1.5;
-   explicit gate for whether Paper 3.5 claim-level control is
    warranted.
