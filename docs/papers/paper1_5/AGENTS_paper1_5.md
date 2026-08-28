# AGENTS — Paper 1.5

## Mission
Test the smallest credible **epistemic coprocessor** by separating semantic
epistemic-risk detection from confidence-based active retrieval during
generation, using one controlled retrieval source.

## Core hypothesis
Model uncertainty and need for evidence are not equivalent. Semantic
epistemic-risk rules may catch confident stale or source-dependent commitments
that a FLARE-like uncertainty trigger misses, while confidence may fire on
low-confidence generations that need no external evidence.

## Required implementation
- One controlled retrieval source.
- Incremental generation watcher.
- Transparent trigger rules.
- Typed retrieval-request IR.
- Compact evidence reinjection.
- Status: SUPPORTED / CONTRADICTED / UNVERIFIED / STALE / AMBIGUOUS / CONFLICT.
- Full provenance logging.
- A documented FLARE-like controller that predicts/inspects upcoming generation,
  triggers below a token/model-confidence threshold, queries with upcoming
  content, and regenerates or continues with evidence.

## Controlled source
Prefer a versioned synthetic or relational fact store with:
- stable facts;
- changed facts;
- unavailable facts;
- contradictory records;
- plausible distractors;
- mid-generation-only information needs.

Web is secondary only.

## Baselines
1. Base LLM.
2. Anti-hallucination prompt.
3. Upfront RAG.
4. Explicit retrieval tool.
5. FLARE-like confidence trigger.
6. Semantic heuristic trigger.
7. Confidence OR semantic trigger.
8. Confidence AND semantic trigger where meaningful.
9. Retrospective verification.
10. Oracle trigger and query formulation.

## Measured confidence-risk design
Assign examples to quadrants using confidence from the actual evaluated
checkpoint, not intuition:

| Confidence | Epistemic risk | Expected behavior |
|---|---|---|
| High | Low | Continue without retrieval |
| Low | Low | Detect possible FLARE over-retrieval |
| High | High | Semantic trigger should catch confident hallucination |
| Low | High | Both controllers should retrieve |

The dataset must include confidently wrong or stale facts, low-confidence cases
that need no evidence, stable high-confidence facts, and low-confidence facts
that require external grounding.

## Allowed triggers
Regex, lexical cues, entity patterns, temporal patterns, deterministic sentence-prefix heuristics.

## Forbidden in core result
Learned classifier/router, hidden-state probe, multi-source selection.

## Prospective/retrospective ablation
Compare retrieval before a factual slot with verification after a candidate claim.

## Metrics
- final accuracy;
- unsupported claim rate;
- trigger P/R;
- retrieval P/R;
- false/missed retrieval;
- correction/abstention;
- evidence calibration;
- tokens;
- retrieval calls;
- latency/cost.
- confident-hallucination catch rate;
- low-confidence/no-deficit false-retrieval rate;
- retrieval rate by measured confidence/risk quadrant;
- uncertainty distributions around trigger thresholds;
- hallucination-reduction versus retrieval-cost Pareto curves.

## Failure taxonomy
Missed interrupt, false interrupt, malformed query, retrieval miss, stale/conflicting evidence, correct evidence ignored, model overrides evidence.

## Hard scope
No multiple retrievers. No symbolic engine. No backtracking. No PRA/native KV. Do not claim retrieval proves truth.

Do not claim active or generation-time retrieval as novel. The narrower claim is
supported only if semantic epistemic-risk detection catches evidence needs missed
by uncertainty at a competitive cost.

## Falsification
Semantic triggering adds no value if the FLARE-like controller catches the same
cases at equal or lower cost, semantic rules mostly duplicate measured
uncertainty, or high-confidence/high-risk cases are immaterial.

## Evidence gate
Paper 2.5 is justified only if heterogeneous sources solve a limitation exposed here.

## Deliverables
`paper1_5.tex`, trigger tests, controlled-source and quadrant-aware benchmark
generators, FLARE-like/upfront-RAG/explicit-tool baselines, machine-readable
confidence and retrieval traces, Pareto curves, result tables, and scaling plots.
