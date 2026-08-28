# Paper 1.5 AGENTS Patch — FLARE-Aware

## Revised focus
Reframe from “generation-time retrieval” to:
**Semantic Epistemic Risk vs Confidence-Based Active Retrieval During Generation.**

## Mandatory FLARE-like baseline
Implement an appropriate FLARE-like condition:
- inspect/predict upcoming generation;
- use token/model uncertainty;
- trigger retrieval below confidence threshold;
- retrieve using predicted/upcoming content;
- regenerate/continue with evidence.
Document simplifications.

## Core hypothesis
Model uncertainty and need-for-evidence are not equivalent.

Construct/measured-test the 2x2:
| Confidence | Epistemic risk | Key expectation |
|---|---|---|
| High | Low | continue |
| Low | Low | possible FLARE over-retrieval |
| High | High | semantic trigger should catch confident hallucination |
| Low | High | both should retrieve |

Do not assign quadrants by intuition: measure confidence on the actual checkpoint.

## Required baselines
1. LLM only
2. anti-hallucination prompt
3. upfront RAG
4. explicit retrieval tool
5. FLARE-like confidence trigger
6. semantic heuristic trigger
7. confidence OR semantic
8. confidence AND semantic where meaningful
9. retrospective verification
10. oracle trigger/query

## Dataset requirements
Include confidently wrong/stale facts, low-confidence cases needing no evidence, stable high-confidence facts, and low-confidence externally grounded facts.

## Extra metrics
Confident-hallucination catch rate; low-confidence/no-deficit false retrieval; retrieval rate by confidence/risk quadrant; uncertainty around triggers; hallucination-reduction vs retrieval-cost Pareto curve.

## Falsification
Semantic triggering fails to add value if FLARE-like confidence catches the same cases at equal/lower cost, semantic rules mostly duplicate uncertainty, or high-confidence/high-risk cases are immaterial.

## Claim discipline
Do not claim active retrieval. A possible claim, only if supported: epistemic-risk detection catches evidence needs missed by uncertainty, especially confident stale/source-dependent commitments.
