# Paper 0 AGENTS Patch — FLARE / Self-RAG

## Required comparison

| Dimension | Conventional RAG | FLARE | Self-RAG | Cognitive-coprocessor program |
|---|---|---|---|---|
| Retrieval timing | Mostly pre-generation | During generation | Adaptive during generation | During generation + upfront retrieval supported |
| Trigger | Request/harness | Low-confidence predicted tokens | Learned retrieval/reflection behavior | Semantic/epistemic interrupt; later learned |
| Forward-looking | Usually no | Yes | Adaptive segment-level | Yes; prefix/text/semantic state, later latent |
| Learned trigger | Usually no | No | Yes | Paper 3.5 |
| Multiple retrieval engines | External concern | Not central | Not central | Paper 2.5/3.5 central |
| Learned source routing | Not central | No | Not central | Paper 3.5 |
| DB/vector/web distinction | Orchestration-specific | No | No | First-class |
| Evidence/reflection state | Passages | Evidence | Relevance/support/reflection | Typed persistent epistemic micro-state |
| Calculator/logic/SMT | Separate tools | No | No | Parallel compute track |
| Automatic symbolic inference | No | No | No | Paper 2+ |
| Transaction/retraction | No | No | Limited critique | Paper 4 |
| PRA/KV micro-state | No | No | No | Paper 5 |
| Target architecture | Retrieval augmentation | Active retrieval | Self-reflective adaptive RAG | Heterogeneous cognitive runtime |

## Novelty discipline
Never claim generation-time retrieval or learned adaptive retrieval alone as novel. FLARE and Self-RAG establish important prior art.

Frame the program as generalizing from “when should the LM retrieve?” to “what specialized computational or epistemic substrate can assist the evolving neural computation now?”

## Critical distinction
Neural uncertainty != epistemic risk.

Evaluate four regimes:
- high confidence / low epistemic risk;
- low confidence / low epistemic risk;
- high confidence / high epistemic risk — critical semantic-trigger regime;
- low confidence / high epistemic risk.

## Self-RAG architectural distinction
Compare main-LM learned retrieval/reflection control against model-independent sidecar semantic control. Do not assume the sidecar is better.

## Strongest broader differentiation
Compute/retrieval symmetry:
- competence deficit -> calculator / logic / SMT / graph / code;
- epistemic deficit -> DB / lexical / vector / KG / web.

## Series consequences
Paper 1.5: mandatory FLARE-like confidence baseline.
Paper 2.5: emphasize heterogeneous evidence substrates/source routing.
Paper 3.5: mandatory Self-RAG-style main-model-control comparison.
Paper 4: do not claim relevance/support labels as novel; focus on persistent provenance, dependencies, transactions and retraction.

## Literature work
Verify canonical FLARE and Self-RAG mechanisms from primary papers. Expand search to active/adaptive retrieval, corrective RAG, confidence-triggered retrieval, token-level retrieval, retrieval routers, and multi-source RAG.
