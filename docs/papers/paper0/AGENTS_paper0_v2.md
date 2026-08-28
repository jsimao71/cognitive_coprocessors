# AGENTS — Paper 0 v3

## Mission
Define the position and research program for a heterogeneous cognitive Transformer with two interleaved coprocessor tracks.

## Core distinction
- **Competence deficit:** enough information exists; specialized computation would help.
- **Knowledge deficit:** reliable/fresh/source-specific evidence is missing.

Do not collapse both into generic tool use.

## Twin tracks
- Computational coprocessors: calculator, logic, constraints, graph, algebra, code/simulation.
- Epistemic coprocessors: DB, vector DB, lexical/document index, KG, web/search.

## Unifying mechanism
Generation-time **semantic interrupts** detect an opportunity for help, normalize it into typed micro-IR, invoke a specialized engine, and expose resulting micro-state to subsequent neural computation.

## Trigger ladder
1. strict syntax
2. lexical/syntactic
3. heuristic routing
4. learned text detector/parser
5. hidden-state probe/router
6. jointly trained interface

## Interleaved roadmap
- Paper 0: Position: computational + epistemic coprocessors
- Paper 1: Reflex calculator
- Paper 1.5: Reflex retrieval, single source
- Paper 2: Multiple compute/reasoning engines
- Paper 2.5: Multiple retrieval engines
- Paper 3: Learned computational semantic interrupts
- Paper 3.5: Learned epistemic interrupts and routing
- Paper 4: Transactional provenance/backtracking
- Paper 5: Structured/PRA/KV interfaces
- Paper 6: Co-adaptation / learning to offload and acquire
- Paper 7: Integrated heterogeneous cognitive runtime

## Required related-work framing
Cover tool/function calling, ReAct/planner-executor, PAL/program-aided reasoning, RAG/Self-RAG, neuro-symbolic systems, theorem-prover interaction, dynamic solver selection, and context/memory virtualization.

FLARE and Self-RAG are mandatory retrieval comparisons. FLARE predicts upcoming
content and uses token uncertainty to decide whether to retrieve and regenerate;
Self-RAG trains the main LM to retrieve and emit relevance, support, and utility
reflection tokens. Never claim generation-time retrieval or learned adaptive
retrieval alone as novel. Also cover corrective RAG, confidence-triggered and
token-level retrieval, retrieval routers, and multi-source RAG.

## Required FLARE / Self-RAG comparison
Compare conventional RAG, FLARE, Self-RAG, and this program on retrieval timing,
trigger, forward-looking behavior, learned control, source heterogeneity, source
routing, evidence/reflection state, computational engines, transactions, and
PRA/KV micro-state. Do not assume external semantic control is better than
main-model learned control; Paper 3.5 must test that architectural choice.

## Critical retrieval distinction
Neural uncertainty is not equivalent to epistemic risk. Evaluate measured cases
in all four confidence/risk regimes: high/high, high/low, low/high, and low/low.
The high-confidence/high-risk regime is the critical test for semantic triggers;
the low-confidence/low-risk regime measures confidence-driven over-retrieval.

The broadest differentiation is compute/retrieval symmetry:
- competence deficit -> calculator, logic, SMT, graph, code;
- epistemic deficit -> DB, lexical, vector, KG, web.

## Program metrics
Reliability, unsupported-claim rate, trigger precision/recall, engine/retrieval correctness, tokens, latency, external calls, state growth, scaling with problem size, and model-size sensitivity.

## Guardrails
- Transformers remain central.
- Explicit tool calling remains supported.
- Upfront RAG remains supported.
- Do not require PRA in Papers 1–3.5.
- Do not equate retrieved evidence with truth.
- Do not equate formal derivability with truth of premises.
- Later mechanisms require evidence gates.
- Paper 1.5 must include a FLARE-like confidence baseline.
- Paper 2.5 must use genuinely heterogeneous evidence substrates and an oracle source-routing gate.
- Paper 3.5 must compare Self-RAG-style main-model control with text and hidden-state sidecars.
- Paper 4 must not claim relevance/support labels as novel; its contribution is persistent provenance, dependencies, transactions, and retraction.

## Deliverables
- `paper0_v3.tex`
- related-work matrix
- novelty comparison table
- conventional RAG / FLARE / Self-RAG / program comparison table
- architecture figure
- final per-paper hypothesis/falsification roadmap
