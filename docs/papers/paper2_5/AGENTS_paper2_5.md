# AGENTS — Paper 2.5

## Mission
Test heterogeneous active retrieval across structurally different epistemic
substrates while keeping the proposed trigger and routing policies transparent
and heuristic. Adaptive generation-time retrieval itself is prior art.

## Minimum source suite
- relational/structured DB;
- lexical/document index;
- vector DB / semantic retrieval;
- web/search.
Optional: knowledge graph.

These must be genuine source types, not wrappers around the same text retriever.
Source-specific semantics must matter: DB aggregation for structured argmax,
lexical retrieval for exact clauses, vector retrieval for semantic report
questions, web search for current public facts, and KG traversal for graph paths.

## Common evidence IR
Every source must normalize to:
source type, source ID, query, value/content, time/freshness, relevance score if applicable, support/contradict/ambiguous/unverified status, provenance.

## Heuristic routing only
Examples:
- structured attribute/aggregate -> DB
- exact phrase/document -> lexical
- semantic doc request -> vector
- current/latest/recent -> web
- relation neighborhood -> KG

## Small helper operations allowed
Relative-date resolution, entity normalization, deterministic query rewrite, filter/aggregation construction. Do not let this become a hidden planner.

Bounded deterministic compositions such as date resolver -> web and entity
resolver -> DB are allowed. Log their dependency DAG and component cost.

## Baselines
1. Base LLM
2. Upfront RAG
3. Single universal retriever
4. FLARE-like confidence + universal retriever
5. FLARE-like confidence + heuristic heterogeneous router
6. Self-RAG-inspired adaptive retrieval, where feasible and clearly labeled
7. Explicit LLM source/tool selection
8. Broadcast to all sources under matched cost
9. Heuristic semantic reflex routing
10. Oracle source router
11. Oracle trigger + source

## Tasks
Structured business/data questions; exact and semantic document retrieval; current facts; conflicts; no-retrieval; mid-generation source needs.

## Metrics
Answer accuracy, unsupported claims, trigger P/R, routing accuracy, query correctness, retrieval P/R, fanout, conflicts, source cost, tokens, tool-selection tokens, wall clock, scaling with source count and factual subgoals.

Report the factorized pipeline separately:
TRIGGER -> SOURCE -> QUERY -> RETRIEVE/EXECUTE -> EVIDENCE USE. Do not hide a
routing failure inside final-answer accuracy.

## Failure taxonomy
Wrong trigger, wrong source, over-broadcast, malformed query, unavailable/stale source, evidence conflict, provenance error, correct evidence ignored.

## Hard scope
No learned router/trigger, hidden-state probe, transactional state, backtracking, PRA/native KV.

## Evidence gate
Oracle source routing is mandatory. Paper 3.5 is justified only if oracle
heterogeneous routing beats a universal retriever under matched cost and brittle
source selection or excessive fanout remains a measured limitation.

## Falsification
Evidence for heterogeneous routing is weak if universal retrieval matches it,
oracle routing adds little, matched-cost broadcast dominates, source-specific
query semantics do not matter, or explicit model source selection is equally
efficient.

## Deliverables
`paper2_5.tex`, common retrieval interface, genuinely distinct source adapters,
heuristic router, mixed-source benchmark, FLARE-like and Self-RAG-inspired
controls, factorized traces, oracle-routing gate, and result tables/plots.
