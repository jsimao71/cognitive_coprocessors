# Paper 2.5 AGENTS Patch — Heterogeneous Active Retrieval

## Revised differentiation
Do not frame this as merely adaptive generation-time retrieval. FLARE/Self-RAG already cover major parts of that space.

Test:
**Heterogeneous active retrieval across structurally different epistemic substrates.**

## Genuine source types
- relational DB / structured query
- lexical/document index
- vector/semantic retrieval
- web/search
- optional knowledge graph

Avoid multiple wrappers over essentially the same text retriever.

## Source semantics must matter
Examples:
- argmax sales -> DB aggregation
- exact policy clause -> lexical
- semantic report question -> vector
- current public fact -> web
- graph neighborhood/path -> KG

## New mandatory baselines
- FLARE-like confidence + universal retriever
- FLARE-like confidence + heuristic heterogeneous router
- Self-RAG-inspired adaptive retrieval condition where feasible
- explicit LLM source/tool selection
- broadcast-to-all under matched cost
- oracle source router
- oracle trigger + source

## Factorized metrics
TRIGGER -> SOURCE -> QUERY -> RETRIEVE/EXECUTE -> EVIDENCE USE.
Do not hide routing failure inside final accuracy.

## Oracle gate
Oracle source routing is mandatory. If oracle heterogeneous routing does not beat a universal retriever under matched cost, Paper 3.5 learned routing is poorly motivated.

## Composition
Bounded deterministic compositions such as date resolver -> web or entity resolver -> DB are allowed. Log dependency DAG and cost. No general learned planner.

## Self-RAG positioning
Treat adaptive retrieval/critique as prior art. The paper's differentiator is heterogeneous source semantics and routing.

## Falsification
Weak evidence if universal retrieval matches, oracle routing adds little, broadcast-to-all dominates, source-specific query semantics do not matter, or explicit model source selection is equally efficient.
