# AGENTS PATCH — Paper 2.5 Next Iteration

## Why this patch is needed
Paper 2.5's core discipline remains correct:
- multiple genuinely different retrieval substrates;
- transparent heuristic routing first;
- oracle source routing before learned routing.

Paper 1 now adds a production-relevant architectural lesson:
> stable interface policy may live in small adapter weights; context is best for cold-start; runtime owns source registry, credentials, validation, retrieval, provenance, evidence status, and enforcement.

Paper 2.5 should therefore test source heterogeneity first while keeping the architecture ready for later learned routing.

Do not turn Paper 2.5 into Paper 3.5.

## Revised central question
Do heterogeneous evidence substrates provide enough source-specific value that a cognitive runtime should treat DB, lexical, vector, KG, and web as distinct epistemic coprocessors instead of one generic retrieval tool?

Secondary:
Can capability count grow largely outside neural context?

## Required sources
1. relational/structured DB
2. lexical/document index
3. vector/semantic retrieval
4. web/search or controlled fresh-web analogue
Optional:
5. knowledge graph

They must differ semantically, not merely by backend.

## Structured DB
Use a real embedded relational engine such as DuckDB or equivalent.

Tasks:
- lookup
- SUM/COUNT/AVG
- argmax
- join

Treat DB primarily as an epistemic source, but log source-native computation.

Prefer typed relational IR compiled to SQL rather than unrestricted generated SQL.

## Source-specific request IR
Examples:

```db
select=max_sales
table=sales
year=2026
group_by=product
```

```lexical
document=policy_17
query="termination notice"
```

```vector
collection=reports
query="reasons for Q2 margin decline"
```

```web
entity=...
relation=...
time_window=...
```

Normalize all outputs into common evidence state with source, ID, query, value/content, freshness, relevance, support status, provenance, latency/cost.

## Primary routing remains heuristic
Examples:
- structured attribute/aggregate -> DB
- exact document/phrase -> lexical
- semantic report question -> vector
- current/recent/public -> web
- relation path -> KG

Why:
Paper 2.5 must first prove source heterogeneity matters before learning the router.

## Capability/context scaling
Add this as a primary systems experiment.

Compare:
- explicit source/tool schemas in context
- heuristic runtime router with source registry outside context
- optional ICL source examples

Use nested source catalogs:
- 1
- 2
- 3
- 4
- optional 5

Measure:
- prompt tokens vs source count
- source-selection accuracy
- query correctness
- fanout
- final accuracy
- unsupported claims
- latency/cost

## Factorized pipeline
Always report:
NEED RETRIEVAL -> SOURCE -> QUERY -> RETRIEVE -> EVIDENCE STATUS -> USE

Paper 1.5 informs NEED.
Paper 2.5 primarily studies SOURCE and source-specific QUERY semantics.

## Bridge from Paper 1.5
If Paper 1.5 later produces a learned retrieval-required policy, Paper 2.5 should be able to consume that event:

retrieval-required -> heuristic source router -> typed source request -> runtime adapter -> evidence state

Do not make the learned router primary here.

## Credentials/policy
Each source adapter should expose:
- local/remote
- credential scope
- latency class
- cost class
- privacy class
- freshness
- side-effect class

Paper 2.5 remains READ-only.
The model never receives raw credentials.

## Retrieval economics
Track:
- CPU time
- network time
- source calls
- bytes retrieved
- evidence tokens reinjected
- source-description/prompt tokens
- query-generation tokens
- total wall time

Separate prototype latency from recurring neural token burden and external-source cost.

## Universal retriever baseline
Make it strong.

Possible universal condition:
- textualize all source records into a unified semantic/vector index.

Compare with source-native execution.

For DB tasks, maintain a source-native oracle to quantify what is lost by textualization.

## Broadcast baseline
Mandatory where feasible.

Ask:
> Is routing worth it, or can we cheaply query everything?

Measure fanout, latency, source cost, conflicts, answer quality.

## Bounded composition
Allow only deterministic 2-stage compositions:
- date resolver -> web
- entity resolver -> DB
- lexical -> vector rerank
- DB result -> exact document lookup

Log dependency DAG.
No general planner.

## Evidence status
Carry:
- SUPPORTED
- CONTRADICTED
- UNVERIFIED
- STALE
- AMBIGUOUS
- CONFLICT

Do not let the LLM silently resolve conflicting sources without trace.

## Oracle matrix
Include:
1. oracle need + oracle source + oracle query
2. real need + oracle source
3. real need + heuristic source
4. heuristic need + heuristic source
5. explicit LLM source/tool selection
6. universal retriever
7. broadcast

This decomposes trigger, source and query headroom.

## Paper 3.5 gate
Learned source routing is justified only if:
1. source-native heterogeneity has oracle value;
2. heuristic routing leaves a measurable gap;
3. context/schema burden grows enough to matter;
4. broadcast is not already dominant.

If these fail, do not train a router.

## Future placement architecture
Document but do not claim as proven:
- weights: stable source-selection policy
- runtime: registry, credentials, validation, retrieval, provenance, evidence enforcement
- context: task/session source hints and newly introduced sources

This mirrors Paper 1 without prematurely importing its result to retrieval.

## Failure taxonomy
Need detection:
- missed retrieval
- false retrieval

Source routing:
- wrong source
- unnecessary fanout
- unavailable source

Query:
- malformed
- wrong entity/relation/time
- over/underspecified

Retrieval:
- miss
- stale evidence
- ranking failure
- conflict

Runtime:
- credential denial
- timeout
- budget violation

Integration:
- evidence ignored
- unsupported answer despite no evidence
- wrong source treated as authoritative

## Immediate order
1. Keep heuristic router primary.
2. Implement common source/evidence interface.
3. Add structured DB + lexical + vector.
4. Add controlled web/fresh source.
5. Build source-optimal benchmark.
6. Run universal vs source-native oracle.
7. Run heuristic vs oracle routing.
8. Run source-count/context scaling.
9. Run broadcast baseline.
10. Only if gate passes, hand learned routing to Paper 3.5.

## Deliverables
- patched `AGENTS_paper2_5.md`
- common retrieval registry
- DB/lexical/vector/web adapters
- typed source request/evidence IR
- source-native vs universal benchmark
- oracle routing matrix
- source-count/context-cost plots
- broadcast comparison
- credential/policy metadata
- explicit Paper 3.5 go/no-go decision
