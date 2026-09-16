# AGENTS PATCH --- Paper 2.5: Public Data/Retrieval Coprocessor Benchmarks

## Purpose

Move Paper 2.5 from production-shaped local fixtures to recognized
public data/retrieval benchmarks while keeping the source-native
execution thesis.

## Priority benchmarks

### TAT-QA

Highest-priority bridge benchmark. Use for: - table/text retrieval; -
arithmetic over retrieved values; - RETRIEVE -\> COMPUTE composition; -
financial interpretation.

Compare: - LLM only; - generic tools; - source-native data
coprocessor; - universal textual retrieval; - oracle evidence/operation.

### Spider 2.0

Use the most locally reproducible subset first, especially
SQLite-compatible tasks where feasible.

Test: - model-generated SQL tool; - generic `__retrieve`/`__help` routed
to data runtime; - typed DataCoprocessor request; -
deterministic/source-native execution.

Measure execution correctness, query safety, malformed SQL/IR,
context/tool-schema cost.

### CRAG

Use as dynamic retrieval benchmark shared with Paper 1.5, but Paper 2.5
focuses on source/back-end execution and heterogeneous retrieval rather
than epistemic trigger theory.

### FRAMES

Use a bounded subset for multi-source/multi-hop composition: -
retrieval; - numerical reasoning; - temporal; - tabular; -
post-processing.

This is the strongest candidate for RETRIEVE -\> COMPUTE -\> VERIFY
chains.

## Later benchmark

BIRD-Interact after Postgres live integration is stable. This is better
suited to later runtime/product work but may provide a Paper 2.5 pilot.

## Existing enterprise fixture

Keep the current 11-case Iceberg/semantic/ontology benchmark because it
tests features not cleanly covered by public datasets: - Iceberg
snapshots/time travel; - governed metric definitions; -
ontology-\>metric composition; - typed provenance.

Use public benchmarks for external validity, not as a replacement.

## Conditions

1.  LLM only.
2.  Four generic tools.
3.  Generic RETRIEVE/VERIFY/HELP CogCop.
4.  CPU/heuristic source-native runtime.
5.  Universal text retriever.
6.  Source-specific explicit tools where useful.
7.  Oracle source/query.
8.  Latent Paper 3/3.5 later.

## TAT-QA composition metrics

Separate: - evidence selection; - table/text extraction; - numerical
operation selection; - compute exactness; - final answer; - whether
generic tool/CogCop needs one or multiple assistance episodes.

## Spider metrics

-   execution accuracy;
-   SQL/IR validity;
-   semantic correctness;
-   unsafe/unbounded query rate;
-   generated query tokens;
-   retries;
-   backend latency.

## CRAG/FRAMES metrics

-   retrieval recall;
-   UCR;
-   evidence provenance;
-   source calls/fanout;
-   final correctness;
-   Automatic Rescue Rate for automatic vs voluntary generic retrieval;
-   composition depth.

## Source-native vs textualization claim

For every applicable public benchmark, create a matched universal-text
condition only when fair.

Do not intentionally cripple it. Use a competent BM25/vector/hybrid
baseline.

## Deliverables

-   TAT-QA integration;
-   Spider 2.0 local subset integration;
-   CRAG/FRAMES bounded integrations;
-   shared generic-tool baseline;
-   source-native vs universal comparison;
-   public benchmark section in Paper 2.5;
-   decision on when BIRD-Interact moves to later papers.
