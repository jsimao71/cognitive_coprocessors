# AGENTS — Paper 2.5 Next Iteration: Production Data Coprocessor Stack

## Mission
Evolve Paper 2.5 from controlled source-native adapters into a production-shaped data-coprocessor evaluation. The present result already supports source-native data access, heuristic routing, lower fanout than broadcast, and capability descriptors outside recurring context. The next step is real data infrastructure, not learned routing.

## Primary question
> Does the heterogeneous DataCoprocessor abstraction retain its reliability and efficiency advantages when backed by real open-source databases, indexes, Iceberg tables, semantic layers, and ontologies?

## Common DataCoprocessor interface
Each backend exposes source ID/type, capabilities, typed request IR, validation, credential/policy metadata, execute/read, result/evidence IR, freshness/snapshot, provenance, and cost/latency. Backend implementation must remain swappable.

## Phase 1 — local in-process baselines
### DuckDB
Use DuckDB for lookup/filter/SUM/AVG/COUNT/GROUP BY/argmax/joins/window-time operations. Prefer typed relational IR compiled to parameterized SQL. No unrestricted generated SQL in the primary condition.

### FAISS
Use FAISS first for semantic retrieval to avoid service/deployment confounds. Start with exact/flat index; add ANN only if dataset size warrants it. Separate embedding cost from retrieval cost.

### Lexical
Start with SQLite FTS5. Optionally add Tantivy/PyTantivy later if fielded ranking/index control is needed.

## Phase 2 — WSL Docker sidekick for Postgres
Windows-host Docker is not an accepted execution target for this project. Create `sidekick/data_stack/` intended for WSL2 execution.

Include:
- docker-compose.yml
- .env.example
- init SQL
- healthcheck script
- README with WSL commands
- pytest integration marker/config

### Postgres
Test relational adapter, typed parameterized queries, joins/aggregates, schema introspection, read-only policy, provenance.

### pgvector
Enable pgvector in the same stack where practical. Compare with FAISS for retrieval quality, metadata filtering, latency, persistence, and operational cost.

Integration tests must skip cleanly when services are unavailable and never silently fall back to synthetic adapters.

## Phase 3 — Apache Iceberg
Treat Iceberg as the open table/storage abstraction, not a query engine.

### I1 DuckDB direct Iceberg read
Read a local Iceberg table directly from metadata/storage. Validate adapter semantics without catalog complexity.

### I2 Iceberg REST catalog
Add a REST-compatible catalog. Apache Polaris is a preferred candidate if local setup is acceptable; another REST-compatible catalog is allowed if simpler. Keep catalog behind an adapter.

### I3 snapshot/time-travel/schema evolution
Test snapshot selection, time travel, schema evolution, snapshot provenance, and write support only where the selected engine/catalog path supports it reliably.

Use local storage first; add MinIO/S3-compatible storage only if the catalog path requires it or after the local path works.

## Phase 4 — dedicated vector DBs
Only after FAISS and pgvector are stable.

Recommended order:
1. Qdrant — simple local Docker deployment and strong metadata filtering.
2. Weaviate — useful second architecture and hybrid-search comparison.
3. Milvus — optional later, only if scale/distributed behavior justifies added ops complexity.
4. Chroma — optional lightweight developer comparison.

The minimal useful set is FAISS + pgvector + Qdrant, optionally Weaviate. Do not turn Paper 2.5 into a general vector-DB benchmark.

## WSL Docker requirements
README must document WSL distro requirements, `docker compose up -d`, health checks, test command, teardown, volume cleanup, and ports. Keep compose modular so Postgres/pgvector can run without Iceberg/vector services.

## Phase 5 — governed semantic layer
Add an optional business semantic/metric layer above relational/Iceberg data. Candidate adapters include Cube Core and MetricFlow/dbt-semantic-style interfaces where practical.

Semantic layer defines metrics, dimensions, joins, filters, and governed business definitions.

Compare:
1. raw model-authored SQL/tool query;
2. typed relational IR;
3. semantic metric request compiled by semantic layer.

Measure business-definition correctness, not only SQL validity.

## Phase 6 — ontology / knowledge semantics
Keep ontology distinct from metric semantics. Add one standards-compatible RDF/SPARQL path, initially Apache Jena or lightweight Oxigraph/pyoxigraph.

Use cases:
- entity/type/relation resolution;
- taxonomy/ISA;
- schema/domain mappings;
- concept normalization;
- ontology-mediated grouping;
- validation of semantic relations.

## Phase 7 — Iceberg + semantic + ontology composition
Build a redistributable enterprise-style dataset: customers, products, sales, costs, inventory, organizations, plus documents/policies, metric definitions, and ontology/taxonomy.

Question classes:
- direct lookup;
- aggregate;
- join;
- governed metric;
- ontology-mediated grouping;
- snapshot/time-travel;
- document evidence;
- mixed data + document evidence.

Example path: natural language -> semantic request -> ontology resolves product family -> governed metric resolves margin -> query engine executes against Iceberg snapshot -> typed result/provenance -> answer.

## Conditions
Retain source-native oracle, transparent heuristic routing, explicit source/tool schemas, universal textual retriever, broadcast, source-native typed IR, semantic-layer request, and ontology-assisted request where applicable. Do not add learned source routing unless a new heuristic-to-oracle gap appears.

## Stronger universal baseline
Do not retain only a weak top-1 lexical universal retriever. Build a stronger textualized baseline over relational/Iceberg rows and semantic descriptions, and test whether it can preserve aggregates, snapshot status, provenance, and conflict semantics. If it closes the gap, report that.

## Provenance
Every result should retain backend, table/index/collection, normalized query, row/document/vector IDs, timestamps, Iceberg snapshot ID, semantic metric definition/version, ontology concept/relation IDs, evidence status, and latency/cost.

## Credentials/policy
Secrets remain runtime-owned and never enter model context. Record scope identifiers, not secret values. Paper stays read-only unless a later explicit write-policy experiment is approved.

## Metrics
Quality: final answer, source/query correctness, retrieval recall, aggregate/join correctness, metric correctness, ontology mapping, conflict handling, UCR.

Cost: source descriptor/prompt tokens, CPU query time, DB latency, vector latency, bytes scanned/retrieved, network calls, total wall time.

Operations: startup complexity, memory, service count, index build time, failure recovery. Keep operational complexity descriptive/checklist-based rather than inventing a pseudo-scientific scalar.

## Test strategy
### Unit — no Docker
Typed IR, routing, provenance, DuckDB, FAISS, SQLite FTS5.

### Integration — WSL Docker
Postgres, pgvector, Iceberg REST catalog, Qdrant, optional Weaviate.

### End-to-end
Mixed benchmark across local, Postgres, Iceberg, vector, semantic layer, ontology.

## Immediate order
1. Replace synthetic relational adapter with DuckDB-backed adapter while preserving the current benchmark.
2. Replace synthetic vector adapter with FAISS.
3. Add SQLite FTS5 lexical adapter.
4. Create WSL `sidekick/data_stack` Docker Compose for Postgres + pgvector.
5. Add Postgres adapter/tests.
6. Add direct Iceberg read via DuckDB.
7. Add REST-catalog Iceberg path.
8. Add Qdrant Docker integration.
9. Optionally add Weaviate.
10. Add semantic-layer adapter.
11. Add ontology/RDF adapter.
12. Build Iceberg+semantic+ontology mixed benchmark.
13. Re-run universal vs source-native vs heuristic/broadcast.
14. Re-evaluate Paper 3.5 gate only if heuristic routing develops measurable headroom.

## Deliverables
Updated Paper 2.5 manuscript/AGENTS, DataCoprocessor interface, DuckDB/FAISS/FTS5 adapters, WSL Docker sidekick, Postgres+pgvector tests, Iceberg direct/REST-catalog adapter, Qdrant adapter, optional Weaviate, semantic-layer adapter, ontology adapter, enterprise benchmark, provenance schema, backend-substitution tests, and updated Paper 3.5 go/no-go.
