# AGENTS PATCH — Paper 7: Full Cognitive Runtime + Data/Semantic Coprocessors

## Purpose
Patch the current Paper 7 integration plan so the final system reflects the full architecture developed across Papers 1–6 and the intended production direction: a small/medium neural semantic core, learned cognitive-interface policy, cheap local computational coprocessors, epistemic/data coprocessors, persistent typed cognitive state, PRA/selective materialization, and conventional tools as universal fallback.

Paper 7 is an integration/evaluation paper, not a license to add unvalidated mechanisms. Maintain a broad candidate registry, but only promote mechanisms into headline experiments when earlier papers establish value or when they are required as explicit baselines/fallbacks.

## Revised title candidate
**A Heterogeneous Cognitive Runtime for Language Models: Virtualized Knowledge, Data, and Computation**

Alternative:
**Cognitive Coprocessors at System Scale: A Heterogeneous Runtime for Neural, Symbolic, and Data Intelligence**

## Revised mission
Integrate the mechanisms validated by Papers 1–6 into a production-shaped heterogeneous cognitive runtime and test the program-level hypothesis:

> Can a substantially smaller neural model, augmented by tightly coupled computational and epistemic/data coprocessors, approach or exceed larger monolithic models on mixed workloads while reducing accelerator work, recurring context burden, and unsupported factual commitments?

The runtime must preserve ordinary tool/function calling as the universal fallback.

## Architecture principle
Use the Paper 1 placement result as the default starting architecture unless later papers overturn it:

### Weights
Stable semantic selection and serialization policy:
- computational-family discrimination;
- epistemic/retrieval-required policy;
- common execution/request protocol;
- optional learned routing where Papers 3/3.5 support it.

### Runtime
Deterministic mechanism and governance:
- typed parsing/IR;
- capability/source registry;
- bounds and budgets;
- exact execution;
- credentials;
- provenance;
- evidence status;
- state/dependency management;
- transaction/retraction if Paper 4 validates it;
- safety enforcement;
- scheduling;
- observability.

### Context
Dynamic/session-specific information:
- task instructions;
- newly introduced/rare capabilities;
- source hints;
- user policy;
- cold-start interfaces;
- information that should not be baked into weights.

### PRA/materialization layer
If Paper 5 validates it:
- selectively materialize only relevant cognitive records, evidence, tool state, and references into neural computation;
- do not dump the full state/capability catalog into the prompt.

## Full candidate coprocessor registry

### Tier 0 — reflex CPU cognition
Candidates:
- calculator / exact integer-rational arithmetic;
- date/time arithmetic;
- units/dimensional conversion;
- simple statistics;
- string/regex/parser primitives;
- deterministic normalization/validation.

These should be extremely cheap, bounded, and usually local.

### Tier 1 — local structured computational cognition
Candidates:
- symbolic algebra (e.g. SymPy-class engine);
- Horn/Datalog;
- ISA/frame/graph reasoning;
- SMT/constraint solver (Z3/cvc5 class);
- graph algorithms;
- deterministic query planning/relational aggregation.

### Tier 2 — local epistemic/data cognition
Candidates:
- relational SQL engine;
- lexical/full-text index;
- vector/semantic index;
- knowledge graph / RDF-SPARQL store;
- governed metric/semantic layer;
- Apache Iceberg lakehouse/table access;
- local document/index stores.

### Tier 3 — general local compute fallback
Candidates:
- sandboxed Python;
- optionally WASM or another stronger isolation boundary;
- domain simulators.

Do not use general code when a safer bounded engine suffices.

### Tier 4 — remote epistemic/data sources
Candidates:
- web/search;
- remote enterprise SQL/lakehouse;
- remote Iceberg catalogs/query engines;
- APIs;
- source-specific knowledge services.

These are READ-only by default in Paper 7 evaluation unless a separate action policy is explicitly tested.

### Tier 5 — side-effectful conventional tools
Examples:
- send email;
- mutate files;
- deploy;
- purchases;
- production writes;
- external transactions.

Keep explicit tool invocation, authorization, and confirmation policies. Do not automatically convert all world-changing actions into reflexes.

## Data coprocessors are first-class
Paper 7 should explicitly distinguish **data coprocessors** from generic retrieval.

A data coprocessor can perform source-native operations such as:
- relational joins;
- aggregation;
- filtering;
- time travel;
- schema-aware lookup;
- governed metric calculation;
- ontology/knowledge-graph traversal;
- source provenance;
- policy-aware access.

This is stronger than converting every data source to text and doing vector RAG.

## Apache Iceberg integration
Add Apache Iceberg as a strategic production data substrate, but do not treat Iceberg itself as a semantic layer.

Iceberg role:
- open table format / lakehouse substrate;
- schema evolution;
- snapshot/time-travel semantics;
- partition evolution;
- multi-engine interoperability;
- catalog abstraction;
- large analytic datasets;
- provenance through snapshot/table metadata.

The runtime should integrate through an adapter boundary, not couple directly to one compute engine.

### Preferred Iceberg architecture
```text
Cognitive runtime
      |
DataCoprocessor interface
      |
Semantic / ontology layer (optional)
      |
Query planner / engine
      |
Iceberg catalog + tables
      |
object storage
```

### Catalog support
Design around Iceberg catalog abstractions and preferably REST-catalog compatibility.

Do not assume a single catalog implementation.

Potential catalog backends may include REST, JDBC, Hive, Nessie, or other Iceberg-compatible catalogs.

### Query engines
Treat Spark/Trino/DuckDB/other Iceberg-capable engines as replaceable execution adapters where practical.

Paper 7 should not become a benchmark of every Iceberg engine.

## Semantic-layer family
Support semantic/metric layers as optional data coprocessors above relational/Iceberg data.

Initial open-source candidates to evaluate:
- Cube Core;
- MetricFlow/dbt semantic interfaces where licensing/version is suitable;
- other open semantic-layer implementations that expose governed metrics/dimensions/joins.

Semantic-layer role:
- certified metrics;
- dimensions;
- joins;
- business definitions;
- access rules;
- compile high-level semantic requests into source-native queries.

Example:
```text
user/model concept:
"revenue growth by product in Q2"

semantic request:
metric = revenue_growth
dimension = product
period = Q2

semantic layer
   -> governed relational query
   -> Iceberg tables
```

The LLM should not repeatedly reinvent metric definitions from raw schemas.

## Ontology / knowledge-semantic family
Metric semantic layers and ontologies are different and should remain distinct.

Add an ontology/knowledge-graph adapter family for:
- entity classes;
- relations;
- taxonomies;
- mappings across schemas;
- domain concepts;
- constraints;
- inference.

Open-source candidates:
- Apache Jena for RDF, SPARQL, OWL/RDFS and rule/inference support;
- Oxigraph/pyoxigraph for a lightweight RDF/SPARQL path;
- other standards-compatible RDF/SPARQL/OWL/SHACL systems where justified.

Do not hard-code Paper 7 to one ontology engine.

## Semantic federation over Iceberg
A strategic integration experiment should test whether the runtime can answer a question using:

1. ontology/domain semantics to identify concepts/relations;
2. governed metric layer to resolve business meaning;
3. Iceberg-backed data for authoritative records;
4. source-native relational computation for joins/aggregates;
5. typed evidence/provenance returned to cognitive state;
6. PRA/selective materialization to expose only the necessary evidence to the neural model.

Example conceptual path:
```text
"Which product family had the greatest YoY margin improvement,
according to the current finance definition?"

natural language
  -> semantic/epistemic event
  -> ontology resolves product-family relation
  -> semantic layer resolves governed margin metric
  -> query engine executes over Iceberg snapshot
  -> typed result + snapshot + metric provenance
  -> cognitive state
  -> selective materialization
  -> answer
```

This is a stronger production target than generic text-to-SQL.

## Semantic/ontology adapter contract
Define a generic runtime contract such as:

- `resolve_concept(term, context)`
- `resolve_metric(metric_name, dimensions, filters)`
- `resolve_relation(subject_type, relation, object_type)`
- `plan_query(semantic_request)`
- `execute_query(plan, snapshot/policy)`
- `explain_provenance(result)`
- `validate_result(result, constraints)`

Concrete backends may implement only the subset they support.

## Do not collapse all semantics into one layer
Keep separate:
1. LLM semantic understanding;
2. business semantic/metric definitions;
3. ontology/knowledge graph;
4. physical data schema/catalog;
5. source-native query engine;
6. cognitive runtime state/provenance.

Paper 7 should test composition across these layers.

## Product-oriented plugin boundaries
Although Paper 7 is a research paper, implement the runtime with product-shaped extension points:

### ComputationalCoprocessor
- capability ID
- accepted typed IR
- result IR
- cost/latency class
- purity/side-effect class
- resource bounds
- execute

### DataCoprocessor
- source ID
- source type
- schema/catalog metadata
- credential/policy requirements
- query capabilities
- freshness/snapshot semantics
- execute/read
- provenance

### SemanticLayerAdapter
- concepts/metrics/dimensions
- query planning/compilation
- policy metadata
- provenance

### OntologyAdapter
- entity/type/relation resolution
- SPARQL/graph query where supported
- inference/constraint capabilities
- provenance

### RetrievalCoprocessor
- lexical/vector/web/document retrieval
- ranking/relevance metadata
- provenance

### FallbackTool
- conventional descriptor/schema
- explicit invocation
- side-effect policy

## Capability virtualization
Paper 7 should explicitly test the thesis that capability descriptions need not all live in neural context.

Compare catalogs of:
- 1
- 2
- 4
- 8
- 16
- optionally 32+ capabilities

Conditions:
1. explicit tool schemas/descriptions in context;
2. ICL capability demonstrations;
3. learned interface adapter + minimal contract;
4. runtime registry + learned/heuristic interrupts;
5. oracle routing.

Measure:
- recurring prompt tokens;
- selection accuracy;
- false activation;
- payload/query correctness;
- latency;
- model calls;
- final quality.

## Context virtualization + capability virtualization
If Paper 5/PRA evidence supports it, make this a central integrated concept:

### Context virtualization
Large information/state space -> selectively materialized neural working set.

### Capability virtualization
Large capability registry -> selectively activated computational/data working set.

Test both independently and jointly.

Do not claim the analogy as an empirical result by itself.

## Small-model augmentation curve
This should be a headline Paper 7 experiment.

Compare several neural scales where hardware permits:
- small baseline;
- small + cognitive runtime;
- medium baseline;
- medium + runtime;
- larger baseline;
- larger + runtime.

Progressively add validated coprocessors:
- C0: none
- C1: reflex primitives
- C2: symbolic/structured compute
- C3: local data/retrieval
- C4: semantic/ontology layer
- C5: remote epistemic sources
- C6: full validated runtime

Plot:
- task quality vs neural parameters;
- task quality vs accelerator seconds;
- maximum solvable task size vs neural parameters;
- unsupported factual commitment vs neural parameters;
- total cost vs quality.

Primary question:
> How much neural scaling can heterogeneous cognitive augmentation substitute for on offloadable workloads?

## Workload families
Use mixed workloads so the runtime cannot win by specializing to one benchmark.

Include representative:
- hard arithmetic;
- date/units;
- formal logic/constraints;
- graph/ontology;
- structured analytics;
- document retrieval;
- current/source-specific facts;
- semantic analytics over governed data;
- bounded multi-stage tasks combining compute + retrieval/data.

Include non-trigger controls.

## Data/semantic benchmark
Add a production-shaped benchmark over synthetic or redistributable enterprise-style data:

Example tables:
- customers;
- products;
- sales;
- costs;
- inventory;
- organizations;
- policies/documents.

Store analytic tables in Iceberg.

Layer:
- a metric semantic model;
- an ontology/taxonomy;
- lexical/vector document indexes.

Questions should require different paths:
- raw lookup;
- aggregate;
- governed metric;
- ontology-mediated grouping;
- document evidence;
- mixed data + document evidence;
- time-travel/snapshot-specific answer.

The benchmark must be reproducible and must not require proprietary enterprise data.

## Iceberg-specific evaluation
Measure:
- correct snapshot selection;
- schema evolution robustness;
- source/table provenance;
- query correctness;
- semantic-layer correctness;
- ontology mapping correctness;
- result accuracy;
- bytes scanned where available;
- query latency;
- evidence/materialization tokens.

Do not claim Iceberg improves reasoning by itself; it provides a reliable open data substrate.

## CPU/GPU/XPU economics
Track:
- accelerator prompt/decode time;
- accelerator tokens;
- CPU coprocessor time;
- data-engine time;
- network time;
- bytes scanned/retrieved;
- state bytes;
- semantic-layer planning time;
- total wall time.

Develop an accelerator-displacement analysis:
how much neural work is avoided by cheap local/source-native computation?

Avoid simplistic conversion of CPU milliseconds into GPU FLOPs; report raw resource dimensions and only derive cost ratios under explicit assumptions.

## Reliability metrics
In addition to final accuracy:
- trigger P/R;
- engine/source routing accuracy;
- false intervention;
- typed IR correctness;
- engine correctness;
- source/query correctness;
- evidence support status;
- unsupported commitment rate;
- result-use/override;
- provenance completeness;
- stale-state rate;
- transaction/retraction correctness if applicable;
- PRA/materialization recall if applicable.

## Production safety/policy
Classify capabilities:
- PURE deterministic compute: automatic allowed;
- READ local/remote: automatic if policy permits;
- EXPENSIVE READ: budget gate;
- WRITE: explicit intent/policy;
- HIGH-RISK WRITE: explicit confirmation.

Credentials remain runtime-owned and must never be materialized into the model context.

## Fallback hierarchy
1. deterministic cheap reflex;
2. learned typed execution/data protocol;
3. learned semantic interrupt/router;
4. explicit conventional tool;
5. user confirmation for high-risk actions.

The runtime must degrade gracefully when a coprocessor is missing or uncertain.

## Strong baselines
Paper 7 must compare against:
- base LLM;
- larger unassisted LLM where feasible;
- matched prompting;
- conventional agent with explicit tools;
- RAG/upfront retrieval;
- strong source-native data agent/text-to-SQL baseline for analytics;
- proposed heterogeneous runtime;
- oracle routing/materialization where useful.

Do not compare only against intentionally weak tool use.

## Integration discipline
Only headline-integrate mechanisms validated earlier.

The candidate registry may contain additional engines, but label them:
- VALIDATED;
- EXPERIMENTAL;
- FALLBACK;
- PRODUCT-CANDIDATE.

Product relevance alone is not scientific evidence.

## Research/product separation
Paper 7 should remain vendor-neutral and architecture-first.

The implementation may be designed to support an eventual eInnovator product combining PRA + Cognitive Coprocessors, but the manuscript should:
- avoid product marketing;
- define open adapter interfaces;
- support multiple data/semantic/ontology backends;
- report reproducible open-source configurations;
- distinguish research conclusions from product roadmap choices.

## Suggested open reference stack
One reproducible reference configuration may use:
- Iceberg tables + REST-compatible catalog;
- an open SQL/query engine suitable for the benchmark;
- Cube Core or MetricFlow-class governed metric layer;
- Apache Jena or Oxigraph-class ontology/RDF layer;
- lexical + vector indexes;
- validated computational coprocessors;
- PRA if Paper 5 supports it.

Do not require this exact stack for the runtime architecture.

## Paper 7 evidence gates
Before claiming the integrated architecture:
1. Paper 2 must show heterogeneous compute value/interface viability.
2. Paper 2.5 must show source heterogeneity has value.
3. Papers 3/3.5 must justify learned routing where used.
4. Paper 4 must justify transactional/retraction machinery where used.
5. Paper 5 must justify PRA/native selective materialization where used.
6. Paper 6 must justify any co-adapted division-of-labor mechanism.

If a mechanism fails its earlier gate, Paper 7 should omit it or include it only as an ablation/fallback.

## Key falsification
The integrated thesis is weakened if:
- larger monolithic models dominate augmented smaller models at comparable total cost;
- explicit tools remain equally reliable with negligible context/routing burden;
- capability-count scaling causes learned/runtime routing to collapse;
- source-native data/semantic layers add no value over generic RAG/textualization;
- CPU/data-engine work does not meaningfully displace accelerator work;
- persistent state/PRA overhead exceeds saved context/compute;
- ontology/semantic layers add complexity without measurable query/reliability benefit.

## Required Paper 7 outputs
- integrated runtime;
- plugin/adapter contracts;
- broad candidate coprocessor registry;
- validated reference stack;
- Iceberg data adapter;
- semantic-layer adapters;
- ontology/RDF adapters;
- computational and retrieval adapters;
- capability-count scaling benchmark;
- small-model augmentation curves;
- enterprise-style Iceberg + semantics benchmark;
- context/capability virtualization ablations;
- CPU/XPU/data-engine accounting;
- provenance/reliability dashboard;
- full failure taxonomy;
- production-oriented deployment notes separated from scientific claims.
