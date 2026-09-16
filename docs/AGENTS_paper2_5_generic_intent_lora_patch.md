# PATCH — Paper 2.5: Generic RETRIEVE/VERIFY/HELP Intent

## Motivation
Do not couple model weights to DuckDB/Postgres/Iceberg/lexical/vector/web/ontology implementation names.

Test a stable model-facing intent:
- NONE
- RETRIEVE
- VERIFY
- HELP
- COMPUTE

The runtime resolves the concrete data/retrieval coprocessor from the deployment-specific registry.

## Hypothesis
Generic retrieval intent should survive source/backend changes without LoRA retraining or large source descriptions in recurring context.

## Syntax conditions

### A — paired tags
`<RETRIEVE>` evidence need `</RETRIEVE>`
and VERIFY/HELP/COMPUTE analogues.

### B — fenced block
Conceptually:
three-backtick RETRIEVE
evidence need
three-backtick

### C — label only
Emit only RETRIEVE/VERIFY/HELP/COMPUTE/NONE; runtime uses original active task as payload.

Measure exact token cost and parsing reliability for all model tokenizers.

## Dataset
Create a new freeze independent of the existing source-routing test.

### RETRIEVE
Mix needs whose eventual sources are:
- structured DB;
- lexical;
- vector;
- web/fresh;
- later Iceberg/semantic/ontology.

Target says RETRIEVE only.

### VERIFY
Model has a candidate factual claim/value and asks runtime to validate it. Verification may use existing evidence, DB, web, or another configured source.

### HELP
Evidence/capability need is real but source family is unclear.

### COMPUTE
Hard negatives for retrieval: arithmetic, units, date, formal computation.

### NONE
Context-sufficient, stable facts, quotations, hypotheticals, ordinary prose.

## Source-registry change experiment
Train generic R1 with a small registry such as:
- DB
- lexical

Evaluate without retraining after adding:
- vector
- web
- Postgres
- Iceberg-backed source
- ontology/semantic adapters as they become available.

The generic RETRIEVE policy should not care which backend is installed.

## Conditions
1. current runtime semantic retrieval trigger;
2. generic ICL;
3. generic label-only LoRA;
4. paired-tag LoRA;
5. fenced-block LoRA;
6. prior source-specific/context block where useful;
7. oracle generic intent.

Do NOT train source-specific learned routing in this experiment.

## R2 runtime source resolution
After RETRIEVE compare:
- existing heuristic router;
- tokenizer/BM25 router;
- capability/source-native rules;
- oracle source;
- broadcast.

For HELP, allow broader registry search.

## PRA integration mode
Paper 2.5 should document and, if implementation is cheap, optionally prototype:

RETRIEVE/HELP
-> runtime/PRA search capability/source registry
-> shortlist relevant source/skill descriptions
-> materialize only shortlist
-> resolve query
-> execute.

This is optional. Generic intent must work without PRA.

## Production data-stack compatibility
The generic protocol must remain unchanged while the Paper 2.5 backend evolves through:
- DuckDB;
- Postgres/pgvector;
- FAISS/Qdrant;
- Iceberg;
- semantic layer;
- ontology/RDF;
- web.

This is a major product-oriented test:
> can the data capability registry evolve independently of the model?

## Metrics
R1:
- RETRIEVE recall;
- VERIFY/HELP recall;
- COMPUTE-vs-RETRIEVE confusion;
- NONE FAR;
- UCR when connected to enforcement.

Syntax:
- token cost;
- parse success;
- latency;
- accidental markdown/tag activation.

R2:
- source/capability correctness;
- final support/accuracy;
- source calls;
- route latency;
- broadcast cost.

Registry scaling:
- R1 quality vs number/type of installed sources;
- retraining required: ideally zero;
- context tokens vs registry size.

## Runtime enforcement
Keep the successful Paper 1.5 principle:
generic RETRIEVE/VERIFY only requests assistance; runtime owns evidence status, provenance, credentials, support policy, and enforcement.

## Gate
Generic intent is valuable if:
- retrieval recall remains high;
- NONE FAR remains low;
- COMPUTE/RETRIEVE confusion is low;
- registry expansion does not materially degrade R1;
- R2 heuristic/source-native routing remains strong;
- no model retraining is needed when adding backends.

## Immediate order
1. freeze generic-intent data;
2. measure paired-tag/fence/label tokenization;
3. run ICL;
4. train one small generic LoRA;
5. compare syntax variants;
6. registry-expansion test;
7. connect best R1 to existing heuristic and tokenizer/BM25 R2;
8. repeat after DuckDB/Postgres/Iceberg/vector backends are installed;
9. re-evaluate Paper 3.5 only if meaningful headroom remains.

## Deliverables
- generic epistemic-intent specification;
- syntax/token comparison;
- generic LoRA/ICL artifacts;
- registry-change benchmark;
- R1/R2 factorization;
- production-backend invariance test;
- updated Paper 2.5 manuscript and Paper 3.5 gate.
