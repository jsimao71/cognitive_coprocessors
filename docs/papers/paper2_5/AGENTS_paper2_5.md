# AGENTS - Paper 2.5

## Mission
Test whether structurally different evidence needs benefit from distinct
read-only epistemic coprocessors rather than one generic retriever. Keep need and
source routing transparent and heuristic until source-native oracle value and a
measurable routing gap justify learning.

## Central question
Do heterogeneous evidence substrates provide enough source-specific value that
the runtime should treat relational, lexical, vector, and fresh web evidence as
distinct coprocessors?

Secondary: can capability count grow outside recurring neural context?

## Frozen next-iteration outcome
- The retained benchmark has 22 examples: six DB, three lexical, three vector,
  four controlled-web, and six supplied-context controls.
- Source-native oracle, heuristic routing, explicit selection, and broadcast all
  reach final accuracy 1.0. The universal textual retriever reaches 0.5909,
  evidence support 0.4375, and UCR 0.5625 on retrieval-required rows.
- Explicit descriptor burden grows from 4.0 to 22.545 mean lexical tokens from
  one to four sources. Broadcast averages 2.909 calls versus 0.727 for routing.
- The heuristic closes the oracle routing gap, so the machine-readable Paper 3.5
  gate is `no_go`. Do not train a learned source router on this freeze.
- A separately versioned local production-backend substitution replaces the DB,
  lexical, and vector implementations with DuckDB 1.5.5, SQLite 3.49.1 FTS5,
  and FAISS 1.15.0 exact-flat search. It preserves all 154 matched four-source
  final/support decisions and complete required provenance for the three
  substituted sources.
- This substitution validates the interface, not production data or service
  operation. The corpus, embeddings, and web source remain controlled; Docker
  and remote services were not used.
- A WSL2-only Postgres/pgvector sidekick and explicit DSN-gated integration tests
  are present but unexecuted. They must never fall back to controlled adapters,
  and they are not evidence until a separately retained service run exists.
- Retrieval-only Qdrant and Iceberg REST-catalog adapters are also present with
  injected-client unit tests and explicit endpoint gates. Qdrant is an optional
  localhost-bound WSL profile. Neither path was executed and neither contributes
  a paper result.
- A local enterprise fixture creates four real PyIceberg tables, two sales
  snapshots, one schema evolution, governed metric definitions, and an Oxigraph
  product ontology. Native governed execution is 11/11 versus 2/11 for top-5
  textualized retrieval. This is a deterministic six-sale diagnostic, not a
  production-scale or language-model result.
- The enterprise result does not measure a new heuristic routing gap and does
  not reopen the Paper 3.5 gate.
- A pinned public TAT-QA development diagnostic freezes 320/1,644 questions by
  answer type, evidence source, scale, and comparison requirement. It retains
  IDs, source coordinates, strata, and content hashes only.
- In that sample, 129/320 questions require arithmetic or counting and 130/320
  require joint table--text evidence. The safe decimal evaluator reproduces all
  106/106 annotated arithmetic derivations exactly.
- Current source-native TAT-QA adapter coverage is 0/320. Treat this as an
  end-to-end operation-adapter gap, not final-answer accuracy.
- At matched top-5 lexical retrieval, flattened word BM25 reaches 0.704 mean
  evidence recall, character BM25 0.756, and their flattened hybrid 0.751. A
  row/column-header-aware hybrid reaches 0.799 over 308 evaluable questions.
- The paired structured-minus-flat recall gain is 0.048 with a deterministic
  10,000-sample bootstrap 95% interval [0.020, 0.077]. Complete evidence has 24
  wins, 8 losses, and 276 ties. Text-only recall is unchanged; gains occur on
  table and table--text questions.
- Ranking is gold-free, but scoring uses annotated paragraph orders and table
  labels induced from gold spans/arithmetic operands. Twelve rows lack
  defensible labels. This is not yet a typed source-native, generic-tool,
  LLM-only, semantic-vector, or final-answer comparison.

These are controlled deterministic runtime results, not language-model or live
web-search evidence. New natural-language routing work requires a separately
versioned benchmark and gate.

## Source suite
1. Embedded relational DB with typed lookup, SUM, COUNT, AVG, argmax, and join.
2. Lexical document index for exact document/phrase retrieval.
3. Dense semantic report retriever for paraphrased topical questions.
4. Controlled fresh-web analogue for current public values and conflict.
5. Knowledge graph is optional only if it adds a distinct access pattern.

These must differ semantically, not merely wrap one text index.

## Common source and evidence IR
Every source request contains request ID, source type, operation, typed payload,
and bounds. Every evidence record contains source type/ID, record ID, value,
content, observation time, relevance, evidence status, provenance, latency, and
retrieved bytes.

Evidence status is one of SUPPORTED, CONTRADICTED, UNVERIFIED, STALE, AMBIGUOUS,
or CONFLICT. The runtime must not silently resolve conflicts.

## Registry and policy
The source registry owns credentials, validation, source invocation, provenance,
and enforcement. Public source descriptors exclude raw credential scopes. Paper
2.5 is strictly read-only. Log locality, latency/cost class, privacy, freshness,
and side-effect class for each adapter.

## Primary router
Use transparent source semantics:

- structured attributes and aggregates -> DB;
- exact document or clause -> lexical;
- semantic report question -> vector;
- latest/current/public fact -> web.

Paper 1.5 may supply a typed retrieval-required event, but Paper 2.5 studies
SOURCE and source-specific QUERY. Do not make a learned router primary.

## Source-count scaling
Use nested catalogs of 1, 2, 3, and 4 sources. Compare explicit source schemas in
context with a runtime registry outside context. Measure descriptor tokens,
source selection, query correctness, fanout, answer support, UCR, calls, bytes,
evidence tokens, source latency, and total wall time.

## Oracle matrix
1. Oracle need + oracle source + oracle query.
2. Real typed need + oracle source.
3. Real typed need + heuristic source.
4. Heuristic need + heuristic source.
5. Explicit source/tool selection with all descriptors.
6. Universal textual retriever.
7. Broadcast to every available source.

The universal baseline textualizes all available records into one matched index.
The source-native DB oracle quantifies aggregate/join information lost by
textualization. Broadcast is mandatory and must report fanout and source cost.

## Factorized evaluation
Always report:

NEED -> SOURCE -> QUERY -> RETRIEVE -> EVIDENCE STATUS -> USE

Do not hide routing or query errors inside final answer accuracy. Report
unsupported commitments when evidence is absent or insufficient.

## Bounded composition
Allow deterministic two-stage helpers such as entity resolver -> DB and date
resolver -> web. Preserve a dependency DAG and component cost. Do not implement a
general planner.

## Economics
Track CPU/source time, network-analogue time, source calls, fanout, bytes,
evidence tokens, descriptor tokens, query tokens, and total wall time. Separate
prototype latency from recurring neural-context burden and external-source cost.

## Failure taxonomy
Missed/false retrieval need, wrong source, unnecessary fanout, unavailable
source, malformed or wrong query, stale evidence, ranking miss, conflict,
credential denial, timeout, budget violation, ignored evidence, unsupported
answer, and wrong source treated as authoritative.

## Scope
No learned source router, hidden-state trigger, write operation, general planner,
transactional rollback, or PRA/native KV integration. The controlled web source
is a freshness-aware analogue, not a claim about live search quality.

## Paper 3.5 gate
Learned source routing is justified only if all are measured:

1. Source-native oracle value over a universal retriever.
2. A non-zero heuristic routing gap.
3. Source descriptor/context burden grows with source count.
4. Broadcast does not dominate quality and cost.

If the heuristic closes the oracle gap, record `no_go`; do not manufacture hard
examples or train a router merely to progress the series. The frozen next
iteration records `no_go` because the heuristic gap criterion fails.

## Falsification
Heterogeneous routing is weak if universal retrieval matches native execution,
oracle routing adds little, broadcast dominates at matched cost, source-specific
query semantics do not matter, or explicit selection is equally efficient.

## Deliverables
The repository must contain the common retrieval IR, source registry, four
adapters, source-optimal benchmark, source-count sweeps, complete oracle matrix,
universal and broadcast controls, conflict enforcement, bounded compositions,
factorized traces and economics, plots, rebuilt paper, and explicit Paper 3.5
go/no-go artifact.

The production-data-stack continuation additionally requires swappable backend
descriptors, typed validation, normalized-query and snapshot provenance, local
DuckDB/FTS5/FAISS substitution, and an explicit executed-versus-deferred backend
boundary. Service, Iceberg, semantic-layer, and ontology results require their
own versioned artifacts before manuscript claims are expanded.

## Tokenizer add-on scope review
The Paper 1.5/Paper 2 tokenizer-aware trigger add-on was reviewed against the
production-data-stack v2 roadmap. It changes NEED/engine trigger evidence, not
Paper 2.5 SOURCE/QUERY selection or backend substitution. Do not reuse Paper 2
engine-token scores as source-routing evidence. No Paper 2.5 rerun is required;
the measured heuristic-to-oracle source gap remains zero and Paper 3.5 stays
`no_go`.

## Public benchmark checkpoint
The TAT-QA source is `next-tat/TAT-QA` at revision
`c96247f5077eac447f63527fd3dcfdc58bb56d6a`, with the development JSON checksum
recorded in `configs/paper2_5/public_tatqa.json`. The source file must remain in a
verified local cache. Never commit question text, tables, paragraphs, answers,
or derivations.

The current audit separates retrieval requirement, compute requirement,
composition depth, gold-compute availability, gold-compute exactness, and
source-native adapter availability. Gold arithmetic validates only the compute
stage. It must not be reported as model execution or end-to-end QA accuracy.

The retrieval continuation compares word BM25, character BM25, reciprocal-rank
hybrid, and a structure-aware hybrid at the same top-5 budget. Gold evidence may
be used only after ranking. Preserve the paired bootstrap seed and report the 12
non-evaluable questions rather than assigning them zero recall.
