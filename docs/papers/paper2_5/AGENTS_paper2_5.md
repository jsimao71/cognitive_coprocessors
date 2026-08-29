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
