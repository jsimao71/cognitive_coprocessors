# AGENTS - Paper 3.5 v3

## Mission

Test whether a causal parallel controller can mark RETRIEVE, VERIFY, or HELP
claim spans early enough to prevent unsupported commitment, and whether an
independent watchdog safely rescues voluntary misses.

## Gate inheritance

Paper 2.5's source-specific learned-router gate remains `no_go` because its
transparent heuristic closes the controlled oracle gap. The generic latent
watchdog is an independent exploratory protocol about timing and error
independence. It never predicts DB, lexical, vector, web, Iceberg, or another
backend in R1.

## Control and action

R1 uses `NONE`, `RETRIEVE`, `VERIFY`, `HELP`, with `COMPUTE` as a shared hard
negative. Reuse Paper 3's causal OUTSIDE/INSIDE START/type/END state machine.

Evidence requirement is separate from evidence availability. After a block,
runtime checks active context, typed task state, and optional PRA state before
calling Paper 2.5 R2. Reuse sufficient authoritative evidence when available.
The controller requests assistance; runtime owns source resolution,
credentials, provenance, evidence status, and enforcement.

## Dataset

Use generation trajectories. Annotate earliest safe START, latest acceptable
START, canonical END, first unsupported-value token, evidence requirement,
evidence availability, epistemic role, and authorized response policy. Include
current/private/changed facts, attribution, unavailable/conflicting evidence,
context-sufficient cases, quotations, hypotheticals, user-given/computed facts,
COMPUTE needs, and non-factual prose. Split paraphrase families, entities,
relations, domains, and model families.

## Required conditions

- Paper 1.5 semantic runtime rule.
- FLARE-like confidence.
- Paired, fenced, and label-only voluntary generic intent.
- Text-prefix sidecar.
- Final-layer and multi-layer latent controllers.
- Voluntary plus independent watchdog hybrid.
- Faithfully labeled Self-RAG or Self-RAG-inspired comparator.
- Oracle timing/type.

Compare voluntary-only, watchdog-only, and hybrid errors. Report Watchdog Rescue
Rate only with FAR, unnecessary retrieval, abstention cost, lead time, and error
correlation.

## R2 and enforcement

After R1, compare Paper 2.5 heuristic routing, token-aware BM25,
capability/source-native rules, broadcast, and oracle source. Do not learn source
identity without a new heuristic-to-oracle gap.

For UNVERIFIED, STALE, CONFLICT, AMBIGUOUS, or unavailable required evidence,
runtime may abstain, qualify, report conflict, or retrieve more. Measure UCR,
Authorized Commitment Coverage, and evidence override. Integration LoRA is a
separate gated experiment after trigger quality.

## Metrics and falsification

Report span/type/FAR, Early/Late Catch, missed-before-commitment, lead tokens,
Watchdog Rescue Rate, error correlation, UCR, Authorized Commitment Coverage,
unnecessary retrieval, override, abstention quality, R2 correctness, fanout,
latency/cost, calibration, and OOD/cross-model transfer. Report cost/frontier
curves and high-confidence/high-risk slices, not F1 alone.

The watchdog is not justified if semantic rules match the UCR-cost frontier,
voluntary control catches nearly all needs, watchdog errors correlate with
generator misses, costs dominate rescues, or hidden-state dependence is not
portable.

## Scope and source of truth

No source identity in R1, transactional/backtracking contribution, silent truth
promotion, or PRA/native-KV dependency. PRA is an optional active-evidence and
capability-materialization hook.

- Manuscript: `paper3_5_v3.tex`
- Built artifact: `paper3_5_v3.pdf`
- Prior drafts: `paper3_5.tex`, `paper3_5_v2.tex`
- Roadmap input: `../../AGENTS_paper3_5_v2.md`
