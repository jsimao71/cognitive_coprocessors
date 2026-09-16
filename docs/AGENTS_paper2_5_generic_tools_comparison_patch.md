# Paper 2.5 Patch --- Data Coprocessors vs Four Generic Tools

## Mandatory four-tool baseline

Use exactly `__compute(...)`, `__retrieve(...)`, `__verify(...)`,
`__help(...)`. These generic gateway tools MUST use the same R2
registry, parsers, engines, provenance, policies, typed state, and
result integration as CogCop. Concrete registry entries remain outside
the recurring schema.

Compare voluntary generic tools, generic CogCop token/block, CPU
automatic trigger, latent parallel control where applicable, and oracle
timing/type. Match checkpoint, prompts, decoding, backend
registry/results, R2 policy, assistance opportunities, and safety
limits.

Measure reliability (assistance recall, FAR, R1/R2 accuracy, final
accuracy, UCR, result use, malformed args), interface cost
(schema/ICL/generated/argument/result tokens), execution cost (model
calls, decode steps, harness/network round trips, time-to-assistance,
CPU/XPU/wall time), registry scaling, and autonomy.

Define
`Automatic Rescue Rate = voluntary-tool misses rescued by automatic/latent interrupt / assistance-required cases missed by voluntary tools`.

If tools match quality/cost, tighter coupling is not justified. If
CogCop only saves tokens/network, claim engineering efficiency only. A
deeper claim requires automatic rescue, better JIT timing, CONTINUE, or
scaling benefits.

## Production invariant

All conditions reach the same DuckDB/Iceberg, governed metrics,
Oxigraph, FTS5, FAISS, and gated Postgres/pgvector/Qdrant/Iceberg-REST
adapters. Never weaken the tool backend or silently fallback.

## Matrix

D0 model only; D1 four generic tools; D2 optional source-specific tools;
D3 generic RETRIEVE/VERIFY/HELP block; D4 semantic automatic trigger; D5
latent Paper-3.5 control later; D6 oracle.

## Enterprise benchmark

Re-run direct lookup, join, aggregate, historical snapshot, governed
metric, ontology, ontology-\>metric, document, and mixed cases through
the generic-tool interface. The model need not name
DuckDB/Iceberg/Oxigraph/etc.; `__retrieve`/`__help` enters the same R2
composition runtime.

## LLM-facing benchmark

The existing 11/11 run is deterministic dispatch. Add model-facing
NONE/RETRIEVE/VERIFY/HELP/COMPUTE choice while keeping R2 deterministic
initially. Compare generic tools vs generic CogCop.

## Evidence timing

Create stale-weight/current-source cases. Measure voluntary `__retrieve`
recall, automatic trigger recall, unsupported commitment before call,
UCR after enforcement, and Automatic Rescue Rate.

## PRA factorial

Where feasible compare tools, tools+PRA, CogCop, CogCop+PRA. Attribute
progressive registry/state context savings to PRA, not CogCop.

## Registry scaling

Keep four schemas fixed while registry grows: DuckDB/FTS5/FAISS;
+Iceberg/metrics/Oxigraph; +Postgres/pgvector/Qdrant; optional synthetic
expansion. Measure model-visible tokens, R2 time, source/final accuracy,
and zero-retraining behavior.

## Structured-authority use

Matched cases: historical vs current snapshot, governed metric vs
invented formula, ontology-normalized family vs lexical guess. Compare
standard tool result vs typed CogCop result while backend output is
identical.

## Claim gate

If four generic tools + same R2 + enforcement match CogCop, conclude
that source-native data cognition/registry abstraction is supported but
a novel invocation mechanism is not. Stronger claims require
automatic/JIT rescue, better evidence timing/integration, or overhead
gains beyond PRA/schema savings.
