# AGENTS PATCH --- Paper 3: Public Benchmarks for Token vs Latent Cognitive Control

## Purpose

Paper 3 should reuse the public benchmark suite from Papers 1/2/2.5 to
evaluate control transport, not introduce a completely new task
distribution.

The scientific question is: \> Given the same four generic cognitive
actions and the same R2/backend, does parallel latent START/type/END
control outperform voluntary generic tools or explicit token control?

## Core benchmark set

Use representative subsets rather than every prior benchmark:

### Compute

-   GSM8K
-   BIG-bench Unit Conversion or Date Understanding
-   RuleTaker/ProofWriter
-   CLUTRR

### Data/retrieval

-   TAT-QA
-   CRAG
-   bounded FRAMES subset

This gives arithmetic, structured compute, symbolic depth, relational
depth, retrieval, data composition, and epistemic timing.

## Conditions

1.  Four generic tools.
2.  Paired tags.
3.  Fenced blocks.
4.  Label-only generic intent.
5.  CPU automatic trigger.
6.  Final-layer latent controller.
7.  Multi-layer latent controller.
8.  Voluntary tools + latent watchdog.
9.  Oracle timing/type.

Same R2 and backends for all conditions.

## Required span annotations

For a manageable audited subset of each benchmark annotate: - earliest
useful START; - latest safe START; - canonical END; - payload span; -
assistance type; - first wrong/unsupported token where applicable.

Headline latent-control claims should use these audited span subsets.

## Public benchmark outcome metrics

In addition to task accuracy: - assistance recall/FAR; - exact/overlap
span quality; - intervention lead time; - Automatic Rescue Rate; -
malformed tool arguments vs latent-span parse failures; - generated
control tokens avoided; - model calls; - wall time; - CONTINUE
correctness.

## Multi-layer analysis

Across public tasks, measure whether layer-wise decodability differs by
assistance type: - COMPUTE; - RETRIEVE; - VERIFY; - HELP.

Test whether multi-layer aggregation generalizes across benchmark
families rather than only the synthetic Paper 3 span dataset.

## Registry invariance

Keep model-facing actions fixed while swapping or expanding concrete
runtime capabilities across benchmark families.

No retraining solely because Paper 3 changes from calculator to Datalog
to Iceberg/web.

## Tool comparison

Four generic tools are the principal skeptical baseline.

Paper 3 may claim a deeper control advantage only if latent/automatic
control shows: - meaningful rescue of voluntary tool misses; - earlier
safe intervention; - better task accuracy at matched FAR; - lower
decode/model-call cost; - better CONTINUE; - or scaling benefits.

If public benchmarks show parity, conclude generic tools are sufficient
for portable deployments.

## Deliverables

-   cross-paper public benchmark adapter layer;
-   audited latent-span subsets;
-   token/tool/CPU/latent matched runs;
-   cross-task layer-decoding plots;
-   rescue/timing/cost tables;
-   final evidence-based recommendation on whether latent control merits
    native inference-engine integration.
