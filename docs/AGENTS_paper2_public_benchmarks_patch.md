# AGENTS PATCH --- Paper 2: Public Heterogeneous Compute Benchmark Suite

## Purpose

Make Paper 2 the main public compute/symbolic benchmark paper for
heterogeneous coprocessors.

Paper 1 establishes calculator transfer. Paper 2 adds multiple compute
substrates and compares: - reasoning in weights; - four generic tools; -
explicit generic CogCop control; - CPU runtime triggers; - TwIL; -
hybrid TwIL + coprocessors; - later latent control.

## Required public benchmarks

### GSM8K

Retain as shared calculator anchor with Paper 1. Use harder subsets and
multi-step arithmetic.

### BIG-bench Unit Conversion

Primary units benchmark.

### BIG-bench Date Understanding

Primary date/temporal arithmetic benchmark.

### RuleTaker / ProofWriter

Primary Datalog/Horn reasoning benchmark.

Require balanced TRUE/FALSE/UNKNOWN or a carefully selected benchmark
configuration that supports non-degenerate labels.

### CLUTRR

Primary graph/relational-depth benchmark.

Exploit relation-length/depth scaling.

## Optional benchmark

Add a small SAT/SMT/formal reasoning dataset only after the core
five-engine suite is stable.

## Common conditions

For every benchmark where meaningful: 1. LLM only. 2. TwIL-LM3 neural.
3. Four generic tools: `__compute`, `__retrieve`, `__verify`, `__help`.
4. Generic COMPUTE/VERIFY/HELP token/block. 5. CPU trigger/router +
exact engine. 6. TwIL + exact coprocessor. 7. Oracle intent + exact
engine. 8. Paper 3 latent control when available.

Use same R2/engine backend across tool/CogCop conditions.

## Component decomposition

Score separately: - assistance need; - coarse intent; - capability
choice; - formalization/argument extraction; - engine execution; -
result integration; - final answer.

## Scaling analyses

### GSM8K

Accuracy vs number/difficulty of arithmetic steps.

### Unit conversion

Accuracy vs conversion chain length and novel-unit difficulty.

### Date understanding

Accuracy vs temporal-step complexity.

### RuleTaker/ProofWriter

Accuracy vs proof depth, fact/rule count, distractors.

### CLUTRR

Accuracy vs relational chain length and distractors.

The paper should emphasize curves, not only averages.

## Closure

For RuleTaker/ProofWriter/graph tasks, measure closure precision/recall
when possible.

## TwIL comparison

Use public logic tasks aligned with TwIL's training/evaluation style.
Compare: - SmolLM3; - TwIL; - SmolLM3 + coproc; - TwIL + coproc; -
oracle formalization + engine.

Use matched thinking budgets and report token/XPU cost.

## Generic tools skeptical baseline

Mandatory across all public compute benchmarks.

A fundamental CogCop claim requires: - automatic rescue of voluntary
tool misses; - lower JIT cost; - better CONTINUE; - better scaling; - or
earlier intervention.

Otherwise describe gains as interface/runtime efficiency.

## Deliverables

-   unified benchmark harness;
-   public dataset loaders;
-   immutable benchmark manifests;
-   depth/difficulty stratification;
-   tools-vs-CogCop-vs-TwIL tables;
-   scaling plots;
-   updated Paper 2 public-benchmark section;
-   revised Paper 3 gate.
