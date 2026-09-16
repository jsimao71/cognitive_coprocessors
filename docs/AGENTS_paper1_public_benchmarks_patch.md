# AGENTS PATCH --- Paper 1: Public Calculator/JIT Compute Benchmarks

## Purpose

Extend Paper 1 beyond synthetic arithmetic by adding public benchmarks
that test whether automatic calculator assistance improves real
natural-language reasoning tasks without changing the paper's narrow
single-coprocessor scope.

## Primary public benchmark: GSM8K

Use GSM8K as the main external benchmark.

Why it fits Paper 1: - one dominant coprocessor: calculator; -
natural-language interpretation remains neural; - arithmetic execution
can be offloaded; - supports JIT intervention rather than only direct
expression evaluation.

## Conditions

Run at minimum: 1. LLM only. 2. Matched prompt/ICL. 3. Generic
`__compute(...)` tool. 4. Existing calculator block/reflex. 5.
Runtime-trigger calculator if available. 6. Oracle calculator
opportunity/operation. 7. LoRA calculator block condition from Paper 1.

Do not require Paper 2 heterogeneous routing.

## Opportunity annotation

For each GSM8K example, derive or annotate arithmetic
subexpressions/opportunities.

Separate: - semantic decomposition correctness; - arithmetic operation
correctness; - calculator invocation correctness; - final answer
correctness.

If automatic extraction of all gold arithmetic steps is noisy, use: -
final arithmetic-bearing subset first; - manually audited development
slice; - full benchmark only for final-answer evaluation.

## Difficulty ladder

Create subsets by: - number of arithmetic operations; -
multiplication/division presence; - operand magnitude; -
intermediate-value dependency; - word-problem length.

The calculator advantage should increase with arithmetic difficulty if
the mechanism is real.

## Tool-vs-CogCop comparison

Use the same generic `__compute(...)` backend as the calculator
coprocessor.

Measure: - final accuracy; - assistance recall; - FAR; - malformed
tool/block rate; - generated control/argument tokens; - model calls; -
time to calculator execution; - Automatic Rescue Rate where runtime
trigger invokes compute when the voluntary tool path does not.

## JIT intervention

Where technically feasible, record whether the model begins producing an
incorrect arithmetic value before assistance.

Metrics: - first wrong-token prevention; - assistance lead time; -
voluntary-tool miss rescued by reflex.

## Secondary public benchmarks

Add small, scope-compatible slices from BIG-bench: - unit conversion
only if it can still be handled by the Paper 1 calculator without
introducing a dedicated unit engine; - date arithmetic only as an
optional transfer diagnostic, not as a headline result.

Dedicated units/date engines belong to Paper 2.

## Claim boundary

Paper 1 may claim: - calculator assistance transfers from synthetic
arithmetic to public natural-language arithmetic tasks.

It may NOT claim: - general heterogeneous coprocessor reasoning; -
general tool superiority; - multi-engine composition.

## Deliverables

-   GSM8K loader and immutable split config;
-   arithmetic-opportunity annotations/audit;
-   matched LLM/tool/CogCop conditions;
-   difficulty-stratified plots;
-   public-benchmark section in Paper 1;
-   explicit comparison between synthetic held-out results and GSM8K.
