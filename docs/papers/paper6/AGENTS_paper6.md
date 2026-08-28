# AGENTS — Paper 6

## Title
Learning to Offload: Co-Adaptation Between Transformers and Cognitive Coprocessors

## Mission
Study whether models trained with permanent coprocessors learn a different division of labor: better formulation and result use with less redundant internal emulation.

## Required work
- Train matched with/without-coprocessor regimes.
- Measure offloading, redundant computation, result use, and robustness.
- Probe learned representations cautiously.
- Sweep model sizes.
- Ablate delayed/noisy/incorrect coprocessor outputs and coprocessor removal.

## Hard scope boundaries
- No mechanistic capacity-reallocation claim without probes.
- Keep engine correctness separate from interface learning.
- Always evaluate no-coprocessor fallback.

## Vocabulary
- **Neural plane:** Transformer-based semantic/generative computation.
- **Semantic interrupt:** automatically detected opportunity for specialized computation.
- **Micro-IR:** canonical typed representation between recognition and execution.
- **Coprocessor:** specialized computational subsystem.
- **Micro-state:** facts, expressions, results, derivations, constraints, provenance, or other structured state.
- **Reinjection:** exposing selected coprocessor results/state to subsequent neural computation.
- **Fallback tool:** conventional explicit function/tool invocation.

## Experimental invariants
- Pin model/tokenizer revisions and environment.
- Log every trigger, candidate/accepted IR, engine operation, derived item, reinjection, and failure.
- Separate detection, normalization, execution, state-management, reinjection/use, and final-answer errors.
- Use matched checkpoints/tasks across baselines.
- Include adversarial and non-trigger controls.
- Report multi-seed variability/confidence intervals for headline results.
- Preserve raw artifacts plus machine-readable summaries.
- Do not infer success from qualitative examples.

## Minimum baselines
1. Base LLM without assistance.
2. Matched prompting baseline.
3. Conventional explicit tool/function calling when applicable.
4. Proposed mechanism.
5. Oracle trigger/selection where useful.

## Metrics
- Final accuracy / exact match.
- Trigger precision/recall and false-intervention rate.
- Normalization and engine correctness.
- Generated tokens and model calls.
- Wall-clock and engine latency.
- State/context growth.
- Intervention count.
- Scaling with problem depth/size.
- Sensitivity to model size.

## Engineering rules
- Build a small reusable runtime core; avoid benchmark-specific shortcuts.
- Engines use typed deterministic interfaces and unit tests.
- Bound expression size, CPU time, memory, recursion, and search.
- Never use arbitrary generated-code execution where a bounded calculator/logic engine suffices.
- Keep ordinary tool calling available.
- Smoke-test cheaply before full sweeps.

## Evidence gate
Do not import later-paper mechanisms merely because they are attractive. Progress to tighter coupling only after a reproducible regime shows improved reliability, efficiency, or scaling, or after a negative result clearly motivates the next mechanism.

## Series dependency
Paper 0 defines the position. Paper 1 tests strict automatic calculator assistance. Paper 2 adds heterogeneous engines and persistent micro-state. Paper 3 learns semantic interrupts. Paper 4 adds transactional/backtracking state. Paper 5 studies structured/PRA/KV interfaces. Paper 6 studies co-adaptation. Paper 7 integrates validated mechanisms.

## Deliverables
- Tests and implementation for empirical papers.
- Reproducible task/dataset generators.
- Experiment manifests/configuration.
- Machine-readable result tables.
- Updated `paper{n}.tex` including negative findings and limitations.
- Reproduction commands.
