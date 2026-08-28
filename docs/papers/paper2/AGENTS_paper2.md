# AGENTS — Paper 2

## Title
Heterogeneous Reflex Reasoning: Multiple Symbolic Coprocessors with Typed Micro-State

## Mission
Generalize automatic assistance to multiple engines and persistent typed micro-state while retaining strict syntax and transparent heuristic triggers.

## Required work
- Retain calculator; add Horn/Datalog logic, ISA/frame graph reasoning, and optionally simple constraints.
- Define common typed micro-IR and provenance.
- Implement bounded persistent state and forward closure.
- Create mixed-engine tasks.
- Compare no-engine, explicit tools, single engines, multiple engines, and oracle selection.
- Measure scaling with depth, entities, rules, distractors, state size, tokens, and latency.

## Hard scope boundaries
- No learned trigger.
- Use a controlled FOL/Horn fragment, not unrestricted theorem proving.
- No rollback/backtracking.
- No PRA dependency.

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
