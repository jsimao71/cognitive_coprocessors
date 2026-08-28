# AGENTS — Paper 1

## Title
Reflex Computation: Automatic Calculator Assistance During Autoregressive Decoding

## Mission
Minimal empirical proof: a calculator watches generation for strict arithmetic syntax, evaluates completed expressions automatically, and reinjects results without an explicit model tool-call decision.

## Required work
- Implement incremental strict arithmetic recognition.
- Implement bounded deterministic evaluation.
- Compare LLM-only, explicit calculator tool calling, and automatic calculator assistance.
- Sweep arithmetic difficulty and model size.
- Measure accuracy, trigger precision/recall, tokens, latency, interventions, and scaling curves.

## Hard scope boundaries
- Calculator only.
- Strict syntax only.
- No learned trigger.
- No symbolic memory/backtracking/PRA/native KV.

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

## ICL and adaptation iteration

The current developmental sequence is ICL, then matched multi-model diagnosis,
then an explicit LoRA decision. Do not train an adapter before the interface gap
is demonstrated across capability levels.

- Preserve the original seven-condition prompt version and `hard_heldout_xpu`
  artifacts.
- Endpoint rescoring uses a parallel condition-independent label and never
  replaces `predicted_answer` or the reported summary.
- Calculator-block evaluation decomposes opening, closing, payload presence,
  exact copying, semantic equivalence, execution, result use, and false blocks.
- ICL-G (`paper1_calculator_block_icl_v2_order_control`) is the frozen generic
  multi-model prompt. It places a negative demonstration before a final positive
  calculator-block demonstration.
- On the four-arithmetic/four-control Qwen3-0.6B developmental set, ICL-G had
  100% semantically equivalent payloads, execution, final accuracy, and result
  use with zero false blocks. Verbatim payload exactness was 0% because redundant
  parentheses were removed. This passes the developmental gate but is not
  confirmatory evidence.
- Use `--smoke` before every larger XPU run; it selects four arithmetic examples
  and two controls and records device, dtype, revision, chat settings, time, and
  memory.
- Authorization-gated Llama and Gemma checkpoints remain skipped until a
  Hugging Face token with accepted licenses is available.
- The 196-row phase remains developmental. Freeze a new untouched and powered
  confirmatory set only after model/interface development ends.
- On the 16-arithmetic/12-control diagnostic, Qwen3-0.6B ICL-G executed all
  arithmetic blocks but falsely blocked 2 controls. Qwen3-1.7B executed 15/16
  blocks with zero false controls and gained 9/lost 1 against LLM-only.
- Qwen3-4B passed the six-item XPU smoke with perfect block execution and no
  false block at 8.21 GB peak memory. Do not report it as a held-out estimate.
- Current evidence is Qwen-only; Llama/Gemma authorization skips are part of
  the result and community mirrors must not be substituted.
- LoRA is scientifically justified under Regime A for Qwen3-0.6B control
  discrimination. PEFT 0.20.0 and Accelerate 1.14.0 passed an XPU optimizer-step
  gate with the existing Torch 2.13.0+XPU and Transformers 4.57.6 stack.
- The frozen protocol dataset contains 80 arithmetic/80 control training rows
  and 20/20 development rows. Its audit must retain zero operand/expression
  overlap with the held-out benchmark; targets never contain arithmetic answers.
- Train Qwen3-0.6B first. Llama/Gemma remain license-gated, so the predeclared
  ungated second-family fallback is official HuggingFaceTB SmolLM2-1.7B-Instruct.
- The primary placement comparison is context (base+ICL-G), weights
  (LoRA+minimal), and runtime (base+normalized reflex). Also measure base+minimal
  and LoRA+ICL to isolate instruction and interaction effects.
- Regenerate cross-model artifacts with `paper1 compare-models` and
  `configs/paper1/block_icl_comparison.json`.

## Series dependency
Paper 0 defines the position. Paper 1 tests strict automatic calculator assistance. Paper 2 adds heterogeneous engines and persistent micro-state. Paper 3 learns semantic interrupts. Paper 4 adds transactional/backtracking state. Paper 5 studies structured/PRA/KV interfaces. Paper 6 studies co-adaptation. Paper 7 integrates validated mechanisms.

## Deliverables
- Tests and implementation for empirical papers.
- Reproducible task/dataset generators.
- Experiment manifests/configuration.
- Machine-readable result tables.
- Updated `paper{n}.tex` including negative findings and limitations.
- Reproduction commands.
