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
- Approved official Gemma3-1B access is now authenticated and pinned at
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`; Llama remains unavailable.
- The 196-row phase remains developmental. Freeze a new untouched and powered
  confirmatory set only after model/interface development ends.
- On the 16-arithmetic/12-control diagnostic, Qwen3-0.6B ICL-G executed all
  arithmetic blocks but falsely blocked 2 controls. Qwen3-1.7B executed 15/16
  blocks with zero false controls and gained 9/lost 1 against LLM-only.
- Qwen3-4B passed the six-item XPU smoke with perfect block execution and no
  false block at 8.21 GB peak memory. Do not report it as a held-out estimate.
- Capability-scaling evidence remains Qwen-only because its sweep preceded
  authentication. The adapter-placement comparison adds official SmolLM2 and
  approved official Gemma3-1B; community mirrors must not be substituted.
- LoRA is scientifically justified under Regime A for Qwen3-0.6B control
  discrimination. PEFT 0.20.0 and Accelerate 1.14.0 passed an XPU optimizer-step
  gate with the existing Torch 2.13.0+XPU and Transformers 4.57.6 stack.
- The frozen protocol dataset contains 80 arithmetic/80 control training rows
  and 20/20 development rows. Its audit must retain zero operand/expression
  overlap with the held-out benchmark; targets never contain arithmetic answers.
- Train Qwen3-0.6B first and use official HuggingFaceTB SmolLM2-1.7B-Instruct as
  the ungated replication. Gemma3-1B is the approved targeted-family replication.
- The primary placement comparison is context (base+ICL-G), weights
  (LoRA+minimal), and runtime (base+normalized reflex). Also measure base+minimal
  and LoRA+ICL to isolate instruction and interaction effects.
- Qwen3-0.6B LoRA+minimal achieved 16/16 block execution and answers with 0/12
  false blocks; base+minimal achieved 0/16. Base+ICL-G had 2/12 false blocks and
  LoRA+ICL-G had 4/12, so context and weights show negative interaction.
- The Qwen adapter has 2,293,760 trainable parameters (0.383%), is 9.2 MB, used
  7,362 target tokens, trained for 394.5 seconds, and peaked at 1.48 GB on XPU.
- SmolLM2 LoRA+minimal also achieved 16/16 execution and answers with 0/12 false
  blocks; base+minimal achieved 0/16. Base+ICL-G reached 15/16 execution, 14/16
  answers, and 1/12 false block; LoRA+ICL retained that false block.
- The SmolLM2 adapter has 3,145,728 trainable parameters (0.183%), is 12.6 MB,
  used 7,542 target tokens, trained for 768.0 seconds, and peaked at 3.59 GB.
- Gemma3-1B uses BF16 and has 1,490,944 trainable parameters (0.149%), is 6.0 MB,
  used 7,362 target tokens, trained for 365.0 seconds, and peaked at 2.52 GB.
- Gemma LoRA+minimal achieved 16/16 execution, 15/16 answers, and 0/12 false
  blocks; LoRA+ICL-G achieved 16/16 answers and 0/12 false blocks. Base ICL-G
  executed 5/16 with 1/12 false block; normalized runtime executed 4/16 with
  1/12 false intervention. Oracle executed 16/16 but the model used 0/16 results.
- Hugging Face generation must honor every model `generation_config.eos_token_id`,
  including Gemma `<end_of_turn>` token 106. Access-smoke `v1`, BF16-smoke `v1`,
  and adapter-eval `v1` are excluded EOS-bug diagnostics; use BF16-smoke `v2`,
  adapter-eval `v2`, and base-eval `v1`.
- Current placement conclusion: context is for development/cold start, adapter
  weights store a stable semantic selection/serialization contract, and the
  runtime retains parsing, bounds, exact execution, reinjection, and enforced
  result use. Residual context is model-specific, not universally additive.
- Regenerate cross-model artifacts with `paper1 compare-models` and
  `configs/paper1/block_icl_comparison.json`.
- Regenerate placement artifacts with `paper1 analyze-placement` and
  `configs/paper1/lora_placement_comparison.json`.
- The current machine-readable placement decision is
  `artifacts/paper1/lora_protocol/lora_decision_final_v4.json`.

## Public GSM8K transfer checkpoint

- The developmental public slice is frozen at
  `artifacts/paper1/public_gsm8k_v1/data/selection.jsonl`: 120 test rows, with
  40 each in the `2_steps`, `3_4_steps`, and `5plus_steps` strata. Its selection
  SHA-256 is `71a2d67253778be7060b990759e852cbfc9ff86ca8bd6001be655dfccb72b48e`.
- The run uses Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`,
  greedy seed 23011, FP16 XPU, thinking disabled, and a 160-token cap. It has one
  checkpoint and one seed and is developmental, not a benchmark-wide estimate.
- The oracle ledger must be generated by normalizing and executing supported
  source expressions through `BoundedCalculator`; never trust source-side ledger
  values directly. The source has 425 annotated operations, of which 382 (89.9%)
  are supported. The corrupted-source unit test protects this provenance rule.
- Final accuracies on 120 matched rows are: LLM-only 13/120 (10.8%), matched ICL
  4/120 (3.3%), generic compute 3/120 (2.5%), base block prompt 13/120 (10.8%),
  automatic runtime 4/120 (3.3%), oracle ledger 91/120 (75.8%), and LoRA block
  1/120 (0.8%).
- The base block prompt never opens a valid block. Its tie with LLM-only swaps 12
  paired gains for 12 losses and is a prompt effect, not calculator evidence.
- Generic compute accepts calls on 114/120 rows but has zero registered-operation
  recall; 168/225 calls are the prompt demonstration `7*8`, 111/120 rows repeat a
  call, and 6/120 rows contain malformed calls.
- Automatic runtime accepts assistance on 114/120 rows, has 6.0% operation recall,
  and gains 2/loses 11 pairs versus LLM-only. Automatic Rescue Rate is 0/6 under
  the frozen definition based on voluntary rows with no valid call.
- LoRA block accepts assistance on 79/120 rows, but 109/120 rows contain malformed
  activity, operation recall is 6.5%, only 9/120 rows emit a scorable endpoint,
  and one row is correct. Synthetic 16/16 behavior does not transfer to GSM8K
  decomposition.
- Oracle ledger accuracy is 91/120 (75.8%, 95% Wilson CI 67.4--82.6%), with 79
  paired gains and one loss versus LLM-only. This is arithmetic/formalization
  headroom, not deployable routing evidence.
- GSM8K has no matched no-compute controls or token alignment. FAR and
  first-wrong-token prevention remain undefined. The concrete `7*8` syntax example
  was frozen before the full run and causes strong anchoring; do not generalize
  this protocol failure to every possible tool prompt.
- Current evidence gate: Paper 1 establishes synthetic interface feasibility and
  public oracle headroom, but not public calculator-assistance transfer. Public
  semantic decomposition must be solved and replicated before a stronger claim.

## ASL/CCIR dataset bootstrap checkpoint

- The new target is `NL -> ASL-Arith -> generic AST -> typed CCIR -> scoped
  workspace -> operator registry`. ASL is model-facing; JSON/Python dictionaries
  remain canonical storage.
- `src/ccpu/dsl` implements an open-vocabulary Pratt parser, canonical lowering,
  hierarchical scope workspace, arithmetic registry, deterministic execution,
  and fail-closed validation. Extensibility tests parse `:-`, variables, calls,
  queries, `<-`, and explicit `SCOPE/END` without claiming those semantics.
- `src/ccpu/dsl_dataset` provides a Click CLI for mining, chop audits,
  deterministic selection, answer-free local batches, labeled repair batches,
  semantic validation, annotation bootstrap, and remote LiteLLM generation.
  Tokens are environment-owned and `.env.example` contains names only.
- The pinned raw checkpoint mines 7,473 GSM8K train, 1,644 TAT-QA development,
  1,146 CLUTRR test, and 360 ProofWriter test records. CLUTRR and ProofWriter are
  ingestion-only for Paper 2; never force them into ASL-Arith.
- One public item equals one external root scope. The audit has zero scope/ID
  collisions and zero teacher-facing chop failures. Remaining 6 GSM8K, 10
  CLUTRR, and 3 ProofWriter warnings occur only in non-teacher rationale or
  ingestion fragments and remain visible.
- The semantic-teacher seed is 100 stratified GSM8K plus 50 arithmetic TAT-QA
  rows. They produce 343 and 50 potential clause requests under the current
  skill hash. No remote teacher has been called.
- TAT-QA rows include source table and paragraph context; question-only requests
  are insufficient for grounded metric/year names. Primary teacher requests hide
  answers, rationales, equations, and existing programs.
- The preserved annotation-derived Q0 operation ledger accepts 95/100 GSM8K and 50/50 TAT-QA
  programs after parser, scope, execution, and final-answer verification. Four
  GSM8K chains mismatch the final answer and one has no equation; do not repair
  them semantically or hide the rejection.
- Q0 rows are operation plans derived from dataset annotations with generic step
  names. They are not semantic teacher gold and cannot support an NL-to-ASL,
  ICL, LoRA, or placement claim.
- The revised local skill is `skills/ccir-arith-compiler/SKILL.md`, with detailed
  semantic rules in `references/semantic-annotation.md`. It requires entities,
  measured quantities, source/derived and temporal state, forward dependencies,
  named returns, and legal identifiers such as `y2019` rather than `.2019`.
  Grammar or semantic
  changes require a version bump and new artifacts; never overwrite prior raw or
  teacher outputs.
- The semantic corpus accepts 150/150 programs: 98 answer-blind primary Codex
  mappings, 45 rationale-assisted round-one repairs, and seven round-two repairs.
  Deterministic checks cover syntax, CCIR, scope, grounded execution, final answer,
  named semantic state, and named return. Grade is Q1, not Q4; no row is marked
  manually reviewed. Current runs used Codex CLI and zero LiteLLM calls.
- ICL and LoRA remain blocked until a manual semantic audit and lineage-safe split,
  unless an explicitly labeled Q1 pilot is authorized.

## Series dependency
Paper 0 defines the position. Paper 1 tests strict automatic calculator assistance. Paper 2 adds heterogeneous engines and persistent micro-state. Paper 3 learns semantic interrupts. Paper 4 adds transactional/backtracking state. Paper 5 studies structured/PRA/KV interfaces. Paper 6 studies co-adaptation. Paper 7 integrates validated mechanisms.

## Deliverables
- Tests and implementation for empirical papers.
- Reproducible task/dataset generators.
- Experiment manifests/configuration.
- Machine-readable result tables.
- Updated `paper{n}.tex` including negative findings and limitations.
- Reproduction commands.
