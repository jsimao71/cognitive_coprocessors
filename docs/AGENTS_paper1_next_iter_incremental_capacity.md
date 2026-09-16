# AGENTS — Paper 1 Next Iteration
## Incremental Executed-State Compiler, Error Propagation, and Adapter Capacity

## Purpose

Continue Paper 1 from the frozen 500-original ASL checkpoint.

Current established checkpoint:
- Qwen3-0.6B
- ASL-Core / ASL-Arith
- 450 train / 25 dev / 25 test
- 397 unique normalized train semantic signatures
- zero dev/test semantic-pattern leakage
- same frozen 25-program test set used for prior whole-program comparisons
- rank-8 QKVO LoRA, 2,293,760 trainable parameters (0.383% of model)
- LoRA-500 whole-program result: 11/25 final answers (44%)
- TAT-QA: 7/8
- GSM8K: 4/17
- whole-program parse validity: 88%
- executable programs: 76%
- incremental corpus already built:
  - 1,403 train transitions
  - 65 dev transitions
- closed-loop predicted-state evaluation already implemented
- current incremental XPU training should complete before changing the protocol

Primary question:

> Does converting whole-problem NL→ASL into repeated local `NL_part + executed workspace -> ASL delta` transitions materially reduce the neural burden on a 0.6B model?

Only after answering that should the iteration investigate whether the remaining limitation is:
1. local semantic grounding,
2. state-error propagation,
3. adapter capacity/location,
4. data diversity,
5. base-model size.

Do not scale all axes at once.

## 1. Preserve the frozen evidence chain

Do not modify:
- existing 500-original split;
- frozen 25-test programs;
- semantic-pattern exclusion rules;
- previous whole-program artifacts;
- LoRA-25/50/100/500 results;
- LoRA-B augmentation checkpoint.

Every new condition must remain linked to the same source IDs, semantic-pattern IDs, model revision, and evaluation definitions where applicable.

If a bug requires rescoring, rescore saved predictions before regeneration whenever possible.

## 2. Finish the currently running incremental LoRA

Complete the current incremental rank-8 QKVO training before any architecture change.

Record:
- model revision;
- adapter config;
- trainable parameter count;
- optimizer steps;
- train transitions;
- target tokens;
- wall time;
- peak XPU memory;
- final train/dev loss;
- adapter hashes.

Report both:
- 1,403 transition rows;
- number of unique source programs;
- number of unique semantic signatures.

## 3. Primary closed-loop evaluation

Run predicted-state closed loop on the same frozen 25 test programs.

Per part:

```text
current executed/predicted workspace
+ next NL part
        ↓
Qwen + incremental LoRA
        ↓
ASL delta
        ↓
parse / lower / type validate
        ↓
if valid: execute
        ↓
new workspace
        ↓
next part
```

Strict policy:
- reject invalid ASL deltas;
- fail closed;
- stop at first syntactic/lowering/type-invalid transition;
- no silent repair;
- no semantic oracle correction.

This is the primary deployed-system condition.

## 4. Report program-level and transition-level metrics

Program-level:
- parse-valid complete program;
- lowerable complete program;
- type-valid complete program;
- executable complete program;
- final-answer correctness;
- semantic-state equivalence;
- semantic-return equivalence;
- dependency correctness.

Transition-level:
- attempted transitions;
- accepted transitions;
- parse-valid deltas;
- lowerable deltas;
- type-valid deltas;
- executable deltas;
- correct operator;
- correct semantic path/entity;
- correct source fact;
- correct references/dependencies;
- correct state delta.

Also report:
- accepted transitions / attempted transitions;
- completed programs / total programs;
- mean completed fraction before stop.

## 5. Add oracle-state incremental evaluation

After predicted-state closed loop is frozen, add:

### Oracle-state incremental

For each test transition, provide the gold executed `state_before` derived from the verified semantic program, independent of previous student mistakes.

The model still predicts the current ASL delta.

This isolates:

`P(correct ASL delta | correct current state, current NL clause)`

from state-error propagation.

Compare:

```text
incremental_oracle_state
vs
incremental_predicted_state
```

Primary metric:

```text
state_error_propagation_gap =
oracle_state_final_accuracy
-
predicted_state_final_accuracy
```

Interpretation:
- small gap -> local semantic compilation is dominant bottleneck;
- large gap -> early mistakes poison later state;
- high oracle-state + low predicted-state -> recovery/state correction deserves future work.

Do not use oracle state for the headline deployed condition.

## 6. Add symbolic-state vs executed-state ablation

If cheap, compare:

### Symbolic prior-state
Provide prior accepted ASL relations/expressions without resolved runtime values.

### Executed-state
Provide runtime-grounded values plus unresolved relations.

Question:

> Does exact CogCop feedback help beyond merely chopping the program?

Compare:
- local ASL correctness;
- reference accuracy;
- final answers;
- token count;
- execution rate.

If implementation cost is high, defer until after oracle-state analysis.

## 7. Error taxonomy

Generate per-transition failure categories:

### Surface/syntax
- no ASL extracted;
- malformed syntax;
- invalid operator token.

### Lowering/type
- unsupported AST form;
- type mismatch;
- invalid arithmetic on Boolean/non-numeric value.

### Semantic grounding
- wrong entity;
- wrong field/path;
- wrong temporal qualifier;
- wrong quantity;
- wrong operator;
- wrong relation direction.

### Reference/dependency
- missing reference;
- wrong dependency;
- premature literal collapse;
- stale/wrong scope path.

### Query/return
- wrong return target;
- premature return;
- missing return.

### State propagation
- current transition would be correct under gold prior state but fails under predicted prior state;
- correct local parse but uses poisoned prior state.

Produce counts and representative examples for GSM8K and TAT-QA separately.

## 8. Diagnose GSM8K/TAT-QA asymmetry

Current whole-program LoRA-500:
- TAT-QA 7/8
- GSM8K 4/17

For incremental, report the same breakdown.

Inspect whether GSM8K errors are disproportionately:
- entity/path grounding;
- relation direction;
- temporal relations;
- compositional dependencies;
- question-to-return mapping.

Inspect whether TAT-QA benefits from:
- explicit numeric/table structure;
- regular semantic schema;
- shorter dependency chains;
- lexical regularity.

If possible compute by dataset:
- mean parts/program;
- mean ASL statements/program;
- mean dependency depth;
- mean unique entities;
- mean semantic paths;
- mean target tokens.

## 9. Decision gates for incremental result

Use whole-program LoRA-500 = 44% as reference.

Approximate interpretation:

### <=45%
Little/no architectural gain.
Focus next on local semantic compilation/data/adapter.

### 50–60%
Moderate gain.
State decomposition helps, but local grounding still dominates.

### 60–75%
Strong architectural evidence.
External executed state materially reduces model burden.

### >75%
Very strong pilot evidence.

These are decision gates, not significance thresholds.

Report Wilson intervals and paired per-example gains/losses.

## 10. Paired analysis

Compute:
- whole wrong -> incremental correct;
- whole correct -> incremental wrong;
- both correct;
- both wrong.

Inspect changed cases.

A McNemar/binomial paired test may be exploratory due to n=25.

Do not overclaim significance.

## 11. Context-loss control

Incremental compilation can lose useful global information.

Compare:

### Causal-only
Current clause + prior state + prior/available context.

### Full-question context
Current clause + prior state + full original question, but still predict only the local delta.

Do not leak gold rationale or future gold state.

If full-question context materially improves local semantics, keep it as a runtime option.

## 12. Do not add retry/recovery yet to the primary result

Keep strict closed-loop clean.

Do NOT initially add:
- auto-repair;
- second model;
- semantic correction;
- oracle retry;
- rule-based patching.

After freezing strict results, optional recovery can test:

```text
invalid delta
→ validator error
→ one retry
```

as a separate condition.

## 13. Choose next axis from diagnostics

### If oracle-state incremental is high but predicted-state lower
Prioritize:
- recovery;
- state consistency;
- confidence/provenance;
- retry;
- verifier;
- conflict detection.

Do not first increase LoRA rank.

### If both oracle and predicted incremental remain weak
Prioritize local compiler:
- more semantic originals;
- MLP adaptation;
- rank;
- larger base model.

### If syntax/type validity dominates
Improve representation/prompt/canonicalization before scaling model.

### If paths/entities dominate
Increase semantic naming/grounding diversity and consider auxiliary objectives.

## 14. Adapter-capacity ablation

Only after incremental diagnosis.

Current:
- all transformer layers;
- Q/K/V/O only;
- rank 8;
- alpha 16;
- dropout .05;
- 2,293,760 trainable params.

Recommended minimal matrix:

### A — Current
`QKVO r8`
~2.29M trainable params.

### B — More rank
`QKVO r16`
~4.59M.

### C — More locations
`QKVO + gate_proj + up_proj + down_proj, r8`
~5.05M.

Interpret:
- B > A, C ~ B -> low-rank capacity bottleneck;
- C > B/A -> MLP semantic transformation matters;
- B ~ C ~ A -> data/architecture/base model dominates.

Keep split/training schedule otherwise matched.

## 15. Optional layer-depth ablation

Only if capacity experiments justify it.

Compare:
- all layers;
- middle+late;
- late only.

Prefer parameter-matched runs.

Optional probes:
- operator class;
- entity/path;
- ASL mode;
- return/query intent.

Use probes only to guide placement.

## 16. Data scaling after 500

Do not jump directly to tens of thousands.

If learning remains data-limited:
- next gate: 1,000 originals;
- then 2,000.

For every expansion:
- exclude frozen dev/test IDs;
- exclude test semantic-pattern families;
- preserve provenance/repair;
- track unique semantic signatures;
- diversify relation structures deliberately.

Track both:
- row count;
- unique semantic-signature count.

## 17. Semantic coverage analysis

Maintain structural tags:
- relative quantity;
- ratio/fraction;
- percentage;
- aggregation;
- rates;
- equal allocation;
- temporal shift;
- before/after;
- remaining/difference;
- max/min/comparison;
- multi-entity;
- chained relation;
- multi-step dependency.

Use tags for coverage, not as perfect gold classes.

## 18. Base-model scaling only after diagnosis

Do not jump to 8B yet.

After 0.6B incremental/capacity analysis, repeat the best protocol on:
- Qwen3-1.7B if feasible;
- SmolLM2-1.7B;
- later Qwen3-4B.

Goal:

> Measure how much base-model size is still needed once semantic compilation uses an executed workspace.

Eventually compare native model size curve vs ASL+CogCop model size curve.

## 19. ICL hybrid after LoRA-500 / incremental

Because LoRA-100 + 3-shot ICL was complementary, test after primary incremental result:

- LoRA-500 + 0-shot;
- LoRA-500 + 1-shot;
- LoRA-500 + 3-shot;
- best incremental LoRA + 0/1/3-shot.

Use leakage-safe demonstrations.

Measure:
- accuracy;
- executable rate;
- prompt tokens;
- wall time.

Question:

> Does LoRA internalize stable ASL while tiny context supplies local specialization?

## 20. Robustness

For best whole/incremental adapter, evaluate:
- untouched original;
- numeric remapping;
- very large numbers;
- entity renaming;
- field renaming where safe;
- paraphrase;
- safe distractors.

Report robustness gaps:

```text
gap_numeric = original - numeric_remap
gap_large = original - large_number
gap_paraphrase = original - paraphrase
```

## 21. State representation ablation

If incremental is promising, compare:

### Current JSON-like runtime state

### Compact ASL state

```text
claire.age_in_2y = 20
claire.age_now = 18
jessica.age_now = 24
```

### Values + unresolved derivations

```text
VALUES
claire.age_in_2y = 20
claire.age_now = 18

UNRESOLVED
jessica.age_now = claire.age_now + 6
```

Measure prompt tokens and accuracy.

Prefer the shortest state form that preserves semantics.

## 22. Unresolved-expression runtime semantics

Confirm/document support for:

```text
jessica.age_now = claire.age_now + 6
```

before `claire.age_now` is grounded.

Runtime should:
- store unresolved dependency;
- record dependency edge;
- reevaluate once references ground;
- preserve symbolic derivation/provenance.

Add tests for:
- single unresolved dependency;
- multi-hop dependency;
- later grounding;
- invalid arithmetic cycles.

## 23. Scope

Continue using externally supplied benchmark-case scope.

Do not add explicit `SCOPE/END` to GSM8K/TAT-QA unless source has real nested subproblems.

Each transition keeps its parent root scope.

No cross-program state leakage.

## 24. Manuscript update

After incremental result, add:
1. 500-original checkpoint;
2. unique semantic-signature count;
3. whole-program vs incremental architecture;
4. transition corpus size;
5. predicted-state closed-loop protocol;
6. oracle-state diagnostic;
7. transition/program metrics;
8. GSM8K/TAT-QA breakdown;
9. state-error propagation analysis;
10. paired outcomes;
11. strict claim boundary.

If positive, frame as:

> Local semantic compilation plus externally executed state reduces the burden on the small Transformer.

Do NOT claim:
- general reasoning solved;
- universal ASL compiler;
- robust general cognitive architecture;
- 0.6B matches large models.

## 25. Key hypotheses

H1 — Semantic data scaling:
Genuinely different semantic programs improve NL→ASL more than surface perturbation alone.

H2 — Incremental-state advantage:
`NL_part + executed state -> ASL delta` should outperform whole-program synthesis.

H3 — Executed-state advantage:
CogCop feedback should outperform purely symbolic prior-state continuation.

H4 — State error propagation:
Oracle-state incremental should exceed predicted-state if early mistakes poison downstream state.

H5 — Adapter location:
If local mapping remains weak, MLP adaptation may help more than simply increasing attention rank.

H6 — Numeric invariance:
Large-number perturbation should affect ASL compilation less than direct neural-answer accuracy.

## 26. Immediate execution order

P0. Finish current incremental r8-QKVO training.

P1. Run strict predicted-state closed-loop on frozen 25 programs.

P2. Freeze artifacts and summarize transition/program metrics.

P3. Add/run oracle-state incremental evaluation.

P4. Paired whole-program vs incremental analysis.

P5. Diagnose GSM8K vs TAT-QA errors.

P6. If justified, symbolic-state vs executed-state ablation.

P7. Choose state-propagation vs local-compiler bottleneck.

P8. If local compiler bottleneck:
- QKVO-r16;
- QKVO+MLP-r8.

P9. If state-propagation bottleneck:
- design one-retry validator-feedback experiment.

P10. Test compact ASL state representation if state prompt cost matters.

P11. Robustness on best selected condition.

P12. Update manuscript/PDF/artifacts/tests.

P13. Only then decide on 1,000+ originals or larger base models.

## 27. Verification

Before every push:
- full CPU tests;
- XPU-focused tests where relevant;
- Ruff;
- deterministic hashes;
- no credentials;
- PDF rebuild;
- no raw benchmark data leakage into tracked artifacts;
- confirm branch/remote sync;
- preserve unrelated user files and untracked artifacts.
