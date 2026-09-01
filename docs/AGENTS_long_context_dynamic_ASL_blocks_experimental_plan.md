# AGENTS — Long-Context Dynamic ASL Blocks Experimental Plan

## Status

Future experiment. Execute only after the ASL/semantic representation has converged and short-context NL→ASL compilation is reasonably stable. Do not block current Paper 1/2 work.

Prerequisites:
- frozen ASL/successor representation;
- deterministic parser/lowering/CogCop runtime;
- stable result/blackboard representation;
- short-context evidence that the representation is useful;
- frozen, provenance-separated Codex and OpenRouter training corpora;
- frozen long-content benchmark protocol.

Central question:

> In long mostly-natural-language content, can the system sparsely detect formalizable spans, temporarily enter an ASL-specialized neural mode, generate ASL without future leakage, execute it, inject appropriate derived state, and continue normal processing?

## 1. Do not compile the whole document

Most text should remain NL. Only selected spans become formal cognitive blocks.

Factor the problem:

1. BLOCK DETECTION — START/END and whether formalization is useful.
2. CONDITIONAL NEURAL MODE — when ASL LoRA/modules are active.
3. CAUSAL VISIBILITY — what source/state ASL may attend to.
4. EXECUTION — deterministic CogCop processing.
5. REINJECTION — what result/state later processing may see.
6. CONTINUATION — how source/generation resumes after intervention.

Keep these separately measurable.

## 2. Typed streams

Maintain an immutable source stream `D=d1...dn`.

For selected block `Bi=[si,ei]` create:
- `Ai`: generated ASL;
- `Ri`: CogCop/blackboard result.

Treat source and cognitive events as conceptually separate:

```text
D1 ─ D2 ─ [B1] ─ D3 ─ [B2] ─ D4
            │             │
            ▼             ▼
           A1            A2
            │             │
            ▼             ▼
           R1            R2
```

Possible segment types:

```text
SOURCE
SELECTED_BLOCK
ASL
COG_RESULT
BLACKBOARD
QUERY
ANSWER
```

Long term, visibility should be a causal graph over typed segments, not merely token index.

## 3. Experimental ladder

Run in order:

```text
L0 oracle blocks + post-hoc ASL
L1 learned block detection
L2 oracle dynamic LoRA switching
L3 learned dynamic LoRA switching
L4 block-causal ASL attention
L5 one result injection
L6 multiple result injections
L7 segmented online source processing
L8 separate cognitive memory
L9 PRA-backed cognitive memory
```

Do not jump directly to the full architecture.

## 4. Long-content benchmark

Construct long documents containing:
- 2–20 candidate formalizable spans;
- ordinary prose between them;
- independent and dependent spans;
- irrelevant numbers/dates;
- non-formalizable numeric statements;
- formalizable non-numeric relations;
- multiple scopes;
- queries needing zero, one, or several blocks;
- early results relevant much later.

Do not use only concatenated GSM8K questions. Include report/manual/exercise-book/conversation/financial-narrative styles.

Preserve exact source offsets and scope IDs.

## 5. Scope

Support at least:
- document;
- section;
- exercise/problem;
- subproblem.

Test:
- within-scope reuse;
- parent/subscope visibility;
- sibling isolation;
- no cross-exercise leakage.

If scope is available from document structure, runtime supplies it; do not make the model regenerate it.

## 6. Block detection

Start with labels:

```text
NONE
FORMALIZE
QUERY
```

Later optionally add:

```text
STATE
COMPUTE
RETRIEVE
VERIFY
HELP
```

Metrics:
- START P/R/F1;
- END P/R/F1;
- exact span match;
- span IoU;
- token inside/outside F1;
- false blocks / 1k source tokens;
- missed critical spans;
- unnecessary formalization rate.

Compare:
- sentence classifier;
- clause classifier;
- token START/INSIDE/END detector;
- lexical/CPU heuristic;
- small classifier;
- hidden-state controller;
- oracle.

Prefer the simplest granularity that works.

## 7. Dynamic adapter modes

Initial modes:

```text
NORMAL
FORMAL
```

Compare:

```text
A0 base model
A1 static ASL LoRA for entire request
A2 oracle-switched ASL LoRA
A3 explicit control-token switching
A4 latent START/CONTINUE/END controller
```

Measure:
- ordinary-language quality/interference;
- ASL quality;
- boundary quality;
- adapter-switch overhead;
- latency.

The detector should not need to memorize the concrete CogCop registry.

## 8. Adapter placement / KV cache

Dynamic Q/K/V LoRA changes the representation used to create cached K/V. Explicitly compare:

```text
C1 conditional QKVO
C2 conditional MLP-only
C3 shared always-on QKVO + conditional MLP
```

Record which mode created each cached segment.

Do not assume mixed base/LoRA K/V is harmless.

MLP-only conditional adaptation is an important condition because attention K/V remains in a common space.

## 9. Whole-document post-hoc architecture

First architecture:

1. encode whole source once;
2. detect blocks;
3. generate ASL for each selected block;
4. execute;
5. use derived results at final query/answer.

This is efficient but ASL appended after the document would normally see future source. Prevent that with block-causal masking.

## 10. Block-causal ASL mask

For `Bi=[si,ei]`:

```text
Ai may see:
  source D[1:ei]
  allowed prior results R<i
  its own previous ASL tokens

Ai may NOT see:
  source D[ei+1:n]
  future ASL/results
```

Formally:

```text
Visible(Ai,t) = D<=ei ∪ allowed(R<i) ∪ Ai,<t
```

When ASL is physically appended after the full source, this creates a masked rectangle over future-source K/V in the ASL rows.

Add leakage tests proving future source cannot affect ASL logits.

## 11. ASL visibility ablation

Compare:

```text
M0 full source (leaky diagnostic only)
M1 causal source prefix through block end
M2 selected block only
M3 block + bounded preceding window
M4 block + selected prior source/state
```

M4 later becomes PRA-enabled.

This determines how much earlier linguistic context ASL compilation actually needs.

## 12. Result reinjection

Do not assume raw ASL should enter normal context.

Compare:

```text
R0 NONE
R1 RESULT_ONLY
R2 SEMANTIC_STATE
R3 ASL_PLUS_RESULT
R4 REFERENCE_ONLY
```

`REFERENCE_ONLY` stores a typed handle that can be materialized later.

Measure:
- downstream accuracy;
- context tokens;
- interference;
- state reuse;
- result override;
- hallucination.

Hypothesis: semantic/result state is more useful than raw ASL trace for later NL processing.

## 13. One-injection experiment

Before multiple blocks, use documents with exactly one formalizable block.

Compare:
- no intervention;
- post-hoc ASL/result used only at final answer;
- segmented processing with result injected before later source.

This isolates reinjection effects.

## 14. Multiple-injection experiment

Then test 2, 4, 8, 16 blocks/document.

Measure:
- block detection;
- ASL semantic accuracy by block index;
- cumulative state correctness;
- error propagation;
- final answer;
- model calls/tokens;
- CogCop calls;
- injected tokens;
- latency;
- KV growth.

Include documents where most earlier results are irrelevant to the final query.

## 15. Segmented online processing

Second architecture:

```text
encode source
→ detect block end
→ pause source
→ enable FORMAL adapter
→ generate ASL
→ execute
→ inject selected result/state
→ disable FORMAL adapter
→ resume source
```

Logical order:

```text
D1 B1 -> A1 -> R1 -> D2 B2 -> A2 -> R2 -> D3 ...
```

This allows earlier cognitive results to affect later interpretation.

## 16. Resumed-source visibility

Compare:

```text
S0 SOURCE_PURE
later source sees no ASL/result

S1 RESULT_VISIBLE
later source sees Ri, not Ai

S2 ASL_RESULT_VISIBLE
later source sees Ai and Ri

S3 SEPARATE_COGNITIVE_CHANNEL
normal source SA + separate cognitive-state access
```

S1/S3 are preferred hypotheses. Do not assume raw ASL should contaminate ordinary source encoding.

## 17. Block-sparse causal DAG

Ordinary triangular causality becomes insufficient.

Example physical sequence:

```text
D1 D2 A1 R1 D3
```

Desired policy may be:

```text
D3 sees D1,D2,R1,D3
D3 does not see A1
```

Define:

```text
M(i,j) = Visible(segment_j, segment_i)
```

not merely `j <= i`.

Example policy:

| Query segment | Visible history |
|---|---|
| SOURCE | prior SOURCE + selected RESULT |
| ASL | source through triggering block + selected prior RESULT + own ASL |
| RESULT | triggering ASL + required state |
| ANSWER | source + selected RESULT + optional references |

Implement as a typed block-sparse causal DAG rather than ad-hoc masks.

## 18. Source encoding strategy

Compare only after simpler stages stabilize.

### E0 Whole-source once
Pros: efficient prefill/global detection.
Cons: later source representations cannot incorporate earlier derived state; requires custom ASL future masking.

### E1 Segmented online
Pros: true causal intervention.
Cons: interruptions/cache complexity.

### E2 Whole source + separate cognitive stream
Source stays immutable; ASL/results live separately; later ASL/answer attends to both.

E2 is an important long-term compromise.

## 19. Separate cognitive memory

Late-stage architecture:

```text
source/self KV
     ↑
     SA
     ↑
text hidden
     ↓
cognitive attention
     ↓
blackboard/CogCop memory
```

Do not require cognitive records to masquerade as ordinary text tokens.

Measure source-quality preservation, state utilization, memory, latency, and final quality.

## 20. PRA integration

Only after non-PRA conditions work.

PRA may store/select:
- source spans;
- semantic blocks;
- ASL;
- CogCop results;
- blackboard records;
- provenance.

Compare:
- full prior cognitive state;
- PRA top-k;
- lexical/hybrid selection;
- oracle relevant state.

Do not confound initial mask/switching results with PRA routing quality.

## 21. End-to-end factorized metrics

Report:

```text
DETECT_START
DETECT_END
FORMALIZE_TYPE
ASL_PARSE
ASL_SEMANTIC
LOWER
EXECUTE
INJECT
USE
FINAL
```

Headline learned conditions must include detector errors; oracle-block ASL accuracy is only a component ceiling.

## 22. False positives / negatives

False blocks are costly in long documents.

Track:
- false blocks / 1k tokens;
- unnecessary ASL tokens;
- unnecessary CogCop calls;
- corrupted blackboard records;
- downstream degradation.

Hard negatives:
- descriptive numbers;
- narrative dates;
- hypothetical calculations;
- quoted equations;
- opinions/comparatives;
- irrelevant examples.

For false negatives distinguish critical vs non-critical missed spans.

## 23. Boundary sensitivity

Measure ASL quality versus boundary IoU.

Test:
- missing subject/entity;
- missing trailing coreference;
- extra commentary;
- merged unrelated clauses.

This determines whether exact token boundaries are necessary or clause-level boundaries suffice.

## 24. Dependency structures

Construct:

```text
independent: B1   B2
chain:       B1 -> B2 -> query
fork:        B1 -> B2
                -> B3
join:        B1 -> B3 -> query
             B2 -> B3
```

Measure whether injection/selection preserves the required graph.

## 25. Scope + dependency

Exercise-book benchmark:

```text
Exercise 1
 a) derive X
 b) uses X

Exercise 2
 X has unrelated meaning
```

Test:
- reuse inside Exercise 1;
- no state leakage into Exercise 2;
- correct scope reset;
- subproblem inheritance.

## 26. Multiple-intervention failure taxonomy

Classify:
- missed block;
- false block;
- early START;
- late START;
- early END;
- late END;
- wrong ASL;
- correct ASL/wrong execution;
- wrong result injection;
- stale result;
- wrong-scope result;
- result not used;
- raw-ASL interference;
- state propagation error.

## 27. Controls

Every major experiment should include matched:

```text
base
static LoRA
oracle block + static/conditional LoRA
learned block + conditional LoRA
no result injection
oracle result injection
```

Do not compare an oracle-block condition directly against an end-to-end learned detector without labeling the distinction.

## 28. Efficiency

Track:
- source prefill tokens;
- ASL generated tokens;
- result-injection tokens;
- model forward passes;
- adapter switches;
- source KV bytes;
- ASL KV bytes;
- cognitive-memory bytes;
- CPU CogCop time;
- accelerator time;
- total wall time.

Report accuracy/cost tradeoffs.

## 29. Primary research questions

H1 Sparse formalization:
Only a minority of long-document spans need ASL/CogCop intervention.

H2 Conditional adapters:
Dynamic ASL LoRA preserves normal-language behavior better than static LoRA while retaining formal-span quality.

H3 Block-causal masking:
ASL can be generated causally without access to source after the selected block.

H4 Result-vs-trace:
Injecting compact semantic result/state is better than injecting raw ASL traces.

H5 Multiple interventions:
Repeated formalization can compose without catastrophic state/error accumulation.

H6 Segmented vs post-hoc:
Online segmented encoding helps when later source interpretation genuinely depends on earlier derived state.

H7 Separate cognitive memory:
A typed cognitive channel can reduce context pollution versus flattening ASL/results into ordinary tokens.

H8 PRA:
Selective materialization can bound cognitive-state cost as block count grows.

## 30. Go/no-go gates

### Block detection GO
Low false activation and adequate critical-block recall on long mixed documents.

### Dynamic LoRA GO
Oracle-switched adapter matches static-LoRA ASL quality while reducing ordinary-language interference.

### Learned switching GO
Learned boundaries approach oracle-switched quality with acceptable FAR.

### Reinjection GO
Injected result/state improves downstream dependent tasks without unacceptable unrelated-text degradation.

### Multi-injection GO
Accuracy remains stable enough as blocks increase and state reuse provides measurable benefit.

### Separate-memory GO
Only if flat/block-mask approaches show context pollution or scaling problems.

### PRA GO
Only after cognitive-state relevance selection becomes a measured bottleneck.

## 31. Recommended first milestone

Once ASL is frozen:

1. Build long mixed documents with oracle block spans.
2. Use whole-document source encoding.
3. Generate ASL post-hoc for each oracle block.
4. Implement block-causal future-source masking.
5. Compare static LoRA vs oracle-switched LoRA.
6. No reinjection into source yet.
7. Execute all ASL and use results only at final query.
8. Verify no future leakage.
9. Measure ASL quality, ordinary-text interference, and cost.

This isolates dynamic adapter switching + attention masking before the difficult multi-injection problem.

## 32. Second milestone

Add learned block detection:
- sentence;
- clause;
- token/span controller.

Keep oracle-block ceiling.

Do not add result injection until detection quality is understood.

## 33. Third milestone

Add one result injection and segmented continuation.

Compare SOURCE_PURE vs RESULT_VISIBLE.

Only then scale to multiple injections.

## 34. Fourth milestone

Scale to 2/4/8/16 interventions and introduce typed block-sparse causal DAG masks.

Add PRA only after measuring state-selection pressure.

## 35. Claim boundary

Do not call oracle boundaries learned cognition.

Do not claim long-context cognitive execution from post-hoc ASL alone.

Do not claim speedup without measured end-to-end timing.

Do not claim dynamic LoRA novelty merely because adapters are switched; compare against existing dynamic/token-routed adapter concepts in related work.

The stronger research contribution, if supported, is the combination of:
- sparse semantic block detection;
- stateful START/END cognitive modes;
- conditional adapters;
- block-causal visibility;
- external CogCop execution;
- repeated typed result injection;
- segmented causal continuation.

## 36. Deliverables

When this project begins, create:
- benchmark generator + frozen manifests;
- oracle block annotations;
- block detector baselines;
- conditional adapter controller;
- block-causal attention-mask implementation;
- leakage tests;
- typed segment/context graph;
- result-injection policies;
- single- and multi-injection experiments;
- cache/memory/economics reports;
- failure taxonomy;
- OpenRouter corpus validation, audit, and training comparison;
- manuscript draft/addon;
- explicit gates for each architectural stage.

Preserve all short-context ASL evidence independently.

## 37. OpenRouter corpus training track

Include the OpenRouter-generated GSM8K and TAT-QA programs as a separately
provenanced training source. Do not silently merge them with the Codex corpus and
do not describe API-accepted programs as semantic gold.

Before training, require each remote program to pass:

```text
JSON/schema
-> ASL parse
-> deterministic lowering
-> type validation
-> execution
-> benchmark-answer verification
-> semantic lint
-> deduplication and leakage audit
```

Answer verification is a screening signal, not proof that the intermediate
program is semantically faithful. Manually audit a stratified sample by dataset,
semantic-pattern family, program length, model route, and validation outcome.
Record source ID, teacher model, attempt number, prompt/skill hashes, and all
validator results for every retained program.

Freeze splits by source ID and normalized semantic-pattern family. Exclude all
short-context and long-context evaluation IDs, near duplicates, and held-out
semantic-pattern families before creating training data. Do not perform a random
row split. Preserve the existing Codex 500 split and test set unchanged.

Run these matched conditions before selecting the formal adapter used in the
long-context experiments:

```text
T0 CODEX_500
   Existing execution-verified Codex corpus.

T1 OPENROUTER_MATCHED_500
   Quality-filtered remote rows, matched to T0 by dataset and semantic pattern.

T2 OPENROUTER_ALL
   All quality-filtered remote rows available at corpus freeze.

T3 UNION
   Codex 500 plus quality-filtered OpenRouter rows.

T4 REMOTE_THEN_CODEX
   Train on remote rows, then finish on the higher-confidence Codex corpus.

T5 QUALITY_WEIGHTED_UNION
   Weight or sample by provenance, answer verification, audit status, and pattern
   rarity without changing the frozen evaluation protocol.
```

For T2-T5, include learning curves at approximately:

```text
500 / 1,000 / 2,000 / all retained rows
```

Use the same base model, LoRA placement/rank, optimizer schedule, decoding, and
frozen test programs for the primary corpus comparison. Add a token- or
optimizer-step-matched control when the larger corpus receives more training.
Report both raw row count and unique normalized semantic-pattern count.

Primary short-context selection metrics remain:

```text
parse and lower validity
type validity
execution
source-fact grounding
entity/path binding
operator and dependency correctness
semantic-state equivalence
final-answer execution
```

Select the long-context FORMAL adapter using frozen development criteria, not
the final long-context test. Keep corpus provenance as an experimental factor in
at least one oracle-block long-context comparison:

```text
best Codex-only adapter
vs
best OpenRouter-inclusive adapter
```

## 38. OpenRouter integration boundaries

OpenRouter ASL programs train the formal-span compiler. They do not by themselves
train block detection, adapter switching, reinjection policy, or state selection.
Those components require separately frozen long-document labels and controls.

When constructing synthetic long documents from public examples:

- partition source records before composing documents;
- never place a training source or paraphrase in a held-out document;
- keep benchmark answers and rationales hidden from teacher prompts;
- never expose future source, executed state, or blackboard values to ASL generation;
- use one fixed prompt and fixed ICL set within each declared condition;
- preserve source offsets, source IDs, scope IDs, and teacher provenance;
- prevent repeated lexical variants of one semantic family from dominating training.

Do not begin dynamic long-context training while the OpenRouter queue is still
changing. Freeze a named corpus checkpoint with manifests and hashes first.
Later OpenRouter additions become a new corpus version and require a separately
reported training run.
