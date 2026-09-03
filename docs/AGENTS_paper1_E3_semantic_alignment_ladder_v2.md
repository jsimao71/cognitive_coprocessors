# AGENTS - Paper 1 E3 Semantic Alignment Ladder v2

## Status and authority

This is the canonical execution order for the Paper 1 E3 experiments. It
refines `AGENTS_paper1_E3_semantic_alignment_ladder.md` by separating seven
questions that must not be confounded:

1. Does modality-specific adaptation help?
2. Can the ASL channel represent and canonicalize a symbolic world by itself?
3. Can paired NL and ASL converge on shared world content?
4. Does a grounded-to-NL-only curriculum improve autonomous internal ASL?
5. Does executing and reinjecting predicted internal ASL improve continuation?
6. Does a factorized output representation help independently of its loss and
   checkpoint-selection policy?
7. Does a larger execution-verified remote-teacher corpus help independently
   of representation and objective?

The design is developmentally inspired. Do not claim that it models human
child cognition or physical grounding. External ASL is a privileged symbolic
model of observable world state.

## Frozen references

Keep the existing Qwen3-0.6B revision, 450/25/25 identities, generation seed,
prompt, target representation, optimizer budget, and autonomous evaluator.

| ID | Condition | Final answer | Role |
|---|---|---:|---|
| R0 | Q0 plain F0 QKVO-r8 | 11/25 | Primary autonomous control |
| R1 | Q1 serialized mixed context | 6/25 | Extra-text control |
| R2 | Q3 native merged K/V, shared LoRA | 10/25 | Shared-adapter memory control |
| S1 | Q3S1 shared plus ASL capture delta | 8/25 | Extra capture-capacity control |

Q3S1 does not justify replication. It has 4.589M trainable parameters, 16/25
executable programs, 5/25 semantic returns, and 8/25 correct answers.

## Representation contract

Treat external and internal ASL as different roles even when their current
canonical strings are identical.

```yaml
source_representation: asl_external_v1
target_representation: asl_internal_canonical_v1
mapping: identity | denoise | lower | abstract | update
```

Use three state variables:

```text
W_ext(t)  externally supplied symbolic world observation
U(t)      natural-language utterance or document span
W_int(t)  internal canonical ASL workspace

W_int(t) = Update(W_int(t-1), W_ext(t), Compile(U(t)))
```

Shared semantic alignment applies only to facts represented in both views.
NL-only additions must retain source, time, modality, and confidence rather
than being forced to equal the immediate external world.

```text
Z_NL_overlap ~= Z_ASL_world
Z_NL_additional remains provenance-distinct
```

Required provenance classes are `observed`, `asserted`, `inferred`, and
`hypothetical`. Required temporal distinctions are current, prior, planned,
and conditional state.

## Target architecture

Reuse one frozen pretrained Qwen backbone with adapter banks:

```text
ASL_ext -> LoRA_A lower --\
                            -> LoRA_S shared upper -> Z_A
NL      -> LoRA_N lower --/                       -> Z_N

NL-derived Z_N -> native merged K/V -> causal decoder -> ASL_int
ASL_int -> parser/runtime -> canonical state/result -> continuation
```

External gold ASL may supervise representation learning. It must not be the
memory used by the primary autonomous generation path. Main generation always
uses NL-derived memory so train and test have the same information boundary.

Start without a separate decoder adapter. Add `LoRA_D` only as a named capacity
ablation after a positive semantic signal.

## Experiment ladder

### Phase 0 - Architecture controls

Run in the current order:

```text
S2: separate full-depth NL and ASL LoRAs
S3: modality-specific lower LoRAs plus shared upper LoRA
```

Interpretation:

- S2 above R2 suggests that surface-language interference matters.
- S3 above S2 suggests that a shared upper transformation helps.
- Neither result alone demonstrates representational convergence.

S3 is the adapter topology used by later E3 cells even if it does not beat R0,
unless routing diagnostics show that it is broken.

### Phase 1 - ASL-channel sufficiency

Build ASL-only examples with no NL present.

```text
C0 identity:
ASL_ext canonical -> ASL_int canonical

C1 denoising:
ASL_ext corrupted -> ASL_int canonical
```

Use matched optimizer steps for C0 and C1. Initial C1 corruption policies are:

```text
argument_mask
value_mask
record_dropout
```

Identity reconstruction is a syntax/copy control. Denoising is the primary
ASL-channel condition because it requires structural recovery. Report exact
text, parse, lower, type, execution, semantic state, dependency, and corruption
recovery by policy and severity.

Do not infer semantic convergence from high identity-copy accuracy.

### Phase 2 - Matched NL-memory generation baseline

Train E3a with the S3 hybrid topology and random adapter initialization:

```text
NL -> LoRA_N + LoRA_S -> Z_N
Z_N -> native merged K/V -> causal ASL_int generation
```

Do not inject external gold ASL into the decoder memory. This is the clean
generation baseline for every preconditioned condition below. It is distinct
from the existing S3 mixed-teacher architecture control.

### Phase 3 - Paired semantic alignment

Train E3b on matched NL/ASL descriptions from the 450 training identities:

```text
Z_N(i) = Encode_N(NL_i)
Z_A(i) = Encode_A(ASL_i)
L_align = symmetric InfoNCE(Z_N(i), Z_A(i))
```

Initialize the ASL path from C1. Use mean pooling over non-padding tokens for
the first run. Do not use raw MSE as the primary objective.

The XPU-safe microbatch is one, so use a deterministic detached negative bank.
Exclude the same source ID and the same normalized semantic-pattern family from
negative candidates. Gradient accumulation is not an in-batch negative method.

The primary E3b run retains the same causal attention mask as E3a. A
bidirectional encoder-mode run is a later named ablation; changing both the
objective and mask in the primary comparison would confound the result.

After preconditioning, transfer the adapters into the exact E3a causal
generation schedule. Compare:

```text
E3a: random hybrid -> NL-memory generation
E3b: C1 + contrastive hybrid -> identical NL-memory generation
```

Measure paired retrieval and layerwise similarity before and after causal
fine-tuning to detect alignment drift.

### Phase 3A - Factorized representation and objective matrix

The specialization and alignment experiments above do not directly address
the dominant measured error: choosing and binding the correct semantic
structure. The earlier `M0`-`M4` ladder mixed a representation change with loss
and checkpoint-selection changes. Treat these as independent axes before
scaling the adapter or base model.

Representation axis:

```text
F0 direct ASL:
NL -> canonical ASL text

F4 canonical semantic IR:
NL -> symbol table + ASL-isomorphic expression graph
   -> deterministic lowering to canonical ASL
```

`F4` is a generated language and a lightweight structural ontology/IR. It is
not a domain ontology and introduces no new facts, operators, rationales, or
answer supervision. It deliberately remains isomorphic to F0 ASL: it factors
path grounding from the operation/dependency graph and makes bindings, source
facts, references, dependencies, and the query target explicit. Its generated
JSON is accepted only when it has unique contiguous slots, exactly one final
query, lowers to ordinary ASL, and passes the existing parse, lower, type,
scope, and execution gates.

Objective and selection axis:

```text
L0 ordinary:
ordinary causal token loss

L1 semantic-weighted:
higher normalized causal-loss weight on explicit semantic decisions

L2 within-example ranking:
L1 + score(gold | NL) > score(executable semantic corruption | NL)

L3 semantic selection:
L2 + development-checkpoint selection by safety and semantic structure,
not token loss alone
```

Preserve the historical aliases in existing paths and reports, but do not use
them as the primary scientific factor names:

| Historical alias | Factorized condition | Meaning |
|---|---|---|
| M0 | F0 x L0 | Frozen direct-ASL Q0 control |
| M0.5 | F0 x L1 | Semantic-weighted direct ASL |
| M0.6 | F0 x L2 | Direct ASL plus within-example ranking |
| M1 | F4 x L0 | Canonical semantic IR with ordinary loss |
| M2 | F4 x L1 | Canonical semantic IR with decision weighting |
| M3 | F4 x L2 | Canonical semantic IR with hard-negative ranking |
| M4 | F4 x L3 | M3 trajectory with semantic checkpoint selection |

New manifests and reports must record `representation_id` and `objective_id`
separately. They may additionally record `historical_alias` for compatibility.
Do not create new bare `M*` configuration names because the 3D architecture
matrix already uses `M1/M2` for attention modes.

Data scale and provenance are a third independent axis:

```text
D0 Codex-500:
the frozen 450/25/25 manually/Codex-curated semantic-program checkpoint

D1 OpenRouter-strict:
the versioned, deduplicated, execution-verified remote-teacher corpus
```

Use D0 for every primary F/L causal comparison. D1 is a later scale and
teacher-quality intervention, not a replacement test set and not semantic
gold. Before constructing any D1 training split, exclude all frozen D0 dev/test
source IDs and all frozen test semantic-pattern families. Keep OpenRouter model,
attempt, validation, recovery, and corpus-version provenance attached.

The September 3 recovery checkpoint contains 6,950 strict unique programs from
8,179 sources; 1,229 sources remain in a provenance-separated retry queue.
Freeze a new D1 version only after the active recovery pass is consolidated.
Train F0 x L0 on D1 first. If that shows a scale signal, train the already
selected objective within F0, then deterministically derive F4 from the same
strict programs and run the matched representation comparison. Do not tune F,
L, and D simultaneously.

The minimum causal comparisons are:

```text
representation effect: F4 x L0 versus F0 x L0
weighting effect:      F0 x L1 versus F0 x L0
ranking effect:        F0 x L2 versus F0 x L1
F4 weighting effect:   F4 x L1 versus F4 x L0
F4 ranking effect:     F4 x L2 versus F4 x L1
selection effect:      F4 x L3 versus F4 x L2 on the same trajectory
data-scale effect:     D1 versus D0 within one frozen F x L condition
```

Do not attribute an F4 gain to ontology or factorization if the matched L0
comparison does not improve. Do not attribute an L1/L2 gain to F4 when the
same objective also improves F0 by a similar amount.

Hard-negative classes are operator swaps, reference/dependency rebinding,
query-target swaps, path-binding swaps, and source-fact perturbations. Keep
only syntactically valid, lowerable, executable negatives that are not
semantically equivalent to the positive program. Never generate test
negatives or select negative policy from test outcomes.

Checkpoint ranking is lexicographic on the development set: lowerability,
type validity, execution, semantic return, semantic state, dependency,
operator/path/source-fact scores, answer, then token loss. This ordering makes
an unsafe or structurally wrong checkpoint unable to win merely through lower
surface-form loss.

Use the same Qwen3-0.6B QKVO-r8 adapter, frozen 450/25/25 identities, optimizer
budget, fixed zero-shot prompt, and autonomous test path for every D0 F x L
cell. Neither gold ASL, rationale, answers, intermediate values, retrieved
examples, nor a record-dependent ICL prefix may enter the model input.
Retrieved demonstrations remain a separate later intervention.

Treat the L1 weights as hyperparameters rather than facts. First compare the
already-frozen F0 x L0 control against one predeclared F0 x L1 strong-weight
condition. If
that gives a development semantic signal, compare mild and strong weighting
with the adapter rank, optimizer, examples, and exposures fixed. Normalize the
weighted token loss by the sum of active weights, report ordinary loss beside
it, and freeze the selected weighting before applying it to another
architecture or model. Do not independently tune every architecture on the
test set. Later optimizer sweeps vary one axis at a time in this order:

```text
semantic-weight ratio -> learning rate -> selected epoch -> dropout -> rank
```

This ordering tests the semantic-objective hypothesis before adding capacity.
The 25-case development set is small, so retain exact counts and treat close
settings as tied rather than selecting on a one-example fluctuation.

L2 uses no new teacher annotation. Define the pair score as mean
semantic-weighted conditional log likelihood:

```text
score(NL, ASL) = mean_w log p(ASL tokens | NL)
loss_rank = softplus((score_negative - score_positive) / temperature)
```

Cycle deterministically across available operator, dependency, query, binding,
and source-fact corruptions over the ten logical epochs. Every negative must
parse, lower, type-check, and execute, and it must share the exact NL prompt
with its positive. Preserve negatives whose final numeric answer happens to
match when their binding is wrong, but report them explicitly; they test
semantic grounding rather than answer discrimination. Generate no test
preferences. L2 retains the L1 positive SFT loss, and changes only the
additional ranking term.

Gates:

1. F4 requires 500/500 gold IR programs to round-trip with equivalent state,
   return, dependencies, and answer.
2. F4 x L1 requires unit-tested token/component weighting and reports both
   weighted and ordinary development loss.
3. Every L2 cell reports every negative class and the fraction whose final
   answer is accidentally unchanged; such cases remain useful binding
   negatives.
4. L3 may inspect only the 25 development programs. The frozen test is run
   once after checkpoint selection.
5. A gain is attributed to the F4 representation only if F4 x L0 beats F0 x L0
   on semantic return/state or answer under matched identities and budget.
6. Objective gains are reported within representation first; pooled claims
   across F0 and F4 are secondary.
7. D1 must publish pre-exclusion and post-exclusion source/pattern counts,
   teacher-model composition, strict acceptance rate, and overlap audit before
   training.

### Phase 4 - Developmental curriculum

Only continue if E3b improves at least one autonomous semantic metric over E3a
without catastrophic answer loss.

Construct three training stages:

```text
D0 co-present grounding:
complete ASL-world + NL describe overlapping content

D1 grounded extension:
partial ASL-world + NL add asserted, temporal, reported, or hypothetical facts

D2 displaced language:
NL only -> construct or update ASL_int
```

Late stages retain 10-20% grounded/denoising rehearsal to test and limit
catastrophic forgetting. Alignment loss covers only overlap annotations;
discourse-only facts are trained as provenance-tagged internal updates.

Run the following exposure-matched controls:

```text
K0 ordered D0 -> D1 -> D2 curriculum
K1 static mixture with identical examples and total exposures
K2 reverse D2 -> D1 -> D0 curriculum
K3 NL-only training
K4 shuffled/mismatched ASL-world pairs
```

The curriculum claim requires K0 to exceed K1 and K2, not merely K3. K4 tests
whether correct world-language pairing matters.

### Phase 5 - Predicted-ASL runtime reinjection

For R0, E3a, and the best aligned/curriculum condition, run:

```text
NL only
  -> generate ASL_int
  -> parse, lower, type-check, and execute
  -> canonical predicted state/result
  -> reinject into the continuation
  -> produce final NL answer
```

Never substitute gold ASL or a gold result after failure. Invalid programs
produce an explicit error/abstention record. Compare:

```text
no reinjection
predicted ASL text reinjection
canonical runtime state/result reinjection
```

Report downstream answer accuracy, runtime faithfulness, rescue rate, harm
rate, abstention, and additional tokens/latency.

## Primary evaluation

Autonomous behavioral evaluation remains primary:

```text
NL only -> ASL_int
```

Report exact counts and rates for:

- parse, lowerability, type validity, and execution;
- source facts, paths, operators, and dependencies;
- relation direction, argument roles, and query target;
- semantic return and semantic state;
- final executed answer.

Representation diagnostics are secondary:

- NL-to-ASL and ASL-to-NL retrieval at top-1 and top-k;
- paired cosine and linear CKA by depth;
- ASL identity and corruption recovery;
- alignment drift after causal generation training.

Gold ASL may be used after autonomous generation for offline diagnostic scoring.
It must never be passed into the generator on the test path.

## Gates

1. **Routing gate:** S2/S3 must pass activation, gradient, and no-overlap tests.
2. **ASL sufficiency gate:** C1 must recover corrupted ASL above C0 chance/error
   behavior; identity copying alone is insufficient.
3. **Alignment gate:** E3b must improve an autonomous semantic metric over E3a
   without losing more than one answer on the 25-case test.
4. **Replication gate:** any apparent behavioral win is repeated with seeds 23
   and 37 before architecture claims.
5. **Capacity gate:** a positive condition with materially more parameters is
   compared with Q0 rank-16 or another parameter-matched plain adapter.
6. **Curriculum gate:** K0 must beat exposure-matched static and reverse order.
7. **Reinjection gate:** runtime reinjection must improve answers or faithfulness
   without an unacceptable harm rate.
8. **Scale gate:** Qwen3-1.7B, F0-large, component alignment, reconstruction
   extensions, and a dedicated encoder follow only a credible 0.6B signal.

With only 25 test programs, report exact paired outcomes, Wilson intervals, and
paired exact tests as exploratory. Do not promote a single-seed one-answer
difference to a positive result.

## Required implementation artifacts

```text
src/ccpu/paper1/e3/
  data.py
  bottleneck.py
  components.py
  negatives.py
  selection.py
  adapters.py
  alignment.py
  pretrain.py
  generation.py
  reinjection.py
  diagnostics.py

configs/paper1/e3/
  c0_identity.json
  c1_denoise.json
  e3a_nl_memory.json
  e3b_contrastive.json
  curriculum_ordered.json
  curriculum_static.json
  curriculum_reverse.json

artifacts/paper1/e3_v2/
  manifests/
  channel/
  alignment/
  generation/
  reinjection/
  analysis/
```

Required deterministic tests include adapter routing, ASL-only view isolation,
corruption provenance, false-negative filtering, contrastive-loss correctness,
checkpoint transfer, overlap-only alignment, curriculum exposure matching,
test-split isolation, and rejection of external/gold ASL on autonomous paths.

## Execution order

```text
P0  Finish S2 and S3.
P1  Freeze their reports and update the architecture table.
P2  Build and audit C0/C1 ASL-only views.
P3  Train C0 and C1; freeze the ASL-channel checkpoint.
P4  Train/evaluate F0 x L1 (historical M0.5); tune weights only after a dev signal.
P4A Train/evaluate F0 x L2 (historical M0.6) with within-example negatives.
P5  Build F4, require the 500/500 exact semantic round-trip, and freeze data.
P6  Train/evaluate the matched F0 x L0 versus F4 x L0 representation comparison.
P6A Run F4 x L1 and F4 x L2, then apply L3 selection to the same L2 trajectory.
P6B Freeze leakage-audited D1; run F0 x L0 scale, then matched winner/F4 cells.
P7  Implement and train E3a NL-memory generation.
P8  Precondition E3b with C1 plus paired contrastive alignment.
P9  Measure retrieval and layerwise convergence before generation.
P10 Transfer E3b into the identical E3a generation schedule.
P11 Evaluate E3a/E3b autonomously on the frozen 25 programs.
P12 If positive, replicate and run the capacity control.
P13 Build D0/D1/D2 with provenance and overlap masks.
P14 Run ordered, static, reverse, NL-only, and shuffled-pair controls.
P15 Run predicted-ASL execution and reinjection for frozen controls and winner.
P16 Compare architecture, bottleneck, alignment, curriculum, reinjection, and F0-large data.
P17 Update Paper 1, build the PDF, commit, and push each major result.
```

Do not change ICL examples between records or steps. Do not select prompts,
states, demonstrations, or checkpoints using test outcomes.
