# AGENTS - Paper 1 E3 Semantic Alignment Ladder v2

## Status and authority

This is the canonical execution order for the Paper 1 E3 experiments. It
refines `AGENTS_paper1_E3_semantic_alignment_ladder.md` by separating five
questions that must not be confounded:

1. Does modality-specific adaptation help?
2. Can the ASL channel represent and canonicalize a symbolic world by itself?
3. Can paired NL and ASL converge on shared world content?
4. Does a grounded-to-NL-only curriculum improve autonomous internal ASL?
5. Does executing and reinjecting predicted internal ASL improve continuation?

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
P4  Implement and train E3a NL-memory generation.
P5  Precondition E3b with C1 plus paired contrastive alignment.
P6  Measure retrieval and layerwise convergence before generation.
P7  Transfer E3b into the identical E3a generation schedule.
P8  Evaluate E3a/E3b autonomously on the frozen 25 programs.
P9  If positive, replicate and run the capacity control.
P10 Build D0/D1/D2 with provenance and overlap masks.
P11 Run ordered, static, reverse, NL-only, and shuffled-pair controls.
P12 Run predicted-ASL execution and reinjection for frozen controls and winner.
P13 Compare architecture, alignment, curriculum, reinjection, and F0-large data.
P14 Update Paper 1, build the PDF, commit, and push each major result.
```

Do not change ICL examples between records or steps. Do not select prompts,
states, demonstrations, or checkpoints using test outcomes.
