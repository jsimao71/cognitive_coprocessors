# AGENTS.md — ASL Grounding 3D Experiment Matrix

## 0. Purpose

This experiment series tests whether training with an externally supplied symbolic representation (`ASL_ext`) improves the model's ability to construct the same or a compatible symbolic world representation internally (`ASL_int`) from natural language (`NL`) **when external ASL is absent at inference time**.

The experimental space has three independent axes:

1. **Encoder architecture**
   - A1: separate NL and ASL encoders
   - A2: fully shared multilingual encoder
   - A3: hybrid shared/specialized/aligned encoder
2. **Attention / memory fusion mode**
   - M1: separate cross-attention
   - M2: merged/native-style K/V attention
3. **Training regime**
   - T1: full external-ASL assistance
   - T2: partial/corrupted external-ASL assistance
   - T3: autonomous NL-only reconstruction

Keep these axes orthogonal in code, configuration, reporting, and analysis.

---

## 1. Terminology

### NL
Natural-language source input.

Example:

```text
John removes six pink hats. He removes twice as many green hats.
```

### ASL_ext
Externally supplied symbolic representation used as a training-time grounding signal.

Example:

```text
e1 := remove(john, hats.pink, 6)
e2 := remove(john, hats.green, 2 * e1.quantity)
```

This is a teacher/world representation. It must not be assumed available at autonomous inference time.

### ASL_int
The symbolic representation generated internally by the model. This is the main prediction target.

### Teacher-assisted inference
The decoder has access to NL plus full or partial `ASL_ext`.

### Autonomous inference
The decoder has access to NL only. This is the headline evaluation condition.

---

## 2. Main hypotheses

### H1 — External symbolic grounding helps autonomous inference

Training with `ASL_ext` should improve later NL-only generation of `ASL_int`.

Primary test:

```text
Autonomous score after mixed ASL-grounded training
>
Autonomous score after matched NL-only training
```

### H2 — Shared representation increases transfer

Architectures that force NL and ASL to share encoder computation should transfer symbolic structure into autonomous NL processing better than fully separate encoders.

Possible ordering, to be tested rather than assumed:

```text
hybrid >= shared > separate
```

### H3 — Hybrid sharing avoids negative transfer

A fully shared encoder may overconstrain low-level NL and ASL processing. A hybrid model may perform better by allowing modality/language-specific processing while forcing common higher-level semantic computation.

### H4 — Merged K/V encourages direct source competition

Merged/native-style K/V puts NL and ASL memory positions under one attention softmax. This may create stronger semantic interchange than separately normalized cross-attention branches.

### H5 — Partial assistance may be more useful than full assistance

Full `ASL_ext` may permit copying. Partial/corrupted ASL should force the model to combine NL evidence with symbolic structure.

### H6 — Useful grounding reduces the teacher/autonomous gap

A successful model should not merely perform well while the teacher is present.

```text
teacher_gap = teacher_assisted_score - autonomous_score
```

The gap should fall without sacrificing autonomous quality.

---

## 3. Full experiment matrix

The primary factorial space is:

```text
3 encoder architectures
×
2 attention modes
×
3 training regimes
=
18 primary cells
```

The three questions are distinct:

```text
ARCHITECTURE:
Where are NL and ASL forced to share computation?

ATTENTION:
How does the decoder access and combine NL and ASL memories?

TRAINING REGIME:
How much external symbolic information is present for an example?
```

Do not encode regime behavior into architecture names or attention modules.

---

# 4. Axis A — Encoder architectures

## A1 — Separate encoders

```text
NL
 │
 ▼
E_N
 │
 └──────────────► KV_N ─────────┐
                                ▼
                              decoder ─► ASL_int
                                ▲
ASL_ext                         │
 │                              │
 ▼                              │
E_A                             │
 │                              │
 └──────────────► KV_A ─────────┘
```

Definitions:

```text
H_N = E_N(NL)
H_A = E_A(ASL_ext)
```

`E_N` and `E_A` do not share Transformer blocks.

### Interpretation

This is the weakest representation-alignment assumption. The decoder may simply learn two interpretation channels:

```text
NL representation  -> decoder path A
ASL representation -> decoder path B
```

Success therefore does **not** imply that the two encoders learn a shared geometry.

### Required diagnostics

Measure:
- NL/ASL hidden-state similarity;
- layerwise CKA and/or SVCCA;
- paired-state retrieval accuracy;
- linear-probe transfer;
- decoder source usage;
- autonomous task performance.

A1 is the baseline for determining whether stronger sharing is necessary.

---

## A2 — Fully shared multilingual encoder

```text
NL ─────────┐
            ▼
        E_shared
            │
            ├────────► KV_N
            │
ASL_ext ────┘
            │
            └────────► KV_A
```

Use shared Transformer weights. It is acceptable to retain:
- source/type embeddings;
- distinct token embeddings if required;
- source marker tokens.

Prefer shared tokenization where practical, but do not distort either language solely to force tokenizer identity.

### Purpose

Both NL and ASL pass through the same representational machinery. This increases pressure toward a common semantic space.

### Caveat

Weight sharing does not guarantee state alignment. The model may still allocate different subspaces to each language. Continue to measure representation similarity directly.

---

## A3 — Hybrid encoder

Canonical layout:

```text
NL  ─► E_N lower/mid ─┐
                      ├──► E_NA shared/aligned semantic layers ─► KV_N_world
ASL ─► E_A lower/mid ─┘
                                                   └────────────► KV_A_world
```

The lower and middle encoder stacks are **language-specific**. Only the upper semantic/world layers are shared.

Suggested notation:

```text
E_N   = NL-specific lower/mid encoder
E_A   = ASL-specific lower/mid encoder
E_NA  = shared/aligned semantic/world encoder
```

Forward equations:

```text
H_N = E_N(NL)
Z_N = E_NA(H_N)

H_A = E_A(ASL_ext)
Z_A = E_NA(H_A)
```

### Purpose

This architecture directly tests the hypothesis that:
- NL and ASL have substantially different surface statistics and should not be forced through the same low-level syntax machinery;
- they nevertheless describe the same world and should converge into a common higher-level semantic/world representation.

Conceptually:

```text
different surface languages
        ↓
language-specific abstraction
        ↓
shared internal world representation
```

### Why the bottom layers are different

NL lower/mid layers must model:
- syntax;
- phrase structure;
- lexical ambiguity;
- pronouns/coreference;
- discourse context;
- natural-language surface variation.

ASL lower/mid layers see:
- regular symbolic syntax;
- explicit paths;
- explicit relation names;
- explicit arguments;
- stable symbolic references.

Forcing these two surfaces through the same lower layers may create unnecessary negative transfer.

The sharing hypothesis applies primarily to the upper semantic/world layers.

### Layer split

Do not hard-code a single split.

At minimum support:

```text
nl_specific_layers
asl_specific_layers
shared_semantic_layers
```

or fractions:

```text
specific_lower_fraction
shared_top_fraction
```

Recommended initial split for a 12-layer effective path:

```text
E_N:   8 NL-specific layers
E_A:   8 ASL-specific layers
E_NA:  4 shared semantic layers
```

The effective depth for either source remains 12 layers.

This is only a starting point.

### Optional pre-convergence and post-convergence memories

The decoder should optionally receive both language-specific and shared semantic memories:

```text
NL  ─► E_N ─────► KV_N_surface ─────┐
          │                         │
          └► E_NA ─► KV_N_world ───┤
                                    ├──► ASL decoder
ASL ─► E_A ─────► KV_A_surface ─────┤
          │                         │
          └► E_NA ─► KV_A_world ───┘
```

This prevents the shared semantic representation from having to preserve every literal/token-level detail.

Use cases:
- `KV_*_surface` for exact names, numbers, lexical evidence, and local syntax;
- `KV_*_world` for entities, relations, roles, dependencies, and semantic state.

Do not enable all four memories in the first baseline. Start with only `KV_N_world` / `KV_A_world`, then ablate additional surface-memory access.

### Optional alignment projection

Support an optional projection before `E_NA`:

```text
P_N(H_N)
P_A(H_A)
```

Use only if dimensions differ or if alignment is unstable.

Do not add it by default if unnecessary.

# 5. Multi-depth encoder memories

Architectures should eventually support exposing multiple encoder depths:

```text
KV_low
KV_mid
KV_semantic
```

Possible interpretation:

```text
lower layer  -> lexical/entity evidence
middle layer -> relational/compositional evidence
upper layer  -> high-level semantic/world representation
```

For the first matrix, use final-layer states only. Add multi-depth memory after baseline effects are established.

---

# 6. Axis M — Attention / memory fusion

The decoder should expose one common memory interface with two interchangeable fusion implementations.

## M1 — Separate cross-attention

Classical encoder-decoder-style source access.

Preferred initial form:

```text
O_N = CrossAttn(Q, K_N, V_N)
O_A = CrossAttn(Q, K_A, V_A)

O = H + g_N * O_N + g_A * O_A
```

Each source has:
- separate K/V;
- separate normalization/softmax;
- optionally separate output projection;
- explicit gate or scale.

### Why M1 matters

It is:
- closest to classical encoder-decoder Transformers;
- easy to debug;
- easy to ablate;
- easy to attribute by source;
- easy to disable ASL memory at autonomous inference.

### Required logs

Per decoder layer:
- NL attention entropy;
- ASL attention entropy;
- maximum attention;
- output norm per branch;
- gates/scales;
- token-wise source dominance.

---

## M2 — Merged/native-style K/V

Build one joint encoder memory:

```text
K_mem = concat(K_N, K_A)
V_mem = concat(V_N, V_A)
```

then:

```text
O = softmax(Q @ K_mem^T / sqrt(d)) @ V_mem
```

NL and ASL positions directly compete under one softmax.

### Source identity

Provide a source-type signal:

```text
NL_MEMORY
ASL_MEMORY
```

using type/segment embeddings or a comparable mechanism. Do not rely solely on vocabulary differences.

### Required logs

Record:
- total attention mass to NL;
- total attention mass to ASL;
- source of maximum-attended memory position;
- attention entropy;
- source mass by decoder layer;
- source mass by generated ASL token category.

### Extended native mode — later only

Later, optionally test:

```text
K_total = concat(K_decoder_self, K_N, K_A)
V_total = concat(V_decoder_self, V_N, V_A)
```

under a unified attention computation.

Do **not** use this as the initial M2 condition. First compare separate vs merged **encoder memory** while keeping causal decoder self-attention fixed.

---

# 7. Relation of M1/M2 to classical Transformers and PRA

### Classical Transformer

The conventional encoder-decoder pattern is:

```text
source
  ↓
bidirectional encoder self-attention
  ↓
source K/V

causal decoder
  ↓
cross-attention(Q_decoder, K_source, V_source)
  ↓
target
```

M1 is the conventional multi-source extension of this pattern.

### PRA analogy

The current canonical PRA native-K/V design conceptually uses:

```text
K_total = concat(K_retrieved_memory, K_local)
V_total = concat(V_retrieved_memory, V_local)
```

with one attention softmax.

Older/legacy PRA cross-attention instead computes memory attention separately and combines it with local attention.

For this ASL experiment:

```text
M1 separate cross-attention
≈ older/separate PRA memory branch philosophy

M2 merged K/V
≈ current PRA native-K/V philosophy
```

Do not import PRA retrieval/routing into this first experiment series. The present question is representation and grounding, not retrieval efficiency.

---

# 8. Axis T — Training regimes

Each paired NL/ASL example should be transformable at runtime into T1, T2, or T3 views.

## T1 — Full external-ASL assistance

Input:

```text
NL
+
full ASL_ext
```

Target:

```text
ASL_int
```

Purpose:
- establish teacher-assisted upper bound;
- teach the decoder to read symbolic memory;
- establish a high-quality symbolic semantic channel.

### Main risk: trivial copying

Always report:
- direct input/target token overlap;
- attention to matching teacher tokens;
- copy rate;
- performance after teacher removal.

Do not interpret T1 teacher-assisted performance as evidence of autonomous grounding.

---

## T2 — Partial/corrupted external-ASL assistance

Input:

```text
NL
+
corrupted ASL_ext
```

Target:

```text
full ASL_int
```

Supported corruption operators should include:

```text
record dropout
entity masking
relation masking
argument masking
value masking
dependency masking
attribute masking
span masking
random token masking
record permutation
noise-record insertion
```

Example gold:

```text
e1 := remove(john, hats.pink, 6)
e2 := remove(john, hats.green, 2 * e1.quantity)
```

Possible partial teacher:

```text
e1 := remove(john, hats.pink, ?)
e2 := remove(john, hats.green, 2 * ?)
```

or:

```text
remove(john, hats.pink, ?)
remove(john, hats.green, ?)
```

or only symbolic anchors:

```text
john
remove
hats.pink
hats.green
```

### Corruption severity

Support configurable severity, e.g.:

```text
easy   = ~20% information removed
medium = ~50%
hard   = ~80%
```

Do not collapse all T2 examples into one aggregate score. Evaluate by corruption type and severity.

---

## T3 — Autonomous NL-only reconstruction

Input:

```text
NL
```

Target:

```text
ASL_int
```

No external ASL memory is supplied.

This is inference-matched training and must remain a substantial component of any model intended for autonomous use.

---

# 9. Training schedule families

## 9.1 Static mixtures

Support configurable mixtures such as:

```text
S0: T1=0.00 T2=0.00 T3=1.00   # critical NL-only baseline
S1: T1=0.25 T2=0.35 T3=0.40
S2: T1=0.10 T2=0.30 T3=0.60
S3: T1=0.40 T2=0.40 T3=0.20
```

Do not assume one mixture is best.

## 9.2 Curriculum mixtures

Example:

```text
early:
T1 60%
T2 30%
T3 10%

middle:
T1 30%
T2 40%
T3 30%

late:
T1 10%
T2 30%
T3 60%
```

Purpose:
1. establish symbolic decoding;
2. require multimodal integration;
3. emphasize autonomous reconstruction.

Compare curriculum against static mixtures rather than assuming it is superior.

---

# 10. Optional auxiliary objectives

Do not introduce these until the basic matrix is stable.

## 10.1 ASL → ASL denoising

```text
corrupted ASL_ext
    ↓
ASL encoder
    ↓
decoder
    ↓
full ASL
```

Never train a trivial exact-copy autoencoder. Corrupt the source or otherwise prevent identity copying.

## 10.2 Teacher/student consistency

For one example compute:

```text
P_teacher = p(y_t | NL, ASL_ext, y_<t)
P_auto    = p(y_t | NL, y_<t)
```

Optional loss:

```text
L_consistency = KL(stopgrad(P_teacher) || P_auto)
```

Total loss scaffold:

```text
L = L_generation
  + lambda_consistency * L_consistency
  + lambda_alignment * L_alignment
  + lambda_aux * L_ASL_denoising
```

Defaults:

```text
lambda_consistency = 0
lambda_alignment   = 0
lambda_aux         = 0
```

Add one auxiliary mechanism at a time.

## 10.3 Explicit representation alignment

Possible later objectives:
- cosine alignment;
- InfoNCE;
- projected MSE;
- entity-level alignment;
- relation-level alignment.

Representation similarity is diagnostic. Autonomous ASL task performance remains primary.

---

# 11. Evaluation conditions

Every trained model should be evaluated under all compatible conditions.

## EVAL-A — Autonomous

```text
NL only
```

This is the headline result.

## EVAL-F — Full teacher

```text
NL + full ASL_ext
```

Teacher-assisted upper bound.

## EVAL-P — Partial teacher

Evaluate multiple corruption severities and corruption operators.

## EVAL-A0 — ASL teacher only

Where architecture permits:

```text
ASL_ext only
```

Measures how much the decoder can solve from symbolic memory alone.

## EVAL-NOISE — Incorrect teacher

Inject:
- wrong entity;
- wrong value;
- wrong relation;
- wrong dependency;
- unrelated ASL.

Measure whether the model blindly follows incorrect symbolic memory.

---

# 12. Core metrics

Do not rely on raw string exact match alone.

Report at least:

```text
exact ASL match
syntax validity
entity accuracy
relation accuracy
attribute accuracy
argument-role accuracy
numeric/value accuracy
dependency/reference accuracy
record-level precision/recall/F1
semantic/graph equivalence
```

If ASL admits equivalent serializations, canonicalize before exact matching.

---

# 13. Derived metrics

## 13.1 Grounding gain

```text
grounding_gain =
    autonomous_score(mixed_ASL_training)
    - autonomous_score(matched_NL_only_training)
```

This is a main headline metric.

## 13.2 Teacher gap

```text
teacher_gap = full_teacher_score - autonomous_score
```

## 13.3 Partial-teacher robustness

Plot score against retained symbolic information, for example:

```text
100%, 80%, 50%, 20%, 0%
```

## 13.4 Wrong-teacher susceptibility

Measure both performance drop and the frequency with which generated output follows the incorrect teacher.

## 13.5 Representation alignment

For paired NL/ASL inputs measure:
- cosine similarity;
- CKA;
- SVCCA;
- linear mapping quality;
- paired-state retrieval accuracy.

Do not substitute these diagnostics for behavioral results.

---

# 14. Attention diagnostics

For every decoder layer and generation position log source usage.

For M1:

```text
NL branch attention entropy
ASL branch attention entropy
NL branch output norm
ASL branch output norm
g_N
g_A
```

For M2:

```text
attention_mass_NL
attention_mass_ASL
source_of_max_attention
joint_attention_entropy
```

Useful derived statistic:

```text
ASL_source_ratio =
ASL_attention_mass /
(NL_attention_mass + ASL_attention_mass)
```

Analyze by generated ASL token type:
- entity;
- relation;
- argument;
- numeric value;
- dependency/reference token.

---

# 15. Critical matched controls

## C0 — NL-only baseline

Architecture and decoder matched; external-ASL branch disabled; train T3 only.

## C1 — Random teacher

Replace `ASL_ext` with unrelated ASL of matched length.

Tests whether gains arise from meaningful grounding rather than extra tokens, regularization, or computation.

## C2 — Scrambled teacher

Preserve the token multiset where possible but corrupt symbolic structure.

## C3 — Surface paraphrase teacher

Replace ASL with a natural-language paraphrase of comparable information/length.

Tests whether symbolic structure itself matters.

## C4 — Parameter-count control

If architectures have unequal capacity, add width-adjusted or otherwise parameter-matched controls.

## C5 — Compute-matched control

Where feasible match:
- optimizer steps;
- examples;
- source/target tokens;
- approximate FLOPs.

---

# 16. Leakage rules

Prohibited:
- target ASL present in NL-only inputs;
- gold-ASL-derived features in autonomous evaluation;
- checkpoint selection using test results;
- corruption metadata accidentally containing target content;
- tokenizer metadata carrying target records;
- test ASL examples used as retrieval/memory;
- future target access beyond normal decoder teacher forcing.

Every batch/view should explicitly store:

```text
has_external_asl
external_asl_fraction
external_asl_corruption
source_fields_visible_to_model
```

Add assertions in the collator and evaluation harness.

---

# 17. Architectural fairness

When comparing A1/A2/A3:
- keep decoder architecture identical;
- keep hidden width identical where possible;
- keep total encoder depth comparable;
- report parameter counts;
- use the same optimizer/LR schedule unless instability forces a documented exception;
- preserve dataset order and seeds;
- preserve tokenizer where feasible.

If A3 has extra capacity, provide a parameter-matched control or explicitly quantify the capacity difference.

---

# 18. Experiment staging

Do not start with all 18 primary cells.

## P0 — Dataset/evaluator validation

Before model runs:
- canonicalize ASL;
- validate parser;
- validate semantic-equivalence evaluator;
- verify corruption operators;
- verify train/dev/test disjointness;
- verify no target leakage;
- create tiny deterministic smoke cases.

Gate:

```text
all evaluator tests pass
all corruption operators expose correct metadata
T3 batches contain no external ASL
```

## P1 — Establish whether symbolic grounding helps

Use the pinned pretrained T5-small checkpoint with a frozen base and QKVO
rank-8 LoRA. Train the source-type embedding and M1 gates alongside the adapters.
Keep full-model T5 fine-tuning only as a separately named capacity control.

Run:

```text
A1 + M1 + T3-only
A1 + M1 + mixed T1/T2/T3
```

Question:

> Does external symbolic training improve autonomous NL→ASL at all?

Do not move to elaborate architectures before this is understood.

## P1Q — Patch a small pretrained causal model

After the pretrained T5-small B0/B1 grounding gate, patch a small pretrained
causal language model, beginning with **Qwen3-0.6B**. The purpose is to preserve
the model's existing natural-language competence while changing only how it can
learn from and attend to external ASL during training.

Freeze the pretrained backbone by default. Train only:

- QKVO rank-8 LoRA adapters;
- source-type embeddings;
- small external-memory projection modules;
- M1 cross-attention branches or M2 native-K/V injection modules;
- explicit gates and normalizations introduced by the patch.

Do not train a new language model from scratch for this track. Do not fully
fine-tune Qwen unless a later, separately named control justifies that cost.

The first Qwen gate is:

```text
Q0: existing Qwen3-0.6B F0 QKVO-r8, T3 only
Q1: same backbone/LoRA budget, serialized mixed T1/T2/T3 context baseline
Q2: same backbone/LoRA budget, M1 side cross-attention patch
Q3: same backbone/LoRA budget, M2 native merged-K/V patch
```

Evaluate every condition autonomously with NL only. Q1 is required to separate
the benefit of extra symbolic training text from the benefit of an architectural
memory patch. Q2/Q3 must report patch parameters separately from LoRA parameters.

Only after Q1--Q3 establish a signal should the Qwen track test shared versus
hybrid source adaptation or move to Qwen3-1.7B.

## P2 — Attention-mode comparison

Fix A1 and compare:

```text
A1 + M1
A1 + M2
```

with matched training mixtures.

Question:

> Does joint K/V competition outperform separately normalized cross-attention?

## P3 — Shared encoder

Use the winning or simpler attention mode and compare:

```text
A1 vs A2
```

Question:

> Does encoder sharing improve autonomous transfer?

## P4 — Hybrid encoder

Compare:

```text
A2 vs A3
```

Question:

> Does specialization plus a shared semantic top beat full sharing?

## P5 — Full factorial confirmation

Only now run the complete or statistically justified subset of:

```text
3 architectures × 2 attention modes × 3 regimes
```

with multiple seeds.

## P6 — Auxiliary objectives

Test one at a time:
- consistency KD;
- explicit representation alignment;
- ASL denoising.

## P7 — Multi-depth memory

Expose lower/mid/top encoder representations to decoder layers and test whether different decoder layers prefer different semantic depths.

---

# 19. Minimal first compute sweep

Run this ladder first:

| ID | Architecture | Attention | Adaptation | Training |
|---|---|---|---|---|
| B0 | Separate | Cross | frozen T5 + QKVO LoRA-r8 | T3 only |
| B1 | Separate | Cross | frozen T5 + QKVO LoRA-r8 | mixed T1/T2/T3 |
| B2 | Separate | Merged K/V | frozen T5 + QKVO LoRA-r8 | mixed T1/T2/T3 |
| B3 | Shared | best M1/M2 | frozen T5 + QKVO LoRA-r8 | mixed T1/T2/T3 |
| B4 | Hybrid | best M1/M2 | frozen T5 + QKVO LoRA-r8 | mixed T1/T2/T3 |

Add matched T3-only versions of B2–B4 if architecture/capacity changes autonomous performance materially.

This ladder should determine whether the complete factorial sweep is worth the compute.

Current execution status: the seed-11 pretrained T5-small LoRA B0 and B1 cells
both produced 0/25 autonomous parse-valid, executable, semantic, and
answer-correct programs. The B1-minus-B0 autonomous grounding gain is zero, so
the gate stops B2--B4 and advances to the Qwen3-0.6B pretrained-LoRA ladder.

Run the Qwen pretrained-model patch ladder after the T5-small B0/B1 gate and
before scaling to a larger backbone. No language model in this roadmap is
trained from scratch:

| ID | Pretrained backbone | Patch | Trainable components | Training |
|---|---|---|---|---|
| Q0 | Qwen3-0.6B | none | QKVO LoRA r8 | T3 only |
| Q1 | Qwen3-0.6B | serialized teacher | QKVO LoRA r8 | mixed T1/T2/T3 |
| Q2 | Qwen3-0.6B | M1 side cross-attention | QKVO LoRA r8 + patch | mixed T1/T2/T3 |
| Q3 | Qwen3-0.6B | M2 native merged K/V | QKVO LoRA r8 + patch | mixed T1/T2/T3 |

Use the same frozen 450/25/25 F0 identities, runtime views, corruption seeds,
target-token budget, optimizer steps, and autonomous evaluator as B0--B4. If a
patch changes sequence length or FLOPs materially, report both optimizer-matched
and token/compute-matched controls.

---

# 20. Suggested config schema

```yaml
model:
  d_model: 512
  n_heads: 8
  encoder_layers: 12
  decoder_layers: 12

encoder:
  architecture: separate  # separate | shared | hybrid
  hybrid:
    shared_lower_layers: 4
    specialized_layers: 4
    shared_top_layers: 4

attention:
  mode: cross             # cross | merged_kv
  source_type_embeddings: true
  separate_output_proj: true

training:
  schedule: static        # static | curriculum
  static:
    full_teacher: 0.25
    partial_teacher: 0.35
    autonomous: 0.40

  corruption:
    policy:
      - record_dropout
      - value_mask
      - argument_mask
    severity: 0.50

loss:
  generation: 1.0
  consistency: 0.0
  alignment: 0.0
  asl_denoising: 0.0

evaluation:
  autonomous: true
  full_teacher: true
  partial_teacher: true
  wrong_teacher: true
```

---

# 21. Recommended code structure

Prefer one compositional implementation rather than separate end-to-end models.

```text
src/
  encoders/
    separate.py
    shared.py
    hybrid.py

  attention/
    cross_memory.py
    merged_kv.py

  training/
    regimes.py
    corruption.py
    schedules.py

  losses/
    generation.py
    consistency.py
    alignment.py

  eval/
    asl_parser.py
    semantic_metrics.py
    attention_metrics.py
    representation_metrics.py

  experiments/
    matrix.py
```

Use:
- one decoder implementation;
- one source-memory abstraction;
- interchangeable encoder factories;
- interchangeable attention/fusion modules.

---

# 22. Source-memory interface

All encoders should emit a common object, conceptually:

```python
SourceMemory(
    hidden_states=...,
    attention_mask=...,
    source_type="nl" | "asl",
    layer_states={
        4: ...,
        8: ...,
        12: ...,
    },
    metadata=...,
)
```

The decoder/fusion layer should not contain architecture-specific logic.

---

# 23. Training-regime interface

Prefer runtime view generation over three duplicated datasets.

Conceptual API:

```python
view = regime_builder.make_view(
    example,
    regime="partial",
    corruption_policy=...,
)
```

Returned view:

```text
nl_input
external_asl_input or None
target_asl
regime
corruption_metadata
```

This makes matched T1/T2/T3 examples easy to generate and audit.

---

# 24. Results table

Store one row per run in a tidy table.

Required fields:

```text
run_id
seed
model
encoder_architecture
attention_mode
training_schedule
T1_fraction
T2_fraction
T3_fraction
corruption_policy
corruption_severity
parameter_count
training_tokens
training_flops_estimate
autonomous_exact
autonomous_semantic
full_teacher_exact
full_teacher_semantic
partial_teacher_exact
partial_teacher_semantic
wrong_teacher_exact
teacher_gap
grounding_gain
syntax_validity
entity_accuracy
relation_accuracy
argument_accuracy
value_accuracy
dependency_accuracy
```

Keep raw predictions, attention traces, and representation probes separately.

---

# 25. Required plots

Generate at least:

1. Autonomous semantic accuracy by architecture, grouped by attention mode.
2. Grounding gain by architecture × attention.
3. Teacher gap by architecture × attention.
4. Accuracy vs ASL corruption severity.
5. NL vs ASL attention mass by decoder layer.
6. NL/ASL representation CKA by encoder layer.

For A3, mark shared-lower, specialized-middle, and shared-top layer regions.

---

# 26. Statistical requirements

Exploration:

```text
minimum 3 seeds
```

Confirmation:

```text
prefer 5+ seeds
```

Report:
- mean;
- standard deviation;
- confidence intervals where useful.

Use paired comparisons when runs share seed, dataset ordering, initialization family, and schedule.

Do not claim superiority from a single run.

---

# 27. Initial paper questions

The first paper should stay focused on:

1. Does symbolic teacher exposure improve autonomous NL→ASL?
2. Is partial symbolic assistance more effective than full assistance?
3. Does encoder sharing improve autonomous transfer?
4. Does merged K/V outperform separate cross-attention?
5. Does the hybrid architecture offer a better specialization/alignment tradeoff than full sharing?

Do not dilute the first study with:
- PRA routing;
- long-context retrieval;
- online learning;
- persistent world memory;
- multiple cognitive coprocessors;
- agent integration.

Those are follow-up directions.

---

# 28. Success criteria

A positive result is **not** simply:

```text
teacher-assisted generation is excellent
```

A compelling result is:

```text
1. symbolic-grounded training
2. significantly improves autonomous NL-only ASL generation
3. relative to matched NL-only controls
4. especially on binding, relations, dependencies, and values
5. and the gain remains when external ASL is removed at inference.
```

Stronger evidence would additionally show a systematic advantage for shared/hybrid representations and/or merged K/V, with behavioral gains supported by representation and attention diagnostics.

---

# 29. Falsification criteria

Treat the central hypothesis as unsupported if, after adequate tuning:

```text
autonomous_score(mixed ASL training)
<=
autonomous_score(matched NL-only training)
```

Also count as negative or confounded evidence if:
- gains disappear under compute matching;
- random/scrambled teachers provide the same gain;
- improvement comes only from extra tokens;
- teacher-assisted success is mostly copying;
- shared/hybrid encoders improve alignment metrics but not autonomous behavior.

Negative results are publishable and should be preserved.

---

# 30. Immediate implementation order

```text
1. dataset + ASL canonicalizer + semantic evaluator
2. T1/T2/T3 runtime view generation
3. A1 separate encoders
4. M1 separate cross-attention
5. matched T3-only baseline
6. mixed-regime A1+M1 experiment
7. M2 merged K/V
8. A2 shared encoder
9. A3 hybrid encoder
10. representation diagnostics
11. optional consistency/alignment losses
12. multi-depth memories
```

The first milestone is not the complete matrix. It is to establish whether external symbolic grounding produces a measurable autonomous benefit.

For the constrained-compute pretrained track, run the following steps after the
T5-small B0/B1 gate. Complete any B2--B4 cells admitted by that gate before
starting Qwen so the encoder-decoder architecture evidence is not confounded by
simultaneous backbone changes:

```text
Q0. Reuse/freeze the existing Qwen3-0.6B F0 QKVO-r8 autonomous result.
Q1. Train a matched mixed-regime serialized-teacher LoRA control.
Q2. Add per-layer M1 external-memory side adapters and train LoRA + adapters.
Q3. Add per-layer M2 external K/V injection into native attention.
Q4. Compare autonomous behavior, teacher gap, wrong-teacher susceptibility,
    trainable parameters, peak memory, and wall time.
Q5. Confirm a positive cell with seeds 11, 23, and 37.
Q6. Only then test modality-specific versus shared/hybrid LoRA placement.
Q7. Only after a replicated gain consider Qwen3-1.7B.
```

## 31. Pretrained Qwen patch design

### 31.1 Preserve pretrained language competence

The Qwen track starts from the same pinned Qwen3-0.6B checkpoint already used by
F0. Natural-language input remains in the model's ordinary causal context. ASL
is an additional training-time memory source, not a replacement tokenizer or a
new language model objective.

Each cell must preserve:

```text
pretrained checkpoint revision
QKVO LoRA rank and alpha
F0 450/25/25 source identities
fixed prompt (no document-dependent ICL)
runtime T1/T2/T3 mixture
answer/rationale hiding boundary
autonomous NL-only test condition
```

### 31.2 Q1 serialized-teacher control

Q1 requires no attention patch. Serialize external ASL behind a fixed source
marker for T1/T2 examples and omit it completely for T3 examples. This is a
strong but conventional context-training baseline. It determines whether mixed
symbolic exposure can improve autonomous generation using LoRA alone.

Q1 must not be described as architectural evidence.

### 31.3 Q2 M1 side cross-attention patch

At selected Qwen layers, retain ordinary causal self-attention and add a
separately normalized external-ASL branch:

```text
h_local = NativeSelfAttention(Q, K_local, V_local)
h_asl   = CrossAttention(Q, K_asl, V_asl)
h       = h_local + sigmoid(g_asl) * P_asl(h_asl)
```

Initialize the ASL gate near zero so the initial model approximates the
pretrained checkpoint. Encode ASL with frozen shared Qwen blocks plus a small
trainable source projection, or with cached hidden states from those blocks.
Do not duplicate and train a second 0.6B backbone.

### 31.4 Q3 M2 native merged-K/V patch

At selected Qwen layers, project external ASL states into that layer's native
K/V geometry and concatenate them with local causal K/V before one softmax:

```text
K_total = concat(K_asl_ext, K_local)
V_total = concat(V_asl_ext, V_local)
h = NativeSelfAttention(Q, K_total, V_total)
```

Use source-type embeddings and an explicit visibility mask. Keep routing,
retrieval, persistent memory, and PRA runtime behavior out of this experiment.
The shared-softmax transport idea is informed by PRA, but Q3 is a local
training-time grounding patch.

### 31.5 Mapping A1/A2/A3 onto a frozen causal backbone

Do not create three trainable copies of Qwen. Interpret the encoder axis through
the source-adaptation path:

- `A1/separate`: ASL uses distinct lightweight source adapters; NL uses native
  Qwen states.
- `A2/shared`: NL and ASL use the same frozen Qwen blocks and shared LoRA/source
  adapters, with source-type markers.
- `A3/hybrid`: lower source adapters are modality-specific while upper LoRA and
  memory projections are shared/aligned.

This mapping preserves the conceptual sharing test while respecting the local
compute budget.

### 31.6 Required pretrained controls

Report at least:

```text
LoRA parameters
patch parameters
total trainable parameters
frozen backbone parameters
peak XPU memory
training wall time
teacher tokens processed
autonomous exact/semantic/answer metrics
```

Add a parameter-matched Q0/Q1 adapter when Q2 or Q3 has materially more trainable
capacity. A positive architectural result requires Q2 or Q3 to exceed both Q0
and Q1 autonomously, not merely under teacher-assisted evaluation.
