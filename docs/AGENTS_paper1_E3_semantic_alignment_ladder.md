# AGENTS — Paper 1 E3* Semantic-Convergence Architecture
## Encoder-Only Preconditioning, NL↔ASL Alignment, and Causal ASL Generation

## Status

Future Paper 1 experiment family.

Run after:
- Q0–Q3 matrix is frozen;
- the selected ASL/F3 target representation is sufficiently stable;
- the 450/25/25 Paper 1 split remains frozen;
- the Qwen3-0.6B autonomous Q0 baseline remains reproducible;
- the current Qwen external-memory path is validated.

Do NOT wait for the OpenRouter F0-large corpus to begin E3*.

The OpenRouter/F0-large experiment is a separate **data-scale** axis and should continue independently.

E3* tests the **representation/architecture/training** axis.

---

# 1. Motivation

Current Qwen3-0.6B matrix:

```text
Q0 plain autonomous F0 LoRA        44% final
Q1 cross-memory mixed              24%
Q2 higher-capacity memory variant  40%
Q3 native/merged-KV mixed          40%
```

No ASL-memory condition beats Q0.

Q3 is important because it achieves approximately Q2 performance with almost the same trainable parameter count as Q0, showing that external ASL can be transported cheaply but does not yet improve autonomous NL→ASL compilation.

Interpretation:

> Memory transport is not currently the main missing capability. The representation produced when encoding NL and ASL may be insufficiently aligned.

E3* therefore tests whether NL and external ASL can be trained to converge toward a common upper semantic/world representation before and during causal ASL generation.

---

# 2. Core hypothesis

There are three representational roles:

```text
NL                 natural-language evidence
ASL_ext            external symbolic description of the world
ASL_int            internally generated symbolic world representation
```

Operationally:

```text
ASL_ext and ASL_int use the same symbolic language.
```

The desired architecture is:

```text
NL  -> E_N -> E_NA -> Z_N
ASL -> E_A -> E_NA -> Z_A
```

where:
- `E_N` is NL-specific lower/mid processing;
- `E_A` is ASL-specific lower/mid processing;
- `E_NA` is a shared upper semantic/world encoder.

Main representational hypothesis:

```text
Z_N(world_i) ≈ Z_A(world_i)
```

for paired NL and ASL descriptions of the same world/problem, while:

```text
Z_N(world_i) != Z_A(world_j)
```

for unrelated worlds.

The goal is NOT to make all raw hidden states identical.

The goal is to create a shared semantic subspace useful for:
- entity identity;
- attributes;
- relations;
- argument roles;
- dependencies;
- query targets;
- world state.

---

# 3. Architectural principle

The bottom/middle layers should remain modality-specific.

Canonical hybrid:

```text
NL  -> E_N lower/mid  --\
                         > E_NA shared semantic top -> Z_N / KV_N_world
ASL -> E_A lower/mid --/
                                             -> Z_A / KV_A_world
```

Do not use shared lower layers in the primary E3 design.

Rationale:
- NL surface syntax is noisy/ambiguous/contextual;
- ASL surface syntax is regular/explicit;
- forcing both through identical low-level adaptation may cause negative transfer;
- the commonality should emerge at the semantic/world level.

---

# 4. Implementation strategy on Qwen

Do NOT initially build two full new Transformers.

Approximate E3 using one pretrained Qwen backbone plus adapter banks.

Suggested adapter sets:

```text
LoRA_N   = active on NL-specific lower/mid layers
LoRA_A   = active on ASL-specific lower/mid layers
LoRA_NA  = shared semantic adapter on upper layers
LoRA_D   = decoder/generation adapter if needed
```

Where possible reuse Q0/F0 decoder adaptation rather than retraining an unrelated generation path.

Keep base Qwen frozen in the first E3 sweep.

---

# 5. Suggested layer partition

Expose configurable split points.

Example for a 28-layer backbone:

```text
layers 0..13:
  modality-specific
  NL pass  -> LoRA_N
  ASL pass -> LoRA_A

layers 14..27:
  shared semantic
  both passes -> LoRA_NA
```

Do not hard-code this split.

Test a bounded set later:

```text
25% specific / 75% shared
50% specific / 50% shared
75% specific / 25% shared
```

Do not sweep split ratios until E3b shows a signal.

---

# 6. E3 experiment ladder

Keep the following variants distinct.

```text
E3a  hybrid architecture, generation loss only
E3b  E3a + encoder-only global contrastive pretraining
E3c  E3b + component/role-level alignment
E3d  E3c + masked/corrupted ASL reconstruction pretraining
E3e  joint causal generation + weak semantic alignment
```

Each variant should be compared against:
- Q0;
- Q3;
- previous E3 variant.

---

# 7. E3a — Hybrid architecture, no explicit alignment

Purpose:

> Establish whether modality-specific lower adapters plus shared semantic upper adapters help even without explicit contrastive alignment.

Architecture:

```text
NL:
Qwen base
+ LoRA_N(lower/mid)
+ LoRA_NA(upper)

ASL_ext capture:
Qwen base
+ LoRA_A(lower/mid)
+ LoRA_NA(upper)
```

Decoder uses the current best memory transport initially:

```text
Q3 native/merged K/V
```

Keep:
- same dataset;
- same training mixture;
- same prompt;
- same decoder;
- same rank;
- same test split.

This isolates encoder specialization.

---

# 8. E3b — Encoder-only global contrastive preconditioning

This is the first major new experiment.

Before causal ASL generation, train only the encoding pathway.

Use complete paired views:

```text
NL_i
ASL_i
```

Both are encoded bidirectionally / non-causally.

No autoregressive decoder is needed in this phase.

Compute:

```text
Z_N_i = Pool(E_NA(E_N(NL_i)))
Z_A_i = Pool(E_NA(E_A(ASL_i)))
```

Then optimize a paired contrastive objective.

Preferred first objective:

```text
InfoNCE
```

Conceptually:

```text
same example:
  sim(Z_N_i, Z_A_i) -> high

different examples:
  sim(Z_N_i, Z_A_j) -> low
```

Do not use raw unregularized MSE as the only objective because of collapse risk.

---

# 9. E3b pooling

Support:

```text
mean pooling
attention pooling
special semantic anchor token
```

First run:
- mean pool non-padding encoder states;
- no learned pooling network unless necessary.

The purpose is to test the hypothesis, not optimize pooling.

---

# 10. E3b negative construction

Use in-batch negatives first.

A positive pair:

```text
NL_i <-> ASL_i
```

Negatives:

```text
NL_i <-> ASL_j, j != i
```

Avoid false negatives where two examples are semantically equivalent.

Use semantic-pattern metadata to exclude known equivalent/near-equivalent negatives where possible.

Log:
- batch false-negative filtering;
- unique semantic signatures;
- effective negative count.

---

# 11. E3b encoder masking

During encoder-only alignment:

```text
NL encoder self-attention:
  full/bidirectional within the NL example

ASL encoder self-attention:
  full/bidirectional within the ASL example
```

No causal mask is needed because:
- both views are fully observed;
- this phase is representation learning, not generation.

This is intentionally closer to classical Transformer encoding / BERT-like representation learning than decoder-only LM training.

---

# 12. E3b preconditioning schedule

Recommended sequence:

```text
Stage 1:
train LoRA_N, LoRA_A, LoRA_NA
with contrastive objective only

Stage 2:
freeze E_N/E_A/E_NA
train/reuse ASL decoder on NL -> ASL

Stage 3:
unfreeze LoRA_NA + decoder
fine-tune NL -> ASL

Stage 4:
optionally unfreeze all adapters
with weak alignment regularization
```

Do not immediately jointly optimize everything.

The point is to determine whether semantic-space preconditioning helps.

---

# 13. E3b primary comparison

Critical matched experiment:

```text
E3a:
same hybrid architecture
random/unpreconditioned adapters
-> causal NL->ASL training

E3b:
same hybrid architecture
contrastive encoder preconditioning
-> identical causal NL->ASL training
```

If E3b > E3a autonomously, that is the cleanest evidence for semantic preconditioning.

---

# 14. E3c — Component / role-level alignment

Global example alignment may only capture topic/world similarity.

Current failures are finer:
- entity binding;
- path selection;
- relation selection;
- argument role;
- reference/coreference;
- dependency;
- query target.

E3c therefore aligns semantic components.

Possible annotations:

```text
NL span                    ASL component
------------------------------------------------
"John"                     john
"pink hats"                hats.pink
"removes"                  remove
"6"                        quantity=6
"twice"                    multiplier=2
"that many"                e1.quantity
"green hats"               hats.green
```

Do not require every example to have full token-level alignments initially.

Start with automatically derivable/high-confidence alignments.

---

# 15. E3c component losses

For every aligned component pair:

```text
z_nl_component
z_asl_component
```

maximize similarity relative to unrelated components.

Use contrastive rather than only pointwise MSE.

Component types:

```text
ENTITY
PATH
ATTRIBUTE
RELATION
ROLE
VALUE
REFERENCE
QUERY
```

Track metrics by component type.

---

# 16. E3c hard semantic negatives

Generate deterministic hard negatives from ASL.

Examples:

## Direction swap

Correct:
```text
older_than(jessica.age, claire.age, 6)
```

Negative:
```text
older_than(claire.age, jessica.age, 6)
```

## Attribute swap

Correct:
```text
company.acquisition_cost@2019
```

Negative:
```text
company.total_cost@2019
```

## Role swap

Correct:
```text
remove(john, hats.pink, 6)
```

Negative:
```text
remove(hats.pink, john, 6)
```

## Reference swap

Correct:
```text
2 * e1.quantity
```

Negative:
```text
2 * e0.quantity
```

Use only semantically guaranteed-invalid transformations.

---

# 17. E3d — Masked / corrupted ASL reconstruction pretraining

E3d adds a second encoder-pretraining signal.

Input:

```text
NL
+
corrupted ASL_ext
```

Target:
- masked ASL components or records.

Examples:

```text
remove(john, ?, 6)
```

recover:
```text
hats.pink
```

or:

```text
older_than(?, claire.age, 6)
```

recover:
```text
jessica.age
```

This explicitly trains NL evidence to fill missing symbolic structure.

---

# 18. E3d corruption policies

Reuse/adapt existing Paper 1 corruption operators:

```text
record_dropout
value_mask
argument_mask
entity_mask
attribute_mask
relation_mask
reference_mask
query_mask
```

Do not use all at once initially.

Start with:

```text
argument_mask
value_mask
record_dropout
```

then add semantically targeted masks.

---

# 19. E3d reconstruction architecture

Two acceptable implementations:

## D1 — small reconstruction head

```text
encoded NL
+
encoded corrupted ASL
-> reconstruction head
-> masked component
```

## D2 — causal ASL decoder

Use the normal ASL decoder to reconstruct the missing ASL.

Prefer D1 initially if it isolates encoder learning more cleanly.

Do not let the reconstruction task dominate later causal generation.

---

# 20. E3e — Joint generation + weak alignment

After preconditioning, train autonomous NL→ASL generation.

Total loss:

```text
L =
L_generation
+ lambda_global * L_global_align
+ lambda_component * L_component_align
```

Optional later:

```text
+ lambda_consistency * L_teacher_student
```

Start with small alignment weights.

The objective is to preserve the common world space, not prevent the encoder from adapting to generation.

---

# 21. Progressive alignment across upper layers

Do not necessarily align only the last layer.

For shared semantic layers:

```text
l1 < l2 < ... < lk
```

define:

```text
L_align =
sum_l lambda_l * D(Z_N^l, Z_A^l)
```

with increasing weights toward the top.

Example:

```text
first shared layer: 0.25
middle shared:      0.50
top shared:         1.00
```

Hypothesis:

```text
lower modality-specific states:
  different

early shared semantic states:
  partially converged

top shared semantic states:
  strongly converged
```

---

# 22. Do not force literal hidden-state equality

The target is NOT:

```text
H_N == H_A
```

for every token/dimension.

Use:
- contrastive geometry;
- retrieval;
- semantic probes;
- projected alignment.

Allow modality-specific residual information to survive.

If raw MSE is tested, treat it as an ablation.

---

# 23. Representation diagnostics

For every E3* condition measure layerwise:

```text
paired cosine
linear CKA
SVCCA if practical
paired NL->ASL retrieval accuracy
paired ASL->NL retrieval accuracy
linear probe accuracy
```

For hybrid architecture mark:

```text
E_N/E_A modality-specific layers
E_NA shared semantic layers
```

Plot whether alignment increases across depth.

---

# 24. Behavioral diagnostics remain primary

Do not declare success because CKA/cosine improves.

Autonomous evaluation remains:

```text
NL only -> ASL_int
```

Report:
- parse;
- lowerable;
- executable;
- final answer;
- semantic return;
- semantic state;
- dependency;
- source facts;
- paths;
- operators;
- attribute F1;
- relation/role binding where available.

---

# 25. Core expected result patterns

## P1 — Alignment + behavioral improvement

Best case:

```text
E3b/c > E3a
```

on semantic state, dependency, binding, and autonomous answer.

Strong support for common-world preconditioning.

## P2 — Alignment improves, behavior flat

Then:
- shared geometry is insufficient;
- decoder may not consume it;
- inspect memory/attention;
- consider stronger component objectives.

## P3 — Teacher-assisted improves, autonomous flat

Internalization failure.

Try:
- teacher/student KL;
- more autonomous T3;
- progressive teacher dropout;
- stronger shared-top alignment.

## P4 — No alignment and no behavior gain

Revisit:
- adapter capacity;
- layer split;
- pooling;
- ASL representation.

Do not jump immediately to larger model.

---

# 26. Teacher/student consistency as optional E3f

Only after E3b–E3e.

For the same example:

```text
P_teacher = p(ASL | NL, ASL_ext)
P_auto    = p(ASL | NL)
```

Add:

```text
KL(stopgrad(P_teacher) || P_auto)
```

Purpose:

> Make autonomous decoding imitate decisions supported by externally grounded symbolic memory.

Do not mix this into the first E3 preconditioning result.

---

# 27. Training data

Initial E3* uses the frozen ~450 training programs.

Do NOT use F0-large initially.

Reason:
- isolate representation/training effect;
- maintain direct comparability to Q0–Q3;
- avoid confounding more data with better architecture.

Later compare:

```text
Q0-450
E3*-450
F0-large
```

---

# 28. Train/dev/test discipline

Keep:

```text
train ~450
dev 25
test 25
```

with exact frozen IDs/signatures.

Do not change test examples.

Checkpoint selection:
- dev only;
- preferably autonomous dev semantic/generation objective;
- never test.

---

# 29. Pretraining dataset views

Every source record can generate:

```text
NL view
ASL view
paired NL/ASL view
corrupted ASL view
component pairs
hard negatives
```

Store provenance.

---

# 30. Parameter accounting

For every run record:

```text
base frozen parameters
LoRA_N parameters
LoRA_A parameters
LoRA_NA parameters
decoder LoRA parameters
alignment-head parameters
projection parameters
total trainable
```

Q0/Q3 parameter controls remain mandatory.

---

# 31. Capacity-matched controls

If E3* has substantially more trainable parameters than Q0:

run a capacity-matched Q0 control using:
- higher rank;
- more target modules;
- or extra MLP LoRA.

Do not attribute a gain to architecture if parameter count alone could explain it.

---

# 32. Compute-matched controls

Where practical match:
- source examples;
- target exposures;
- optimizer steps;
- total tokens;
- approximate FLOPs.

Encoder-only preconditioning adds compute.

Report it explicitly.

---

# 33. Recommended first E3 sweep

Run:

```text
R0 = Q0 frozen reference
R1 = Q3 frozen reference
R2 = E3a
R3 = E3b
```

Question:

> Does encoder-only contrastive preconditioning improve the same hybrid architecture?

If R3 does not outperform R2 in meaningful autonomous semantic metrics, stop before E3c/d.

---

# 34. Second sweep

If E3b shows signal:

```text
R4 = E3c component alignment
R5 = E3d masked symbolic reconstruction
```

Keep architecture fixed.

Do not alter layer split simultaneously.

---

# 35. Third sweep

If E3c/d helps:

```text
R6 = E3e joint weak alignment during causal generation
```

Then test:
- alignment weights;
- one or two layer split alternatives.

---

# 36. Fourth sweep

Only after a positive E3 result:

repeat best E3* on:

```text
Qwen3-1.7B
```

Do not move immediately to 4B.

---

# 37. F0-large comparison

When OpenRouter corpus freezes:

train/evaluate:

```text
F0-large
```

Then compare:

```text
Q0-450
best E3*-450
F0-large
```

This tests:

```text
better architecture/representation
vs
much more supervision
```

Potential later:

```text
best E3*-large
```

only if E3*-450 is positive.

---

# 38. Key semantic metrics

Prioritize:

```text
attribute_f1
path_f1
source_fact_f1
dependency_f1
relation_class_accuracy
argument_role_accuracy
exact_binding_accuracy
query_target_accuracy
semantic_state_equivalence
final_answer
```

---

# 39. Layerwise convergence experiment

For E3b/c, produce a plot:

```text
x-axis: layer/depth
y-axis:
  paired cosine
  CKA
  paired retrieval
```

Expected if hypothesis holds:

```text
E_N/E_A region:
  low alignment

E_NA region:
  increasing alignment

top:
  highest semantic alignment
```

Do not force this expected shape in analysis.

---

# 40. Semantic retrieval diagnostic

Test:

```text
given NL_i:
retrieve matching ASL_i
```

and inverse:

```text
given ASL_i:
retrieve matching NL_i
```

Report top-1/top-k.

---

# 41. Component retrieval diagnostic

For E3c:

```text
given NL span "that many":
retrieve ASL component e1.quantity
```

and:

```text
given ASL relation remove:
retrieve corresponding NL predicate span
```

Report by component class.

---

# 42. Preconditioning checkpoint transfer

Store encoder-only checkpoints separately.

Compare:

```text
preconditioned encoder + same decoder setup
```

against:

```text
unpreconditioned encoder + same decoder setup
```

This establishes whether the encoder itself carries transferable semantic structure.

---

# 43. Freeze/unfreeze ablation

For E3b:

```text
A:
freeze preconditioned encoders during decoder training

B:
unfreeze only E_NA

C:
unfreeze all adapters
```

This tests whether causal fine-tuning destroys useful alignment.

---

# 44. Alignment drift

Measure before and after causal generation training:

```text
paired cosine
CKA
retrieval
component alignment
```

If generation fine-tuning destroys alignment:
- lower encoder LR;
- freeze lower adapters;
- retain weak alignment regularizer;
- use separate optimizer groups.

---

# 45. Optimizer groups

Support separate learning rates:

```text
lr_N
lr_A
lr_NA
lr_decoder
lr_alignment_head
```

Recommended initial strategy:
- lower LR for shared semantic layers during joint fine-tuning;
- standard LoRA LR for decoder;
- alignment head may use higher LR.

---

# 46. Non-causal encoder implementation

For Qwen-based reuse, explicitly choose:

## B1 — remove causal mask during capture/alignment pass

Use Qwen blocks bidirectionally for encoder-only preconditioning.

## B2 — dedicated encoder-mode wrapper

Reuse Qwen weights but construct encoder-mode attention.

## B3 — small dedicated encoder

Only later if B1/B2 are unstable.

Prefer B1/B2 first to maximize pretrained reuse.

Record clearly that encoder-mode attention differs from ordinary Qwen causal inference.

---

# 47. Positional treatment

If Qwen/RoPE is used non-causally:
- preserve consistent positions;
- verify full-attention RoPE behavior;
- test padding/mask correctness;
- no future-leakage concern exists because preconditioning sees the full example by design.

Do not reuse inference-cache assumptions in encoder-only mode.

---

# 48. Generation phase

During causal NL→ASL generation:

```text
NL source encoding:
  may be bidirectional if full source block is available

ASL decoder:
  causal
```

This is classical encoder-decoder behavior.

Do not unnecessarily force source encoding to be causal for short complete benchmark problems.

---

# 49. Relation to classical translation

E3* is structurally related to:

```text
source language
-> bidirectional encoder
-> cross-attended causal decoder
-> target language
```

But the research hypothesis differs:

```text
NL and ASL are two descriptions of the same world,
and ASL is also the desired internal cognitive language.
```

Do not claim basic encoder-decoder attention as novel.

---

# 50. Relation to external/internal grounding

During preconditioning:

```text
ASL_ext -> E_A -> E_NA -> Z_world
NL      -> E_N -> E_NA -> Z_world
```

During inference:

```text
NL -> E_N -> E_NA -> Z_internal
                 -> ASL_int
```

Empirical hypothesis:

> External symbolic world descriptions can serve as a training-time anchor for an internal symbolic world representation reconstructed from NL.

---

# 51. Relation to long-context architecture

If E3* works, later long-context system can use:

```text
NL block -> E_N -> E_NA -> source/world semantic memory

previous ASL/world records
-> E_A -> E_NA -> world memory

ASL decoder
cross/merged attention over selected memories
```

Do not include long-context block detection in E3* Paper 1 experiments.

---

# 52. Required artifacts

Suggested:

```text
docs/papers/paper1/AGENTS_E3_semantic_alignment.md

src/ccpu/paper1/e3/
  adapters.py
  encoder_mode.py
  contrastive.py
  alignment.py
  components.py
  corruption.py
  pretrain.py
  train_generation.py
  evaluate.py
  diagnostics.py

configs/paper1/e3/
  e3a.yaml
  e3b.yaml
  e3c.yaml
  e3d.yaml
  e3e.yaml

artifacts/paper1/e3_v1/
  manifests/
  pretraining/
  checkpoints/
  generation/
  eval/
  representation/
  analysis/
```

---

# 53. Required tests

Add deterministic tests for:
- modality-specific adapter activation;
- shared upper adapter activation;
- no accidental LoRA_N/LoRA_A overlap;
- non-causal encoder mask;
- paired batch construction;
- false-negative filtering;
- contrastive-loss correctness;
- component alignment;
- corruption validity;
- checkpoint freeze/unfreeze behavior;
- test split isolation;
- autonomous evaluation has no `ASL_ext`.

---

# 54. Leakage gate

Autonomous test must contain:

```text
NL only
```

No:
- external ASL memory;
- teacher embeddings;
- ASL-derived hidden states;
- alignment annotations;
- component mappings;
- corruption metadata.

Add runtime assertions.

---

# 55. Primary table

Final E3 table:

```text
Q0
Q3
E3a
E3b
E3c
E3d
E3e
```

Columns:

```text
trainable params
pretraining compute
generation compute
parse
executable
source facts
paths
relations
roles
binding
dependency
semantic state
final answer
paired retrieval
top-layer CKA
```

---

# 56. Go / no-go gates

## Gate 1 — E3a

If E3a is substantially worse than Q3/Q0, inspect adapter/layer routing before alignment experiments.

## Gate 2 — E3b

Proceed to component alignment only if E3b improves at least one meaningful autonomous semantic metric over E3a without catastrophic final-answer loss.

## Gate 3 — E3c/d

Proceed to joint fine-tuning if component/reconstruction objectives improve binding/dependency or clear diagnostics.

## Gate 4 — Larger model

Only repeat on 1.7B after a credible 0.6B signal.

## Gate 5 — Large corpus

Only train E3-large if E3*-450 shows useful representation or sample-efficiency gains.

---

# 57. Falsification

The common-world preconditioning hypothesis is weakened if:

```text
E3b <= E3a
```

across autonomous semantic metrics after matched training.

It is further weakened if:
- alignment metrics rise sharply but behavior does not;
- random/scrambled pairings give similar gains;
- capacity-matched Q0 explains the gain;
- preconditioning only improves teacher-assisted conditions;
- causal fine-tuning erases all representation benefits.

Negative results must be preserved.

---

# 58. Immediate implementation order

```text
P0  Freeze Q0/Q3 references and manifests.

P1  Implement modality-specific adapter routing:
    LoRA_N / LoRA_A / LoRA_NA.

P2  Implement E3a generation-only baseline.

P3  Implement bidirectional/non-causal encoder capture mode.

P4  Implement paired global contrastive pretraining.

P5  Train E3b encoder-only on frozen 450 train examples.

P6  Measure layerwise convergence/retrieval before generation.

P7  Train E3a and E3b with identical causal NL->ASL schedule.

P8  Evaluate autonomous 25-test set.

P9  If E3b signal is positive, add component alignment (E3c).

P10 Add masked/corrupted symbolic reconstruction (E3d) only after E3c audit.

P11 Add joint weak alignment during generation (E3e).

P12 Compare against Q0/Q3 and capacity controls.

P13 When F0-large freezes, compare best E3*-450 against F0-large.

P14 Patch Paper 1 with representation-vs-data-vs-architecture conclusions.
