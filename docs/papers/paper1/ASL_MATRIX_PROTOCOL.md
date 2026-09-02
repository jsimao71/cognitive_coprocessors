# ASL Grounding Architecture Matrix Protocol

This track implements the staged ladder in `docs/AGENTS_ASL_3D_MATRIX_v2.md`
against the frozen F0 500-source partition. It is separate from F1/F2/F3 and
from the OpenRouter expansion corpus.

Two pretrained tracks are maintained. The first B ladder uses T5-small as a
clean encoder-decoder architectural control. After its B0/B1 grounding gate,
the Q ladder patches the existing Qwen3-0.6B F0 model with rank-8 LoRA and small
memory modules. Neither track trains language understanding from scratch.

## Fixed inputs

- Training: 450 F0 programs, 397 normalized semantic patterns.
- Development: 25 programs, used only for autonomous-loss checkpoint selection.
- Test: 25 untouched programs and 23 semantic patterns.
- Backbone: `google-t5/t5-small` at pinned revision
  `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`.
- Decoder, tokenizer, source lengths, target lengths, optimizer family, ordering,
  and seed are fixed within a matched comparison.

T5-small has six encoder and six decoder layers. The 12-layer dimensions in the
experiment brief are an illustrative schema, not a checkpoint requirement. A
small pretrained encoder-decoder is used because 450 examples cannot establish
natural-language competence for a Transformer trained from scratch and because
the local XPU can run the complete ladder.

## Axes

`A1/separate` uses two encoder block stacks with a shared token embedding.
`A2/shared` sends both sources through one encoder. `A3/hybrid` uses distinct
lower blocks and object-identical shared top blocks. The initial hybrid split is
four specialized plus two shared layers.

`M1/cross` replaces every decoder cross-attention module with two separately
normalized branches. The NL branch starts near the pretrained path; the ASL
branch has its own K/V/output projections and a learned gate. An unavailable
source is multiplied by an explicit zero availability mask.

`M2/merged_kv` concatenates typed NL and ASL encoder states and uses the original
T5 decoder cross-attention. Both sources therefore compete under one softmax.
This is analogous to PRA's native-K/V transport distinction, but no PRA code,
retrieval, routing, persistent memory, or runtime dependency is imported.

## Training views

Views are generated at runtime from one source dataset:

- T1: full external ASL;
- T2: deterministic record dropout, value masking, or argument masking;
- T3: NL only, with the external-ASL attention mask asserted to zero.

B0 and B1 instantiate exactly the same A1+M1 model. B0 trains only T3 while B1
uses T1=0.25, T2=0.35, T3=0.40. This keeps capacity fixed for the primary
grounding-gain test. Every epoch resamples regimes deterministically from the
run seed. Checkpoints are written after every epoch and selection uses autonomous
development loss only.

## Evaluation

Every completed model is evaluated on all 25 test identities under autonomous,
full-teacher, 20/50/80-percent corruption, teacher-only, and unrelated-teacher
conditions. The deterministic ASL/CCIR evaluator reports syntax, lowering,
typing, execution, semantic state/return equivalence, dependencies, and final
answer execution. Representation diagnostics include paired cosine, paired
retrieval, and layerwise linear CKA. Attention diagnostics are source-specific
for M1 and source-mass based for M2.

Single-seed ladder results are exploratory. A positive B1-B0 autonomous gain is
the gate for three-seed confirmation and later B2-B4 runs. Material architecture
effects require T3-only and parameter/compute controls before a causal claim.

Execution order is fixed: complete B0, B1, and any B2--B4 cells admitted by the
T5 gate before launching Q1--Q3. This preserves the lower-cost pretrained
encoder-decoder comparison and avoids simultaneous accelerator contention.

For Qwen, Q0 is the existing F0 QKVO-r8 result. Q1 adds mixed external-ASL
exposure as serialized context without changing attention. Q2 adds separately
normalized cross-attention side adapters, and Q3 injects external ASL K/V into
native self-attention. Q2/Q3 must beat Q1 autonomously before the gain can be
attributed to architecture rather than extra symbolic tokens.

## Commands

```powershell
python -m ccpu paper1 prepare-asl-matrix-data `
  --train artifacts/paper1/asl_pilot_500_v1/data/sft/train_450.jsonl `
  --dev artifacts/paper1/asl_pilot_500_v1/data/sft/dev.jsonl `
  --test artifacts/paper1/asl_pilot_v1/freeze/splits/test.jsonl `
  --output-dir artifacts/paper1/asl_matrix_v1/data

python -m ccpu paper1 train-asl-matrix `
  --config configs/paper1/asl_matrix_b0_t3_xpu.json `
  --data-dir artifacts/paper1/asl_matrix_v1/data `
  --output-dir artifacts/paper1/asl_matrix_v1/runs/b0_seed11

python -m ccpu paper1 evaluate-asl-matrix `
  --config configs/paper1/asl_matrix_b0_t3_xpu.json `
  --data-dir artifacts/paper1/asl_matrix_v1/data `
  --checkpoint artifacts/paper1/asl_matrix_v1/runs/b0_seed11/checkpoint_best.pt `
  --output-dir artifacts/paper1/asl_matrix_v1/eval/b0_seed11
```
