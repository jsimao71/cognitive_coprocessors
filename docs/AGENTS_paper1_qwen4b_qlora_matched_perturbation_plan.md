# Paper 1 Qwen3-4B QLoRA Matched Perturbation Plan

## Objective

Test whether additional model capacity improves NL-to-ASL semantic compilation
enough for deterministic execution to match or exceed direct neural reasoning as
numeric magnitude, value jitter, operator complexity, and expression depth rise.

The central comparison is within Qwen3-4B. Do not compare a trained 4B ASL route
against a smaller Direct model or against different test identities.

## Frozen model and data

- Base model: `Qwen/Qwen3-4B`.
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- ASL training data: the existing U2000/E4500 F0 corpus only.
- Training rows: 4,500 exposures from 2,000 unique GSM8K semantic programs.
- Development set: the existing 17 frozen development identities.
- Training seed: `99173`.
- Perturbation seed: `17011`.
- Decoding seed: `44017`.
- No magnitude, jitter, operator-ladder, complexity, or test-parent descendants
  may enter ASL training.

## Conditions

| ID | Condition | Purpose |
| --- | --- | --- |
| D0 | Base Qwen3-4B Direct, 4,096-token ceiling | Primary ordinary neural-reasoning route |
| D0-2K | Base Qwen3-4B Direct, 2,048-token ceiling | Token-budget sensitivity control |
| A0 | Base Qwen3-4B zero-shot ASL plus runtime | Untrained representation control only |
| A1 | Qwen3-4B U2000/E4500 QKVO-r8 QLoRA ASL plus runtime | Primary CogCop route |
| D1 | Budget-matched Qwen3-4B Direct-answer QLoRA | Supervision-budget fairness control |

A0 must never be presented as the main ASL result. The primary architecture
comparison is D0 versus A1. D1 is required before making a definitive claim that
the difference comes from delegation rather than task-specific supervision.

## QLoRA training gate

Use the existing configuration
`configs/paper1/e3_g1_gsm8k_u2000_e4500_f0_l0_qwen4b_qlora_cuda.json`:

- NF4 4-bit base weights with double quantization;
- FP16 compute;
- QKVO LoRA, rank 8, alpha 16, dropout 0.05;
- batch size 1 and gradient accumulation 8;
- gradient checkpointing;
- 1,152-token maximum with truncation rejected;
- ten logical epoch views totaling exactly 4,500 exposures;
- optimizer checkpoint every 25 optimizer steps.

Run the eight-train/two-dev smoke first. The full run may begin only when:

1. all smoke losses and gradients are finite;
2. forward, backward, optimizer step, adapter save, and adapter reload pass;
3. no example is truncated;
4. peak allocated and reserved GPU memory are recorded;
5. the reconstructed adapter provenance matches model revision, corpus hashes,
   target modules, rank, seed, and optimizer settings.

The preferred host is the 8 GB RTX 5060. A 4 GB GPU may run the smoke but must
not silently reduce sequence length, change precision, or skip long records.

## Frozen perturbation cells

All routes consume the exact same frozen IDs and question hashes from
`configs/paper1/qwen4b_perturbation_schedule_v1.json`.

Primary confirmation cells:

- ordinary original;
- magnitude x1, x100, x1000, x10000, and x1000000;
- operator O1 and O6;
- independent uniform `[-30%, +30%]` jitter at x1 and x1000;
- mixed x1/O1 and x1000/O6;
- expression complexity C1 and C4.

Secondary diagnostic cells:

- operators O3 and O5;
- the other four operator-by-jitter mixtures;
- expression complexity C2 and C3.

The primary list is frozen before A1 is trained. Secondary cells are reported as
diagnostics and cannot replace a failed primary cell.

## Sample-size ladder

Each cell contains 100 identities split into ten disjoint modulo shards.

1. Complete shard 0 for every D0 and A1 cell: `n=10` early read.
2. Complete shard 1 for every primary D0 and A1 cell: `n=20` stability read.
3. Complete shards 2 through 9 for every primary D0 and A1 cell: `n=100`
   confirmation.
4. Expand secondary cells beyond `n=20` only as explicitly exploratory work or
   after all primary cells are complete.
5. Run D0-2K on original, x1000, O6, x1000/O6, and C4 using the same identities.
6. Run D1 on all primary cells after its own memory/provenance smoke.

Do not expand only cells whose first ten examples favor ASL. The n=20 and n=100
decisions are fixed by cell class, not observed direction.

## Metrics

For D0, D0-2K, and D1 report:

- exact final-answer accuracy;
- strict endpoint and scorer-v2 accuracy;
- generated tokens, ceiling hits, latency, and peak memory;
- semantic/structure, arithmetic, formatting, and no-scorable-value failures.

For A0 and A1 report each boundary separately:

- parseable;
- lowerable to CCIR;
- type-valid;
- executable;
- runtime final-answer correct;
- post-reinjection final-answer correct when reinjection is enabled;
- semantic selection/binding error;
- literal-copy error and oracle-literal-correction result;
- generated ASL tokens, runtime latency, total latency, and peak memory.

The headline comparison uses runtime final-answer correctness for A1 and final
answer correctness for D0/D1. Never count parseability or execution alone as a
correct ASL answer.

## Analysis

- Join routes by exact example ID and question hash.
- Report paired deltas and exact McNemar tests per primary cell.
- Report Wilson intervals and paired bootstrap intervals for accuracy deltas.
- Plot accuracy and generated tokens against `log10(magnitude)`.
- Plot Direct-minus-ASL accuracy over operator, magnitude, jitter, and complexity.
- Report the crossover point separately for 0.6B, 1.7B, and 4B.
- Decompose A1 failures into compilation versus deterministic-runtime failures.
- Decompose Direct failures into semantic, arithmetic, endpoint, and ceiling
  failures.

## Interpretation gates

- If A1 improves over the smaller ASL adapters but remains below D0 everywhere,
  model scale improved compilation without establishing a 4B crossover.
- If D0 wins ordinary cells but A1 wins high-magnitude, high-operator, or mixed
  cells, report a regime-dependent crossover rather than universal dominance.
- If D1 removes the A1 advantage, attribute the earlier difference partly to
  supervision rather than delegation.
- If A1 has correct structure but literal-copy failures dominate, prioritize
  constrained literal grounding instead of a larger adapter.
- If A1 remains primarily binding-limited, do not infer that QLoRA rank or CPU
  execution is the bottleneck without a matched semantic diagnostic.

## Execution order

```text
finish D0 shard-0 schedule
        -> QLoRA memory smoke
        -> train A1 U2000/E4500 QKVO-r8
        -> A1 shard 0 across all cells
        -> D0/A1 primary shard 1
        -> D0/A1 primary shards 2-9
        -> D0-2K controls
        -> train/evaluate D1 fairness control
        -> replicate material interactions with adapter seeds 99189 and 99203
```

Commit and push after the smoke, full training, shard-0 comparison, n=20 primary
comparison, and n=100 primary comparison. Build the paper after each completed
comparison tier, not after every individual shard.
