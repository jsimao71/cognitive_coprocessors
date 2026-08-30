# Paper 1 Q1 ASL Compiler Pilot

This is a diagnostic learning-signal experiment, not evidence for a robust
general semantic compiler. All 150 source mappings are execution-verified Q1
local-Codex records; none is Q4 manually adjudicated gold.

## Frozen split

- Seed: `731993`
- Train/dev/test: `100/25/25`
- Grouping: renaming-invariant CCIR topology, semantic quantity family, and
  clause layout
- Pattern groups: `92/24/23`
- Cross-split semantic-pattern overlap: zero
- Dataset mix: train `66 GSM8K / 34 TAT-QA`; dev and test each
  `17 GSM8K / 8 TAT-QA`
- Freeze manifest: `artifacts/paper1/asl_pilot_v1/freeze/freeze_manifest.json`

The test set was frozen before model runs. The 25- and 50-row training subsets
are nested in the 100-row training split.

## Conditions

- Base Qwen3-0.6B
- 5-shot and 10-shot ICL
- LoRA-A learning curve at 25, 50, and 100 original training programs
- LoRA-A-100 plus 3-shot ICL
- LoRA-B on 100 originals plus 900 controlled perturbations

LoRA uses rank 8 over `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The base model
revision, XPU dtype, seed, and optimizer settings are pinned in
`configs/paper1/asl_pilot_qwen_*_xpu.json`.

## Robustness suites

- Untouched original semantic programs: 25
- Safe numeric remapping: 20 eligible programs
- Safe very-large-number remapping: 20 eligible programs
- Natural-language paraphrase: 25 programs

Augmentation is train-only and execution-verified. It preserves percentage
parameters and year-like values. The leakage audit proves that augmented parents
come only from train and evaluation parents only from test.

## Metrics

Exact ASL is reported but does not define semantic correctness. The scorer also
reports parse validity, semantic lint, path grounding, source-fact extraction,
operator and reference/dependency accuracy, executable ASL, dependency
correctness, canonical return/state equivalence, and final-answer execution.
Canonical expression graphs tolerate harmless identifier renaming and normalize
commutative arithmetic.

## Reproduction

Set `PYTHONPATH=src`, then use `python -m ccpu paper1 freeze-asl-pilot` and
`build-asl-pilot-data` to rebuild local corpora from the accepted source. Run one
condition with `run-asl-pilot`; score an existing prediction file with
`evaluate-asl-pilot`. Generated corpora, predictions, and adapter weights are
ignored, while manifests, split IDs, summaries, and training reports are tracked.
