# Dataset Registry

This file records dataset roles, frozen counts, provenance, and leakage policy
for the cognitive-coprocessor papers. Counts here refer to local frozen
artifacts unless explicitly marked as approximate upstream sizes.

## Paper 1 Scope

Paper 1 now studies semantic compilation for simple arithmetic word problems.
All newly created Paper 1 training, development, historical-test,
confirmatory, and robustness artifacts must be arithmetic-only.

- The immediate dataset is GSM8K only.
- TAT-QA is retired from all new Paper 1 work.
- Legacy mixed GSM8K/TAT-QA artifacts remain immutable provenance.
- A legacy mixed checkpoint must never be described as GSM8K-only.
- New dataset and checkpoint IDs must identify their corpus scope.

Table/document retrieval belongs to the retrieval and heterogeneous-runtime
papers rather than the forward Paper 1 experiment ladder.

## Current Paper 1 Inventory

### Upstream materialization

| Dataset and upstream split | Local source rows | Current role |
|---|---:|---|
| GSM8K `train` | 7,473 | Annotation source and current training universe |
| TAT-QA `development`, arithmetic-compatible subset | 706 | Legacy provenance only; retired from Paper 1 |
| **Total OpenRouter source requests** | **8,179** | Historical annotation campaign |

The current OpenRouter campaign did not materialize or annotate the official
GSM8K test split. That split remains the intended source for the larger
untouched confirmatory evaluation.

Source manifests:

- `artifacts/paper1/dsl/openrouter_full_v1/inputs/gsm8k_full.manifest.json`
- `artifacts/paper1/dsl/openrouter_full_v1/inputs/tatqa_arithmetic_full.manifest.json`

### Final OpenRouter strict corpus

"Strict-valid" means that the generated ASL passed deterministic syntax,
lowering, type, execution, semantic-lint, and final-answer checks. It does not
mean that every symbolic path was manually reviewed or that strict-valid rows
form an untouched benchmark test set.

| Dataset | Source rows | Strict-valid ASL | Strict yield |
|---|---:|---:|---:|
| GSM8K `train` | 7,473 | 6,569 | 87.9% |
| TAT-QA `development` | 706 | 480 | 68.0% |
| **Total** | **8,179** | **7,049** | **86.2%** |

Final corpus:

- `artifacts/paper1/dsl/openrouter_full_v1/recovery_v2/consolidated_run3/combined_strict.jsonl`
- `artifacts/paper1/dsl/openrouter_full_v1/recovery_v2/consolidated_run3/summary.json`

The final strict corpus is annotation material. Membership in it does not by
itself assign a record to training, development, or test.

### Legacy D1 mixed training freeze

D1 was frozen before the final recovery additions. It trained all three
reported seeds on the same 4,500 unique programs:

| Dataset | D1 training rows |
|---|---:|
| GSM8K | 4,044 |
| TAT-QA | 456 |
| **Total** | **4,500** |

D1 excluded the then-frozen development/test identities and protected test
semantic patterns. Its manifest reports zero selected source-ID overlap and
zero protected-pattern overlap. D1 is historical mixed-data evidence and is
not the forward Paper 1 corpus.

- Manifest: `artifacts/paper1/e3_v2/d1_f0_v1/manifest.json`
- Training rows: `artifacts/paper1/e3_v2/d1_f0_v1/train.jsonl`
- Replication summary:
  `artifacts/paper1/e3_v2/d1_f0_v1/replications/aggregate/summary.json`

Relative to the final strict corpus, 2,549 strict-valid annotations were not
used by D1: 2,525 GSM8K and 24 TAT-QA. These are unused annotation candidates,
not an official public test set.

### G1_GSM8K forward training freeze

`G1_GSM8K` is the first forward GSM8K-only corpus. The builder reads the final
strict annotation corpus but rejects every non-GSM8K annotation at the data
boundary.

| Stage | Count |
|---|---:|
| Final strict corpus input | 7,049 |
| Non-GSM8K annotations scope-rejected | 480 |
| Strict GSM8K programs | 6,569 |
| Frozen-identity or protected-pattern exclusions | 29 |
| Eligible GSM8K programs | 6,540 |
| Eligible semantic signatures | 6,375 |
| Selected training programs | 4,500 |
| Selected semantic signatures | 4,500 |
| Eligible programs outside training | 2,040 |

The 2,040 records outside G1 training are an internal reserve. They have not
been registered as a confirmatory test and must not be selected post hoc based
on model behavior.

- Manifest: `artifacts/paper1/gsm8k_scale_v1/g1_f0_4500/manifest.json`
- Training rows: `artifacts/paper1/gsm8k_scale_v1/g1_f0_4500/train.jsonl`
- Eligible pool: `artifacts/paper1/gsm8k_scale_v1/g1_f0_4500/eligible.jsonl`
- Exclusions: `artifacts/paper1/gsm8k_scale_v1/g1_f0_4500/excluded.jsonl`

#### Exposure-matched scaling cells

The first learning-curve derivative is `G1_GSM8K_U2000_E4500`. It selects
2,000 unique source IDs and 2,000 semantic signatures from the frozen eligible
pool, then creates exactly 4,500 deterministic training exposures. Every source
appears two or three times, and each of ten logical epochs contains 450 rows.
This matches G1-4500 on row exposures and optimizer steps while changing unique
semantic diversity.

- Manifest: `artifacts/paper1/gsm8k_scale_v1/u2000_e4500/manifest.json`
- Training stream: `artifacts/paper1/gsm8k_scale_v1/u2000_e4500/train.jsonl`

All three declared runs are complete on `TEST-GSM17`. U2000/E4500 reaches
11, 6, and 6 answers across seeds 11, 23, and 37 (mean 7.67/17, 45.1%; range
35.3--64.7%), versus 6, 7, and 6 for same-seed G1-4500. The corresponding
same-seed answer deltas are therefore +5, -1, and 0: the seed-11 gain does not
replicate uniformly. Mean alpha-state F1 is .620, .571, and .572, exceeding
the corresponding G1 values in every seed, so repetition has a more consistent
soft-semantic than final-answer effect. This small historical test cannot select
the final condition; the official confirmation freeze below remains decisive.

The next predeclared derivative, `G1_GSM8K_U1000_E4500`, freezes 1,000 unique
source IDs and semantic signatures before any official-test result is known.
It preserves 4,500 exposures and 570 optimizer steps, with every source reused
four or five times across ten 450-row logical epochs. Its role is to distinguish
the effect of additional repetition from the U2000 and G1-4500 diversity cells;
it is not selected in response to confirmatory performance.

#### Official GSM8K confirmation freeze

`GSM8K_OFFICIAL_TEST_V1` freezes the pinned `openai/gsm8k` official test
Parquet at revision `a05f38c23a0e9ab0b71de8a2b4947e20f74f68f7`. The source
contains 1,319 rows and has SHA-256
`ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
The full view retains all 1,319 identities for answer/execution confirmation.
A separately frozen 250-row diagnostic view uses seed 22,901 and balances the
observed solution-step strata at 83 low, 84 medium, and 83 high examples.

The model prompt contains only the question. Gold rationales and final answers
are never passed to generation. Final answers are retained only in the scoring
record because official GSM8K does not provide teacher ASL. Exact normalized
question hashing finds zero overlaps with the U2000/E4500 training freeze. The
manifest and immutable views are under
`artifacts/paper1/gsm8k_scale_v1/official_test_v1/`. Neither view may be used
for checkpoint, architecture, objective, or hyperparameter selection.

#### Matched direct and large-number controls

`PAPER1_GSM8K_MATCHED_DIRECT_V1` freezes two base-Qwen3-0.6B controls on the
same 250 official identities before direct inference. `direct_concise` uses
thinking-disabled greedy decoding with a 128-token ceiling;
`direct_reasoning` uses native thinking with a 1,024-token ceiling. Both use the
same pinned revision, XPU FP16 backend, chat template, question ordering, and
numeric endpoint scorer. Prompts contain only the question and never expose
ASL demonstrations, benchmark rationales, answers, state, or intermediate
values. This comparison was added after observing the first ASL seed and is not
presented as preregistered.

- Protocol manifest:
  `artifacts/paper1/gsm8k_scale_v1/matched_direct_v1/protocol/manifest.json`

`PAPER1_GSM8K_LARGE_NUMBER_V1` is a separately frozen exploratory paired
robustness view. A deterministic factor-1,000 transform changes registered
digit source quantities, propagates the changes through every hidden GSM8K
arithmetic equation, and retains a row only when the source trace verifies, all
transformed source values are trace-grounded, operator signatures are
unchanged, and the terminal result is an integer. Unsafe or ambiguous
percentages, ages/calendar values, clocks, bounded time ratios, numeric
ordinals, lexicalized quantities, hyphenated unit modifiers, collisions, and
incomplete traces are excluded with recorded reasons.

The final freeze retains 59/250 parent identities: 26 low-, 22 medium-, and 11
high-step cases. Transformed answers comprise 38 four-to-six-digit, 14
seven-to-nine-digit, and seven ten-or-more-digit values. Only transformed
questions are model-visible; hidden traces and answers remain scorer-only. The
suite was frozen before any transformed inference and remains exploratory
because its design followed the first original-set ASL result.

- Paired rows: `artifacts/paper1/gsm8k_scale_v1/large_number_v1/data/large.jsonl`
- Exclusions: `artifacts/paper1/gsm8k_scale_v1/large_number_v1/data/excluded.jsonl`
- Manifest: `artifacts/paper1/gsm8k_scale_v1/large_number_v1/data/manifest.json`

### G1 development and historical test

The run boundary projects the legacy mixed development and test artifacts onto
GSM8K before model training or evaluation:

| Role | Mixed input | GSM8K selected | TAT-QA scope-rejected |
|---|---:|---:|---:|
| Development, checkpoint selection only | 25 | 17 | 8 |
| Historical paired test (`TEST-GSM17`) | 25 | 17 | 8 |

The split manifest verifies:

- zero train/development source overlap;
- zero train/test source overlap;
- zero development/test source overlap;
- zero train/test protected semantic-pattern overlap.

The archived mixed `TEST-25` and old scores remain available for historical
reproduction. New Paper 1 runs use only `TEST-GSM17`. Neither set is the future
official-test confirmation set.

- Eval manifest:
  `artifacts/paper1/gsm8k_scale_v1/g1_f0_4500/eval/manifest.json`
- GSM8K development view:
  `artifacts/paper1/gsm8k_scale_v1/g1_f0_4500/eval/dev.jsonl`
- GSM8K historical-test view:
  `artifacts/paper1/gsm8k_scale_v1/g1_f0_4500/eval/test.jsonl`

## Paper 1 Arithmetic Dataset Ladder

Approximate sizes below are planning values. Each dataset must receive a pinned
upstream version, license record, source hash, split manifest, and overlap audit
when it is imported.

| Dataset | Approximate size | Character | Registered Paper 1 role | Status |
|---|---:|---|---|---|
| **GSM8K** | 8.8K | Diverse 2-8 step grade-school problems | Immediate core training; official test for later confirmation | Active |
| **ASDiv** | 2.3K | Diverse elementary problems with equations | Later training diversity | Planned |
| **SVAMP** | 1K | Adversarial variations of simple arithmetic problems | Held-out adversarial test | Planned |
| **MAWPS** | 3.3K | Classic elementary word-problem collection with equations | Later training diversity after overlap audit | Planned |
| **MultiArith** | about 600 | Multi-step arithmetic stories | Secondary set only after legacy-overlap audit | Planned |
| **GSM-Plus** | 10.5K | Adversarial GSM8K perturbations | Untouched robustness test | Planned |
| **GSM-Symbolic** | GSM-derived | Controlled symbolic/template perturbations | Untouched invariance and generalization test | Planned |

Do not train on SVAMP, GSM-Plus, or GSM-Symbolic before their registered
held-out evaluation role has been completed. MAWPS and MultiArith require
cross-collection deduplication because classic arithmetic-word-problem
collections may share source problems or templates.

## Split Vocabulary

Use these terms consistently:

| Term | Meaning |
|---|---|
| Upstream split | Split published by the source dataset |
| Annotation corpus | Teacher-generated ASL plus deterministic validation metadata |
| Training freeze | Exact immutable rows exposed to adapter optimization |
| Development | May select checkpoints or tune declared settings; never reports final confirmation |
| Historical test | Frozen identities retained for paired continuity across experiments |
| Internal reserve | Unused rows from a training-side upstream split; not automatically a test |
| Confirmatory test | Untouched upstream test identities frozen before final evaluation |
| Robustness test | Registered adversarial or controlled-shift dataset never used for training |

## Leakage Rules

Every new Paper 1 freeze must record and enforce:

1. Dataset scope and upstream split.
2. Stable source and document IDs.
3. Input and output SHA-256 hashes.
4. Exact source-ID intersections across train, development, and test.
5. Normalized semantic-pattern intersections where available.
6. Template and near-duplicate overlap for derived or aggregated datasets.
7. Teacher provenance and whether answers or rationales were visible.
8. A fixed prompt and fixed ICL policy across records.

Public benchmark exposure during the base model's original pretraining is a
separate, generally unknown contamination risk. Source-disjoint adapter splits
do not prove that the pretrained base model never encountered a benchmark.

## Reproduction

Prepare the current GSM8K-only freeze and evaluation boundary without Docker:

```powershell
cmd /c scripts\prepare-paper1-gsm8k-g1.cmd
```

Train and evaluate the Qwen3-0.6B QKVO-r8 G1 adapter using the validated XPU
environment:

```powershell
cmd /c scripts\run-paper1-g1-gsm8k-f0-l0-xpu.cmd
```

The authoritative forward roadmap is
`docs/AGENTS_paper1_next_steps_after_D1.md`.
