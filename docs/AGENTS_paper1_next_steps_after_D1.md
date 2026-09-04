# AGENTS — Paper 1 Next Steps After Replicated D1 Diversity Signal

## Status

This document supersedes broad exploratory prioritization for Paper 1 after the replicated D1 OpenRouter diversity result.

## Forward dataset scope

Paper 1 now studies compilation of simple arithmetic word problems only. From
this revision onward:

- all newly created Paper 1 train, development, test, confirmatory, and
  robustness artifacts must contain only arithmetic word problems;
- the immediate phase uses **GSM8K only**;
- TAT-QA is retired from all new Paper 1 work because its table/document
  arithmetic introduces a different retrieval-and-grounding problem;
- historical mixed GSM8K/TAT-QA artifacts and reported results remain immutable
  provenance, but may not be reused as forward training or evaluation splits;
- historical mixed checkpoints may be analyzed on their frozen GSM8K subset,
  but every new checkpoint must be trained without TAT-QA;
- dataset and checkpoint IDs must state their corpus scope. Do not call a mixed
  D1 checkpoint a GSM8K-only checkpoint.

The arithmetic dataset ladder is:

| Dataset | Approximate size | Character | Paper 1 role |
|---|---:|---|---|
| **GSM8K** | 8.8K | Linguistically diverse, 2--8 step grade-school problems | **Immediate core** |
| **ASDiv** | 2.3K | Diverse elementary word problems with supplied equations | **Later training diversity** |
| **SVAMP** | 1K | Adversarial variations of simple arithmetic word problems | **Later held-out test** |
| **MAWPS** | 3.3K | Collection of classic elementary word problems with equations | **Later training diversity** |
| **MultiArith** | about 600 | Multi-step arithmetic stories | Later secondary set after overlap audit |
| **GSM-Plus** | 10.5K | Adversarial GSM8K perturbations | **Later robustness test** |
| **GSM-Symbolic** | GSM-derived | Controlled symbolic/template perturbations | **Later invariance/generalization test** |

Before adding any dataset, freeze its upstream version, license, split, source
IDs, hashes, overlap audit, and role. Never train on SVAMP, GSM-Plus, or
GSM-Symbolic before their registered test role has been completed. Audit MAWPS
and MultiArith against each other and against any shared legacy word-problem
collections before use.

Latest frozen evidence:

```text
Q0 F0 ordinary likelihood, 450 unique programs:
  11/25 answers
  19/25 executable

D1 OpenRouter 4,500 unique programs:
  seed 11: 16/25 answers, 23/25 executable
  seed 23: 13/25 answers, 24/25 executable
  seed 37: 17/25 answers, 22/25 executable
```

Across all three declared D1 seeds:
- autonomous answer accuracy exceeds Q0;
- executable rate exceeds Q0;
- alpha-normalized return equivalence exceeds Q0;
- mean alpha-normalized state F1 exceeds Q0;
- strict path-sensitive state/return equivalence does not improve.

This is currently the strongest replicated positive Paper 1 intervention.

Future work must distinguish four axes:

```text
1. semantic diversity / data scale
2. representation design
3. training objective
4. architecture
```

Do not conflate them.

---

## 1. Current evidence hierarchy

### Positive / replicated
**D1 semantic diversity.** Replacing repeated exposure to 450 programs with 4,500 unique leakage-audited programs improves autonomous answer generation and execution across three seeds.

### Positive but incomplete
**Model size.** Qwen3-1.7B improves raw answers relative to 0.6B in earlier matched runs, but semantic-state/dependency quality does not improve proportionally.

### Negative / null so far
- Q1/Q2/Q3 external-ASL memory do not beat Q0.
- Q3S1/Q3S2/Q3S3 encoder-specialization variants do not beat Q0.
- M0.5 semantic token weighting changes component F1s but worsens answers.
- M0.6 hard-negative ranking learns the ranking task but hurts autonomous generation.
- F4 explicit graph bottleneck does not solve grounding.
- Stateful incremental compilation does not beat whole-program Q0.

---

## 2. Main interpretation

The strongest current conclusion is:

> Genuine semantic/program diversity improves autonomous compilation more reliably than the tested representation, memory, objective, and adapter-specialization interventions.

However:

> D1 improves executed/computational semantics more clearly than canonical teacher-path semantics.

Do not equate strict path-sensitive mismatch with wrong computation without alpha-normalized analysis.

---

## 3. Alpha-equivalent semantics

Maintain two notions:

```text
STRICT SEMANTICS
teacher path names / exact symbolic vocabulary matter

ALPHA-NORMALIZED SEMANTICS
arbitrary internal names are canonicalized
while computation/dependency structure is preserved
```

Future metrics and representation design must separate:

```text
ontology-vocabulary agreement
vs
computation/world-structure agreement
```

Classify identities as:

```text
LOCAL_TEMP
DERIVED_STATE
SOURCE_GROUNDED
ENTITY_ATTRIBUTE
QUERY_TARGET
EVENT_ID
```

Alpha-renaming may be permissive for `LOCAL_TEMP` and selected derived/event IDs, but must not erase genuine grounded identity errors.

---

## 4. Immediate research questions

1. Why does D1 improve answers/execution but not strict canonical semantic equivalence?
2. Is the D1 gain driven by raw rows, surface diversity, semantic-signature diversity, dependency-graph diversity, source-fact diversity, or dataset balance?
3. Can revised F3 preserve computation while making world-state semantics more stable/reusable?
4. Can E3 NL↔ASL preconditioning improve sample efficiency relative to simply adding more examples?
5. At what data scale does D1 saturate?

---

# WORKSTREAM A — D1 Error Decomposition

## 5. Compare Q0 and all D1 seeds per identity

For every prediction classify:

```text
A syntax failure
B lowering/type failure
C execution failure
D source-fact grounding error
E operator error
F dependency/reference error
G query-target error
H wrong computation graph
I correct computation + noncanonical symbols
J correct answer + semantically different state
K strict semantic match
```

Add alpha-normalized equivalents.

Produce one deterministic paired table containing:
- source_id;
- dataset;
- Q0 and all D1 predictions/scores;
- strict/alpha state;
- strict/alpha return;
- source fact F1;
- path F1;
- operator F1;
- edge/dependency F1;
- semantic-state F1.

Add paired categories:

```text
all D1 better
all D1 worse
mixed
Q0 only
D1 only
stable correct
stable wrong
```

---

## 6. Canonical alpha graph

Build a deterministic alpha-normalized computation graph for every gold/predicted program.

Canonicalize:
- local variable names;
- temporary statement IDs;
- arbitrary event IDs.

Preserve:
- grounded entity/attribute identity;
- source identity;
- constants;
- operators;
- dependency edges;
- query target;
- semantically relevant order.

Do not alpha-normalize away actual semantic errors.

---

# WORKSTREAM B — Diversity Learning Curve

## 7. Build controlled D1 subsets

Create frozen subsets:

```text
450
1,000
2,000
4,500
```

Optional:
```text
3,000
```

Keep model/revision/rank/prompt/test protocol fixed.

### B1 — Exposure-matched
Keep optimizer steps approximately fixed. As unique rows grow, repeat frequency falls.

### B2 — Epoch-matched
Keep epoch count approximately fixed. Total optimizer steps grow with data.

Do not mix the interpretations.

---

## 8. Learning-curve metrics

Report:

```text
answer
parse
execute
strict_state
strict_return
alpha_state
alpha_return
mean_alpha_state_f1
source_fact_f1
path_f1
operator_f1
edge_f1
```

Plot against:
- unique examples;
- unique alpha semantic signatures;
- unique dependency graphs.

---

# WORKSTREAM C — Surface vs Semantic Diversity

## 9. Controlled diversity cells

If feasible create:

### C1 Surface-heavy / semantic-limited
Many paraphrases/number variants over a bounded semantic-signature set.

### C2 Semantic-diverse / surface-limited
Many distinct dependency/operator/source-fact structures with controlled surface variation.

### C3 Natural D1
Original OpenRouter diversity.

Question:

> Is D1 mainly benefiting from semantic structure diversity or generic lexical exposure?

---

## 10. Semantic signatures

Derive deterministic signatures from gold ASL/CCIR using:

```text
operator multiset
dependency DAG shape
number of source facts
number of derived states
query type
path-depth profile
reference count
arithmetic stage count
dataset/task family
```

Create:

```text
signature_strict
signature_alpha
```

The alpha form must ignore arbitrary local variable names.

---

# WORKSTREAM D — Larger Confirmatory Evaluation

## 11. Preserve historical TEST-25 as provenance

Do not delete or mutate the 25 frozen historical examples. Retain their old
scores for provenance, but define `TEST-GSM17` as the immutable GSM8K projection
used by every new Paper 1 run. Do not generate new TAT-QA predictions.

Add a larger untouched **GSM8K-only** confirmatory set from the official GSM8K
test split:

```text
100–250 examples
```

Stratify by:
- semantic signature;
- dependency depth;
- entity count;
- source-fact count;
- operator family;
- query type.

Roles:

```text
DEV       tuning/checkpoint selection
TEST-25   archived mixed-result provenance only
TEST-GSM17 historical paired GSM8K continuity
CONFIRM   untouched final confirmation
```

No architecture/hyperparameter decision may use `CONFIRM`.

---

# WORKSTREAM E — Revised F3

## 12. F3 identity policy

Revise F3 around:

```text
GROUNDED
  source/entity/attribute identity must remain stable

EVENT
  stable inside a scope but alpha-renamable globally

LOCAL
  temporary computation names freely alpha-renamable
```

F3 should preserve:
- grounded entities;
- attributes;
- events;
- relations;
- dependencies;
- query semantics.

It should not over-penalize teacher-specific temporary naming.

---

## 13. F3 comparison

Run:

```text
F0-450
F3-450
D1-F0-4500
```

Metrics:
- strict semantics;
- alpha semantics;
- grounded identity;
- dependencies;
- execution;
- answer.

Positive F3 result can be:
- improved alpha/world semantics;
- improved grounded identity;
- better dependency F1;
- better sample efficiency;
- or better answers.

Exact teacher syntax is not required.

---

# WORKSTREAM F — Revised E3 Preconditioning

## 14. Architecture

Use:

```text
NL  -> E_N -> E_NA -> Z_N
ASL -> E_A -> E_NA -> Z_A
```

with modality-specific lower/mid processing and shared upper semantic/world layers.

Initial E3 variants:

```text
E3a  hybrid architecture, generation only
E3b  + global alpha-aware contrastive pretraining
E3c  + grounded/component-level alignment
E3d  + masked/corrupted ASL reconstruction
E3e  + weak alignment during causal generation
```

Run `E3a vs E3b` first.

---

## 15. Alpha-aware alignment

During E3 contrastive pretraining, treat alpha-equivalent ASL variants as positives:

```text
NL_i <-> ASL_i
NL_i <-> alpha_rename_1(ASL_i)
NL_i <-> alpha_rename_2(ASL_i)
```

Hard negatives must alter semantics:
- wrong grounded source fact;
- wrong relation;
- wrong role;
- wrong operator;
- wrong dependency;
- wrong query target.

A mere temporary-variable rename is not a negative.

---

## 16. Component alignment

Component types:

```text
GROUNDED_ENTITY
GROUNDED_ATTRIBUTE
RELATION
ROLE
VALUE
REFERENCE
QUERY
LOCAL_TEMP
```

Use local-temp alignment only in a rename-invariant manner.

Do not force exact raw hidden-state equality.

Measure:
- paired cosine;
- CKA;
- NL→ASL retrieval;
- ASL→NL retrieval;
- component retrieval;
- autonomous semantic behavior.

Behavior remains the primary criterion.

---

# WORKSTREAM G — Objective Controls

## 17. Preserve current negatives

Do not broadly repeat:
- M0.5 naive semantic weighting;
- M0.6 hard-negative ranking;
- F4 graph bottleneck.

Only revisit with a materially new hypothesis.

A possible later objective is an **alpha-aware structured loss**:

```text
L =
L_token
+ λ_ground * L_grounded_identity
+ λ_edge * L_dependency
+ λ_query * L_query
```

with no semantic penalty for arbitrary local renaming.

Do not implement until the alpha-equivalence taxonomy is stable.

---

# WORKSTREAM H — Combine Data and Architecture

## 18. Only after a positive F3/E3 signal

Then compare:

```text
Q0-450
best F3/E3-450
D1-F0-4500
best F3/E3-4500
```

This tests the interaction:

```text
better representation
×
more semantic diversity
```

Do not spend large teacher/compute budget on F3/E3-large before the 450-row gate.

---

# 19. Architecture priority

Do not immediately return to:
- more Q3 memory variants;
- larger adapter banks;
- new attention modes;
- dedicated full encoders.

Architecture becomes high priority again only if:
- E3 preconditioning yields a representation/behavior signal;
- F3 improves sample efficiency;
- long-context typed-memory work requires it.

---

# 20. D1 corpus freeze

Freeze the replicated D1 dataset as:

```text
D1_v1
```

Store:
- source IDs;
- hashes;
- teacher provenance;
- strict/accepted status;
- dedup policy;
- semantic signatures.

Any later recovered rows go to:

```text
D1_v2_candidate
```

Do not silently mutate D1_v1.

---

# 21. Main Paper 1 framing

Use the evidence-supported narrative:

> The main obstacle in autonomous NL→ASL transfer is not parser validity or deterministic execution. Memory transport, adapter specialization, semantic weighting, hard-negative ranking, and a stronger explicit graph bottleneck do not outperform the 450-program Q0 control. In contrast, replacing repeated exposure with 4,500 unique leakage-audited programs improves autonomous answer generation and executable program rate across three matched seeds. Alpha-normalized semantic metrics also improve, while strict teacher-path equivalence does not. The evidence therefore supports semantic diversity as the first replicated lever and motivates separating computational/world equivalence from arbitrary symbolic naming.

Do not claim a solved semantic compiler.

---

# 22. Tables to maintain

### Table A — baseline/architecture/objective controls

```text
Q0
Q1/Q2/Q3
Q3S1/Q3S2/Q3S3
M0.5
M0.6
F4
```

### Table B — D1 replication

```text
Q0
D1 seed11
D1 seed23
D1 seed37
```

### Table C — strict vs alpha semantics

```text
strict state
alpha state
strict return
alpha return
mean alpha-state F1
```

### Table D — scaling curve

```text
450
1000
2000
4500
```

For legacy mixed experiments, continue reporting GSM8K and TAT-QA separately so
the historical record is auditable. All new tables must report GSM8K alone until
the registered arithmetic dataset ladder is activated.

---

# 23. Statistical discipline

For D1:
- report mean/range across seeds;
- do not pool repeated predictions on the same 25 identities as independent samples.

For future key scale points:
- use at least 3 seeds for confirmatory cells;
- single-seed exploratory cells are gating evidence only.

Use exact McNemar for paired answer transitions where appropriate and label exploratory p-values.

---

# 24. New code/artifacts

Suggested code:

```text
src/ccpu/paper1/
  alpha_normalize.py
  semantic_signature.py
  paired_diversity_analysis.py
  scaling_dataset.py
  confirmatory_split.py
```

Suggested artifacts:

```text
artifacts/paper1/d1_v1/
  manifest.json
  semantic_signatures.jsonl
  alpha_equivalence_audit.json
  paired_analysis.jsonl
  scaling/
  confirmatory/

artifacts/paper1/f3_v2/
artifacts/paper1/e3_v2/
```

---

# 25. Required tests

Add deterministic tests for:

```text
alpha-equivalent local rename
grounded path not alpha-renamed
event-ID rename
dependency preservation
query-target preservation
signature invariance to local rename
signature sensitivity to operator changes
signature sensitivity to source-fact changes
```

---

# 26. Go / no-go gates

### Gate A — D1 diagnosis
Complete before prioritizing new architecture.

### Gate B — scaling curve
If 1k/2k/4.5k shows coherent improvement, semantic diversity remains primary.

### Gate C — F3-450
Proceed to F3-large only if F3 improves alpha/world semantics or sample efficiency.

### Gate D — E3b
Proceed to E3c/d only if preconditioning improves meaningful autonomous semantic metrics over E3a.

### Gate E — confirmatory set
No broad semantic-compiler claim until the best condition is tested on the larger untouched set.

---

# 27. Immediate execution order

```text
P0  Freeze legacy mixed D1_v1 and all three D1 seeds without mutation.

P0a Freeze a new GSM8K-only corpus and manifest from strict OpenRouter GSM8K
    programs. Reject every non-GSM8K row at the data boundary. Give all derived
    datasets and checkpoints GSM-specific IDs.

P1  Run full Q0 vs D1 paired semantic decomposition.

P2  Finalize alpha-equivalence taxonomy:
    LOCAL_TEMP / DERIVED_STATE / GROUNDED / QUERY / EVENT.

P3  Generate alpha-normalized graphs and semantic signatures.

P4  Build GSM8K-only subsets:
    450 / 1000 / 2000 / 4500.
    Preserve the official GSM8K test split for confirmation.

P5  Run exposure-matched 1k/2k cells.
    Replicate only if the trend is coherent.

P6  Freeze a larger confirmatory set from official GSM8K test.
    Never use it for tuning.

P7  Revise F3 around grounded identity + alpha-renamable locals.

P8  Run F3-450.

P9  Revise E3 contrastive pretraining to be alpha-aware.

P10 Run E3a vs E3b on 450.

P11 If positive, add E3c component alignment.

P12 Compare:
     Q0-450
     best F3/E3-450
     D1-F0-4500

P13 Confirm best conditions on larger untouched evaluation.

P14 Only then consider:
     F3/E3-large
     dedicated encoders
     additional memory architecture
     long-context integration.

P15 After the GSM8K conclusions are frozen, add arithmetic datasets one role at
    a time: ASDiv and overlap-audited MAWPS for training diversity; SVAMP,
    GSM-Plus, and GSM-Symbolic for untouched robustness/generalization; and
    MultiArith only after legacy-overlap analysis.
```

---

# 28. Final research principle

Current evidence favors:

> Teach the model many genuinely different semantic worlds before adding increasingly elaborate machinery.

But the D1 strict-vs-alpha gap also says:

> Do not confuse failure to imitate a teacher's internal symbol vocabulary with failure to construct the correct computation or world relation.

The next phase should jointly pursue:

```text
MORE SEMANTIC DIVERSITY
+
BETTER SEMANTIC EQUIVALENCE
+
SAMPLE-EFFICIENT REPRESENTATION LEARNING
```

before returning to heavier architecture.
