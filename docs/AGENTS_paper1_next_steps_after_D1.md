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
  init 99173 (historical run label seed11): 16/25 answers, 23/25 executable
  init 23: 13/25 answers, 24/25 executable
  init 37: 17/25 answers, 22/25 executable

U2000 official GSM8K confirmation, 250 untouched question-only examples:
  init 99173 (artifact label seed11): 110/250 answers, 212/250 executable
  init 23: 94/250 answers, 222/250 executable
  init 37: 85/250 answers, 208/250 executable
  mean answer accuracy: 38.5% (range 34.0--44.0%)

U2000 paired factor-1,000 view, 59 pre-frozen eligible examples:
  init 99173: 27/59 answers
  init 23: 28/59 answers
  init 37: 23/59 answers

Matched Qwen3-0.6B direct reasoning:
  ordinary: 149/250 answers (59.6%)
  eligible original: 39/59 answers (66.1%)
  factor-1,000: 25/59 answers (42.4%), degradation -23.7 pp

ASL ordinary contribution versus direct reasoning:
  init 99173: -15.6 pp, bootstrap 95% [-23.2, -8.4]
  init 23: -22.0 pp, bootstrap 95% [-30.0, -14.0]
  init 37: -25.6 pp, bootstrap 95% [-32.8, -18.4]

ASL factor-1,000 relative robustness versus direct reasoning:
  init 99173: +20.3 pp, bootstrap 95% [+5.1, +37.3]
  init 23: +27.1 pp, bootstrap 95% [+10.2, +44.1]
  init 37: +18.6 pp, bootstrap 95% [+3.4, +33.9]

Full official GSM8K confirmation, init 37 only:
  509/1,319 answers (38.6%), 1,123/1,319 executable
```

Dataset selection and model initialization are separate seed axes. The frozen
G1, U2000, and U1000 first-run datasets use selection seed 11, while their
matched first model run uses initialization/training seed 99,173. Historical
artifact names containing `seed11` are retained for provenance and must not be
described as training seed 11.

Across all three declared D1 initialization runs:
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
**D1 semantic diversity.** Replacing repeated exposure to 450 programs with 4,500 unique leakage-audited programs improves autonomous answer generation and execution across three initialization seeds.

### Positive but incomplete
**Model size.** Qwen3-1.7B improves raw answers relative to 0.6B in earlier matched runs, but semantic-state/dependency quality does not improve proportionally.

The next model-size gate uses the exact U2000/E4500 F0 representation rather
than the older 500-program comparison. Qwen3-1.7B must use the same 2,000 unique
programs, 4,500 exposures, 570 optimizer steps, QKVO rank 8, initialization
99,173, and frozen 250/59 original/large identities as Qwen3-0.6B. Compare each
model against its own matched direct-reasoning control and report:

```text
answer interaction =
  (ASL_1.7B - direct_1.7B) - (ASL_0.6B - direct_0.6B)

robustness interaction =
  differential_degradation_1.7B - differential_degradation_0.6B
```

One 1.7B initialization is an exploratory resource gate. Replicate seeds 23 and
37 only when at least one interaction is positive and scientifically material;
do not infer ASL-specific leverage from a standalone 1.7B ASL improvement.

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

# PRIMARY WORKSTREAM — Matched ASL Contribution

## Main Paper 1 estimand

The main Paper 1 question is not whether an ASL adapter can produce correct
answers in isolation. It is:

> How much does the learned `NL -> ASL -> deterministic runtime` path add over
> the same pretrained model answering the same arithmetic questions directly,
> and does that advantage survive large changes in numeric magnitude?

This is the central result of Paper 1. Architecture ablations, semantic error
analysis, dataset scaling, and parser/runtime diagnostics explain the result,
but they do not replace it. The primary endpoint hierarchy is frozen as:

```text
1. Original answer contribution:
   accuracy(A0_original) - accuracy(B1_original)

2. Large-number robustness contribution:
   [accuracy(A0_large) - accuracy(A0_original_eligible)]
   - [accuracy(B1_large) - accuracy(B1_original_eligible)]

3. Supporting controls:
   repeat 1 and 2 against B0; report ASL execution-stage failures and paired
   question-level wins/losses against both direct conditions.
```

Endpoint 2 is especially important: preserving the dependency graph while
changing only numeric magnitude tests whether the symbolic route contributes
systematic computation rather than merely another learned answer surface.

No score from a different identity set, prompt budget, model revision, backend,
or decoding regime is an acceptable substitute. In particular, do not subtract
the historical 120-item direct-Qwen score from an official 250-item ASL score.

## Matched conditions

Freeze and run these conditions on identical official GSM8K identities:

```text
B0 DIRECT-CONCISE
   Base Qwen3-0.6B, question -> concise final answer, thinking disabled.

B1 DIRECT-REASONING
   Base Qwen3-0.6B, question -> natural-language reasoning + final answer.
   Give this control a sufficient but bounded output budget so ASL is not
   compared only against an artificially answer-constrained base model.

B1L DIRECT-REASONING-LONG (post-hoc capacity sensitivity)
   If at least 10% of B1 generations reach the 1,024-token ceiling, rerun B1
   with an otherwise identical 2,048-token ceiling. Report B1 unchanged as the
   registered primary control and B1L as a post-hoc sensitivity. Any ASL
   contribution claim must also survive comparison with the stronger of B1 and
   B1L; token-ceiling failures may not be presented as arithmetic failures
   without separate disclosure.

A0 ASL-RUNTIME
   Qwen3-0.6B U2000 adapter, question -> ASL -> validated deterministic
   execution -> returned answer.
```

Pin the base-model revision, question text, chat template, backend/dtype,
generation mode, endpoint extraction, and random seed. Record prompt and output
budgets explicitly. B0 and B1 must never see ASL demonstrations, benchmark
rationales, reference answers, runtime state, or intermediate values. A0 must
remain the autonomous zero-shot ASL condition already frozen. Where a control
requires a genuinely different mode, such as reasoning-enabled generation,
declare that difference rather than calling the protocols identical.

A0 is a three-training-seed condition. Report each adapter seed, its mean and
range, and paired outcomes against each direct control. Do not pool the three
predictions per question as independent observations. Deterministic direct
decoding needs one run; stochastic direct decoding requires the same declared
seed policy as A0.

For every generative condition, report the observed maximum generated-token
count and how many rows at that maximum are scorable and correct. Equality
between the observed maximum and the configured ceiling is treated as possible
truncation, not a successful termination signal.

## Paired large-number suite

Large-number robustness is a primary comparison, not an appendix. Create a
separate deterministic transformation of eligible frozen official questions:

```text
original question
  -> registered source-number spans
  -> magnitude transformation
  -> mechanically updated hidden arithmetic trace
  -> execution-verified transformed answer
```

Requirements:

- transform source quantities, not merely the final answer;
- preserve the arithmetic dependency graph and linguistic relation;
- use integer-safe mappings and retain required divisibility;
- exclude unsafe dates, percentages, ordinals, unit conventions, lexicalized
  numbers, and real-world constraints unless a deterministic validator proves
  that the transformation preserves meaning;
- validate every transformed hidden arithmetic equation and final answer;
- expose only the transformed question to generation;
- freeze source IDs, spans, mapping, transformed hashes, eligibility reasons,
  exclusions, and answers before model inference;
- keep transformed descendants out of all training and development data;
- report the eligible denominator rather than silently replacing exclusions.

Run B0, B1, and every declared A0 seed on the same transformed descendants. The
primary robustness statistic is the paired change from original to large:

```text
ASL degradation     = accuracy(A0_large) - accuracy(A0_original)
direct degradation  = accuracy(Bx_large) - accuracy(Bx_original)
ASL robustness gain = ASL degradation - direct degradation
```

Also report paired retention (correct original and transformed), gains, losses,
parse/lower/type/execute failures for A0, answer-extraction failures for direct
conditions, and results by original difficulty and magnitude band. Because this
suite is added after observing one A0 seed on the original set, label its first
result exploratory and freeze it before inspecting any direct or transformed
outputs.

## Claim gate

Paper 1 may claim that ASL adds value only if the matched comparison supports at
least one clearly delimited result:

```text
accuracy:           A0 improves over both B0 and the stronger B1 control;
numeric robustness: A0 has materially smaller paired large-number degradation;
mechanism:          executable ASL converts correct semantic bindings into exact
                    answers that the direct model misses.
```

If A0 does not beat B1, report that faithfully: ASL generation may still be a
useful inspectable interface, but the current experiment would not establish an
answer-accuracy advantage over direct reasoning.

If A0 and B1 are similar on original questions but A0 degrades materially less
on paired large-number variants, the paper may claim a bounded robustness
advantage, not a general accuracy advantage. If neither endpoint favors A0,
frame ASL as an inspectable runtime interface and treat the added generation
stage as an unresolved accuracy cost.

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
6. Can execution-verified semantic augmentation improve binding and dependency
   correctness, rather than only surface or numeric invariance?

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

## 9.1 Offline semantic augmentation ladder

These are offline dataset-construction interventions. They must not change the
test-time prompt, retrieve problem-dependent demonstrations, expose hidden
answers/rationales, or feed intermediate execution values back to the model.
Use one frozen instruction and either zero ICL or one globally fixed ICL prefix
for every record in a condition. The primary comparison remains zero-shot.

Split source parents before generating descendants. Every paraphrase,
counterfactual, query variant, reordered story, and recombination lineage must
remain in its parent's partition. When a recombination has multiple parents, its
partition is the most restrictive partition of any parent and it may not cross a
protected semantic-pattern family. Record parent IDs, transformation IDs,
generator versions, seeds, and pre/post CCIR hashes.

### C1 Relation-specific lexical and syntactic paraphrases

Replace the historical six-phrase paraphraser with operator-aware families. At
minimum cover addition/aggregation, difference/remaining, multiplicative
comparison, ratio/rate, equal allocation, percentage, and temporal shift. Include
equivalent realizations such as `twice`, `double`, `two times`, and `two for every
one`, and grammatical alternations rather than isolated random synonyms.

Meaning-preserving variants must retain the normalized CCIR graph, ordered
dependencies, query target, and executed answer. Reject a paraphrase if a
deterministic checker cannot establish preservation or if a teacher audit marks
it ambiguous.

### C2 Meaning-changing positive counterfactuals

Mutate one registered semantic slot in the gold AST/CCIR and generate a complete
new NL/ASL positive pair. Target the measured confusions: source entity, source
attribute, operator, noncommutative argument order, multiplier owner, temporal
state, and query target. For example, contrast `twice John's pink amount` with
`twice Carl's pink amount` while holding vocabulary and most numbers fixed.

Unlike M0.6, these are not merely rejected programs. Each counterfactual must
have a newly lowered, type-valid, executable gold program and a recomputed answer.
Require the intended semantic slot and answer to change unless the registered
mutation class explicitly tests an answer-preserving equivalence.

### C3 Query-target augmentation

Keep a verified world and ask for multiple valid quantities: an intermediate
source/derived fact, one category's remaining value, an aggregate, a difference,
or the original terminal quantity. Generate a new gold RETURN and recompute the
answer. This cell directly targets the observed RETURN-grounding failures.

### C4 Dependency-direction minimal pairs

Create tightly matched positive pairs such as `A has six more than B` versus `B
has six more than A`, and `A is twice B` versus `B is twice A`. Preserve lexical
content and constants where possible so success requires direction and ordered
argument binding rather than keyword recognition.

### C5 Coreference and entity-collision augmentation

Vary names, descriptions, pronouns, repeated entity types, and multiple agents
performing similar actions. Include only deterministically resolvable references
in positive training data. Register deliberately ambiguous variants as rejection
or abstention tests, not as ordinary positive programs.

### C6 Clause-order and distractor augmentation

Reorder independent clauses, vary question position, and add plausible irrelevant
facts using entities/units that cannot alter the gold dependency graph. Preserve
dependent-clause order unless a verified rewrite also updates references. The
checker must prove that the source-fact subgraph selected by RETURN and the answer
remain unchanged.

### C7 Compositional recombination

Compose compatible verified subgraphs into new worlds, prioritizing thin semantic
signatures and longer chains such as `rate -> aggregation -> remaining`. Unify
units and symbol identities explicitly, lower the merged graph, and generate NL
from the merged semantics. Reject accidental shortcuts, disconnected query
targets, duplicated facts, unit conflicts, and any recombination that reproduces
a protected test signature.

### Augmentation conditions

Run a cumulative ladder over the same frozen GSM8K source parents:

```text
AUG0 originals only
AUG1 AUG0 + relation-specific paraphrases
AUG2 AUG1 + positive semantic counterfactuals
AUG3 AUG2 + query-target and dependency-direction pairs
AUG4 AUG3 + coreference/entity collision, clause order/distractors,
     and compositional recombination
```

Use a greedy continuation protocol for the fast-track pass. Start from the
frozen U2000 adapter, train only on one new increment with a fresh optimizer,
and advance the resulting adapter only when its correct-answer count strictly
improves on a separately frozen 250-case augmentation-selection view. This view
must be sampled from the 1,069 official GSM8K cases outside the final
confirmatory 250, balanced by difficulty, and immutable across augmentation
stages. A tie or negative delta retains the previous adapter. Run the untouched
confirmatory set and factor-1,000 comparison only after the greedy ladder and
its decisions are frozen.

Fast-track increment order is `AUG1 relation paraphrase`, then `AUG2 synchronized
entity/path rename`. AUG2 is generated and audited independently, but its
training parent is AUG1 only when AUG1 passes; otherwise it starts from U2000.
Later increments follow the same single-addition rule rather than a factorial
combination sweep.

For every cell, include an originals-repeat control with the same optimizer steps
and approximately matched target tokens. Also report an epoch-matched view when
affordable. Do not attribute a gain to semantic augmentation when it can be
explained solely by greater exposure. Report unique parents, descendant rows,
alpha signatures, dependency DAGs, relation classes, and per-parent sampling
weights.

Primary augmentation gates are answer accuracy, source-fact F1, path/attribute
grounding, ordered dependency F1, relation direction, and RETURN grounding on
untouched parents. Parse/lower improvements alone do not pass the gate. Evaluate
ordinary official GSM8K, the frozen factor-1,000 suite, and separately frozen
paraphrase/binding challenge descendants of test-only parents.

---

## 9.2 Additional training and inference approaches

Apply these only after the corresponding labels/candidates can be derived
deterministically from verified gold ASL/CCIR. Keep the base model, source parents,
exposures, prompt, decoding seed, and evaluation identities matched to AUG0.

### T0 Ordinary autoregressive control

Retain the current QKVO-r8 token-likelihood model as the mandatory control.

### T1 Slot-level semantic auxiliary objectives

Add supervised heads or tagged losses for entity identity, attribute identity,
source-fact selection, operator class, ordered arguments, dependency edges, and
RETURN target. Normalize each objective independently and report gradient norms
and component losses so syntax-token volume cannot silently dominate semantics.
Compare fixed weights, uncertainty-normalized weights, and a bounded focal variant
on one development-only sweep; freeze the selected setting before confirmation.

### T2 Runtime-symbol pointer grounding

Build a deterministic per-question symbol table from model-visible spans and
canonical runtime aliases. Predict source/target slots by pointer or constrained
classification, then lower selected symbols into ASL paths. The table may contain
only information extractable from the question at test time. Measure extraction,
candidate recall, pointer accuracy, and downstream ASL accuracy separately.

### T3 Clause-local semantic contrastive training

Replace M0.6's successful-but-nontransferring whole-program ranking loss with
local comparisons at the erroneous slot. Rank the gold source path, ordered
argument, operator, edge, or RETURN target against valid confounders from the same
question. Keep the generative objective active and gate on autonomous generation,
not ranking accuracy alone.

### T4 Generate, execute, and semantically rerank

Train a verifier on gold programs plus deterministic local semantic corruptions.
At inference, generate a bounded grammar-valid candidate set, lower/type-check and
execute each candidate, then select by NL-to-ASL semantic consistency. The
verifier sees only the question and candidate program, never the benchmark answer
or hidden rationale. Report candidate oracle coverage, verifier selection
accuracy, final answers, generated tokens, and latency separately.

### T5 Adapter placement after semantic-supervision gate

Only after T1--T3 show a positive semantic signal, compare matched QKVO-r8 against
QKVO+MLP-r8 and modality-specific lower/shared upper adapters. Rank-only expansion
is lower priority because QKVO-r16 already improved fit/syntax without improving
answers. Run the same best semantic objective at 1.7B before advancing to a 4B
QLoRA gate.

No condition in this ladder may use dynamic or problem-dependent ICL. Grammar
constraints remain a safety/validity control and cannot be presented as semantic
supervision.

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

> The main obstacle in autonomous NL→ASL transfer is not parser validity or deterministic execution. Memory transport, adapter specialization, semantic weighting, hard-negative ranking, and a stronger explicit graph bottleneck do not outperform the 450-program Q0 control. In contrast, replacing repeated exposure with 4,500 unique leakage-audited programs improves autonomous answer generation and executable program rate across three matched initialization runs. Alpha-normalized semantic metrics also improve, while strict teacher-path equivalence does not. The evidence therefore supports semantic diversity as the first replicated lever and motivates separating computational/world equivalence from arbitrary symbolic naming.

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
D1 init99173
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

### Table E — primary matched contribution

```text
condition
original answer accuracy
large-number answer accuracy
paired retention / gains / losses
large-number degradation
ASL minus direct original accuracy
ASL minus direct large-number degradation (difference-in-differences)
paired exact McNemar p-value and identity-bootstrap 95% interval
latency and generated tokens
ASL parse / lower / type / execute (A0 only)
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

For the primary ASL contribution comparison:
- use identical frozen identities for A0, B0, and B1;
- report paired gains/losses and exact McNemar intervals/tests, not only the
  difference between marginal percentages;
- report each A0 training seed separately plus its mean/range;
- bootstrap identities, not repeated seed-question rows, for the exploratory
  large-number difference-in-differences interval.

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
  e3/direct_answer_eval.py
  e3/large_number_suite.py
  e3/contribution_analysis.py
  e3/semantic_augmentation.py
  e3/semantic_auxiliary.py
  e3/symbol_pointer.py
  e3/semantic_reranker.py

scripts/
  run-paper1-gsm8k-matched-contribution-xpu.ps1
  run-paper1-gsm8k-matched-contribution-xpu.cmd
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
artifacts/paper1/gsm8k_scale_v1/matched_direct_v1/
artifacts/paper1/gsm8k_scale_v1/large_number_v1/
artifacts/paper1/gsm8k_scale_v1/analysis/matched_contribution_v1.json
artifacts/paper1/gsm8k_semantic_augmentation_v1/
  parent_manifest.json
  transformation_ledger.jsonl
  aug0_originals/
  aug1_relation_paraphrase/
  aug2_positive_counterfactual/
  aug3_query_dependency/
  aug4_grounding_composition/
  test_challenges/
artifacts/paper1/gsm8k_semantic_training_v1/
```

After all three official A0 summaries exist, run the complete resumable matched
campaign with:

```bat
scripts\run-paper1-gsm8k-matched-contribution-xpu.cmd
```

The script checkpoints every prediction, skips completed conditions, and emits
the paired analysis only after original and transformed B0, B1, and all three A0
seed files pass exact identity checks.

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
direct prompt contains question but no hidden rationale or answer
direct endpoint extraction handles declared final-answer forms
large-number transformation is deterministic
large-number transformed trace recomputes the stored answer
unsafe numeric contexts are excluded with reasons
original/transformed identities remain one-to-one
augmentation descendants cannot cross parent partitions
meaning-preserving rewrites retain normalized CCIR and answer
positive counterfactuals change the registered semantic slot and recompute answer
query-target variants return the newly registered target
dependency-direction pairs preserve constants but reverse the registered edge
coreference positives have one deterministic resolution
distractors remain disconnected from the RETURN dependency closure
compositions lower, type-check, execute, and avoid protected test signatures
symbol-table candidates contain model-visible spans only
semantic reranker never reads hidden benchmark answers or rationales
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
Complete for U2000: all three adapters were tested on the untouched frozen 250
set. The 34.0--44.0% answer range and cross-seed disagreement preclude a broad
semantic-compiler claim.

### Gate F — matched direct and magnitude controls
Primary 1,024-token comparison complete. Direct reasoning wins ordinary answer
accuracy (59.6% versus 34.0--44.0% for ASL), so no ordinary ASL benefit is
supported at 0.6B. ASL has +18.6 to +27.1 percentage-point differential
factor-1,000 robustness across the three seeds, with all paired bootstrap
intervals above zero. Because B1 reached its token ceiling on 58/250 ordinary
and 18/59 large generations, the registered result remains primary while the
post-hoc B1L 2,048-token sensitivity tests whether the robustness conclusion
survives a less truncated direct baseline.

### Gate G - semantic augmentation and supervision

Run AUG0--AUG4 only after the parent split, transformation ledger, and semantic
preservation/mutation tests are frozen. Advance an augmentation cell only when it
improves answer accuracy and at least one binding-sensitive metric without a
material regression in another: source facts, grounded paths/attributes, ordered
dependencies, relation direction, or RETURN target. Parse/lower gains alone fail
the gate. Run T1--T5 sequentially; do not combine multiple unvalidated training
or architecture changes in one cell.

---

# 27. Immediate execution order

```text
P0  Freeze legacy mixed D1_v1 and all three D1 initialization runs without mutation.

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

    For U1000, run the unchanged historical 17-item projection and then the
    frozen official 250-item question-only view. The official result is
    confirmatory evaluation, never checkpoint or hyperparameter selection.

P6  Freeze a larger confirmatory set from official GSM8K test.
    Never use it for tuning.
    COMPLETE: frozen 250-item view and all three U2000 evaluations.

P6a Freeze B0 direct-concise and B1 direct-reasoning prompts and endpoint
    scorers on the same official identities. Run both before interpreting the
    ASL-vs-direct effect.

P6b Build and execution-verify the paired official GSM8K large-number suite.
    Freeze eligibility, transformations, exclusions, and hashes before any
    transformed inference.
    COMPLETE: 59 paired factor-1,000 variants; all three A0 seeds evaluated.

P6c Run B0, B1, and all A0 adapter seeds on both original and large-number
    questions. Treat answer accuracy and differential magnitude degradation as
    the primary Paper 1 comparisons.
    COMPLETE for the registered 1,024-token B1 control; B1L post-hoc sensitivity
    is running. Ordinary ASL contribution is negative for all three seeds;
    differential factor-1,000 robustness is positive for all three seeds.

P6d Run the matched Qwen3-1.7B U2000/E4500 initialization-99173 gate on the same
    250/59 identities. Compute answer and robustness model-size interactions.
    Replicate seeds 23 and 37 only if the one-seed gate is positive and material.

P6e Freeze the offline GSM8K semantic-augmentation protocol over the same source
    parents: relation paraphrases, positive counterfactuals, query-target variants,
    dependency-direction pairs, coreference/entity collisions, clause order and
    distractors, and compositional recombinations. Freeze parent lineages and the
    test-only paraphrase/binding challenge before training.

P6f Run AUG0--AUG4 with Qwen3-0.6B QKVO-r8. Match optimizer steps and target-token
    exposure against originals-repeat controls. Use one exploratory initialization;
    replicate only cells that pass Gate G on autonomous untouched-parent metrics.

P6g On the best passed data cell, run T1 slot-level auxiliary supervision, then T2
    runtime-symbol pointers, then T3 clause-local contrastive binding, and finally
    T4 bounded semantic reranking. Test QKVO+MLP and modality-specific adapters only
    after a semantic-supervision condition passes. Never use dynamic ICL.

P7  Revise F3 around grounded identity + alpha-renamable locals.

P8  Run F3-450.

P9  Revise E3 contrastive pretraining to be alpha-aware.

P10 Run E3a vs E3b on 450.

P11 If positive, add E3c component alignment.

P12 Compare:
     Q0-450
     best F3/E3-450
     D1-F0-4500

P13 Confirm best conditions and matched direct controls on the larger untouched
    evaluation.

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
