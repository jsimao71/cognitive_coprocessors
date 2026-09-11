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

### Cross-dataset LoRA sequence (frozen v1)

Run ASDiv, SVAMP, MAWPS, and GSM-Plus in this exact order of conditions:

1. **E0, existing GSM LoRA:** evaluate the immutable U2000/E4500 GSM adapter
   without target examples, target ICL, or additional training. Freeze these
   predictions before creating target ASL supervision.
2. **E1, continued target LoRA:** copy the GSM adapter into a new dataset-
   specific run and continue training. Never update the GSM-only checkpoint in
   place. Report target accuracy and GSM/O0 retention.
3. **E2, fresh target LoRA:** initialize the same LoRA architecture from the
   base Qwen checkpoint and train only on that target dataset.

E1 and E2 must use the same target train/dev identities, optimizer-step and
example-exposure budget, seed policy, prompt, decoder, and frozen 250-example
diagnostic. Their difference estimates transfer from the learned GSM NL-to-ASL
compiler; E0 measures zero-shot transfer. Preserve the initial E0 results for
SVAMP and GSM-Plus as OOD evidence before using any disjoint reserve families
for E1/E2. Dataset-specific training is ordered last, after the quick E0 pass
and the transferred E1 pass.

The pinned source registry is
`configs/paper1/cross_dataset_transfer_v1.json`; frozen manifests and splits
are under `artifacts/paper1/cross_dataset_transfer_v1/`. GSM-Plus is grouped by
seed-question family. Only question text is model-visible during evaluation.
Answer-blind teacher seed scaffolds contain 1,898 ASDiv, 600 SVAMP, 3,635
MAWPS, and 6,782 GSM-Plus training candidates. They are not ASL labels: do not
start teacher generation, accept/reject repair, E1, or E2 until E0 predictions
for that dataset have been copied back and checksummed.

MAWPS has passed that per-dataset gate. Its immutable E0 run scored 202/250
answers (80.8%) and 236/250 executable programs (94.4%). A relation-diverse
500-record training pilot and disjoint 100-record development split were then
annotated. The answer-blind primary pass accepted 458/500 train and 82/100 dev;
one provenance-marked rationale-assisted repair round recovered the remaining
42 train and 18 dev records. The final E1/E2 corpus therefore contains 500
train and 100 dev programs, all execution- and answer-verified, with zero
train/dev/diagnostic identity overlap. E1 continues from a copied GSM adapter;
E2 uses the same ordered corpus and optimizer budget from a fresh base-model
adapter. The source GSM checkpoint remains immutable.

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

## Inference-token budget and Qwen reasoning-mode ladder

Preserve B1 at 1,024 tokens and B1L at 2,048 tokens as the registered and
post-hoc direct controls. Their current Qwen3-0.6B configurations use thinking
enabled, whereas A0 uses thinking disabled with a 384-token ceiling. Treat this
as a declared protocol difference and investigate it directly rather than
interpreting maximum token limits as actual compute consumption.

First recover the following telemetry from every existing B0, B1, B1L, and A0
prediction without rerunning inference where the records are sufficient:

- prompt, generated, and total tokenizer-token counts per example;
- mean, median, p90, and maximum generated tokens;
- explicit stop reason and ceiling-hit rate;
- generated tokens for correct and incorrect answers separately;
- generation wall time, followed separately by ASL parse/lower/execute time;
- final-answer accuracy and paired correctness at each budget.

All neural tokens, including thinking tokens emitted by the local Qwen model,
count toward inference cost. Do not compare only configured maxima. Report
correct answers per 1,000 generated tokens as a descriptive aggregate, but use
the full accuracy-versus-token-cost Pareto frontier as the primary efficiency
analysis because a short incorrect response must not appear efficient merely by
terminating early.

Use one fixed prompt per output mode and define reasoning level operationally;
do not claim a model-native reasoning-effort control that Qwen does not expose:

```text
R0  thinking disabled
R1  thinking enabled, 256 generated-token ceiling
R2  thinking enabled, 512 generated-token ceiling
R3  thinking enabled, 1,024 generated-token ceiling (registered B1)
R4  thinking enabled, 2,048 generated-token ceiling (post-hoc B1L)
```

Run R0 at matched ceilings where needed to separate the effect of thinking mode
from the effect of available tokens. For direct answers, start with the frozen
55-parent magnitude cohort at ceilings `128`, `256`, `512`, `1024`, and `2048`.
For ASL, start with `64`, `128`, `256`, and `384` tokens with thinking disabled.
Add a bounded ASL-thinking diagnostic at `384` and `512` tokens only after its
output contract is frozen: the model may think internally, but the scored final
payload must be an ASL program processed by the unchanged parser and runtime.
No benchmark rationale, answer, runtime state, dynamic ICL, or intermediate
value may enter either prompt.

For each budget and reasoning mode, plot both accuracy and actual generated
tokens against `log10(source-number scale)`. Estimate:

```text
reasoning compression = mean direct generated tokens / mean ASL generated tokens
token-cost slope       = change in generated tokens per log10 magnitude unit
accuracy slope         = change in accuracy per log10 magnitude unit
```

The intended mechanistic test is whether ASL generation cost depends mainly on
program structure rather than operand magnitude, while direct neural reasoning
requires more tokens or loses accuracy as magnitude grows. CPU execution cost
must be reported, but separately from autoregressive token cost. Promote only
non-dominated or scientifically diagnostic settings from the 55-parent screen
to the untouched 250-item ordinary comparison. Freeze those selected settings
before inspecting their 250-item predictions.

After the 0.6B frontier and magnitude curves are complete, repeat only the
frozen frontier endpoints with Qwen3-1.7B. This tests whether model scaling
improves accuracy by consuming a similar reasoning budget, needs a larger
budget, shifts the direct/ASL crossover, or removes magnitude sensitivity.

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

## Common-parent magnitude curve and failure audit

Keep the registered 250-question ordinary comparison and its 59 eligible
factor-1,000 descendants as the primary causal test. Add a separate exploratory
curve over one frozen intersection of parents that passes the conservative
transformer at every scale:

```text
x1, x10^2, x10^3, x10^4, x10^6
```

The frozen intersection contains 55 parents (10 high, 20 medium, and 25 low
difficulty). Every B1, B1L, and A0 point in the curve must use exactly these 55
ordered parent identities. Never increase a scale's denominator independently;
that would confound magnitude with changing question composition. The curve is
an exploratory visualization and does not replace the larger 250/59 estimands.

For each transformed A0 prediction, use three mutually exclusive outcome bins:

```text
wrong semantic structure
literal-copy error
runtime failure
```

`runtime failure` includes parse, lower, type, execute, and missing-RETURN
failures, with the failing stage retained. `literal-copy error` is deliberately
conservative: the corresponding x1 prediction must be correct, every nonliteral
ASL token must either match or have the same literal-free alpha structure, and
replacing only position-aligned numeric literals according to the frozen
source/trace mapping must execute to the right transformed answer. All remaining
wrong executable programs are `wrong semantic structure`. Report both raw ASL
accuracy and oracle-literal-corrected accuracy;
the latter estimates a lower bound on errors attributable only to numeric
transcription and is not a deployable model result.

The main Paper 1 figure is answer accuracy versus numeric scale for B1, B1L,
and each A0 seed. The desired robustness pattern is a declining direct curve and
a comparatively horizontal ASL curve. If this pattern does not occur, report it
without weakening the fixed-identity protocol.

### Magnitude-curve hypothesis and model-size decision

The next primary experiment is the complete magnitude curve, not another
two-point comparison. Plot final-answer accuracy against
`log10(source-number scale)` at the frozen factors `x1`, `x10^2`, `x10^3`,
`x10^4`, and `x10^6`. Show B1/B1L and every A0 seed, together with the ASL
mean and uncertainty or seed range. Lines may connect the points because every
factor uses the same fixed 55 parent problems, but the figure and table must
also report counts and paired uncertainty.

The directional hypothesis is:

```yaml
Direct: 66 -> 60 -> 52 -> 44 -> ...
ASL:    49 -> 48 -> 47 -> 46 -> ...
```

These values are schematic and must never be reported as observations. If the
measured direct curve declines materially with log magnitude while the ASL
curve remains comparatively horizontal, Paper 1 gains direct mechanistic
evidence that deterministic execution removes magnitude sensitivity after
successful semantic compilation. Do not make that claim before the complete
curve is available.

After completing the 0.6B curve, interpret the frozen 1.7B matched gate as a
model-size decision test:

- `crossover_shift`: the larger direct model moves the ASL crossover to a
  larger magnitude but remains magnitude-sensitive;
- `sensitivity_elimination`: the larger direct model stays approximately flat
  and closes the ASL robustness gap.

The first outcome supports a capacity-dependent or small-model advantage. A
persistent direct-versus-ASL slope difference supports a more fundamental
neural-computation versus deterministic-execution effect. Elimination of the
difference bounds the present claim to the tested small-model regime. Compare
only identical parent IDs and transformations, and never infer the gate from an
incomplete arm or mismatched denominator.

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

### Cross-size augmentation gate before 4B

Complete the current magnitude, full-audit, and Qwen3-1.7B matched-control runs
before interpreting augmentation. Select increments greedily with Qwen3-0.6B,
then retrain and evaluate the frozen winning cumulative dataset independently at
both Qwen3-0.6B and Qwen3-1.7B. Each size must use its own matched AUG0 adapter
with the same initialization, source parents, optimizer-step budget, approximate
target-token exposure, prompt, decoding policy, and untouched 250/59 identities.

Report the within-size augmentation delta and the model-size-by-augmentation
interaction for final answers and every semantic component. An increment is
portable only if its answer or binding gain is positive at both sizes without a
material regression in the other primary semantic metrics. A gain at only one
size is still informative but must be reported as size-dependent rather than
pooled into a general augmentation claim.

Advance to Qwen3-4B only after both size checks are complete. For each model,
carry forward the best passed cumulative dataset; if no augmentation passes at a
given size, retain AUG0 rather than forcing a negative increment. The first 4B
condition is a one-initialization QKVO-r8 QLoRA gate using the same U2000/E4500
parents and the frozen 250/59 evaluation. Require a 4-bit load/training memory
smoke, gradient checkpointing, no truncation, and recorded peak memory before the
full run. Compare 4B ASL against its own matched 4B direct control, not against a
smaller model's direct result. Replicate or extend the magnitude ladder only if
the one-seed gate is positive or resolves the registered scaling hypothesis.

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

B1L is complete: 154/250 (61.6%) ordinary and 26/59 (44.1%) transformed
answers. Its paired transformed degradation is -22.0 points versus -23.7 for
B1. ASL's differential robustness remains +16.9 to +25.4 points across seeds,
with all paired bootstrap intervals above zero. The bounded robustness result
therefore survives the long-output sensitivity.

The full 1,319-question A0 seed-37 audit is descriptive confirmation only:
509/1,319 answers (38.6%) are correct and 1,123/1,319 programs execute. Preserve
the frozen 250/59 experiment as the causal comparison. An independent full-audit
replication may verify reproducibility but must not be used for selection.

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
    COMPLETE for the registered 1,024-token B1 control and post-hoc B1L
    sensitivity. Ordinary ASL contribution is negative for all three seeds;
    differential factor-1,000 robustness is positive for all three seeds.

P6c.1 Finish B1L on the 59 transformed descendants and regenerate only the 18
      rows that reached the old token ceiling. Then rerun the registered
      contribution analysis against the stronger direct control.
      COMPLETE: 154/250 ordinary, 26/59 transformed, and the ASL robustness
      result survives comparison with B1L.

P6c.2 Run B1, B1L, and all A0 seeds on the frozen 55-parent magnitude ladder at
      x1, x10^2, x10^3, x10^4, and x10^6. Build the common-denominator accuracy
      curve and the ASL semantic/literal/runtime failure audit. Include the
      literal-only oracle correction as a diagnostic, never as model accuracy.

P6c.3 Make accuracy versus `log10(source-number scale)` the primary robustness
      plot. Once the 0.6B curve is complete, use the frozen 1.7B matched gate to
      test whether scaling shifts the crossover rightward or eliminates
      magnitude sensitivity. Extend the complete ladder to 1.7B only when the
      matched gate warrants the additional compute.

P6c.4 Recover actual token counts, stop reasons, ceiling hits, and timing from
      existing B0/B1/B1L/A0 records. Produce accuracy-versus-generated-token
      Pareto plots and generated-token cost versus log magnitude; rerun only
      records whose saved telemetry is insufficient.

P6c.5 Screen the frozen Qwen reasoning ladder R0--R4 on the identical 55-parent
      cohort. Cross thinking on/off at matched ceilings where needed, test the
      bounded no-thinking ASL budget ladder, and admit only predeclared Pareto or
      diagnostic endpoints to the 250-item confirmation and 1.7B replication.

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

P6f.1 Freeze the winning cumulative augmentation after the 0.6B greedy ladder.
      Retrain matched AUG0 and winning-augmentation adapters at both 0.6B and
      1.7B, then evaluate the same untouched 250/59 identities and semantic
      components. Report within-size deltas and the size-by-augmentation
      interaction; never carry a negative increment into the next model.

P6f.2 After the two-size augmentation gate, implement and memory-smoke Qwen3-4B
      QKVO-r8 QLoRA. Train the best passed data condition, run the same 250/59
      ASL and matched direct controls, and only then decide whether 4B seeds or
      the complete magnitude ladder are warranted.

P6g On the best passed data cell, run T1 slot-level auxiliary supervision, then T2
    runtime-symbol pointers, then T3 clause-local contrastive binding, and finally
    T4 bounded semantic reranking. Test QKVO+MLP and modality-specific adapters only
    after a semantic-supervision condition passes. Never use dynamic ICL.

P7  After B1L, the full-audit verification, the 1.7B gate, and magnitude
    diagnostics are frozen, revise F3 around grounded identity + alpha-renamable
    locals. Optimize F3/E3 for the observed 38.5% semantic compiler accuracy,
    not for arithmetic that the deterministic runtime already solves.

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

P16 Start `AGENTS_paper1_operator_complexity_ladder.md` after the active matched
    Qwen3-1.7B magnitude gate, running Qwen3-0.6B first. Augmentation selection
    may continue concurrently, but augmentation portability and the first 4B
    gate no longer block the 0.6B operator ladder. Build its registry/runtime/
    generator on CPU while the 1.7B control finishes. Treat O0--O6 as
    categorical capability families;
    estimate complexity slopes and crossovers only along frozen operation-count,
    dependency-depth, nesting-depth, and magnitude axes within a family. Compare
    direct reasoning, a generic expression tool, primitive ASL, and matched
    semantic-functor ASL before attributing gains to CogCop decomposition.

P17 After the 0.6B operator-complexity ladder is complete, add the matched R1J
    randomized-magnitude control. Apply independent deterministic
    `Uniform(-0.30,+0.30)` jitter to each eligible source quantity after its
    registered `10^k` scaling for `k in {0,2,3,4,6}`; use at least three frozen
    perturbation seeds, recompute intermediates and answers through the hidden
    authoritative trace, and freeze one common parent intersection. Evaluate
    identical transformed questions under Direct and ASL and compare against
    the existing strict power-of-ten curve. Do not mix R1J into the completed
    R1 estimand.
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

---

# 29. Journal-readiness program

Paper 1 is evidence-rich enough to stand alone, but it is not submission-ready
until the matched robustness data, result-use bridge, transformed-data validity,
and comparator fairness are closed.  Do not add the full long-context
architecture to this manuscript.  Keep the main paper centered on:

```text
NL
-> generated ASL
-> deterministic execution
-> controlled result reinjection
-> continued generation
```

The complete long/multipart ladder---learned block boundaries, dynamic adapter
switching, 2/4/8/16 interventions, persistent world memory, block-causal masks,
typed K/V injection, and PRA selection---is a separate follow-up paper.  Paper 1
may include one short, oracle-boundary, two-part cascade only as the bridge from
terminal execution to downstream result use.

## 29.1 Execution order

The following order supersedes any older ordering where running more inference
precedes validation of the transformed panel.  Existing predictions remain
preserved, but do not spend multiple seeds on a panel or scorer that may change.

```text
J0  Preserve the current seed-17011 checkpoints as developmental evidence.

J1  Audit transformation semantic consistency and freeze eligibility rules.
    Detect contradictory totals, impossible remaining counts, fractional-object
    answers, unit/world-plausibility violations, broken cross-sentence numeric
    dependencies, and stale aggregate quantities.  Rules must be condition-blind.

J2  Audit Direct endpoints and freeze scorer v2.
    Separate semantic, arithmetic, formatting, and token-ceiling failures.
    Deterministically rescore saved generations before rerunning inference.

J3  Freeze journal panel v2, manifests, exclusions, hashes, and common parent
    intersections.  Report both the original all-row analysis and the audited
    validity subset; never silently replace the observed panel.

J4  Complete Direct/ASL O0/O1/O5/O6 x scale x jitter for seeds
    17011/17023/17037 on panel v2.  Run seed-major and preserve paired outcomes.

J5  Run the long-token Direct sensitivity on the same panel and report ceiling
    hits, formatting-only recoveries, semantic errors, and arithmetic errors.

J6  Run short-context result reinjection with existing ASL checkpoints.

J7  Run one bounded two-part cascade with oracle boundaries.

J8  Train and evaluate a compute/exposure-matched Direct-answer LoRA control.

J9  Evaluate untouched SVAMP and GSM-Plus before any training on either source.

J10 Freeze analyses, figures, qualitative examples, and claim table; then perform
    the editorial revision, build, commit, and push.
```

## 29.2 Transformation audit

The independent `Uniform(-0.30,+0.30)` jitter tests non-round numeric changes,
not generic randomness.  A transformed row is admissible only when all textual
mentions, hidden computations, and the reference answer remain mutually
consistent.  The audit must distinguish:

```text
arithmetically replayable
semantically coherent
world-plausible
object-integral where required
unit-consistent
free of stale totals or durations
```

World-plausibility is a declared sensitivity stratum, not a license to remove
hard examples after observing model outcomes.  Freeze deterministic exclusions
and a stratified manual audit before confirmatory inference.  In particular,
flag examples that imply negative inventory, fractional people/objects, a target
already exceeded when the wording asks for "more," or a stated total that no
longer equals transformed components.

## 29.3 Direct failure taxonomy and scorer v2

Do not equate every endpoint mismatch with failed reasoning.  Rescore existing
text into mutually exclusive primary labels:

```text
CORRECT_ENDPOINT
CORRECT_VALUE_FORMAT_MISS
CORRECT_INTERMEDIATE_THEN_OVERRIDE
CORRECT_INTERMEDIATE_THEN_TRUNCATED
SEMANTIC_STRUCTURE_ERROR
RELATION_OR_BINDING_ERROR
ARITHMETIC_EXECUTION_ERROR
NO_SCORABLE_VALUE
```

The deterministic endpoint parser must recognize the frozen allowed answer
wrappers, including plain numbers, commas, decimals, fractions, `Answer:`,
boxed forms, and angle-bracket wrappers.  Never infer an answer from an arbitrary
earlier number when no endpoint is present.  Report registered strict scoring
and scorer-v2 sensitivity together.

## 29.4 Short-context reinjection matrix

The current official ASL evaluator stops after executing the complete generated
program.  Restore result-use measurement without changing ASL generation:

```text
I0 TERMINAL_RUNTIME_VALUE       current endpoint
I1 VALUE_TEXT                  VALUE: <computed-value>
I2 TYPED_RESULT_TEXT           value + provenance/status
I3 ASL_PLUS_RESULT_TEXT        generated ASL + typed result
I4 ORACLE_VALUE                result-use ceiling
I5 CORRUPTED_OR_STALE_VALUE    trust/calibration control
I6 CONTINUATION_NO_VALUE       extra-generation control
```

The primary I1--I3 values come only from the model-generated ASL, whether right
or wrong.  A runtime-success flag verifies execution, not semantic truth.  Use a
fixed continuation prompt and fixed formatting across every record.  Never
select the format per question or outcome.

Minimum typed record:

```text
VALUE: <value>
EXECUTION_STATUS: VERIFIED
SEMANTIC_GROUNDING: UNVERIFIED
SOURCE: MODEL_GENERATED_ASL
```

Report runtime-value correctness separately from post-injection final-answer
correctness.  Also report value uptake conditional on a correct runtime value,
wrong-value uptake, correction, correct-value override, contradiction, injected
tokens, continuation tokens, and latency.

## 29.5 Two-part cascade boundary

Paper 1 includes only a bounded bridge experiment:

```text
part A -> generated ASL A -> execute -> inject result A
part B depends on result A -> final answer
```

Use short inputs, oracle part boundaries, one intermediate result, and no
retrieval or persistent cross-document memory.  Compare no value, generated-ASL
value, oracle value, and corrupted value.  This tests whether a correct result
can help a later step and whether a wrong result propagates.  Learned selection,
multiple interventions, long distractor regions, dynamic adapters, typed K/V,
and PRA remain out of Paper 1.

## 29.6 Comparator and OOD requirements

Train one Direct-answer LoRA from the same base model using the same eligible
training questions, optimizer-step budget, and approximate target-token exposure
as the primary ASL adapter.  Compare:

```text
base Direct
budget-matched Direct LoRA
ASL LoRA + deterministic runtime
ASL LoRA + runtime + reinjection
```

This control separates architectural delegation from generic supervised
adaptation.  Preserve SVAMP and GSM-Plus as evaluation-only OOD datasets until
their first reported evaluation is frozen.  Report source-overlap and semantic-
pattern audits before interpreting transfer.

## 29.7 Manuscript structure

Keep the journal main text to the causal argument and move implementation and
diagnostic detail to appendices/supplement.

Main text:

```text
1. Motivation and precise claim boundary
2. NL->ASL->runtime->reinjection architecture
3. Frozen datasets, models, and matched controls
4. Ordinary accuracy and magnitude/jitter robustness
5. Operator/expression complexity
6. Result use and bounded cascade
7. OOD and matched-Direct-LoRA results
8. Limitations and conclusion
```

Appendices/supplement:

```text
ASL grammar and lowering
dataset construction and provenance
all prompts and hyperparameters
semantic-validity audit and exclusions
scorer-v2 taxonomy and fixtures
full per-seed tables and confidence intervals
failure decomposition and qualitative examples
token/latency/memory accounting
negative architecture/loss/augmentation results
reproducibility commands and artifact hashes
```

## 29.8 Submission gates

The journal draft is evidence-complete only when:

- all three audited matrix seeds are complete;
- the result survives scorer-v2 and long-token sensitivity;
- runtime correctness and post-injection use are reported separately;
- the two-part cascade has no hidden gold state or future-source leakage;
- the matched Direct LoRA control is complete;
- at least SVAMP and GSM-Plus have untouched OOD results;
- transformed-data audit decisions are frozen independently of outcomes;
- all headline claims map to tables, figures, manifests, and scripts.

Editorial readiness additionally requires a shortened main narrative, complete
appendices, cross-reference and terminology normalization, reproducibility audit,
PDF build, and one claim-by-claim consistency pass.
