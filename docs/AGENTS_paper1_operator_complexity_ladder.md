# AGENTS — Paper 1 Operator-Complexity Ladder
## GSM8K-Derived O0→O6 Benchmark for Direct Qwen vs ASL/CogCop

## Status

Future Paper 1 / Paper 1.x experiment family extending the current matched GSM8K result beyond operand magnitude.

Current evidence suggests:
- direct Qwen3-0.6B is stronger on ordinary GSM8K;
- direct reasoning degrades materially on ×1000 numeric perturbations;
- ASL/CogCop is much flatter under magnitude changes;
- ASL generation is compact and execution is deterministic;
- the dominant ASL bottleneck is semantic compilation, not syntax/runtime.

The next question is:

> Does this factorization advantage persist when computational difficulty increases through a richer operator/function vocabulary rather than only larger numbers?

The benchmark must measure three headline quantities:

```text
accuracy
robustness
neural reasoning tokens
```

## Integration and causal-design constraints

This is a Paper 1.x extension that starts immediately after the active matched
Qwen3-1.7B magnitude gate. Run the operator-complexity ladder first with
Qwen3-0.6B; augmentation selection may continue concurrently on another worker
but does not block the ladder. Defer the first 4B gate until the 0.6B ladder has
produced its initial O1 and gate-family results. Registry, runtime, generator,
and validator implementation may proceed on CPU while the 1.7B gate is active,
but it must not delay or alter that frozen experiment.

O0--O6 are capability-family labels, not points on a validated scalar measure
of difficulty. Trigonometry, geometry, and algebra change domain vocabulary and
semantic grounding as well as mathematical operations. Therefore the primary
causal analysis must cross a categorical `operator_family` with independently
measured complexity axes:

```text
primitive_operation_count: 1, 2, 4
dependency_depth:          1, 2, 4
function_nesting_depth:    1, 2, 3
operand_magnitude:         frozen bands
entity_count:              frozen bands
prompt_tokens:             matched or adjusted
```

Never interpret a slope across the arbitrary O0--O6 numbering as an operator-
complexity effect. Estimate ordered slopes within an operator family while
holding the other axes fixed; treat cross-family differences as categorical.

Every selected family must include two matched symbolic representations:

```text
AP  primitive ASL whose execution graph is isomorphic to the required math
AS  semantic functor ASL with deterministic lowering to the same primitive graph
```

AP versus AS separates exact-execution benefits from ontology/task-
decomposition benefits. In addition to direct concise/reasoning controls, add a
generic expression-tool baseline in which Qwen emits a standard arithmetic or
symbolic expression for the same deterministic backend. If the generic tool
matches ASL, the evidence supports tool delegation rather than an ASL-specific
advantage. If AS improves over AP, attribute the gain to semantic decomposition,
not parentheses, syntax, or deterministic execution alone.

Expose one fixed versioned operator-registry description to every ASL example.
Add a matched direct registry-context control that sees the same description but
cannot execute it, so extra prompt context is not confounded with runtime access.
No condition receives the gold operator, gold program, answer, rationale,
intermediate state, or problem-dependent demonstrations.

---

# 1. Operator ladder

Freeze these capability families:

```text
O0: + - × ÷
O1: square, cube, sqrt
O2: powers, roots, abs, min/max
O3: sin/cos, logarithms
O4: distance, area, volume
O5: solve linear equation
O6: solve quadratic equation
```

The labels describe operator coverage. Evaluation sets are balanced,
non-cumulative family cells. Cumulative combinations are permitted only as an
explicitly labeled training curriculum after level-specific gates pass.

---

# 2. Models

Primary family:

```text
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
```

Run policy:

## M0 — Qwen3-0.6B
Run the full operator ladder first.

## M1 — Qwen3-1.7B
Run after the 0.6B datasets/test sets are frozen. Use identical data, prompts, ASL, runtime, evaluation, and token accounting.

## M2 — Qwen3-4B
Gate 4B. Do not run immediately.

Run 4B only if 0.6B→1.7B shows at least one material result:
- ASL compiler improves substantially;
- ASL-vs-direct interaction changes;
- robustness crossover shifts;
- reasoning-token scaling differs;
- O5/O6 remain sufficiently difficult to justify more capacity.

---

# 3. Matched conditions

For every model and level:

```text
D0  direct concise
D1  direct native reasoning/thinking
D1L direct long-budget reasoning

A0  NL -> ASL -> deterministic runtime
```

The primary direct baseline is D1L, with enough output budget to avoid an artificial ceiling.

A0 must receive:
- the identical NL question;
- no gold operator choice;
- no gold ASL;
- no oracle state.

---

# 4. Main hypotheses

## H1 — Direct neural degradation
As operator sophistication grows, direct accuracy falls and/or neural reasoning-token usage grows.

## H2 — CogCop execution invariance
Once ASL is semantically correct, deterministic execution remains nearly exact across O0–O6.

## H3 — Compiler bottleneck
CogCop failures are increasingly attributable to NL→ASL semantic errors rather than mathematical execution.

## H4 — Model-size leverage
Larger Qwen models improve NL→ASL semantic compilation more than they improve deterministic execution.

## H5 — Token advantage
CogCop replaces part of long autoregressive reasoning with compact symbolic programs and cheap deterministic compute.

## H6 — Crossover
Direct reasoning may dominate at low operator complexity, while ASL/CogCop may become competitive or superior at higher complexity or under perturbation.

Do not assume a crossover exists; measure it.

---

# 5. Dataset design

Create a new GSM8K-derived family:

```text
GSM8K-OC-v1
```

with subsets:

```text
GSM8K-OC-O0
...
GSM8K-OC-O6
```

The objective is to hold linguistic style and semantic-story complexity relatively stable while increasing mathematical operator sophistication.

Do not simply concatenate unrelated external math datasets into this experiment.

---

# 6. Dataset construction principle

Generate the symbolic world/program first, then render natural language.

Pipeline:

```text
1. choose story template
2. choose operator level
3. sample valid parameters
4. construct gold ASL/CCIR
5. execute gold program
6. generate NL realization
7. verify answer
8. semantic lint
9. deduplicate / leakage audit
10. freeze
```

The deterministic program/runtime is authoritative.

A large teacher LLM may produce paraphrases, but it must not define mathematical truth.

---

# 7. Splits and leakage

Do not derive training data from official held-out test identities.

Split by parent semantic/template identity, not merely row.

Every record should include:

```text
parent_template_id
operator_level
operator_signature
semantic_signature
surface_variant_id
numeric_variant_id
generator_version
validation_status
```

Prevent parent/template overlap across train/dev/test.

---

# 8. Initial dataset size

Per operator level, initial target:

```text
train: 2,000
dev:     100
test:    250
```

This mirrors the useful U2000 regime.

If generation is high quality and compute permits:

```text
train: 4,000–8,000
dev:       250
test:      500
```

Freeze test sets before training.

---

# 9. Difficulty control inside each level

Each level should have:

```text
easy
medium
hard
```

based on:
- operator count;
- dependency depth;
- function nesting depth;
- entity count;
- intermediate variable count;
- number of references.

Store all of these as numeric fields and balance or condition on them. In
particular, compare AP and AS on identical latent programs, parameters, NL
realizations, and split identities. Hold out complete generator-template and
paraphrase families, not only rendered rows.

Do not let O6 simply become much longer prose than O1.

The goal is operator sophistication, not prompt-length scaling.

---

# 10. O0 — Arithmetic baseline

Operators:

```text
+
-
*
/
```

Anchor to current GSM8K-style ASL.

Example:

```text
may = april / 2
total = april + may
RETURN total
```

Use O0 to recover:
- current accuracy baseline;
- magnitude robustness;
- reasoning-token baseline.

---

# 11. O1 — Square, cube, sqrt

Add:

```text
square(x)
cube(x)
sqrt(x)
```

Example templates:
- square area from side;
- side from square area;
- cube volume;
- cubed quantity;
- simple Pythagorean-style substeps if they use only square/sqrt.

ASL:

```text
area = square(side)
RETURN area
```

```text
side = sqrt(area)
RETURN side
```

Prefer exact roots first.

---

# 12. O2 — Powers, roots, abs, min/max

Add:

```text
pow(x,n)
root(x,n)
abs(x)
min(a,b,...)
max(a,b,...)
```

Examples:

```text
value = pow(base, exponent)
distance = abs(a-b)
largest = max(a,b,c)
```

Start with integer powers and exact roots, then add deterministic approximate cases.

---

# 13. O3 — Trigonometry and logarithms

Add:

```text
sin(x)
cos(x)
log(x)
log10(x)
ln(x)
```

Angle units must be explicit:

```text
degrees
radians
```

Start with exact/common values:
- 0°, 30°, 45°, 60°, 90°;
- log10(10^k);
- ln(e^k).

Then add general deterministic numeric cases.

---

# 14. O4 — Distance, area, volume

Introduce higher-level mathematical functors:

```text
distance2d(x1,y1,x2,y2)
circle_area(r)
rectangle_area(w,h)
triangle_area(base,height)
box_volume(l,w,h)
cylinder_volume(r,h)
sphere_volume(r)
```

Example:

```text
r = distance2d(cx, cy, px, py)
RETURN r
```

This level tests whether the LLM can select/bind a higher-level formal operation rather than manually expanding every primitive operation.

---

# 15. O5 — Linear equation solving

Add a deterministic algebra solver.

Support at minimum:

```text
a*x + b = c
```

and simple equivalent forms.

Freeze one ASL representation, e.g.:

```text
eq = equation(a*x + b, c)
x = solve(eq, x)
RETURN x
```

or:

```text
x = solve_linear(a,b,c)
RETURN x
```

Do not mix representations in the first experiment.

Natural-language templates:
- age;
- price/count;
- distance/time;
- balance;
- simple mixture;
- unknown quantity from total.

---

# 16. O6 — Quadratic equations

Add:

```text
solve_quadratic(a,b,c)
```

or a generic equation representation.

Example:

```text
roots = solve_quadratic(a,b,c)
answer = max(roots)
RETURN answer
```

Initial questions should make root selection unambiguous:
- positive root;
- larger root;
- physically valid length.

Complex roots can be deferred.

---

# 17. Runtime/operator registry

Use a versioned registry:

```text
operator_registry = {
  O0: arithmetic,
  O1: power_basic,
  O2: extended_numeric,
  O3: transcendental,
  O4: geometry,
  O5: linear_solver,
  O6: quadratic_solver
}
```

Every operator records:

```text
name
arity
argument_types
return_type
domain restrictions
exactness class
implementation version
```

Use deterministic symbolic/numeric execution, e.g. SymPy where appropriate.

The LLM must not implement the math operator itself once it has emitted ASL.

---

# 18. Exactness classes

Classify outputs as:

```text
EXACT_INTEGER
EXACT_RATIONAL
EXACT_SYMBOLIC
DETERMINISTIC_APPROX
```

Examples:

```text
sqrt(49)       -> EXACT_INTEGER
sqrt(2)        -> EXACT_SYMBOLIC or fixed deterministic approximation
log10(1000)    -> EXACT_INTEGER
```

Freeze tolerance rules before model evaluation.

---

# 19. Gold runtime ceiling

Before any model training/evaluation, gold ASL should achieve approximately:

```text
100% parse
100% type-valid
100% execution
100% answer
```

Quarantine dataset rows that fail deterministic validation.

Runtime correctness is infrastructure evidence, not model evidence.

---

# 20. Training protocol

Start with the successful U2000/E4500 pattern where possible:

```text
2,000 unique programs
4,500 total exposures
QKVO LoRA-r8
```

Two designs:

## T1 — Level-specific
Separate adapter per O-level.

Best for causal diagnosis.

## T2 — Cumulative
Train on:

```text
O0
O0+O1
O0+O1+O2
...
```

Best for an extensible general math CogCop.

Run T1 first on gate levels. Add T2 only after individual levels work.

---

# 21. Initial execution ladder

Avoid training every possible cell immediately.

## Phase 1 — Qwen3-0.6B

Run:

```text
O0
O1
O3
O5
O6
```

first.

If coherent, fill:

```text
O2
O4
```

## Phase 2 — Qwen3-1.7B

Run the same frozen tests, starting with:

```text
O0
O1
O3
O5
O6
```

then complete the ladder if justified.

## Phase 3 — Qwen3-4B

After analyzing 0.6B and 1.7B, run only:

```text
O0
one mid-level
O6
observed crossover level, if different
```

Run full 4B O0–O6 only if the interaction is important enough.

---

# 22. Accuracy metrics

For all conditions:

```text
final_answer_accuracy
```

For ASL additionally:

```text
parse_valid
lowerable
type_valid
executable
semantic_return
semantic_state
alpha_return
alpha_state
operator_selection_accuracy
argument_binding_accuracy
literal_binding_accuracy
dependency_accuracy
query_target_accuracy
```

For O5/O6 add:

```text
equation_setup_accuracy
unknown_variable_accuracy
solution_selection_accuracy
```

---

# 23. Robustness metrics

For each frozen test identity derive paired perturbations.

## R0 — Original
Base question.

## R1 — Magnitude
Where valid:

```text
×10
×100
×1000
×10000
×1000000
```

After the operator ladder is complete, add a separate randomized-magnitude
control rather than folding it into R1. For each eligible source quantity and
registered scale factor, derive an independent deterministic jitter from the
frozen parent identity, semantic role, factor, and perturbation seed:

```text
scaled_value = original_value * 10^k
jitter       ~ Uniform(-0.30, +0.30)
new_value    = deterministic_quantize(scaled_value * (1 + jitter))
k            in {0, 2, 3, 4, 6}
```

Use at least three frozen jitter seeds. Recompute every hidden intermediate and
the final answer through the authoritative program; never adjust the answer by
an independent shortcut. Preserve protected dimensionless constants, reject
zero denominators and invalid domains, require exact execution, and freeze one
common parent intersection across all factors and seeds. Direct and ASL arms
must receive identical transformed questions. Report this as `R1J`, separately
from strict global scaling, because independent role-wise jitter changes
relative quantities and tests semantic binding rather than scale recognition.

## R2 — Entity renaming
Names/objects changed, program preserved.

## R3 — Surface paraphrase
Wording changed, semantics preserved.

## R4 — Distractor numbers
Add irrelevant plausible quantities.

## R5 — Equivalent formulation
Use mathematically equivalent wording only when deterministic.

Analyze each perturbation independently first.

---

# 24. Robustness statistic

For condition C:

```text
degradation(C) =
  Acc_transformed(C) - Acc_original(C)
```

## Primary operator-complexity dashboard

Freeze the following dashboard before inspecting model outcomes. Report it for
every pilot and promoted operator family on identical test identities:

```text
1. direct answer accuracy by operator family
2. direct change from matched O0
3. AP and AS answer accuracy by operator family
4. AP and AS parse, operator-selection, argument-binding, dependency,
   executable, and semantic-state accuracy
5. AP-minus-direct and AS-minus-direct paired answer gaps
6. AS-minus-AP paired answer gap
7. deterministic runtime accuracy conditional on a semantically correct program
8. generated neural tokens and ceiling-hit rate for every condition
```

The three primary questions are:

```text
Does direct accuracy decline as required computation becomes more complex?
Does NL-to-ASL compilation itself decline when the operator vocabulary changes?
Does the ASL-minus-direct gap become less negative or positive on harder families?
```

Do not treat O0--O6 as equally spaced scalar levels. Compare categorical
operator families and estimate complexity effects only within matched operation
count, dependency-depth, nesting-depth, entity-count, prompt-length, and
operand-magnitude cells. Report paired flips as well as aggregate rates so a
stable total cannot hide different solved examples.

Run strict operator-family comparisons first. Only after those results are
frozen, generate and evaluate R1J independently jittered numeric variants. R1J
is a second-stage interaction test:

```text
condition x operator family x randomized magnitude
```

Its purpose is to determine whether any operator-complexity crossover survives
non-power-of-ten numeric changes, not to tune or select the operator datasets.

CogCop relative robustness:

```text
Delta_robust =
  degradation(ASL) - degradation(Direct)
```

Use paired identity bootstrap and exact paired tests.

---

# 25. Neural reasoning-token metrics

For direct Qwen, count all generated thinking/reasoning output tokens.

For ASL, count generated ASL tokens.

Do not count deterministic runtime operations as neural tokens.

Report:

```text
mean
median
p90
p95
max
ceiling-hit rate
tokens on correct examples
tokens on incorrect examples
```

Derived:

```text
token_ratio =
  direct_reasoning_tokens / asl_tokens
```

and:

```text
tokens_per_correct_answer
```

---

# 26. Accuracy-token Pareto frontier

For each model and O-level plot:

```text
x = neural generated tokens
y = accuracy
```

Conditions:

```text
D1
D1L
A0
```

A CogCop result can be valuable even if raw accuracy is lower when it is materially:
- more robust;
- much shorter;
- cheaper;
- exact after compilation.

Do not collapse the tradeoff into one scalar in the main paper.

---

# 27. Token-budget sensitivity

For selected levels on 0.6B:

Direct:

```text
256
512
1024
2048
```

ASL:

```text
64
128
256
384
```

Measure:

```text
accuracy(token_budget)
```

This separates model capability from reasoning-budget limitation.

---

# 28. Core derived quantities

For model M and operator level O:

```text
Delta_accuracy(M,O) =
  Acc_ASL(M,O) - Acc_Direct(M,O)
```

Model-size interaction:

```text
I_accuracy(O) =
  [ASL_1.7(O) - Direct_1.7(O)]
  -
  [ASL_0.6(O) - Direct_0.6(O)]
```

Token interaction:

```text
I_tokens(O) =
  [Tokens_Direct - Tokens_ASL]_1.7
  -
  [Tokens_Direct - Tokens_ASL]_0.6
```

---

# 29. Crossover analysis

Because O0--O6 are categorical families, do not define a first crossover across
their numeric labels. Define a crossover only along an ordered, measured axis
within one frozen family:

```text
C*(family, axis) =
first registered axis value where
ASL accuracy >= direct accuracy
```

and separately:

```text
C*_pareto(family, axis) =
first registered axis value where ASL is Pareto-superior
on a chosen combination of
accuracy / robustness / neural-token cost
```

Cross-family AP/AS/direct contrasts remain valid, but they are not crossover
locations. Do not assume either within-family crossover exists.

---

# 30. Failure decomposition

ASL:

```text
syntax
type
unsupported operator
wrong operator
wrong argument
wrong literal
wrong entity/path
wrong dependency
wrong query target
runtime domain error
```

Direct reasoning, where deterministically detectable:

```text
wrong semantic setup
arithmetic/math execution error
reasoning truncation
answer extraction error
```

Use LLM judges only for optional diagnostics, never headline scores.

---

# 31. Fairness

Direct:
- same NL question;
- thinking enabled;
- generous enough token budget;
- no calculator/ASL.

ASL:
- same NL question;
- no gold ASL;
- no gold operator hint;
- no oracle intermediate values;
- access only to the registered runtime/operator vocabulary.

If deployment assumes operator definitions are available, expose the same fixed registry description at test time for all ASL examples.

---

# 32. Statistics

Accuracy:
- Wilson interval per condition;
- exact McNemar for paired direct-vs-ASL.

Robustness:
- paired identity bootstrap;
- exact original/transformed flip analysis.

Tokens:
- bootstrap identity intervals;
- medians/distributions, not just mean.

Model-size interactions:
- bootstrap over shared identities.

---

# 33. Seed policy

0.6B:
- exploratory seed first;
- three seeds for positive/material cells.

1.7B:
- one exploratory matched seed first;
- replicate only if interaction is scientifically meaningful.

4B:
- one exploratory gate;
- replicate only if it establishes a major new interaction.

---

# 34. Primary plots

Create:

```text
A. accuracy vs O0...O6
B. neural tokens vs O0...O6
C. accuracy vs neural tokens
D. robustness degradation vs O-level
E. ASL-minus-direct accuracy vs O-level
F. model-size difference-in-differences
G. accuracy vs numeric magnitude within selected O-levels
```

---

# 35. Main scientific interpretations

Possible outcomes:

## A — Direct wins accuracy at all levels
CogCop may still win robustness/token efficiency.

## B — Direct falls with O-level while ASL is flatter
Strong evidence for compute factorization.

## C — 1.7B improves ASL much more than direct
Strong evidence that model scaling helps semantic compilation.

## D — Both improve similarly with model size
CogCop advantage remains mainly robustness/cost.

## E — ASL collapses as operator vocabulary grows
The semantic representation/training strategy is the bottleneck; prioritize F3/E3.

---

# 36. Go/no-go gates

Before the full gates, run a one-seed behavior pilot at every new operator
family:

```text
train: 250 unique records
dev:    30 records
test:  100 records
epochs: 3
seed:   99173
```

Run direct reasoning, AP, and semantic AS first. Add the generic-expression
baseline in the same pilot analysis, but do not let it delay the first AP/AS
readout. Promote a family to 2,000/100/250 training only if the pilot shows a
useful accuracy, representation, robustness, or token signal. A null pilot is
diagnostic and should stop expensive expansion rather than trigger a larger run
automatically.

## G0
Gold runtime ceiling must be essentially perfect.

## G1
0.6B must learn O1 above a meaningful baseline before expanding aggressively.

## G2
At least one mid-level O3/O4 condition must show usable ASL compilation before heavy O5/O6 campaigns.

## G3
Run 1.7B only on frozen datasets/tests.

## G4
Run 4B only after 1.7B leaves a scientifically meaningful scaling/crossover question.

---

# 37. Suggested artifact layout

```text
artifacts/paper1/operator_complexity_v1/
  manifests/
  operator_registry/
  o0/
  o1/
  o2/
  o3/
  o4/
  o5/
  o6/
  robustness/
  token_metrics/
  analysis/
```

Configs:

```text
configs/paper1/operator_complexity/
```

Code:

```text
src/ccpu/paper1/operator_complexity/
  registry.py
  generate.py
  validate.py
  perturb.py
  runtime.py
  evaluate.py
  token_metrics.py
  analysis.py
```

---

# 38. Required tests

Add deterministic tests for:
- operator registry versions;
- argument type checking;
- exactness classes;
- root/log/trig domain rules;
- geometry formulas;
- linear solver;
- quadratic solver;
- ASL→CCIR lowering;
- perturbation semantic preservation;
- parent-template split isolation;
- token accounting;
- direct/ASL identity matching.

---

# 39. Immediate work order

```text
P-2 Finish the frozen magnitude, token-budget, and 1.7B matched controls.

P-1 Complete the active matched 1.7B magnitude control. Allow the independent
     0.6B augmentation gate to continue concurrently, but do not make it or the
     first 4B gate a prerequisite for 0.6B operator-complexity runs.

P0  Freeze GSM8K-OC-v1 schema.

P0a Freeze independent complexity axes and non-overlapping template splits.

P0b Freeze AP primitive, AS semantic-functor, generic-expression-tool, and
    matched registry-context controls.

P1  Implement versioned O0–O6 operator registry.

P2  Implement O1 runtime and generator.

P3  Generate/validate O1 train/dev/test.

P3a Before full O1 training, freeze and run the 250/30/100, one-seed O1 pilot.
    Use three short epochs (750 exposures per adapter) and matched AP/AS test
    identities. Apply the same pilot-first gate independently to O3, O5, and O6.

P4  Run the 0.6B direct + generic-tool + AP + AS O1 gate immediately after the
    active 1.7B magnitude control completes.

P5  Implement O2/O3 operators and data.

P6  Run 0.6B O3 gate.

P7  Implement O4 geometry.

P8  Implement O5 linear equations.

P9  Implement O6 quadratics.

P10 Run 0.6B O0/O1/O3/O5/O6.

P11 Fill O2/O4 if trends are coherent.

P12 Freeze robustness variants and token-budget analysis.

P13 Run matched 1.7B ladder.

P14 Analyze within-family measured complexity axes:
    accuracy
    robustness
    neural reasoning tokens
    model-size interactions
    crossover/Pareto behavior.

P15 Decide 4B gate.

P16 If justified, run 4B on:
    O0
    selected mid-level
    O6
    observed crossover level.

P17 Update Paper 1 / Paper 1.x with operator-complexity scaling.

P18 After the operator ladder is complete, freeze and run the matched R1J
    independently jittered magnitude campaign. Compare its Direct/ASL curves
    with strict R1 scaling and report whether the crossover survives when
    source quantities are no longer exact powers-of-ten copies.
```

---

# 40. Paper endpoint

The central question is:

> As mathematical computation becomes more sophisticated, how do direct neural reasoning and learned semantic compilation into deterministic CogCops differ in accuracy, robustness, and neural inference cost?

The result need not show CogCop winning every task.

A meaningful outcome may be:

```text
Direct:
  stronger easy-task accuracy
  increasingly long reasoning
  increasing sensitivity to computational complexity

CogCop:
  weaker semantic compiler initially
  compact neural output
  deterministic execution
  flatter robustness curve
```

If increasing model size preferentially improves semantic compilation while deterministic execution preserves its robustness/token advantage, this supports the broader CogCop scaling hypothesis:

> Scale LLMs for semantic interpretation and formalization; delegate well-defined computation to specialized processors instead of spending increasingly large autoregressive reasoning budgets on exact execution.
