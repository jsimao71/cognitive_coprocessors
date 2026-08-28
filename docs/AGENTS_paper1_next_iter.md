# AGENTS — Paper 1 Next Iteration

## Title
Reflex Computation: Hard-Arithmetic Scaling and Semantic Execution Blocks

## Mission
Run the next evidence-focused iteration of Paper 1 after the first XPU pilot showed that:
- reflex calculator accuracy tied LLM-only on easy arithmetic;
- explicit tool use was numerically better;
- calculator execution itself was exact when invoked;
- the main failures were expression exposure, premature/intermediate-expression selection, and result use/override;
- strict ASCII syntax also missed equivalent LaTeX/Unicode arithmetic forms.

This iteration should not merely enlarge the original easy benchmark. It should deliberately create regimes where exact arithmetic is difficult for the LLM but trivial for the calculator, and compare multiple interface designs under matched calculator capability.

The central question is:

> When arithmetic is genuinely hard, can automatic or lightweight semantic execution interfaces preserve exactness with lower invocation overhead than conventional explicit tool calling?

## Main hypotheses

### H1 — hard-arithmetic advantage
As arithmetic difficulty increases, LLM-only and matched-prompt accuracy should degrade substantially while any condition that successfully reaches the exact calculator should remain high.

The decisive evidence is the difficulty-by-condition interaction, not easy arithmetic accuracy.

### H2 — reflex value becomes visible only above a difficulty threshold
The original pilot used expressions that the model frequently solved unaided. The next benchmark should identify the region where:
- LLM-only exactness falls;
- calculator engine exactness remains 100%;
- reflex/tool interface quality becomes the dominant variable.

### H3 — explicit semantic execution blocks improve selection reliability
A delimited execution block such as:

```calculator
15246377 * 746647383
```

should reduce:
- premature subexpression interception;
- ambiguous expression boundaries;
- LaTeX/Unicode surface-form misses;
- unnecessary verbose tool-selection syntax.

This is a bridge between strict reflex syntax and later learned semantic interrupts.

### H4 — explicit tools remain an upper interface baseline when selected correctly
Do not expect the reflex or block interface to beat a perfectly selected explicit calculator on engine correctness.

The potential advantage is in:
- invocation token cost;
- selection reliability;
- robustness as capability count grows;
- lower formatting burden;
- simpler local execution.

### H5 — many-capability pressure may expose explicit-tool routing cost
If feasible, add a controlled tool-catalog ablation where several irrelevant tools are registered alongside calculator.

Test whether explicit tool selection degrades or becomes more expensive as the catalog grows while calculator-specific reflex/block routing remains stable.

This is secondary to the hard-arithmetic experiment.

## Experimental conditions

Minimum conditions:

1. **LLM only**
   - no calculator;
   - direct exact-answer instruction.

2. **Matched prompt**
   - no calculator;
   - encourage careful arithmetic/checking.

3. **Explicit calculator tool**
   - model must explicitly choose calculator and provide arguments;
   - same calculator engine as all assisted conditions.

4. **Strict reflex**
   - existing generation-time arithmetic watcher;
   - automatic evaluation on recognized complete expression.

5. **Normalized reflex**
   - strict reflex plus deterministic surface normalization:
     - `\\times`, `×` -> `*`
     - `\\div`, `÷` -> `/`
     - Unicode minus -> `-`
     - compatible parenthesis/spacing normalization
   - no semantic classifier.

6. **Semantic execution block**
   - model emits a delimited block:
     ```text
     ```calculator
     EXPR
     ```
     ```
   - runtime intercepts the completed block and executes locally.
   - no function name/JSON schema beyond the block type.
   - this is not a normal explicit tool-call condition.

7. **Oracle expression selection**
   - exact gold expression supplied to calculator;
   - estimates maximum arithmetic headroom.

Optional:
8. **Native function-calling baseline**
   - only if the selected model/runtime supports stable native function calling.

Optional:
9. **Distractor-tool catalog**
   - explicit tool condition with increasing tool count;
   - calculator remains available but competes with irrelevant tools.

## Hard-arithmetic benchmark

The next benchmark must include arithmetic that is genuinely difficult for the selected LLM.

### Required difficulty axes
- operand digit width;
- operator count;
- multiplication depth;
- nested parentheses;
- exact integer/rational division where safe;
- mixed operations;
- long intermediate values.

Example difficulty ladder:

- Easy:
  - `12 * 37`
  - `128 + 745`

- Moderate:
  - `8462 * 719`
  - `(9274 + 3811) * 47`

- Hard:
  - `15246377 * 746647383`
  - `(18273645 * 918273) + (99887766 * 554433)`
  - nested exact-rational expressions with large intermediate values.

### Safety
All generated expressions must remain within calculator bounds:
- no arbitrary code;
- bounded exponentiation;
- bounded integer bit growth;
- bounded AST depth;
- bounded wall time.

Do not increase difficulty by making the calculator itself unstable.

## Benchmark design

### Factorial grid
Cross at least:
- 4 operand-width bands;
- 4 operator-count bands;
- 3 expression-structure classes;
- multiple seeds.

Use enough examples per cell to estimate a real difficulty curve.

### Adaptive difficulty pilot
Before the full sweep:
- run LLM-only over a broad candidate pool;
- estimate exact-match difficulty;
- select cells spanning roughly:
  - 90%+ LLM accuracy;
  - 50–80%;
  - 10–50%;
  - near 0%.

Do not choose individual examples based on assisted-condition outcomes.

### Held-out confirmatory set
After selecting the difficulty regime and freezing normalization/interface rules:
- generate a new held-out benchmark;
- do not tune detector or prompts on it.

## Expression exposure and selection diagnostics

Every assisted generation must classify:

### EXPOSURE
Did the generated stream contain a representation of the intended full arithmetic expression?

### RECOGNITION
Did the runtime recognize a candidate?

### SELECTION
Was the selected candidate the intended full expression rather than an intermediate subexpression?

### NORMALIZATION
Was the recognized surface form converted to the correct canonical expression?

### EXECUTION
Did the calculator return the exact correct result?

### REINJECTION
Was the result inserted at the intended point?

### USE
Did the model use the result in its final answer?

### OVERRIDE
Did the model later contradict/change a correct calculator result?

Report these separately.

Do not infer interface quality from final exact match alone.

## LaTeX and Unicode normalization

The original miss on `3 \\times 2 =` is a surface-normalization issue, not a deep semantic failure.

Implement a deterministic normalization layer before parsing.

Required supported aliases:
- `\\times`, `×` -> multiplication;
- `\\cdot` -> multiplication where unambiguous;
- `\\div`, `÷` -> division;
- Unicode minus variants -> ASCII minus;
- harmless whitespace/bracket variants.

Add adversarial tests to ensure normalization does not trigger inside:
- prose;
- quoted code;
- unrelated LaTeX;
- variable equations outside the arithmetic task grammar.

Keep normalization separate from semantic selection.

## Semantic execution block

### Motivation
The strict watcher fires on any valid arithmetic suffix and can therefore execute an intermediate step too early.

A semantic execution block provides an explicit boundary without requiring a verbose API schema.

### Initial syntax
Use a Markdown-like fenced block:

```text
```calculator
15246377 * 746647383
```
```

Alternative short tags may be tested only as secondary variants.

### Runtime behavior
- detect opening calculator block;
- collect contents incrementally;
- do not execute intermediate expressions inside the block;
- execute only at the closing delimiter;
- normalize arithmetic surface syntax;
- pass canonical micro-IR to the existing bounded calculator;
- insert a typed result immediately after the block or as a compact result record;
- continue generation.

### Training requirement
First test zero-shot prompting.

If the selected model does not reliably emit execution blocks:
- create a small SFT/LoRA experiment;
- train only the execution-delimiting behavior;
- do not train arithmetic answers into the model.

The target behavior is:
> recognize arithmetic work and externalize it into a calculator block.

Not:
> memorize multiplication.

### LoRA/SFT data
Create synthetic pairs across diverse natural-language prompts:
- arithmetic question -> calculator block;
- prose with numbers but no arithmetic -> no block;
- already-known literal answer -> no unnecessary block;
- multiple arithmetic subgoals -> separate or appropriately scoped blocks.

Include negatives for:
- quoted calculator blocks;
- Markdown examples;
- code snippets discussing arithmetic;
- variable expressions outside supported grammar.

## Comparison with explicit tool calling

Explicit tool calling is expected to match or exceed calculator correctness once selected.

The next iteration should therefore measure:
- selection success;
- valid argument formation;
- model-authored invocation tokens;
- reinjected/result tokens;
- total model tokens;
- wall time;
- tool-selection errors;
- wrong-tool rate under distractor-tool catalog;
- result-use/override;
- final exact accuracy.

Do not claim lower cost unless measured.

## Tool-catalog stress test

Secondary experiment only.

Register:
- calculator;
- date resolver;
- web/search placeholder;
- DB lookup placeholder;
- unit converter;
- graph query placeholder;
- several irrelevant synthetic tools.

Vary tool count:
- 1
- 4
- 8
- 16
- optionally 32

The arithmetic task always requires calculator.

Compare:
- explicit tool selection accuracy;
- tool-selection tokens;
- wrong-tool calls;
- calculator block/reflex behavior.

Do not make this the primary Paper 1 contribution unless the effect is substantial and replicated.

## Primary metrics

### End-task
- exact answer accuracy;
- 95% CI;
- accuracy by difficulty cell.

### Interface
- exposure recall;
- recognition recall;
- full-expression selection precision/recall;
- normalization correctness;
- engine correctness;
- reinjection success;
- result-use rate;
- override rate.

### Cost
- prompt tokens;
- generated tokens;
- tool-selection tokens;
- reinjected tokens;
- number of model calls;
- engine calls;
- wall-clock latency;
- engine latency;
- trace/state bytes.

### Scaling
Fit/plot:
- accuracy vs operand width;
- accuracy vs operator count;
- accuracy vs estimated baseline difficulty;
- assisted-minus-LLM gain vs difficulty;
- interface failure composition vs difficulty.

The key expected curve is:
- LLM-only accuracy drops;
- oracle remains flat;
- assisted conditions differ mainly by interface success.

## Required failure modes to surface

1. **Surface miss**
   - e.g. LaTeX `\\times` not recognized.

2. **Premature selection**
   - intermediate valid expression executed before full target.

3. **Wrong normalization**
   - recognized candidate maps to incorrect canonical expression.

4. **No exposure**
   - model never emits an executable arithmetic representation.

5. **Tool selection failure**
   - explicit condition chooses no tool/wrong tool.

6. **Malformed tool/block**
   - arguments or delimiters invalid.

7. **Correct result ignored**
   - engine right, final answer wrong.

8. **Correct result overridden**
   - model initially uses value then contradicts it.

9. **Unnecessary intervention**
   - easy task where model already correct and assistance adds cost.

10. **Catalog interference**
    - explicit tool routing degrades as unrelated tools increase.

## Evidence gates

### Gate A — hard arithmetic
Do not continue using easy arithmetic as headline evidence if LLM-only remains above ~90%.

The benchmark must contain a substantial regime where exact neural arithmetic fails.

### Gate B — normalized reflex
If LaTeX/Unicode normalization repairs most misses but premature selection remains dominant, stop adding surface heuristics.

That motivates semantic boundaries rather than more regex.

### Gate C — semantic execution block
The block interface is promising if it:
- materially improves full-expression selection;
- preserves calculator exactness;
- reduces override/malformed invocation;
- and is competitive with explicit tools on token/cost frontier.

### Gate D — LoRA
Only introduce LoRA/SFT if zero-shot block emission is insufficient.

A LoRA result must be compared to:
- same model without LoRA;
- explicit tool prompting;
- strict/normalized reflex.

### Gate E — Paper 2
Do not open heterogeneous symbolic-engine empirical runs merely because block syntax works for calculator.

Require a reproducible automatic-compute regime that improves the quality/cost frontier or establishes a clear interface mechanism that generalizes beyond arithmetic.

## Relationship to later papers

### Paper 2
Can reuse semantic execution blocks such as:
- `calculator`
- `datalog`
- `graph`

but must not assume a learned semantic router.

### Paper 3
Can study learned detection/routing from ordinary language into these execution families.

The execution-block interface may become a supervised intermediate target:
- natural language -> `<calculator>` block
- natural language -> `<datalog>` block
- natural language -> `<graph>` block

### Longer-term local execution language
Do not implement a broad `lll-lang` in this iteration.

However, document the design direction:
- small bounded language;
- suitable for local LLM-server execution;
- typed operations;
- deterministic parsing;
- safe resource bounds;
- extensible engine families;
- lower overhead than arbitrary Python/tool JSON.

The Paper 1 semantic block should be designed so it can evolve toward that language later.

## Engineering requirements
- Preserve the existing bounded calculator and typed micro-IR.
- Keep old strict-reflex condition unchanged as a baseline.
- Add normalization as a separate layer/config flag.
- Add execution blocks as a separate condition, not a silent detector modification.
- Version all prompts, detector policies, normalizers, and block grammars.
- Preserve raw generations and event traces.
- Add unit/adversarial tests before XPU sweeps.
- Keep CPU scripted plumbing marked `empirical: false`.
- Freeze rules before confirmatory evaluation.

## Deliverables
- updated `docs/papers/paper1/paper1.tex`;
- next-iteration config(s);
- hard-arithmetic benchmark generator;
- normalized reflex implementation/tests;
- semantic execution-block runtime/tests;
- optional LoRA/SFT dataset and training script if gate requires it;
- tool-catalog stress config;
- machine-readable component metrics;
- plots for difficulty scaling and interface-failure decomposition;
- explicit go/no-go conclusion for Paper 2 and Paper 3.
