# AGENTS — Paper 1 Next Iteration
## NL → Extensible Cognitive DSL for Arithmetic Coprocessor Assistance

## 0. Purpose

The current Paper 1 GSM8K results show that calculator execution is not the main bottleneck. The main failure is converting natural-language quantitative reasoning into the correct executable intermediate operations.

This iteration changes the Paper 1 target from:

`natural language -> calculator trigger/block`

to:

`natural language -> small typed declarative DSL -> deterministic execution -> typed state -> continuation`

The DSL should be deliberately small for Paper 1, but its AST, parser, typing, validation, state model, and dataset-generation tooling must be designed so Paper 2 can extend it to logic, graph, temporal, retrieval, data, and additional coprocessors without replacing the Paper 1 representation.

The scientific question is:

> Can a small model learn to compile natural-language arithmetic/state relations into a compact executable DSL more robustly than it can directly solve the same problems or learn a brittle calculator-call syntax?

Do not overclaim a universal cognitive language in Paper 1.

## 1. Core decomposition

Treat every benchmark example as four separable problems:

A. Semantic compilation — map a natural-language clause/span to a typed DSL state update, expression, or query.

B. Boundary / executable-unit detection — identify when enough semantic structure exists to execute a statement or expression.

C. Execution — deterministic runtime evaluates calculator-compatible DSL nodes.

D. Integration — executed values are stored as typed state and made available to later clauses and final answer generation.

Paper 1 primarily improves and measures A, while keeping B/C/D explicit.

## 2. DSL design goals

The DSL must be:
- declarative rather than general-purpose imperative code;
- small enough for small LLMs to learn;
- deterministic to parse;
- easy to serialize as JSON and later optionally as a compact text syntax;
- typed;
- stateful;
- safe to execute;
- reference-based;
- provenance-friendly;
- easy to extend with new operator families;
- independent of concrete coprocessor engine names.

The model should emit semantic operators such as `MUL`, not `calculator`.

The runtime maps operators/capabilities to concrete coprocessors.

## 3. Paper 1 DSL: CCIR-Arith v0

Use a stable internal AST. JSON is the canonical interchange format.

### Core state/record operators
- `FRAME`
- `SET`
- `REF`
- `QUERY`
- `RETURN`

### Arithmetic
- `CONST`
- `ADD`
- `SUB`
- `MUL`
- `DIV`
- `NEG`
- `ABS`

### Aggregation
- `SUM`
- `MIN`
- `MAX`
- `MEAN`

### Percentage/rate helpers
- `PERCENT_OF`
- `INCREASE_BY_PERCENT`
- `DECREASE_BY_PERCENT`
- `RATE_TIMES_DURATION`

These may lower to arithmetic internally, but explicit semantic forms can make NL→DSL learning easier.

### Comparison/selection
- `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`
- `ARGMAX`, `ARGMIN`

### Basic value types
Prepare modest typing support:
- scalar
- count
- money
- percentage
- duration
- rate

Units/date engines remain mostly Paper 2 concerns unless required by Paper 1 data.

## 4. Example

Question:

> A program has 60 downloads in month 1. Month 2 has three times as many. Month 3 is 30% lower than month 2. What is the total over three months?

Chopped clauses:
1. `A program has 60 downloads in month 1.`
2. `Month 2 has three times as many.`
3. `Month 3 is 30% lower than month 2.`
4. `What is the total over three months?`

Candidate canonical AST:

```json
[
  {
    "source_part": 0,
    "delta": [
      {"op": "FRAME", "id": "month1", "type": "time_period"},
      {"op": "SET", "target": "month1.downloads",
       "expr": {"op": "CONST", "value": 60}}
    ]
  },
  {
    "source_part": 1,
    "delta": [
      {"op": "FRAME", "id": "month2", "type": "time_period"},
      {"op": "SET", "target": "month2.downloads",
       "expr": {"op": "MUL",
                "args": [
                  {"op": "REF", "path": "month1.downloads"},
                  {"op": "CONST", "value": 3}
                ]}}
    ]
  },
  {
    "source_part": 2,
    "delta": [
      {"op": "FRAME", "id": "month3", "type": "time_period"},
      {"op": "SET", "target": "month3.downloads",
       "expr": {"op": "DECREASE_BY_PERCENT",
                "value": {"op": "REF", "path": "month2.downloads"},
                "percent": {"op": "CONST", "value": 30}}}
    ]
  },
  {
    "source_part": 3,
    "delta": [
      {"op": "QUERY", "id": "total_downloads",
       "expr": {"op": "SUM",
                "args": [
                  {"op": "REF", "path": "month1.downloads"},
                  {"op": "REF", "path": "month2.downloads"},
                  {"op": "REF", "path": "month3.downloads"}
                ]}},
      {"op": "RETURN", "ref": "total_downloads"}
    ]
  }
]
```

Runtime state after execution:

```json
{
  "month1.downloads": 60,
  "month2.downloads": 180,
  "month3.downloads": 126,
  "total_downloads": 366
}
```

## 5. Paper 2 extensibility

Do not implement the full Paper 2 language now, but reserve clean operator namespaces and a registry-driven AST.

Potential future families:

Logic:
- `FACT`, `RULE`, `ENTAIL`, `UNKNOWN`

Graph:
- `NODE`, `EDGE`, `PATH`, `REACHABLE`

Time:
- `DATE`, `DATE_ADD`, `DATE_DIFF`, `INTERVAL`

Units:
- `QUANTITY`, `CONVERT`

Retrieval/data:
- `LOOKUP`, `FILTER`, `PROJECT`, `AGGREGATE`, `JOIN`, `VERIFY_SOURCE`

Constraints:
- `ASSERT`, `CONSTRAINT`, `SAT`, `SOLVE`

The parser/runtime must support operator registration. Avoid hard-coding Paper 1 operators throughout the codebase.

## 6. Adapter principle

The adapter should mostly learn:
- DSL syntax;
- state/reference conventions;
- clause-to-IR structural mapping;
- stable semantic primitives;
- boundary formatting.

It should NOT memorize concrete coprocessor engine names or a deployment registry.

Adding/replacing a concrete coprocessor implementation should ideally require runtime registration only.

For Paper 2, test continued LoRA from the Paper 1 adapter versus fresh adapters and measure arithmetic retention.

## 7. Default dataset sources

Default dataset list:

1. GSM8K
2. GSM-Symbolic
3. GSM-Plus
4. GSM-Ranges
5. MAWPS
6. TAT-QA
7. CLUTRR
8. RuleTaker

Paper 1 should prioritize arithmetic-compatible subsets.

Do not force CLUTRR/RuleTaker into CCIR-Arith. Prepare ingestion only; their semantic operators belong to Paper 2.

TAT-QA should initially contribute only examples expressible with the Paper 1 arithmetic/state DSL.

## 8. Dataset mining CLI

Implement a modular Python package/CLI able to ingest configurable datasets.

Suggested command:

```bash
python -m ccpu.dsl_dataset mine   --datasets gsm8k gsm_symbolic gsm_plus gsm_ranges mawps tatqa clutrr ruletaker   --output-dir artifacts/paper1/dsl/raw_v1
```

Each loader emits a normalized record:

```json
{
  "dataset": "gsm8k",
  "split": "train",
  "source_id": "...",
  "question": "...",
  "answer": "...",
  "gold_reasoning": "...",
  "metadata": {},
  "parts": []
}
```

Preserve source IDs, original text, hashes, and source/license metadata when available.

Never overwrite source data.

## 9. Heuristic chopping

Implement dataset-aware `chop_example()` logic.

Potential boundaries:
- sentence punctuation;
- newline-separated rationale;
- equation separators;
- `####` in GSM8K;
- table/text section boundaries in TAT-QA;
- rule/fact separators in RuleTaker;
- story sentence boundaries in CLUTRR;
- dataset-native rationale steps.

Each chopped part stores:
- `part_id`;
- text;
- kind;
- original character offsets;
- heuristic used;
- confidence;
- warnings.

Do not assume chopping is correct.

## 10. Output layout

Create one JSONL file per dataset:

```text
artifacts/paper1/dsl/raw_v1/
  gsm8k.jsonl
  gsm_symbolic.jsonl
  gsm_plus.jsonl
  gsm_ranges.jsonl
  mawps.jsonl
  tatqa.jsonl
  clutrr.jsonl
  ruletaker.jsonl
  manifest.json
```

Each item includes original QA plus the array of chopped parts.

## 11. Audit chopped examples before DSL generation

Add:

```bash
python -m ccpu.dsl_dataset audit-chops   --input-dir artifacts/paper1/dsl/raw_v1   --sample-per-dataset 50   --output artifacts/paper1/dsl/raw_v1/chop_audit.json
```

Review:
- missing clauses;
- bad sentence splits;
- split numbers/equations;
- answer/rationale leakage into question clauses;
- empty/duplicate parts;
- table corruption;
- source ID collisions.

Produce automatic stats and a human-readable failure sample.

Do not scale teacher generation until obvious chopping problems are fixed.

## 12. Bootstrap skill

Create:

```text
skills/ccir_arith_compiler/SKILL.md
```

The skill should define:
1. CCIR-Arith grammar.
2. Operator semantics.
3. State/reference semantics.
4. Clause-level mapping policy.
5. Rules against inventing state.
6. Rules against solving only in prose.
7. Handling unresolved/ambiguous references.
8. Source fact vs derived value distinction.
9. Required JSON output.
10. Good and bad examples.
11. Ambiguity reporting.

Recommended teacher output:

```json
{
  "status": "ok|ambiguous|unsupported",
  "part_id": 2,
  "state_delta": [],
  "assumptions": [],
  "confidence": 0.0
}
```

Codex may improve the skill during bootstrap, but every semantic DSL/skill change must bump a version and preserve prior artifacts.

## 13. Codex bootstrap set

Before bulk cloud generation, use Codex + the skill to build a small trusted seed set.

Suggested selection command:

```bash
python -m ccpu.dsl_dataset select   --input artifacts/paper1/dsl/raw_v1/gsm8k.jsonl   --max-examples 100   --strategy stratified   --output artifacts/paper1/dsl/bootstrap_v1/gsm8k_seed.jsonl
```

Start with 50–100 GSM8K examples stratified by operation count/difficulty.

Then add a small MAWPS sample.

Keep Codex/teacher raw output separate from accepted/verified mappings.

## 14. Teacher input/output

Teacher input should include:
- DSL version;
- dataset/source ID;
- full question;
- correct answer;
- chopped parts;
- current part;
- current state;
- allowed operators;
- skill/spec hash.

Prefer incremental supervision:

`state_before + current NL part -> state_delta`

Whole-problem compilation may be generated as an auxiliary consistency check.

## 15. Remote teacher generator with LiteLLM

Implement a provider-neutral client using LiteLLM.

Suggested CLI:

```bash
python -m ccpu.dsl_dataset generate   --input artifacts/paper1/dsl/bootstrap_v1/gsm8k_seed.jsonl   --provider openrouter   --model <configured-model>   --skill skills/ccir_arith_compiler/SKILL.md   --output artifacts/paper1/dsl/teacher_v1/gsm8k.jsonl
```

Support:

OpenRouter:
- `OPENROUTER_API_KEY`

OpenAI API:
- `OPENAI_API_KEY`

Hugging Face/provider:
- `HF_TOKEN`

Also allow normal LiteLLM provider configuration.

Never commit tokens.

Add `.env.example` with variable names only.

Model names belong in config, not hard-coded logic.

## 16. Teacher strategy

Support:
- single teacher;
- multiple free/cheap OpenRouter teachers;
- consensus;
- bounded retry;
- escalation.

For hard rows:
1. call 2–4 configured teachers;
2. normalize candidate ASTs;
3. test semantic/execution equivalence;
4. accept consensus when validation passes;
5. otherwise queue for Codex/stronger-model review.

Do not trust syntax validity alone.

## 17. Validation pipeline

Every candidate must pass:

### Syntax
- JSON parse;
- AST schema;
- required fields.

### Types
- valid operator;
- valid arity;
- compatible value types.

### References
- paths resolve;
- no invalid/dangling state refs.

### State
- deterministic update order;
- valid targets;
- queries/returns resolve.

### Execution
Execute all Paper 1 arithmetic-compatible nodes deterministically.

### Answer consistency
Final executed result must match benchmark answer where exact comparison is meaningful.

### Intermediate consistency
When a dataset provides trustworthy equations/traces, compare operations/values.

Do not treat free-form rationale as perfect formal gold by default.

## 18. Quality grades

Store a supervision-quality grade:

- `Q4_MANUAL_GOLD`
- `Q3_STRONG_TEACHER_AUDITED`
- `Q2_MULTI_TEACHER_EXEC_VERIFIED`
- `Q1_SINGLE_TEACHER_EXEC_VERIFIED`
- `Q0_PERTURBED_FROM_VERIFIED`

Also store flags:
- syntax_verified
- type_verified
- execution_verified
- final_answer_verified
- intermediate_trace_verified
- manually_reviewed

## 19. Reject/repair policy

Candidate mappings may be:
- accepted;
- deterministically repaired;
- retried;
- escalated;
- rejected.

Safe deterministic repair:
- numeric formatting;
- canonical AST key ordering;
- harmless field normalization;
- redundant parentheses.

Never silently repair semantic operators/references unless equivalence is provable.

## 20. Controlled perturbation engine

Once a mapping is verified, generate many variants mechanically.

Supported transformations:

### Numeric
- replace values;
- scale magnitude;
- very large integers;
- decimals;
- percentages;
- negative values where valid.

### Operator-aware
Only when NL and DSL are changed together:
- twice -> three times;
- 20% more -> 30% less;
- sum -> difference;
- max -> min.

### Entity/field renaming
- downloads -> visitors;
- apples -> books;
- month1 -> week1.

### Paraphrase
Generate lexical variants while preserving DSL.

### Clause ordering
Reorder only when dependency analysis proves semantic independence.

### Distractors
Add irrelevant clauses whose DSL effect is `NOOP` or unrelated state.

Track parent ID and transformation provenance.

## 21. Perturbation validation

Every transformed row must:
- parse;
- type-check;
- execute;
- produce expected answer;
- preserve dependency validity;
- match NL numeric/operator changes;
- avoid rationale leakage.

## 22. Leakage-safe splits

Split by semantic-program lineage, not row identity.

All descendants/paraphrases/numeric variants of one source program stay in the same split unless a deliberate counterfactual generalization protocol says otherwise.

## 23. Training views

Create at least:

### Clause compiler
Input:
- state_before;
- current clause.

Target:
- state_delta.

### Whole-problem compiler
Input:
- complete problem.

Target:
- full DSL.

### Incremental execution view
Input:
- prior state + clause + executed results.

Target:
- next DSL delta/query/return.

Paper 1 headline should emphasize clause/incremental compilation.

## 24. LoRA training

Train on the small model families already used in this project where feasible:
- Qwen3-0.6B;
- SmolLM2-1.7B;
- Gemma if runtime/access permits.

Pin:
- base revision;
- dataset hashes;
- tokenizer;
- LoRA rank/alpha/dropout;
- target modules;
- seed.

Start with attention projections. Add attention+MLP only if needed.

Do not train on benchmark eval splits.

## 25. Training objectives

Primary:
- valid DSL rate;
- semantic-equivalent DSL accuracy.

Secondary:
- normalized AST exact match;
- operator accuracy;
- reference accuracy;
- state-target accuracy;
- execution correctness;
- final answer correctness.

Optional auxiliary tasks later:
- operator classification;
- reference target selection;
- boundary type;
- state slot prediction.

## 26. Evaluation

Report A/B/C/D separately.

### A — Semantic compilation
- exact AST;
- semantic-equivalent AST;
- operator accuracy;
- reference accuracy;
- state-target accuracy;
- syntax validity.

### B — Executable-unit detection
- complete executable statement rate;
- premature execution;
- missed executable statement.

### C — Execution
- exact execution conditional on valid DSL.

### D — Integration
- downstream state-reference correctness;
- final answer;
- override after exact state exists.

### Robustness
Evaluate:
- untouched GSM8K;
- numeric remapping;
- very large numbers;
- paraphrases;
- operator perturbations;
- entity/field renaming;
- safe clause reordering;
- distractors.

## 27. Required baselines

Compare:
1. direct LLM answer;
2. CoT/direct reasoning;
3. generic `__compute(...)` tool;
4. calculator block ICL;
5. calculator-block LoRA;
6. NL→CCIR ICL;
7. NL→CCIR LoRA;
8. NL→CCIR LoRA + ICL;
9. oracle CCIR;
10. oracle execution;
11. runtime-copy/direct return where applicable.

This determines whether improvements come from semantic compilation, triggering, execution, or integration.

## 28. Key hypothesis

Under controlled numeric remapping/scaling:

`P(correct CCIR | same semantic structure)`

should degrade substantially less than:

`P(correct direct answer)`.

Replacing small values with huge values should not materially change semantic compilation when the linguistic relation is unchanged.

If DSL accuracy strongly collapses with number magnitude, the core hypothesis is weakened.

## 29. Syntax optimization comes later

Canonical supervision remains JSON AST.

Only after the representation works, compare compact syntaxes such as:

```text
SET month2.downloads = MUL(REF month1.downloads, 3)
```

or S-expressions.

Measure:
- model token count;
- syntax validity;
- semantic accuracy;
- training cost.

Do not optimize syntax before semantics.

## 30. Runtime architecture

Target flow:

```text
NL clause
  -> DSL compiler (LLM)
  -> parser/type/ref validator
  -> operator registry
  -> calculator/state runtime
  -> typed state update
  -> next clause
```

Example registry:

```python
registry.register("ADD", arithmetic_executor)
registry.register("MUL", arithmetic_executor)
registry.register("PERCENT_OF", arithmetic_executor)
```

Paper 2 later registers additional operator families without changing the core state/reference model.

## 31. Suggested code layout

```text
src/ccpu/dsl/
  ast.py
  schema.py
  parser.py
  types.py
  state.py
  registry.py
  execute.py
  normalize.py
  equivalence.py

src/ccpu/dsl_dataset/
  loaders/
    gsm8k.py
    gsm_symbolic.py
    gsm_plus.py
    gsm_ranges.py
    mawps.py
    tatqa.py
    clutrr.py
    ruletaker.py
  chop.py
  audit.py
  select.py
  teacher.py
  validate.py
  perturb.py
  split.py
  cli.py

skills/ccir_arith_compiler/
  SKILL.md
```

Reuse existing repo config/model/XPU helpers when possible.

## 32. Reproducibility

Every stage writes:
- config;
- input/output hashes;
- model/provider;
- model revision when known;
- skill hash;
- DSL version;
- git revision;
- seed;
- validation counts;
- rejection reasons.

Raw teacher outputs are immutable artifacts. Accepted datasets are derived artifacts.

## 33. Credentials/security

Teacher code must:
- read keys only from environment;
- never print/store keys;
- never commit `.env`;
- support dry-run;
- support `--max-examples`;
- bound concurrency/retries;
- redact provider headers.

Environment variables:
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `HF_TOKEN`

## 34. OpenRouter bootstrap execution

After user provides credentials:
1. configure free models in YAML/JSON;
2. smoke 5 examples;
3. validate syntax/execution;
4. run 50;
5. inspect failures;
6. only then scale.

Prefer multi-model consensus for ambiguous rows.

## 35. Review queues

Produce:

```text
review/
  invalid_syntax.jsonl
  dangling_refs.jsonl
  answer_mismatch.jsonl
  teacher_disagreement.jsonl
  ambiguous.jsonl
  accepted_sample.jsonl
```

Use Codex + skill to review:
- initial bootstrap batch;
- random accepted samples;
- all new operator families;
- all high-impact grammar changes.

## 36. Paper 1 manuscript integration

Add a section explaining why GSM8K changed the direction:
- calculator execution is easy;
- semantic decomposition is hard;
- tool/block syntax does not transfer robustly;
- oracle operations expose large headroom;
- therefore test NL→DSL semantic compilation.

Report:
- DSL spec;
- mining/bootstrap pipeline;
- teacher quality;
- validation/rejection rates;
- LoRA results;
- robustness under numeric remapping;
- direct/tool/block/DSL comparison;
- A/B/C/D factorization.

Separate dataset-generation evidence from model-evaluation evidence.

## 37. Paper 2 handoff

Freeze after Paper 1:
- `CCIR-Core` state/reference semantics;
- `CCIR-Arith` operators;
- parser/type checker;
- operator registry;
- dataset generation framework;
- best Paper 1 adapter(s).

Paper 2 extends to dates, units, logic, graph, retrieval/data, and verification.

Test:
- continue training Paper 1 adapter;
- fresh adapter;
- arithmetic retention;
- new-operator learning;
- registry expansion without retraining when DSL semantics are unchanged.

Desired property:

> The adapter primarily learns the stable semantic compiler/DSL interface. Triggers, dispatchers, validators, coprocessor implementations, and registry changes remain runtime concerns.

## 38. Immediate execution order

P0. Freeze `CCIR-Arith v0`.

P1. Implement GSM8K loader/chopper/audit.

P2. Generate 50–100 Codex-reviewed GSM8K mappings using the skill.

P3. Implement parser/type/ref/execution validation.

P4. Revise/freeze DSL and skill; version changes.

P5. Implement LiteLLM teacher client.

P6. Smoke free OpenRouter models after credentials are supplied.

P7. Build larger validated GSM8K corpus.

P8. Add perturbations and lineage-safe splits.

P9. Train first Qwen/SmolLM LoRA compilers.

P10. Evaluate untouched GSM8K + numeric-remap/large-number variants.

P11. Add MAWPS and compatible GSM-Symbolic/GSM-Plus/GSM-Ranges.

P12. Add compatible TAT-QA arithmetic subset.

P13. Keep CLUTRR/RuleTaker ingestion prepared for Paper 2.

P14. Integrate results into Paper 1 and update Paper 2 handoff.

## 39. Stop conditions

Do not scale teacher generation if:
- chopping is poor;
- DSL semantics are still changing;
- teacher validity is low;
- answer consistency is low;
- perturbation lineage is unsafe.

Do not scale LoRA training if:
- mappings are not execution-verified;
- semantic-program leakage exists;
- evaluation splits contaminate training.

Prefer a smaller verified compiler dataset to a large weak one.
