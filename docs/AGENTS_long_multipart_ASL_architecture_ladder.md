# AGENTS — Long / Multipart ASL Cognitive Architecture Experimental Ladder

## Purpose

Future experiment series to compare the already-planned **vanilla Transformer + selectors + LoRA/adapters + textual result injection** baseline against progressively more architectural variants using dedicated NL/ASL encoders, typed source/world memories, non-textual K/V injection, state-aware selectors, block-sparse visibility, and later PRA-backed materialization.

Run only after:
- ASL/F3 representation is sufficiently frozen;
- short-context NL→ASL is stable;
- deterministic parser/lowering/CogCop execution is reliable;
- simple selector and textual-injection baselines exist.

Central question:

> How much of long/multipart cognitive processing can be solved by selectors/adapters around a conventional Transformer, and when do explicit changes to representation, attention, memory, and injection become necessary?

---

## 1. Factor the problem

Always evaluate these separately:

### P1 — Span / block selection
Determine:
- whether a region should be formalized;
- START/END;
- intervention type;
- relevant prior source;
- relevant prior world state.

### P2 — Semantic compilation
For selected block `Bi`:

```text
Bi
+ selected prior source
+ selected world state
→ ASL/F3 Ai
```

### P3 — Execution / injection
```text
Ai → CogCop → Ri → world update Wi
```

Then decide what later source, later ASL blocks, and final generation may see.

---

## 2. Logical records

Do not assume the long input must physically be:

```text
NL ... <ASL> ... </ASL> ... <RESULT> ...
```

Conceptually keep:

```text
SOURCE:          D = d1...dn
SELECTED BLOCK:  Bi=[si,ei]
ASL:             Ai
COGCOP RESULT:   Ri
WORLD RECORD:    Wi
```

with:

```text
Bi -> Ai -> Ri -> Wi
```

and optional edges:

```text
depends_on_source
depends_on_world
derived_from
updates
same_scope
```

---

## 3. Two experiment tracks

### Track V — Vanilla / minimally modified

Use:
- standard decoder-only or encoder-decoder Transformer;
- static or switched LoRA;
- START/END selector;
- ordinary prompt/KV cache;
- ASL as generated tokens;
- CogCop result reinserted as text;
- custom masking only where needed.

This is the compatibility baseline.

### Track A — Architectural cognitive variants

Allow:
- dedicated NL encoder `E_N`;
- dedicated ASL/world encoder `E_A`;
- shared semantic/world encoder `E_NA`;
- typed source/world memories;
- non-textual result K/V;
- separate or merged source/world attention;
- state-aware controllers;
- typed segment DAGs;
- PRA selection later.

Architectural variants must earn their complexity empirically.

---

## 4. Master ladder

Run progressively:

```text
L0  Oracle blocks, no intermediate injection
L1  Learned blocks, no intermediate injection
L2  Textual result injection
L3  Dynamic adapter switching
L4  Multipart source-context selection
L5  Typed result encoder + cross-attention
L6  ASL/world result encoder
L7  Merged source/world K/V
L8  Separate E_N / E_A
L9  Shared semantic E_NA
L10 State-aware selector
L11 Typed segment DAG / block-sparse attention
L12 Separate persistent world memory
L13 PRA-backed source/world selection
L14 Online segmented multi-intervention inference
```

Introduce one major mechanism per stage.

---

## 5. Long-content benchmark

Construct documents with:

```text
0, 1, 2, 4, 8, 16
```

formalizable spans.

Include:
- ordinary prose between formal spans;
- irrelevant numbers/dates;
- numeric statements that should NOT trigger formalization;
- formalizable non-numeric relations;
- independent spans;
- dependency chains/forks/joins;
- nested scopes/subproblems;
- repeated names with changed meaning;
- early results relevant much later;
- final queries requiring only a subset of earlier blocks.

Use multiple styles:
- exercise books;
- financial reports;
- manuals;
- conversations;
- mixed table+prose;
- task/agent logs.

Do not use only concatenated GSM8K.

---

## 6. Gold annotations

For every candidate/gold block record:

```text
block_id
source_start
source_end
scope_id
block_type
gold_asl
gold_result
critical_to_query
source_dependencies[]
world_dependencies[]
derived_records[]
```

This allows selector/retrieval evaluation separately from ASL generation.

---

# VANILLA TRACK

## 7. V0 — Oracle blocks + final-only execution

```text
full source
→ oracle blocks
→ generate ASL
→ execute
→ store results externally
→ final answer sees results
```

No learned selector and no intermediate influence on later source.

Compare:
- block-only input;
- causal source prefix;
- full-source leaky diagnostic only.

Establish the long-document compiler ceiling.

---

## 8. V1 — Learned selector

Compare:
- sentence classifier;
- clause classifier;
- token START/INSIDE/END;
- final-layer hidden-state controller;
- multi-layer controller;
- oracle.

Metrics:
- START/END P/R/F1;
- exact span;
- span IoU;
- false blocks / 1k tokens;
- missed critical spans;
- ASL quality conditioned on detected boundary.

---

## 9. V2 — Textual result injection

Baseline:

```text
ASL
→ CogCop
→ serialize result
→ append/reinsert into prompt
```

Compare:

```text
I0 NONE
I1 RESULT_ONLY_TEXT
I2 STRUCTURED_STATE_TEXT
I3 ASL_PLUS_RESULT_TEXT
```

Measure:
- injected token count;
- later ASL accuracy;
- final answer;
- result-use rate;
- result override;
- hallucination;
- context growth.

---

## 10. V3 — Dynamic adapter switching

Compare:

```text
static ASL LoRA
oracle-switched LoRA
learned-switched LoRA
```

Initial modes:

```text
NORMAL
FORMAL
```

Later optionally:
```text
VERIFY
RETRIEVE
HELP
```

Run at least:
- conditional QKVO;
- conditional MLP-only;
- shared always-on QKVO + conditional MLP.

Record KV-cache mode and switch overhead.

---

## 11. V4 — Multipart source-context selection

For current block select:

```text
current block
+
relevant earlier source
+
prior textual results
```

Compare:

```text
S0 all prefix
S1 local window
S2 lexical
S3 semantic
S4 oracle source dependencies
```

This is the simple PRA-like baseline without typed world memory.

---

# ARCHITECTURAL TRACK

## 12. A0 — Typed result encoder

Replace text reinjection:

```text
ResultRecord
→ E_R
→ K_R,V_R
→ decoder cross-attention
```

Example:

```text
path=hats.green.count
value=3
type=integer
provenance=e3
scope=exercise1
status=RUNTIME_VERIFIED
```

Keep selector and ASL model unchanged.

This first architectural comparison must isolate:

```text
text injection
vs
typed K/V injection
```

---

## 13. A1 — Reuse ASL/world encoder

Prefer eventually:

```text
CogCop result
→ canonical ASL/F3 state record
→ E_A
→ world K/V
```

Example:

```text
hats.green.count@now := 3
```

The same `E_A` should ideally encode:
- external ASL during training;
- internal ASL;
- CogCop-derived world state;
- persistent symbolic memory.

---

## 14. A2 — Separate source/world cross-attention

Decoder:

```text
causal self-attention
+
cross-attention(source NL)
+
cross-attention(world ASL/state)
```

Formally:

```text
O_N = Attn(Q,K_N,V_N)
O_W = Attn(Q,K_W,V_W)
O = H + g_N O_N + g_W O_W
```

Log:
- source/world attention;
- branch norms;
- gates;
- decoder-layer utilization.

---

## 15. A3 — Merged source/world K/V

Construct:

```text
K = concat(K_source, K_world)
V = concat(V_source, V_world)
```

one shared softmax:

```text
O = softmax(QK^T)V
```

Use source-type embeddings/bias.

This is analogous to PRA native-KV integration.

Compare directly against A2.

---

## 16. A4 — Separate NL and ASL/world encoders

```text
NL  → E_N → source memory
ASL → E_A → world memory
```

`E_A` should not process long NL no-man's-land regions.

This directly addresses sparse symbolic annotations in long text: the ASL path is activated only for symbolic records, not for every NL token.

Compare against a shared source encoder with the same decoder and memory fusion.

---

## 17. A5 — Hybrid semantic convergence

Use:

```text
NL  → E_N lower/mid ─┐
                     ├→ E_NA shared semantic/world top → KV_N_world
ASL → E_A lower/mid ─┘                              → KV_A_world
```

Bottom/middle layers are different because NL and ASL surfaces differ.

Top layers are shared because both describe the same world.

Initially expose only:

```text
KV_N_world
KV_A_world
```

Later optionally expose:

```text
KV_N_surface
KV_A_surface
```

for exact names/literals alongside semantic memory.

---

## 18. A6 — Multi-depth memories

Compare:

```text
top-only
mid+top
layer-routed
```

Potential interpretation:
- lower/mid: names, literals, local syntax;
- upper: entities, relations, roles, world state.

Do not globally compress long source/world state into one hidden vector.

---

## 19. A7 — State-aware block controller

Baseline:

```text
p(block | NL)
```

Advanced:

```text
p(block | NL, selected world state)
```

Architecture:

```text
NL encoder
+
world-memory query
→ START / CONTINUE / END / TYPE
```

Use dependent examples such as:

```text
"That value is twice the previous result."
```

Compare:
- NL only;
- NL + textual result;
- NL + typed world K/V.

---

## 20. A8 — Multipart selection becomes graph construction

Evolve output from:

```text
[start,end]
```

to:

```text
BlockPlan {
  source_span
  source_dependencies[]
  world_dependencies[]
  intervention_type
}
```

ASL gets:

```text
current block
+ selected source
+ selected world
```

Metrics:
- span F1;
- source dependency P/R;
- world dependency P/R;
- intervention-type accuracy;
- final semantic accuracy.

---

## 21. A9 — Typed segment DAG / block-sparse attention

Typed segments:

```text
SOURCE
ASL
RESULT
WORLD
QUERY
ANSWER
```

Define visibility:

```text
M(i,j) = Visible(segment_j, segment_i)
```

rather than only:

```text
j <= i
```

Example:

```text
later SOURCE may see prior SOURCE + RESULT
but not raw ASL trace

ASL may see source through block end + prior WORLD
```

Compare to flat causal prompt.

---

## 22. A10 — Separate persistent world memory

Maintain:

```text
SOURCE MEMORY
immutable evidence

WORLD MEMORY
evolving symbolic state
```

World records include:
- entities;
- paths;
- values;
- events;
- relations;
- constraints;
- verified results;
- provenance.

New block:

```text
selected source
+
selected world
→ ASL update
```

Preferred long-term architecture.

---

## 23. A11 — World-state storage policy

Compare:

### Append-only version history

```text
count@t0=26
count@t1=22
count@t2=16
```

### Current-state record + version history/reference

```text
current count=16
history -> prior records
```

Measure:
- retrieval;
- stale-state errors;
- temporal queries;
- memory growth.

Never mutate state without provenance/versioning.

---

## 24. A12 — Residual/direct vector injection control

Ablation only:

```text
r = Encode(R)
h' = h + g W_r r
```

Potentially useful for tiny scalar results.

Risks:
- lossy compression;
- poor provenance;
- hard selective retrieval;
- weak scaling to structured state.

Use as control, not preferred architecture.

---

# PRA / SELECTION SCALING

## 25. PRA ladder

Only after typed source/world memory works:

```text
P0 full source + full world
P1 selected source
P2 selected world
P3 selected source + world
P4 hierarchical typed PRA records
P5 oracle relevant subgraph
```

PRA may select:
- source spans;
- ASL;
- CogCop results;
- world-state records.

Do not confound architecture quality with retrieval quality.

---

## 26. Source encoding strategy

Compare later:

### E0 Whole source once
```text
E_N(document) → immutable source memory
```

### E1 Segmented online
```text
source → block → ASL → result → resume source
```

### E2 Whole immutable source + evolving world
Encode source once and update world separately.

**E2 is the preferred long-term hypothesis** because source evidence remains stable while cognition evolves.

---

# COMPARISON MATRICES

## 27. Compact first matrix

Do not run everything immediately.

| ID | Selector | ASL mechanism | Injection | Memory |
|---|---|---|---|---|
| V0 | oracle | static LoRA | final-only | flat |
| V1 | learned | switched LoRA | text | flat |
| V2 | learned+retrieval | switched LoRA | structured text | flat |
| A0 | same selector | same ASL target | typed result cross-attn | result memory |
| A1 | same selector | dedicated ASL decoder | E_A world K/V | source+world |
| A2 | world-aware | hybrid E_N/E_A/E_NA | merged K/V | typed source+world |

This ladder locates where architectural changes begin to matter.

---

## 28. Injection matrix

```text
I0 NONE
I1 TEXT_RESULT
I2 TEXT_STRUCTURED_STATE
I3 RESULT_ENCODER_CROSS_ATTN
I4 ASL_WORLD_ENCODER_CROSS_ATTN
I5 ASL_WORLD_MERGED_KV
I6 RESIDUAL_VECTOR
```

Metrics:
- later-block semantic accuracy;
- final answer;
- result-use rate;
- state consistency;
- injected token count;
- K/V bytes;
- latency.

---

## 29. Selection matrix

```text
S0 ORACLE
S1 SENTENCE
S2 CLAUSE
S3 TOKEN_START_END
S4 START_END + SOURCE_RETRIEVAL
S5 START_END + SOURCE + WORLD_RETRIEVAL
S6 MULTI_LAYER CONTROLLER
```

Metrics:
- span quality;
- FAR;
- critical recall;
- source/world dependency recall;
- end-to-end accuracy.

---

# SCALING

## 30. Block-count scaling

Report for:

```text
0,1,2,4,8,16
```

blocks/document:
- final accuracy;
- ASL accuracy by block index;
- cumulative world-state error;
- false activation;
- records/KV/tokens;
- runtime calls;
- wall time.

---

## 31. Dependency-depth scaling

Test:

```text
depth 0,1,2,4,8
```

independently of number of blocks.

Measure:
- dependency selection;
- propagation;
- stale references;
- final answer.

---

## 32. Scope scaling

Test:
- one scope;
- nested subscopes;
- sibling scopes;
- repeated symbols with new meaning.

Measure:
- cross-scope leakage;
- wrong symbol reuse;
- missed parent reuse;
- scope reset.

---

# NO-MAN'S-LAND / SPARSITY

## 33. Sparse ASL activity

Long documents have large regions with no formal semantics.

For vanilla switched-adapter conditions measure:
- ASL-adapter activation density;
- false activation in ordinary prose;
- controller drift over long gaps.

For separate `E_N/E_A` architectures:
- `E_A` should process only ASL/world records;
- no full parallel ASL sequence should be created;
- symbolic memory should remain sparse/addressable.

This is a key architectural motivation.

---

# RESULT AUTHORITY / ROBUSTNESS

## 34. Result corruption

Inject:
- correct;
- stale;
- wrong;
- conflicting;
- unrelated;
- correct value/wrong provenance.

Measure:
- trust calibration;
- conflict detection;
- override behavior;
- downstream degradation.

Typed records should include:

```text
MODEL_ASSERTED
RUNTIME_VERIFIED
RETRIEVED
ORACLE
UNRESOLVED
CONFLICT
```

Do not encode authority only as prose.

---

# METRICS / COST

## 35. Metric hierarchy

Always factor:

```text
SPAN
→ DEPENDENCY_SELECTION
→ ASL_PARSE
→ ASL_SEMANTIC
→ EXECUTION
→ RESULT_ENCODING
→ RESULT_RETRIEVAL
→ RESULT_USE
→ FINAL
```

---

## 36. Efficiency

Track:
- source tokens encoded;
- ASL tokens generated;
- textual injection tokens;
- typed records encoded;
- K/V memory bytes;
- selector FLOPs;
- ASL decoder FLOPs;
- CogCop CPU time;
- adapter switches;
- attention positions;
- end-to-end wall time.

Architectural variants should report one-time source encoding separately from incremental world writes.

---

# HYPOTHESES / GATES

## 37. Main hypotheses

H1. Vanilla selector + LoRA + text injection solves a substantial subset cheaply.

H2. Text injection degrades in efficiency/reliability as interventions grow.

H3. Typed world K/V improves reuse and avoids neural re-interpretation of deterministic results.

H4. Separate `E_N/E_A` avoids sparse/no-man's-land symbolic processing.

H5. Shared `E_NA` improves NL↔world semantic alignment.

H6. World-aware selectors improve dependent block detection.

H7. Multipart selection becomes graph construction rather than simple START/END.

H8. Merged native-style K/V can beat separate cross-attention for source/world integration.

H9. PRA becomes useful only after memory-selection pressure is demonstrated.

---

## 38. Falsification

Architectural complexity is not justified if:

```text
vanilla selector + switched LoRA + text injection
≈
typed-memory architecture
```

at matched quality/cost.

Typed world memory is weakened if it does not improve:
- result use;
- semantics;
- state reuse;
- or cost.

Shared semantic layers are weakened if alignment improves but autonomous behavior does not.

PRA is unnecessary if full source/world memory remains cheap and reliable.

---

## 39. Compute gates

```text
G0 oracle long-block compiler works
G1 learned selector has acceptable FAR/critical recall
G2 text injection improves dependent tasks
G3 typed K/V injection beats/matches text at lower cost
G4 separate source/world encoders improve semantics/scaling
G5 shared E_NA improves semantic binding
G6 world-aware selector improves dependent blocks
G7 PRA only after measured memory pressure
```

Do not automatically proceed through every gate.

---

# RECOMMENDED EXECUTION ORDER

## 40. First architectural comparison

After existing simple long-context plan:

**Baseline**
```text
same block selector
+ ASL LoRA
+ textual CogCop result injection
```

**Architectural**
```text
same selector
+ same ASL target
+ result canonicalized to ASL/F3
+ E_A(result)
+ decoder cross-attends to result K/V
```

Keep all other variables fixed.

This isolates **text reinjection vs symbolic world-memory injection**.

---

## 41. Second comparison

If typed result memory helps:

```text
shared source/result encoder
vs
separate E_N + E_A
```

Keep selector, decoder, and memory fusion fixed.

---

## 42. Third comparison

If separation helps:

```text
E_N + E_A
vs
E_N + E_A + shared E_NA
```

Measure:
- binding;
- world-state correctness;
- final answer;
- selector/retriever quality.

---

## 43. Fourth comparison

Using winning encoder architecture:

```text
separate cross-attention
vs
merged source/world K/V
```

Do not change selector simultaneously.

---

## 44. Fifth comparison

Add:

```text
NL-only block controller
vs
NL + typed-world controller
```

using dependent long-document blocks.

---

# RELATION TO OTHER PROJECT LINES

## 45. ASL grounding

The same `E_A` should preferably encode:

```text
external ASL teacher
internal ASL
CogCop result ASL/world records
persistent world state
```

Do not create unrelated symbolic encoders unless evidence requires it.

---

## 46. PRA

PRA contributes:
1. preserve detailed K/V rather than global hidden-vector compression;
2. cheap selection representations before detailed materialization;
3. native K/V concatenation under one shared softmax where appropriate.

Apply only after typed memories and actual selection pressure exist.

---

# DELIVERABLES

## 47. Expected files

When work begins:

```text
docs/AGENTS_long_multipart_architecture.md

src/.../
  block_controller.py
  block_planner.py
  source_memory.py
  world_memory.py
  result_encoder.py
  injection.py
  dependency_selector.py
  segment_visibility.py

configs/.../
  long_vanilla_*.yaml
  long_arch_*.yaml

artifacts/.../
  blocks/
  selection/
  injection/
  source_world_memory/
  scaling/
  analysis/
```

Add deterministic tests for:
- future-source leakage;
- cross-scope leakage;
- result provenance;
- typed-memory visibility;
- dependency selection;
- gold text-vs-K/V semantic equivalence;
- segment-DAG masks.

---

## 48. Final research question

The experiment series should determine:

> At what point does long/multipart cognitive processing stop being adequately modeled as prompt engineering around a flat Transformer, and become better modeled as interaction among linguistic evidence, a persistent symbolic world, deterministic processors, and typed attention memories?

The vanilla track is the strongest simple baseline.

The architectural track must earn its complexity through better:
- semantic binding;
- state reuse;
- intervention precision;
- long-context scaling;
- result integration;
- or compute efficiency.
