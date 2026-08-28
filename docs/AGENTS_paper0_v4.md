# AGENTS — Paper 0 v4

## Title
Heterogeneous Cognitive Transformers: Computational and Epistemic Coprocessors

## Mission
Revise Paper 0 v3 into a stronger evidence-aware position paper without changing its central thesis. Preserve the twin-track architecture: computational coprocessors address competence deficits; epistemic coprocessors address knowledge deficits.

The revision must integrate what Papers 1 and 1.5 have already taught us about the neural↔coprocessor boundary, refine the coupling ladder, add semantic execution blocks as an intermediate interface, and make the research gates depend on resolving interface questions rather than requiring every earlier mechanism to “win.”

This remains a position paper. Preliminary program evidence may be summarized, but do not convert Paper 0 into an empirical paper.

## Core thesis to preserve
A Transformer remains the general learned semantic/generative substrate, but it need not approximate every computational operation internally and need not acquire every fact from parametric memory.

The complete system can include specialized, tightly integrated components that:
- recognize opportunities for assistance during inference;
- normalize those opportunities into typed operations or information needs;
- invoke bounded computational or epistemic engines;
- preserve provenance-bearing cognitive state;
- materialize only the relevant result back into neural computation;
- fall back to explicit tool calling when automatic coupling is uncertain, expensive, unsafe, or unsupported.

Do not frame the architecture as “tools, but automatic.”

The broader systems question is:

> Which specialized computational or epistemic substrate can assist the evolving neural computation now, through what interface, and at what coupling level?

## Required v4 additions

### 1. Add an “Early Evidence from the Program” section

Keep this concise and explicitly developmental.

#### Paper 1 — strict calculator reflex
Report:
- Qwen3-0.6B XPU pilot;
- LLM-only and strict reflex both at 75% exact match;
- explicit calculator at 87.5%;
- oracle at 100%;
- calculator engine exact whenever invoked;
- only about 50% of accepted reflex candidates represented the intended full expression;
- result-use/override failures occurred.

Interpretation:
- calculator execution is not the bottleneck;
- exposure, selection, normalization, timing, and result use are first-class interface problems;
- automatic invocation does not automatically beat explicit invocation;
- the naive thesis “automatic reflex > tool call” is falsified in this easy regime.

Do not overstate significance.

#### Paper 1.5 — semantic epistemic risk vs confidence
Report:
- Qwen3-0.6B controlled-source developmental pilot;
- FLARE-like confidence retrieval: 83.3%;
- semantic OR confidence: 91.7%;
- difference driven by one high-confidence/high-epistemic-risk stale case;
- confidence over-retrieved low-confidence/low-risk examples;
- upfront RAG also reached 91.7% but retrieved on every example;
- protocol was tuned during development, so no confirmatory claim.

Interpretation:
- neural uncertainty and epistemic risk are observably distinct in at least one controlled case;
- semantic risk may complement confidence;
- retrieval coupling currently has a more promising early signal than strict arithmetic reflex coupling.

Required framing:
> The difficult part of heterogeneous cognitive systems is often the neural↔coprocessor boundary, not the specialized engine itself.

Do not present Paper 1.5 as confirmed superiority over FLARE.

### 2. Add a capability-specific coupling principle

Explicitly state:
> Different coprocessor families may have different optimal interfaces.

Add a table like:

| Capability | Likely interface | Main failure risk | Fallback |
|---|---|---|---|
| Calculator | strict/semantic block | premature subexpression | explicit tool |
| Logic | typed block/parser | wrong formalization | explicit symbolic request |
| Retrieval | epistemic interrupt | false/missed evidence need | explicit search/RAG |
| Web/current facts | semantic+temporal trigger | stale/source conflict | explicit search |
| Code | fenced executable block | unsafe execution | sandbox/tool call |
| High-risk action | explicit invocation | unintended side effect | confirmation |

This is a design hypothesis, not an established result.

### 3. Add semantic execution blocks to the coupling ladder

Introduce an intermediate coupling mode where the model explicitly delimits a machine-executable cognitive region without authoring a verbose API schema.

Examples:

```calculator
15246377 * 746647383
```

```datalog
fact isa(penguin,bird)
fact isa(bird,animal)
query isa(penguin,animal)
```

```python
# bounded sandboxed code
```

Runtime behavior:
- detect opening block type;
- wait for the closing delimiter;
- normalize enclosed content;
- execute with the corresponding bounded local engine;
- return a typed result;
- continue decoding.

Position this between explicit tool calls and implicit semantic interrupts.

Suggested revised ladder:
1. explicit tool/function call;
2. strict syntax/grammar reflex;
3. semantic execution blocks / typed fenced regions;
4. lexical/syntactic semantic interrupt;
5. heuristic routing;
6. learned text-level interrupt;
7. hidden-state routing;
8. jointly trained/native interface.

Mention Self-RAG-style control tokens as an architectural precedent for learned control signals, while clarifying that execution blocks generalize control across computational families.

### 4. Operationally define “cognitive state”

Use a technical criterion:

> Runtime state is treated as cognitive state when it persists beyond an isolated service response and can participate in subsequent interpretation, reasoning, selection, verification, or generation.

Distinguish ordinary one-shot tool results from typed, provenance-bearing, potentially persistent cognitive micro-state.

Example fields:
- stable ID;
- type;
- payload;
- engine/source version;
- epistemic status;
- confidence;
- scope;
- dependencies;
- provenance;
- timestamp.

### 5. Revise the evidence gates

Replace:
> Paper N must produce a positive win before Paper N+1 can proceed.

With:
> A later paper may proceed when the earlier interface question has been resolved sufficiently to motivate the next mechanism.

Examples:

#### Paper 1 → Paper 2
Paper 2 should not require strict reflex to win. It requires that the computational interface problem be sufficiently characterized.

Possible outcomes:
- strict reflex works → reuse it;
- semantic execution block works better → Paper 2 uses blocks;
- strict reflex fails but exposes semantic-selection failure → Paper 3 becomes more urgent;
- explicit tools remain best → retain them as fallback and narrow the automatic-compute claim.

#### Paper 1.5 → Paper 2.5
Require evidence that generation-time retrieval has value in some regime and/or oracle source heterogeneity has measurable headroom.

#### Paper 2.5 → Paper 3.5
Learned routing is justified only if heterogeneous source selection matters and heuristic routing leaves a measurable oracle gap.

#### Paper 4+
Transactional state is justified only when persistent multi-engine/multi-source state creates concrete provenance/retraction problems.

General rule:
> Do not generalize an unresolved interface.

### 6. Expand the failure taxonomy

Use a unified decomposition:

- Detection: missed interrupt; false interrupt.
- Exposure: relevant operation never represented in usable form.
- Selection: wrong candidate/subexpression selected.
- Normalization: semantic meaning mapped incorrectly to micro-IR.
- Routing: wrong engine/source.
- Execution/retrieval: engine error, timeout, stale/irrelevant/conflicting evidence.
- State update: wrong provenance, duplicated/stale state, dependency loss.
- Materialization: too much/too little/wrong state reinjected.
- Use: model ignores correct result/evidence.
- Override: model later contradicts correct state.
- Safety: unsafe execution or unintended external action.

Use Papers 1 and 1.5 to show that engine correctness alone is insufficient.

### 7. Strengthen the fallback hierarchy

Describe the final architecture as hierarchical rather than replacement-oriented:

- Level A — automatic cheap reflexes;
- Level B — bounded semantic execution blocks;
- Level C — automatic semantic/epistemic interrupts;
- Level D — learned routing;
- Level E — explicit tool calling for arbitrary/ambiguous/expensive/side-effectful capabilities;
- Level F — user confirmation for high-risk actions.

Explicit tools remain the universal slow/general fallback.

### 8. Tighten FLARE / Self-RAG positioning

FLARE:
- precedent for generation-time retrieval;
- confidence is one trigger signal.

Our question:
- is semantic epistemic risk complementary to uncertainty?
- can routing span heterogeneous evidence substrates?

Self-RAG:
- precedent for learned retrieval/reflection control in the main LM;
- precedent for special control tokens.

Our question:
- must control live in the main LM?
- can sidecar detectors/routers be model-independent?
- can the same general control architecture extend from retrieval to computation?

Do not claim learned adaptive retrieval, support/relevance tokens, or generation-time retrieval as novel.

### 9. Broaden architectural antecedents

Required related-work categories:
- Toolformer/function calling;
- ReAct/planner-executor;
- PAL/program-aided LMs;
- FLARE/Self-RAG/adaptive retrieval;
- neuro-symbolic reasoning;
- theorem-prover interaction;
- solver composition;
- blackboard architectures;
- production systems;
- classical cognitive architectures;
- neural module networks;
- heterogeneous computing/coprocessor analogies;
- shared-state/event-driven systems;
- typed intermediate representations;
- safe execution sandboxes.

Position this as a modern Transformer-era synthesis rather than invention of heterogeneous cognition.

### 10. Add “What would falsify the broad program?”

The broad program is weakened if, across multiple domains:
- explicit tool calling consistently matches or dominates automatic coupling on accuracy/cost;
- semantic interrupts create more false interventions than useful assistance;
- state/provenance overhead exceeds saved neural computation;
- heterogeneous routing provides no benefit over universal tools/retrievers;
- small models do not gain disproportionately;
- tighter interfaces do not improve scaling with task difficulty;
- hidden/native coupling adds complexity without measurable value.

A successful program does not require every coprocessor family to win. A valid outcome may be:
> some capabilities benefit from reflex coupling, others from execution blocks, others from learned routing, and high-risk actions remain explicit.

### 11. Add a program-level claim hierarchy

- Claim A — specialized engines are often more reliable/efficient for exact operations or external evidence.
- Claim B — automatic generation-time coupling can reduce explicit invocation burden in selected regimes.
- Claim C — persistent typed cognitive state improves multi-step reasoning/reuse.
- Claim D — learned semantic interrupts outperform hand-built routing.
- Claim E — native/KV/latent coupling improves over text/structured materialization.
- Claim F — jointly trained heterogeneous systems develop a better division of labor than monolithic neural models.

Do not write later claims as if established by early papers.

### 12. Preserve the twin-track roadmap

Keep:
- Paper 0 — position;
- Paper 1 — calculator/interface;
- Paper 1.5 — single-source epistemic retrieval;
- Paper 2 — heterogeneous compute;
- Paper 2.5 — heterogeneous retrieval;
- Paper 3 — learned compute interrupts;
- Paper 3.5 — learned epistemic interrupts;
- Paper 4 — transactions/provenance/retraction;
- Paper 5 — structured/PRA/native materialization;
- Paper 6 — co-adaptation;
- Paper 7 — integrated runtime.

Annotate each transition with its evidence question.

### 13. Relationship to the Cognitive Machines B-series

Mention only briefly in Discussion/Future Work:

> If heterogeneous coprocessors become numerous and persistent, a later research direction can ask whether natural language should remain the universal internal interchange format, or whether typed/learned cognitive interlingua are more appropriate.

Do not expand B-series material inside Paper 0 v4.

## Required figures/tables

- Figure 1 — twin computational/epistemic architecture with common interrupt/state path.
- Figure 2 — capability-specific coupling map.
- Table 1 — competence vs knowledge deficit.
- Table 2 — conventional RAG / FLARE / Self-RAG / program comparison.
- Table 3 — capability-specific likely interface/failure/fallback.
- Table 4 — early program evidence from Papers 1 and 1.5, clearly marked developmental.
- Table 5 — roadmap + evidence gate per paper.

## Style requirements

- Keep tone architectural and falsifiable.
- Do not oversell “cognitive.”
- Avoid AGI/consciousness language.
- State clearly that Papers 1 and 1.5 are preliminary developmental evidence.
- Make negative results part of the argument.
- Prefer “interface/coupling problem” over “tool-use failure.”
- Preserve explicit tools and upfront RAG as legitimate baselines and fallbacks.
- Do not make PRA central before Paper 5.
- Do not let the paper become a catalog of mechanisms; maintain one coherent systems thesis.

## Deliverables

- `paper0_v4.tex`
- `paper0_v4.pdf`
- updated AGENTS context
- early-evidence table
- capability-specific coupling table
- revised coupling-ladder figure
- expanded related-work matrix
- roadmap/evidence-gate table
- explicit broad-program falsification section
