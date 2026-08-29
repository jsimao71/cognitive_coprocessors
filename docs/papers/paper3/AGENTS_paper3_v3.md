# AGENTS - Paper 3 v3

## Mission

Test whether a parallel causal control channel can expose START, coarse TYPE,
and END of cognitive-assistance spans more reliably or cheaply than generic
language-bus control. This is not an engine-identity router.

## Gate inheritance

Paper 2's engine-specific learned-routing gate remains `no_go`. Paper 3 v3 is an
independent registered protocol for generic latent span detection, not evidence
that the failed gate passed. Do not claim latent-control success before retained
matched artifacts exist.

## Stable protocol

R1 predicts only `NONE`, `COMPUTE`, `RETRIEVE`, `VERIFY`, or `HELP`. R2 resolves
the active task, span, registry, policy, and state to a semantic capability or
shortlist. Never train concrete engine/backend identity into the primary R1
controller.

The preferred state machine is OUTSIDE -> START(type) -> INSIDE(type) -> END ->
execute/materialize -> OUTSIDE. No nesting or type switching. Headline inference
must be causal and use no future token.

## Required comparisons

- Paired generic tags.
- Fenced generic blocks.
- Label-only intent using the active task as payload.
- Transparent CPU trigger.
- Final-layer frozen probe.
- Multi-layer latent controller.
- Oracle span and type.

All conditions derive from the same examples and use matched runtime
capabilities. Separate control, normalization, execution, state update, result
integration, and final-answer errors.

## Registered ladders

Probe approximately 0.25L, 0.40L, 0.55L, 0.70L, 0.85L, and L. Compare final
layer, best dev layer, fixed logit mean, voting, learned scalar layer weights,
and a tiny MLP over logits. Hidden-state fusion is deferred.

Train frozen linear probes first, then frozen ensembles. A small control LoRA is
allowed only if needed. Integration LoRA is a separate factorial axis. Runtime
copy remains the authoritative default for exact final values.

Compare neural END, deterministic parser completeness, conservative AND, and a
policy-bounded safe OR. Report premature and late END separately.

## Dataset and invariance

Freeze a causal span benchmark covering calculator/date/units, graph/Datalog,
retrieval, verification, HELP, NONE, quoted syntax, tool discussion without
need, and longer multi-opportunity generations. Split paraphrase families and
namespaces.

The retained public control registry contains 280 safe references: 40 each from
GSM8K, BIG-bench Date Understanding, BIG-bench Unit Conversion, balanced
ProofWriter, CLUTRR, CRAG, and TAT-QA. It stores IDs/hashes and assistance types,
not benchmark content. TAT-QA compute-after-retrieve rows retain a secondary
`COMPUTE` label.

The registry manifest is `headline_ready: false`. FRAMES is absent and every
row still needs audited earliest START, latest safe START, canonical END,
payload span, assistance type, and first wrong/unsupported token. Never infer
these annotations automatically from answers for headline evaluation.

The mandatory public matrix is four generic tools, paired tags, fenced blocks,
label-only intent, CPU trigger, final-layer latent, multi-layer latent,
tools-plus-watchdog, and oracle timing/type with identical R2/backends. Existing
Paper 2/2.5 oracle transport audits are invariants, not model-facing conditions.
Bind the latest Paper 1.5 CRAG diagnostic and Paper 2.5 public-suite readiness
manifests by SHA-256. These add provenance only: both remain gated and neither
provides model-facing timing, evidence execution, final answers, or causal-span
labels. Keep `matched_model_facing_public_condition_results` as an explicit
headline blocker.

Train R1 with a small registry, then add graph/Datalog and later capabilities
without R1 retraining or prompt changes. Score R1 intent and R2 capability
selection separately.

## Metrics and falsification

Report START/END P/R/F1, exact span, IoU, type accuracy, FAR, completion timing,
parseability, R2 success, runtime/final exactness, cost, tokens avoided,
layer-wise decodability, agreement/calibration, and registry-change deltas.

Latent control is not justified if generic explicit blocks are equally reliable
and cheap, END is brittle, layer aggregation adds no value, portability is poor,
or CPU triggers dominate. Early exit remains gated and exploratory.

## Scope

No nested latent blocks, general planner, learned engine registry, primary
backtracking contribution, or unsupported positive claims. PRA is an optional
capability-materialization hook and is not required by the controller.

## Source of truth

- Manuscript: `paper3_v3.tex`
- Built artifact: `paper3_v3.pdf`
- Prior drafts: `paper3.tex`, `paper3_v2.tex`
- Roadmap input: `../../AGENTS_paper3_v2.md`
