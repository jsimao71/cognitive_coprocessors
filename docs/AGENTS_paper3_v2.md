# AGENTS --- Paper 3 v2

## Latent Cognitive Blocks: Parallel Causal Span Control for JIT Coprocessor Invocation

## Reframe

Replace the old engine-specific semantic-router plan. Paper 2 shows
exact parsing/execution is strong once capability is known, while
engine-specific learned routing has poor FAR/generalization. Paper 3 now
asks whether a Transformer can expose START, TYPE, and END of a
cognitive-assistance span through a parallel causal control channel
without serializing control syntax into language.

Paper 2's engine-specific `no_go` remains valid; generic latent block
detection is an independent exploratory `go`.

## Stable control vocabulary

`NONE, COMPUTE, RETRIEVE, VERIFY, HELP`. The controller never predicts
calculator/Datalog/etc.; R2 runtime/PRA resolves concrete capabilities.

## State machine

OUTSIDE: NONE or START\_{COMPUTE,RETRIEVE,VERIFY,HELP}. INSIDE(type):
CONTINUE or END. No nesting/type switching. On END, pause generation,
extract naturally generated span, route/parse/execute, store provenance,
inject/materialize result, resume OUTSIDE.

## Controller formulations

1.  BIO/BIES: O and B/I/E per type.
2.  Preferred minimal START/END: outside predicts start/type; inside
    predicts end. Runtime constrains transitions.

## Payload ladder

Use marked ordinary text span first. If insufficient: deterministic
parser; PRA/ICL selected capability skill; short capability-specific
normalization; only then learned structured-argument head.

## Multi-layer control

Final layer is baseline only. Probe \~0.25L, .40L, .55L, .70L, .85L, L
for START/type/END.

Aggregation ladder: - final-layer; - best dev-selected layer; - fixed
mean of layer logits; - majority/threshold vote; - learned scalar layer
weights; - tiny MLP over layer logits; - hidden-state fusion only if
needed.

Allow different layer mixtures for START and END.

## Hybrid END

Compare neural END, deterministic parser completeness, AND, and safe OR.
Arithmetic/date/units may use deterministic completeness more strongly
than RETRIEVE/HELP.

## Training ladder

P0 frozen linear probes. P1 frozen multi-layer ensemble. P2 small
control LoRA only if needed. P3 optional multitask LM+control training.
Never train concrete registry identity into R1.

## Baselines

Matched examples: - paired tags; - fenced blocks; - label-only intent; -
CPU trigger; - latent span controller; - oracle span/type.

This isolates language-bus vs parallel-control-bus transport.

## Dataset

New causal span freeze covering calculator/date/units, graph/Datalog,
retrieval, verification, HELP, NONE, quoted tags/fences, tool discussion
without need, and longer generations with multiple opportunities. Split
paraphrase families/namespaces. Headline inference is causal; no future
tokens.

## Metrics

START/END P/R/F1; exact span; IoU; type accuracy; FAR; premature/late
END; parseable payload; R2 success; final accuracy; controller
params/FLOPs/latency; explicit control tokens avoided; earliest reliable
layer; cross-layer agreement/calibration.

## Registry invariance

Train R1 with a small registry, then add graph/Datalog and later
SMT/algebra/retrieval backends without retraining. Score R1 and R2
separately.

## Result integration

Keep triggering and assimilation separate. Runtime-copy remains default
for final exact values. For CONTINUE compare typed/text authoritative
result and optional separate integration LoRA. Where feasible use
factorial token-vs-latent control × base-vs-integration-LoRA.

## PRA

Optional hook: latent intent -\> runtime/PRA capability search -\>
materialize 1-k relevant skills -\> execute -\> task-aware typed
state/materialization.

## Early exit

Only after strong multi-layer evidence, test high-confidence START
before full layer stack. Exploratory only.

## Falsification

Not justified if explicit generic blocks are equally reliable/cheap, END
is brittle, multi-layer aggregation adds no value, hidden-state coupling
is too model-specific, or CPU triggers dominate.

## Immediate order

1.  Freeze span dataset.
2.  Derive explicit and latent labels from same examples.
3.  Final-layer frozen probe.
4.  Layer sweep.
5.  Multi-layer aggregation.
6.  BIO vs START/END.
7.  Parser-assisted END.
8.  Compare tags/fences/label-only/CPU.
9.  Registry expansion.
10. Control LoRA only if needed.
11. Integration LoRA separately.
12. Early exit only after gate.

## Deliverables

New `AGENTS_paper3.md`, `paper3.tex`, span benchmark, probes,
aggregator/state machine, explicit-vs-latent comparison,
registry-invariance test, control/integration factorial,
cost/portability analysis, Paper 7 recommendation.
