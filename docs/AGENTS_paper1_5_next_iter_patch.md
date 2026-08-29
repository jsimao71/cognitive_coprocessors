# AGENTS PATCH — Paper 1.5 Next Iteration

## Why this patch is needed
Paper 1 now provides a general interface-placement lesson:
- context/ICL teaches a protocol quickly but costs recurring tokens and can cause false interventions;
- small LoRA adapters can store stable semantic selection/serialization behavior;
- runtime should keep deterministic retrieval mechanics, bounds, provenance, reinjection, and enforcement.

Paper 1.5 should preserve its original primary scientific question:
> Is epistemic risk distinct from model uncertainty?

Do not prematurely turn it into Paper 3.5.

## Revised sequencing
Phase A — confirm the original semantic-risk vs confidence result.
Phase B — only after Phase A is frozen/evaluated, run a secondary context-vs-weights-vs-runtime placement experiment for one-source retrieval-required policy.

## Terminology
Keep conceptual label:
- high/low epistemic risk

Add operational label:
- retrieval-required / retrieval-not-required

Definition:
> retrieval-required means a reliable answer depends on an external/source-specific source rather than model weights or already supplied active context.

## Phase A benchmark expansion
The current pilot is too small and tuned.

Build a new untouched set with materially larger measured quadrants. Aim, after checkpoint confidence is measured, for roughly >=20 examples in each:
- high confidence / retrieval-not-required
- low confidence / retrieval-not-required
- high confidence / retrieval-required
- low confidence / retrieval-required

Oversample candidates then assign quadrants from actual checkpoint confidence.

## Retrieval-required subclasses
Include:
1. fresh/current facts
2. source-version-specific facts
3. private/controlled-store facts
4. changed familiar facts
5. unavailable evidence
6. conflicting evidence
7. exact source attribution
8. structured-source value if still one controlled source

## Retrieval-not-required subclasses
Include:
1. stable familiar facts
2. answer already in context
3. freshness lexical distractors
4. quotations
5. hypotheticals
6. non-factual prose
7. computational needs that should go to a compute coprocessor instead of retrieval

## Stronger hallucination metric
Add:
### Unsupported Commitment Rate (UCR)
Unsupported emitted factual commitments / retrieval-required opportunities.

Also add:
### Authorized Commitment Coverage
Fraction of retrieval-required cases where sufficient evidence exists and the system emits a supported answer.

## Evidence-enforcement ablation
Current pilot showed no-evidence and retrospective-correction failures.

Compare:
1. evidence advisory only
2. evidence + abstention instruction
3. runtime epistemic gate

For the runtime gate, if retrieval is required but support is UNVERIFIED/CONFLICT/insufficient, do not allow an unsupported factual value to be accepted.

This enforces support policy relative to configured sources; it does not claim truth.

## Phase B — placement extension
After Phase A:

### Context
Few-shot retrieval-required / not-required demonstrations.

### Weights
Train a small adapter to emit a one-source retrieval control/request, e.g.

```retrieve
entity=...
attribute=...
as_of=...
source=atlas
```

The adapter learns retrieve/not-retrieve and request serialization, not answer values.

### Runtime
Keep:
- confidence thresholds
- semantic heuristics
- source access
- request validation
- evidence status
- bounds
- provenance
- enforcement

Compare whether stable retrieval policy benefits from weight placement as calculator policy did.

## Leakage controls
Because the source is synthetic/finite:
- disjoint entity seeds
- disjoint record IDs
- disjoint answer values
- held-out paraphrases
- overlap audit

A model that memorizes registry values invalidates the epistemic experiment.

## FLARE remains mandatory
Compare:
- confidence-only
- semantic rules
- confidence OR semantic
- learned retrieval-required adapter
- adapter + confidence
- oracle

The enhanced question becomes:
> Does semantic epistemic policy add information beyond uncertainty, and can that stable policy be stored efficiently in weights?

## Multi-model extension
Use:
- Qwen3-0.6B
- SmolLM2-1.7B
- Gemma if available

Confidence thresholds should be calibrated per checkpoint unless explicitly testing cross-model calibration.

## Scope boundary
Paper 1.5 remains ONE source.

Do not learn DB-vs-vector-vs-web routing here.
That belongs to Paper 2.5/3.5.

## Revised Paper 2.5 gate
Paper 2.5 is justified when:
1. generation-time retrieval has a confirmed regime with a favorable unsupported-claim/cost tradeoff; and/or
2. oracle source-heterogeneity experiments show distinct sources have real value.

Do not require heuristic semantic rules to be the final production trigger.

## Immediate order
1. Freeze current Paper 1.5 policy.
2. Build larger untouched quadrant-aware set.
3. Run Qwen replication.
4. Replicate on second family.
5. Add runtime evidence-enforcement ablation.
6. Only then run context-vs-LoRA placement extension.
7. Decide Paper 2.5 gate.

## Deliverables
- patched Paper 1.5 AGENTS
- larger measured-quadrant benchmark
- UCR + authorized-coverage metrics
- evidence-enforcement condition
- second-family replication
- optional one-source retrieval-policy LoRA
- leakage audit
- context/weights/runtime placement table
- explicit Paper 2.5 gate decision
