# AGENTS - Paper 1.5

## Mission
Test the smallest credible epistemic coprocessor by separating semantic
epistemic risk from checkpoint uncertainty during generation, using exactly one
controlled retrieval source. After that primary experiment is frozen, test where
stable one-source retrieval policy should live: context, adapter weights, or the
runtime.

## Central question
Is epistemic risk distinct from model uncertainty?

The placement extension is secondary: does semantic epistemic policy add
information beyond uncertainty, and can that stable policy be stored efficiently
in weights?

## Terminology
- **High/low epistemic risk** is the conceptual distinction.
- **Retrieval-required** means a reliable answer depends on an external or
  source-specific record rather than model weights or supplied active context.
- **Retrieval-not-required** means the answer is stable, already supplied, or
  belongs to another coprocessor class.
- **Unsupported Commitment Rate (UCR)** is the fraction of retrieval-required
  opportunities that emit unsupported factual commitments.
- **Authorized Commitment Coverage** is the fraction of retrieval-required cases
  with sufficient evidence that emit a supported answer.

## Sequencing
Phase A confirms semantic risk versus confidence. Phase B begins only after the
Phase A benchmark, confidence probes, threshold policy, source, and evaluation
are frozen.

## Phase A benchmark
Oversample candidates, measure confidence on each evaluated checkpoint, and only
then assign quadrants. The untouched test set must retain at least 20 examples in
each Qwen-measured quadrant:

| Confidence | Retrieval requirement |
|---|---|
| High | Not required |
| Low | Not required |
| High | Required |
| Low | Required |

Retrieval-required subclasses include fresh/current, source-version-specific,
private source, changed familiar, unavailable, conflicting, exact-attribution,
and structured one-source values. Retrieval-not-required subclasses include
stable facts, supplied context, freshness distractors, quotations, hypotheticals,
non-factual prose, and compute-coprocessor needs.

## Phase A conditions
1. LLM only.
2. Anti-hallucination prompt.
3. Upfront RAG.
4. Explicit retrieval.
5. FLARE-like confidence trigger.
6. Semantic heuristic trigger.
7. Confidence OR semantic.
8. Confidence AND semantic.
9. Retrospective verification.
10. Evidence advisory.
11. Evidence plus abstention instruction.
12. Runtime epistemic gate.
13. Oracle.

FLARE remains mandatory. Document its short-span simplifications and calibrate
confidence thresholds per checkpoint unless cross-checkpoint calibration is the
explicit experiment.

## Evidence enforcement
The runtime gate enforces support relative to configured sources; it does not
claim truth. If retrieval is required and evidence is UNVERIFIED, CONFLICT,
AMBIGUOUS, stale, or otherwise insufficient, an unsupported factual value must
not be accepted. Keep statuses and record provenance explicit.

## Phase B placement
Compare:

- **Context:** few-shot retrieve/not-retrieve demonstrations.
- **Weights:** a small LoRA adapter emits a typed one-source retrieval request or
  `NO_RETRIEVAL`; it learns selection and serialization, never answer values.
- **Runtime:** confidence thresholds, semantic heuristics, request validation,
  source access, evidence status, bounds, provenance, and enforcement.
- **Combinations:** confidence OR semantic, adapter plus confidence, and oracle.

The typed request is:

```retrieve
entity=...
attribute=...
as_of=...
source=atlas
```

Use Qwen3-0.6B, SmolLM2-1.7B, and Gemma3-1B when checkpoint access and XPU
execution are stable.

## Leakage controls
- Disjoint entity namespaces, record IDs, and answer values across train/dev/test.
- Held-out benchmark entities and values excluded from adapter data.
- Targets contain protocol requests, not source answers.
- Machine-readable overlap audit must pass before training.

Memorizing finite registry values invalidates the placement experiment.

## Metrics
- Final exact accuracy and Wilson intervals where appropriate.
- UCR and Authorized Commitment Coverage.
- Retrieval precision/recall, false retrieval, and missed retrieval.
- Confident hallucination catch rate.
- Low-confidence/no-deficit false retrieval.
- Retrieval by measured confidence/risk quadrant.
- Evidence status, abstention, advisory, and runtime-enforcement rates.
- Selection, serialization, and interface success for Phase B.
- Prompt, generated, and reinjected tokens; model/source calls; wall time.
- Hallucination reduction versus retrieval-cost Pareto curves.

## Failure taxonomy
Missed or false interrupt, malformed request, wrong entity/attribute/time,
retrieval miss, stale/conflicting evidence, correct evidence ignored, model
override, unsupported acceptance, adapter answer leakage, and threshold
miscalibration.

## Scope
Paper 1.5 uses one source. No DB-versus-vector-versus-web routing, learned
multi-source router, symbolic reasoning, rollback, or PRA/native KV dependency.
Learned selection is forbidden in the Phase A primary result and allowed only in
the frozen Phase B one-source placement experiment.

Do not claim active retrieval as novel. The primary claim is supported only if
semantic epistemic-risk policy catches evidence needs missed by uncertainty at a
competitive cost.

## Falsification
Semantic policy adds no value if FLARE catches the same cases at equal or lower
cost, rules duplicate uncertainty, false triggers erase reliability gains, or
high-confidence/high-risk cases are immaterial. Weight placement adds no value if
context or runtime matches its reliability and recurring cost, or if an adapter
succeeds by answer memorization.

## Paper 2.5 gate
Paper 2.5 is justified when at least two model families reduce UCR with the
runtime gate while retrieving less often than upfront RAG, and/or a source
heterogeneity oracle shows genuine value. Learned source routing is not implied.

## Deliverables
The repository must contain the frozen quadrant-aware benchmark, confidence
probes, controlled source, all condition traces, UCR/coverage summaries,
evidence-enforcement ablation, multi-family replication, leakage-audited
retrieval-policy adapters, context/weights/runtime placement analysis, plots,
manifests, the rebuilt paper, and an explicit Paper 2.5 gate decision.

## Natural-language iteration outcome
Natural-v5 contains 96 train, 32 development, and 64 test examples. The freezer
reports zero duplicate questions, zero normalized-template overlap, and zero
source-key collisions, plus zero split-template/gold-answer mismatches. Its
strongest shallow lexical baseline reaches 85.94%,
so it passes the 90% screen but remains an easy synthetic benchmark.

Across Qwen3-0.6B, SmolLM2-1.7B, and Gemma3-1B, the frozen semantic policy has
100% retrieval recall and 12.5% false activation. Runtime enforcement has zero
UCR and full authorized coverage for all three. Advisory and support-contract
behavior remains checkpoint-dependent. Retrospective verification detects all
wrong required forecasts but does not correct them without enforcement.

The learned-policy decision is **no-go** on this consumed freeze. Any retry needs
richer natural data, a new untouched freeze, and must compare a CPU classifier,
rules, confidence, LoRA, and hybrids. The 12-opportunity long-form result is a
deterministic controller/cache stress test, not free-running model generation.
