# AGENTS PATCH — Paper 3: Semantic Compute Interrupts After Learned Blocks

## Purpose
Reframe Paper 3 in light of Paper 1's strong LoRA+block result.

Paper 3 must no longer assume that semantic detection is automatically the next/better coupling mechanism. A LoRA-trained model emitting typed execution blocks already performs semantic detection, engine selection, serialization, and boundary declaration.

The burden of proof is now:

> Does an implicit semantic compute interrupt solve a measured limitation of learned typed blocks strongly enough to justify its additional complexity?

## Default baseline architecture
Treat the simpler architecture as primary:

natural language -> model + interface LoRA -> typed engine block -> deterministic runtime -> exact result -> continuation

Paper 3 compares against this, not merely against explicit tools or regex reflexes.

## Revised research questions
1. Do learned blocks leave systematic computational opportunities unexposed?
2. Does block serialization consume enough tokens/latency to matter under frequent micro-interventions?
3. Does block reliability degrade with very small models or large engine catalogs?
4. Can an external/text-level or hidden-state detector recover those failures without unacceptable false activation?
5. Is independent compute-side detection useful as a watchdog rather than a primary router?

## Entry gate
Do NOT implement a complex semantic router until Paper 2 identifies at least one residual limitation:

- missed offload opportunities despite engine availability;
- block token overhead materially affects quality/cost;
- engine-family selection degrades with catalog size;
- very small models cannot reliably emit the protocol;
- latent computation needs occur before an explicit block can be formed;
- block omission creates a measurable reliability gap.

If Paper 2 finds none of these, Paper 3 should be narrowed, postponed, or published as evidence that explicit learned cognitive protocols are sufficient for compute-side coupling.

## Conditions
1. LoRA + typed block baseline.
2. Runtime deterministic reflex where applicable.
3. Explicit tools.
4. Text-level semantic sidecar detector.
5. Optional hidden-state detector only after text-level evidence.
6. Block + watchdog hybrid.
7. Oracle opportunity detector.
8. Oracle engine/payload.

## Semantic watchdog mode
Prioritize a low-complexity watchdog before replacing block routing.

Behavior:
- model normally emits learned blocks;
- sidecar observes ordinary generation;
- if a high-confidence computational opportunity appears without the expected block, sidecar may interrupt;
- runtime validates the proposed event before execution.

This tests recovery rather than competing control systems.

## Candidate residual use cases

### Frequent micro-computation
Many small arithmetic/date/unit events where explicit block tokens become expensive.

### Latent formal need
The model is reasoning about constraints/relations but does not voluntarily emit Datalog/SMT/graph blocks.

### Very small models
Test whether sidecars permit smaller neural cores than block protocols alone.

### Large capability catalogs
If 16/32+ engine families cause block selection confusion, compare external routing.

## Detector ladder
Escalate only when justified:

1. deterministic grammar/lexical detector;
2. classical/tiny CPU classifier;
3. small neural text classifier;
4. main-model text-level semantic head;
5. hidden-state probe/router.

Every level must beat or solve a failure of the previous level.

## Hidden-state routing gate
Hidden-state integration is NOT a default Paper 3 deliverable.

Only attempt it if:
- block/text-level routing leaves a substantial measured gap; and
- the expected benefit is large enough to justify model/inference-server coupling.

Require explicit comparison of:
- quality;
- tokens;
- latency;
- false activations;
- model-version portability;
- observability/debuggability;
- engineering complexity.

## Metrics
- offload-opportunity recall;
- block omission rate;
- watchdog recovery rate;
- false-intervention rate;
- correct engine selection;
- payload correctness;
- final answer;
- generated block tokens avoided;
- detector CPU/GPU cost;
- wall time;
- engine calls;
- result use/override;
- performance vs engine-count;
- performance vs model size.

## Engineering-complexity accounting
Paper 3 should explicitly report implementation burden:
- additional learned parameters;
- runtime components;
- model hooks;
- hidden-state access requirements;
- calibration requirements;
- portability across inference engines/checkpoints;
- failure/debugging surface.

A small accuracy gain with large integration complexity is not enough.

## Falsification
Implicit semantic compute interrupts are NOT justified if:
- LoRA blocks already capture nearly all useful opportunities;
- watchdog recovery is negligible;
- block-token savings are small;
- false triggers offset recovered opportunities;
- hidden-state routing is brittle across checkpoints;
- external routing costs more than the neural work it displaces;
- explicit blocks remain better for safety/auditability.

A negative result is scientifically useful.

## Relationship to Paper 2
Paper 2 establishes heterogeneous explicit learned execution protocols and engine-count scaling.

Paper 3 exists only to solve residual compute-side coupling problems revealed there.

## Relationship to Paper 3.5
Do not assume compute and epistemic interrupts have equal value.

Independent epistemic monitoring has a stronger architectural motivation because the generator may be confidently wrong and fail to request evidence.

## Relationship to Paper 6/7
If semantic compute interrupts add little, Paper 6/7 should retain blocks/reflexes rather than forcing tighter coupling.

The final runtime may legitimately use:
- reflex for trivial compute;
- learned blocks for most compute;
- semantic watchdog only for selected cases.

## Immediate execution order
1. Wait for/inspect Paper 2 residual failure analysis.
2. Select ONE measured block limitation.
3. Build oracle headroom for that limitation.
4. Implement cheapest detector capable of addressing it.
5. Test watchdog hybrid.
6. Escalate to learned/hidden detector only if justified.
7. Produce explicit go/no-go decision for implicit compute interrupts.

## Deliverables
- revised Paper 3 manuscript/AGENTS;
- residual-failure benchmark from Paper 2;
- oracle opportunity annotations;
- text-level/watchdog implementation;
- block-vs-watchdog quality/cost comparison;
- optional hidden-state experiment only after gate;
- engineering-complexity table;
- explicit recommendation for Paper 7 coupling mode.
