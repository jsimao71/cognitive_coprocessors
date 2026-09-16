# AGENTS — Paper 1.5 Next Iteration: Natural-Language Epistemic Robustness

## Mission
Extend Paper 1.5 from the controlled-source result into a substantially more natural-language, multi-opportunity robustness evaluation while preserving the central question: **epistemic risk / retrieval requirement is not the same as model uncertainty**.

Current evidence: the transparent semantic runtime policy achieves 100% retrieval recall with 2.5% false activation across Qwen3-0.6B, SmolLM2-1.7B, and Gemma3-1B; FLARE-like confidence is checkpoint-dependent; runtime enforcement drives UCR to 0; the tested retrieval-policy LoRAs fail held-out generalization. The next iteration tests whether these findings survive less templated language.

## Primary questions
1. Does transparent semantic retrieval-required detection survive broad paraphrase and cue removal?
2. Does it still complement or beat confidence across model families?
3. Can it interrupt before unsupported commitments in longer generation?
4. Can runtime support enforcement keep UCR low without excessive abstention/retrieval?
5. Which semantic features genuinely generalize?

## Scope
Keep one controlled/versioned evidence source for the primary experiment. No multi-source routing; that remains Paper 2.5. Do not retune LoRA on the existing held-out set.

## Natural-language benchmark redesign

### Paraphrase diversity
For each underlying information need generate unrelated phrasings rather than one lexical template family. Include explicit, indirect, conversational, terse, and document-style forms.

### Cue removal
Include retrieval-required examples without obvious tokens such as `current`, `latest`, `today`, `registry`, or year markers. Retrieval should instead be required because of source version, private ID, authoritative attribution, unfamiliar dynamic field, or changed familiar fact.

### Hard negatives
Include lexical distractors: electric current, historical dates, quoted “latest/current” phrases, hypotheticals, fictional claims, source names used in prose, prompt-contained answers, and compute needs that belong to calculator/date/units rather than retrieval.

### Context-sufficient cases
Unusual or low-confidence answers that are already supplied in active context must remain retrieval-not-required.

### Familiar-but-stale cases
Include facts for which the model has a plausible strong parametric answer but the configured source authoritatively differs.

## Lexical-triviality audit
Before expensive evaluation train cheap classifiers on the benchmark labels:
- bag-of-words logistic regression;
- TF-IDF word n-grams;
- character n-grams.

Report their held-out accuracy. If labels are nearly perfectly separable by shallow lexical features, increase paraphrase and hard-negative diversity before claiming semantic robustness.

## Multi-opportunity long-form benchmark
Add short paragraphs/multi-step answers with several possible factual commitments. At least one evidence need should emerge after generation begins.

Measure per factual opportunity:
- retrieval before first unsupported value token;
- missed retrieval-required opportunity;
- late trigger after draft value;
- repeated unnecessary retrieval;
- evidence reuse later in the same response;
- UCR per commitment, not just per example.

## Prospective vs retrospective
Retain both prospective interruption before commitment and retrospective verification after a candidate claim. Do not assume retrospective correction is equivalent; current evidence suggests draft preservation/override is a separate failure mode.

## Runtime semantic policy
Keep features transparent and independently logged:
- temporal/freshness;
- source-specificity;
- private/controlled identifiers;
- attribution requirements;
- version/change semantics;
- active-context sufficiency;
- quote/hypothetical suppression;
- compute-vs-retrieval discrimination.

## Feature ablations
Run temporal-only, source-only, context-sufficiency-only, combined semantic, confidence-only, confidence OR semantic, and confidence AND semantic.

## Cross-model evaluation
Minimum: Qwen3-0.6B, SmolLM2-1.7B, Gemma3-1B. Add one larger model only as a robustness check if practical. Fit confidence thresholds per checkpoint unless explicitly studying transfer calibration.

## Enforcement ladder
### A — advisory
Evidence is supplied but model remains unconstrained.

### B — explicit support contract
Model is instructed to use supported evidence and abstain/report conflict otherwise.

### C — runtime epistemic enforcement
If retrieval is required and evidence is UNVERIFIED/CONFLICT/insufficient, reject unsupported factual commitment and force abstention/qualification/conflict state. This enforces policy relative to configured sources; it is not a truth oracle.

## Metrics
- Unsupported Commitment Rate (UCR)
- Authorized Commitment Coverage
- retrieval trigger precision/recall
- Early Catch Rate
- Late Catch Rate
- unnecessary retrieval rate
- evidence override rate
- abstention precision/recall
- final accuracy
- tokens/calls/latency
- per-feature trigger contributions

## Learned-policy follow-up gate
Do not immediately retry LoRA. Only reopen learned policy if the new benchmark is richer, runtime semantics leaves measurable headroom, training data covers that semantic diversity, and a new untouched freeze is created. If reopened, compare runtime rules, tiny CPU classifier, LoRA, confidence, and hybrids.

## Immediate order
1. Build natural-language candidate generator.
2. Run lexical-triviality audit.
3. Freeze a new untouched test set.
4. Evaluate semantic vs confidence across three families.
5. Add long-form multi-opportunity tasks.
6. Run advisory/instruction/enforcement ladder.
7. Run semantic feature ablations.
8. Decide whether learned epistemic policy deserves a new experiment.

## Deliverables
Updated manuscript/AGENTS, natural-language dataset generator, triviality audit, long-form benchmark, feature-level traces, cross-model confidence analysis, enforcement ablation, UCR/coverage/early-catch plots, and a new learned-policy go/no-go.
