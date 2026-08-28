# AGENTS - Paper 0 v4

## Role

Paper 0 v4 is the evidence-aware position paper for the heterogeneous cognitive
Transformer program. It preserves the twin-track thesis:

- computational coprocessors address competence deficits;
- epistemic coprocessors address knowledge deficits;
- both use a common observable interrupt and typed-state contract;
- explicit tools and upfront RAG remain baselines and general fallbacks.

The paper is architectural and falsifiable. It is not an empirical paper and
must not present developmental program results as confirmed general findings.

## Source of truth

- Manuscript: `paper0_v4.tex`
- Built artifact: `paper0_v4.pdf`
- Revision specification: `../../AGENTS_paper0_v4.md`
- Prior version: `paper0_v3.tex`

Paper 0 has no dedicated runtime, configuration, or artifact tree. Reusable
program code belongs under `src/ccpu/common`; paper-specific experiment code
belongs under the relevant later paper tree.

## Frozen v4 decisions

- The central question is which computational or epistemic substrate should
  assist evolving neural computation, through which interface and coupling.
- Different coprocessor families may have different optimal interfaces.
- The coupling ladder has eight rungs and places semantic execution blocks
  between strict reflexes and implicit semantic interrupts.
- Runtime state is called cognitive state only when it persists beyond an
  isolated response and can participate in later interpretation, reasoning,
  selection, verification, or generation.
- The unified failure taxonomy is detection, exposure, selection,
  normalization, routing, execution/retrieval, state update, materialization,
  use, override, and safety.
- Evidence gates resolve interface questions. They do not require each earlier
  mechanism to outperform every baseline.
- Later claims about persistent state, learned routing, native materialization,
  and co-adaptation remain hypotheses.
- PRA is not central before Paper 5.

## Developmental evidence

Paper 1's strict Qwen3-0.6B XPU pilot reported 75% LLM-only and strict-reflex
accuracy, 87.5% explicit-tool accuracy, and 100% oracle accuracy. The calculator
was exact when invoked; candidate selection and result use were limiting.

Paper 1's later hard held-out extension reported 43.75% LLM-only, 62.5%
explicit, 37.5% strict reflex, 75% normalized reflex, 12.5% calculator blocks,
and 87.5% oracle accuracy over 16 arithmetic cases. Normalized reflex versus
LLM-only had unadjusted exact McNemar p=0.0625. No condition falsely intervened
on 12 controls. This extension is reported separately and remains preliminary.

Paper 1.5's tuned controlled-source pilot reported 83.3% accuracy at 83.3%
retrieval for FLARE-like confidence and 91.7% at 91.7% retrieval for semantic
OR confidence. One high-confidence, high-risk stale case drove the difference.
Upfront RAG also reached 91.7% while retrieving all examples. Do not state
confirmed superiority over FLARE.

## Update guardrails

- Preserve negative and mixed evidence; do not rewrite the history as a steady
  sequence of wins.
- Attribute failures to the boundary stage rather than calling every issue a
  generic tool-use failure.
- Keep engine correctness separate from end-to-end correctness.
- Preserve explicit invocation and confirmation for ambiguous, expensive,
  side-effectful, or high-risk capabilities.
- Update the early-evidence table only from frozen paper artifacts.
- Advance roadmap language only when the preceding interface question is
  characterized; do not generalize an unresolved interface.
- Mention the Cognitive Machines B-series in one future-work sentence only.

## Build

Build directly from this directory without Docker:

```powershell
& 'C:\Users\j.simao\AppData\Local\ccpu-tools\tectonic-0.17.0\tectonic.exe' `
  'paper0_v4.tex' --keep-logs --keep-intermediates
```

A release build must produce `paper0_v4.pdf` without LaTeX errors, undefined
references, undefined citations, or overfull boxes.
