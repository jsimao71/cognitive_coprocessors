# Cognitive Coprocessor Research Series

- Paper 0: Heterogeneous Cognitive Transformers: From Tool Use to Cognitive Coprocessors
- Paper 1: Reflex Computation: Automatic Calculator Assistance During Autoregressive Decoding
- Paper 2: Heterogeneous Reflex Reasoning: Multiple Symbolic Coprocessors with Typed Micro-State
- Paper 3: Learned Semantic Interrupts for Cognitive Coprocessors
- Paper 4: Transactional Cognitive State: Hypotheses, Provenance, Retraction, and Backtracking
- Paper 5: Native Coprocessor Context: From Text Reinjection to Structured and KV-Level Interfaces
- Paper 6: Learning to Offload: Co-Adaptation Between Transformers and Cognitive Coprocessors
- Paper 7: A Heterogeneous Cognitive Runtime for Language Models

## Experiment code

Reusable runtime, trace, artifact, generation, and metric contracts live under
`src/ccpu/common`. Paper-specific implementations live under `src/ccpu/paper{n}`;
Paper 1's strict reflex calculator is under `src/ccpu/paper1`. See
`src/ccpu/README.md` for smoke and pretrained reproduction commands.

## Paper PDFs

Install Tectonic 0.17.0, then rebuild every paper beside its TeX source:

```powershell
$env:TECTONIC = 'C:\path\to\tectonic.exe'
.\scripts\build-papers.cmd
```

The script pins `SOURCE_DATE_EPOCH` so unchanged sources produce byte-stable
PDFs across repeated builds.
