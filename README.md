# Cognitive Coprocessor Series — Retrieval Track Patch

Reusable experiment infrastructure lives under `src/ccpu/common`; Paper 1's
strict reflex-calculator implementation lives under `src/ccpu/paper1`. See
`src/ccpu/README.md` for reproduction commands.

This patch adds the interleaved retrieval track and broadens Paper 0 to define twin coprocessor families.

- Paper 0 v2 — computational + epistemic coprocessor position
- Paper 1.5 — single-source reflex retrieval
- Paper 2.5 — heterogeneous retrieval engines with heuristic routing
- Paper 3.5 — learned epistemic interrupts and source routing

The intended sequence is:
0 → 1 compute → 1.5 retrieve → 2 compute → 2.5 retrieve → 3 learned compute trigger → 3.5 learned retrieval trigger → 4 unified transactional state → 5 structured/PRA/KV interface → 6 co-adaptation → 7 integrated runtime.
