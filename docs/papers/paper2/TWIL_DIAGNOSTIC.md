# Paper 2 TwIL comparison diagnostic

## Status

The first local TwIL-LM3 comparison is **diagnostic only**. It is valid evidence
about the tested low-latency typed-interface prompt, but it is not a fair ranking
of reasoning in weights versus reasoning in coprocessors.

## Preserved result

The run uses the official BF16 checkpoints at immutable revisions:

- `HuggingFaceTB/SmolLM3-3B@a07cc9a04f16550a088caea529712d1d335b0ac1`
- `webAI-Official/TwIL-LM3@715bde26ca65dc386eb7a035635182d8f361c096`

Both use greedy decoding, the same prompts, `/no_think`, a 160-token ceiling,
and Intel XPU execution. Strict rescoring requires exact formalization and exact
execution; an engine returning the right Boolean for the wrong IR receives no
credit.

| Condition | Exact tasks | Overall | Truncated |
|---|---:|---:|---:|
| SmolLM3 neural | 16/22 | 18/26 | 1/26 |
| TwIL neural | 16/22 | 18/26 | 2/26 |
| SmolLM3 hybrid | 6/22 | 6/26 | 3/26 |
| TwIL hybrid | 7/22 | 7/26 | 4/26 |

The neural 16/22 result is not substantive: all 16 Horn/graph answers are
`true`, so a constant `true` response is perfect. Both models fail all six
calculator/date/units tasks without engines. With the interface, both solve the
two calculators, two unit conversions, and one date. SmolLM3 formalizes one of
eight Datalog cells exactly; TwIL formalizes two. Neither formalizes a graph cell
exactly. Every exact IR executes correctly.

Persistent typed state is separately valid deterministic evidence. At 100
queries over one world, amortized execution improves by 13.7x for Datalog and
3.1x for graph relative to rebuilding closure for each query.

## Why it is not rankable

1. The Horn/graph labels are all positive.
2. TwIL documents greedy thinking-mode evaluation with 2048 generated tokens
   and one 4096-token retry; this pilot forces `/no_think` and 160 tokens.
3. The suite has four semantic controls but no reproduced FOL translation,
   semantic parsing, Lean, or rule-induction lane.
4. The hybrid path uses the exact runtime display as the final answer and does
   not test neural result integration.
5. The sample is too small for a headline model comparison.

## Failure diagnosis

The low-latency ICL interface is the bottleneck. SmolLM3 hybrid records 20
failures: five malformed IRs, three missed delegations, three wrong-fact sets,
three truncations, two false activations, two wrong semantic answers, one wrong
engine, and one wrong query. TwIL hybrid records 19 failures: five wrong-engine
selections, four truncations, three wrong-fact sets, three false activations, two
malformed IRs, one wrong query, and one wrong semantic answer. TwIL changes the
failure distribution but yields only one additional exact item.

## Required correction before manuscript use

- Freeze balanced true/false/unknown Horn and graph cells.
- Add a small TwIL-aligned entailment/FOL/semantic-parsing subset.
- Run matched thinking-mode internal reasoning with a declared resource-bounded
  token budget and truncation audit.
- Keep the 160-token no-thinking condition only as a latency/interface ablation.
- Test compact IR and, if XPU memory permits, a small interface LoRA.
- Add an explicit neural result-use pass before claiming hybrid final accuracy.

Paper 3 and Paper 3.5 remain paused. This diagnostic does not reopen the existing
Paper 3 gate.
