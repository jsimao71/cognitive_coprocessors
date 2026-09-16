# Paper 3 Patch --- Generic Tools vs Parallel Latent Cognitive Blocks

## Mandatory four-tool baseline

Use exactly `__compute(...)`, `__retrieve(...)`, `__verify(...)`,
`__help(...)`. These generic gateway tools MUST use the same R2
registry, parsers, engines, provenance, policies, typed state, and
result integration as CogCop. Concrete registry entries remain outside
the recurring schema.

Compare voluntary generic tools, generic CogCop token/block, CPU
automatic trigger, latent parallel control where applicable, and oracle
timing/type. Match checkpoint, prompts, decoding, backend
registry/results, R2 policy, assistance opportunities, and safety
limits.

Measure reliability (assistance recall, FAR, R1/R2 accuracy, final
accuracy, UCR, result use, malformed args), interface cost
(schema/ICL/generated/argument/result tokens), execution cost (model
calls, decode steps, harness/network round trips, time-to-assistance,
CPU/XPU/wall time), registry scaling, and autonomy.

Define
`Automatic Rescue Rate = voluntary-tool misses rescued by automatic/latent interrupt / assistance-required cases missed by voluntary tools`.

If tools match quality/cost, tighter coupling is not justified. If
CogCop only saves tokens/network, claim engineering efficiency only. A
deeper claim requires automatic rescue, better JIT timing, CONTINUE, or
scaling benefits.

## Core comparison

Same semantic actions, different transport: T0 native voluntary
four-tool calling; T1 paired cognitive tags; T2 fenced blocks; T3
label-only intent; T4 CPU automatic trigger; T5 final-layer latent
START/END; T6 multi-layer latent controller; T7 voluntary tools + latent
watchdog; T8 oracle.

## Main distinction

Both strongest baselines have only four actions. The question is
voluntary language/action control versus independent decoding-coupled
control. Report voluntary assistance recall, watchdog rescue, error
correlation, and false interventions.

## Payload

Compare model-generated structured tool args, generic-block
natural-language payload, latent marked natural-language span, and
oracle span + deterministic parser. Score malformed args separately from
intent.

## Timing annotations

Record earliest useful interrupt, latest safe interrupt, voluntary
tool-call time/token, latent START/END, and first wrong/unsupported
token. Report lead time, late calls, premature interrupts, and prevented
commitments.

## Multi-layer hypothesis

Multi-layer latent control earns its complexity only if it gives earlier
reliable START, lower FAR, or useful signals before the language/tool
action becomes available. Otherwise reject it.

## Local-vs-local first

Compare in-process tool and CogCop execution before network tests. Never
attribute a network advantage if only tools are remote. Report
remote/harness deployment separately.

## Result representation

Feed identical engine results as standard tool message, immediate text
injection, typed authoritative record, runtime-copy, and gated
integration LoRA. This separates trigger gains from injection gains.

## PRA factorial

Where possible: tools; tools+PRA; token CogCop; token CogCop+PRA; latent
CogCop; latent+PRA. PRA owns progressive registry/state materialization;
Paper 3 owns control transport.

## Fundamental-advantage criteria

Claim more than engineering efficiency only if latent/automatic control
shows reproducibly higher recall at matched FAR, meaningful Automatic
Rescue Rate, earlier prevention of wrong/unsupported tokens, better
CONTINUE under matched result representation, better depth/size scaling,
or useful pre-language multi-layer control signals. Otherwise prefer
four generic tools for portability.
