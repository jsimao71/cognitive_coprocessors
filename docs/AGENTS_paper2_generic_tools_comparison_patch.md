# Paper 2 Patch --- CogCop vs Four Generic Tools

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

## Matrix

C0 LLM only; C1 four generic tools; C2 optional engine-specific tools;
C3 generic COMPUTE/VERIFY/HELP block; C4 CPU trigger; C5 latent Paper-3
control; C6 oracle.

## Questions

-   Does voluntary `__compute()` miss cases automatic triggers catch?
-   Does structured tool-argument generation fail more than marked
    natural-language spans plus deterministic parsing?
-   Compare identical exact results as tool-result message vs typed
    CogCop record.
-   Does normal tool-call turn structure add latency/model calls versus
    JIT interruption?
-   Do four generic tools eliminate the original
    registry-size/engine-LoRA problem?

## COPY/INTERPRET/CONTINUE

Use identical backend outputs. Compare tool result, CogCop text result,
authoritative typed record, runtime-copy, and optional integration LoRA.
COPY keeps direct runtime render as production upper baseline. CONTINUE
is the key test of deeper integration.

## Registry mutation

Keep four tool schemas fixed while registry grows from
calculator/date/units to graph/Datalog and optionally SMT/algebra. No
model retraining. Compare generic CogCop R1 under identical mutations.

## JIT ladder

Use hard arithmetic/date/closure cases where the model may begin a wrong
answer. Record whether voluntary tool invocation happens before
commitment, whether automatic control catches it, first-wrong-token
prevention, and rescue rate.

## Claim gate

A fundamental CogCop claim requires higher assistance recall at
controlled FAR, better CONTINUE, lower JIT model/decode cost, better
problem-size scaling, or meaningful automatic rescue. Otherwise report
token/context/network/runtime engineering gains only.
