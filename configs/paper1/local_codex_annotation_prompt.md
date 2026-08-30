# Local Codex Semantic Annotation Prompt

Read and follow `skills/ccir-arith-compiler/SKILL.md` and both references linked
from it. Read only the supplied answer-free request batch; do not inspect seed,
raw, accepted, rejected, rationale, or benchmark-answer files.

Annotate every item and every supplied part. Produce clause-level semantic ASL
deltas in source order. Preserve entities, measured quantities, relations,
time/state, forward dependencies, and query intent. Use several statements for a
part when needed. Never use anonymous operation names such as `step_N`, `tmp`,
`result`, or `value_N`. Do not solve backward from an answer.

Set `status` to `ambiguous` or `unsupported` rather than inventing semantics. Set
`answer_hidden=true`, `rationale_hidden=true`, `rationale_assisted=false`,
`repair_round=0`, and `manually_reviewed=false`. Return only the JSON value
required by the output schema.
