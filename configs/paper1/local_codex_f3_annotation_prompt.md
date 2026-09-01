# Local Codex F3 Annotation Prompt

Read and follow `skills/f3-grounded-compiler/SKILL.md` and its linked F3-v1
reference. Read only the supplied request batch. Do not inspect any seed, ASL,
F0/F1/F2, accepted, rejected, runtime-state, rationale, or benchmark-answer
artifact.

Annotate every item independently from raw `question` and optional raw
`source_context`. Preserve source assertions and evidence; do not generate a
solution ledger or calculate hidden values.

Set `answer_hidden=true`, `rationale_hidden=true`,
`prior_programs_hidden=true`, `runtime_state_hidden=true`, and
`annotator="local_codex"`. Return only the JSON value required by the output
schema.
