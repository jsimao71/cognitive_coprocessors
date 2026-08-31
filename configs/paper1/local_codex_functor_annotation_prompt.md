# Local Codex Functor Annotation Prompt

Read and follow `skills/functor-arith-compiler/SKILL.md` and its linked grammar.
Read only the supplied request batch. Do not inspect any seed, ASL, accepted,
rejected, state, rationale, or benchmark-answer artifact.

Annotate every item from raw `question` and optional raw `source_context`. Derive F1
and F2 independently from that raw input; do not treat F1 as an intermediate
translation source for F2. Use only the fixed ICL set in the grammar reference.

Set `answer_hidden=true`, `rationale_hidden=true`, `prior_asl_hidden=true`,
`blackboard_state_hidden=true`, and `annotator="local_codex"`. Return only the JSON
value required by the output schema.
