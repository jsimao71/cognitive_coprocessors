# Local Codex F3 Categorical Repair Prompt

Read and follow `skills/f3-grounded-compiler/SKILL.md` and its linked F3-v1
reference. Read only the supplied repair batch. Each item contains raw source,
your prior F3 draft when available, and one categorical failure class.

Repair each draft from the raw source. The failure class is directional lint,
not semantic supervision. Do not infer or request benchmark answers, expected or
actual runtime values, numeric deltas, rationales, ASL/F0/F1/F2 targets,
blackboard state, validator details, or intermediate execution values. Do not
calculate hidden observations.

For `not_lowerable`, ensure every query-relevant symbol is connected to explicit
observations through registered F3 forms and remove no source assertions merely
to force closure. For `evidence_invalid`, use exact contiguous source spans or
exact non-empty table cells. For `answer_mismatch`, recompile independently from
the raw source without guessing the answer. An actually unrepresentable record
must remain `unsupported`.

Set `answer_hidden=true`, `rationale_hidden=true`,
`prior_programs_hidden=false`, `runtime_state_hidden=true`, and
`annotator="local_codex_guided_repair"`. Return only the JSON value required by
the output schema.
