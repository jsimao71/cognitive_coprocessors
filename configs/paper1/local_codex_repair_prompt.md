# Local Codex Semantic Repair Prompt

Read and follow `skills/ccir-arith-compiler/SKILL.md` and both linked references.
The supplied batch contains only rows rejected by deterministic validation. This
is an adjudication pass: the expected answer and dataset derivation are visible
and must be recorded as rationale-assisted provenance.

For every item, diagnose the stated validation failure and return a complete
replacement annotation covering every source part. Preserve or improve semantic
entities, quantities, relations, time/state, and query intent. Use the rationale
only as an arithmetic constraint. Do not replace semantic state with `step_N`,
`tmp`, `result`, `value_N`, an answer literal, or a single opaque expression.
Ensure every path segment starts with a letter or underscore; use `y2019`, never
`.2019`. Include every source value needed for grounded execution. The final
dependency graph must be acyclic and grounded from literals; when the text gives
a relation in the opposite direction from the query, algebraically invert it
while retaining the semantic relation (for example, derive white bears as
`black_bears / 2` once black bears are grounded).

Set `annotator="codex"`, `answer_hidden=false`, `rationale_hidden=false`,
`rationale_assisted=true`, `repair_round` to the batch's supplied round, and
`manually_reviewed=false`.
Return only the JSON value required by the output schema.
