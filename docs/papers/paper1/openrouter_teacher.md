# OpenRouter ASL Teacher Campaign

This campaign is separate from the local Codex corpus. It queues all 7,473 GSM8K
training programs and all 706 arithmetic-compatible TAT-QA development programs.
Each request hides the benchmark answer and rationale and asks for every clause
mapping in one response.

The configuration in `configs/paper1/dsl_teacher_openrouter_free.json` contains
five zero-price OpenRouter model routes, but no credential. Set the credential
only in the process environment:

```powershell
$env:OPENROUTER_API_KEY = (Get-Content C:\Users\j.simao\zgit\rd\openrouter-token.txt -Raw).Trim()
```

Run or resume either queue with the isolated teacher environment:

```powershell
& C:\Users\j.simao\.venvs\ccpu-teacher\Scripts\python.exe -m ccpu.dsl_dataset generate-programs `
  --input artifacts\paper1\dsl\openrouter_full_v1\inputs\gsm8k_full.jsonl `
  --config configs\paper1\dsl_teacher_openrouter_free.json `
  --skill skills\ccir-arith-compiler\SKILL.md `
  --output-dir artifacts\paper1\dsl\openrouter_full_v1\gsm8k
```

For every program, the runner tries at most five models. JSON-shape, ASL syntax,
lowering, type, and execution failures trigger the next model. Attempts,
accepted annotations, failures, model identity, and skill hash are checkpointed.
Completed rows are skipped on restart. If every route is rate limited, the
current and remaining rows stay pending for a later resume.

Refresh conversion and quality statistics with:

```powershell
python -m ccpu.dsl_dataset analyze-programs `
  --source artifacts\paper1\dsl\openrouter_full_v1\inputs\gsm8k_full.jsonl `
  --source artifacts\paper1\dsl\openrouter_full_v1\inputs\tatqa_arithmetic_full.jsonl `
  --remote-dir artifacts\paper1\dsl\openrouter_full_v1\gsm8k `
  --remote-dir artifacts\paper1\dsl\openrouter_full_v1\tatqa `
  --baseline artifacts\paper1\dsl\annotated_v1\accepted.jsonl `
  --baseline artifacts\paper1\dsl\expansion500_v1\freeze\expansion_train.jsonl `
  --output-dir artifacts\paper1\dsl\openrouter_full_v1\analysis
```

The aggregate report distinguishes generated, executable, lint-valid, and
answer-verified programs. Detailed question/ASL sample pairs remain in ignored
JSONL; only IDs and aggregate statistics need to be tracked. Free-route results
are a screening corpus, not semantic gold. Hard failures can later be isolated
for a stronger paid teacher or local Codex review without mixing provenance.
