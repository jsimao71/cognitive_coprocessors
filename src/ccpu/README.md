# CCPU experiment runtime

`ccpu.common` contains the reusable typed interrupt/runtime, trace, artifact, generation,
and metric contracts. `ccpu.paper1` contains only the strict reflex-calculator experiment.

Run the dependency-free protocol smoke test from the repository root:

```powershell
python -m pip install -e .
python -m ccpu paper1 generate --config configs/paper1/smoke.json --output artifacts/paper1/smoke/dataset.jsonl
python -m ccpu paper1 validate --dataset artifacts/paper1/smoke/dataset.jsonl
python -m ccpu paper1 simulate --dataset artifacts/paper1/smoke/dataset.jsonl --output-dir artifacts/paper1/smoke/run
python -m ccpu paper1 plot --summary artifacts/paper1/smoke/run/summary.json --output artifacts/paper1/smoke/run/scaling.png
```

The simulator verifies plumbing only and marks every summary `empirical: false`. It is not
evidence for the paper. Saved outputs from any model harness can be evaluated with `replay`.

For the pinned Qwen model-size sweep:

```powershell
python -m pip install -e ".[hf,analysis]"
python -m ccpu paper1 generate --config configs/paper1/primary.json --output artifacts/paper1/primary/dataset.jsonl
python -m ccpu paper1 run-hf --dataset artifacts/paper1/primary/dataset.jsonl --config configs/paper1/primary.json --output-dir artifacts/paper1/primary/run
```

Use `--limit`, `--model`, and repeated `--condition` flags for cheap preflight runs before the
full sweep. The Hugging Face adapter is correctness-first and recomputes the prefix per token;
optimized KV-cache interception is intentionally outside Paper 1.

## Paper 1 XPU pilot

The reported diagnostic pilot uses the validated PyTorch XPU environment. Its sample is too
small for confirmatory claims, but the preserved generations, traces, and replayed scores are
fully reproducible:

```powershell
$py = 'C:\Users\j.simao\.venvs\modal-llm-xpu\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path

& $py -m ccpu paper1 generate --config configs/paper1/xpu_pilot.json --output artifacts/paper1/xpu_pilot/dataset.jsonl
& $py -m ccpu paper1 run-hf --dataset artifacts/paper1/xpu_pilot/dataset.jsonl --config configs/paper1/xpu_pilot.json --output-dir artifacts/paper1/xpu_pilot/run96
& $py -m ccpu paper1 replay --dataset artifacts/paper1/xpu_pilot/dataset.jsonl --completions artifacts/paper1/xpu_pilot/run96/predictions.jsonl --output-dir artifacts/paper1/xpu_pilot/final
& $py -m ccpu paper1 plot --summary artifacts/paper1/xpu_pilot/final/summary.json --output docs/papers/paper1/paper1_xpu_pilot.png
```

For `device: auto`, the adapter selects CUDA, then XPU, then CPU. Replay preserves the original
backend metadata and records the source prediction hash plus evaluator provenance.

## Paper 1 protocol LoRA

The LoRA experiment teaches calculator-block selection and serialization, never arithmetic
answers. Generate the balanced data and run one pinned model at a time:

```powershell
$py = 'C:\Users\j.simao\.venvs\modal-llm-xpu\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
& $py -m ccpu paper1 generate-lora-data --config configs/paper1/lora_protocol_xpu.json --excluded-dataset artifacts/paper1/hard_heldout_xpu/dataset.jsonl --output-dir artifacts/paper1/lora_protocol/data_v1
& $py -m ccpu paper1 train-lora --config configs/paper1/lora_protocol_xpu.json --model Qwen/Qwen3-0.6B --train artifacts/paper1/lora_protocol/data_v1/train.jsonl --dev artifacts/paper1/lora_protocol/data_v1/dev.jsonl --output-dir artifacts/paper1/lora_protocol/qwen3_0_6b_v1
& $py -m ccpu paper1 train-lora --config configs/paper1/lora_protocol_xpu.json --model HuggingFaceTB/SmolLM2-1.7B-Instruct --train artifacts/paper1/lora_protocol/data_v1/train.jsonl --dev artifacts/paper1/lora_protocol/data_v1/dev.jsonl --output-dir artifacts/paper1/lora_protocol/smollm2_1_7b_v1
& $py -m ccpu paper1 train-lora --config configs/paper1/lora_protocol_xpu.json --model google/gemma-3-1b-it --train artifacts/paper1/lora_protocol/data_v1/train.jsonl --dev artifacts/paper1/lora_protocol/data_v1/dev.jsonl --output-dir artifacts/paper1/lora_protocol/gemma3_1b_v1
& $py -m ccpu paper1 run-hf --dataset artifacts/paper1/hard_heldout_xpu/dataset.jsonl --config configs/paper1/lora_gemma3_base_xpu.json --condition llm_only --condition calculator_block_minimal --condition calculator_block_icl_g --condition normalized_reflex --condition explicit_tool --condition oracle --output-dir artifacts/paper1/lora_protocol/gemma3_1b_base_eval_v1
& $py -m ccpu paper1 run-hf --dataset artifacts/paper1/hard_heldout_xpu/dataset.jsonl --config configs/paper1/lora_gemma3_adapter_xpu.json --condition calculator_block_minimal --condition calculator_block_icl_g --output-dir artifacts/paper1/lora_protocol/gemma3_1b_adapter_eval_v2
python -m ccpu paper1 analyze-placement --config configs/paper1/lora_placement_comparison.json --output-dir artifacts/paper1/lora_protocol/placement_comparison_v2
```

The generator rejects operand or expression overlap with the untouched held-out benchmark.
Adapter evaluation keeps `calculator_block_minimal` separate from the frozen ICL-G prompt so
context, weights, and deterministic runtime placement can be compared directly.
Manual generation honors the complete model EOS set; this includes Gemma's
`<end_of_turn>` token 106 in addition to tokenizer EOS.

## Paper 1.5 epistemic-risk replication and placement

The next iteration freezes a larger Qwen-measured four-quadrant benchmark before evaluation,
adds UCR/authorized-coverage and evidence-enforcement conditions, and replicates Phase A on
three model families. The secondary adapter learns one-source request policy, never answers:

```powershell
$py = 'C:\Users\j.simao\.venvs\modal-llm-xpu\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
& $py -m ccpu paper1.5 freeze-next --config configs/paper1_5/next_iter_freeze_qwen_xpu.json --output-dir artifacts/paper1_5/next_iter/freeze_qwen_v2
& $py -m ccpu paper1.5 run-hf --config configs/paper1_5/next_iter_qwen_xpu.json --output-dir artifacts/paper1_5/next_iter/qwen_v2
& $py -m ccpu paper1.5 run-hf --config configs/paper1_5/next_iter_smollm2_xpu.json --output-dir artifacts/paper1_5/next_iter/smollm2_v1
& $py -m ccpu paper1.5 run-hf --config configs/paper1_5/next_iter_gemma3_xpu.json --output-dir artifacts/paper1_5/next_iter/gemma3_v1
python -m ccpu paper1.5 analyze-next --config configs/paper1_5/next_iter_analysis.json --output-dir artifacts/paper1_5/next_iter/analysis_v1

& $py -m ccpu paper1.5 generate-policy-data --config configs/paper1_5/policy_lora_xpu.json --output-dir artifacts/paper1_5/next_iter/policy_data_v2
& $py -m ccpu paper1.5 train-policy-lora --config configs/paper1_5/policy_lora_xpu.json --model Qwen/Qwen3-0.6B --train artifacts/paper1_5/next_iter/policy_data_v2/train.jsonl --dev artifacts/paper1_5/next_iter/policy_data_v2/dev.jsonl --output-dir artifacts/paper1_5/next_iter/policy_qwen_v1
python -m ccpu paper1.5 analyze-policy --config configs/paper1_5/policy_placement.json --output-dir artifacts/paper1_5/next_iter/policy_analysis_v1
```

Each checkpoint fits its confidence threshold on development rows. Phase A has 13 conditions;
the policy-data audit rejects held-out entity/value overlap or answer-bearing targets.

## Paper 2 heterogeneous engines

The developmental run is complete and the confirmatory Paper 3 gate is `no_go`. The runtime
supports calculator, Datalog, graph, date, and dimension-checked units blocks through a reusable
registry. Generate the disjoint adapter data, run deterministic scaling/compositions, then train
one multi-engine adapter at a time:

```powershell
python -m ccpu paper2 generate-next --config configs/paper2/next_iter_xpu.json --output-dir artifacts/paper2/next_iter/data_v1
python -m ccpu paper2 run-next --config configs/paper2/next_iter_xpu.json --dataset artifacts/paper2/next_iter/data_v1/test.jsonl --condition runtime --catalog-size 5 --output-dir artifacts/paper2/next_iter/runtime_n5_v4
python -m ccpu paper2 compositions --count-per-family 20 --output-dir artifacts/paper2/next_iter/compositions_v1
& $py -m ccpu paper2 train-next-lora --config configs/paper2/next_iter_xpu.json --model qwen3_0_6b --train artifacts/paper2/next_iter/data_v1/train.jsonl --dev artifacts/paper2/next_iter/data_v1/dev.jsonl --output-dir artifacts/paper2/next_iter/lora_qwen_v1
```

Runtime-only and composition rows remain explicitly non-empirical. Model runs preserve
factorized detect/select/normalize/execute/reinject/use metrics and token/latency accounting.
The retained three-family adapters do not clear the five-engine selection or result-use gate.

## Paper 2.5 heterogeneous retrieval

Paper 2.5 uses a read-only source registry with embedded relational, lexical, vector, and
controlled fresh-web adapters. Run all source counts before deciding the learned-router gate:

```powershell
python -m ccpu paper2.5 freeze --output-dir artifacts/paper2_5/next_iter/data_v2
python -m ccpu paper2.5 run --benchmark artifacts/paper2_5/next_iter/data_v2/benchmark.jsonl --source-count 4 --output-dir artifacts/paper2_5/next_iter/source_n4_v2
python -m ccpu paper2.5 analyze --predictions artifacts/paper2_5/next_iter/source_n1_v2/predictions.jsonl artifacts/paper2_5/next_iter/source_n2_v2/predictions.jsonl artifacts/paper2_5/next_iter/source_n3_v2/predictions.jsonl artifacts/paper2_5/next_iter/source_n4_v2/predictions.jsonl --output-dir artifacts/paper2_5/next_iter/analysis_v2
python -m ccpu paper2.5 compositions --count-per-family 12 --output-dir artifacts/paper2_5/next_iter/compositions_v1
```

The oracle matrix decomposes need, source, query, retrieval, evidence status, and use. Public
registry metadata never exposes credential scopes, and every source is read-only. The retained
22-example freeze records a Paper 3.5 `no_go`: native sources outperform universal textualization,
but the transparent heuristic closes the oracle routing gap.

Install `.[data]` to substitute real local in-process engines without changing the frozen IR or
benchmark. This path uses DuckDB, SQLite FTS5, and FAISS; it does not start Docker or silently
fall back when a requested backend is missing:

```powershell
python -m pip install -e ".[data]"
python -m ccpu paper2.5 run --benchmark artifacts/paper2_5/production_v1/data/benchmark.jsonl --source-count 4 --backend-suite local_production --output-dir artifacts/paper2_5/production_v1/source_n4
python -m ccpu paper2.5 analyze-production --controlled-predictions artifacts/paper2_5/next_iter/source_n4_v2/predictions.jsonl --production-predictions artifacts/paper2_5/production_v1/source_n4/predictions.jsonl --production-traces artifacts/paper2_5/production_v1/source_n4/traces.jsonl --output-dir artifacts/paper2_5/production_v1/substitution
```

The optional `sidekick/data_stack` boundary is for WSL2 only. Its Postgres/pgvector integration
test requires `CCPU_POSTGRES_DSN` and otherwise skips explicitly; no service-backed result is
included in the paper. See the sidekick README for startup, health, test, teardown, and volume
cleanup commands.
