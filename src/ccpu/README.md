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
