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
```

The generator rejects operand or expression overlap with the untouched held-out benchmark.
Adapter evaluation keeps `calculator_block_minimal` separate from the frozen ICL-G prompt so
context, weights, and deterministic runtime placement can be compared directly.

## Paper 1.5 controlled retrieval pilot

Paper 1.5 measures checkpoint confidence and semantic epistemic risk separately against one
versioned fact store. The current XPU run is a developmental pilot, not held-out evidence:

```powershell
python -m ccpu paper1.5 validate --config configs/paper1_5/xpu_pilot.json
$py = 'C:\Users\j.simao\.venvs\modal-llm-xpu\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
& $py -m ccpu paper1.5 run-hf --config configs/paper1_5/xpu_pilot.json --output-dir artifacts/paper1_5/xpu_pilot
python -m ccpu paper1.5 plot --summary artifacts/paper1_5/xpu_pilot/summary.json --output docs/papers/paper1_5/paper1_5_xpu_pilot.png
```

The run writes the measured token probabilities, fitted threshold, typed source requests and
statuses, raw explicit-retrieval forecasts, ten-condition predictions, threshold sweeps, Pareto
frontiers, hashes, and environment provenance.

## Paper 2 heterogeneous protocol smoke

Paper 2 currently validates deterministic calculator, Horn, and ISA/frame routing plus
persistent typed state. The smoke run is explicitly non-empirical and requires no accelerator:

```powershell
python -m ccpu paper2 generate --config configs/paper2/smoke.json --output artifacts/paper2/smoke/dataset.jsonl
python -m ccpu paper2 validate --dataset artifacts/paper2/smoke/dataset.jsonl
python -m ccpu paper2 simulate --dataset artifacts/paper2/smoke/dataset.jsonl --output-dir artifacts/paper2/smoke/run
python -m ccpu paper2 plot --summary artifacts/paper2/smoke/run/summary.json --output docs/papers/paper2/paper2_protocol_smoke.png
```

Do not cite scripted accuracy as model evidence. The Paper 1 evidence gate for a Paper 2 model
experiment remains closed.
