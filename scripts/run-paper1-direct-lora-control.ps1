param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][ValidateSet("xpu", "cuda")][string]$Device,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [ValidateSet("Prepare", "Train", "Evaluate", "All")][string]$Phase = "All"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$env:PYTHONPATH = Join-Path $root "src"
$scale = Join-Path $root "artifacts\paper1\gsm8k_scale_v1"
$source = Join-Path $scale "u2000_e4500"
$dev = Join-Path $scale "g1_f0_4500\eval\dev.jsonl"
$raw = Join-Path $root "artifacts\paper1\dsl\raw_v1\gsm8k.jsonl"
$control = Join-Path $root "artifacts\paper1\direct_lora_control_v1"
$data = Join-Path $control "data"
$run = Join-Path $control "qwen06_r8_seed99173_$Device"
$trainConfig = Join-Path $root "configs\paper1\e3_gsm8k_direct_lora_budget_qwen_$Device.json"
$evalConfig = Join-Path $root "configs\paper1\gsm8k_direct_reasoning_qwen_$Device.json"
$adapterId = "Qwen3-0.6B-GSM8K-Direct-U2000-E4500-r8-seed99173-$Device"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python executable is unavailable: $PythonExecutable"
}

Push-Location $root
try {
    if ($Phase -in @("Prepare", "All") -and -not (Test-Path (Join-Path $data "manifest.json"))) {
        & $PythonExecutable -u -m ccpu.paper1.e3 prepare-gsm8k-direct-lora `
            --train (Join-Path $source "train.jsonl") --dev $dev `
            --raw-gsm8k $raw --output-dir $data
        if ($LASTEXITCODE -ne 0) { throw "Direct LoRA data preparation failed" }
    }

    if ($Phase -in @("Train", "All") -and -not (Test-Path (Join-Path $run "training_report.json"))) {
        & $PythonExecutable -u -m ccpu paper1 train-lora `
            --config $trainConfig --model "Qwen/Qwen3-0.6B" `
            --train (Join-Path $data "train.jsonl") --dev (Join-Path $data "dev.jsonl") `
            --output-dir $run
        if ($LASTEXITCODE -ne 0) { throw "Direct LoRA training failed" }
    }

    if ($Phase -in @("Evaluate", "All")) {
        foreach ($item in @(
            @{ Name = "ordinary"; Eval = (Join-Path $scale "official_test_v1\confirmatory.jsonl") },
            @{ Name = "large"; Eval = (Join-Path $scale "large_number_v1\data\large.jsonl") }
        )) {
            $output = Join-Path $run "eval\$($item.Name)"
            if (-not (Test-Path (Join-Path $output "summary.json"))) {
                & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-direct-shard `
                    --eval $item.Eval --config $evalConfig --condition direct_reasoning `
                    --adapter-path (Join-Path $run "adapter") --adapter-id $adapterId `
                    --output-dir $output --shard-index 0 --shard-count 1 --checkpoint-every 1
                if ($LASTEXITCODE -ne 0) { throw "Direct LoRA $($item.Name) evaluation failed" }
            }
        }
    }
}
finally {
    Pop-Location
}
