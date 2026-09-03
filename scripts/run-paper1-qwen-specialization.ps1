param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = "C:\Users\j.simao\.venvs\modal-llm-xpu\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "XPU Python is unavailable: $Python"
}

$data = Join-Path $RepoRoot "artifacts\paper1\asl_matrix_v1\qwen_data\q1_seed11"
$source = Join-Path $RepoRoot "artifacts\paper1\asl_matrix_v1\data\source"
$runs = Join-Path $RepoRoot "artifacts\paper1\asl_matrix_v1\qwen_runs"
$evals = Join-Path $RepoRoot "artifacts\paper1\asl_matrix_v1\eval"

$conditions = @(
    @{
        Name = "q3s1_seed11"
        Config = "configs\paper1\asl_matrix_q3s1_capture_delta_xpu.json"
    },
    @{
        Name = "q3s2_seed11"
        Config = "configs\paper1\asl_matrix_q3s2_separate_xpu.json"
    },
    @{
        Name = "q3s3_seed11"
        Config = "configs\paper1\asl_matrix_q3s3_hybrid_xpu.json"
    }
)

foreach ($condition in $conditions) {
    $run = Join-Path $runs $condition.Name
    $summary = Join-Path $run "training_report.json"
    if (-not (Test-Path -LiteralPath $summary)) {
        & $Python -u -m ccpu paper1 train-qwen-asl-patch `
            --config $condition.Config `
            --train (Join-Path $data "train.jsonl") `
            --dev (Join-Path $data "dev.jsonl") `
            --output-dir $run
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $($condition.Name) with exit code $LASTEXITCODE"
        }
    }

    $evaluation = Join-Path (Join-Path $evals $condition.Name) "autonomous"
    $evaluationSummary = Join-Path $evaluation "summary.json"
    if (-not (Test-Path -LiteralPath $evaluationSummary)) {
        & $Python -u -m ccpu paper1 evaluate-qwen-asl-patch `
            --config $condition.Config `
            --state (Join-Path $run "trainable_patch_state.safetensors") `
            --eval (Join-Path $source "test.jsonl") `
            --train-split (Join-Path $source "train.jsonl") `
            --output-dir $evaluation `
            --checkpoint-every 1
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for $($condition.Name) with exit code $LASTEXITCODE"
        }
    }
}
