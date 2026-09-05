param(
    [string]$RepositoryRoot,
    [string]$PythonPath
)

$ErrorActionPreference = "Stop"
$RepoRoot = if ($RepositoryRoot) {
    (Resolve-Path $RepositoryRoot).Path
}
else {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$Python = if ($PythonPath) { $PythonPath } else {
    Join-Path $env:USERPROFILE ".venvs\ccpu-cuda\Scripts\python.exe"
}
$env:PYTHONPATH = Join-Path $RepoRoot "src"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "CUDA Python is unavailable: $Python"
}
& $Python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA validation failed" }

$Scale = Join-Path $RepoRoot "artifacts\paper1\gsm8k_scale_v1"
$Data = Join-Path $Scale "u2000_e4500"
$Dev = Join-Path $Scale "g1_f0_4500\eval\dev.jsonl"
$OfficialEval = Join-Path $Scale "official_test_v1\confirmatory.jsonl"
$LargeEval = Join-Path $Scale "large_number_v1\data\large.jsonl"
$Root = Join-Path $Scale "qwen1_7b_u2000_e4500\seed99173_cuda"
$Run = Join-Path $Root "qwen_run"
$AslOriginal = Join-Path $Root "asl\original"
$AslLarge = Join-Path $Root "asl\large"
$DirectOriginal = Join-Path $Root "direct\original\reasoning"
$DirectLarge = Join-Path $Root "direct\large\reasoning"
$DirectLongOriginal = Join-Path $Root "direct\original\reasoning_long"
$DirectLongLarge = Join-Path $Root "direct\large\reasoning_long"
$Analysis = Join-Path $Root "analysis\matched_contribution_v1.json"
$AnalysisLong = Join-Path $Root "analysis\matched_contribution_b1l_v1.json"
$TrainConfig = Join-Path $RepoRoot "configs\paper1\e3_g1_gsm8k_u2000_e4500_f0_l0_qwen1_7b_lora_cuda.json"
$AslConfig = Join-Path $RepoRoot "configs\paper1\asl_pilot_qwen1_7b_base_cuda.json"
$DirectConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_qwen1_7b_cuda.json"
$DirectLongConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_long_qwen1_7b_cuda.json"
$AdapterId = "Qwen3-1.7B-G1-GSM8K-U2000-E4500-F0-L0-r8-init99173-cuda"

foreach ($required in @(
    (Join-Path $Data "train.jsonl"), $Dev, $OfficialEval, $LargeEval,
    $TrainConfig, $AslConfig, $DirectConfig, $DirectLongConfig
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required input: $required" }
}

function Invoke-Step {
    param([string]$Name, [string]$CompletionPath, [string[]]$Arguments)
    if (Test-Path -LiteralPath $CompletionPath) {
        Write-Host "SKIP $Name (complete)"
        return
    }
    Write-Host "START $Name"
    & $Python -u -m ccpu @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $CompletionPath)) {
        throw "$Name exited without completion artifact: $CompletionPath"
    }
    Write-Host "COMPLETE $Name"
}

function Invoke-E3Step {
    param([string]$Name, [string]$CompletionPath, [string[]]$Arguments)
    if (Test-Path -LiteralPath $CompletionPath) {
        Write-Host "SKIP $Name (complete)"
        return
    }
    Write-Host "START $Name"
    & $Python -u -m ccpu.paper1.e3 @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $CompletionPath)) {
        throw "$Name exited without completion artifact: $CompletionPath"
    }
    Write-Host "COMPLETE $Name"
}

Push-Location $RepoRoot
try {
    Invoke-Step -Name "Qwen3-1.7B U2000 training" `
        -CompletionPath (Join-Path $Run "training_report.json") `
        -Arguments @(
            "paper1", "train-lora", "--config", $TrainConfig,
            "--model", "Qwen/Qwen3-1.7B", "--train", (Join-Path $Data "train.jsonl"),
            "--dev", $Dev, "--output-dir", $Run
        )

    foreach ($item in @(
        @{ Name = "ASL-original"; Eval = $OfficialEval; Dir = $AslOriginal },
        @{ Name = "ASL-large"; Eval = $LargeEval; Dir = $AslLarge }
    )) {
        Invoke-E3Step -Name $item.Name -CompletionPath (Join-Path $item.Dir "summary.json") `
            -Arguments @(
                "run-gsm8k-official-shard", "--eval", $item.Eval,
                "--config", $AslConfig, "--adapter-path", (Join-Path $Run "adapter"),
                "--adapter-id", $AdapterId, "--output-dir", $item.Dir,
                "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
            )
    }

    foreach ($item in @(
        @{ Name = "direct-original"; Eval = $OfficialEval; Dir = $DirectOriginal },
        @{ Name = "direct-large"; Eval = $LargeEval; Dir = $DirectLarge }
    )) {
        Invoke-E3Step -Name $item.Name -CompletionPath (Join-Path $item.Dir "summary.json") `
            -Arguments @(
                "run-gsm8k-direct-shard", "--eval", $item.Eval,
                "--config", $DirectConfig, "--condition", "direct_reasoning",
                "--output-dir", $item.Dir, "--shard-index", "0", "--shard-count", "1",
                "--checkpoint-every", "1"
            )
    }

    Invoke-E3Step -Name "Qwen3-1.7B matched analysis" -CompletionPath $Analysis `
        -Arguments @(
            "analyze-gsm8k-contribution", "--original-eval", $OfficialEval,
            "--large-eval", $LargeEval,
            "--original-direct", "direct_reasoning=$(Join-Path $DirectOriginal 'predictions.jsonl')",
            "--original-asl", "qwen1_7b_seed99173=$(Join-Path $AslOriginal 'predictions.jsonl')",
            "--large-direct", "direct_reasoning=$(Join-Path $DirectLarge 'predictions.jsonl')",
            "--large-asl", "qwen1_7b_seed99173=$(Join-Path $AslLarge 'predictions.jsonl')",
            "--output", $Analysis
        )

    $sourceRows = @(Get-Content -LiteralPath (Join-Path $DirectOriginal "predictions.jsonl") |
        ForEach-Object { $_ | ConvertFrom-Json })
    $ceilingHits = @($sourceRows.Where({ [int]$_.generated_tokens -ge 1024 })).Count
    if ($ceilingHits -ge 25) {
        foreach ($item in @(
            @{ Name = "direct-long-original"; Eval = $OfficialEval; Source = $DirectOriginal; Dir = $DirectLongOriginal },
            @{ Name = "direct-long-large"; Eval = $LargeEval; Source = $DirectLarge; Dir = $DirectLongLarge }
        )) {
            if (-not (Test-Path -LiteralPath (Join-Path $item.Dir "long_budget_resume_manifest.json"))) {
                Invoke-E3Step -Name "$($item.Name)-prepare" `
                    -CompletionPath (Join-Path $item.Dir "long_budget_resume_manifest.json") `
                    -Arguments @(
                        "prepare-gsm8k-long-budget-resume", "--source-predictions",
                        (Join-Path $item.Source "predictions.jsonl"), "--output-dir", $item.Dir,
                        "--source-ceiling", "1024", "--target-ceiling", "2048"
                    )
            }
            Invoke-E3Step -Name $item.Name -CompletionPath (Join-Path $item.Dir "summary.json") `
                -Arguments @(
                    "run-gsm8k-direct-shard", "--eval", $item.Eval,
                    "--config", $DirectLongConfig, "--condition", "direct_reasoning",
                    "--output-dir", $item.Dir, "--shard-index", "0", "--shard-count", "1",
                    "--checkpoint-every", "1"
                )
        }
        Invoke-E3Step -Name "Qwen3-1.7B long-budget matched analysis" `
            -CompletionPath $AnalysisLong `
            -Arguments @(
                "analyze-gsm8k-contribution", "--original-eval", $OfficialEval,
                "--large-eval", $LargeEval,
                "--original-direct", "direct_reasoning_long=$(Join-Path $DirectLongOriginal 'predictions.jsonl')",
                "--original-asl", "qwen1_7b_seed99173=$(Join-Path $AslOriginal 'predictions.jsonl')",
                "--large-direct", "direct_reasoning_long=$(Join-Path $DirectLongLarge 'predictions.jsonl')",
                "--large-asl", "qwen1_7b_seed99173=$(Join-Path $AslLarge 'predictions.jsonl')",
                "--output", $AnalysisLong
            )
    }
    else {
        Write-Host "SKIP 1.7B long-budget sensitivity: ceiling hits=$ceilingHits/250"
    }
}
finally {
    Pop-Location
}
