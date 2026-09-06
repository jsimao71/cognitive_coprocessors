param(
    [switch]$WaitForB1L,
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$RepoRoot = if ($RepositoryRoot) {
    (Resolve-Path $RepositoryRoot).Path
}
else {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$Python = Join-Path $env:USERPROFILE ".venvs\modal-llm-xpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $RepoRoot "src"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "XPU Python is unavailable: $Python"
}

$Scale = Join-Path $RepoRoot "artifacts\paper1\gsm8k_scale_v1"
$Stage = Join-Path $RepoRoot "artifacts\paper1\gsm8k_semantic_augmentation_v1\aug1_relation_paraphrase"
$Increment = Join-Path $Stage "increment.jsonl"
$Config = Join-Path $RepoRoot "configs\paper1\e3_gsm8k_sem_aug1_relation_qwen_lora_xpu.json"
$ModelConfig = Join-Path $RepoRoot "configs\paper1\asl_pilot_qwen_base_xpu.json"
$Parent = Join-Path $Scale "u2000_e4500"
$ParentAdapter = Join-Path $Parent "qwen_run\adapter"
$HistoricalEval = Join-Path $Scale "g1_f0_4500\eval\test.jsonl"
$Dev = Join-Path $Scale "g1_f0_4500\eval\dev.jsonl"
$OfficialEval = Join-Path $Scale "official_test_v1\confirmatory.jsonl"
$LargeEval = Join-Path $Scale "large_number_v1\data\large.jsonl"
$Run = Join-Path $Stage "qwen_run"
$Historical = Join-Path $Stage "eval\historical"
$Official = Join-Path $Stage "eval\official"
$Large = Join-Path $Stage "eval\large"
$Analysis = Join-Path $Stage "analysis\greedy_gate.json"
$AdapterId = "Qwen3-0.6B-U2000-plus-AUG1-relation-F0-L0-r8-init99173"
$B1L = Join-Path $Scale "analysis\matched_contribution_b1l_v1.json"

foreach ($path in @($Increment, $Config, $ParentAdapter, $HistoricalEval, $Dev, $OfficialEval, $LargeEval)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required campaign input is unavailable: $path"
    }
}

if ($WaitForB1L) {
    while (-not (Test-Path -LiteralPath $B1L)) {
        Write-Host "WAIT local XPU: B1L is still active"
        Start-Sleep -Seconds 30
    }
}

function Invoke-ResumableStep {
    param(
        [string]$Name,
        [string]$CompletionPath,
        [string[]]$Arguments,
        [switch]$MainCli
    )
    if (Test-Path -LiteralPath $CompletionPath) {
        Write-Host "SKIP $Name (complete)"
        return
    }
    Write-Host "START $Name"
    if ($MainCli) {
        & $Python -u -m ccpu @Arguments
    }
    else {
        & $Python -u -m ccpu.paper1.e3 @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $CompletionPath)) {
        throw "$Name exited without completion artifact: $CompletionPath"
    }
    Write-Host "COMPLETE $Name"
}

Push-Location $RepoRoot
try {
    Invoke-ResumableStep `
        -Name "AUG1 warm-start training" `
        -CompletionPath (Join-Path $Run "training_report.json") `
        -MainCli `
        -Arguments @(
            "paper1", "train-lora",
            "--config", $Config,
            "--model", "Qwen/Qwen3-0.6B",
            "--train", $Increment,
            "--dev", $Dev,
            "--initial-adapter-path", $ParentAdapter,
            "--output-dir", $Run
        )

    Invoke-ResumableStep `
        -Name "AUG1 historical diagnostic" `
        -CompletionPath (Join-Path $Historical "summary.json") `
        -MainCli `
        -Arguments @(
            "paper1", "run-asl-pilot",
            "--eval", $HistoricalEval,
            "--train-split", $Increment,
            "--config", $ModelConfig,
            "--adapter-path", (Join-Path $Run "adapter"),
            "--adapter-id", $AdapterId,
            "--condition", "lora",
            "--shots", "0",
            "--output-dir", $Historical,
            "--checkpoint-every", "1"
        )

    foreach ($evaluation in @(
        @{ Name = "official"; Eval = $OfficialEval; Output = $Official },
        @{ Name = "large-number"; Eval = $LargeEval; Output = $Large }
    )) {
        Invoke-ResumableStep `
            -Name "AUG1 $($evaluation.Name) evaluation" `
            -CompletionPath (Join-Path $evaluation.Output "summary.json") `
            -Arguments @(
                "run-gsm8k-official-shard",
                "--eval", $evaluation.Eval,
                "--config", $ModelConfig,
                "--adapter-path", (Join-Path $Run "adapter"),
                "--adapter-id", $AdapterId,
                "--output-dir", $evaluation.Output,
                "--shard-index", "0",
                "--shard-count", "1",
                "--checkpoint-every", "1"
            )
    }

    Invoke-ResumableStep `
        -Name "AUG1 greedy gate" `
        -CompletionPath $Analysis `
        -Arguments @(
            "analyze-gsm8k-augmentation",
            "--stage", "AUG1-relation-paraphrase",
            "--official-eval", $OfficialEval,
            "--historical-eval", $HistoricalEval,
            "--large-eval", $LargeEval,
            "--baseline-official", (Join-Path $Scale "official_eval_v1\confirmatory\seed11_xpu\predictions.jsonl"),
            "--candidate-official", (Join-Path $Official "predictions.jsonl"),
            "--baseline-historical", (Join-Path $Parent "historical_test_eval\predictions.jsonl"),
            "--candidate-historical", (Join-Path $Historical "predictions.jsonl"),
            "--baseline-historical-summary", (Join-Path $Parent "historical_test_eval\summary.json"),
            "--candidate-historical-summary", (Join-Path $Historical "summary.json"),
            "--baseline-large", (Join-Path $Scale "large_number_v1\eval\asl\seed11_xpu\predictions.jsonl"),
            "--candidate-large", (Join-Path $Large "predictions.jsonl"),
            "--output", $Analysis
        )
}
finally {
    Pop-Location
}
