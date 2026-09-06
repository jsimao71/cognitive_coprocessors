param(
    [switch]$WaitForAUG1,
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
$Scale = Join-Path $RepoRoot "artifacts\paper1\gsm8k_scale_v1"
$AugRoot = Join-Path $RepoRoot "artifacts\paper1\gsm8k_semantic_augmentation_v1"
$Aug1 = Join-Path $AugRoot "aug1_relation_paraphrase"
$Stage = Join-Path $AugRoot "aug2_entity_rename"
$SelectionRoot = Join-Path $AugRoot "selection_gate_v1"
$SelectionEval = Join-Path $SelectionRoot "gate.jsonl"
$Aug1Gate = Join-Path $Aug1 "analysis\greedy_gate.json"
$Increment = Join-Path $Stage "increment.jsonl"
$Config = Join-Path $RepoRoot "configs\paper1\e3_gsm8k_sem_aug2_entity_qwen_lora_xpu.json"
$ModelConfig = Join-Path $RepoRoot "configs\paper1\asl_pilot_qwen_base_xpu.json"
$Dev = Join-Path $Scale "g1_f0_4500\eval\dev.jsonl"
$HistoricalEval = Join-Path $Scale "g1_f0_4500\eval\test.jsonl"
$Run = Join-Path $Stage "qwen_run"
$Historical = Join-Path $Stage "eval\historical"
$CandidateSelection = Join-Path $Stage "eval\selection"
$Analysis = Join-Path $Stage "analysis\greedy_gate.json"
$AdapterId = "Qwen3-0.6B-greedy-plus-AUG2-entity-F0-L0-r8-init99173"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "XPU Python is unavailable: $Python"
}
if ($WaitForAUG1) {
    while (-not (Test-Path -LiteralPath $Aug1Gate)) {
        Write-Host "WAIT AUG1 greedy decision"
        Start-Sleep -Seconds 30
    }
}
if (-not (Test-Path -LiteralPath $Aug1Gate)) {
    throw "AUG1 greedy decision is unavailable: $Aug1Gate"
}

$Aug1Decision = Get-Content -LiteralPath $Aug1Gate -Raw | ConvertFrom-Json
if ([bool]$Aug1Decision.decision.accepted) {
    $ParentAdapter = Join-Path $Aug1 "qwen_run\adapter"
    $BaselineSelection = Join-Path $Aug1 "eval\selection"
    $BaselineHistorical = Join-Path $Aug1 "eval\historical"
    $ParentLabel = "AUG1"
}
else {
    $ParentAdapter = Join-Path $Scale "u2000_e4500\qwen_run\adapter"
    $BaselineSelection = Join-Path $SelectionRoot "eval\u2000_seed11"
    $BaselineHistorical = Join-Path $Scale "u2000_e4500\historical_test_eval"
    $ParentLabel = "U2000"
}

foreach ($path in @(
    $Increment,
    $Config,
    $ParentAdapter,
    (Join-Path $BaselineSelection "predictions.jsonl"),
    (Join-Path $BaselineHistorical "summary.json")
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required AUG2 input is unavailable: $path"
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
    Write-Host "AUG2 parent checkpoint: $ParentLabel"
    Invoke-ResumableStep `
        -Name "AUG2 warm-start training" `
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
        -Name "AUG2 historical diagnostic" `
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

    Invoke-ResumableStep `
        -Name "AUG2 selection evaluation" `
        -CompletionPath (Join-Path $CandidateSelection "summary.json") `
        -Arguments @(
            "run-gsm8k-official-shard",
            "--eval", $SelectionEval,
            "--config", $ModelConfig,
            "--adapter-path", (Join-Path $Run "adapter"),
            "--adapter-id", $AdapterId,
            "--output-dir", $CandidateSelection,
            "--shard-index", "0",
            "--shard-count", "1",
            "--checkpoint-every", "1"
        )

    Invoke-ResumableStep `
        -Name "AUG2 greedy gate" `
        -CompletionPath $Analysis `
        -Arguments @(
            "analyze-gsm8k-augmentation",
            "--stage", "AUG2-synchronized-entity-rename",
            "--selection-eval", $SelectionEval,
            "--historical-eval", $HistoricalEval,
            "--baseline-selection", (Join-Path $BaselineSelection "predictions.jsonl"),
            "--candidate-selection", (Join-Path $CandidateSelection "predictions.jsonl"),
            "--baseline-historical", (Join-Path $BaselineHistorical "predictions.jsonl"),
            "--candidate-historical", (Join-Path $Historical "predictions.jsonl"),
            "--baseline-historical-summary", (Join-Path $BaselineHistorical "summary.json"),
            "--candidate-historical-summary", (Join-Path $Historical "summary.json"),
            "--output", $Analysis
        )
}
finally {
    Pop-Location
}
