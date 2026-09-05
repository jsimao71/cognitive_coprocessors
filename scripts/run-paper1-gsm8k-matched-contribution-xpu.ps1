param(
    [switch]$WaitForOfficial,
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
$OfficialEval = Join-Path $Scale "official_test_v1\confirmatory.jsonl"
$LargeEval = Join-Path $Scale "large_number_v1\data\large.jsonl"
$OfficialRoot = Join-Path $Scale "official_eval_v1\confirmatory"
$DirectRoot = Join-Path $Scale "matched_direct_v1\original"
$LargeRoot = Join-Path $Scale "large_number_v1\eval"
$Analysis = Join-Path $Scale "analysis\matched_contribution_v1.json"
$BaseConfig = Join-Path $RepoRoot "configs\paper1\asl_pilot_qwen_base_xpu.json"
$ConciseConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_concise_qwen_xpu.json"
$ReasoningConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_qwen_xpu.json"
$U2000 = Join-Path $Scale "u2000_e4500"

$AslRuns = @(
    @{
        Label = "seed11"
        Adapter = Join-Path $U2000 "qwen_run\adapter"
        AdapterId = "Qwen3-0.6B-G1-GSM8K-U2000-E4500-F0-L0-r8-seed11"
        Original = Join-Path $OfficialRoot "seed11_xpu\predictions.jsonl"
        LargeDir = Join-Path $LargeRoot "asl\seed11_xpu"
    },
    @{
        Label = "seed23"
        Adapter = Join-Path $U2000 "replications\seed23_xpu\qwen_run\adapter"
        AdapterId = "Qwen3-0.6B-G1-GSM8K-U2000-E4500-F0-L0-r8-seed23-xpu"
        Original = Join-Path $OfficialRoot "seed23_xpu\predictions.jsonl"
        LargeDir = Join-Path $LargeRoot "asl\seed23_xpu"
    },
    @{
        Label = "seed37"
        Adapter = Join-Path $U2000 "replications\seed37_xpu\qwen_run\adapter"
        AdapterId = "Qwen3-0.6B-G1-GSM8K-U2000-E4500-F0-L0-r8-seed37"
        Original = Join-Path $OfficialRoot "seed37_xpu\predictions.jsonl"
        LargeDir = Join-Path $LargeRoot "asl\seed37_xpu"
    }
)

function Invoke-ResumableStep {
    param(
        [string]$Name,
        [string]$CompletionPath,
        [string[]]$Arguments
    )
    if (Test-Path -LiteralPath $CompletionPath) {
        Write-Host "SKIP $Name (complete)"
        return
    }
    Write-Host "START $Name"
    & $Python -u -m ccpu.paper1.e3 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $CompletionPath)) {
        throw "$Name exited without completion artifact: $CompletionPath"
    }
    Write-Host "COMPLETE $Name"
}

$RequiredOfficial = $AslRuns | ForEach-Object {
    Join-Path (Split-Path $_.Original -Parent) "summary.json"
}
if ($WaitForOfficial) {
    while ($RequiredOfficial.Where({ -not (Test-Path -LiteralPath $_) }).Count -gt 0) {
        $remaining = $RequiredOfficial.Where({ -not (Test-Path -LiteralPath $_) }).Count
        Write-Host "WAIT official ASL summaries remaining=$remaining"
        Start-Sleep -Seconds 30
    }
}
foreach ($path in $RequiredOfficial) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Original ASL evaluation is incomplete: $path"
    }
}

Push-Location $RepoRoot
try {
    foreach ($condition in @(
        @{ Name = "direct_concise"; Config = $ConciseConfig },
        @{ Name = "direct_reasoning"; Config = $ReasoningConfig }
    )) {
        foreach ($dataset in @(
            @{ Name = "original"; Eval = $OfficialEval; Dir = Join-Path $DirectRoot $condition.Name },
            @{ Name = "large"; Eval = $LargeEval; Dir = Join-Path $LargeRoot "direct\$($condition.Name)" }
        )) {
            Invoke-ResumableStep `
                -Name "$($condition.Name)-$($dataset.Name)" `
                -CompletionPath (Join-Path $dataset.Dir "summary.json") `
                -Arguments @(
                    "run-gsm8k-direct-shard",
                    "--eval", $dataset.Eval,
                    "--config", $condition.Config,
                    "--condition", $condition.Name,
                    "--output-dir", $dataset.Dir,
                    "--shard-index", "0",
                    "--shard-count", "1",
                    "--checkpoint-every", "1"
                )
        }

        if ($condition.Name -eq "direct_concise") {
            foreach ($run in $AslRuns) {
                Invoke-ResumableStep `
                    -Name "ASL-$($run.Label)-large" `
                    -CompletionPath (Join-Path $run.LargeDir "summary.json") `
                    -Arguments @(
                        "run-gsm8k-official-shard",
                        "--eval", $LargeEval,
                        "--config", $BaseConfig,
                        "--adapter-path", $run.Adapter,
                        "--adapter-id", $run.AdapterId,
                        "--output-dir", $run.LargeDir,
                        "--shard-index", "0",
                        "--shard-count", "1",
                        "--checkpoint-every", "1"
                    )
            }
        }
    }

    $AnalysisArguments = @(
        "analyze-gsm8k-contribution",
        "--original-eval", $OfficialEval,
        "--large-eval", $LargeEval,
        "--original-direct", "direct_concise=$(Join-Path $DirectRoot 'direct_concise\predictions.jsonl')",
        "--original-direct", "direct_reasoning=$(Join-Path $DirectRoot 'direct_reasoning\predictions.jsonl')",
        "--large-direct", "direct_concise=$(Join-Path $LargeRoot 'direct\direct_concise\predictions.jsonl')",
        "--large-direct", "direct_reasoning=$(Join-Path $LargeRoot 'direct\direct_reasoning\predictions.jsonl')",
        "--output", $Analysis
    )
    foreach ($run in $AslRuns) {
        $AnalysisArguments += @("--original-asl", "$($run.Label)=$($run.Original)")
        $AnalysisArguments += @(
            "--large-asl",
            "$($run.Label)=$(Join-Path $run.LargeDir 'predictions.jsonl')"
        )
    }
    Invoke-ResumableStep `
        -Name "matched-contribution-analysis" `
        -CompletionPath $Analysis `
        -Arguments $AnalysisArguments
}
finally {
    Pop-Location
}
