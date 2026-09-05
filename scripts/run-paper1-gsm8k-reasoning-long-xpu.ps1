param(
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
$Config = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_long_qwen_xpu.json"
$PrimaryOriginal = Join-Path $DirectRoot "direct_reasoning"
$PrimaryLarge = Join-Path $LargeRoot "direct\direct_reasoning"
$LongOriginal = Join-Path $DirectRoot "direct_reasoning_long"
$LongLarge = Join-Path $LargeRoot "direct\direct_reasoning_long"
$Analysis = Join-Path $Scale "analysis\matched_contribution_b1l_v1.json"

foreach ($summary in @(
    (Join-Path $PrimaryOriginal "summary.json"),
    (Join-Path $PrimaryLarge "summary.json")
)) {
    while (-not (Test-Path -LiteralPath $summary)) {
        Write-Host "WAIT primary reasoning summary: $summary"
        Start-Sleep -Seconds 30
    }
}

$SourceCeiling = 1024
$TargetCeiling = 2048
$sourceRows = @(Get-Content -LiteralPath (Join-Path $PrimaryOriginal "predictions.jsonl") |
    ForEach-Object { $_ | ConvertFrom-Json })
$ceilingHits = @($sourceRows.Where({ [int]$_.generated_tokens -ge $SourceCeiling })).Count
if ($ceilingHits -lt 25) {
    Write-Host "SKIP B1L: original ceiling hits $ceilingHits/250 are below 10% gate"
    exit 0
}

function Invoke-LongBudgetRun {
    param(
        [string]$Name,
        [string]$Eval,
        [string]$SourceDir,
        [string]$OutputDir
    )
    if (Test-Path -LiteralPath (Join-Path $OutputDir "summary.json")) {
        Write-Host "SKIP $Name (complete)"
        return
    }
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDir "long_budget_resume_manifest.json"))) {
        & $Python -u -m ccpu.paper1.e3 prepare-gsm8k-long-budget-resume `
            --source-predictions (Join-Path $SourceDir "predictions.jsonl") `
            --output-dir $OutputDir `
            --source-ceiling $SourceCeiling `
            --target-ceiling $TargetCeiling
        if ($LASTEXITCODE -ne 0) { throw "$Name preparation failed" }
    }
    & $Python -u -m ccpu.paper1.e3 run-gsm8k-direct-shard `
        --eval $Eval `
        --config $Config `
        --condition direct_reasoning `
        --output-dir $OutputDir `
        --shard-index 0 `
        --shard-count 1 `
        --checkpoint-every 1
    if ($LASTEXITCODE -ne 0) { throw "$Name generation failed" }
}

Push-Location $RepoRoot
try {
    Invoke-LongBudgetRun -Name "B1L-original" -Eval $OfficialEval `
        -SourceDir $PrimaryOriginal -OutputDir $LongOriginal
    Invoke-LongBudgetRun -Name "B1L-large" -Eval $LargeEval `
        -SourceDir $PrimaryLarge -OutputDir $LongLarge

    $arguments = @(
        "analyze-gsm8k-contribution",
        "--original-eval", $OfficialEval,
        "--large-eval", $LargeEval,
        "--original-direct", "direct_concise=$(Join-Path $DirectRoot 'direct_concise\predictions.jsonl')",
        "--original-direct", "direct_reasoning=$(Join-Path $PrimaryOriginal 'predictions.jsonl')",
        "--original-direct", "direct_reasoning_long=$(Join-Path $LongOriginal 'predictions.jsonl')",
        "--large-direct", "direct_concise=$(Join-Path $LargeRoot 'direct\direct_concise\predictions.jsonl')",
        "--large-direct", "direct_reasoning=$(Join-Path $PrimaryLarge 'predictions.jsonl')",
        "--large-direct", "direct_reasoning_long=$(Join-Path $LongLarge 'predictions.jsonl')",
        "--output", $Analysis
    )
    foreach ($seed in @("seed11", "seed23", "seed37")) {
        $suffix = if ($seed -eq "seed11") { "seed11_xpu" } else { "${seed}_xpu" }
        $arguments += @(
            "--original-asl", "$seed=$(Join-Path $OfficialRoot "$suffix\predictions.jsonl")",
            "--large-asl", "$seed=$(Join-Path $LargeRoot "asl\$suffix\predictions.jsonl")"
        )
    }
    & $Python -u -m ccpu.paper1.e3 @arguments
    if ($LASTEXITCODE -ne 0) { throw "B1L contribution analysis failed" }
    Write-Host "COMPLETE B1L matched contribution analysis: $Analysis"
}
finally {
    Pop-Location
}
