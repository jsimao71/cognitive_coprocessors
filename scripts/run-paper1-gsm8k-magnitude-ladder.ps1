param(
    [string]$RepositoryRoot,
    [ValidateSet("xpu", "cuda")]
    [string]$Device = "xpu",
    [string]$PythonPath,
    [string]$Conditions = "b1,b1l,asl_seed11,asl_seed23,asl_seed37"
)

$ErrorActionPreference = "Stop"
$RepoRoot = if ($RepositoryRoot) {
    (Resolve-Path $RepositoryRoot).Path
}
else {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$Python = if ($PythonPath) { $PythonPath } elseif ($Device -eq "cuda") {
    Join-Path $env:USERPROFILE ".venvs\ccpu-cuda\Scripts\python.exe"
}
else {
    Join-Path $env:USERPROFILE ".venvs\modal-llm-xpu\Scripts\python.exe"
}
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$Selected = @($Conditions.Split(",").ForEach({ $_.Trim() }).Where({ $_ }))
$Allowed = @("b1", "b1l", "asl_seed11", "asl_seed23", "asl_seed37")
foreach ($condition in $Selected) {
    if ($condition -notin $Allowed) { throw "Unknown condition: $condition" }
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Python is unavailable: $Python" }

$Scale = Join-Path $RepoRoot "artifacts\paper1\gsm8k_scale_v1"
$Root = Join-Path $Scale "magnitude_ladder_v1"
$Data = Join-Path $Root "data"
$EvalRoot = Join-Path $Root "eval"
$OfficialEval = Join-Path $Scale "official_test_v1\confirmatory.jsonl"
$LargeEval = Join-Path $Scale "large_number_v1\data\large.jsonl"
$OriginalAsl = Join-Path $Scale "official_eval_v1\confirmatory"
$LargeAsl = Join-Path $Scale "large_number_v1\eval\asl"
$OriginalDirect = Join-Path $Scale "matched_direct_v1\original"
$LargeDirect = Join-Path $Scale "large_number_v1\eval\direct"
$Factors = @(1, 100, 1000, 10000, 1000000)
$NewFactors = @(100, 10000, 1000000)
$AslConfig = Join-Path $RepoRoot "configs\paper1\asl_pilot_qwen_base_${Device}.json"
$DirectConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_qwen_${Device}.json"
$DirectLongConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_long_qwen_${Device}.json"

$Adapters = @{
    asl_seed11 = Join-Path $Scale "u2000_e4500\qwen_run\adapter"
    asl_seed23 = Join-Path $Scale "u2000_e4500\replications\seed23_xpu\qwen_run\adapter"
    asl_seed37 = Join-Path $Scale "u2000_e4500\replications\seed37_xpu\qwen_run\adapter"
}
$AdapterIds = @{
    asl_seed11 = "Qwen3-0.6B-G1-GSM8K-U2000-E4500-F0-L0-r8-seed11"
    asl_seed23 = "Qwen3-0.6B-G1-GSM8K-U2000-E4500-F0-L0-r8-seed23"
    asl_seed37 = "Qwen3-0.6B-G1-GSM8K-U2000-E4500-F0-L0-r8-seed37"
}
$OriginalPredictions = @{
    b1 = Join-Path $OriginalDirect "direct_reasoning\predictions.jsonl"
    b1l = Join-Path $OriginalDirect "direct_reasoning_long\predictions.jsonl"
    asl_seed11 = Join-Path $OriginalAsl "seed11_xpu\predictions.jsonl"
    asl_seed23 = Join-Path $OriginalAsl "seed23_xpu\predictions.jsonl"
    asl_seed37 = Join-Path $OriginalAsl "seed37_xpu\predictions.jsonl"
}
$LargePredictions = @{
    b1 = Join-Path $LargeDirect "direct_reasoning\predictions.jsonl"
    b1l = Join-Path $LargeDirect "direct_reasoning_long\predictions.jsonl"
    asl_seed11 = Join-Path $LargeAsl "seed11_xpu\predictions.jsonl"
    asl_seed23 = Join-Path $LargeAsl "seed23_xpu\predictions.jsonl"
    asl_seed37 = Join-Path $LargeAsl "seed37_xpu\predictions.jsonl"
}

function Invoke-E3 {
    param([string[]]$Arguments)
    & $Python -u -m ccpu.paper1.e3 @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Paper 1 E3 command failed: $Arguments" }
}

function Project-Cached {
    param([string]$Condition, [int]$Factor, [string]$SourceEval, [string]$SourcePredictions)
    $output = Join-Path $EvalRoot "$Condition\factor_$Factor"
    if (Test-Path -LiteralPath (Join-Path $output "projection_manifest.json")) { return }
    Invoke-E3 @(
        "project-gsm8k-magnitude-predictions",
        "--source-eval", $SourceEval,
        "--target-eval", (Join-Path $Data "factor_$Factor.jsonl"),
        "--source-predictions", $SourcePredictions,
        "--output-dir", $output
    )
}

function Run-Asl {
    param([string]$Condition, [int]$Factor)
    $output = Join-Path $EvalRoot "$Condition\factor_$Factor"
    if (Test-Path -LiteralPath (Join-Path $output "summary.json")) { return }
    Invoke-E3 @(
        "run-gsm8k-official-shard",
        "--eval", (Join-Path $Data "factor_$Factor.jsonl"),
        "--config", $AslConfig,
        "--adapter-path", $Adapters[$Condition],
        "--adapter-id", $AdapterIds[$Condition],
        "--output-dir", $output,
        "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
    )
}

function Run-Direct {
    param([string]$Condition, [int]$Factor)
    $output = Join-Path $EvalRoot "$Condition\factor_$Factor"
    if (Test-Path -LiteralPath (Join-Path $output "summary.json")) { return }
    if ($Condition -eq "b1l") {
        $source = Join-Path $EvalRoot "b1\factor_$Factor\predictions.jsonl"
        if (-not (Test-Path -LiteralPath (Join-Path $output "long_budget_resume_manifest.json"))) {
            Invoke-E3 @(
                "prepare-gsm8k-long-budget-resume", "--source-predictions", $source,
                "--output-dir", $output, "--source-ceiling", "1024", "--target-ceiling", "2048"
            )
        }
    }
    $config = if ($Condition -eq "b1l") { $DirectLongConfig } else { $DirectConfig }
    Invoke-E3 @(
        "run-gsm8k-direct-shard",
        "--eval", (Join-Path $Data "factor_$Factor.jsonl"),
        "--config", $config, "--condition", "direct_reasoning",
        "--output-dir", $output,
        "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
    )
}

Push-Location $RepoRoot
try {
    foreach ($condition in $Selected) {
        Project-Cached $condition 1 $OfficialEval $OriginalPredictions[$condition]
        Project-Cached $condition 1000 $LargeEval $LargePredictions[$condition]
        foreach ($factor in $NewFactors) {
            if ($condition.StartsWith("asl_")) { Run-Asl $condition $factor }
            else { Run-Direct $condition $factor }
        }
        if ($condition.StartsWith("asl_")) {
            foreach ($factor in $Factors.Where({ $_ -ne 1 })) {
                $analysis = Join-Path $Root "analysis\failure_audit_v1\$condition\factor_$factor"
                Invoke-E3 @(
                    "analyze-gsm8k-magnitude-failures",
                    "--eval", (Join-Path $Data "factor_$factor.jsonl"),
                    "--original-predictions", (Join-Path $EvalRoot "$condition\factor_1\predictions.jsonl"),
                    "--transformed-predictions", (Join-Path $EvalRoot "$condition\factor_$factor\predictions.jsonl"),
                    "--output-dir", $analysis
                )
            }
        }
    }
    Write-Host "COMPLETE magnitude conditions: $($Selected -join ',')"
}
finally {
    Pop-Location
}
