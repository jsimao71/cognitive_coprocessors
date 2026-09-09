param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][ValidateSet("cpu", "xpu", "cuda")][string]$Device,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][ValidateSet(17011, 17023, 17037)][int]$JitterSeed,
    [ValidateSet("ASL", "Direct", "All")][string]$Phase = "All",
    [ValidateSet(1, 1000)][int[]]$Factor = @(1, 1000),
    [ValidateSet("O0", "O1", "O5", "O6")][string[]]$Level = @("O0", "O1", "O5", "O6"),
    [string]$BaseAdapterPath,
    [string]$OperatorAdapterPath
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$panel = Join-Path $root "artifacts\paper1\operator_complexity_v2\gsm8k_matched_seed99173"
$matrix = Join-Path $root "artifacts\paper1\operator_jitter_matrix_v1\gsm8k_matched_seed99173"
$runRoot = Join-Path $root "artifacts\paper1\operator_jitter_matrix_v1\runs\seed_$JitterSeed"
$directConfig = Join-Path $root "configs\paper1\gsm8k_direct_reasoning_qwen_$Device.json"
$aslConfig = Join-Path $root "configs\paper1\operator_complexity\natural_v2_qwen06_$Device.json"
if (-not $BaseAdapterPath) {
    $BaseAdapterPath = Join-Path $root "artifacts\paper1\gsm8k_scale_v1\u2000_e4500\qwen_run\adapter"
}
if (-not $OperatorAdapterPath) {
    $OperatorAdapterPath = Join-Path $panel "runs\operator_adapter_r8_seed99173_cuda\adapter"
}
$env:PYTHONPATH = Join-Path $root "src"

function Invoke-Checked([string]$Name, [string]$Completion, [string[]]$Arguments) {
    if (Test-Path -LiteralPath $Completion) {
        Write-Host "SKIP $Name"
        return
    }
    Write-Host "START $Name"
    & $PythonExecutable -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $Completion)) {
        throw "$Name did not create $Completion"
    }
    Write-Host "COMPLETE $Name"
}

function Get-EvalPath([int]$Scale, [string]$OperatorLevel) {
    if ($OperatorLevel -eq "O0") {
        return Join-Path $panel "jitter\seed_$JitterSeed\x$Scale\test.jsonl"
    }
    return Join-Path $matrix "seed_$JitterSeed\x$Scale\$($OperatorLevel.ToLowerInvariant())\test.jsonl"
}

if (-not (Test-Path -LiteralPath (Join-Path $matrix "manifest.json"))) {
    throw "Frozen operator-jitter matrix is unavailable: $matrix"
}
$seedOrder = @(17011, 17023, 17037)
$seedIndex = [Array]::IndexOf($seedOrder, $JitterSeed)
if ($seedIndex -gt 0) {
    $previousSeed = $seedOrder[$seedIndex - 1]
    $previousRoot = Join-Path (Split-Path $runRoot -Parent) "seed_$previousSeed"
    foreach ($arm in @("asl", "direct")) {
        foreach ($scale in @(1, 1000)) {
            foreach ($operatorLevel in @("o0", "o1", "o5", "o6")) {
                $cell = Join-Path $previousRoot "$arm\x$scale\$operatorLevel"
                $complete = Get-ChildItem $cell -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue
                if (-not $complete) {
                    throw "Seed $JitterSeed is gated on complete seed $previousSeed cell $arm/x$scale/$operatorLevel"
                }
            }
        }
    }
}
if ($Phase -in @("ASL", "All") -and -not (Test-Path -LiteralPath $BaseAdapterPath)) {
    throw "Base GSM adapter is unavailable: $BaseAdapterPath"
}
if (
    $Phase -in @("ASL", "All") -and
    ($Level | Where-Object { $_ -ne "O0" }) -and
    -not (Test-Path -LiteralPath $OperatorAdapterPath)
) {
    throw "Operator adapter is unavailable: $OperatorAdapterPath"
}

Push-Location $root
try {
    foreach ($scale in $Factor) {
        foreach ($operatorLevel in $Level) {
            $eval = Get-EvalPath $scale $operatorLevel
            $levelName = $operatorLevel.ToLowerInvariant()
            if ($Phase -in @("ASL", "All")) {
                $adapter = if ($operatorLevel -eq "O0") {
                    $BaseAdapterPath
                } else {
                    $OperatorAdapterPath
                }
                $adapterId = if ($operatorLevel -eq "O0") {
                    "Qwen3-0.6B-GSM8K-U2000-E4500-F0-L0-r8-seed99173"
                } else {
                    "Qwen3-0.6B-GSM8K-OC-natural-v2-r8-seed99173"
                }
                $output = Join-Path $runRoot "asl\x$scale\$levelName\$Device\shard_0"
                Invoke-Checked "seed-$JitterSeed-ASL-x$scale-$operatorLevel" (Join-Path $output "summary.json") @(
                    "-m", "ccpu.paper1.e3", "run-gsm8k-official-shard",
                    "--eval", $eval, "--config", $aslConfig,
                    "--adapter-path", $adapter, "--adapter-id", $adapterId,
                    "--output-dir", $output,
                    "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
                )
            }
            if ($Phase -in @("Direct", "All")) {
                $output = Join-Path $runRoot "direct\x$scale\$levelName\$Device\shard_0"
                Invoke-Checked "seed-$JitterSeed-Direct-x$scale-$operatorLevel" (Join-Path $output "summary.json") @(
                    "-m", "ccpu.paper1.e3", "run-gsm8k-direct-shard",
                    "--eval", $eval, "--config", $directConfig,
                    "--condition", "direct_reasoning", "--output-dir", $output,
                    "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
                )
            }
        }
    }
}
finally {
    Pop-Location
}
