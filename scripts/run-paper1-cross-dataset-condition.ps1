param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)]
    [ValidateSet("asdiv", "svamp", "mawps", "gsm_plus")]
    [string]$Dataset,
    [Parameter(Mandatory = $true)][ValidateSet("E1", "E2")][string]$Condition,
    [Parameter(Mandatory = $true)][string]$AdapterPath,
    [Parameter(Mandatory = $true)][string]$AdapterId,
    [ValidateSet("cpu", "xpu", "cuda")][string]$Device = "cpu",
    [string]$PythonExecutable = "python",
    [int]$ShardIndex = 0,
    [int]$ShardCount = 1
)

$ErrorActionPreference = "Stop"
$artifactRoot = Join-Path $RepositoryRoot "artifacts/paper1/cross_dataset_transfer_v1"
$evalPath = Join-Path $artifactRoot "$Dataset/diagnostic.jsonl"
$configPath = Join-Path $RepositoryRoot "configs/paper1/asl_pilot_qwen_base_$Device.json"
$conditionName = $Condition.ToLowerInvariant()
$outputRoot = Join-Path $artifactRoot "$Dataset/$conditionName"
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"

function Invoke-Evaluation([string]$InputPath, [string]$OutputDir) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-official-shard `
        --eval $InputPath `
        --config $configPath `
        --adapter-path $AdapterPath `
        --adapter-id $AdapterId `
        --output-dir $OutputDir `
        --shard-index $ShardIndex `
        --shard-count $ShardCount `
        --checkpoint-every 1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($exitCode -ne 0) {
        throw "$Condition evaluation failed with exit code $exitCode"
    }
}

Push-Location $RepositoryRoot
try {
    Invoke-Evaluation $evalPath (Join-Path $outputRoot "$Device/target/shard_$ShardIndex")
    if ($Condition -eq "E1") {
        $gsmEval = Join-Path $RepositoryRoot `
            "artifacts/paper1/gsm8k_scale_v1/official_test_v1/confirmatory.jsonl"
        Invoke-Evaluation $gsmEval (Join-Path $outputRoot "$Device/gsm_retention/shard_$ShardIndex")
    }
}
finally {
    Pop-Location
}
