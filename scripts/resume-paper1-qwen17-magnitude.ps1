param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$AdapterPath
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"
$outputRoot = Join-Path $RepositoryRoot `
    "artifacts/paper1/evaluation_model_ladder_v1/qwen3_1_7b"
$directConfig = Join-Path $RepositoryRoot `
    "configs/paper1/gsm8k_direct_reasoning_long_qwen1_7b_cuda.json"
$aslConfig = Join-Path $RepositoryRoot `
    "configs/paper1/asl_pilot_qwen1_7b_base_cuda.json"

function Invoke-MagnitudeCondition {
    param(
        [Parameter(Mandatory = $true)][string]$Factor,
        [Parameter(Mandatory = $true)]
        [ValidateSet("direct", "asl_gsm")][string]$Condition
    )

    $eval = Join-Path $RepositoryRoot `
        "artifacts/paper1/gsm8k_scale_v1/magnitude_ladder_v1/data/factor_$Factor.jsonl"
    $conditionRoot = Join-Path $outputRoot `
        "gsm8k_scale_x$Factor/$Condition/cuda"
    $shardRoot = Join-Path $conditionRoot "shard_0"
    $summary = Join-Path $shardRoot "summary.json"
    if (Test-Path -LiteralPath $summary) {
        Write-Host "SKIP x$Factor $Condition"
        return
    }
    $lock = Join-Path $shardRoot ".run.lock"
    if (Test-Path -LiteralPath $lock) {
        Remove-Item -LiteralPath $lock -Force
    }

    Write-Host "START x$Factor $Condition"
    if ($Condition -eq "direct") {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-direct-shard `
            --eval $eval `
            --config $directConfig `
            --condition direct_reasoning `
            --output-dir $shardRoot `
            --shard-index 0 `
            --shard-count 1 `
            --seed 44017 `
            --checkpoint-every 5
    }
    else {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-official-shard `
            --eval $eval `
            --config $aslConfig `
            --adapter-path $AdapterPath `
            --adapter-id Qwen3-1.7B-G1-GSM8K-U2000-E4500-F0-L0-r8-init99173-cuda `
            --output-dir $shardRoot `
            --shard-index 0 `
            --shard-count 1 `
            --seed 44017 `
            --checkpoint-every 5
    }
    if ($LASTEXITCODE -ne 0) {
        throw "x$Factor $Condition failed with exit code $LASTEXITCODE"
    }
}

Invoke-MagnitudeCondition -Factor "1000" -Condition "direct"
Invoke-MagnitudeCondition -Factor "1000" -Condition "asl_gsm"
Invoke-MagnitudeCondition -Factor "1000000" -Condition "direct"
Invoke-MagnitudeCondition -Factor "1000000" -Condition "asl_gsm"
Write-Host "QWEN17_MAGNITUDE_QUEUE_COMPLETE"
