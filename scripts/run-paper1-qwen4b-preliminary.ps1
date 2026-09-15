param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$DirectConfig,
    [Parameter(Mandatory = $true)][string]$AslConfig,
    [ValidateSet("direct", "asl", "both")][string]$Route = "both",
    [ValidateSet("operator_o1", "operator_o5", "operator_o6", "jitter_x1_o0", "jitter_x1000_o0", "mixed_x1000_o5", "mixed_x1000_o6", "all")]
    [string]$Panel = "all",
    [ValidateSet("cuda", "xpu")][string]$Device = "xpu",
    [ValidateRange(0, 99)][int]$ShardIndex = 0,
    [ValidateRange(1, 100)][int]$ShardCount = 10,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"
$outputRoot = Join-Path $RepositoryRoot `
    "artifacts/paper1/evaluation_model_ladder_v1/qwen3_4b/preliminary_seed_17011"

foreach ($configPath in @($DirectConfig, $AslConfig)) {
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    if ($config.model.model_id -ne "Qwen/Qwen3-4B") {
        throw "expected Qwen/Qwen3-4B in $configPath"
    }
    if ($config.model.device -ne $Device) {
        throw "device mismatch: runner=$Device config=$($config.model.device) in $configPath"
    }
}

# Shard 0/10 freezes ten evenly spaced identities from each 100-row panel.
$panels = @(
    [pscustomobject]@{ Id = "operator_o1"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o1/test.jsonl" },
    [pscustomobject]@{ Id = "operator_o5"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o5/test.jsonl" },
    [pscustomobject]@{ Id = "operator_o6"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o6/test.jsonl" },
    [pscustomobject]@{ Id = "jitter_x1_o0"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/jitter/seed_17011/x1/test.jsonl" },
    [pscustomobject]@{ Id = "jitter_x1000_o0"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/jitter/seed_17011/x1000/test.jsonl" },
    [pscustomobject]@{ Id = "mixed_x1000_o5"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o5/test.jsonl" },
    [pscustomobject]@{ Id = "mixed_x1000_o6"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o6/test.jsonl" }
)

if ($ShardIndex -ge $ShardCount) {
    throw "ShardIndex must be smaller than ShardCount"
}

$selected = @($panels | Where-Object { $Panel -eq "all" -or $_.Id -eq $Panel })
foreach ($item in $selected) {
    $eval = Join-Path $RepositoryRoot $item.Eval
    if (!(Test-Path -LiteralPath $eval)) {
        throw "frozen evaluation file is missing: $eval"
    }
    $count = @(Get-Content -LiteralPath $eval).Count
    if ($count -ne 100) {
        throw "$($item.Id) has $count rows; expected 100"
    }
}

$rowsPerPanel = [math]::Ceiling((100 - $ShardIndex) / $ShardCount)
if ($ValidateOnly) {
    Write-Host "VALID model=Qwen3-4B seed=17011 panels=$($selected.Count) rows_per_panel=$rowsPerPanel shard=$ShardIndex/$ShardCount route=$Route"
    return
}

function Remove-StaleLock([string]$LockPath) {
    if (!(Test-Path -LiteralPath $LockPath)) {
        return
    }
    $owner = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($owner.pid)" `
        -ErrorAction SilentlyContinue
    $isExperimentOwner = $process -and $process.Name -match '^python(?:\.exe)?$' -and `
        $process.CommandLine -match 'ccpu\.paper1\.e3'
    if ($isExperimentOwner) {
        throw "refusing to remove live run lock owned by PID $($owner.pid): $LockPath"
    }
    Write-Host "REMOVE stale lock PID=$($owner.pid) current_process=$($process.Name)"
    Remove-Item -LiteralPath $LockPath -Force
}

function Invoke-Cell($Item, [string]$Condition) {
    $eval = Join-Path $RepositoryRoot $Item.Eval
    $conditionName = if ($Condition -eq "direct") { "direct_long" } else { "asl_base_zero_shot" }
    $output = Join-Path $outputRoot "$($Item.Id)/$conditionName/$Device/shard_$ShardIndex-of-$ShardCount"
    $summary = Join-Path $output "summary.json"
    if (Test-Path -LiteralPath $summary) {
        Write-Host "SKIP $($Item.Id) $conditionName"
        return
    }
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    Remove-StaleLock (Join-Path $output ".run.lock")
    Write-Host "START $($Item.Id) $conditionName shard=$ShardIndex/$ShardCount"
    if ($Condition -eq "direct") {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-direct-shard `
            --eval $eval `
            --config $DirectConfig `
            --condition direct_reasoning `
            --output-dir $output `
            --shard-index $ShardIndex `
            --shard-count $ShardCount `
            --seed 44017 `
            --checkpoint-every 1
    }
    else {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-official-shard `
            --eval $eval `
            --config $AslConfig `
            --adapter-id Qwen3-4B-base-zero-shot `
            --output-dir $output `
            --shard-index $ShardIndex `
            --shard-count $ShardCount `
            --seed 44017 `
            --checkpoint-every 1
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$($Item.Id) $conditionName failed with exit code $LASTEXITCODE"
    }
}

foreach ($item in $selected) {
    if ($Route -in @("direct", "both")) {
        Invoke-Cell $item "direct"
    }
    if ($Route -in @("asl", "both")) {
        Invoke-Cell $item "asl"
    }
}
Write-Host "QWEN4B_PRELIMINARY_COMPLETE route=$Route panel=$Panel shard=$ShardIndex/$ShardCount"
