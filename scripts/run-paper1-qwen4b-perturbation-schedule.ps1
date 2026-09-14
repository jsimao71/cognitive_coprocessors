param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$DirectConfig,
    [Parameter(Mandatory = $true)][string]$AslConfig,
    [ValidateSet("direct", "asl", "both")][string]$Route = "both",
    [ValidateSet("base", "lora")][string]$AslCondition = "base",
    [string]$AdapterPath,
    [string]$AdapterId = "Qwen3-4B-U2000-E4500-F0-L0-r8-init99173",
    [ValidateSet("baseline", "magnitude", "operator", "jitter", "mixed", "complexity", "all")]
    [string]$Stage = "all",
    [string]$Panel = "all",
    [ValidateSet("cuda", "xpu")][string]$Device = "xpu",
    [ValidateSet(2048, 4096)][int]$DirectBudget = 4096,
    [ValidateRange(0, 9)][int]$ShardIndex = 0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"
$shardCount = 10
$outputRoot = Join-Path $RepositoryRoot `
    "artifacts/paper1/evaluation_model_ladder_v1/qwen3_4b/perturbation_schedule_seed_17011"

$direct = Get-Content -LiteralPath $DirectConfig -Raw | ConvertFrom-Json
$asl = Get-Content -LiteralPath $AslConfig -Raw | ConvertFrom-Json
foreach ($config in @($direct, $asl)) {
    if ($config.model.model_id -ne "Qwen/Qwen3-4B") {
        throw "all configs must use Qwen/Qwen3-4B"
    }
    if ($config.model.device -ne $Device) {
        throw "device mismatch: runner=$Device config=$($config.model.device)"
    }
}
if ([int]$direct.model.max_new_tokens -ne $DirectBudget) {
    throw "DirectBudget=$DirectBudget does not match config max_new_tokens=$($direct.model.max_new_tokens)"
}
if ($AslCondition -eq "lora" -and [string]::IsNullOrWhiteSpace($AdapterPath)) {
    throw "AdapterPath is required for the LoRA ASL condition"
}

$panels = @(
    [pscustomobject]@{ Stage = "baseline"; Id = "original"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/original/test.jsonl" },
    [pscustomobject]@{ Stage = "magnitude"; Id = "magnitude_x1"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/magnitude/x1/test.jsonl" },
    [pscustomobject]@{ Stage = "magnitude"; Id = "magnitude_x100"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/magnitude/x100/test.jsonl" },
    [pscustomobject]@{ Stage = "magnitude"; Id = "magnitude_x1000"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/magnitude/x1000/test.jsonl" },
    [pscustomobject]@{ Stage = "magnitude"; Id = "magnitude_x10000"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/magnitude/x10000/test.jsonl" },
    [pscustomobject]@{ Stage = "magnitude"; Id = "magnitude_x1000000"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/magnitude/x1000000/test.jsonl" },
    [pscustomobject]@{ Stage = "operator"; Id = "operator_o1"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o1/test.jsonl" },
    [pscustomobject]@{ Stage = "operator"; Id = "operator_o3"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o3/test.jsonl" },
    [pscustomobject]@{ Stage = "operator"; Id = "operator_o5"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o5/test.jsonl" },
    [pscustomobject]@{ Stage = "operator"; Id = "operator_o6"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o6/test.jsonl" },
    [pscustomobject]@{ Stage = "jitter"; Id = "jitter_x1_o0"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/jitter/seed_17011/x1/test.jsonl" },
    [pscustomobject]@{ Stage = "jitter"; Id = "jitter_x1000_o0"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/jitter/seed_17011/x1000/test.jsonl" },
    [pscustomobject]@{ Stage = "mixed"; Id = "mixed_x1_o1"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1/o1/test.jsonl" },
    [pscustomobject]@{ Stage = "mixed"; Id = "mixed_x1_o5"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1/o5/test.jsonl" },
    [pscustomobject]@{ Stage = "mixed"; Id = "mixed_x1_o6"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1/o6/test.jsonl" },
    [pscustomobject]@{ Stage = "mixed"; Id = "mixed_x1000_o1"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o1/test.jsonl" },
    [pscustomobject]@{ Stage = "mixed"; Id = "mixed_x1000_o5"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o5/test.jsonl" },
    [pscustomobject]@{ Stage = "mixed"; Id = "mixed_x1000_o6"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o6/test.jsonl" },
    [pscustomobject]@{ Stage = "complexity"; Id = "complexity_c1"; Eval = "artifacts/paper1/compositional_complexity_v1/pilot_seed124001/C1/test.jsonl" },
    [pscustomobject]@{ Stage = "complexity"; Id = "complexity_c2"; Eval = "artifacts/paper1/compositional_complexity_v1/pilot_seed124001/C2/test.jsonl" },
    [pscustomobject]@{ Stage = "complexity"; Id = "complexity_c3"; Eval = "artifacts/paper1/compositional_complexity_v1/pilot_seed124001/C3/test.jsonl" },
    [pscustomobject]@{ Stage = "complexity"; Id = "complexity_c4"; Eval = "artifacts/paper1/compositional_complexity_v1/pilot_seed124001/C4/test.jsonl" }
)

$selected = @($panels | Where-Object {
    ($Stage -eq "all" -or $_.Stage -eq $Stage) -and
    ($Panel -eq "all" -or $_.Id -eq $Panel)
})
if (!$selected) {
    throw "no panel matches Stage=$Stage Panel=$Panel"
}
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
if ($ValidateOnly) {
    Write-Host "VALID model=Qwen3-4B stage=$Stage panels=$($selected.Count) rows_per_panel=10 shard=$ShardIndex/10 route=$Route asl=$AslCondition direct_budget=$DirectBudget"
    return
}

function Remove-StaleLock([string]$LockPath) {
    if (!(Test-Path -LiteralPath $LockPath)) { return }
    $owner = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $owner.pid -ErrorAction SilentlyContinue) {
        throw "refusing to remove live run lock owned by PID $($owner.pid): $LockPath"
    }
    Remove-Item -LiteralPath $LockPath -Force
}

function Invoke-Cell($Item, [string]$Condition) {
    $eval = Join-Path $RepositoryRoot $Item.Eval
    if ($Condition -eq "direct") {
        $conditionName = "direct_$DirectBudget"
    }
    elseif ($AslCondition -eq "lora") {
        $conditionName = "asl_lora_r8"
    }
    else {
        $conditionName = "asl_base_zero_shot"
    }
    $output = Join-Path $outputRoot "$($Item.Id)/$conditionName/$Device/shard_$ShardIndex-of-10"
    if (Test-Path -LiteralPath (Join-Path $output "summary.json")) {
        Write-Host "SKIP $($Item.Id) $conditionName shard=$ShardIndex/10"
        return
    }
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    Remove-StaleLock (Join-Path $output ".run.lock")
    Write-Host "START $($Item.Id) $conditionName shard=$ShardIndex/10"
    if ($Condition -eq "direct") {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-direct-shard `
            --eval $eval --config $DirectConfig --condition direct_reasoning `
            --output-dir $output --shard-index $ShardIndex --shard-count $shardCount `
            --seed 44017 --checkpoint-every 1
    }
    elseif ($AslCondition -eq "lora") {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-official-shard `
            --eval $eval --config $AslConfig --adapter-path $AdapterPath `
            --adapter-id $AdapterId --output-dir $output `
            --shard-index $ShardIndex --shard-count $shardCount `
            --seed 44017 --checkpoint-every 1
    }
    else {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-official-shard `
            --eval $eval --config $AslConfig --adapter-id Qwen3-4B-base-zero-shot `
            --output-dir $output --shard-index $ShardIndex --shard-count $shardCount `
            --seed 44017 --checkpoint-every 1
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$($Item.Id) $conditionName failed with exit code $LASTEXITCODE"
    }
}

foreach ($item in $selected) {
    if ($Route -in @("direct", "both")) { Invoke-Cell $item "direct" }
    if ($Route -in @("asl", "both")) { Invoke-Cell $item "asl" }
}
Write-Host "QWEN4B_SCHEDULE_COMPLETE stage=$Stage route=$Route shard=$ShardIndex/10"
