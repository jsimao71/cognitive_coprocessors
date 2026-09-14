param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$AdapterPath,
    [Parameter(Mandatory = $true)][string]$DirectConfig,
    [Parameter(Mandatory = $true)][string]$AslConfig,
    [ValidateSet("direct", "asl", "both")][string]$Route = "both",
    [ValidateSet("operators", "jitter", "mixed", "all")][string]$Phase = "all",
    [ValidateSet("cuda", "xpu")][string]$Device = "cuda",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"
$outputRoot = Join-Path $RepositoryRoot `
    "artifacts/paper1/evaluation_model_ladder_v1/qwen3_1_7b/seed_17011"

$panels = @(
    [pscustomobject]@{ Phase = "operators"; Id = "operator_o1"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o1/test.jsonl" },
    [pscustomobject]@{ Phase = "operators"; Id = "operator_o3"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o3/test.jsonl" },
    [pscustomobject]@{ Phase = "operators"; Id = "operator_o5"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o5/test.jsonl" },
    [pscustomobject]@{ Phase = "operators"; Id = "operator_o6"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/operators/o6/test.jsonl" },
    [pscustomobject]@{ Phase = "jitter"; Id = "jitter_x1_o0"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/jitter/seed_17011/x1/test.jsonl" },
    [pscustomobject]@{ Phase = "jitter"; Id = "jitter_x1000_o0"; Eval = "artifacts/paper1/operator_complexity_v2/gsm8k_matched_seed99173/jitter/seed_17011/x1000/test.jsonl" },
    [pscustomobject]@{ Phase = "mixed"; Id = "mixed_x1_o1"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1/o1/test.jsonl" },
    [pscustomobject]@{ Phase = "mixed"; Id = "mixed_x1_o5"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1/o5/test.jsonl" },
    [pscustomobject]@{ Phase = "mixed"; Id = "mixed_x1_o6"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1/o6/test.jsonl" },
    [pscustomobject]@{ Phase = "mixed"; Id = "mixed_x1000_o1"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o1/test.jsonl" },
    [pscustomobject]@{ Phase = "mixed"; Id = "mixed_x1000_o5"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o5/test.jsonl" },
    [pscustomobject]@{ Phase = "mixed"; Id = "mixed_x1000_o6"; Eval = "artifacts/paper1/operator_jitter_matrix_v1/gsm8k_matched_seed99173/seed_17011/x1000/o6/test.jsonl" }
)

if ($ValidateOnly) {
    $selected = @($panels | Where-Object { $Phase -eq "all" -or $_.Phase -eq $Phase })
    foreach ($panel in $selected) {
        $eval = Join-Path $RepositoryRoot $panel.Eval
        if (!(Test-Path -LiteralPath $eval)) {
            throw "frozen evaluation file is missing: $eval"
        }
        $count = @(Get-Content -LiteralPath $eval).Count
        if ($count -ne 100) {
            throw "$($panel.Id) has $count rows; expected 100"
        }
    }
    Write-Host "VALID seed=17011 phase=$Phase panels=$($selected.Count) rows_per_panel=100"
    return
}

function Remove-StaleLock([string]$LockPath) {
    if (!(Test-Path -LiteralPath $LockPath)) {
        return
    }
    $owner = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
    if (Get-Process -Id $owner.pid -ErrorAction SilentlyContinue) {
        throw "refusing to remove live run lock owned by PID $($owner.pid): $LockPath"
    }
    Remove-Item -LiteralPath $LockPath -Force
}

function Invoke-Cell($Panel, [string]$Condition) {
    $eval = Join-Path $RepositoryRoot $Panel.Eval
    if (!(Test-Path -LiteralPath $eval)) {
        throw "frozen evaluation file is missing: $eval"
    }
    $conditionName = if ($Condition -eq "direct") { "direct" } else { "asl_gsm" }
    $output = Join-Path $outputRoot "$($Panel.Id)/$conditionName/$Device/shard_0"
    $summary = Join-Path $output "summary.json"
    if (Test-Path -LiteralPath $summary) {
        Write-Host "SKIP $($Panel.Id) $conditionName"
        return
    }
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    Remove-StaleLock (Join-Path $output ".run.lock")
    Write-Host "START $($Panel.Id) $conditionName"
    if ($Condition -eq "direct") {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-direct-shard `
            --eval $eval `
            --config $DirectConfig `
            --condition direct_reasoning `
            --output-dir $output `
            --shard-index 0 `
            --shard-count 1 `
            --seed 44017 `
            --checkpoint-every 5
    }
    else {
        & $PythonExecutable -u -m ccpu.paper1.e3 run-gsm8k-official-shard `
            --eval $eval `
            --config $AslConfig `
            --adapter-path $AdapterPath `
            --adapter-id Qwen3-1.7B-G1-GSM8K-U2000-E4500-F0-L0-r8-init99173 `
            --output-dir $output `
            --shard-index 0 `
            --shard-count 1 `
            --seed 44017 `
            --checkpoint-every 5
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$($Panel.Id) $conditionName failed with exit code $LASTEXITCODE"
    }
}

foreach ($panel in $panels) {
    if ($Phase -ne "all" -and $panel.Phase -ne $Phase) {
        continue
    }
    if ($Route -in @("direct", "both")) {
        Invoke-Cell $panel "direct"
    }
    if ($Route -in @("asl", "both")) {
        Invoke-Cell $panel "asl"
    }
}
Write-Host "QWEN17_SEED17011_MATRIX_COMPLETE route=$Route phase=$Phase"
