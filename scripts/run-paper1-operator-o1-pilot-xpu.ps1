param([string]$RepositoryRoot)

$ErrorActionPreference = "Stop"
$RepoRoot = if ($RepositoryRoot) {
    (Resolve-Path $RepositoryRoot).Path
}
else {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$Python = Join-Path $env:USERPROFILE ".venvs\modal-llm-xpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$Root = Join-Path $RepoRoot "artifacts\paper1\operator_complexity_v1"
$O1 = Join-Path $Root "o1"
$Pilot = Join-Path $Root "o1_pilot_seed99173"
$BaseConfig = Join-Path $RepoRoot "configs\paper1\asl_pilot_qwen_base_xpu.json"
$DirectConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_qwen_xpu.json"

if (-not (Test-Path -LiteralPath $Python)) { throw "XPU Python is unavailable: $Python" }
& $Python -c "import torch; assert torch.xpu.is_available(); print(torch.xpu.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "XPU validation failed" }

function Invoke-Step {
    param([string]$Name, [string]$CompletionPath, [string[]]$Arguments, [switch]$MainCli)
    if (Test-Path -LiteralPath $CompletionPath) {
        Write-Host "SKIP $Name (complete)"
        return
    }
    Write-Host "START $Name"
    if ($MainCli) { & $Python -u -m ccpu @Arguments }
    else { & $Python -u -m ccpu.paper1.e3 @Arguments }
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $CompletionPath)) {
        throw "$Name exited without completion artifact: $CompletionPath"
    }
    Write-Host "COMPLETE $Name"
}

Push-Location $RepoRoot
try {
    Invoke-Step -Name "freeze O1 pilot" -CompletionPath (Join-Path $Pilot "manifest.json") `
        -Arguments @(
            "prepare-operator-o1-pilot", "--source-dir", $O1, "--output-dir", $Pilot,
            "--train-count", "200", "--dev-count", "30", "--test-count", "100",
            "--seed", "99173"
        )

    foreach ($representation in @("ap", "as")) {
        $upper = $representation.ToUpperInvariant()
        $run = Join-Path $Pilot "$representation\qwen_run"
        $eval = Join-Path $Pilot "$representation\eval"
        $config = Join-Path $RepoRoot "configs\paper1\operator_complexity\o1_$($representation)_qwen06_xpu.json"
        $adapterId = "Qwen3-0.6B-GSM8K-OC-O1-$upper-pilot-r8-seed99173"
        Invoke-Step -Name "train O1 $upper pilot" -CompletionPath (Join-Path $run "training_report.json") `
            -MainCli -Arguments @(
                "paper1", "train-lora", "--config", $config, "--model", "Qwen/Qwen3-0.6B",
                "--train", (Join-Path $Pilot "$representation\train.jsonl"),
                "--dev", (Join-Path $Pilot "$representation\dev.jsonl"), "--output-dir", $run
            )
        Invoke-Step -Name "evaluate O1 $upper pilot" -CompletionPath (Join-Path $eval "summary.json") `
            -Arguments @(
                "run-gsm8k-official-shard", "--eval", (Join-Path $Pilot "$representation\test.jsonl"),
                "--config", $BaseConfig, "--adapter-path", (Join-Path $run "adapter"),
                "--adapter-id", $adapterId, "--output-dir", $eval,
                "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
            )
    }

    Invoke-Step -Name "evaluate O1 direct pilot" `
        -CompletionPath (Join-Path $Pilot "direct\eval\summary.json") `
        -Arguments @(
            "run-gsm8k-direct-shard", "--eval", (Join-Path $Pilot "test.jsonl"),
            "--config", $DirectConfig, "--condition", "direct_reasoning",
            "--output-dir", (Join-Path $Pilot "direct\eval"),
            "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
        )
}
finally {
    Pop-Location
}
