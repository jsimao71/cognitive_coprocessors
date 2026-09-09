param(
    [string]$RepositoryRoot,
    [string[]]$Levels = @("O5", "O6")
)

$ErrorActionPreference = "Stop"
$RepoRoot = if ($RepositoryRoot) { (Resolve-Path $RepositoryRoot).Path } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$Python = Join-Path $env:USERPROFILE ".venvs\ccpu-cuda\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$Root = Join-Path $RepoRoot "artifacts\paper1\operator_complexity_v1"
$BaseConfig = Join-Path $RepoRoot "configs\paper1\asl_pilot_qwen_base_cuda.json"
$DirectConfig = Join-Path $RepoRoot "configs\paper1\gsm8k_direct_reasoning_qwen_cuda.json"

function Invoke-Step {
    param([string]$Name, [string]$CompletionPath, [string[]]$Arguments, [switch]$MainCli)
    if (Test-Path -LiteralPath $CompletionPath) { Write-Host "SKIP $Name"; return }
    Write-Host "START $Name"
    if ($MainCli) { & $Python -u -m ccpu @Arguments } else { & $Python -u -m ccpu.paper1.e3 @Arguments }
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $CompletionPath)) { throw "$Name produced no completion artifact" }
    Write-Host "COMPLETE $Name"
}

Push-Location $RepoRoot
try {
    & $Python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw "CUDA validation failed" }
    foreach ($level in $Levels) {
        $lower = $level.ToLowerInvariant()
        $pilot = Join-Path $Root "${lower}_pilot250_seed99173"
        foreach ($representation in @("ap", "as")) {
            $upper = $representation.ToUpperInvariant()
            $run = Join-Path $pilot "$representation\qwen_run"
            $eval = Join-Path $pilot "$representation\eval"
            $config = Join-Path $RepoRoot "configs\paper1\operator_complexity\${lower}_${representation}_qwen06_cuda.json"
            $adapterId = "Qwen3-0.6B-GSM8K-OC-$level-$upper-pilot-r8-seed99173"
            Invoke-Step "train $level $upper" (Join-Path $run "training_report.json") @(
                "paper1", "train-lora", "--config", $config, "--model", "Qwen/Qwen3-0.6B",
                "--train", (Join-Path $pilot "$representation\train.jsonl"),
                "--dev", (Join-Path $pilot "$representation\dev.jsonl"), "--output-dir", $run
            ) -MainCli
            Invoke-Step "evaluate $level $upper" (Join-Path $eval "summary.json") @(
                "run-gsm8k-official-shard", "--eval", (Join-Path $pilot "$representation\test.jsonl"),
                "--config", $BaseConfig, "--adapter-path", (Join-Path $run "adapter"),
                "--adapter-id", $adapterId, "--output-dir", $eval,
                "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
            )
        }
        Invoke-Step "evaluate $level direct" (Join-Path $pilot "direct\eval\summary.json") @(
            "run-gsm8k-direct-shard", "--eval", (Join-Path $pilot "test.jsonl"),
            "--config", $DirectConfig, "--condition", "direct_reasoning",
            "--output-dir", (Join-Path $pilot "direct\eval"),
            "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
        )
        Invoke-Step "analyze $level pilot" (Join-Path $pilot "analysis\summary.json") @(
            "analyze-operator-pilot", "--eval", (Join-Path $pilot "test.jsonl"),
            "--direct-predictions", (Join-Path $pilot "direct\eval\predictions.jsonl"),
            "--ap-predictions", (Join-Path $pilot "ap\eval\predictions.jsonl"),
            "--as-predictions", (Join-Path $pilot "as\eval\predictions.jsonl"),
            "--output-dir", (Join-Path $pilot "analysis")
        )
    }
}
finally { Pop-Location }
