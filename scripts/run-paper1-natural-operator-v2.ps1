param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][ValidateSet("xpu", "cuda")][string]$Device,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [ValidateSet("Train", "ASL", "Direct", "All")][string]$Phase = "All",
    [ValidateSet("O1", "O3", "O5", "O6")][string[]]$Level = @("O1", "O3", "O5", "O6")
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$data = Join-Path $root "artifacts\paper1\operator_complexity_v2\gsm8k_matched_seed99173"
$runRoot = Join-Path $data "runs"
$adapterRun = Join-Path $runRoot "operator_adapter_r8_seed99173_$Device"
$adapter = Join-Path $adapterRun "adapter"
$config = Join-Path $root "configs\paper1\operator_complexity\natural_v2_qwen06_$Device.json"
$directConfig = Join-Path $root "configs\paper1\gsm8k_direct_reasoning_qwen_$Device.json"
$initialAdapter = Join-Path $root "artifacts\paper1\gsm8k_scale_v1\u2000_e4500\qwen_run\adapter"
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

Push-Location $root
try {
    if ($Phase -in @("Train", "All")) {
        Invoke-Checked "natural-v2-train" (Join-Path $adapterRun "training_report.json") @(
            "-m", "ccpu", "paper1", "train-lora",
            "--config", $config,
            "--model", "Qwen/Qwen3-0.6B",
            "--train", (Join-Path $data "operators\sft\train.jsonl"),
            "--dev", (Join-Path $data "operators\sft\dev.jsonl"),
            "--initial-adapter-path", $initialAdapter,
            "--adapter-id", "Qwen3-0.6B-GSM8K-OC-natural-v2-r8-seed99173",
            "--output-dir", $adapterRun
        )
    }
    if ($Phase -in @("ASL", "All")) {
        if (-not (Test-Path -LiteralPath $adapter)) {
            throw "Natural-v2 adapter is unavailable: $adapter"
        }
        foreach ($operatorLevel in $Level) {
            $levelName = $operatorLevel.ToLowerInvariant()
            $eval = Join-Path $data "operators\$levelName\test.jsonl"
            $output = Join-Path $runRoot "asl\$levelName\$Device\shard_0"
            Invoke-Checked "natural-v2-ASL-$operatorLevel" (Join-Path $output "summary.json") @(
                "-m", "ccpu.paper1.e3", "run-gsm8k-official-shard",
                "--eval", $eval,
                "--config", $config,
                "--adapter-path", $adapter,
                "--adapter-id", "Qwen3-0.6B-GSM8K-OC-natural-v2-r8-seed99173",
                "--output-dir", $output,
                "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
            )
        }
    }
    if ($Phase -in @("Direct", "All")) {
        foreach ($operatorLevel in $Level) {
            $levelName = $operatorLevel.ToLowerInvariant()
            $eval = Join-Path $data "operators\$levelName\test.jsonl"
            $output = Join-Path $runRoot "direct\$levelName\$Device\shard_0"
            Invoke-Checked "natural-v2-Direct-$operatorLevel" (Join-Path $output "summary.json") @(
                "-m", "ccpu.paper1.e3", "run-gsm8k-direct-shard",
                "--eval", $eval,
                "--config", $directConfig,
                "--condition", "direct_reasoning",
                "--output-dir", $output,
                "--shard-index", "0", "--shard-count", "1", "--checkpoint-every", "1"
            )
        }
    }
}
finally {
    Pop-Location
}
