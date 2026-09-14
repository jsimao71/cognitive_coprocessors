param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"
$scale = Join-Path $RepositoryRoot "artifacts/paper1/gsm8k_scale_v1"
$train = Join-Path $scale "u2000_e4500/train.jsonl"
$dev = Join-Path $scale "g1_f0_4500/eval/dev.jsonl"
$root = Join-Path $scale "qwen4b_u2000_e4500/seed99173_qlora_cuda"

& $PythonExecutable -c "import bitsandbytes, torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), bitsandbytes.__version__)"
if ($LASTEXITCODE -ne 0) { throw "CUDA/bitsandbytes validation failed" }

if ($Smoke) {
    $smokeData = Join-Path $root "smoke_data"
    New-Item -ItemType Directory -Force -Path $smokeData | Out-Null
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllLines(
        (Join-Path $smokeData "train.jsonl"),
        [string[]]@(Get-Content -LiteralPath $train -First 8),
        $utf8
    )
    [IO.File]::WriteAllLines(
        (Join-Path $smokeData "dev.jsonl"),
        [string[]]@(Get-Content -LiteralPath $dev -First 2),
        $utf8
    )
    $train = Join-Path $smokeData "train.jsonl"
    $dev = Join-Path $smokeData "dev.jsonl"
    $config = Join-Path $RepositoryRoot "configs/paper1/e3_g1_gsm8k_u2000_e4500_f0_l0_qwen4b_qlora_smoke_cuda.json"
    $output = Join-Path $root "memory_smoke"
}
else {
    $config = Join-Path $RepositoryRoot "configs/paper1/e3_g1_gsm8k_u2000_e4500_f0_l0_qwen4b_qlora_cuda.json"
    $output = Join-Path $root "qwen_run"
}

if (Test-Path -LiteralPath (Join-Path $output "training_report.json")) {
    Write-Host "SKIP completed Qwen3-4B QLoRA run: $output"
    return
}

& $PythonExecutable -u -m ccpu paper1 train-lora `
    --config $config `
    --model Qwen/Qwen3-4B `
    --train $train `
    --dev $dev `
    --output-dir $output
if ($LASTEXITCODE -ne 0) { throw "Qwen3-4B QLoRA training failed" }
Write-Host "QWEN4B_QLORA_COMPLETE smoke=$Smoke output=$output"
