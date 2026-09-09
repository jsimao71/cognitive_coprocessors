param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][ValidateSet("asdiv", "svamp", "mawps", "gsm_plus")][string]$Dataset,
    [ValidateSet("cpu", "xpu", "cuda")][string]$Device = "cpu",
    [int]$ShardIndex = 0,
    [int]$ShardCount = 1
)

$ErrorActionPreference = "Stop"
$artifactRoot = Join-Path $RepositoryRoot "artifacts/paper1/cross_dataset_transfer_v1"
$evalPath = Join-Path $artifactRoot "$Dataset/diagnostic.jsonl"
$adapterPath = Join-Path $RepositoryRoot "artifacts/paper1/gsm8k_scale_v1/u2000_e4500/qwen_run/adapter"
$configPath = Join-Path $RepositoryRoot "configs/paper1/asl_pilot_qwen_base_$Device.json"
$outputDir = Join-Path $artifactRoot "$Dataset/e0_gsm_lora/$Device/shard_$ShardIndex"
$logDir = Join-Path $artifactRoot "$Dataset/e0_gsm_lora/$Device/logs"
New-Item -ItemType Directory -Force -Path $outputDir, $logDir | Out-Null

$stdout = Join-Path $logDir "shard_$ShardIndex.stdout.log"
$stderr = Join-Path $logDir "shard_$ShardIndex.stderr.log"
Push-Location $RepositoryRoot
try {
    & python -u -m ccpu.paper1.e3 run-gsm8k-official-shard `
        --eval $evalPath `
        --config $configPath `
        --adapter-path $adapterPath `
        --adapter-id "Qwen3-0.6B-GSM8K-U2000-E4500-F0-L0-r8-seed99173-E0" `
        --output-dir $outputDir `
        --shard-index $ShardIndex `
        --shard-count $ShardCount `
        --checkpoint-every 1 1>> $stdout 2>> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "E0 shard failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
