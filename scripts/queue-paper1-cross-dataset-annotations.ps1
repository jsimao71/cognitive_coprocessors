param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)]
    [ValidateSet("asdiv", "svamp", "mawps", "gsm_plus")]
    [string]$Dataset,
    [Parameter(Mandatory = $true)][int]$WaitProcessId,
    [Parameter(Mandatory = $true)]
    [ValidateSet("cpu", "xpu", "cuda")]
    [string]$E0Device,
    [string]$PythonExecutable = "python",
    [Parameter(Mandatory = $true)][string]$CodexExecutable,
    [string]$CodexModel = "gpt-5.6-sol"
)

$ErrorActionPreference = "Stop"
Wait-Process -Id $WaitProcessId -ErrorAction SilentlyContinue

$datasetRoot = Join-Path $RepositoryRoot `
    "artifacts/paper1/cross_dataset_transfer_v1/$Dataset"
$summaryPath = Join-Path $datasetRoot "e0_gsm_lora/$E0Device/shard_0/summary.json"
$manifestPath = Join-Path $datasetRoot "manifest.json"
if (!(Test-Path $summaryPath)) {
    throw "E0 summary was not produced: $summaryPath"
}
$summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($summary.prediction_count -ne 250) {
    throw "E0 is incomplete: expected 250 predictions, found $($summary.prediction_count)"
}
if ($summary.eval_sha256 -ne $manifest.outputs.diagnostic.sha256) {
    throw "E0 diagnostic checksum differs from the frozen manifest"
}

$work = Join-Path $datasetRoot "e1_e2_pilot500"
$prompt = Join-Path $RepositoryRoot "configs/paper1/local_codex_annotation_prompt.md"
$schema = Join-Path $RepositoryRoot "configs/paper1/semantic_annotation_batch.schema.json"
$devName = if ($Dataset -eq "gsm_plus") { "dev100" } else { "dev" }
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"

Push-Location $RepositoryRoot
try {
    & $PythonExecutable -u -m ccpu.dsl_dataset run-local `
        --requests-dir (Join-Path $work "primary_requests/requests") `
        --output-dir (Join-Path $work "primary_codex") `
        --prompt $prompt `
        --schema $schema `
        --repo-root $RepositoryRoot `
        --executable $CodexExecutable `
        --model $CodexModel `
        --reasoning-effort medium `
        --concurrency 4
    if ($LASTEXITCODE -ne 0) {
        throw "primary annotation failed with exit code $LASTEXITCODE"
    }
    & $PythonExecutable -u -m ccpu.dsl_dataset run-local `
        --requests-dir (Join-Path $work "${devName}_requests/requests") `
        --output-dir (Join-Path $work "${devName}_codex") `
        --prompt $prompt `
        --schema $schema `
        --repo-root $RepositoryRoot `
        --executable $CodexExecutable `
        --model $CodexModel `
        --reasoning-effort medium `
        --concurrency 2
    if ($LASTEXITCODE -ne 0) {
        throw "development annotation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
