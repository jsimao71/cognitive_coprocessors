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
$summaries = @()

Push-Location $RepoRoot
try {
    foreach ($level in @("O0", "O1", "O3", "O5", "O6")) {
        $pilot = Join-Path $Root "$($level.ToLowerInvariant())_pilot250_seed99173"
        $required = @(
            (Join-Path $pilot "test.jsonl"),
            (Join-Path $pilot "direct\eval\predictions.jsonl"),
            (Join-Path $pilot "ap\eval\predictions.jsonl"),
            (Join-Path $pilot "as\eval\predictions.jsonl")
        )
        if (($required | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -gt 0) {
            Write-Host "SKIP $level analysis (incomplete predictions)"
            continue
        }
        & $Python -u -m ccpu.paper1.e3 analyze-operator-pilot `
            --eval $required[0] `
            --direct-predictions $required[1] `
            --ap-predictions $required[2] `
            --as-predictions $required[3] `
            --output-dir (Join-Path $pilot "analysis")
        if ($LASTEXITCODE -ne 0) { throw "$level analysis failed" }
        $summaries += @("--summary", "$level=$(Join-Path $pilot 'analysis\summary.json')")
    }
    if ($summaries.Count -eq 0) { throw "no complete operator pilots to analyze" }
    & $Python -u -m ccpu.paper1.e3 analyze-operator-ladder @summaries `
        --output-dir (Join-Path $Root "analysis")
    if ($LASTEXITCODE -ne 0) { throw "operator ladder analysis failed" }
}
finally {
    Pop-Location
}
