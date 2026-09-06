param([string]$RepositoryRoot)

$ErrorActionPreference = "Stop"
$RepoRoot = if ($RepositoryRoot) {
    (Resolve-Path $RepositoryRoot).Path
}
else {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$B1lAnalysis = Join-Path $RepoRoot `
    "artifacts\paper1\gsm8k_scale_v1\analysis\matched_contribution_b1l_v1.json"
$MagnitudeScript = Join-Path $RepoRoot `
    "scripts\run-paper1-gsm8k-magnitude-ladder.ps1"
$AugmentationCommand = Join-Path $RepoRoot `
    "scripts\run-paper1-gsm8k-sem-aug1-xpu.cmd"

while (-not (Test-Path -LiteralPath $B1lAnalysis)) {
    Write-Host "WAIT B1L contribution analysis"
    Start-Sleep -Seconds 60
}

$source = [IO.File]::ReadAllText($MagnitudeScript)
$campaign = [ScriptBlock]::Create($source)
& $campaign -RepositoryRoot $RepoRoot -Device xpu -Conditions "b1,b1l"
if ($LASTEXITCODE -ne 0) { throw "Local magnitude campaign failed" }

& cmd.exe /d /c $AugmentationCommand
if ($LASTEXITCODE -ne 0) { throw "AUG1 resume failed" }
