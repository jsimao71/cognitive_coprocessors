param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$DatasetList,
    [ValidateSet("cpu", "xpu", "cuda")][string]$Device = "cuda",
    [string]$WaitTaskName = "",
    [int]$WaitProcessId = 0
)

$ErrorActionPreference = "Stop"
if ($WaitProcessId -gt 0) {
    Wait-Process -Id $WaitProcessId -ErrorAction SilentlyContinue
}
if ($WaitTaskName) {
    do {
        $task = schtasks.exe /Query /TN $WaitTaskName /FO LIST 2>$null
        $running = $LASTEXITCODE -eq 0 -and ($task -match "(?m)^Status:\s+Running\s*$")
        if ($running) {
            Start-Sleep -Seconds 60
        }
    } while ($running)
}

$runner = Join-Path $RepositoryRoot "scripts/run-paper1-cross-dataset-e0.ps1"
foreach ($name in $DatasetList.Split(",")) {
    & $runner -RepositoryRoot $RepositoryRoot -Dataset $name -Device $Device
}
