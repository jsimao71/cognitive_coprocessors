param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$DatasetList,
    [ValidateSet("cpu", "xpu", "cuda")][string]$Device = "cuda",
    [string]$PythonExecutable = "python",
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
$runnerBlock = [scriptblock]::Create((Get-Content -LiteralPath $runner -Raw))
foreach ($name in $DatasetList.Split(",")) {
    & $runnerBlock -RepositoryRoot $RepositoryRoot -Dataset $name -Device $Device `
        -PythonExecutable $PythonExecutable
}
