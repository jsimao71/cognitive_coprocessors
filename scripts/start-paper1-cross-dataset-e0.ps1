param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][ValidateSet("asdiv", "svamp", "mawps", "gsm_plus")][string]$Dataset,
    [ValidateSet("cpu", "xpu", "cuda")][string]$Device = "cpu",
    [int]$ShardIndex = 0,
    [int]$ShardCount = 1
)

$runner = Join-Path $RepositoryRoot "scripts/run-paper1-cross-dataset-e0.ps1"
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $runner,
    "-RepositoryRoot", $RepositoryRoot,
    "-Dataset", $Dataset,
    "-Device", $Device,
    "-ShardIndex", $ShardIndex,
    "-ShardCount", $ShardCount
)
Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WindowStyle Hidden
