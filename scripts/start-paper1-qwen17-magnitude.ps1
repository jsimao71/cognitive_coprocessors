param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$AdapterPath
)

$ErrorActionPreference = "Stop"
$stateRoot = Join-Path $RepositoryRoot `
    "artifacts/paper1/evaluation_model_ladder_v1/qwen3_1_7b/queue_state"
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
$pidPath = Join-Path $stateRoot "magnitude.pid"
if (Test-Path -LiteralPath $pidPath) {
    $recordedPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    if (Get-Process -Id $recordedPid -ErrorAction SilentlyContinue) {
        throw "Qwen 1.7B magnitude queue already runs as PID $recordedPid"
    }
}

$runner = Join-Path $RepositoryRoot "scripts/resume-paper1-qwen17-magnitude.ps1"
$stdout = Join-Path $stateRoot "magnitude.stdout.log"
$stderr = Join-Path $stateRoot "magnitude.stderr.log"
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $runner,
    "-RepositoryRoot", $RepositoryRoot,
    "-PythonExecutable", $PythonExecutable,
    "-AdapterPath", $AdapterPath
)
$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Host "STARTED Qwen 1.7B magnitude queue PID=$($process.Id)"
