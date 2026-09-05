@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
if not defined CCPU_PYTHON set "CCPU_PYTHON=%USERPROFILE%\.venvs\ccpu-cuda\Scripts\python.exe"

powershell.exe -NoProfile -Command ^
  "$source=[IO.File]::ReadAllText('%~dp0run-paper1-qwen1-7b-matched-cuda.ps1'); $campaign=[ScriptBlock]::Create($source); & $campaign -RepositoryRoot '%REPO_ROOT%' -PythonPath '%CCPU_PYTHON%'"

exit /b %ERRORLEVEL%
