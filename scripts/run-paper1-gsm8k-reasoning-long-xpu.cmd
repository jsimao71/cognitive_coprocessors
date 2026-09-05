@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

powershell.exe -NoProfile -Command ^
  "$source=[IO.File]::ReadAllText('%~dp0run-paper1-gsm8k-reasoning-long-xpu.ps1'); $campaign=[ScriptBlock]::Create($source); & $campaign -RepositoryRoot '%REPO_ROOT%'"

exit /b %ERRORLEVEL%
