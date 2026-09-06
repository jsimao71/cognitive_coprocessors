@echo off
setlocal EnableExtensions

powershell.exe -NoProfile -Command "Invoke-Command -ScriptBlock ([scriptblock]::Create((Get-Content -LiteralPath '%~dp0run-paper1-gsm8k-sem-aug2-xpu.ps1' -Raw))) -ArgumentList $true,'%~dp0..'"
exit /b %ERRORLEVEL%
