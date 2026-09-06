@echo off
setlocal EnableExtensions

powershell.exe -NoProfile -Command "^& ([scriptblock]::Create((Get-Content -LiteralPath '%~dp0run-paper1-gsm8k-sem-aug1-xpu.ps1' -Raw))) -WaitForB1L"
exit /b %ERRORLEVEL%
