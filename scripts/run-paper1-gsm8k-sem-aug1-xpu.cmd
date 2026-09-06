@echo off
setlocal EnableExtensions

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-paper1-gsm8k-sem-aug1-xpu.ps1" -WaitForB1L
exit /b %ERRORLEVEL%
