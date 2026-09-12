@echo off
setlocal EnableExtensions

if "%~2"=="" (
    echo Usage: %~nx0 SCALE OPERATOR_LEVEL 1>&2
    exit /b 2
)
set "SCALE=%~1"
set "LEVEL=%~2"
if /i not "%SCALE%"=="1" if /i not "%SCALE%"=="1000" (
    echo SCALE must be 1 or 1000. 1>&2
    exit /b 2
)
if /i not "%LEVEL%"=="o0" if /i not "%LEVEL%"=="o1" if /i not "%LEVEL%"=="o5" if /i not "%LEVEL%"=="o6" (
    echo OPERATOR_LEVEL must be o0, o1, o5, or o6. 1>&2
    exit /b 2
)

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=%USERPROFILE%\.venvs\ccpu-cpu\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "EVAL=%REPO_ROOT%\artifacts\paper1\operator_jitter_matrix_v1\gsm8k_matched_seed99173\seed_17011\x%SCALE%\%LEVEL%\test.jsonl"
set "OUTPUT=%REPO_ROOT%\artifacts\paper1\operator_jitter_matrix_v1\runs\seed_17011\direct\x%SCALE%\%LEVEL%\cpu\shard_0"
set "CONFIG=%REPO_ROOT%\configs\paper1\gsm8k_direct_reasoning_qwen_cpu.json"

if not exist "%PYTHON%" exit /b 3
if not exist "%EVAL%" exit /b 4
if exist "%OUTPUT%\summary.json" exit /b 0
if not exist "%OUTPUT%" mkdir "%OUTPUT%"
if exist "%OUTPUT%\.run.lock" del /q "%OUTPUT%\.run.lock"

"%PYTHON%" -u -m ccpu.paper1.e3 run-gsm8k-direct-shard ^
    --eval "%EVAL%" --config "%CONFIG%" --condition direct_reasoning ^
    --output-dir "%OUTPUT%" --shard-index 0 --shard-count 1 ^
    --checkpoint-every 1 >>"%OUTPUT%\stdout.log" 2>>"%OUTPUT%\stderr.log"
exit /b %ERRORLEVEL%
